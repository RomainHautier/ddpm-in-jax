import os, subprocess, json, pickle
import numpy as np
os.chdir("/home/rhautier/ddpm-jax")
REC="monitoring/sequence_reconstructions"
GCS="gs://ddpm-thesis-rh/monitoring/sparse_reconstructions"
MEAN,STD,N,DT=0.0,4.7988,256,1.0/32.0
MODELS=["base","grad_frozen60","grad_full60","field_frozen60","field_full60"]
RE_CFG={
 500: dict(prefix="ood_re500_",    seqs=list(range(8)),     gt="flow-data/kf_re500_256_20seed.npy",   gcs="gs://ddpm-thesis-rh/flow-data/generated_kf/kf_re500_256_20seed.npy"),
 1000:dict(prefix="indist_re1000_",seqs=list(range(32,40)), gt="flow-data/kf_2d_re1000_256_40seed.npy",gcs="gs://ddpm-thesis-rh/flow-data/kf_2d_re1000_256_40seed.npy"),
 2000:dict(prefix="ood_re2000_",   seqs=list(range(8)),     gt="flow-data/kf_re2000_256_20seed.npy",   gcs="gs://ddpm-thesis-rh/flow-data/generated_kf/kf_re2000_256_20seed.npy"),
}
k=np.fft.fftfreq(N,d=1.0/N); KX=k[:,None]*np.ones((1,N)); KY=k[None,:]*np.ones((N,1))
KSQ=KX**2+KY**2; KSQ_NZ=KSQ.copy(); KSQ_NZ[0,0]=1.0
coord=np.linspace(0,2*np.pi,N,endpoint=False); _,YY=np.meshgrid(coord,coord,indexing="ij")
FORCING=-4.0*np.cos(4.0*YY); KMAG=np.round(np.sqrt(KX**2+KY**2)).astype(int); NK=int(KMAG.max()+1)
VBINS=np.linspace(-30,30,201)
def resid(t,re):
    visc=1.0/re; wt=(t[...,2]-t[...,0])/(2*DT); wm=t[...,1]; wh=np.fft.fft2(wm); psih=wh/KSQ_NZ
    u=np.fft.ifft2(1j*KY*psih).real; v=np.fft.ifft2(-1j*KX*psih).real
    wx=np.fft.ifft2(1j*KX*wh).real; wy=np.fft.ifft2(1j*KY*wh).real; wlap=np.fft.ifft2(-KSQ*wh).real
    return wt+(u*wx+v*wy-visc*wlap+0.1*wm)-FORCING
def spec(f): 
    P=np.abs(np.fft.fft2(f))**2
    return np.bincount(KMAG.ravel(),P.ravel(),minlength=NK)/np.maximum(np.bincount(KMAG.ravel(),minlength=NK),1)
def get(tag,s):
    loc=f"{REC}/sequence_reconstruction_{tag}seq{s}.pkl"
    if os.path.exists(loc): return loc,False
    tmp=f"/tmp/_ood_{tag}seq{s}.pkl"
    r=subprocess.run(["gcloud","storage","cp",f"{GCS}/sequence_reconstruction_{tag}seq{s}.pkl",tmp],capture_output=True)
    return (tmp,True) if r.returncode==0 else (None,False)

# GT datasets
for re,c in RE_CFG.items():
    if not os.path.exists(c["gt"]):
        os.makedirs("flow-data",exist_ok=True); subprocess.run(["gcloud","storage","cp",c["gcs"],c["gt"]],check=True)

def acc_gt(re):
    c=RE_CFG[re]; GT=np.load(c["gt"],mmap_mode="r"); res=[]; sp=np.zeros(NK); h=np.zeros(len(VBINS)-1); n=0
    for s in c["seqs"]:
        sq=np.asarray(GT[s])
        for f in range(sq.shape[0]-2):
            t=np.stack([sq[f],sq[f+1],sq[f+2]],-1).astype(np.float32)
            res.append(float(np.mean(resid(t,re)**2))); sp+=spec(t[...,1]); h+=np.histogram(t[...,1].ravel(),VBINS)[0]; n+=1
    return dict(residual=float(np.mean(res)),spectrum=(sp/n).tolist(),hist=(h/h.sum()).tolist())

def acc_model(re,m):
    c=RE_CFG[re]; tag=f"{c['prefix']}{m}_"; res=[]; mse=[]; sp=np.zeros(NK); h=np.zeros(len(VBINS)-1); n=0
    for s in c["seqs"]:
        p,tmp=get(tag,s)
        if p is None: return None
        d=pickle.load(open(p,"rb"))
        for fr in d["frames"]:
            t=fr["final"].astype(np.float32)*STD+MEAN
            res.append(float(np.mean(resid(t,re)**2))); mse.append(float(fr["mse"])); sp+=spec(t[...,1]); h+=np.histogram(t[...,1].ravel(),VBINS)[0]; n+=1
        if tmp: os.remove(p)
    return dict(residual=float(np.mean(res)),mse=float(np.mean(mse)),spectrum=(sp/n).tolist(),hist=(h/h.sum()).tolist(),nframes=n)

OUT={"re":{}}
for re in (500,1000,2000):
    g=acc_gt(re); OUT["re"][re]={"GT":g,"models":{}}
    print(f"\n===== Re={re}  (GT residual {g['residual']:.2f}) =====",flush=True)
    gh=sum(g["spectrum"][32:])
    print(f"{'model':16s} {'MSE':>8s} {'residual':>10s} {'res/GT':>8s} {'hi-k ret':>9s}  {'vs base':>22s}")
    base=None
    for m in MODELS:
        d=acc_model(re,m)
        if d is None: print(f"{m:16s}  (missing)"); continue
        OUT["re"][re]["models"][m]=d
        if m=="base": base=d
        note=""
        if m!="base" and base:
            note=f"res {100*(d['residual']-base['residual'])/base['residual']:+.1f}%  mse {100*(d['mse']-base['mse'])/base['mse']:+.1f}%"
        hk=sum(d["spectrum"][32:])/gh
        print(f"{m:16s} {d['mse']:8.4f} {d['residual']:10.2f} {d['residual']/g['residual']:8.2f} {hk:9.2f}  {note:>22s}",flush=True)

json.dump(OUT,open("base_results/ood_metrics.json","w"))
print("\nsaved base_results/ood_metrics.json",flush=True)
