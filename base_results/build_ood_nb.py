import json
cells = []
def md(s):  cells.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s})

md("""# OOD generalisation: conditional models vs plain base — Re 500 / 1000 / 2000

Does the learned residual-conditioning mapping improve **out-of-distribution** generalisation over the
plain (unconditional) base? All models trained at **Re=1000**; run here on Re=500 and Re=2000 generated
flows (8 seqs, idx 0-7) plus the Re=1000 in-dist set (seqs 32-39), learned-only (no linear). Conditional
models get the residual signal at the **target Re** (`conditioning.inference.re`). Metrics per model/Re:
PDE residual (at that Re), MSE, energy-spectrum match, vorticity PDF. numpy/CPU, streaming, skips missing.
""")
code("""import os, pickle
import numpy as np
import matplotlib.pyplot as plt
os.chdir("/home/rhautier/ddpm-jax")
REC="monitoring/sequence_reconstructions"; MEAN,STD,N,DT=0.0,4.7988,256,1.0/32.0
MODELS=["base","grad_frozen60","grad_full60","field_frozen60","field_full60"]
RE_CFG={
 500: dict(prefix="ood_re500_",     seqs=list(range(8)),      gt="flow-data/kf_re500_256_20seed.npy",   gcs="gs://ddpm-thesis-rh/flow-data/generated_kf/kf_re500_256_20seed.npy"),
 1000:dict(prefix="indist_re1000_", seqs=list(range(32,40)),  gt="flow-data/kf_2d_re1000_256_40seed.npy",gcs="gs://ddpm-thesis-rh/flow-data/kf_2d_re1000_256_40seed.npy"),
 2000:dict(prefix="ood_re2000_",    seqs=list(range(8)),      gt="flow-data/kf_re2000_256_20seed.npy",   gcs="gs://ddpm-thesis-rh/flow-data/generated_kf/kf_re2000_256_20seed.npy"),
}
COLORS={m:c for m,c in zip(MODELS,plt.cm.tab10.colors)}
def tag(re,m): return f"{RE_CFG[re]['prefix']}{m}_"
def fp(re,m,s): return f"{REC}/sequence_reconstruction_{tag(re,m)}seq{s}.pkl"
""")
code("""k=np.fft.fftfreq(N,d=1.0/N); KX=k[:,None]*np.ones((1,N)); KY=k[None,:]*np.ones((N,1))
KSQ=KX**2+KY**2; KSQ_NZ=KSQ.copy(); KSQ_NZ[0,0]=1.0
coord=np.linspace(0,2*np.pi,N,endpoint=False); _,YY=np.meshgrid(coord,coord,indexing="ij")
FORCING=-4.0*np.cos(4.0*YY); KMAG=np.round(np.sqrt(KX**2+KY**2)).astype(int); NK=KMAG.max()+1
VBINS=np.linspace(-30,30,201); VCEN=0.5*(VBINS[1:]+VBINS[:-1])
def make_residual(re):
    visc=1.0/re
    def res(t):
        wt=(t[...,2]-t[...,0])/(2*DT); wm=t[...,1]; wh=np.fft.fft2(wm); psih=wh/KSQ_NZ
        u=np.fft.ifft2(1j*KY*psih).real; v=np.fft.ifft2(-1j*KX*psih).real
        wx=np.fft.ifft2(1j*KX*wh).real; wy=np.fft.ifft2(1j*KY*wh).real; wlap=np.fft.ifft2(-KSQ*wh).real
        return wt+(u*wx+v*wy-visc*wlap+0.1*wm)-FORCING
    return res
def radial_spectrum(f):
    P=np.abs(np.fft.fft2(f))**2
    return np.bincount(KMAG.ravel(),P.ravel(),minlength=NK)/np.maximum(np.bincount(KMAG.ravel(),minlength=NK),1)
def accumulate(it,re):
    rf=make_residual(re); res=[]; spec=np.zeros(NK); hist=np.zeros(len(VBINS)-1); n=0
    for t in it:
        res.append(float(np.mean(rf(t)**2))); mid=t[...,1]
        spec+=radial_spectrum(mid); hist+=np.histogram(mid.ravel(),VBINS)[0]; n+=1
    return dict(residual=np.array(res),spectrum=spec/max(n,1),hist=hist/max(hist.sum(),1),nframes=n)
""")
code("""GT={}
for re,c in RE_CFG.items():
    if not os.path.exists(c["gt"]):
        import subprocess; os.makedirs("flow-data",exist_ok=True)
        subprocess.run(["gcloud","storage","cp",c["gcs"],c["gt"]],check=True)
    GT[re]=np.load(c["gt"],mmap_mode="r"); print(f"Re={re}: GT {GT[re].shape}")
""")
code("""def gt_frames(re):
    for s in RE_CFG[re]["seqs"]:
        sq=np.asarray(GT[re][s])
        for f in range(sq.shape[0]-2): yield np.stack([sq[f],sq[f+1],sq[f+2]],-1).astype(np.float32)
def model_frames(re,m):
    files=[fp(re,m,s) for s in RE_CFG[re]["seqs"]]
    if not all(os.path.exists(x) for x in files): return None
    def gen():
        for x in files:
            for fr in pickle.load(open(x,"rb"))["frames"]: yield fr["final"].astype(np.float32)*STD+MEAN
    return gen
def mse_of(re,m):
    xs=[]
    for s in RE_CFG[re]["seqs"]:
        for fr in pickle.load(open(fp(re,m,s),"rb"))["frames"]: xs.append(fr["mse"])
    return float(np.mean(xs))
GTM,RESULTS={},{}
for re in RE_CFG:
    GTM[re]=accumulate(gt_frames(re),re); RESULTS[re]={}
    for m in MODELS:
        g=model_frames(re,m)
        if g is None: print(f"[skip] Re={re} {m}"); continue
        RESULTS[re][m]=accumulate(g(),re)
    print(f"Re={re}: GT res {GTM[re]['residual'].mean():.2f} | loaded {list(RESULTS[re])}")
""")
code("""for re in RE_CFG:
    if not RESULTS[re]: continue
    gh=GTM[re]['spectrum'][32:].sum(); gr=GTM[re]['residual'].mean()
    print(f"\\n===== Re={re}  (GT residual {gr:.2f}) =====")
    print(f"{'model':16s} {'MSE':>8s} {'residual':>10s} {'res/GT':>8s} {'hi-k ret':>9s}  {'vs base':>20s}")
    br=RESULTS[re].get('base',{}).get('residual'); bm=mse_of(re,'base') if 'base' in RESULTS[re] else None
    for m in MODELS:
        if m not in RESULTS[re]: continue
        d=RESULTS[re][m]; mse=mse_of(re,m); rr=d['residual'].mean(); note=""
        if m!='base' and br is not None:
            note=f"res {100*(rr-br.mean())/br.mean():+.1f}%  mse {100*(mse-bm)/bm:+.1f}%"
        print(f"{m:16s} {mse:8.4f} {rr:10.2f} {rr/gr:8.2f} {d['spectrum'][32:].sum()/gh:9.2f}  {note:>20s}")
""")
code("""for re in RE_CFG:
    ms=[m for m in MODELS if m in RESULTS[re]]
    if not ms: continue
    fig,ax=plt.subplots(1,3,figsize=(17,4.2)); kk=np.arange(NK)
    ax[0].bar(ms,[RESULTS[re][m]['residual'].mean() for m in ms],color=[COLORS[m] for m in ms])
    ax[0].axhline(GTM[re]['residual'].mean(),ls='--',c='k',label='GT'); ax[0].legend()
    ax[0].set_title(f"Re={re}  PDE residual (lower=more physical)"); ax[0].tick_params(axis='x',rotation=30)
    for m in ms: ax[1].semilogx(kk[1:],RESULTS[re][m]['spectrum'][1:]/np.maximum(GTM[re]['spectrum'][1:],1e-30),color=COLORS[m],label=m)
    ax[1].axhline(1,ls='--',c='k'); ax[1].axvline(32,ls=':',c='gray'); ax[1].set_ylim(0,2)
    ax[1].set_title(f"Re={re}  E(k) ratio model/GT"); ax[1].set_xlabel("k"); ax[1].legend(fontsize=8)
    ax[2].semilogy(VCEN,GTM[re]['hist'],'k-',lw=2,label='GT')
    for m in ms: ax[2].semilogy(VCEN,RESULTS[re][m]['hist'],color=COLORS[m],alpha=.8,label=m)
    ax[2].set_title(f"Re={re}  vorticity PDF"); ax[2].set_xlabel("vorticity"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.show()
""")
md("""**Read the `vs base` column** — negative `res %` = the conditional model lowers the PDE residual vs the
plain base at that Re (learned mapping helping OOD). If conditioning helps OOD but not in-dist, that's the
interesting result. Compare gradient vs field and frozen vs full across Re 500 / 1000 / 2000.
""")
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
out="/home/rhautier/ddpm-jax/base_results/ood_conditional_comparison.ipynb"
json.dump(nb,open(out,"w"),indent=1); print("wrote",out,len(cells),"cells")
