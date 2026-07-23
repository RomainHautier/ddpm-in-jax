"""SEALED EVALUATION — frozen protocol Appendix B (single sanctioned run).
Order of operations (decisions strictly before grading):
  1. R3.2 checkpoint selection: deployed-vs-anchor band ratio on TRAIN-POOL inputs (GT-free)
  2. R3.3 k_c: first k with E_ddpo >= E_recon for 3 consecutive shells, TRAIN-POOL spectra (GT-free)
  3. §5 grading: TEST seqs 24-39 (96 frames, opened here for the first time), deep cascade K3+lam3,
     inference temps 1.0 and 0.30, base/DDPO/hybrid — reported unedited."""
import os, sys, pickle, glob
os.chdir('/home/rhautier/ddpm-jax')
sys.path.insert(0,'.'); sys.path.insert(0,'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn, make_residual_loss
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution
MEAN,SIG,N,HIK0=0.0,4.7988,256,32
GT_PATH=os.environ.get('SEAL_GT','flow-data/kf_re2000_256_40seed.npy')
# Attempt #1 defaults; attempt #2 (Appendix B8) overrides via env: SEAL_ANCHOR (v2, fingerprinted),
# SEAL_CKDIR, SEAL_TEST (e.g. "8:24" -> seqs 8-23).
ANCHOR=os.environ.get('SEAL_ANCHOR','base_results/regime_stats_re2000_obsfit_floorfix.npz')
CKDIR=os.environ.get('SEAL_CKDIR','monitoring/ddpo_re2000_frozen_confirmatory_ckpts')
# SEAL_TEST: "a:b" range, or comma list "0,5,10,15" for decorrelated picks (C5: seqs are ordered
# segments, adjacent corr ~0.6 — spaced picks restore approximate independence).
_ts=os.environ.get('SEAL_TEST','24:40')
TEST_SEQS=([int(v) for v in _ts.split(',')] if ',' in _ts
           else list(range(*[int(v) for v in _ts.split(':')])))
print(f"SEAL CONFIG: anchor={ANCHOR} ckdir={CKDIR} test={TEST_SEQS}",flush=True)
ddpm,base_params,_=build_base_ddpm(); ab=ddpm.alpha_bar
_RE=float(os.environ.get('SEAL_RE','2000'))
dx=make_dx_func(n=N,re=_RE,std=SIG,mean=MEAN); spec_fn=make_spectrum_fn(N)
resid_fn=jax.jit(make_residual_loss(n=N,re=_RE,std=SIG,mean=0.0))
ddim20=build_ddim_denoiser(ddpm.unet,ab,100,20)
_sa,_s1=float(jnp.sqrt(ab[100])),float(jnp.sqrt(1.0-ab[100]))
sat,s1t=float(jnp.sqrt(ab[150])),float(jnp.sqrt(1.0-ab[150]))
A=np.load(ANCHOR)['spec_ref']
def batched(fn,xin,seed,bs=8):
    k=jax.random.PRNGKey(seed); o=[]
    for i in range(0,len(xin),bs): o.append(np.asarray(fn(jnp.asarray(xin[i:i+bs]),jax.random.fold_in(k,i))))
    return np.concatenate(o)
def pool(seqs,n_per):
    xg,xl=[],[]
    for s in seqs:
        seq=load_sequence(GT_PATH,s)
        g=build_triplets(seq,MEAN,SIG); l=build_triplets(grid_downsample_degrade(seq,4),MEAN,SIG)
        i2=np.linspace(0,len(g)-1,n_per).astype(int); xg.append(g[i2]); xl.append(l[i2])
    return np.concatenate(xg),np.concatenate(xl)

# ---------- 1. R3.2 selection (train pool, GT-free: GT array loaded but only LR used here) ----------
_TRAIN=[int(v) for v in os.environ.get('SEAL_TRAIN','0,1,2,3').split(',')]
_,xl_tr=pool(_TRAIN,6)
xdd_tr=batched(lambda xb,kk: ddim20(base_params,_sa*xb+_s1*jax.random.normal(jax.random.fold_in(kk,1),xb.shape)),xl_tr,500)
deep=make_kchain_ddim_sampler(ddpm.unet,ab,[150,100,50],86,dx,3.0)
print("R3.2 checkpoint selection (deployed[10,96)/anchor[10,96) on train pool):",flush=True)
best=(None,1e9)
_fix=os.environ.get('SEAL_CKPT')
for ck in ([_fix] if _fix else sorted(glob.glob(f'{CKDIR}/ddpo_re1000_iter*.pkl'))):
    P=pickle.load(open(ck,'rb'))['params']
    y=batched(lambda xb,kk: deep(P,sat*xb+s1t*jax.random.normal(jax.random.fold_in(kk,1),xb.shape),
                                 jax.random.fold_in(kk,2)),xdd_tr,700)
    E=np.asarray(spec_fn(jnp.asarray(y))).mean(0)
    ratio=float(E[10:96].sum()/A[10:96].sum())
    print(f"  {os.path.basename(ck)}: ratio={ratio:.3f}  |ratio-1|={abs(ratio-1):.3f}",flush=True)
    if abs(ratio-1)<best[1]: best=(ck,abs(ratio-1))
print(f"SELECTED: {best[0]}",flush=True)
P=pickle.load(open(best[0],'rb'))['params']
# ---------- 2. R3.3 k_c (train pool, GT-free) ----------
y_tr=batched(lambda xb,kk: deep(P,sat*xb+s1t*jax.random.normal(jax.random.fold_in(kk,1),xb.shape),
                                jax.random.fold_in(kk,2)),xdd_tr,700)
E_dd=np.asarray(spec_fn(jnp.asarray(y_tr))).mean(0); E_rc=np.asarray(spec_fn(jnp.asarray(xdd_tr))).mean(0)
kc=8
for k in range(2,17):
    if all(E_dd[k+j]>=E_rc[k+j] for j in range(3)): kc=k; break
print(f"R3.3 crossover: k_c={kc}",flush=True)
# ---------- 3. SEALED GRADING (test seqs opened here) ----------
print(f"\n=== OPENING TEST SET (seqs {TEST_SEQS}) ===",flush=True)
xg,xl=pool(TEST_SEQS,int(os.environ.get('SEAL_NPER','6')))
xdd=batched(lambda xb,kk: ddim20(base_params,_sa*xb+_s1*jax.random.normal(jax.random.fold_in(kk,1),xb.shape)),xl,500)
E_gt=np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
Ehg=local_hik_energy(xg[...,1]*SIG,HIK0,6.0)
GTres=float(np.asarray(resid_fn(jnp.asarray(xg))).mean())
kf=np.fft.fftfreq(N,1.0/N); KR=np.sqrt(kf[:,None]**2+kf[None,:]**2)
w_hi=0.5*(1+np.tanh((KR-float(kc))/2.0))
def metrics(y,nm):
    E=np.asarray(spec_fn(jnp.asarray(y))).mean(0)
    ret=float(E[HIK0:96].sum()/E_gt[HIK0:96].sum()); mse=float(((y-xg)**2).mean())
    Eh=local_hik_energy(y[...,1]*SIG,HIK0,6.0); pl=float(np.corrcoef(Eh.ravel(),Ehg.ravel())[0,1])
    R=float(np.asarray(resid_fn(jnp.asarray(y))).mean()); lo=float(E[1:5].sum()/E_gt[1:5].sum())
    print(f"{nm:<28} ret={ret:.3f} |ret-1|={abs(ret-1):.3f} lowk={lo:.3f} MSE={mse:.4f} place={pl:.3f} resid={R:.1f} k*={eff_resolution(E,E_gt)}",flush=True)
print(f"GT residual^2 on test: {GTres:.1f}",flush=True)
metrics(xdd,'base recon')
for itemp in [float(v) for v in os.environ.get('SEAL_ITEMPS','1.0,0.30').split(',')]:
    smp=make_kchain_ddim_sampler(ddpm.unet,ab,[150,100,50],86,dx,3.0,temp=itemp)
    y=batched(lambda xb,kk: smp(P,sat*xb+s1t*jax.random.normal(jax.random.fold_in(kk,1),xb.shape),
                                jax.random.fold_in(kk,2)),xdd,700)
    metrics(y,f'DDPO itemp={itemp}')
    yh=np.real(np.fft.ifft2(np.fft.fft2(xdd,axes=(1,2))*(1-w_hi[None,:,:,None])
              +np.fft.fft2(y,axes=(1,2))*w_hi[None,:,:,None],axes=(1,2))).astype(np.float32)
    metrics(yh,f'hybrid kc={kc} itemp={itemp}')
print("\nA-PRIORI BAR (B6): PRIMARY ret(itemp 0.30) in [0.80,1.20]; SECONDARY k*>=80, hybrid lowk>=0.97",flush=True)
print("done",flush=True)
