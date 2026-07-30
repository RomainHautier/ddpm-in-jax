"""OOD TRANSFER GRADING at robust sample size — every finetuned model on every unseen regime.

Design (settled with the user): 636+ triplets SPREAD ACROSS ALL available sequences, because
independent units are set by sequence count, not triplet count (adjacent triplets in a sequence are
~98% correlated). 16 test sequences x 40 triplets = 640 triplets / 16 independent units.
Errors: bootstrap BY SEQUENCE. Config: the frozen deployment cascade (K3 [150,100,50] x86, lam3,
itemp 0.30) — i.e. transfer with NO inference retuning, which is the question being asked.
Regimes 1500/3000/4000/5000 are unseen by every model and by every calibration.
"""
import os, sys, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn, make_residual_loss
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution
MEAN,SIG,N,HIK0 = 0.0,4.7988,256,32
TEST_SEQS = list(range(4,20))          # 0-3 reserved (anchor source pool); 16 test sequences
PER_SEQ   = 40                          # 16 x 40 = 640 triplets
KC        = 6                           # fixed (reward rule picked 6-7 at both calibration regimes)
MODELS = {
 'in-dist Re=1000': 'monitoring/ddpo_re1000_k2_s100-75_ddim50_ddiminit_hk10_emabase_ckpts/ddpo_re1000_iter0299.pkl',
 'Re=2000 model':   'monitoring/ddpo_re2000_dt32_ckpts/ddpo_re1000_iter0599.pkl',
 'Re=10000 model':  'monitoring/ddpo_re10000_dt32_ckpts/ddpo_re1000_iter0549.pkl'}
REGIMES = {r: dict(gt=f'flow-data/generated/gen_fnons_re{r}_kf_1024to256_20seq.npy',
                   anchor=f'base_results/regime_stats_re{r}_obsfit_gen.npz', re=float(r))
           for r in (1500,3000,4000,5000)}
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa,_s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0-ab[100]))
sat,s1t = float(jnp.sqrt(ab[150])), float(jnp.sqrt(1.0-ab[150]))
PARAMS = {k: pickle.load(open(v,'rb'))['params'] for k,v in MODELS.items()}
kf=np.fft.fftfreq(N,1.0/N); KR=np.sqrt(kf[:,None]**2+kf[None,:]**2)
w_hi=0.5*(1+np.tanh((KR-float(KC))/2.0))

def batched(fn,x,seed,bs=16):
    k=jax.random.PRNGKey(seed); o=[]
    for i in range(0,len(x),bs): o.append(np.asarray(fn(jnp.asarray(x[i:i+bs]), jax.random.fold_in(k,i))))
    return np.concatenate(o)

for R,cfg in REGIMES.items():
    xg,xl,sid=[],[],[]
    for s in TEST_SEQS:
        seq=load_sequence(cfg['gt'],s)
        g=build_triplets(seq,MEAN,SIG); l=build_triplets(grid_downsample_degrade(seq,4),MEAN,SIG)
        i2=np.linspace(0,len(g)-1,PER_SEQ).astype(int)
        xg.append(g[i2]); xl.append(l[i2]); sid += [s]*PER_SEQ
    xg=np.concatenate(xg); xl=np.concatenate(xl); sid=np.array(sid)
    print(f"\n=== Re={R} (UNSEEN): {len(TEST_SEQS)} sequences x {PER_SEQ} = {len(xg)} triplets ===",flush=True)
    recon=batched(lambda xb,kk: ddim20(base_params,_sa*xb+_s1*jax.random.normal(jax.random.fold_in(kk,1),xb.shape)),xl,500)
    A=np.load(cfg['anchor'])['spec_ref']
    dx=make_dx_func(n=N,re=cfg['re'],std=SIG,mean=MEAN)
    resid_fn=jax.jit(make_residual_loss(n=N,re=cfg['re'],std=SIG,mean=0.0))
    Eg_all=np.asarray(spec_fn(jnp.asarray(xg)))
    E_gt=Eg_all.mean(0); Ehg=local_hik_energy(xg[...,1]*SIG,HIK0,6.0)
    us=np.unique(sid); rng=np.random.default_rng(0)
    idx_by_seq={s:np.where(sid==s)[0] for s in us}
    def report(y,nm):
        Ey_all=np.asarray(spec_fn(jnp.asarray(y))); E=Ey_all.mean(0)
        ret=float(E[HIK0:96].sum()/E_gt[HIK0:96].sum()); lo=float(E[1:5].sum()/E_gt[1:5].sum())
        Eh=local_hik_energy(y[...,1]*SIG,HIK0,6.0); pl=float(np.corrcoef(Eh.ravel(),Ehg.ravel())[0,1])
        bs=[]
        for _ in range(400):
            m=np.concatenate([idx_by_seq[s] for s in rng.choice(us,len(us),replace=True)])
            bs.append(Ey_all[m][:,HIK0:96].sum(1).mean()/Eg_all[m][:,HIK0:96].sum(1).mean())
        print(f"  {nm:<24} ret={ret:.3f} +-{np.std(bs):.3f} lowk={lo:.3f} place={pl:.3f} "
              f"resid={float(np.asarray(resid_fn(jnp.asarray(y))).mean()):.1f} k*={eff_resolution(E,E_gt)} "
              f"| blind={float(E[10:96].sum()/A[10:96].sum()):.3f}",flush=True)
    report(recon,'base recon')
    for nm,_ in MODELS.items():
        smp=make_kchain_ddim_sampler(ddpm.unet,ab,[150,100,50],86,dx,3.0,temp=0.30)
        y=batched(lambda xb,kk: smp(PARAMS[nm],sat*xb+s1t*jax.random.normal(jax.random.fold_in(kk,1),xb.shape),
                                    jax.random.fold_in(kk,2)),recon,700)
        report(y,nm)
        yh=np.real(np.fft.ifft2(np.fft.fft2(recon,axes=(1,2))*(1-w_hi[None,:,:,None])
                  +np.fft.fft2(y,axes=(1,2))*w_hi[None,:,:,None],axes=(1,2))).astype(np.float32)
        report(yh,f'  + hybrid kc={KC}')
print("\nTRANSFER GRADE COMPLETE",flush=True)
