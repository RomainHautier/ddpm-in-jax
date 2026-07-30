"""MAX-N re-grade of the dt32 OOD claims. The dt32 files hold only 2 triplets/sequence, so the
ceiling is 70 triplets (35 non-train sequences x 2). Uses EVERY available one — 9x the basis of the
Re=2000 decorrelated claim, 2.3x the sealed one-shot's."""
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
MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
CASES = {
 're2000': dict(gt='flow-data/kf_re2000_256_40seed_dt32.npy', re=2000.0, exclude=[0,1,2,3,4],
                anchor='base_results/regime_stats_re2000_obsfit_v3.npz', kc=7,
                ck='monitoring/ddpo_re2000_dt32_ckpts/ddpo_re1000_iter0599.pkl',
                ck_r6='monitoring/ddpo_re2000_dt32_r6_ckpts/ddpo_re1000_iter0349.pkl'),
 're10000': dict(gt='flow-data/kf_re10000_256_40seed_dt32.npy', re=10000.0, exclude=[20,21,22,23,24],
                anchor='base_results/regime_stats_re10000_obsfit_dt32.npz', kc=6,
                ck='monitoring/ddpo_re10000_dt32_ckpts/ddpo_re1000_iter0549.pkl', ck_r6=None),
}
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa,_s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0-ab[100]))
sat,s1t = float(jnp.sqrt(ab[150])), float(jnp.sqrt(1.0-ab[150]))
def batched(fn,x,seed,bs=8):
    k=jax.random.PRNGKey(seed); o=[]
    for i in range(0,len(x),bs): o.append(np.asarray(fn(jnp.asarray(x[i:i+bs]), jax.random.fold_in(k,i))))
    return np.concatenate(o)
kf=np.fft.fftfreq(N,1.0/N); KR=np.sqrt(kf[:,None]**2+kf[None,:]**2)

for tag,c in CASES.items():
    seqs=[s for s in range(40) if s not in c['exclude']]
    xg,xl,sid=[],[],[]
    for s in seqs:
        seq=load_sequence(c['gt'],s)
        g=build_triplets(seq,MEAN,SIG); l=build_triplets(grid_downsample_degrade(seq,4),MEAN,SIG)
        xg.append(g); xl.append(l); sid += [s]*len(g)
    xg=np.concatenate(xg); xl=np.concatenate(xl); sid=np.array(sid)
    print(f"\n=== {tag} MAX-N: {len(seqs)} sequences, {len(xg)} triplets (file ceiling) ===",flush=True)
    recon=batched(lambda xb,kk: ddim20(base_params,_sa*xb+_s1*jax.random.normal(jax.random.fold_in(kk,1),xb.shape)),xl,500)
    E_gt=np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
    Ehg=local_hik_energy(xg[...,1]*SIG,HIK0,6.0)
    resid_fn=jax.jit(make_residual_loss(n=N,re=c['re'],std=SIG,mean=0.0))
    A=np.load(c['anchor'])['spec_ref']
    dx=make_dx_func(n=N,re=c['re'],std=SIG,mean=MEAN)
    w_hi=0.5*(1+np.tanh((KR-float(c['kc']))/2.0))
    def report(y,nm):
        E=np.asarray(spec_fn(jnp.asarray(y))).mean(0)
        ret=float(E[HIK0:96].sum()/E_gt[HIK0:96].sum()); lo=float(E[1:5].sum()/E_gt[1:5].sum())
        Eh=local_hik_energy(y[...,1]*SIG,HIK0,6.0); pl=float(np.corrcoef(Eh.ravel(),Ehg.ravel())[0,1])
        # per-sequence bootstrap SE on ret (resample sequences with replacement)
        us=np.unique(sid); rng=np.random.default_rng(0); bs=[]
        Ey=np.asarray(spec_fn(jnp.asarray(y)))
        Eg=np.asarray(spec_fn(jnp.asarray(xg)))
        for _ in range(400):
            pick=rng.choice(us,len(us),replace=True)
            m=np.concatenate([np.where(sid==s)[0] for s in pick])
            bs.append(Ey[m][:,HIK0:96].sum(1).mean()/Eg[m][:,HIK0:96].sum(1).mean())
        print(f"  {nm:<26} ret={ret:.3f} +-{np.std(bs):.3f}(seq-boot) lowk={lo:.3f} place={pl:.3f} "
              f"resid={float(np.asarray(resid_fn(jnp.asarray(y))).mean()):.1f} k*={eff_resolution(E,E_gt)} "
              f"| blind={float(E[10:96].sum()/A[10:96].sum()):.3f}",flush=True)
    report(recon,'base recon')
    for label,ckp in (('DDPO 600-iter',c['ck']),('DDPO R6 early-stop',c['ck_r6'])):
        if not ckp: continue
        P=pickle.load(open(ckp,'rb'))['params']
        smp=make_kchain_ddim_sampler(ddpm.unet,ab,[150,100,50],86,dx,3.0,temp=0.30)
        y=batched(lambda xb,kk: smp(P,sat*xb+s1t*jax.random.normal(jax.random.fold_in(kk,1),xb.shape),
                                    jax.random.fold_in(kk,2)),recon,700)
        report(y,f'{label} itemp0.30')
        yh=np.real(np.fft.ifft2(np.fft.fft2(recon,axes=(1,2))*(1-w_hi[None,:,:,None])
                  +np.fft.fft2(y,axes=(1,2))*w_hi[None,:,:,None],axes=(1,2))).astype(np.float32)
        report(yh,f'{label} hybrid kc={c["kc"]}')
print("\nMAXN REGRADE COMPLETE",flush=True)
