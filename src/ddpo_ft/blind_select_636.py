"""GT-FREE inference-config SELECTION at robust sample size.

Question: given a finetuned model and an unseen regime, can we choose the chain/step configuration
using ONLY the target regime's low-res data? Nothing here touches ground truth: the anchor is built
from the target's own LR (seqs 0-3), and the score is that anchor vs the model's deployed output on
the SAME LR. The GT grading of whichever rung this picks happens separately, afterwards.

R5's earlier 1-in-5 failure used a blind score computed on 8 triplets. The blind score has a FIXED
denominator so it does not self-normalise (base recon: 0.846 at n=70 vs 0.909 at n=8), i.e. it is the
noisiest quantity in the pipeline. This re-runs the selection with 60 triplets per rung — ~2.7x less
selection noise — to separate "the rule is broken" from "the rule was measured badly".
"""
import os, sys, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from src.rewards import make_spectrum_fn
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
MEAN,SIG,N = 0.0,4.7988,256
SRC_SEQS, PER_SEQ = [0,1,2,3], 15       # 60 triplets from the anchor's OWN source sequences
BAND = (0.798, 0.858)                    # calibration-regime band (NOT portable — reported, not trusted)
LADDER = [('K4[200,150,100,50]x110',[200,150,100,50],110), ('K3[150,100,50]x86',[150,100,50],86),
          ('K2[100,75]x50',[100,75],50), ('K1[100]x20',[100],20),
          ('K1[75]x20',[75],20), ('K1[50]x12',[50],12)]
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
PARAMS = {k: pickle.load(open(v,'rb'))['params'] for k,v in MODELS.items()}
def batched(fn,x,seed,bs=16):
    k=jax.random.PRNGKey(seed); o=[]
    for i in range(0,len(x),bs): o.append(np.asarray(fn(jnp.asarray(x[i:i+bs]), jax.random.fold_in(k,i))))
    return np.concatenate(o)
for R,cfg in REGIMES.items():
    xl=[]
    for s in SRC_SEQS:
        l=build_triplets(grid_downsample_degrade(load_sequence(cfg['gt'],s),4),MEAN,SIG)
        xl.append(l[np.linspace(0,len(l)-1,PER_SEQ).astype(int)])
    xl=np.concatenate(xl)
    recon=batched(lambda xb,kk: ddim20(base_params,_sa*xb+_s1*jax.random.normal(jax.random.fold_in(kk,1),xb.shape)),xl,500)
    A=np.load(cfg['anchor'])['spec_ref']
    dx=make_dx_func(n=N,re=cfg['re'],std=SIG,mean=MEAN)
    Eb=np.asarray(spec_fn(jnp.asarray(recon))).mean(0)
    print(f"\n=== Re={R} BLIND SELECTION ({len(xl)} triplets, LR only) | base recon blind="
          f"{float(Eb[10:96].sum()/A[10:96].sum()):.3f} ===",flush=True)
    for nm in MODELS:
        best=(None,9)
        row=[]
        for cname,starts,steps in LADDER:
            smp=make_kchain_ddim_sampler(ddpm.unet,ab,starts,steps,dx,3.0,temp=0.30)
            sa,s1=float(jnp.sqrt(ab[starts[0]])),float(jnp.sqrt(1.0-ab[starts[0]]))
            y=batched(lambda xb,kk: smp(PARAMS[nm],sa*xb+s1*jax.random.normal(jax.random.fold_in(kk,1),xb.shape),
                                        jax.random.fold_in(kk,2)),recon,700)
            b=float(np.asarray(spec_fn(jnp.asarray(y))).mean(0)[10:96].sum()/A[10:96].sum())
            row.append((cname,b))
            d=abs(b-np.clip(b,*BAND))
            if d<best[1]: best=(cname,d)
        print(f"  {nm}:",flush=True)
        for cname,b in row:
            print(f"      {cname:<24} blind={b:.3f}{'  <- IN BAND' if BAND[0]<=b<=BAND[1] else ''}",flush=True)
        print(f"      => BLIND PICK (band {BAND}): {best[0]}",flush=True)
print("\nBLIND SELECT COMPLETE",flush=True)
