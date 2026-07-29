"""CROSS-REGIME INFERENCE TEST on genuinely unseen regimes (Re=1500, Re=3000, new generator).

Question: can a model finetuned for one regime be recovered on an unseen regime by changing ONLY
the inference process? Both DOWN-transfer (hot model on milder flow -> remove dose) and UP-transfer
(cool model on hotter flow -> can we ADD dose beyond the frozen config?) are tested.

Stage 0: set-point calibration in the DEPLOYMENT convention (itemp 0.30) — each healthy model on
         its own source pool vs its own anchor. Establishes the band the blind picks target.
Stage 1: for each (new regime, model): depth ladder, blind score on the ANCHOR'S SOURCE POOL
         (seqs 0-3 LR) vs the obs-fit anchor. No GT.
Stage 2: same ladder graded vs GT on a disjoint TEST pool (seqs 8,11,14,17). Revealed after.
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
MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
ITEMP = 0.30
MODELS = {
 're1000':  'monitoring/ddpo_re1000_k2_s100-75_ddim50_ddiminit_hk10_emabase_ckpts/ddpo_re1000_iter0299.pkl',
 're2000':  'monitoring/ddpo_re2000_dt32_ckpts/ddpo_re1000_iter0599.pkl',
 're10000': 'monitoring/ddpo_re10000_dt32_ckpts/ddpo_re1000_iter0549.pkl'}
HOME = {  # each model's own regime: gt, Re, source pool, anchor
 're1000':  ('flow-data/kf_2d_re1000_256_40seed.npy', 1000.0, [32,33], 'base_results/regime_stats_re1000.npz'),
 're2000':  ('flow-data/kf_re2000_256_40seed_dt32.npy', 2000.0, [0,1,2,3], 'base_results/regime_stats_re2000_obsfit_v3.npz'),
 're10000': ('flow-data/kf_re10000_256_40seed_dt32.npy', 10000.0, [20,21,22,23], 'base_results/regime_stats_re10000_obsfit_dt32.npz')}
NEW = {
 're1500': dict(gt='flow-data/generated/gen_fnons_re1500_kf_1024to256_20seq.npy', re=1500.0,
                anchor='base_results/regime_stats_re1500_obsfit_gen.npz',
                models=['re1000','re2000','re10000']),
 're3000': dict(gt='flow-data/generated/gen_fnons_re3000_kf_1024to256_20seq.npy', re=3000.0,
                anchor='base_results/regime_stats_re3000_obsfit_gen.npz',
                models=['re2000','re10000'])}
SRC_SEQS, TEST_SEQS = [0,1,2,3], [8,11,14,17]
LADDER = [('K4[200,150,100,50]x110',[200,150,100,50],110), ('K3[150,100,50]x86',[150,100,50],86),
          ('K2[100,75]x50',[100,75],50), ('K1[100]x20',[100],20),
          ('K1[75]x20',[75],20), ('K1[50]x12',[50],12)]
UP = {('re1500','re1000'), ('re3000','re2000')}       # up-transfer: keep the extended K4 rung

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
PARAMS = {k: pickle.load(open(v,'rb'))['params'] for k,v in MODELS.items()}

def batched(fn, xin, seed, bs=8):
    k = jax.random.PRNGKey(seed); o=[]
    for i in range(0,len(xin),bs): o.append(np.asarray(fn(jnp.asarray(xin[i:i+bs]), jax.random.fold_in(k,i))))
    return np.concatenate(o)

def pool(gt_path, seqs, n_per):
    xg, xl = [], []
    for s in seqs:
        seq = load_sequence(gt_path, s)
        g = build_triplets(seq, MEAN, SIG); l = build_triplets(grid_downsample_degrade(seq,4), MEAN, SIG)
        i2 = np.linspace(0, len(g)-1, n_per).astype(int); xg.append(g[i2]); xl.append(l[i2])
    return np.concatenate(xg), np.concatenate(xl)

def recon_of(xl, seed): 
    return batched(lambda xb,kk: ddim20(base_params, _sa*xb + _s1*jax.random.normal(jax.random.fold_in(kk,1), xb.shape)), xl, seed)

def run(model, starts, steps, recon, re):
    dx = make_dx_func(n=N, re=re, std=SIG, mean=MEAN)
    smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=ITEMP)
    sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0-ab[starts[0]]))
    return batched(lambda xb,kk: smp(PARAMS[model], sa*xb + s1*jax.random.normal(jax.random.fold_in(kk,1), xb.shape),
                                     jax.random.fold_in(kk,2)), recon, 700)

# ---------------- Stage 0: set-point in the deployment convention ----------------
print(f"=== STAGE 0 — set-point calibration @ itemp {ITEMP} (each healthy model, own pool+anchor) ===", flush=True)
setpts = {}
for m,(gt,re,src,anc) in HOME.items():
    A = np.load(anc)['spec_ref']
    _, xl = pool(gt, src, 2 if 'dt32' in gt else 6)
    y = run(m, [150,100,50], 86, recon_of(xl, 500), re)
    E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
    setpts[m] = float(E[10:96].sum()/A[10:96].sum())
    print(f"  {m:>8} healthy plateau @itemp{ITEMP} = {setpts[m]:.3f}"
          f"{'   (legacy measured-stats anchor — excluded from band)' if m=='re1000' else ''}", flush=True)
obs = [setpts['re2000'], setpts['re10000']]
BAND = (min(obs), max(obs))
print(f"=> DEPLOYMENT-CONVENTION SET-POINT BAND (obs-fit regimes): [{BAND[0]:.3f}, {BAND[1]:.3f}]", flush=True)

# ---------------- Stages 1+2: ladders on the unseen regimes ----------------
for rname, R in NEW.items():
    A = np.load(R['anchor'])['spec_ref']
    _, xl_src = pool(R['gt'], SRC_SEQS, 2)
    xg_te, xl_te = pool(R['gt'], TEST_SEQS, 3)
    rec_src, rec_te = recon_of(xl_src, 500), recon_of(xl_te, 900)
    E_gt = np.asarray(spec_fn(jnp.asarray(xg_te))).mean(0)
    Ehg = local_hik_energy(xg_te[...,1]*SIG, HIK0, 6.0)
    resid_fn = jax.jit(make_residual_loss(n=N, re=R['re'], std=SIG, mean=0.0))
    Eb = np.asarray(spec_fn(jnp.asarray(rec_te))).mean(0)
    Ehb = local_hik_energy(rec_te[...,1]*SIG, HIK0, 6.0)
    print(f"\n=== {rname} (unseen): base recon ret={float(Eb[HIK0:96].sum()/E_gt[HIK0:96].sum()):.3f} "
          f"place={float(np.corrcoef(Ehb.ravel(),Ehg.ravel())[0,1]):.3f} "
          f"k*={eff_resolution(Eb,E_gt)} | blind={float(Eb[10:96].sum()/A[10:96].sum()):.3f} ===", flush=True)
    for m in R['models']:
        rungs = LADDER if (rname,m) in UP else LADDER[1:]
        print(f"--- {m} model @ {rname} ({'UP-transfer' if (rname,m) in UP else 'down-transfer'}) ---", flush=True)
        best = (None, 1e9)
        for cname, starts, steps in rungs:
            ys = run(m, starts, steps, rec_src, R['re'])
            blind = float(np.asarray(spec_fn(jnp.asarray(ys))).mean(0)[10:96].sum()/A[10:96].sum())
            yt = run(m, starts, steps, rec_te, R['re'])
            E = np.asarray(spec_fn(jnp.asarray(yt))).mean(0)
            ret = float(E[HIK0:96].sum()/E_gt[HIK0:96].sum()); lo = float(E[1:5].sum()/E_gt[1:5].sum())
            Eh = local_hik_energy(yt[...,1]*SIG, HIK0, 6.0)
            pl = float(np.corrcoef(Eh.ravel(), Ehg.ravel())[0,1])
            d = abs(blind - np.clip(blind, *BAND))
            mark = ' <- IN BAND' if BAND[0] <= blind <= BAND[1] else ''
            print(f"  {cname:<24} blind={blind:.3f}{mark:11} | ret={ret:.3f} lowk={lo:.3f} place={pl:.3f} "
                  f"resid={float(np.asarray(resid_fn(jnp.asarray(yt))).mean()):.1f} k*={eff_resolution(E,E_gt)}", flush=True)
            if d < best[1]: best = (cname, d, ret, pl)
        print(f"  BLIND PICK: {best[0]}  ->  GT ret={best[2]:.3f} place={best[3]:.3f}", flush=True)
print("\nCROSSREGIME NEW COMPLETE", flush=True)
