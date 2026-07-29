"""BACKTEST — do OOD-finetuned models lose (or over-energize) the milder regimes?

Stage A: best Re=2000 and Re=10000 models, standard frozen inference (K3 [150,100,50], 86 steps,
lam3, itemp 0.30), pointed BACK at lower regimes:
  - @Re=1000: virgin full seqs [0, 16] (never used in any training/eval), stride-4 dense sampling
  - @Re=2000 (dt32): decorrelated picks [24,29,34,39] (seqs 5-7 stay sealed)
References on the same frames: base recon, and the in-dist Re=1000 flagship (at Re=1000).
Graded vs GT (legitimate: Re=1000 owned; Re=2000 burned splits, report-only) + the blind
anchor-ratio view a deployment would see.

Stage B (the user's piste): GT-free INFERENCE-DEPTH selection for the OOD models at Re=1000 —
sweep inference configs (chain starts x step budget), pick blind by |E[10,96)/anchor - 1| of the
TARGET regime's anchor, then reveal each config's GT retention to check whether the blind pick
recovers parity that the full-depth cascade overshoots.
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
MODELS = {
    'indist-re1000': 'monitoring/ddpo_re1000_k2_s100-75_ddim50_ddiminit_hk10_emabase_ckpts/ddpo_re1000_iter0299.pkl',
    'ood-re2000':    'monitoring/ddpo_re2000_dt32_ckpts/ddpo_re1000_iter0599.pkl',
    'ood-re10000':   'monitoring/ddpo_re10000_dt32_ckpts/ddpo_re1000_iter0549.pkl',
}
REGIMES = {
    're1000': dict(gt='flow-data/kf_2d_re1000_256_40seed.npy', re=1000.0, seqs=[0, 16], stride=4,
                   anchor='base_results/regime_stats_re1000.npz',
                   models=['indist-re1000', 'ood-re2000', 'ood-re10000']),
    're2000': dict(gt='flow-data/kf_re2000_256_40seed_dt32.npy', re=2000.0, seqs=[24, 29, 34, 39], stride=1,
                   anchor='base_results/regime_stats_re2000_obsfit_v3.npz',
                   models=['ood-re2000', 'ood-re10000']),
}
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
PARAMS = {k: pickle.load(open(v, 'rb'))['params'] for k, v in MODELS.items()}

def batched(fn, xin, seed, bs=8):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(xin), bs):
        o.append(np.asarray(fn(jnp.asarray(xin[i:i+bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)

def pool(gt_path, seqs, stride):
    xg, xl = [], []
    for s in seqs:
        seq = load_sequence(gt_path, s)
        g = build_triplets(seq, MEAN, SIG); l = build_triplets(grid_downsample_degrade(seq, 4), MEAN, SIG)
        idx = np.arange(0, len(g), stride); xg.append(g[idx]); xl.append(l[idx])
    return np.concatenate(xg), np.concatenate(xl)

def grade(y, xg, E_gt, Ehg, resid_fn, A, tag):
    E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
    ret = float(E[HIK0:96].sum() / E_gt[HIK0:96].sum()); lo = float(E[1:5].sum() / E_gt[1:5].sum())
    mse = float(((y - xg) ** 2).mean())
    Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0); pl = float(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1])
    R = float(np.asarray(resid_fn(jnp.asarray(y))).mean())
    blind = float(E[10:96].sum() / A[10:96].sum())
    print(f"{tag:<38} ret={ret:.3f} lowk={lo:.3f} MSE={mse:.4f} place={pl:.3f} resid={R:.1f} "
          f"k*={eff_resolution(E, E_gt)}  | blind deployed/anchor={blind:.3f}", flush=True)

# ---------------- Stage A: standard frozen inference, pointed back ----------------
for rname, R in REGIMES.items():
    dx = make_dx_func(n=N, re=R['re'], std=SIG, mean=MEAN)
    resid_fn = jax.jit(make_residual_loss(n=N, re=R['re'], std=SIG, mean=0.0))
    A = np.load(R['anchor'])['spec_ref']
    xg, xl = pool(R['gt'], R['seqs'], R['stride'])
    print(f"\n=== STAGE A @ {rname}: seqs {R['seqs']} stride {R['stride']} -> {len(xg)} frames ===", flush=True)
    recon = batched(lambda xb, kk: ddim20(base_params, _sa*xb + _s1*jax.random.normal(jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    print(f"GT residual^2: {float(np.asarray(resid_fn(jnp.asarray(xg))).mean()):.1f}", flush=True)
    grade(recon, xg, E_gt, Ehg, resid_fn, A, 'base recon')
    deep = make_kchain_ddim_sampler(ddpm.unet, ab, [150, 100, 50], 86, dx, 3.0, temp=0.30)
    sat, s1t = float(jnp.sqrt(ab[150])), float(jnp.sqrt(1.0 - ab[150]))
    for m in R['models']:
        y = batched(lambda xb, kk: deep(PARAMS[m], sat*xb + s1t*jax.random.normal(jax.random.fold_in(kk, 1), xb.shape),
                                        jax.random.fold_in(kk, 2)), recon, 700)
        grade(y, xg, E_gt, Ehg, resid_fn, A, f'{m} @ {rname} (frozen K3 inference)')

# ---------------- Stage B: anchor-matched inference-depth selection @ Re=1000 ----------------
R = REGIMES['re1000']
dx = make_dx_func(n=N, re=1000.0, std=SIG, mean=MEAN)
resid_fn = jax.jit(make_residual_loss(n=N, re=1000.0, std=SIG, mean=0.0))
A = np.load(R['anchor'])['spec_ref']
xg, xl = pool(R['gt'], R['seqs'], 16)          # subset for the sweep
recon = batched(lambda xb, kk: ddim20(base_params, _sa*xb + _s1*jax.random.normal(jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
CONFIGS = [('K3 [150,100,50] x86', [150, 100, 50], 86), ('K2 [100,75] x50', [100, 75], 50),
           ('K1 [100] x20', [100], 20), ('K1 [75] x20', [75], 20), ('K1 [50] x12', [50], 12)]
print(f"\n=== STAGE B @ re1000: GT-free inference-depth selection ({len(xg)} frames) ===", flush=True)
for m in ['ood-re2000', 'ood-re10000']:
    print(f"--- {m}: sweep, blind pick = min |E[10,96)/anchor - 1| ---", flush=True)
    best = (None, 1e9)
    for cname, starts, steps in CONFIGS:
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=0.30)
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        y = batched(lambda xb, kk: smp(PARAMS[m], sa*xb + s1*jax.random.normal(jax.random.fold_in(kk, 1), xb.shape),
                                       jax.random.fold_in(kk, 2)), recon, 700)
        E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
        blind = float(E[10:96].sum() / A[10:96].sum())
        grade(y, xg, E_gt, Ehg, resid_fn, A, f'  {m} {cname}')
        if abs(blind - 1) < best[1]: best = (cname, abs(blind - 1))
    print(f"BLIND DEPTH PICK for {m}: {best[0]}", flush=True)
print("\nBACKTEST COMPLETE", flush=True)
