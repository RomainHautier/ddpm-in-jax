"""ONE-SHOT Re=10000 CONFIRMATORY — STAGE 1 (anchor-only; protocol §8.10).

USER DIRECTIVE (2026-07-23): assess the deployment setup ONLY against the extrapolated anchor.
NO ground-truth quantity is computed or read anywhere in this script — not on sealed data, not on
burned splits. Sealed seqs 25-39 are NOT opened. GT grading is Stage 2, on explicit user go only.

What runs here (all on TRAIN-POOL inputs, seqs 20-23; LR is derived from stored frames, HR is
never used as a reference):
  1. ESCALATION RULE (§8.7b, adopted): record the blind plateau of the temp-2.5 run -> trigger;
     record the temp-3.5 run's plateau -> adopt. (Both runs were executed fresh, seed 0, frozen
     recipe, blind decisions; the rule applied to them reproduces what a live escalation would do.)
  2. R3.2 blind checkpoint selection for the adopted run (recomputed fresh for the record).
  3. R3.3' blind k_c: reward-argmax over the hybrid sweep (locked training reward vs the anchor).
  4. Anchor-relative deployment assessment at itemp 0.30: band ratios vs anchor, PDE residual vs
     the blind law floor. These numbers ARE the Stage-1 verdict.
"""
import os, sys, pickle, glob
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from rewards_claude import Reward
from src.rewards import make_spectrum_fn, make_residual_loss
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence

MEAN, SIG, N = 0.0, 4.7988, 256
GT_PATH = 'flow-data/kf_re10000_256_40seed_dt32.npy'
STATS = 'base_results/regime_stats_re10000_obsfit_dt32.npz'
A = np.load(STATS)['spec_ref']
FLOOR = float(np.load(STATS)['residual_ref'])
T25_DIR = 'monitoring/ddpo_re10000_dt32_t25_ckpts'
T35_DIR = 'monitoring/ddpo_re10000_dt32_ckpts'
LOCKED_WEIGHTS = {"spec": 0.5, "spec_highk": 3.0, "energy": 0.1, "w1": 0.0, "pde": 1.0, "align": 2.0}
THETA = 0.70

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
dx = make_dx_func(n=N, re=10000.0, std=SIG, mean=MEAN)
spec_fn = make_spectrum_fn(N)
resid_fn = jax.jit(make_residual_loss(n=N, re=10000.0, std=SIG, mean=0.0))
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sat, s1t = float(jnp.sqrt(ab[150])), float(jnp.sqrt(1.0 - ab[150]))
reward = Reward.from_calibration(STATS, "base_results/reward_calibration.json", re=10000,
                                 weights=LOCKED_WEIGHTS, pde_hinge=True, scales_re=1000,
                                 highk_band=(10, 96))

def batched(fn, xin, seed, bs=8):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(xin), bs):
        o.append(np.asarray(fn(jnp.asarray(xin[i:i+bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)

xl = []
for s in [20, 21, 22, 23]:
    seq = load_sequence(GT_PATH, s)
    l = build_triplets(grid_downsample_degrade(seq, 4), MEAN, SIG)
    xl.append(l)                              # ALL available train triplets; HR never referenced
xl = np.concatenate(xl)
recon = batched(lambda xb, kk: ddim20(base_params, _sa*xb + _s1*jax.random.normal(jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
E_rc = np.asarray(spec_fn(jnp.asarray(recon))).mean(0)

def plateau_of(ckpt, deep):
    P = pickle.load(open(ckpt, 'rb'))['params']
    y = batched(lambda xb, kk: deep(P, sat*xb + s1t*jax.random.normal(jax.random.fold_in(kk, 1), xb.shape),
                                    jax.random.fold_in(kk, 2)), recon, 700)
    E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
    return float(E[10:96].sum() / A[10:96].sum()), y, E

deep_sel = make_kchain_ddim_sampler(ddpm.unet, ab, [150, 100, 50], 86, dx, 3.0)

print("=== STAGE 1 — ANCHOR-ONLY ASSESSMENT (no GT anywhere; sealed 25-39 closed) ===", flush=True)
# 1. escalation record
p25, _, _ = plateau_of(sorted(glob.glob(f'{T25_DIR}/ddpo_re1000_iter*.pkl'))[-1], deep_sel)
print(f"ESCALATION (R3.1'/§8.7b): temp 2.5 plateau = {p25:.3f} vs theta {THETA} -> "
      f"{'ESCALATE to 3.5' if p25 < THETA else 'keep 2.5'}", flush=True)
# 2. R3.2 blind selection on the adopted (3.5) run
best = (None, 1e9)
for ck in sorted(glob.glob(f'{T35_DIR}/ddpo_re1000_iter*.pkl')):
    r, _, _ = plateau_of(ck, deep_sel)
    print(f"  R3.2 {os.path.basename(ck)}: deployed/anchor = {r:.3f}", flush=True)
    if abs(r - 1) < best[1]: best = (ck, abs(r - 1))
print(f"R3.2 SELECTED: {best[0]}", flush=True)
# 3. deployment operating point: itemp 0.30
deep_dep = make_kchain_ddim_sampler(ddpm.unet, ab, [150, 100, 50], 86, dx, 3.0, temp=0.30)
p35, y, E = plateau_of(best[0], deep_dep)
# R3.3' reward-selected kc
kf = np.fft.fftfreq(N, 1.0/N); KR = np.sqrt(kf[:, None]**2 + kf[None, :]**2)
kc_best = (None, -1e9)
for kc in range(2, 17):
    w = 0.5 * (1 + np.tanh((KR - float(kc)) / 2.0))
    H = np.real(np.fft.ifft2(np.fft.fft2(recon, axes=(1, 2)) * (1 - w[None, :, :, None])
               + np.fft.fft2(y, axes=(1, 2)) * w[None, :, :, None], axes=(1, 2))).astype(np.float32)
    r, _ = reward(jnp.asarray(H))
    rm = float(np.asarray(r).mean())
    if rm > kc_best[1]: kc_best = (kc, rm)
print(f"R3.3' reward-selected k_c = {kc_best[0]} (reward {kc_best[1]:.3f})", flush=True)
# 4. anchor-relative verdict
res = float(np.asarray(resid_fn(jnp.asarray(y))).mean())
print("\nSTAGE-1 DEPLOYMENT ASSESSMENT (itemp 0.30, vs extrapolated anchor only):", flush=True)
print(f"  plateau [10,96): {p35:.3f}   (escalation trigger was {p25:.3f} at 2.5)", flush=True)
for a, b in ((1, 5), (10, 32), (32, 64), (64, 96)):
    print(f"  band [{a},{b}): deployed/anchor = {float(E[a:b].sum()/A[a:b].sum()):.3f}"
          f"   recon/anchor = {float(E_rc[a:b].sum()/A[a:b].sum()):.3f}", flush=True)
print(f"  PDE residual: deployed {res:.1f} vs blind law floor {FLOOR:.0f} -> hinge "
      f"{'inactive (below floor)' if res < FLOOR else 'ACTIVE'}", flush=True)
print("\nSealed seqs 25-39 remain CLOSED. GT grading = Stage 2, on explicit user authorization only.", flush=True)
