"""R3.3' CANDIDATE — reward-selected hybrid crossover k_c (GT-free), with GT validation.

Rule under test: sweep k_c over candidates; build the Fourier hybrid H(k_c) = low-k from the base
recon + high-k from DDPO (same tanh crossover as deployment) on TRAIN-POOL outputs; score each
hybrid with THE TRAINING REWARD (spec/spec_highk/energy/pde-hinge/align vs the frozen anchor —
zero GT); pick k_c = argmax reward. The reward trades retention against PDE cost exactly as
training did, so the pick inherits the locked objective instead of a crossing heuristic (old R3.3).

Validation: on held-out TEST picks the same sweep is graded vs GT (ret/lowk/resid/place); the rule
is freezable if the blind pick's GT metrics match the GT-optimal pick's within noise. Run in-dist
(Re=1000, GT legitimate) to validate; OOD rows are report-only diagnostics.

Usage: JAX_PLATFORMS='' BASE_CKPT=... python -m src.ddpo_ft.kc_rule_study <regime: re1000|re2000>
"""
import os, sys, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from rewards_claude import Reward
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn, make_residual_loss
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
CFG = {
    're1000': dict(gt='flow-data/kf_2d_re1000_256_40seed.npy', re=1000,
                   train=[32, 33], test=[37, 38, 39], n_per=6,
                   ckpt='monitoring/ddpo_re1000_k2_s100-75_ddim50_ddiminit_hk10_emabase_ckpts/ddpo_re1000_iter0299.pkl',
                   stats='base_results/regime_stats_re1000.npz'),
    're2000': dict(gt='flow-data/kf_re2000_256_40seed_dt32.npy', re=2000,
                   train=[0, 1, 2, 3], test=[24, 29, 34, 39], n_per=2,
                   ckpt='monitoring/ddpo_re2000_dt32_ckpts/ddpo_re1000_iter0599.pkl',
                   stats='base_results/regime_stats_re2000_obsfit_v3.npz'),
}
LOCKED_WEIGHTS = {"spec": 0.5, "spec_highk": 3.0, "energy": 0.1, "w1": 0.0, "pde": 1.0, "align": 2.0}
KC_GRID = list(range(2, 17))

cfg = CFG[sys.argv[1]]
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
dx = make_dx_func(n=N, re=float(cfg['re']), std=SIG, mean=MEAN)
spec_fn = make_spectrum_fn(N)
resid_fn = jax.jit(make_residual_loss(n=N, re=float(cfg['re']), std=SIG, mean=0.0))
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sat, s1t = float(jnp.sqrt(ab[150])), float(jnp.sqrt(1.0 - ab[150]))
reward = Reward.from_calibration(cfg['stats'], "base_results/reward_calibration.json",
                                 re=cfg['re'], weights=LOCKED_WEIGHTS, pde_hinge=True,
                                 scales_re=1000, highk_band=(10, 96))
P = pickle.load(open(cfg['ckpt'], 'rb'))['params']
deep = make_kchain_ddim_sampler(ddpm.unet, ab, [150, 100, 50], 86, dx, 3.0, temp=0.30)

def batched(fn, xin, seed, bs=8):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(xin), bs):
        o.append(np.asarray(fn(jnp.asarray(xin[i:i+bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)

def pool(seqs, n_per):
    xg, xl = [], []
    for s in seqs:
        seq = load_sequence(cfg['gt'], s)
        g = build_triplets(seq, MEAN, SIG); l = build_triplets(grid_downsample_degrade(seq, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, n_per).astype(int); xg.append(g[i2]); xl.append(l[i2])
    return np.concatenate(xg), np.concatenate(xl)

kf = np.fft.fftfreq(N, 1.0/N); KR = np.sqrt(kf[:, None]**2 + kf[None, :]**2)
def hybrid(recon, y, kc):
    w = 0.5 * (1 + np.tanh((KR - float(kc)) / 2.0))
    return np.real(np.fft.ifft2(np.fft.fft2(recon, axes=(1, 2)) * (1 - w[None, :, :, None])
                   + np.fft.fft2(y, axes=(1, 2)) * w[None, :, :, None], axes=(1, 2))).astype(np.float32)

def outputs(seqs, n_per, seed):
    xg, xl = pool(seqs, n_per)
    recon = batched(lambda xb, kk: ddim20(base_params, _sa*xb + _s1*jax.random.normal(jax.random.fold_in(kk, 1), xb.shape)), xl, seed)
    y = batched(lambda xb, kk: deep(P, sat*xb + s1t*jax.random.normal(jax.random.fold_in(kk, 1), xb.shape),
                                    jax.random.fold_in(kk, 2)), recon, seed + 200)
    return xg, recon, y

# ---- 1. BLIND SELECTION on train pool (reward only; GT loaded but unused here) ----
_, recon_tr, y_tr = outputs(cfg['train'], cfg['n_per'], 500)
print(f"[{sys.argv[1]}] BLIND k_c sweep on train pool ({len(y_tr)} outputs), reward = locked training objective:", flush=True)
best = (None, -1e9)
for kc in KC_GRID:
    H = hybrid(recon_tr, y_tr, kc)
    r, comps = reward(jnp.asarray(H))
    rm = float(np.asarray(r).mean())
    print(f"  kc={kc:2d}: reward={rm:8.3f}  spec_highk={float(np.asarray(comps['spec_highk']).mean()):.3f}"
          f"  pde={float(np.asarray(comps['pde']).mean()):.3f}", flush=True)
    if rm > best[1]: best = (kc, rm)
print(f"BLIND PICK: k_c = {best[0]}  (reward {best[1]:.3f})", flush=True)

# ---- 2. GT VALIDATION on held-out test picks ----
xg_te, recon_te, y_te = outputs(cfg['test'], cfg['n_per'], 900)
E_gt = np.asarray(spec_fn(jnp.asarray(xg_te))).mean(0)
Ehg = local_hik_energy(xg_te[..., 1] * SIG, HIK0, 6.0)
print(f"\n[{sys.argv[1]}] GT grading of the same sweep on test picks {cfg['test']}:", flush=True)
gt_best = (None, 1e9)
for kc in KC_GRID + ['ddpo']:
    H = y_te if kc == 'ddpo' else hybrid(recon_te, y_te, kc)
    E = np.asarray(spec_fn(jnp.asarray(H))).mean(0)
    ret = float(E[HIK0:96].sum() / E_gt[HIK0:96].sum()); lo = float(E[1:5].sum() / E_gt[1:5].sum())
    R = float(np.asarray(resid_fn(jnp.asarray(H))).mean())
    Eh = local_hik_energy(H[..., 1] * SIG, HIK0, 6.0); pl = float(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1])
    print(f"  kc={str(kc):>4}: ret={ret:.3f} |ret-1|={abs(ret-1):.3f} lowk={lo:.3f} resid={R:.1f} "
          f"place={pl:.3f} k*={eff_resolution(E, E_gt)}", flush=True)
    if kc != 'ddpo' and lo >= 0.97 and abs(ret - 1) < gt_best[1]: gt_best = (kc, abs(ret - 1))
print(f"GT-OPTIMAL (min |ret-1| s.t. lowk>=0.97): k_c = {gt_best[0]}", flush=True)
print(f"VERDICT: blind pick {best[0]} vs GT-optimal {gt_best[0]}", flush=True)
