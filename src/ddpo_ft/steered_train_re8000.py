"""STEERED-TRAINING PILOT (user 2026-08-22): the dose dial INSIDE the training rollouts (user 2026-08-20): the training-side form of the placed
consistency idea — teach the policy to keep (and refine) the placement present in its own
input reconstruction, instead of correcting drift at sampling time.

Reward = the rs2k recipe's statistical reward (measured Re=2000 anchors) MINUS
w_place * d_place(x0, recon_input), where d_place is the pilot-validated map term: mean-
normalized Gaussian-smoothed local band-energy maps over [16,32)+[32,64), MSE against the
SAME maps of the sample's own base-DDIM input (per-sample, GT-free; reference maps
stop-gradiented). w_place=2.0 against calibration scale 0.1 (starts ~ the spec_highk term's
influence; the probe tracks both terms).

Everything else identical to the rs2k twin (600 iters, temp 2.5, kl 0.01, ema 0.99, seed 0,
gen re2000 seqs 0-7, base-DDIM init, K2 policy). Ckpts monitoring/ddpo_re2000_placereward_ckpts.
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp, optax, json, pickle
from rewards_claude import Reward
from ppo_claude import DDPOTrainer, per_input_advantage, build_ddim_denoiser
from train_claude import build_base_ddpm
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence

SMOKE = '--smoke' in sys.argv
MEAN, SIG, N = 0.0, 4.7988, 256
R = 8000
GT = 'flow-data/generated/gen_fnons_re8000_kf_1024to256_20seq.npy'
SEQS = list(range(0, 8))
PER_SEQ = 4 if SMOKE else 120
N_OUTER = 4 if SMOKE else 600
SAVE_EVERY = 2 if SMOKE else 50
B, GROUP, N_INNER, TEMP, LR = 8, 4, 4, 2.5, 5e-5
GUIDE_LS = 8.0
SAVE_DIR = 'monitoring/ddpo_re8000_steeredtrain_ckpts'
os.makedirs(SAVE_DIR, exist_ok=True)

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))

from src.rewards import make_spectrum_distance
def make_anchor_dose_dx(stats):
    lref = stats.get('log_spec_ref')
    d1 = make_spectrum_distance(stats['spec_ref'], kband=(1, 96), n=N, log_ref=lref)
    d2 = make_spectrum_distance(stats['spec_ref'], kband=(32, 96), n=N, log_ref=lref)
    def loss(x):
        return jnp.sum(0.5 * d1(x) + 3.0 * d2(x))
    return jax.grad(loss)
_stats = {k: np.load('base_results/regime_stats_re8000_measured_train.npz')[k]
          for k in np.load('base_results/regime_stats_re8000_measured_train.npz').files}
_dose = make_anchor_dose_dx(_stats)
def GUIDE(x0, visc):
    """net x0-shift matching the inference convention at ls=GUIDE_LS (lam*(ls/lam)=ls)."""
    return GUIDE_LS * _dose(x0)

print("building pool", flush=True)
xl = []
for s in SEQS:
    l = build_triplets(grid_downsample_degrade(load_sequence(GT, s), 4), MEAN, SIG)
    xl.append(l[np.linspace(0, len(l) - 1, PER_SEQ).astype(int)])
xl = np.concatenate(xl)
o, key = [], jax.random.PRNGKey(500)
for i in range(0, len(xl), 16):
    key, k = jax.random.split(key)
    o.append(np.asarray(ddim20(base_params, _sa * jnp.asarray(xl[i:i + 16])
                               + _s1 * jax.random.normal(k, xl[i:i + 16].shape))))
POOL = np.concatenate(o)
reward = Reward.from_calibration(
    'base_results/regime_stats_re8000_measured_train.npz',
    'base_results/reward_calibration.json', re=R, scales_re=1000,
    weights={'spec': 0.5, 'spec_highk': 3.0, 'energy': 0.1, 'w1': 0.0, 'pde': 1.0},
    pde_hinge=True)

optimizer = optax.adam(LR)
trainer = DDPOTrainer(ddpm, base_params, reward, optimizer, group_size=GROUP,
                      n_inner=N_INNER, sampler='ddim', ddim_steps=50, eta=1.0,
                      chain_starts=[100, 75], sampling_temp=TEMP, kl_coef=0.003,
                      seed=0, ddim_guide_fn=GUIDE)
POLICY_EMA = 0.99
ema_params = jax.tree_util.tree_map(lambda a: a, trainer.params)


probe_hist = []
for gi in range(N_OUTER):
    idx = rng.choice(len(POOL), B, replace=False)
    m = trainer.train_iter(jnp.asarray(POOL[idx]), visc=1.0 / R)
    ema_params = jax.tree_util.tree_map(lambda e, p: POLICY_EMA * e + (1.0 - POLICY_EMA) * p,
                                        ema_params, trainer.params)
    comp = "  ".join(f"{k}={v:.3f}" for k, v in m.get('components', {}).items())
    print(f"[{gi:04d}] R={m['reward_mean']:.3f}±{m['reward_std']:.3f} "
          f"gstd={m['group_r_std']:.3f} loss {m['loss_first']:.3f}->{m['loss_last']:.3f}  {comp}",
          flush=True)
    if (gi + 1) % SAVE_EVERY == 0 or gi == N_OUTER - 1:
        p0 = jax.tree_util.tree_map(lambda a: np.asarray(a[0]), trainer.params)
        e0 = jax.tree_util.tree_map(lambda a: np.asarray(a[0]), ema_params)
        with open(f'{SAVE_DIR}/ddpo_re1000_iter{gi:04d}.pkl', 'wb') as f:
            pickle.dump(dict(params=p0, ema_params=e0, ema_rate=POLICY_EMA, iter=gi), f)
        x0p = trainer.probe_x0(PROBE, jax.random.PRNGKey(11), visc=1.0 / R)
        rp, cp = reward(x0p)
        probe_hist.append(dict(iter=gi, R=float(np.asarray(rp).mean())))
        json.dump(probe_hist, open(f'{SAVE_DIR}/probe_history.json', 'w'), indent=1)
        print(f"    saved iter{gi:04d}  [probe] R={probe_hist[-1]['R']:+.2f}", flush=True)
print("STEERED-TRAIN FINETUNE COMPLETE", flush=True)
