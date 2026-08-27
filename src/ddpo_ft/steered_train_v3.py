"""STEERED-TRAINING v3 = v2 + KL CONTROL (2026-08-27): the st1k/st2k large-scale drain was
attributed to steered training, but those runs used kl_coef=0.003 (inherited from the Re=8000
recipe where it was loosened to unlock far-regime dose) while the PLAIN fine-tunes they were
compared against used 0.01. This run matches the plain recipe's KL exactly, isolating the
steering from the anchor strength. STEER_KL overrides. ORIGINAL DOC: STEERED-TRAINING v2 (user 2026-08-26): the original steered fine-tunes used the v1 dial
(hard edge, marginal anchor) inside their rollouts and the near-anchor models inherited its
in-regime pathologies (st2k's large-scale drain). This retrains with the REPAIRED guide: the
v2 dose (broadband + mid-band term) with the TAPERED hi-k edge (full weight to k=80, zero at
96). Tests whether fixing the dial fixes the damage of feeding it back into the weights.
ORIGINAL DOC: STEERED-TRAINING PILOT (user 2026-08-22): the dose dial INSIDE the training rollouts (user 2026-08-20): the training-side form of the placed
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
R = int(os.environ.get('STEER_RE', '8000'))
_CFG = {8000: ('flow-data/generated/gen_fnons_re8000_kf_1024to256_20seq.npy', list(range(0, 8))),
        2000: ('flow-data/generated/gen_fnons_re2000_kf_1024to256_20seq.npy', list(range(0, 8))),
        1000: ('flow-data/kf_2d_re1000_256_40seed.npy', [32, 33])}   # r1k's frozen train split (protocol: 5-7 sealed, 8-39 burned)
GT, SEQS = _CFG[R]
PER_SEQ = 4 if SMOKE else 120
N_OUTER = 4 if SMOKE else 600
SAVE_EVERY = 2 if SMOKE else 50
B, GROUP, N_INNER, TEMP, LR = 8, 4, 4, 2.5, 5e-5
GUIDE_LS = 8.0
SAVE_DIR = f'monitoring/ddpo_re{R}_steeredtrain_kl{os.environ.get("STEER_KL", "0.01").replace(".", "")}_ckpts'
os.makedirs(SAVE_DIR, exist_ok=True)

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))

from src.rewards import make_spectrum_distance
def make_anchor_dose_dx(stats):
    # v2 + tapered edge: broadband + mid-band + tapered hi-k (reference never consulted past 96)
    from src.rewards import make_spectrum_fn as _msf
    ref = stats['spec_ref']; lref = stats.get('log_spec_ref')
    d1 = make_spectrum_distance(ref, kband=(1, 96), n=N, log_ref=lref)
    dm = make_spectrum_distance(ref, kband=(16, 32), n=N, log_ref=lref)
    _w = np.ones(64, np.float32); _w[48:] = 0.5 * (1 + np.cos(np.pi * np.arange(16) / 16))
    _wj = jnp.asarray(_w)
    _lr32 = jnp.asarray((lref if lref is not None else np.log(ref + 1e-20))[32:96], jnp.float32)
    _sf = _msf(N)
    def d2(x):
        e = jnp.maximum(_sf(x)[..., 32:96], jnp.exp(_lr32) * 1e-6)
        return jnp.sum(_wj * (jnp.log(e) - _lr32) ** 2, axis=-1) / jnp.sum(_wj)
    def loss(x):
        return jnp.sum(0.5 * d1(x) + 3.0 * dm(x) + 3.0 * d2(x))
    return jax.grad(loss)
_stats = {k: np.load(f'base_results/regime_stats_re{R}_measured_train.npz')[k]
          for k in np.load(f'base_results/regime_stats_re{R}_measured_train.npz').files}
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
    f'base_results/regime_stats_re{R}_measured_train.npz',
    'base_results/reward_calibration.json', re=R, scales_re=1000,
    weights={'spec': 0.5, 'spec_highk': 3.0, 'energy': 0.1, 'w1': 0.0, 'pde': 1.0},
    pde_hinge=True)

optimizer = optax.adam(LR)
trainer = DDPOTrainer(ddpm, base_params, reward, optimizer, group_size=GROUP,
                      n_inner=N_INNER, sampler='ddim', ddim_steps=50, eta=1.0,
                      chain_starts=[100, 75], sampling_temp=TEMP, kl_coef=float(os.environ.get('STEER_KL', '0.01')),
                      seed=0, ddim_guide_fn=GUIDE)
POLICY_EMA = 0.99
ema_params = jax.tree_util.tree_map(lambda a: a, trainer.params)


probe_hist = []
rng = np.random.default_rng(0)
PROBE = jnp.asarray(POOL[np.random.default_rng(7).choice(len(POOL), 8, replace=False)])
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
