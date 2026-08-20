"""PLACEMENT-REWARD FINE-TUNE (user 2026-08-20): the training-side form of the placed
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
R = 2000
GT = 'flow-data/generated/gen_fnons_re2000_kf_1024to256_20seq.npy'
SEQS = list(range(0, 8))
PER_SEQ = 4 if SMOKE else 120
N_OUTER = 4 if SMOKE else 600
SAVE_EVERY = 2 if SMOKE else 50
B, GROUP, N_INNER, TEMP, LR = 8, 4, 4, 2.5, 5e-5
W_PLACE, PLACE_SCALE = 2.0, 0.1
SAVE_DIR = 'monitoring/ddpo_re2000_placereward_ckpts'
os.makedirs(SAVE_DIR, exist_ok=True)

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))

fy = np.fft.fftfreq(N) * N
kmag = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
gsm = jnp.asarray(np.exp(-2.0 * (np.pi * 6.0) ** 2 *
                         ((fy[:, None] / N) ** 2 + (fy[None, :] / N) ** 2)))
PB = [jnp.asarray(((kmag >= lo) & (kmag < hi)).astype(np.float32)) for lo, hi in
      [(16, 32), (32, 64)]]


def nmaps(w):
    F = jnp.fft.fft2(w)
    out = []
    for m in PB:
        bp = jnp.real(jnp.fft.ifft2(F * m))
        e = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(bp ** 2) * gsm))
        out.append(e / (jnp.mean(e, axis=(-2, -1), keepdims=True) + 1e-12))
    return out


def d_place(x0, ref):
    """per-sample placed-consistency distance vs the (stop-gradiented) reference maps."""
    ms = nmaps(x0[..., 1] * SIG)
    rs = [jax.lax.stop_gradient(r) for r in nmaps(ref[..., 1] * SIG)]
    return sum(jnp.mean((m - r) ** 2, axis=(-2, -1)) for m, r in zip(ms, rs))


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
    'base_results/regime_stats_re2000_measured_train.npz',
    'base_results/reward_calibration.json', re=R, scales_re=1000,
    weights={'spec': 0.5, 'spec_highk': 3.0, 'energy': 0.1, 'w1': 0.0, 'pde': 1.0},
    pde_hinge=True)

optimizer = optax.adam(LR)
trainer = DDPOTrainer(ddpm, base_params, reward, optimizer, group_size=GROUP,
                      n_inner=N_INNER, sampler='ddim', ddim_steps=50, eta=1.0,
                      chain_starts=[100, 75], sampling_temp=TEMP, kl_coef=0.01, seed=0)
POLICY_EMA = 0.99
ema_params = jax.tree_util.tree_map(lambda a: a, trainer.params)


def reward_adv(x0, ref):
    r, comps = reward(x0)
    dp = d_place(x0, ref)
    r = r - (W_PLACE / PLACE_SCALE) * dp
    comps = dict(comps); comps['place_d'] = dp
    grp_std = r.reshape(-1, GROUP).std(axis=1).mean()
    return r, per_input_advantage(r, GROUP), comps, grp_std


P_REWARD = jax.pmap(reward_adv, axis_name='dev')
rng = np.random.default_rng(0)
PROBE = jnp.asarray(POOL[np.random.default_rng(7).choice(len(POOL), 8, replace=False)])
json.dump(dict(re=R, per_seq=PER_SEQ, n_outer=N_OUTER, temp=TEMP, kl_coef=0.01,
               policy_ema=POLICY_EMA, lr=LR, w_place=W_PLACE, place_scale=PLACE_SCALE,
               reward='measured re2000 anchors + placed map-consistency vs own recon input '
                      '(bands [16,32)+[32,64), mean-normalized, sigma=6)'),
          open(f'{SAVE_DIR}/config.json', 'w'), indent=1)

nd = trainer.n_dev
K = GROUP
probe_hist = []
for gi in range(N_OUTER):
    idx = rng.choice(len(POOL), B, replace=False)
    xs = POOL[idx]
    ipd = B // nd
    xc = jnp.repeat(jnp.asarray(xs.reshape(nd, ipd, *xs.shape[1:])), K, axis=1)
    noise = jax.random.normal(trainer._next_key(), xc.shape)
    x_start = trainer._sqrt_ab * xc + trainer._sqrt_1mab * noise
    keys = jax.random.split(trainer._next_key(), nd)
    states, actions, logp_old, x0 = trainer._p_rollout(trainer.params, x_start, keys)
    r, adv, comps, grp_std = P_REWARD(x0, xc)
    losses = []
    for _ in range(N_INNER):
        trainer.params, trainer.opt_state, loss = trainer._p_update(
            trainer.params, trainer.opt_state, states, actions, logp_old, adv,
            trainer.base_params)
        losses.append(float(loss[0]))
    ema_params = jax.tree_util.tree_map(lambda e, p: POLICY_EMA * e + (1.0 - POLICY_EMA) * p,
                                        ema_params, trainer.params)
    r_np = np.asarray(r).reshape(-1)
    comp = "  ".join(f"{k}={float(np.asarray(v).mean()):.3f}" for k, v in comps.items())
    print(f"[{gi:04d}] R={r_np.mean():.3f}±{r_np.std():.3f} "
          f"gstd={float(np.asarray(grp_std).mean()):.3f} "
          f"loss {losses[0]:.3f}->{losses[-1]:.3f}  {comp}", flush=True)
    if (gi + 1) % SAVE_EVERY == 0 or gi == N_OUTER - 1:
        p0 = jax.tree_util.tree_map(lambda a: np.asarray(a[0]), trainer.params)
        e0 = jax.tree_util.tree_map(lambda a: np.asarray(a[0]), ema_params)
        with open(f'{SAVE_DIR}/ddpo_re1000_iter{gi:04d}.pkl', 'wb') as f:
            pickle.dump(dict(params=p0, ema_params=e0, ema_rate=POLICY_EMA, iter=gi), f)
        x0p = trainer.probe_x0(PROBE, jax.random.PRNGKey(11))
        rp, cp = reward(x0p)
        dpp = float(np.asarray(d_place(x0p, PROBE)).mean())
        probe_hist.append(dict(iter=gi, R=float(np.asarray(rp).mean()), place_d=dpp))
        json.dump(probe_hist, open(f'{SAVE_DIR}/probe_history.json', 'w'), indent=1)
        print(f"    saved iter{gi:04d}  [probe] R={probe_hist[-1]['R']:+.2f} "
              f"place_d={dpp:.4f}", flush=True)
print("PLACE-REWARD FINETUNE COMPLETE", flush=True)
