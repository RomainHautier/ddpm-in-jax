"""ONE MODEL, ALL REGIMES: fine-tune the base across Re=1000..8000 simultaneously.

HYPOTHESIS. Every single-regime fine-tune we measured applies a near-constant energy dose
regardless of its input - that is exactly why transfer fails. The separability study showed the
coarse input carries a smooth regime signal (rank correlation +0.82, all of it in the spectral
tail). If the network can read that signal, a model trained on a MIXTURE of regimes - each
rewarded against its own regime's anchor - has gradient pressure to make its dose input-dependent.
This run tests that. The alternative outcome (one compromise dose, mediocre everywhere) is equally
informative.

MECHANICS.
- Nine training pools (Re=1000 uses sequences 20-27 of the reference file; the eight generated
  regimes use sequences 0-7), ~120 triplets per sequence, base-DDIM initialised once.
- Nine Reward objects, each built from its regime's own obs-fit anchor with its own viscosity.
  The PPO trainer bakes the reward into a pmapped closure, so nine such closures are built up
  front and swapped by attribute before each iteration - parameters and optimizer state stay
  single and shared. Rollout/loss are regime-independent (no guidance in the training policy).
- Regime ROTATES each iteration (round-robin), so every 9 iterations the gradient has seen all
  regimes. n_outer=1350 => 150 iterations per regime (a 600-iteration specialist saw 4x more of
  its own regime; parameter sharing is the bet).
- DELIBERATE DEVIATION from frozen R3.1, recorded: one sampling temperature (2.5, the middle
  bucket) for all regimes in this first experiment. Per-regime temperatures would need nine
  compiled rollout/loss variants; deferred until this run shows signal.
- No ground truth is read anywhere (anchors + LR pools only). No early stopping; checkpoints
  every 90 iterations (one per 10 full cycles) to a fresh directory.

Usage: python -m src.ddpo_ft.multiregime_ft [--smoke]
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
GT_ANCHORS = '--gt_anchors' in sys.argv   # ORACLE MODE: measured ground-truth spectra as targets.
# Uses target-regime GT (training sequences only) => NOT DEPLOYABLE; separates 'can one model
# learn input-dependent dose' from 'are the extrapolated anchors accurate enough'.
MEAN, SIG = 0.0, 4.7988
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {
    1000: dict(gt='flow-data/kf_2d_re1000_256_40seed.npy', seqs=list(range(20, 28)),
               anchor='base_results/regime_stats_re1000_obsfit.npz'),
    **{R: dict(gt=GEN.format(R), seqs=list(range(0, 8)),
               anchor=f'base_results/regime_stats_re{R}_obsfit_gen.npz')
       for R in (1500, 3000, 4000, 5000, 6000, 7000, 8000)},
    2000: dict(gt=GEN.format(2000), seqs=list(range(0, 8)),
               anchor='base_results/regime_stats_re2000_obsfit_newgen.npz'),
}
ORDER = sorted(REGIMES)
PER_SEQ = 4 if SMOKE else 120
N_OUTER = 4 if SMOKE else 1350
SAVE_EVERY = 2 if SMOKE else 90
B, GROUP, N_INNER, TEMP, LR = 8, 4, 4, 2.5, 5e-5
SAVE_DIR = ('monitoring/ddpo_multiregime_gt_ckpts' if GT_ANCHORS else 'monitoring/ddpo_multiregime_ckpts')
os.makedirs(SAVE_DIR, exist_ok=True)

ddpm, base_params, _ = build_base_ddpm()
ab = ddpm.alpha_bar
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))

print(f"building {len(ORDER)} pools + rewards (smoke={SMOKE})", flush=True)
POOLS, REWARDS = {}, {}
key = jax.random.PRNGKey(500)
for R in ORDER:
    c = REGIMES[R]
    xl = []
    for s in c['seqs']:
        l = build_triplets(grid_downsample_degrade(load_sequence(c['gt'], s), 4), MEAN, SIG)
        xl.append(l[np.linspace(0, len(l) - 1, PER_SEQ).astype(int)])
    xl = np.concatenate(xl)
    o = []
    for i in range(0, len(xl), 16):
        key, k = jax.random.split(key)
        o.append(np.asarray(ddim20(base_params, _sa * jnp.asarray(xl[i:i + 16])
                                   + _s1 * jax.random.normal(k, xl[i:i + 16].shape))))
    POOLS[R] = np.concatenate(o)
    anchor_path = (f'base_results/regime_stats_re{R}_measured_train.npz' if GT_ANCHORS
                   else c['anchor'])
    REWARDS[R] = Reward.from_calibration(
        anchor_path, 'base_results/reward_calibration.json', re=R, scales_re=1000,
        weights={'spec': 0.5, 'spec_highk': 3.0, 'energy': 0.1, 'w1': 0.0, 'pde': 1.0},
        pde_hinge=True)
    print(f"  Re={R}: pool {POOLS[R].shape}, anchor {os.path.basename(anchor_path)}", flush=True)

optimizer = optax.adam(LR)
trainer = DDPOTrainer(ddpm, base_params, REWARDS[ORDER[0]], optimizer, group_size=GROUP,
                      n_inner=N_INNER, sampler='ddim', ddim_steps=50, eta=1.0,
                      chain_starts=[100, 75], sampling_temp=TEMP, kl_coef=0.01,
                      seed=0)
# EMA shadow over the (replicated) policy weights, exactly as in train_claude — the newpool recipe
# all graded checkpoints came from used mu=0.99; kept identical so multi-regime ckpts are comparable
POLICY_EMA = 0.99
ema_params = jax.tree_util.tree_map(lambda a: a, trainer.params)

# nine pmapped reward+advantage closures sharing the trainer's group size; swapped per iteration
def make_p_reward_adv(reward, K):
    def reward_adv(x0):
        r, comps = reward(x0)
        grp_std = r.reshape(-1, K).std(axis=1).mean()
        return r, per_input_advantage(r, K), comps, grp_std
    return jax.pmap(reward_adv, axis_name='dev')

P_REWARD = {R: make_p_reward_adv(REWARDS[R], GROUP) for R in ORDER}
rng = np.random.default_rng(0)
# FIXED per-regime probe inputs: evaluated with the CURRENT weights at every checkpoint, giving
# nine reward trajectories over training. Signatures: all rise together = the conditioning
# attractor; sawtooth locked to the rotation = adapt-and-forget; flat mediocre = compromise dose.
PROBE = {R: jnp.asarray(POOLS[R][np.random.default_rng(7).choice(len(POOLS[R]), 8, replace=False)])
         for R in ORDER}
probe_hist = []
json.dump(dict(regimes=ORDER, per_seq=PER_SEQ, n_outer=N_OUTER, temp=TEMP,
               kl_coef=0.01, policy_ema=POLICY_EMA, lr=LR, group_size=GROUP, batch=B,
               policy=dict(sampler='ddim', ddim_steps=50, eta=1.0, chain_starts=[100, 75]),
               note=('ORACLE: measured GT anchors (train seqs), NOT deployable; single temp 2.5' if GT_ANCHORS else 'multi-regime rotation; single temp 2.5 = recorded deviation from R3.1')),
          open(f'{SAVE_DIR}/config.json', 'w'), indent=1)

for gi in range(N_OUTER):
    R = ORDER[gi % len(ORDER)]
    trainer._p_reward_adv = P_REWARD[R]
    idx = rng.choice(len(POOLS[R]), B, replace=False)
    m = trainer.train_iter(jnp.asarray(POOLS[R][idx]))
    ema_params = jax.tree_util.tree_map(lambda e, p: POLICY_EMA * e + (1.0 - POLICY_EMA) * p,
                                        ema_params, trainer.params)
    print(f"[{gi:04d}] Re={R:<5} R={m['reward_mean']:.3f}±{m['reward_std']:.3f} "
          f"gstd={m['group_r_std']:.3f} loss={m.get('loss', float('nan')):.3f}", flush=True)
    if (gi + 1) % SAVE_EVERY == 0 or gi == N_OUTER - 1:
        p0 = jax.tree_util.tree_map(lambda a: np.asarray(a[0]), trainer.params)
        e0 = jax.tree_util.tree_map(lambda a: np.asarray(a[0]), ema_params)
        with open(f'{SAVE_DIR}/ddpo_multi_iter{gi:04d}.pkl', 'wb') as f:
            pickle.dump(dict(params=p0, ema_params=e0, ema_rate=POLICY_EMA, iter=gi,
                             regime_cycle=ORDER), f)
        print(f"    saved iter{gi:04d}", flush=True)
        import jax as _jax
        row = {}
        for Rp in ORDER:
            x0p = trainer.probe_x0(PROBE[Rp], _jax.random.PRNGKey(11))
            rp, _ = REWARDS[Rp](x0p)
            row[Rp] = float(np.asarray(rp).mean())
        probe_hist.append(dict(iter=gi, **{str(k): v for k, v in row.items()}))
        json.dump(probe_hist, open(f'{SAVE_DIR}/probe_history.json', 'w'), indent=1)
        print("    [probe] " + "  ".join(f"Re{Rp}:{row[Rp]:+.2f}" for Rp in ORDER), flush=True)
print("MULTIREGIME FINETUNE COMPLETE", flush=True)
