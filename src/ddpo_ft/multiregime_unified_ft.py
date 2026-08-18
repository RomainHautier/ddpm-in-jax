"""FIFTH ARM — UNIFIED REWARD, MIXED BATCHES (user's proposal 2026-08-18): one reward over all
flows, several regimes inside EVERY gradient step.

Why this is the causal fix the conditioning arc points at: under regime rotation, every
gradient step was single-regime, and a single-regime step is perfectly satisfiable by a
SHARED-dose adjustment — so no conditioning pathway (input stem, adapter, FiLM) was ever the
cheapest direction, and all three arms converged to the unconditioned dose. With mixed batches
scored by a unified per-sample reward, a shared-dose direction cancels against itself inside
each update (hotter helps the far samples and hurts the near samples simultaneously); the only
directions that raise the batch reward are regime-dependent ones. Combined with the FiLM code
(whose pathway the third arm proved ACTIVE but never recruited), the gradient finally has
nowhere to go except through the conditioning.

Mechanics:
- Batch = 8 inputs from 8 DISTINCT regimes (the ninth rotates out round-robin), each with its
  own viscosity: per-sample codes for the FiLM policy, per-sample reward targets.
- Unified reward: all nine per-regime Rewards evaluated on the batch inside one pmapped
  closure, rows selected by a per-sample regime index. Rewards are FFT-cheap next to rollouts,
  so the 9x overcompute is noise. Group advantage normalization is untouched (each input's
  K samples share one regime).
- Data budget matches the rotation arms: 1350 iters x ~1 input/regime ~= 1200 inputs/regime,
  same as 150 dedicated iters x 8 inputs. Policy, KL, EMA, probes, oracle anchors: identical.
- ppo_claude is UNCHANGED: the ddim_cond_fn path is shape-agnostic, so per-device visc rows of
  shape (ipd*K,) flow through rollout/loss exactly like the scalar did. The iteration loop is
  reimplemented here only because train_iter's reward call has no per-sample index.

ACCEPTANCE TEST unchanged: the single-config mean-dose row vs the unconditioned 0.797 -> 0.281.
Usage: python -m src.ddpo_ft.multiregime_unified_ft [--smoke] [--gt_anchors]
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp, optax, json, pickle
from rewards_claude import Reward
from ppo_claude import DDPOTrainer, per_input_advantage, build_ddim_denoiser
from train_claude import build_base_ddpm
from src.models.model import FiLMUnet
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence

SMOKE = '--smoke' in sys.argv
GT_ANCHORS = '--gt_anchors' in sys.argv
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
SAVE_DIR = ('monitoring/ddpo_multiregime_unified_gt_ckpts' if GT_ANCHORS
            else 'monitoring/ddpo_multiregime_unified_ckpts')
os.makedirs(SAVE_DIR, exist_ok=True)


def RECODE(x, visc):
    """FiLM regime code from viscosity — elementwise, so visc may be a scalar (probes) or a
    per-sample array (mixed training batches)."""
    code = jnp.log(1.0 / (1000.0 * visc)) / jnp.log(8.0)
    return jnp.ones((x.shape[0],), jnp.float32) * code


ddpm, base_params, _ = build_base_ddpm()
ab = ddpm.alpha_bar
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))

u = ddpm.unet
film_unet = FiLMUnet(ch=u.ch, ch_mult=u.ch_mult, out_ch=u.out_ch, in_ch=u.in_ch,
                     n_resnet_blocks=u.n_resnet_blocks, dropout_p=u.dropout_p,
                     freq_dim=u.freq_dim)
film_init = film_unet.init(jax.random.PRNGKey(7), jnp.zeros((1, 256, 256, 3), jnp.float32),
                           jnp.array([1], jnp.int32), train=False,
                           condRes=jnp.zeros((1,), jnp.float32))
film_params = jax.tree_util.tree_map(np.asarray, film_init['params'])
if not isinstance(film_params, dict):
    import flax
    film_params = flax.core.unfreeze(film_params)
extra = set(film_params) - set(base_params)
assert extra == {'re_emb_hidden', 're_emb_out'} and not (set(base_params) - set(film_params))
merged = dict(film_params)
for k in base_params:
    merged[k] = base_params[k]
tx = jax.random.normal(jax.random.PRNGKey(3), (2, 256, 256, 3))
tt = jnp.array([50, 120], jnp.int32)
with jax.default_matmul_precision('float32'):
    d32 = float(jnp.abs(ddpm.unet.apply({'params': base_params}, tx, tt, train=False)
                        - film_unet.apply({'params': merged}, tx, tt, train=False,
                                          condRes=RECODE(tx, jnp.float32(1 / 4000)))).max())
print(f"base-identity check (float32): {d32:.2e}", flush=True)
assert d32 < 1e-3
ddpm.unet = film_unet

print(f"building {len(ORDER)} pools + rewards (smoke={SMOKE}, unified mixed-batch)", flush=True)
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
trainer = DDPOTrainer(ddpm, merged, REWARDS[ORDER[0]], optimizer, group_size=GROUP,
                      n_inner=N_INNER, sampler='ddim', ddim_steps=50, eta=1.0,
                      chain_starts=[100, 75], sampling_temp=TEMP, kl_coef=0.01,
                      seed=0, ddim_cond_fn=RECODE)
POLICY_EMA = 0.99
ema_params = jax.tree_util.tree_map(lambda a: a, trainer.params)

# ---- the UNIFIED pmapped reward: nine rewards on the whole batch, rows picked per sample ----
def unified_reward_adv(x0, idx):
    rs, comps_sel = [], {}
    all_comps = []
    for R in ORDER:
        r, comps = REWARDS[R](x0)
        rs.append(r); all_comps.append(comps)
    rs = jnp.stack(rs)                                   # (9, B)
    sel = jnp.arange(x0.shape[0])
    r = rs[idx, sel]                                     # per-sample regime's reward
    for k in all_comps[0]:
        comps_sel[k] = jnp.stack([c[k] if jnp.ndim(all_comps[0][k]) else jnp.full((x0.shape[0],), c[k])
                                  for c in all_comps])[idx, sel].mean()
    grp_std = r.reshape(-1, GROUP).std(axis=1).mean()
    return r, per_input_advantage(r, GROUP), comps_sel, grp_std

P_REWARD = jax.pmap(unified_reward_adv, axis_name='dev')

rng = np.random.default_rng(0)
PROBE = {R: jnp.asarray(POOLS[R][np.random.default_rng(7).choice(len(POOLS[R]), 8, replace=False)])
         for R in ORDER}
probe_hist = []
json.dump(dict(regimes=ORDER, per_seq=PER_SEQ, n_outer=N_OUTER, temp=TEMP,
               kl_coef=0.01, policy_ema=POLICY_EMA, lr=LR, group_size=GROUP, batch=B,
               policy=dict(sampler='ddim', ddim_steps=50, eta=1.0, chain_starts=[100, 75]),
               conditioning='FiLM log-Re code, PER-SAMPLE (mixed batches)',
               reward='UNIFIED: per-sample regime targets inside every gradient step; '
                      '8 distinct regimes per batch, ninth rotates out round-robin',
               note=('ORACLE: measured GT anchors (train seqs), NOT deployable; single temp 2.5'
                     if GT_ANCHORS else 'obs-fit anchors; single temp 2.5')),
          open(f'{SAVE_DIR}/config.json', 'w'), indent=1)

nd = trainer.n_dev
K = GROUP
for gi in range(N_OUTER):
    skip = ORDER[gi % len(ORDER)]                        # the regime sitting this iteration out
    regs = [R for R in ORDER if R != skip]               # 8 distinct regimes -> B=8 inputs
    xs = np.stack([POOLS[R][rng.integers(len(POOLS[R]))] for R in regs])
    idx_in = np.array([ORDER.index(R) for R in regs], np.int32)          # (B,)
    visc_in = np.array([1.0 / R for R in regs], np.float32)              # (B,)
    ipd = B // nd
    xc = xs.reshape(nd, ipd, *xs.shape[1:])
    xc = jnp.repeat(jnp.asarray(xc), K, axis=1)                          # (nd, ipd*K, ...)
    idx = jnp.repeat(jnp.asarray(idx_in.reshape(nd, ipd)), K, axis=1)    # (nd, ipd*K)
    visc = jnp.repeat(jnp.asarray(visc_in.reshape(nd, ipd)), K, axis=1)  # (nd, ipd*K)
    noise = jax.random.normal(trainer._next_key(), xc.shape)
    x_start = trainer._sqrt_ab * xc + trainer._sqrt_1mab * noise
    keys = jax.random.split(trainer._next_key(), nd)
    states, actions, logp_old, x0 = trainer._p_rollout(trainer.params, x_start, keys, visc)
    r, adv, comps, grp_std = P_REWARD(x0, idx)
    losses = []
    for _ in range(N_INNER):
        trainer.params, trainer.opt_state, loss = trainer._p_update(
            trainer.params, trainer.opt_state, states, actions, logp_old, adv,
            trainer.base_params, visc)
        losses.append(float(loss[0]))
    ema_params = jax.tree_util.tree_map(lambda e, p: POLICY_EMA * e + (1.0 - POLICY_EMA) * p,
                                        ema_params, trainer.params)
    r_np = np.asarray(r).reshape(-1)
    comp = "  ".join(f"{k}={float(np.asarray(v).mean()):.3f}" for k, v in comps.items())
    print(f"[{gi:04d}] mix-{skip:<5} R={r_np.mean():.3f}±{r_np.std():.3f} "
          f"gstd={float(np.asarray(grp_std).mean()):.3f} "
          f"loss {losses[0]:.3f}->{losses[-1]:.3f}  {comp}", flush=True)
    if (gi + 1) % SAVE_EVERY == 0 or gi == N_OUTER - 1:
        p0 = jax.tree_util.tree_map(lambda a: np.asarray(a[0]), trainer.params)
        e0 = jax.tree_util.tree_map(lambda a: np.asarray(a[0]), ema_params)
        with open(f'{SAVE_DIR}/ddpo_multiuni_iter{gi:04d}.pkl', 'wb') as f:
            pickle.dump(dict(params=p0, ema_params=e0, ema_rate=POLICY_EMA, iter=gi,
                             regime_cycle=ORDER, arch='FiLMUnet',
                             cond='FiLM log-Re per-sample', reward='unified mixed-batch'), f)
        print(f"    saved iter{gi:04d}", flush=True)
        row = {}
        for Rp in ORDER:
            x0p = trainer.probe_x0(PROBE[Rp], jax.random.PRNGKey(11), visc=1.0 / Rp)
            rp, _ = REWARDS[Rp](x0p)
            row[Rp] = float(np.asarray(rp).mean())
        probe_hist.append(dict(iter=gi, **{str(k): v for k, v in row.items()}))
        json.dump(probe_hist, open(f'{SAVE_DIR}/probe_history.json', 'w'), indent=1)
        print("    [probe] " + "  ".join(f"Re{Rp}:{row[Rp]:+.2f}" for Rp in ORDER), flush=True)
print("MULTIREGIME-UNIFIED FINETUNE COMPLETE", flush=True)
