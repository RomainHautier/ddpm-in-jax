"""ONE MODEL, ALL REGIMES, WITH RESIDUAL-FEEDBACK CONDITIONING (the user's ENS-style suggestion).

The unconditioned multi-regime run settled the mean-dose question: mixture training alone
recalibrates one shared dose (flat 1.40x ratio to the borrowed specialist at every regime) and
produces ZERO input dependence. The regime signal is in the input, but the plain UNet has no
cheap pathway from that signal to a global output scale.

THIS run gives it the pathway — using architecture that already exists in the repo:
- ConditionalUnet: a `condRes` input processed by cond_in/cond_hidden convs and merged by a
  1x1 conv whose kernel is initialised identity-on-features / ZERO-on-conditioning, so the
  merged model starts BIT-IDENTICAL to the base (verified numerically at startup below).
- The conditioning signal is the ENS-style RAW NS-RESIDUAL FIELD of the current sample,
  computed with the TARGET REGIME'S viscosity (make_field_func_visc; visc = 1/Re is a traced
  scalar, so one compiled rollout/loss serves all nine regimes). Same field, different
  viscosity -> different condRes: the network is explicitly told the regime, through physics.
- condRes is recomputed at EVERY policy step from the current x_t (residual feedback), in the
  rollout, in the PPO loss recomputation, and in the deterministic probe/eval path.

Distinct from sampling-time guidance: the deployment cascade's lambda=3 residual-gradient
nudge was active in every graded cell of the unconditioned run and did not supply the dose.
Here the residual goes INTO the network as an input it can learn to read. Training policy
itself stays guidance-free, as always.

Everything else mirrors multiregime_ft.py exactly (oracle GT anchors, nine-pool rotation,
K2 chain [100,75]x50, temp 2.5, kl 0.01, EMA 0.99, probe battery at every save) so the
mean-dose row of the two runs is directly comparable. ACCEPTANCE TEST: if conditioning works,
the fixed-depth retention row flattens where the unconditioned run decayed 0.80 -> 0.28.

Usage: python -m src.ddpo_ft.multiregime_cond_ft [--smoke] [--gt_anchors]
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp, optax, json, pickle
from rewards_claude import Reward
from ppo_claude import DDPOTrainer, per_input_advantage, build_ddim_denoiser
from train_claude import build_base_ddpm
from src.models.model import ConditionalUnet
from src.physics_guidance import make_field_func_visc
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence

SMOKE = '--smoke' in sys.argv
GT_ANCHORS = '--gt_anchors' in sys.argv   # ORACLE MODE: measured ground-truth spectra as targets.
# --adapter_init PATH: warm-start the three cond_* modules from a pretrained conditioned
# checkpoint (e.g. the GCS conditioned_field_cond_60ep supervised adapter, same field signal,
# trained frozen-base at Re=1000). Backbone stays OUR EMA base either way. In this mode the
# model is deliberately NOT base-identical at start (the adapter already conditions), so the
# identity assert is replaced by a logged deviation.
ADAPTER_INIT = None
if '--adapter_init' in sys.argv:
    ADAPTER_INIT = sys.argv[sys.argv.index('--adapter_init') + 1]
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
SAVE_DIR = ('monitoring/ddpo_multiregime_cond_gt_ckpts' if GT_ANCHORS
            else 'monitoring/ddpo_multiregime_cond_ckpts')
if ADAPTER_INIT:
    SAVE_DIR += '_adapter'
os.makedirs(SAVE_DIR, exist_ok=True)

ddpm, base_params, _ = build_base_ddpm()
ab = ddpm.alpha_bar
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)          # base recon: PLAIN base, as always
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))

# ---- conditional architecture: same hyperparameters, plus the condRes pathway ---------------
u = ddpm.unet
cond_unet = ConditionalUnet(ch=u.ch, ch_mult=u.ch_mult, out_ch=u.out_ch, in_ch=u.in_ch,
                            n_resnet_blocks=u.n_resnet_blocks, dropout_p=u.dropout_p,
                            freq_dim=u.freq_dim)
FIELD = make_field_func_visc(n=256, std=SIG, mean=MEAN)
dummy_x = jnp.zeros((1, 256, 256, 3), jnp.float32)
cond_init = cond_unet.init(jax.random.PRNGKey(7), dummy_x, jnp.array([1], jnp.int32),
                           train=False, condRes=dummy_x)
cond_params = jax.device_get(cond_init['params'])
cond_params = jax.tree_util.tree_map(np.asarray, cond_params)
if not isinstance(cond_params, dict):
    import flax
    cond_params = flax.core.unfreeze(cond_params)
extra = set(cond_params) - set(base_params)
missing = set(base_params) - set(cond_params)
assert extra == {'cond_in', 'cond_hidden', 'cond_combine'} and not missing, \
    f"param-tree mismatch: extra={extra} missing={missing}"
merged = dict(cond_params)
for k in base_params:
    bs = jax.tree_util.tree_map(lambda a: a.shape, base_params[k])
    cs = jax.tree_util.tree_map(lambda a: a.shape, cond_params[k])
    assert bs == cs, f"shape mismatch in shared module {k}: {bs} vs {cs}"
    merged[k] = base_params[k]
if ADAPTER_INIT:
    ad = pickle.load(open(ADAPTER_INIT, 'rb'))['params']
    for k in ('cond_in', 'cond_hidden', 'cond_combine'):
        ash = jax.tree_util.tree_map(lambda a: np.asarray(a).shape, ad[k])
        csh = jax.tree_util.tree_map(lambda a: np.asarray(a).shape, merged[k])
        assert ash == csh, f"adapter module {k} shape mismatch: {ash} vs {csh}"
        merged[k] = jax.tree_util.tree_map(np.asarray, ad[k])
    print(f"ADAPTER WARM-START: cond_* modules loaded from {ADAPTER_INIT} "
          f"(epoch {pickle.load(open(ADAPTER_INIT, 'rb')).get('epoch', '?')}); "
          "backbone remains the EMA base", flush=True)

# ---- BASE-IDENTITY CHECK: the merged conditional model must equal the plain base in exact
# arithmetic. TPU matmuls default to bfloat16, and the extra (identity) cond_combine conv
# reintroduces ~1e-2 rounding through the network, so the ASSERT runs under forced float32
# precision (true identity), while the default-precision deviation is logged for the record —
# it is pure rounding, far below the policy's sampling noise, and harmless to PPO.
tx = jax.random.normal(jax.random.PRNGKey(3), (2, 256, 256, 3))
tt = jnp.array([50, 120], jnp.int32)
cres = FIELD(tx, jnp.float32(1.0 / 4000))
with jax.default_matmul_precision('float32'):
    eps_plain32 = ddpm.unet.apply({'params': base_params}, tx, tt, train=False)
    eps_cond32 = cond_unet.apply({'params': merged}, tx, tt, train=False, condRes=cres)
dmax32 = float(jnp.abs(eps_plain32 - eps_cond32).max())
eps_plain = ddpm.unet.apply({'params': base_params}, tx, tt, train=False)
eps_cond = cond_unet.apply({'params': merged}, tx, tt, train=False, condRes=cres)
dmax_bf16 = float(jnp.abs(eps_plain - eps_cond).max())
print(f"base-identity check: max diff {dmax32:.2e} (float32)  {dmax_bf16:.2e} (device default)",
      flush=True)
if ADAPTER_INIT:
    print("  (adapter warm-start: nonzero deviation from base is EXPECTED and not asserted)",
          flush=True)
else:
    assert dmax32 < 1e-3, "merged conditional model does not reproduce the base — zero-init broken"

ddpm.unet = cond_unet          # trainer builds its policy around ddpm.unet

print(f"building {len(ORDER)} pools + rewards (smoke={SMOKE}, cond=residual-field)", flush=True)
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
                      seed=0, ddim_cond_fn=FIELD)
POLICY_EMA = 0.99
ema_params = jax.tree_util.tree_map(lambda a: a, trainer.params)

def make_p_reward_adv(reward, K):
    def reward_adv(x0):
        r, comps = reward(x0)
        grp_std = r.reshape(-1, K).std(axis=1).mean()
        return r, per_input_advantage(r, K), comps, grp_std
    return jax.pmap(reward_adv, axis_name='dev')

P_REWARD = {R: make_p_reward_adv(REWARDS[R], GROUP) for R in ORDER}
rng = np.random.default_rng(0)
PROBE = {R: jnp.asarray(POOLS[R][np.random.default_rng(7).choice(len(POOLS[R]), 8, replace=False)])
         for R in ORDER}
probe_hist = []
json.dump(dict(regimes=ORDER, per_seq=PER_SEQ, n_outer=N_OUTER, temp=TEMP,
               kl_coef=0.01, policy_ema=POLICY_EMA, lr=LR, group_size=GROUP, batch=B,
               policy=dict(sampler='ddim', ddim_steps=50, eta=1.0, chain_starts=[100, 75]),
               conditioning='ENS residual-field (make_field_func_visc), visc=1/Re per batch, '
                            'ConditionalUnet zero-init merge (base-identical at start)',
               adapter_init=ADAPTER_INIT,
               note=('ORACLE: measured GT anchors (train seqs), NOT deployable; single temp 2.5'
                     if GT_ANCHORS else 'obs-fit anchors; single temp 2.5')),
          open(f'{SAVE_DIR}/config.json', 'w'), indent=1)

for gi in range(N_OUTER):
    R = ORDER[gi % len(ORDER)]
    trainer._p_reward_adv = P_REWARD[R]
    idx = rng.choice(len(POOLS[R]), B, replace=False)
    m = trainer.train_iter(jnp.asarray(POOLS[R][idx]), visc=1.0 / R)
    ema_params = jax.tree_util.tree_map(lambda e, p: POLICY_EMA * e + (1.0 - POLICY_EMA) * p,
                                        ema_params, trainer.params)
    comp = "  ".join(f"{k}={v:.3f}" for k, v in m.get('components', {}).items())
    print(f"[{gi:04d}] Re={R:<5} R={m['reward_mean']:.3f}±{m['reward_std']:.3f} "
          f"gstd={m['group_r_std']:.3f} "
          f"loss {m.get('loss_first', float('nan')):.3f}->{m.get('loss_last', float('nan')):.3f}"
          f"  {comp}", flush=True)
    if (gi + 1) % SAVE_EVERY == 0 or gi == N_OUTER - 1:
        p0 = jax.tree_util.tree_map(lambda a: np.asarray(a[0]), trainer.params)
        e0 = jax.tree_util.tree_map(lambda a: np.asarray(a[0]), ema_params)
        with open(f'{SAVE_DIR}/ddpo_multicond_iter{gi:04d}.pkl', 'wb') as f:
            pickle.dump(dict(params=p0, ema_params=e0, ema_rate=POLICY_EMA, iter=gi,
                             regime_cycle=ORDER, arch='ConditionalUnet',
                             cond='residual-field visc=1/Re'), f)
        print(f"    saved iter{gi:04d}", flush=True)
        row = {}
        for Rp in ORDER:
            x0p = trainer.probe_x0(PROBE[Rp], jax.random.PRNGKey(11), visc=1.0 / Rp)
            rp, _ = REWARDS[Rp](x0p)
            row[Rp] = float(np.asarray(rp).mean())
        probe_hist.append(dict(iter=gi, **{str(k): v for k, v in row.items()}))
        json.dump(probe_hist, open(f'{SAVE_DIR}/probe_history.json', 'w'), indent=1)
        print("    [probe] " + "  ".join(f"Re{Rp}:{row[Rp]:+.2f}" for Rp in ORDER), flush=True)
print("MULTIREGIME-COND FINETUNE COMPLETE", flush=True)
