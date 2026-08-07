"""Runnable DDPO finetuning entry point (Claude reference).

Wires the base DDPM, the physics Reward, and the DDPOTrainer into a training run on the in-dist
Re=1000 sparse-reconstruction task. The base checkpoint is UNCONDITIONAL, so conditioning is forced
off (plain Unet) to match its params.

Usage (from repo root, TPU idle):
    JAX_PLATFORMS=''  python -m src.ddpo_ft.train_claude              # real run (needs TPU/GPU)
    JAX_PLATFORMS=cpu python -m src.ddpo_ft.train_claude --smoke      # tiny CPU smoke test

Assets it reads (all present locally):
    configs/config.yaml                         base model + diffusion schedule
    checkpoints/ddpm/ckpt_epoch_0299.pkl        base params
    base_results/regime_stats_re1000.npz        reward anchors
    base_results/reward_calibration.json        weights / per-Re scales / residual floor
    flow-data/kf_2d_re1000_256_40seed.npy       GT (for sparse-nnfill inputs)
    flow-data/kmflow_idx_lst.npz                sparse sampling mask
"""
import argparse
import copy
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax.numpy as jnp                           # noqa: E402
import numpy as np                                # noqa: E402
import optax                                      # noqa: E402
import yaml                                       # noqa: E402

from ppo_claude import DDPOTrainer                # noqa: E402
from rewards_claude import Reward                 # noqa: E402
from src.models.model import DDPM                 # noqa: E402
from src.sequence_inference import (               # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence, sparse_nnfill_degrade)
from src.utils import load_checkpoint             # noqa: E402

MEAN, STD, N = 0.0, 4.7988, 256
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"

# Per-regime config: in-dist Re=1000 (base's own regime) and the OOD targets Re=500/2000.
# The base is Re=1000-trained; finetuning toward Re=2000 tests DDPO's OOD/extrapolation ability.
# train_seqs = held-out-from-OOD-eval sequences (OOD sets keep seqs 0-7 for eval, so train on 8+).
RE_CFG = {
    1000: dict(gt="flow-data/kf_2d_re1000_256_40seed.npy", stats="base_results/regime_stats_re1000.npz",
               train_seqs=[32, 33], probe_seq=36),
    # Frozen confirmatory split (protocol Appendix B): 40-seed file, train 0-3 (LR only), probe 4
    # record-only. Attempt #1 TEST 24-39 burned; attempt #2 TEST 8-23 burned; spare 5-7 sealed.
    # dt32 file = stride-80 subsample (frames 39/119/199/279) of the fine-dt generation — restores
    # the base model's dt=1/32 frame spacing (raw file lag-1 corr 0.99999 vs required ~0.986).
    # v1.3/v1.4 anchor: regime_stats_re2000_obsfit_v3.npz (rebuilt from THIS file, fingerprinted).
    2000: dict(gt="flow-data/kf_re2000_256_40seed_dt32.npy", stats="base_results/regime_stats_re2000.npz",
               train_seqs=[0, 1, 2, 3], probe_seq=4),
    500:  dict(gt="flow-data/kf_re500_256_20seed.npy",    stats="base_results/regime_stats_re500.npz",
               train_seqs=[8, 9, 10, 11], probe_seq=0),
    # Re=10000: CLEAN disjoint split. NOTE the Re=2000/500 entries above train on [8,9,10,11] while
    # the eval driver uses test=10..19 -> seqs 10,11 leak (measured impact -0.013, inside eval
    # noise, because DDPO uses the pool as a state distribution rather than fitting targets — but
    # it is still a leak). Re=10000 avoids it entirely: train 20-23, probe 24, eval val 0-9 /
    # test 10-19, all disjoint. 40 sequences available.
    # dt32 = stride-80 subsample (frames 39/119/199/279; raw file is the fine-dt generation, B9).
    # Anchor: regime_stats_re10000_obsfit_dt32.npz (frozen §2b procedure on THIS file's train LR).
    # Report-only eval on seqs 10-19; confirmatory seqs 25-39 REMAIN SEALED.
    10000: dict(gt="flow-data/kf_re10000_256_40seed_dt32.npy",
                stats="base_results/regime_stats_re10000_blind.npz",
                train_seqs=[20, 21, 22, 23], probe_seq=24),
}


def build_base_ddpm(config_path="configs/config.yaml", ckpt_path=None):
    """Build the base (unconditional) DDPM + load base params. Conditioning forced off so the plain
    Unet matches the unconditional checkpoint.

    Base-checkpoint resolution order (for old-base vs EMA-base A/Bs without touching call sites):
      1. explicit `ckpt_path` argument
      2. env var BASE_CKPT
      3. configs/config.yaml -> inference.base_ckpt
      4. legacy default checkpoints/ddpm/ckpt_epoch_0299.pkl"""
    cfg = copy.deepcopy(yaml.safe_load(open(config_path)))
    cfg["conditioning"]["train"]["enabled"] = False
    cfg["conditioning"]["inference"]["enabled"] = False
    ddpm = DDPM(cfg)
    _ck_path = (ckpt_path or os.environ.get("BASE_CKPT")
                or cfg.get("inference", {}).get("base_ckpt")
                or "checkpoints/ddpm/ckpt_epoch_0299.pkl")
    import pickle as _pkl
    with open(_ck_path, "rb") as _f:
        _ck = _pkl.load(_f)
    # prefer EMA weights when the checkpoint carries them (named key; see report.md EMA finding)
    _use_ema = "ema_params" in _ck
    params = _ck["ema_params"] if _use_ema else _ck["params"]
    epoch = _ck.get("epoch", "?")
    _ema_note = f" | EMA weights (mu={_ck.get('ema_rate', '?')})" if _use_ema else ""
    print(f"base DDPM: {type(ddpm.unet).__name__} | epoch {epoch} | T={cfg['diffusion']['T']}"
          f"{_ema_note} | ckpt={_ck_path}", flush=True)
    return ddpm, params, cfg


def _degrade(seq, s, grid_factor):
    """grid_factor=None -> random 1024-pt collocation (the task); else clean grid-`factor` downsample."""
    return grid_downsample_degrade(seq, grid_factor) if grid_factor else sparse_nnfill_degrade(seq, s)


def sparse_input_iterator(gt_path, seqs, n_inputs, mean=MEAN, std=STD, seed=0, grid_factor=None,
                          base_denoise=None):
    """Yield (n_inputs, 256, 256, 3) batches of NORMALIZED nnfill input triplets, drawn from the given
    Re=1000 sequences. grid_factor=None -> random-1024 task degradation; else clean grid-`factor`
    (e.g. 4 -> 4096-pt regular grid). Each triplet is one conditioning field; cycles forever.
    base_denoise: optional (pool)->pool callable that replaces the raw low-res pool with the frozen-base
    DDIM reconstruction (the base-DDIM finetuning init)."""
    rng = np.random.default_rng(seed)
    pools = []
    for s in seqs:
        seq = load_sequence(gt_path, s)                       # (n_frames, H, W)
        nn = _degrade(seq, s, grid_factor)                    # sparse degradation (random or grid)
        pools.append(build_triplets(nn, mean, std))          # (n, 256, 256, 3) normalized
    pool = np.concatenate(pools, axis=0)
    print(f"input pool: {pool.shape} triplets from seqs {list(seqs)} "
          f"(degrade={'grid'+str(grid_factor)+'x' if grid_factor else 'random-1024'})", flush=True)
    if base_denoise is not None:
        pool = base_denoise(pool)                             # low-res -> frozen-base DDIM reconstruction
        print(f"    base-DDIM init: input pool replaced with base reconstruction -> {pool.shape}", flush=True)
    while True:
        idx = rng.choice(len(pool), n_inputs, replace=False)
        yield jnp.asarray(pool[idx], dtype=jnp.float32)


def save_ckpt(params, opt_state, outer, save_dir, ema_params=None, ema_rate=None):
    import pickle
    import jax
    os.makedirs(save_dir, exist_ok=True)
    # params/opt_state are REPLICATED across devices (pmap) -> save a single replica
    unrep = lambda t: jax.tree_util.tree_map(lambda a: np.asarray(a[0]), t)
    path = os.path.join(save_dir, f"ddpo_re1000_iter{outer:04d}.pkl")
    payload = {"params": unrep(params), "opt_state": unrep(opt_state), "iter": outer}
    if ema_params is not None:
        # named keys, same convention as base-training checkpoints (never positional)
        payload["ema_params"] = unrep(ema_params)
        payload["ema_rate"] = ema_rate
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"    saved {path}", flush=True)


def make_gt_probe(seq_id=36, n=4, gt_path=GT_PATH, grid_factor=None, base_denoise=None):
    """Fixed held-out probe: (nnfill-input triplets, GT triplets) for the live GT-retention curve.
    grid_factor matches the training degradation so the probe reflects the same regime.
    base_denoise: same base-DDIM transform as training, applied to the probe inputs for consistency."""
    seq = load_sequence(gt_path, seq_id)
    inp = build_triplets(_degrade(seq, seq_id, grid_factor), MEAN, STD)
    gt = build_triplets(seq, MEAN, STD)
    idx = np.linspace(0, len(inp) - 1, n).astype(int)
    probe = inp[idx]
    if base_denoise is not None:
        probe = base_denoise(probe)                          # match the base-DDIM training init
    return jnp.asarray(probe), jnp.asarray(gt[idx])


def build_anchor_monitor(re, cfg, stats_path, grid_factor, base_params, ddpm, n_per=6):
    """R6: a GT-FREE in-loop probe that scores the DEPLOYMENT configuration, not a training proxy.

    Returns score(params) = E_deployed[10,96) / E_anchor[10,96), where E_deployed comes from the
    frozen deployment cascade (K3 [150,100,50], 86 steps, lam3 guidance, itemp 0.30) run on a FIXED
    base-DDIM reconstruction of the TRAIN-POOL low-res inputs — identical in every respect to the
    quantity R3.2 uses post-hoc, so the monitor and the selector measure the same thing.
    """
    import numpy as _np, jax as _jax, jax.numpy as _jnp
    from diag_guided_residual import make_kchain_ddim_sampler
    from ppo_claude import build_ddim_denoiser
    from src.rewards import make_spectrum_fn
    from src.physics_guidance import make_dx_func
    from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence

    ab = ddpm.alpha_bar
    anchor = _np.load(stats_path)["spec_ref"]
    spec_fn = make_spectrum_fn(N)
    dx = make_dx_func(n=N, re=float(re), std=STD, mean=MEAN)
    ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
    sa0, s10 = float(_jnp.sqrt(ab[100])), float(_jnp.sqrt(1.0 - ab[100]))
    sat, s1t = float(_jnp.sqrt(ab[150])), float(_jnp.sqrt(1.0 - ab[150]))
    deep = make_kchain_ddim_sampler(ddpm.unet, ab, [150, 100, 50], 86, dx, 3.0, temp=0.30)

    xl = []
    for s in cfg["train_seqs"]:
        l = build_triplets(grid_downsample_degrade(load_sequence(cfg["gt"], s), grid_factor or 4), MEAN, STD)
        xl.append(l[_np.linspace(0, len(l) - 1, min(n_per, len(l))).astype(int)])
    xl = _np.concatenate(xl)

    def _batched(fn, xin, seed, bs=8):
        k = _jax.random.PRNGKey(seed); o = []
        for i in range(0, len(xin), bs):
            o.append(_np.asarray(fn(_jnp.asarray(xin[i:i + bs]), _jax.random.fold_in(k, i))))
        return _np.concatenate(o)

    recon = _batched(lambda xb, kk: ddim20(base_params, sa0 * xb + s10 * _jax.random.normal(
        _jax.random.fold_in(kk, 1), xb.shape)), xl, 500)

    def score(params):
        y = _batched(lambda xb, kk: deep(params, sat * xb + s1t * _jax.random.normal(
            _jax.random.fold_in(kk, 1), xb.shape), _jax.random.fold_in(kk, 2)), recon, 700)
        E = _np.asarray(spec_fn(_jnp.asarray(y))).mean(0)
        return float(E[10:96].sum() / anchor[10:96].sum())

    print(f"    R6 ANCHOR MONITOR: deployment cascade K3[150,100,50]x86 lam3 itemp0.30 on "
          f"{len(recon)} fixed train-pool inputs vs {os.path.basename(stats_path)} (GT-free)", flush=True)
    return score


def main(smoke=False, n_outer=None, save_dir=None, save_every=20,
         eval_every=10, lr=5e-5, resume=None, re=1000, stats=None, scales_re=None,
         pde_local_weight=0.0, pde_local_patch=2, pde_local_frac=0.1, align_weight=0.0,
         spec_residual_weight=0.0, pde_weight=1.0, grid_factor=None,
         base_ddim_init=False, ddim_steps=20, ddim_t_start=100, t_start=None,
         sampler="ddpm", policy_ddim_steps=20, eta=1.0, chain_starts=None, ddim_stride=None,
         highk_lo=32, policy_ema=0.0, clip_eps=0.2, sampling_temp=None, kl_coef=None, seed=None,
         pde_two_sided=False, fresh_opt=False,
         anchor_monitor_every=0, anchor_band=None, anchor_patience=2,
         gt_override=None, train_seqs_override=None, no_gt_probe=False):
    import json
    import pickle
    import jax
    from src.rewards import make_spectrum_fn

    cfg = dict(RE_CFG[re])            # copy: never mutate the frozen table
    # Data overrides. The PHYSICS still comes from `re` (nu = 1/re); only the file and the
    # training sequences move. Lets a regenerated dataset be used without editing RE_CFG,
    # so every earlier run stays byte-reproducible from the table.
    if gt_override:
        cfg['gt'] = gt_override
    if train_seqs_override:
        cfg['train_seqs'] = [int(x) for x in str(train_seqs_override).split(',')]
    if gt_override or train_seqs_override:
        print(f"    DATA OVERRIDE: gt={cfg['gt']} train_seqs={cfg['train_seqs']}", flush=True)
    save_dir = save_dir or (f"monitoring/ddpo_re{re}{'_ddiminit' if base_ddim_init else ''}"
                            f"{'_ddimpolicy' if sampler == 'ddim' else ''}"
                            f"{f'_k{len(chain_starts)}chain' if chain_starts else ''}_ckpts")
    if sampler == "ddim":
        print(f"    DDIM POLICY: stochastic-DDIM chain, ~{policy_ddim_steps} steps, eta={eta} "
              f"(trajectory memory ~{policy_ddim_steps}/{t_start or 100} of the DDPM policy)", flush=True)
    if chain_starts:
        from ppo_claude import kchain_schedules
        _sch = kchain_schedules(chain_starts, policy_ddim_steps, ddim_stride)
        _steps = "+".join(str(len(s) - 1) for s in _sch)
        budget = (f"stride {ddim_stride} in t -> {_steps} policy steps" if ddim_stride else
                  f"{_steps} policy steps (~{policy_ddim_steps} total budget split "
                  f"proportionally to start t)")
        print(f"    MULTI-PHASE: K={len(chain_starts)} chains from S={list(chain_starts)} "
              f"(K=3 eval used S=[150,100,50]); {budget}; renoise between chains carries no "
              f"log-prob (theta-free forward q)", flush=True)
    stats_path = stats or cfg["stats"]                     # override -> extrapolated anchor (no target data)
    print(f"=== REGIME Re={re} | gt={cfg['gt']} train_seqs={cfg['train_seqs']} probe_seq={cfg['probe_seq']} "
          f"-> {save_dir} ===", flush=True)
    # Protocol v1.3 §2b pre-flight: an obs-fit anchor must match the LR of the data generation it
    # will train against; a fingerprint-less obs-fit anchor or a drifted one aborts the launch.
    import numpy as _np
    _stats_d = _np.load(stats_path)
    if "obs_lr_fingerprint" in _stats_d:
        from anchor_obsfit_builder import verify_freshness
        if not verify_freshness(stats_path, cfg["gt"], cfg["train_seqs"]):
            raise SystemExit(f"ABORT (v1.3 pre-flight): {stats_path} is stale for {cfg['gt']} — "
                             f"rebuild via src/ddpo_ft/anchor_obsfit_builder.py from the training LR")
    elif "obsfit" in stats_path:
        raise SystemExit(f"ABORT (v1.3 pre-flight): obs-fit anchor {stats_path} has no LR fingerprint "
                         f"(pre-v1.3 artifact) — rebuild via src/ddpo_ft/anchor_obsfit_builder.py")
    if stats:
        print(f"    ANCHOR OVERRIDE: stats={stats_path}  scales_re={scales_re or re}  "
              f"({'EXTRAPOLATED' if 'extrap' in stats_path else 'custom'} anchor — no target-regime data)", flush=True)

    # EXPERIMENT 2 (reweight): one-sided pde hinge + heavier spec_highk, drop w1. Targets the
    # spec-vs-pde landscape plateau — let the model add high-k energy without pde fighting it.
    # pde_weight > 1 -> "pde-heavy" mode: make the residual (dominated by the k<8 temporal-spatial
    # balance error, see diag decomposition) the DOMINANT reward term, with spec_highk as the
    # anti-smoothing guard.
    exp2_weights = {"spec": 0.5, "spec_highk": 3.0, "energy": 0.1, "w1": 0.0, "pde": pde_weight}
    if pde_weight != 1.0:
        print(f"    pde-heavy: pde weight={pde_weight} (target the large-scale temporal-balance residual; "
              f"spec_highk=3.0 guards the energy)", flush=True)
    if pde_local_weight > 0:                                # region-targeted residual-cleanup term
        exp2_weights["pde_local"] = pde_local_weight
        print(f"    pde_local ACTIVE: weight={pde_local_weight} patch={pde_local_patch}x{pde_local_patch} "
              f"frac={pde_local_frac} (targets worst-{int(pde_local_frac*100)}% {pde_local_patch}x{pde_local_patch} regions)",
              flush=True)
    if align_weight > 0:                                    # strain-orientation (filament coherence) term
        exp2_weights["align"] = align_weight
        print(f"    align ACTIVE: weight={align_weight} (push small-scale orientation stat toward GT "
              f"ref 0.289; base ~0.394, isotropic 0.5 — added energy must form strain-locked filaments)",
              flush=True)
    if spec_residual_weight > 0:                            # high-k residual-speckle term
        from rewards_claude import SPEC_RESID_FLOOR
        exp2_weights["spec_residual"] = spec_residual_weight
        print(f"    spec_residual ACTIVE: weight={spec_residual_weight} (penalize k>=32 NS-residual power "
              f"above GT floor {SPEC_RESID_FLOOR.get(re):.2e} — kill the residual speckle, keep the energy)",
              flush=True)
    if highk_lo != 32:
        print(f"    EXTENDED HIGH-K BAND: spec_highk over k in [{highk_lo}, 96) — the energy deficit "
              f"starts ~k=10 (deep-cascade spectra), not 32; same scale/weight, wider gradient coverage",
              flush=True)
    reward = Reward.from_calibration(
        stats_path, "base_results/reward_calibration.json",
        re=re, weights=exp2_weights, pde_hinge=not pde_two_sided, scales_re=scales_re,
        pde_local_frac=pde_local_frac, pde_local_patch=pde_local_patch, highk_band=(highk_lo, 96))
    print("reward weights:", reward.weights, f"| pde_hinge={not pde_two_sided}"
          + ("  [TWO-SIDED pde: pushes residual UP toward the floor when below it]" if pde_two_sided else ""), flush=True)

    nd = jax.local_device_count()
    print(f"data-parallel over {nd} device(s)", flush=True)
    if smoke:
        t_start, n_inputs, group_size, n_inner, temp, kl = t_start or 3, nd, 2, 2, 1.5, 0.01
        temp = sampling_temp if sampling_temp is not None else temp
        kl = kl_coef if kl_coef is not None else kl
        n_outer = n_outer or 2
    else:
        # t_start override: policy SDEdit start level (renoise-decorrelation study -> t=50 keeps 93%
        # of the DDIM recon's placement signal vs 59% at the default 100; chains half as long)
        t_start, n_inputs, group_size, n_inner, temp, kl = t_start or 100, nd, 8, 4, 1.5, 0.01
        temp = sampling_temp if sampling_temp is not None else temp
        kl = kl_coef if kl_coef is not None else kl
        n_outer = n_outer or 300
    if chain_starts:
        t_start = int(chain_starts[0])    # multi-phase: SDEdit level = first chain's start

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    ddpm, base_orig, _ = build_base_ddpm()

    # EXPERIMENT: base-DDIM init. Instead of feeding the raw low-res as the conditioning field, first
    # reconstruct it with a FROZEN-base deterministic DDIM chain (SDEdit t=ddim_t_start -> 0, ddim_steps,
    # single chain) and finetune off THAT on-manifold reconstruction. Pool + probe are transformed once.
    base_denoise = None
    if base_ddim_init:
        from ppo_claude import build_ddim_denoiser
        ab = ddpm.alpha_bar
        _ddim = build_ddim_denoiser(ddpm.unet, ab, ddim_t_start, ddim_steps)
        _sa, _s1 = float(jnp.sqrt(ab[ddim_t_start])), float(jnp.sqrt(1.0 - ab[ddim_t_start]))
        _dkey = jax.random.PRNGKey(12345)

        def base_denoise(pool, _bs=16):
            out = []
            for i in range(0, len(pool), _bs):
                xb = jnp.asarray(pool[i:i + _bs], dtype=jnp.float32)
                xs = _sa * xb + _s1 * jax.random.normal(jax.random.fold_in(_dkey, i), xb.shape)
                out.append(np.asarray(_ddim(base_orig, xs)))
            return np.concatenate(out, axis=0)
        print(f"    BASE-DDIM INIT: SDEdit t={ddim_t_start} -> DDIM {ddim_steps} steps (eta=0, single chain) "
              f"with frozen base; finetune starts from the base reconstruction, not raw low-res", flush=True)

    if resume:                                                # continue a checkpoint (keep TRUE base for KL)
        ck = pickle.load(open(resume, "rb"))
        start_iter = ck["iter"] + 1
        # fresh_opt: resume PARAMS but discard the checkpoint's Adam state. Needed when the
        # operating point shifts (e.g. temperature anneal): stale second moments adapted to the
        # old gradient landscape drove the anneal@1.5 collapse (probe 0.24->0.12 in 30 iters).
        _opt_state = None if fresh_opt else ck["opt_state"]
        if fresh_opt:
            print("    FRESH OPTIMIZER: params resumed, Adam state discarded", flush=True)
        trainer = DDPOTrainer(ddpm, ck["params"], reward, optimizer, group_size=group_size,
                              t_start=t_start, clip_eps=clip_eps, kl_coef=kl, n_inner=n_inner,
                              seed=(seed if seed is not None else start_iter),
                              sampling_temp=temp, base_params=base_orig, opt_state=_opt_state,
                              sampler=sampler, ddim_steps=policy_ddim_steps, eta=eta,
                              chain_starts=chain_starts, ddim_stride=ddim_stride)
        print(f"RESUMED from {resume} at iter {start_iter}", flush=True)
    else:
        start_iter = 0
        trainer = DDPOTrainer(ddpm, base_orig, reward, optimizer, group_size=group_size,
                              t_start=t_start, clip_eps=clip_eps, kl_coef=kl, n_inner=n_inner,
                              seed=(seed if seed is not None else 0),
                              sampling_temp=temp, sampler=sampler, ddim_steps=policy_ddim_steps, eta=eta,
                              chain_starts=chain_starts, ddim_stride=ddim_stride)

    inputs = sparse_input_iterator(cfg["gt"], seqs=cfg["train_seqs"], n_inputs=n_inputs,
                                    grid_factor=grid_factor, base_denoise=base_denoise)
    samp = (f"ddim({policy_ddim_steps} steps, eta={eta}"
            f"{f', S={list(chain_starts)}' if chain_starts else ''})" if sampler == "ddim" else "ddpm")
    print(f"\n=== DDPO {'SMOKE' if smoke else 'run'}: sampler={samp} t_start={t_start} B={n_inputs*group_size} "
          f"n_inner={n_inner} start={start_iter} n_outer={n_outer} lr={lr} temp={temp} kl={kl} "
          f"save/{save_every} eval/{eval_every} ===", flush=True)

    # live GT-retention probe (held-out test seq 36); base reference computed once
    spec_fn = make_spectrum_fn(N)
    hik_ret = lambda recon, gt: float((np.asarray(spec_fn(recon))[:, 32:].sum(-1)
                                       / np.asarray(spec_fn(gt))[:, 32:].sum(-1)).mean())
    # --no_gt_probe: skip the record-only GT probe entirely. It never influenced training (no
    # gradient, no checkpoint selection) but it DOES read ground truth, so a GT-free run must not
    # run it at all. Also avoids RE_CFG's probe_seq pointing past the end of a regenerated file.
    if no_gt_probe:
        probe_inp = probe_gt = base_hik = None
        print("    GT PROBE DISABLED — no ground truth is read anywhere in this run", flush=True)
    else:
        probe_inp, probe_gt = make_gt_probe(seq_id=cfg["probe_seq"], gt_path=cfg["gt"],
                                            grid_factor=grid_factor, base_denoise=base_denoise)
    base_hik = None if (smoke or no_gt_probe) else hik_ret(
        trainer.probe_x0(probe_inp, jax.random.PRNGKey(7),
                         params=jax.tree_util.tree_map(lambda a: a[0], trainer.base_params)), probe_gt)
    if base_hik is not None:
        print(f"GT-probe base hik_ret = {base_hik:.3f} (want DDPO to exceed this)", flush=True)

    # ---- provenance: every checkpoint dir carries the FULL run config (anti-confounding — the
    # random-1024-vs-grid4x k2 mix-up of 2026-07-14 must not repeat). config.json is written at
    # launch, before any checkpoint, and includes the git commit of the code that ran.
    if not smoke:
        import subprocess
        try:
            git_rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                              cwd=_ROOT, text=True).strip()
        except Exception:
            git_rev = None
        run_config = dict(
            re=re, gt=cfg["gt"], train_seqs=cfg["train_seqs"], probe_seq=cfg["probe_seq"],
            degradation=("grid" + str(grid_factor) + "x" if grid_factor else "random-1024"),
            grid_factor=grid_factor, base_ddim_init=base_ddim_init,
            ddim_init=dict(steps=ddim_steps, t_start=ddim_t_start) if base_ddim_init else None,
            policy=dict(sampler=sampler, t_start=t_start,
                        ddim_steps=policy_ddim_steps if sampler == "ddim" else None,
                        eta=eta if sampler == "ddim" else None,
                        chain_starts=list(chain_starts) if chain_starts else None,
                        ddim_stride=ddim_stride),
            reward=dict(weights=dict(reward.weights), scales=dict(reward.scales),
                        highk_band=[highk_lo, 96], pde_hinge=not pde_two_sided,
                        stats=stats_path, scales_re=scales_re),
            train=dict(lr=lr, n_outer=n_outer, group_size=group_size, n_inner=n_inner,
                       sampling_temp=temp, kl_coef=kl, resume=resume,
                       policy_ema=policy_ema or None, clip_eps=clip_eps, seed=seed),
            git_commit=git_rev, launched="provenance-v1")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(run_config, f, indent=2, default=str)
        print(f"    run config -> {save_dir}/config.json", flush=True)

    metrics_path = None if smoke else os.path.join(save_dir, "metrics.jsonl")
    if metrics_path:
        os.makedirs(save_dir, exist_ok=True)
        open(metrics_path, "a" if resume else "w").close()

    # optional EMA over policy weights: per-OUTER-iteration shadow (not per grad step) to average
    # PPO plateau noise; off by default (policy_ema=0.0). Shadow is replicated like trainer.params.
    ema_params = None
    if policy_ema > 0.0:
        import jax as _jax
        ema_params = _jax.tree_util.tree_map(lambda a: a, trainer.params)
        print(f"    POLICY EMA: shadow weights, mu={policy_ema} per outer iter "
              f"(~{1.0/(1.0-policy_ema):.0f}-iter horizon); checkpoints carry ema_params+ema_rate", flush=True)

    anchor_score = None
    if anchor_monitor_every and not smoke:
        anchor_score = build_anchor_monitor(re, cfg, stats_path, grid_factor, base_orig, ddpm)
        if anchor_band:
            print(f"    R6 EARLY STOP: halt once the anchor score sits in "
                  f"[{anchor_band[0]:.3f}, {anchor_band[1]:.3f}] for {anchor_patience} consecutive "
                  f"checks (every {anchor_monitor_every} iters)", flush=True)
    in_band = 0
    stopped_early = False

    for outer in range(n_outer):
        gi = start_iter + outer
        m = trainer.train_iter(next(inputs))
        if ema_params is not None:
            import jax as _jax
            ema_params = _jax.tree_util.tree_map(
                lambda e, p: policy_ema * e + (1.0 - policy_ema) * p, ema_params, trainer.params)
        comp = "  ".join(f"{k}={v:.3f}" for k, v in m["components"].items())
        print(f"[{gi:04d}] R={m['reward_mean']:.3f}±{m['reward_std']:.3f} gstd={m['group_r_std']:.3f} "
              f"|A|={m['adv_abs_mean']:.2f} loss {m['loss_first']:.3f}->{m['loss_last']:.3f}  {comp}",
              flush=True)
        rec = {"iter": gi, **{k: v for k, v in m.items() if k != "components"},
               **{f"c_{k}": v for k, v in m["components"].items()}}
        if not smoke and not no_gt_probe and (gi + 1) % eval_every == 0:   # live GT-retention probe
            hr = hik_ret(trainer.probe_x0(probe_inp, jax.random.PRNGKey(gi)), probe_gt)
            rec["gt_hik_ret"] = hr
            print(f"    [GTeval iter={gi}] hik_ret={hr:.3f}  (base {base_hik:.3f}, Δ{hr - base_hik:+.3f})", flush=True)
        if anchor_score is not None and (gi + 1) % anchor_monitor_every == 0:
            _p = jax.tree_util.tree_map(lambda a: a[0], trainer.params)
            sc = anchor_score(_p)
            rec["anchor_score"] = sc
            hit = bool(anchor_band and anchor_band[0] <= sc <= anchor_band[1])
            in_band = in_band + 1 if hit else 0
            print(f"    [R6 anchor iter={gi}] deployed/anchor = {sc:.3f}"
                  + (f"  IN BAND ({in_band}/{anchor_patience})" if hit else ""), flush=True)
            if anchor_band and in_band >= anchor_patience:
                print(f"    [R6 EARLY STOP] anchor score in band for {anchor_patience} consecutive "
                      f"checks -> stopping at iter {gi}", flush=True)
                stopped_early = True
                save_ckpt(trainer.params, trainer.opt_state, gi, save_dir,
                          ema_params=ema_params, ema_rate=policy_ema if ema_params is not None else None)
                if metrics_path:
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                break
        if metrics_path:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if not smoke and (gi + 1) % save_every == 0:
            save_ckpt(trainer.params, trainer.opt_state, gi, save_dir,
                      ema_params=ema_params, ema_rate=policy_ema if ema_params is not None else None)

    if not smoke and not stopped_early:
        save_ckpt(trainer.params, trainer.opt_state, start_iter + n_outer - 1, save_dir,
                  ema_params=ema_params, ema_rate=policy_ema if ema_params is not None else None)
    print("\nDONE.", flush=True)
    return trainer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to validate the loop")
    ap.add_argument("--n_outer", type=int, default=None, help="number of outer iterations")
    ap.add_argument("--lr", type=float, default=5e-5, help="Adam learning rate")
    ap.add_argument("--resume", type=str, default=None, help="checkpoint .pkl to continue from")
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=10, help="live GT-retention probe interval")
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--re", type=int, default=1000, help="target regime (1000 in-dist, 2000/500 OOD)")
    ap.add_argument("--stats", type=str, default=None,
                    help="override reward anchor npz (e.g. extrapolated anchor -> zero target-regime data)")
    ap.add_argument("--scales_re", type=int, default=None,
                    help="pull per-component scales from THIS regime instead of --re (owned normalizers)")
    ap.add_argument("--pde_local_weight", type=float, default=0.0,
                    help="weight for the region-targeted residual-cleanup term (0 = off)")
    ap.add_argument("--pde_local_patch", type=int, default=2, help="PxP region size for pde_local (2..4)")
    ap.add_argument("--pde_local_frac", type=float, default=0.1, help="worst-fraction of regions targeted")
    ap.add_argument("--align_weight", type=float, default=0.0,
                    help="weight for the strain-orientation (filament coherence) term (0 = off)")
    ap.add_argument("--spec_residual_weight", type=float, default=0.0,
                    help="weight for the high-k residual-speckle term (k>=32 residual power; 0 = off)")
    ap.add_argument("--pde_weight", type=float, default=1.0,
                    help="weight for the pde residual term (>1 = pde-heavy: attack the temporal-balance residual)")
    ap.add_argument("--grid_factor", type=int, default=None,
                    help="clean grid-N downsample for the INPUT instead of random-1024 (e.g. 4 -> 4096 pts)")
    ap.add_argument("--base_ddim_init", action="store_true",
                    help="finetune off the frozen-base DDIM reconstruction of the low-res, not the raw low-res")
    ap.add_argument("--ddim_steps", type=int, default=20, help="DDIM steps for the base pre-denoise (eta=0)")
    ap.add_argument("--ddim_t_start", type=int, default=100, help="SDEdit start level for the base pre-denoise")
    ap.add_argument("--t_start", type=int, default=None,
                    help="policy SDEdit start level (default 100; renoise study -> 50 keeps 93% of the "
                         "DDIM recon's placement signal and halves the chain)")
    ap.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"],
                    help="policy chain: 'ddpm' (every step) or 'ddim' (stochastic DDIM over "
                         "~policy_ddim_steps subsampled steps; shorter trajectory, less memory)")
    ap.add_argument("--policy_ddim_steps", type=int, default=20,
                    help="number of subsampled steps for the DDIM POLICY (distinct from --ddim_steps, "
                         "which is the eta=0 base pre-denoise of --base_ddim_init)")
    ap.add_argument("--eta", type=float, default=1.0,
                    help="DDIM policy stochasticity in (0, 1]; eta=1 = DDPM posterior std on the coarse "
                         "schedule. Must be > 0 (eta=0 is deterministic -> no valid log-prob/ratio)")
    ap.add_argument("--chain_starts", type=int, nargs="+", default=None,
                    help="multi-phase renoise-and-denoise starts, strictly descending (DDIM policy "
                         "only), e.g. '--chain_starts 150 100' for K=2 or '150 100 50' = the K=3 eval "
                         "S. Overrides --t_start with the first value; policy_ddim_steps is the TOTAL "
                         "budget split proportionally to start t. Renoise between chains carries no "
                         "log-prob/gradient. NOT group_size (samples-per-input)")
    ap.add_argument("--ddim_stride", type=int, default=None,
                    help="fixed t-interval per DDIM policy step (e.g. 10 -> chain from S runs "
                         "[S, S-10, ..., 10, 1] = S/10 steps). Overrides policy_ddim_steps. With "
                         "'--chain_starts 150 100 50 --ddim_stride 10' -> 15+10+5=30 policy steps")
    ap.add_argument("--highk_lo", type=int, default=32,
                    help="lower edge of the spec_highk reward band (default 32; 10 = extend to where "
                         "the energy deficit starts)")
    ap.add_argument("--policy_ema", type=float, default=0.0,
                    help="OPTIONAL EMA over policy weights, applied once per outer iteration "
                         "(e.g. 0.99 ~ 100-iter horizon). 0 = off (default). Checkpoints then carry "
                         "ema_params+ema_rate under named keys; eval can compare shadow vs online.")
    ap.add_argument("--fresh_opt", action="store_true",
                    help="on --resume: keep params but discard the checkpoint's Adam state "
                         "(for operating-point shifts like a temperature anneal).")
    ap.add_argument("--pde_two_sided", action="store_true",
                    help="use the TWO-SIDED pde log-ratio (ln R^2 - ln floor)^2 instead of the "
                         "one-sided hinge: actively pushes residual UP toward the floor when the "
                         "solution is too smooth (the hinge is slack below the floor and cannot).")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for rollouts (default: start_iter on resume, else 0). Set "
                         "explicitly for seed-repeat variance runs.")
    ap.add_argument("--sampling_temp", type=float, default=None,
                    help="rollout sampling temperature (default 1.5). Higher -> more trajectory "
                         "diversity -> larger within-group advantage spread (stronger signal).")
    ap.add_argument("--kl_coef", type=float, default=None,
                    help="KL leash to the base policy (default 0.01). Lower -> policy free to travel "
                         "further from base.")
    ap.add_argument("--anchor_monitor_every", type=int, default=0,
                    help="R6: every N outer iters, score the DEPLOYMENT cascade (K3 [150,100,50] x86, "
                         "lam3, itemp 0.30) on a fixed train-pool batch against the anchor. 0 = off "
                         "(legacy behaviour: GT probe only, post-hoc R3.2 selection).")
    ap.add_argument("--anchor_band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="R6 early-stop band for the anchor score; stop once inside it for "
                         "--anchor_patience consecutive checks. Requires --anchor_monitor_every.")
    ap.add_argument("--no_gt_probe", action="store_true",
                    help="skip the record-only GT probe so the run reads NO ground truth")
    ap.add_argument("--gt_override", type=str, default=None,
                    help="use this GT file instead of RE_CFG[re]['gt'] (physics still from --re)")
    ap.add_argument("--train_seqs_override", type=str, default=None,
                    help="comma-separated training sequence indices, overriding RE_CFG")
    ap.add_argument("--anchor_patience", type=int, default=2,
                    help="consecutive in-band checks required to stop (guards the ~0.02 check noise)")
    ap.add_argument("--clip_eps", type=float, default=0.2,
                    help="PPO clip epsilon (ratio clamp 1+-eps). Default 0.2 (standard). Wider "
                         "(0.3-0.4) allows larger per-update policy steps -> faster travel when the "
                         "climb is speed-limited, at higher instability risk.")
    main(**vars(ap.parse_args()))
