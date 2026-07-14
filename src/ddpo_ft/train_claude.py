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
    2000: dict(gt="flow-data/kf_re2000_256_20seed.npy",   stats="base_results/regime_stats_re2000.npz",
               train_seqs=[8, 9, 10, 11], probe_seq=0),
    500:  dict(gt="flow-data/kf_re500_256_20seed.npy",    stats="base_results/regime_stats_re500.npz",
               train_seqs=[8, 9, 10, 11], probe_seq=0),
}


def build_base_ddpm(config_path="configs/config.yaml"):
    """Build the base (unconditional) DDPM + load base params. Conditioning forced off so the plain
    Unet matches ckpt_0299."""
    cfg = copy.deepcopy(yaml.safe_load(open(config_path)))
    cfg["conditioning"]["train"]["enabled"] = False
    cfg["conditioning"]["inference"]["enabled"] = False
    ddpm = DDPM(cfg)
    params, _, epoch = load_checkpoint("checkpoints/ddpm/ckpt_epoch_0299.pkl")
    print(f"base DDPM: {type(ddpm.unet).__name__} | epoch {epoch} | T={cfg['diffusion']['T']}", flush=True)
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


def save_ckpt(params, opt_state, outer, save_dir):
    import pickle
    import jax
    os.makedirs(save_dir, exist_ok=True)
    # params/opt_state are REPLICATED across devices (pmap) -> save a single replica
    unrep = lambda t: jax.tree_util.tree_map(lambda a: np.asarray(a[0]), t)
    path = os.path.join(save_dir, f"ddpo_re1000_iter{outer:04d}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"params": unrep(params), "opt_state": unrep(opt_state), "iter": outer}, f)
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


def main(smoke=False, n_outer=None, save_dir=None, save_every=20,
         eval_every=10, lr=5e-5, resume=None, re=1000, stats=None, scales_re=None,
         pde_local_weight=0.0, pde_local_patch=2, pde_local_frac=0.1, align_weight=0.0,
         spec_residual_weight=0.0, pde_weight=1.0, grid_factor=None,
         base_ddim_init=False, ddim_steps=20, ddim_t_start=100, t_start=None,
         sampler="ddpm", policy_ddim_steps=20, eta=1.0, chain_starts=None, ddim_stride=None,
         highk_lo=32):
    import json
    import pickle
    import jax
    from src.rewards import make_spectrum_fn

    cfg = RE_CFG[re]
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
        re=re, weights=exp2_weights, pde_hinge=True, scales_re=scales_re,
        pde_local_frac=pde_local_frac, pde_local_patch=pde_local_patch, highk_band=(highk_lo, 96))
    print("reward weights:", reward.weights, "| pde_hinge=True", flush=True)

    nd = jax.local_device_count()
    print(f"data-parallel over {nd} device(s)", flush=True)
    if smoke:
        t_start, n_inputs, group_size, n_inner, temp, kl = t_start or 3, nd, 2, 2, 1.5, 0.01
        n_outer = n_outer or 2
    else:
        # t_start override: policy SDEdit start level (renoise-decorrelation study -> t=50 keeps 93%
        # of the DDIM recon's placement signal vs 59% at the default 100; chains half as long)
        t_start, n_inputs, group_size, n_inner, temp, kl = t_start or 100, nd, 8, 4, 1.5, 0.01
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
        trainer = DDPOTrainer(ddpm, ck["params"], reward, optimizer, group_size=group_size,
                              t_start=t_start, clip_eps=0.2, kl_coef=kl, n_inner=n_inner, seed=start_iter,
                              sampling_temp=temp, base_params=base_orig, opt_state=ck["opt_state"],
                              sampler=sampler, ddim_steps=policy_ddim_steps, eta=eta,
                              chain_starts=chain_starts, ddim_stride=ddim_stride)
        print(f"RESUMED from {resume} at iter {start_iter}", flush=True)
    else:
        start_iter = 0
        trainer = DDPOTrainer(ddpm, base_orig, reward, optimizer, group_size=group_size,
                              t_start=t_start, clip_eps=0.2, kl_coef=kl, n_inner=n_inner, seed=0,
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
    probe_inp, probe_gt = make_gt_probe(seq_id=cfg["probe_seq"], gt_path=cfg["gt"], grid_factor=grid_factor,
                                        base_denoise=base_denoise)
    base_hik = None if smoke else hik_ret(
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
                        highk_band=[highk_lo, 96], pde_hinge=True,
                        stats=stats_path, scales_re=scales_re),
            train=dict(lr=lr, n_outer=n_outer, group_size=group_size, n_inner=n_inner,
                       sampling_temp=temp, kl_coef=kl, resume=resume),
            git_commit=git_rev, launched="provenance-v1")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(run_config, f, indent=2, default=str)
        print(f"    run config -> {save_dir}/config.json", flush=True)

    metrics_path = None if smoke else os.path.join(save_dir, "metrics.jsonl")
    if metrics_path:
        os.makedirs(save_dir, exist_ok=True)
        open(metrics_path, "a" if resume else "w").close()

    for outer in range(n_outer):
        gi = start_iter + outer
        m = trainer.train_iter(next(inputs))
        comp = "  ".join(f"{k}={v:.3f}" for k, v in m["components"].items())
        print(f"[{gi:04d}] R={m['reward_mean']:.3f}±{m['reward_std']:.3f} gstd={m['group_r_std']:.3f} "
              f"|A|={m['adv_abs_mean']:.2f} loss {m['loss_first']:.3f}->{m['loss_last']:.3f}  {comp}",
              flush=True)
        rec = {"iter": gi, **{k: v for k, v in m.items() if k != "components"},
               **{f"c_{k}": v for k, v in m["components"].items()}}
        if not smoke and (gi + 1) % eval_every == 0:          # live GT-retention probe
            hr = hik_ret(trainer.probe_x0(probe_inp, jax.random.PRNGKey(gi)), probe_gt)
            rec["gt_hik_ret"] = hr
            print(f"    [GTeval iter={gi}] hik_ret={hr:.3f}  (base {base_hik:.3f}, Δ{hr - base_hik:+.3f})", flush=True)
        if metrics_path:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if not smoke and (gi + 1) % save_every == 0:
            save_ckpt(trainer.params, trainer.opt_state, gi, save_dir)

    if not smoke:
        save_ckpt(trainer.params, trainer.opt_state, start_iter + n_outer - 1, save_dir)
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
    main(**vars(ap.parse_args()))
