"""Within-group high-k exploration diagnostic: can the policy's rollout group DISAGREE in the
reward band? For each input (DDIM-reconstructed, as in --base_ddim_init training), sample a group of
G rollouts from the SAME start field (different noise) with the FROZEN BASE params — the policy at
finetune init — at each (t_start, temp) config. Report the within-group std / CV of the high-k
energy sum E(k>=32) per sample: this is the advantage signal available to PPO along the spec_highk
axis. Low CV at t=50 + temp 1.5 (vs t=100) = the exploration constraint that stalls the t=50 run;
whether temp 2.5 restores it discriminates the "more std" fix from the "deeper t_start" fix.

    python -m src.ddpo_ft.diag_group_hik_std [--seqs 32,36,38] [--n_per_seq 2] [--group 8]
        [--re 1000] [--grid_factor 4] [--configs 50:1.5,75:1.5,100:1.5,50:2.5,75:2.5,100:2.5]
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax                                        # noqa: E402
import jax.numpy as jnp                           # noqa: E402
import numpy as np                                # noqa: E402

from diag_guided_residual import make_guided_sampler  # noqa: E402
from ppo_claude import build_ddim_denoiser        # noqa: E402
from train_claude import build_base_ddpm          # noqa: E402
from src.physics_guidance import make_dx_func     # noqa: E402
from src.rewards import make_spectrum_fn          # noqa: E402
from src.sequence_inference import (              # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence)

MEAN, STD, N, HIK0 = 0.0, 4.7988, 256, 32
GT_FILE = {1000: "flow-data/kf_2d_re1000_256_40seed.npy", 2000: "flow-data/kf_re2000_256_20seed.npy"}


def main(seqs="32,36,38", n_per_seq=2, group=8, re=1000, grid_factor=4,
         configs="50:1.5,75:1.5,100:1.5,50:2.5,75:2.5,100:2.5", seed=0):
    seqs = [int(x) for x in seqs.split(",")]
    cfgs = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in configs.split(",")]
    ddpm, base_params, _ = build_base_ddpm()
    ab = ddpm.alpha_bar
    ddim = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
    sa100, s1100 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
    dx = make_dx_func(n=N, re=float(re), std=STD, mean=MEAN)
    spec_fn = make_spectrum_fn(N)

    xin, xgt = [], []
    for s in seqs:
        seq = load_sequence(GT_FILE[re], s)
        g = build_triplets(seq, MEAN, STD)
        l = build_triplets(grid_downsample_degrade(seq, grid_factor), MEAN, STD)
        idx = np.linspace(len(g) // 4, len(g) - 1, n_per_seq).astype(int)
        xin.append(l[idx]); xgt.append(g[idx])
    xin, xgt = np.concatenate(xin), np.concatenate(xgt)
    key = jax.random.PRNGKey(12345)
    xdd = np.asarray(ddim(base_params, sa100 * jnp.asarray(xin) + s1100 * jax.random.normal(key, xin.shape)))
    hik_gt = np.asarray(spec_fn(jnp.asarray(xgt)))[:, HIK0:].sum(-1)
    hik_dd = np.asarray(spec_fn(jnp.asarray(xdd)))[:, HIK0:].sum(-1)
    print(f"\n=== group hi-k exploration diag: re={re} {len(xin)} inputs (seqs {seqs}), group={group}, "
          f"FROZEN BASE params, DDIM-recon inputs ===", flush=True)
    print(f"GT E_hik (per-input mean) {hik_gt.mean():.3e} | DDIM recon {hik_dd.mean():.3e} "
          f"(ratio {hik_dd.mean()/hik_gt.mean():.3f})", flush=True)
    print(f"{'t_start':>8}{'temp':>6} | {'E_hik grp-mean':>15}{'grp-std':>12}{'CV %':>7}"
          f"{'ret vs GT':>11}{'ret spread':>11}", flush=True)

    for t_start, temp in cfgs:
        smp = make_guided_sampler(ddpm.unet, ab, ddpm.beta_schedule, t_start, dx, 0.0, temp)
        sat, s1t = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))
        stds, means, rets, ret_sprd = [], [], [], []
        for i in range(len(xdd)):
            xc = jnp.tile(jnp.asarray(xdd[i:i + 1]), (group, 1, 1, 1))     # same input, G chains
            k1, k2 = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), 1000 * t_start + i))
            x0 = np.asarray(smp(base_params, sat * xc + s1t * jax.random.normal(k1, xc.shape), k2))
            h = np.asarray(spec_fn(jnp.asarray(x0)))[:, HIK0:].sum(-1)     # (G,) per-sample hi-k energy
            stds.append(h.std()); means.append(h.mean())
            r = h / hik_gt[i]
            rets.append(r.mean()); ret_sprd.append(r.max() - r.min())
        cv = 100.0 * np.mean(stds) / np.mean(means)
        print(f"{t_start:>8}{temp:>6.1f} | {np.mean(means):>15.3e}{np.mean(stds):>12.3e}{cv:>7.2f}"
              f"{np.mean(rets):>11.3f}{np.mean(ret_sprd):>11.3f}", flush=True)
    print("\nCV = within-group std/mean of E(k>=32): the PPO advantage signal along the spec_highk axis.\n"
          "ret spread = max-min of hik retention within a group (how different the best/worst sample are).",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=str, default="32,36,38")
    ap.add_argument("--n_per_seq", type=int, default=2)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--re", type=int, default=1000)
    ap.add_argument("--grid_factor", type=int, default=4)
    ap.add_argument("--configs", type=str, default="50:1.5,75:1.5,100:1.5,50:2.5,75:2.5,100:2.5")
    ap.add_argument("--seed", type=int, default=0)
    main(**vars(ap.parse_args()))
