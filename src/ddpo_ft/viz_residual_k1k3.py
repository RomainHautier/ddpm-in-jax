"""Does DDPO make things better? Per frame, reconstruction ERROR (|recon−GT|) and PDE RESIDUAL
(|NS-residual|) for base / DDPO-K=1 / DDPO-K=3, all against GT:

  GT ω | |base−GT| | |DDPO-K1−GT| | |DDPO-K3−GT| | resid(base) | resid(K1) | resid(K3) | resid(GT)

Error cols share a scale; residual cols share a scale. Panels annotate RMS error and mean |resid|,
so you can read straight off whether DDPO shrinks the error/residual or not.

    python -m src.ddpo_ft.viz_residual_k1k3 <ddpo_ckpt.pkl> [--seq 36] [--frames 4] [--grid_factor 4]
"""
import argparse
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax                                        # noqa: E402
import jax.numpy as jnp                           # noqa: E402
import numpy as np                                # noqa: E402
import matplotlib.pyplot as plt                   # noqa: E402

from eval_ddpo import make_sampler                # noqa: E402
from train_claude import build_base_ddpm          # noqa: E402
from src.physics_guidance import make_ns_residual  # noqa: E402
from src.sequence_inference import (              # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence, sparse_nnfill_degrade)

MEAN, STD, N = 0.0, 4.7988, 256
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"
S_MULTI = [150, 100, 50]


def main(ckpt, seq=36, frames=4, t_start=100, grid_factor=4, seed=1,
         out="monitoring/ab_pdelocal/viz_residual_k1k3.png", gt=None, re=1000):
    ddpm, base_params, _ = build_base_ddpm()
    ddpo = pickle.load(open(ckpt, "rb"))["params"]
    ab = ddpm.alpha_bar
    resid = jax.jit(make_ns_residual(n=N, re=float(re)))
    samplers = {t: make_sampler(ddpm.unet, ab, ddpm.beta_schedule, t, temp=1.0)
                for t in sorted({t_start, *S_MULTI})}
    key = jax.random.PRNGKey(seed)

    s = load_sequence(gt or GT_PATH, seq)
    degraded = grid_downsample_degrade(s, grid_factor) if grid_factor else sparse_nnfill_degrade(s, seq)
    inp = build_triplets(degraded, MEAN, STD)
    gt = build_triplets(s, MEAN, STD)
    idx = np.linspace(len(inp) // 4, len(inp) - 1, frames).astype(int)
    xin, xgt = jnp.asarray(inp[idx]), np.asarray(gt[idx])
    print(f"reconstructing base/DDPO-K1/DDPO-K3, {frames} frames seq {seq} (grid_factor={grid_factor}) ...", flush=True)

    def noise_to(x, t, k):
        return float(jnp.sqrt(ab[t])) * x + float(jnp.sqrt(1.0 - ab[t])) * jax.random.normal(k, x.shape)

    def one(p):
        nonlocal key; key, k1, k2 = jax.random.split(key, 3)
        return np.asarray(samplers[t_start](p, noise_to(xin, t_start, k1), k2))

    def k3(p):
        nonlocal key; xc = xin
        for Sj in S_MULTI:
            key, k1, k2 = jax.random.split(key, 3)
            xc = jnp.asarray(samplers[Sj](p, noise_to(xc, Sj, k1), k2))
        return np.asarray(xc)

    base_r, k1_r, k3_r = one(base_params), one(ddpo), k3(ddpo)
    Rf = lambda x0: np.abs(np.asarray(resid(jnp.asarray(x0) * STD)))
    Rb, R1, R3, Rg = Rf(base_r), Rf(k1_r), Rf(k3_r), Rf(np.asarray(xgt))
    wg = xgt[..., 1] * STD
    eb = np.abs(base_r[..., 1] * STD - wg); e1 = np.abs(k1_r[..., 1] * STD - wg); e3 = np.abs(k3_r[..., 1] * STD - wg)
    rms = lambda a: float(np.sqrt((a ** 2).mean()))
    print(f"RMS error   base {rms(eb):.3f}  K=1 {rms(e1):.3f}  K=3 {rms(e3):.3f}", flush=True)
    print(f"mean |resid| base {Rb.mean():.2f}  K=1 {R1.mean():.2f}  K=3 {R3.mean():.2f}  GT {Rg.mean():.2f}", flush=True)

    fig, axes = plt.subplots(frames, 8, figsize=(21, 2.7 * frames))
    axes = np.atleast_2d(axes)
    for r in range(frames):
        vv = np.percentile(np.abs(wg[r]), 99)
        vE = np.percentile(np.concatenate([eb[r], e1[r], e3[r]]).ravel(), 99)
        vR = np.percentile(np.concatenate([Rb[r], R1[r], R3[r], Rg[r]]).ravel(), 99)
        cells = [("GT ω", wg[r], "RdBu_r", -vv, vv),
                 (f"|base−GT| {rms(eb[r]):.2f}", eb[r], "magma", 0, vE),
                 (f"|K=1−GT| {rms(e1[r]):.2f}", e1[r], "magma", 0, vE),
                 (f"|K=3−GT| {rms(e3[r]):.2f}", e3[r], "magma", 0, vE),
                 (f"resid base {Rb[r].mean():.1f}", Rb[r], "inferno", 0, vR),
                 (f"resid K=1 {R1[r].mean():.1f}", R1[r], "inferno", 0, vR),
                 (f"resid K=3 {R3[r].mean():.1f}", R3[r], "inferno", 0, vR),
                 (f"resid GT {Rg[r].mean():.1f}", Rg[r], "inferno", 0, vR)]
        for c, (nm, d, cm, vmn, vmx) in enumerate(cells):
            a = axes[r, c]; a.imshow(d, cmap=cm, vmin=vmn, vmax=vmx)
            a.set_xticks([]); a.set_yticks([]); a.set_title(nm, fontsize=8.5)
        axes[r, 0].set_ylabel(f"frame {idx[r]}", fontsize=9)
    fig.suptitle(f"grid-4× — reconstruction error & PDE residual vs GT (seq {seq})   "
                 f"[RMS err: base {rms(eb):.2f} / K1 {rms(e1):.2f} / K3 {rms(e3):.2f}  |  "
                 f"mean resid: base {Rb.mean():.1f} / K1 {R1.mean():.1f} / K3 {R3.mean():.1f} / GT {Rg.mean():.1f}]",
                 y=1.005, fontsize=10.5)
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seq", type=int, default=36)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--t_start", type=int, default=100)
    ap.add_argument("--grid_factor", type=int, default=4)
    ap.add_argument("--out", type=str, default="monitoring/ab_pdelocal/viz_residual_k1k3.png")
    ap.add_argument("--gt", type=str, default=None, help="GT .npy (default Re=1000)")
    ap.add_argument("--re", type=int, default=1000, help="Reynolds for the NS residual operator")
    main(**vars(ap.parse_args()))
