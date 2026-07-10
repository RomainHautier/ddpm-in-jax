"""Visual K=1 vs K=3 comparison for a finetuned model on the SAME frames. Per frame, 8 panels:
  GT | K=1 recon | K=3 recon | E_hik(GT) | E_hik(K=1) | E_hik(K=3) | |K=1−GT| | |K=3−GT|
to see whether the K=3 overshoot shows up as denser high-k structure and where the extra error lands.

    python -m src.ddpo_ft.viz_k1k3 <ckpt.pkl> [--seq 36] [--frames 4] [--grid_factor 4]
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
from viz_energy import local_hik_energy           # noqa: E402
from src.rewards import make_spectrum_fn          # noqa: E402
from src.sequence_inference import (              # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence, sparse_nnfill_degrade)

MEAN, STD, N, HIK0 = 0.0, 4.7988, 256, 32
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"
S_MULTI = [150, 100, 50]


def main(ckpt, seq=36, frames=4, t_start=100, grid_factor=4, seed=1, kcut=32, sigma=6.0,
         out="monitoring/ab_pdelocal/viz_k1k3.png", gt=None):
    ddpm, _, _ = build_base_ddpm()
    params = pickle.load(open(ckpt, "rb"))["params"]
    ab = ddpm.alpha_bar
    spec_fn = make_spectrum_fn(N)
    samplers = {t: make_sampler(ddpm.unet, ab, ddpm.beta_schedule, t, temp=1.0)
                for t in sorted({t_start, *S_MULTI})}
    key = jax.random.PRNGKey(seed)

    s = load_sequence(gt or GT_PATH, seq)
    degraded = grid_downsample_degrade(s, grid_factor) if grid_factor else sparse_nnfill_degrade(s, seq)
    inp = build_triplets(degraded, MEAN, STD)
    gtt = build_triplets(s, MEAN, STD)
    idx = np.linspace(len(inp) // 4, len(inp) - 1, frames).astype(int)
    xin, xgt = jnp.asarray(inp[idx]), np.asarray(gtt[idx])
    print(f"reconstructing {frames} frames seq {seq} K=1 and K=3 (grid_factor={grid_factor}) ...", flush=True)

    def noise_to(x, t, k):
        return float(jnp.sqrt(ab[t])) * x + float(jnp.sqrt(1.0 - ab[t])) * jax.random.normal(k, x.shape)

    key, k1, k2 = jax.random.split(key, 3)
    K1 = np.asarray(samplers[t_start](params, noise_to(xin, t_start, k1), k2))          # single chain
    xc = xin
    for Sj in S_MULTI:
        key, ka, kb = jax.random.split(key, 3)
        xc = jnp.asarray(samplers[Sj](params, noise_to(xc, Sj, ka), kb))
    K3 = np.asarray(xc)

    wg, w1, w3 = xgt[..., 1] * STD, K1[..., 1] * STD, K3[..., 1] * STD
    Eg, E1, E3 = (local_hik_energy(w, kcut, sigma) for w in (wg, w1, w3))
    e1, e3 = np.abs(w1 - wg), np.abs(w3 - wg)
    Egt = np.asarray(spec_fn(jnp.asarray(xgt)))
    hik = lambda x: float((np.asarray(spec_fn(x))[:, HIK0:].sum(-1) / Egt[:, HIK0:].sum(-1)).mean())
    h1, h3 = hik(K1), hik(K3)
    print(f"hik_ret  K=1 {h1:.3f}   K=3 {h3:.3f}", flush=True)

    fig, axes = plt.subplots(frames, 8, figsize=(21, 2.7 * frames))
    axes = np.atleast_2d(axes)
    for r in range(frames):
        vv = np.percentile(np.abs(wg[r]), 99)
        vH = np.percentile(np.concatenate([Eg[r], E1[r], E3[r]]).ravel(), 99)
        vE = np.percentile(np.concatenate([e1[r], e3[r]]).ravel(), 99)
        m1 = ((K1[r] - xgt[r]) ** 2).mean(); m3 = ((K3[r] - xgt[r]) ** 2).mean()
        panels = [("GT ω", wg[r], "RdBu_r", -vv, vv), ("K=1 recon", w1[r], "RdBu_r", -vv, vv),
                  ("K=3 recon", w3[r], "RdBu_r", -vv, vv),
                  ("E_hik(GT)", Eg[r], "viridis", 0, vH), ("E_hik(K=1)", E1[r], "viridis", 0, vH),
                  ("E_hik(K=3)", E3[r], "viridis", 0, vH),
                  (f"|K=1−GT| mse{m1:.3f}", e1[r], "magma", 0, vE),
                  (f"|K=3−GT| mse{m3:.3f}", e3[r], "magma", 0, vE)]
        for c, (nm, d, cm, vmn, vmx) in enumerate(panels):
            a = axes[r, c]; a.imshow(d, cmap=cm, vmin=vmn, vmax=vmx)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(nm, fontsize=9)
            elif c in (6, 7):
                a.set_title(nm.split()[-1], fontsize=8)
        axes[r, 0].set_ylabel(f"frame {idx[r]}", fontsize=9)
    fig.suptitle(f"grid-4× model — K=1 (single chain) vs K=3 (multi-phase), seq {seq}   "
                 f"[hik_ret K=1={h1:.2f}, K=3={h3:.2f} — K=3 over-adds high-k]", y=1.005, fontsize=11)
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seq", type=int, default=36)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--t_start", type=int, default=100)
    ap.add_argument("--grid_factor", type=int, default=4)
    ap.add_argument("--out", type=str, default="monitoring/ab_pdelocal/viz_k1k3.png")
    ap.add_argument("--gt", type=str, default=None, help="GT .npy (default Re=1000)")
    main(**vars(ap.parse_args()))
