"""Difference-map view: base | DDPO | (DDPO - base) | (GT - base) | GT vorticity fields.
The 3rd column shows what DDPO ADDED over base; the 4th shows the TRUE missing structure (GT-base).
If DDPO added the right high-k energy, col 3 should resemble col 4.

    python -m src.ddpo_ft.viz_diff <ckpt.pkl> [--seq 36] [--frames 4] [--t_start 100]
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
from src.sequence_inference import build_triplets, load_sequence, sparse_nnfill_degrade  # noqa: E402

MEAN, STD = 0.0, 4.7988
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"


def main(ckpt, seq=36, frames=4, t_start=100, seed=1, out="monitoring/ddpo_ckpts/viz_diff.png"):
    ddpm, base_params, _ = build_base_ddpm()
    ddpo_params = pickle.load(open(ckpt, "rb"))["params"]
    sampler = make_sampler(ddpm.unet, ddpm.alpha_bar, ddpm.beta_schedule, t_start, temp=1.0)
    ab = ddpm.alpha_bar
    sqrt_ab, sqrt_1m = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))

    s = load_sequence(GT_PATH, seq)
    inp = build_triplets(sparse_nnfill_degrade(s, seq), MEAN, STD)
    gt = build_triplets(s, MEAN, STD)
    idx = np.linspace(len(inp) // 4, len(inp) - 1, frames).astype(int)
    xin, xgt = jnp.asarray(inp[idx]), np.asarray(gt[idx])
    print(f"reconstructing {frames} frames of seq {seq} for diff map ...", flush=True)

    key = jax.random.PRNGKey(seed)
    def recon(params):
        k1, k2 = jax.random.split(key)
        x_start = sqrt_ab * xin + sqrt_1m * jax.random.normal(k1, xin.shape)
        return np.asarray(sampler(params, x_start, k2))[..., 1] * STD           # de-normalized middle frame
    base, ddpo, gtf = recon(base_params), recon(ddpo_params), xgt[..., 1] * STD

    cols = [("base recon", base, "field"), ("DDPO recon", ddpo, "field"),
            ("DDPO − base", ddpo - base, "diff"), ("GT − base", gtf - base, "diff"),
            ("GT", gtf, "field")]
    fig, axes = plt.subplots(frames, len(cols), figsize=(3.0 * len(cols), 3.0 * frames))
    axes = np.atleast_2d(axes)
    for r in range(frames):
        vf = np.abs(gtf[r]).max()
        vd = 0.35 * vf                                                          # diffs are smaller-amplitude
        for c, (name, data, kind) in enumerate(cols):
            a = axes[r, c]; vlim = vf if kind == "field" else vd
            a.imshow(data[r], cmap="RdBu_r", vmin=-vlim, vmax=vlim)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(name, fontsize=11)
        axes[r, 0].set_ylabel(f"frame {idx[r]}", fontsize=9)
    fig.suptitle(f"Re=1000 seq {seq} — DDPO−base difference vs true missing structure GT−base "
                 f"({os.path.basename(ckpt)}, diff cols at 0.35× field scale)", y=1.005, fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=115, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seq", type=int, default=36)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--t_start", type=int, default=100)
    main(**vars(ap.parse_args()))
