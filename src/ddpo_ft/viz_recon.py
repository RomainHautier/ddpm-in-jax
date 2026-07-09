"""Visualize reconstructions: sparse input | base | DDPO | GT vorticity fields, for a few frames.
Single-chain SDEdit (t_start), de-normalized to physical vorticity, shared symmetric color scale.

    python -m src.ddpo_ft.viz_recon <ckpt.pkl> [--seq 36] [--frames 4] [--t_start 100]
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


def main(ckpt, seq=36, frames=4, t_start=100, seed=1, out="monitoring/ddpo_ckpts/viz_recon.png"):
    ddpm, base_params, _ = build_base_ddpm()
    ddpo_params = pickle.load(open(ckpt, "rb"))["params"]
    sampler = make_sampler(ddpm.unet, ddpm.alpha_bar, ddpm.beta_schedule, t_start, temp=1.0)
    ab = ddpm.alpha_bar
    sqrt_ab, sqrt_1m = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))

    s = load_sequence(GT_PATH, seq)
    inp = build_triplets(sparse_nnfill_degrade(s, seq), MEAN, STD)
    gt = build_triplets(s, MEAN, STD)
    idx = np.linspace(len(inp) // 4, len(inp) - 1, frames).astype(int)      # developed-turbulence frames
    xin, xgt = jnp.asarray(inp[idx]), np.asarray(gt[idx])
    print(f"reconstructing {frames} frames of seq {seq} (t_start={t_start}) ...", flush=True)

    key = jax.random.PRNGKey(seed)
    def recon(params):
        k1, k2 = jax.random.split(key)
        x_start = sqrt_ab * xin + sqrt_1m * jax.random.normal(k1, xin.shape)
        return np.asarray(sampler(params, x_start, k2))
    rb, rd = recon(base_params), recon(ddpo_params)

    # de-normalize middle frame -> physical vorticity
    fields = {"sparse input": np.asarray(xin)[..., 1] * STD, "base recon": rb[..., 1] * STD,
              "DDPO recon": rd[..., 1] * STD, "GT": xgt[..., 1] * STD}
    cols = list(fields)
    fig, axes = plt.subplots(frames, len(cols), figsize=(3.0 * len(cols), 3.0 * frames))
    axes = np.atleast_2d(axes)
    for r in range(frames):
        vlim = np.abs(fields["GT"][r]).max()
        for c, name in enumerate(cols):
            a = axes[r, c]
            a.imshow(fields[name][r], cmap="RdBu_r", vmin=-vlim, vmax=vlim)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(name, fontsize=11)
        axes[r, 0].set_ylabel(f"frame {idx[r]}", fontsize=9)
    fig.suptitle(f"Re=1000 seq {seq} — vorticity reconstructions ({os.path.basename(ckpt)}, single-chain t_start={t_start})", y=1.005)
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
