"""Visualise sequence_inference outputs:
  1. animation.html  — input | ground truth | reconstruction, playing as a video
  2. reconstruction_examples.png — a few frames across one sequence
  3. per_frame_mse.png — per-frame MSE vs frame index, one line per sequence
"""
import os
import pickle

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

RECON_DIR = "monitoring/sequence_reconstructions"
OUT_DIR = "monitoring/sequence_viz"
SEQS = [36, 37, 38, 39]
ANIM_SEQ = 39        # sequence to animate / show examples from
CH = 1               # middle frame of the (t, t+1, t+2) triplet
ANIM_STEP = 2        # subsample frames in the animation to keep the HTML small

plt.rcParams["animation.embed_limit"] = 200  # MB


def load_seq(seq):
    with open(os.path.join(RECON_DIR, f"sequence_reconstruction_seq{seq}.pkl"), "rb") as f:
        return pickle.load(f)


def per_frame_mse_plot():
    plt.figure(figsize=(11, 5))
    for seq in SEQS:
        r = load_seq(seq)
        mse = np.array([fr["mse"] for fr in r["frames"]])
        plt.plot(mse, lw=1.2, label=f"seq {seq} (mean={mse.mean():.3f})")
    plt.xlabel("frame index (sequence length)")
    plt.ylabel("MSE (normalized units)")
    plt.title("Per-frame reconstruction MSE — test sequences")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "per_frame_mse.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"saved {out}", flush=True)


def load_stacks(seq):
    r = load_seq(seq)
    frames = r["frames"]
    inp = np.stack([fr["input"][..., CH] for fr in frames])
    gt = np.stack([fr["ground_truth"][..., CH] for fr in frames])
    rec = np.stack([fr["final"][..., CH] for fr in frames])
    return inp, gt, rec, r["metadata"]


def examples_plot(inp, gt, rec, vmax, seq, n=5):
    idx = np.linspace(0, len(inp) - 1, n, dtype=int)
    rows = [("input (npz)", inp), ("ground truth", gt), ("reconstruction", rec)]
    fig, axes = plt.subplots(3, n, figsize=(3.1 * n, 9.5))
    for ri, (label, d) in enumerate(rows):
        for ci, fi in enumerate(idx):
            ax = axes[ri, ci]
            ax.imshow(d[fi], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"frame {fi}", fontsize=11)
            if ci == 0:
                ax.set_ylabel(label, fontsize=12)
    fig.suptitle(f"Sequence {seq} — reconstruction examples (channel t+1, normalized)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "reconstruction_examples.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}", flush=True)


def animation_html(inp, gt, rec, vmax, seq):
    n = len(inp)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    titles = ["input (npz)", "ground truth", "reconstruction"]
    data = [inp, gt, rec]
    ims = []
    for ax, title, d in zip(axes, titles, data):
        im = ax.imshow(d[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax, animated=True)
        ax.set_title(title)
        ax.axis("off")
        ims.append(im)
    sup = fig.suptitle("")

    frame_idx = list(range(0, n, ANIM_STEP))

    def update(f):
        for im, d in zip(ims, data):
            im.set_array(d[f])
        sup.set_text(f"Sequence {seq} — frame {f}/{n - 1}")
        return ims

    fig.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=frame_idx, interval=120, blit=False)
    html = ani.to_jshtml()
    plt.close(fig)
    out = os.path.join(OUT_DIR, "animation.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"saved {out}  ({len(frame_idx)} frames)", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    per_frame_mse_plot()
    inp, gt, rec, meta = load_stacks(ANIM_SEQ)
    vmax = float(np.percentile(np.abs(gt), 99))  # stable colour scale
    examples_plot(inp, gt, rec, vmax, ANIM_SEQ)
    animation_html(inp, gt, rec, vmax, ANIM_SEQ)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
