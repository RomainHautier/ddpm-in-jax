#!/usr/bin/env python
"""Figure: how a high-resolution vorticity snapshot becomes the model's starting state.

    (a) HR 256^2  ->  (b) every `factor`-th pixel kept (64x64 lattice)  ->  (c) nearest-neighbour
    fill back to 256^2 (the observation / model input)  ->  (d) forward-noised to t_noise with the
    model's schedule (the SDEdit starting state used at inference and finetuning).

Pure numpy/scipy/matplotlib (schedule read from configs/config.yaml). Also prints the per-band
energy ratio input/HR for the text.

Usage (from repo root):
    JAX_PLATFORMS=cpu python plotting/plot_lr_preprocessing.py --seq 36 --frame 160 --t_noise 100
"""
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "font.size": 11})
import numpy as np                      # noqa: E402
import yaml                             # noqa: E402

from src.sequence_inference import grid_downsample_degrade, load_sequence  # noqa: E402

DIVERGING = "RdBu_r"                    # signed vorticity: two hues + neutral midpoint
MEAN, STD = 0.0, 4.7988                 # the model's normalization (training-set statistics)
BANDS = [(1, 16), (16, 32), (32, 64), (64, 96)]


def radial_spectrum(w):
    n = w.shape[-1]
    p = np.abs(np.fft.fft2(w)) ** 2
    k = np.fft.fftfreq(n, d=1.0 / n)
    kr = np.rint(np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)).astype(int)
    return np.bincount(kr.ravel(), weights=p.ravel(), minlength=n)[: n // 2]


def alpha_bar(cfg_path="configs/config.yaml"):
    d = yaml.safe_load(open(cfg_path))["diffusion"]
    beta = np.linspace(d["beta_start"], d["beta_end"], d["T"])
    return np.cumprod(1.0 - beta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="flow-data/kf_2d_re1000_256_40seed.npy")
    ap.add_argument("--seq", type=int, default=36, help="held-out probe sequence")
    ap.add_argument("--frame", type=int, default=160)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--t_noise", type=int, default=100, help="SDEdit start level (inference/finetune config)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="plotting/figs/lr_preprocessing", help="path stem (.png + .pdf)")
    a = ap.parse_args()

    seq = load_sequence(a.gt, a.seq)                                   # (n_frames, 256, 256)
    hr = np.asarray(seq[a.frame], np.float32)
    n, f = hr.shape[0], a.factor
    mask = np.zeros((n, n), bool)
    mask[::f, ::f] = True
    lattice = np.where(mask, hr, np.nan)                               # kept pixels only
    filled = np.asarray(grid_downsample_degrade(seq[a.frame:a.frame + 1], f)[0], np.float32)

    ab = alpha_bar()[a.t_noise]                                        # same schedule as the model
    rng = np.random.default_rng(a.seed)
    x_norm = (filled - MEAN) / STD
    noised = np.sqrt(ab) * x_norm + np.sqrt(1.0 - ab) * rng.standard_normal(x_norm.shape)
    noised = noised * STD + MEAN                                       # back to vorticity units for display

    E_hr, E_in = radial_spectrum(hr), radial_spectrum(filled)
    print(f"seq {a.seq} frame {a.frame}, factor {f}: {int(mask.sum())} of {n * n} points kept; "
          f"t={a.t_noise}: sqrt(ab)={np.sqrt(ab):.3f}, noise std={np.sqrt(1 - ab) * STD:.2f} (vorticity units)")
    for lo, hi in BANDS:
        print(f"  band [{lo:>2},{hi:>2}): energy input/HR = {100 * E_in[lo:hi].sum() / E_hr[lo:hi].sum():6.1f}%")

    vmax = 0.9 * float(np.abs(hr).max())
    cmap = plt.get_cmap(DIVERGING).copy()
    cmap.set_bad("#ececec")
    panels = [
        (hr, rf"a) HR field, $x_{{HR}} \in \mathbb{{R}}^{{{n}^2 \times 1}}$"),   # {n} → 256,
        (lattice, rf"b) every {f}th pixel kept, ${n // f}^2$"),
        (filled, rf"c) NN filled, $x_{{LR}} \in \mathbb{{R}}^{{{n}^2 \times 1}}$"),
        (noised, rf"d) Noised input to $t={a.t_noise}$, $x_t \in \mathbb{{R}}^{{{n}^2 \times 1}}$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2),
                             gridspec_kw=dict(left=0.01, right=0.99, top=0.99, bottom=0.10, wspace=0.05))
    for ax, (w, title) in zip(axes, panels):
        im = ax.imshow(w, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.text(0.5, -0.04, title, transform=ax.transAxes, ha="center", va="top", fontsize=16)
        ax.set_axis_off()
    fig.canvas.draw()                                               # settle the equal-aspect boxes
    pos = axes[-1].get_position()                                   # panel (d) as drawn
    cax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.011, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"vorticity $\omega$", fontsize=16)
    cb.ax.tick_params(labelsize=12)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(a.out + ".pdf", bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {a.out}.png and .pdf")


if __name__ == "__main__":
    main()
