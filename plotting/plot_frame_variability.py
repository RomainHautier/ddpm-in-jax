#!/usr/bin/env python
"""Figure: frames of the same flow do not share a spectral shape.

    (a) three held-out ground-truth frames chosen at the 10th / 50th / 90th percentile of
        fine-scale energy, with a zoom row;
    (b) their spectra, absolute and normalised by total energy (the shape, level removed);
    (c) the distribution across all 120 held-out frames: each band's share of total energy,
        and the shape indicator vs the total level (they are uncorrelated).

Reads STORED ground-truth fields (base_results/fields/re1000/GT.npz) - no inference.

Usage (from repo root):
    JAX_PLATFORMS=cpu python plotting/plot_frame_variability.py
"""
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import matplotlib                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.patches as mpatches    # noqa: E402
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "font.size": 11})
import numpy as np                       # noqa: E402

DIVERGING = "RdBu_r"
SIG = 4.7988
BANDS = [(1, 5), (5, 16), (16, 32), (32, 64), (64, 96)]
BL = ["$[1,5)$", "$[5,16)$", "$[16,32)$", "$[32,64)$", "$[64,96)$"]
COL = ["#1f77b4", "#0f9e78", "#d4770a"]


def radial_spectrum(w):
    n = w.shape[-1]
    p = np.abs(np.fft.fft2(w)) ** 2
    k = np.fft.fftfreq(n, d=1.0 / n)
    kr = np.rint(np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)).astype(int)
    return np.bincount(kr.ravel(), weights=p.ravel(), minlength=n)[: n // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="base_results/fields/re1000/GT.npz")
    ap.add_argument("--zoom", default="64,160,96,192")
    ap.add_argument("--out", default="plotting/figs/frame_variability")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    i0, i1, j0, j1 = (int(v) for v in a.zoom.split(","))

    w = np.load(a.fields)["x"].astype(np.float32)[..., 1] * SIG      # (n,256,256)
    S = np.stack([radial_spectrum(x) for x in w])                    # (n,128)
    bands = np.stack([S[:, lo:hi].sum(1) for lo, hi in BANDS], 1)    # (n,5)
    tot = bands.sum(1)
    frac = bands / tot[:, None]
    fine = bands[:, 3] + bands[:, 4]

    # three representative frames by fine-scale energy
    order = np.argsort(fine)
    idx = [order[int(0.10 * len(order))], order[len(order) // 2], order[int(0.90 * len(order))]]
    labs = ["quiet frame (p10)", "typical frame (p50)", "energetic frame (p90)"]

    fig = plt.figure(figsize=(14.6, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 0.75, 1.05])

    vm = float(np.abs(w[idx[1]]).max())
    for c, (i, lab) in enumerate(zip(idx, labs)):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(w[i], cmap=DIVERGING, vmin=-vm, vmax=vm)
        ax.add_patch(mpatches.Rectangle((j0, i0), j1 - j0, i1 - i0, fill=False, ec="k", lw=1.3))
        ax.set_title(f"{lab}\nfine-scale energy {fine[i] / fine.mean():.2f}$\\times$ mean", color=COL[c])
        ax.set_xticks([]); ax.set_yticks([])
        axz = fig.add_subplot(gs[1, c])
        axz.imshow(w[i][i0:i1, j0:j1], cmap=DIVERGING, vmin=-vm, vmax=vm, interpolation="nearest")
        axz.set_xticks([]); axz.set_yticks([])
        if c == 0:
            axz.set_ylabel("zoom")

    k = np.arange(1, 128)
    ax = fig.add_subplot(gs[0, 3])
    for c, i in enumerate(idx):
        ax.loglog(k, S[i, 1:128], color=COL[c], lw=1.8)
    ax.set(xlabel="wavenumber $k$", ylabel="$E(k)$", title="absolute spectra")
    ax.axvspan(32, 96, color="#e3edf3", zorder=0)

    ax = fig.add_subplot(gs[1, 3])
    for c, i in enumerate(idx):
        ax.loglog(k, S[i, 1:128] / tot[i], color=COL[c], lw=1.8)
    ax.set(xlabel="wavenumber $k$", ylabel="$E(k)\\,/\\,E_{tot}$", title="shape (level removed)")
    ax.axvspan(32, 96, color="#e3edf3", zorder=0)

    # (c) distributions over all frames
    ax = fig.add_subplot(gs[2, 0:2])
    parts = ax.violinplot([frac[:, i] / frac[:, i].mean() for i in range(5)], showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor("#7a4bd0"); b.set_alpha(0.45)
    for i in range(5):
        y = frac[:, i] / frac[:, i].mean()
        ax.scatter(np.full(len(y), i + 1) + np.random.default_rng(0).uniform(-.06, .06, len(y)),
                   y, s=4, color="#232823", alpha=.35)
        ax.text(i + 1, 2.9, f"{y.max() / y.min():.0f}$\\times$", ha="center", fontsize=9.5)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set(xticks=range(1, 6), ylim=(0, 3.2),
           ylabel="band's share of total energy\n(relative to its mean)",
           title="every band's share varies frame to frame (range annotated)")
    ax.set_xticklabels(BL)

    ax = fig.add_subplot(gs[2, 2])
    shape = bands[:, 3] / bands[:, 2]
    r = np.corrcoef(np.log(shape), np.log(tot))[0, 1]
    ax.scatter(tot / tot.mean(), shape / shape.mean(), s=16, color="#b8399e", alpha=.7)
    for c, i in enumerate(idx):
        ax.scatter(tot[i] / tot.mean(), shape[i] / shape.mean(), s=70, color=COL[c], zorder=5,
                   edgecolor="k", linewidth=.8)
    ax.set(xlabel="total energy / mean", ylabel="shape  $E[32,64)/E[16,32)$\n(relative to mean)",
           title=f"shape is independent of level\n$r = {r:+.2f}$")

    ax = fig.add_subplot(gs[2, 3])
    ax.hist(fine / fine.mean(), bins=24, color="#0f9e78", alpha=.8)
    for c, i in enumerate(idx):
        ax.axvline(fine[i] / fine.mean(), color=COL[c], lw=2)
    ax.set(xlabel="fine-scale energy $k\\in[32,96)$ / mean", ylabel="frames",
           title=f"fine-scale energy across frames\nspread {fine.std() / fine.mean():.2f}, "
                 f"range {fine.max() / fine.min():.0f}$\\times$")

    fig.suptitle("Frames of the same flow do not share a spectral shape: the ensemble mean is not "
                 "the right target for any individual frame", fontsize=13)
    for ext in (".pdf", ".png"):
        fig.savefig(a.out + ext, dpi=200)
    plt.close(fig)

    print(f"n = {len(w)} held-out frames")
    for i, b in enumerate(BL):
        f = frac[:, i]
        print(f"  share in {b:>10}: mean {f.mean():.4f}  CV {f.std() / f.mean():.3f}  "
              f"range {f.max() / f.min():.1f}x")
    print(f"  fine-scale energy: CV {fine.std() / fine.mean():.3f}, range {fine.max() / fine.min():.1f}x")
    print(f"  shape vs level correlation: r = {r:+.3f}")
    print(f"figure -> {a.out}.pdf / .png")


if __name__ == "__main__":
    main()
