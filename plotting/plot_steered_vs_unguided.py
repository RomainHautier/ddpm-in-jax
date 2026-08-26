#!/usr/bin/env python
"""Figures for the in-distribution steered-guidance paragraph (Re=1000, held-out seqs 34-39).

    (A) reconstructions side by side: ground truth | observation | unguided base | steered base,
        with a zoom row on a boxed region.
    (B) where the energy is placed: local band-energy maps (sigma=6 smoothed) for the same three
        fields, per band, with the per-band placement correlation annotated, plus a pixel-wise
        scatter of local energy (model vs ground truth) that shows the correlation collapse.

Reads STORED reconstructions (base_results/fields/re1000/) - no inference, no TPU.
Pure numpy/matplotlib.

Usage (from repo root):
    JAX_PLATFORMS=cpu python plotting/plot_steered_vs_unguided.py --sample 47
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
FDIR = "base_results/fields/re1000"
BANDS = [(16, 32), (32, 64), (64, 96)]


def load(name):
    """stored triplets (n,256,256,3) float16 -> middle frame in physical vorticity units"""
    return np.load(f"{FDIR}/{name}.npz")["x"].astype(np.float32)[..., 1] * SIG


def band_energy_map(w, lo, hi, sigma=6.0):
    """local energy of the [lo,hi) band, Gaussian-smoothed - the placement metric's own map"""
    n = w.shape[-1]
    fy = np.fft.fftfreq(n) * n
    km = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
    g = np.exp(-2.0 * (np.pi * sigma) ** 2 * ((fy[:, None] / n) ** 2 + (fy[None, :] / n) ** 2))
    bp = np.real(np.fft.ifft2(np.fft.fft2(w) * ((km >= lo) & (km < hi))))
    return np.real(np.fft.ifft2(np.fft.fft2(bp ** 2) * g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=47, help="index into the 120 held-out samples")
    ap.add_argument("--unguided", default="base0__K3__none")
    ap.add_argument("--steered", default="base0__K3__rewardv2")
    ap.add_argument("--zoom", default="64,160,96,192", help="i0,i1,j0,j1 of the zoom window")
    ap.add_argument("--band", type=int, default=1, help="which BANDS entry for the scatter panel")
    ap.add_argument("--out", default="plotting/figs/steered_vs_unguided")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    i0, i1, j0, j1 = (int(v) for v in a.zoom.split(","))
    i = a.sample

    gt, lr = load("GT"), load("LR")
    un, st = load(a.unguided), load(a.steered)

    # ---------- (A) reconstructions ----------
    panels = [("ground truth", gt[i]), ("observation (4$\\times$ coarse)", lr[i]),
              ("unguided", un[i]), ("statistics-steered", st[i])]
    vm = float(np.abs(gt[i]).max())
    fig, ax = plt.subplots(2, 4, figsize=(13.2, 6.9), constrained_layout=True)
    for c, (lab, img) in enumerate(panels):
        ax[0, c].imshow(img, cmap=DIVERGING, vmin=-vm, vmax=vm)
        ax[0, c].add_patch(mpatches.Rectangle((j0, i0), j1 - j0, i1 - i0, fill=False, ec="k", lw=1.3))
        ax[0, c].set_title(lab)
        ax[1, c].imshow(img[i0:i1, j0:j1], cmap=DIVERGING, vmin=-vm, vmax=vm, interpolation="nearest")
        for r in (0, 1):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[1, 0].set_ylabel("zoom")
    fig.suptitle(f"Re=1000 held-out sample {i}: statistics-steered sampling adds fine-scale energy "
                 f"the unguided model smooths away", fontsize=12)
    for ext in (".pdf", ".png"):
        fig.savefig(a.out + "_recon" + ext, dpi=200)
    plt.close(fig)

    # ---------- (B) where the energy goes ----------
    fig, ax = plt.subplots(len(BANDS), 4, figsize=(13.6, 3.5 * len(BANDS)), constrained_layout=True)
    for b, (lo, hi) in enumerate(BANDS):
        mg, mu, ms = (band_energy_map(x[i], lo, hi) for x in (gt, un, st))
        vmax = float(np.percentile(mg, 99.5))
        for c, (lab, m) in enumerate((("ground truth", mg), ("unguided", mu), ("steered", ms))):
            ax[b, c].imshow(m, cmap="magma", vmin=0, vmax=vmax)
            ax[b, c].set_xticks([]); ax[b, c].set_yticks([])
            if b == 0:
                ax[b, c].set_title(lab)
            if c == 0:
                ax[b, c].set_ylabel(f"$k\\in[{lo},{hi})$")
            elif c > 0:
                r = np.corrcoef(m.ravel(), mg.ravel())[0, 1]
                ax[b, c].set_xlabel(f"map correlation {r:.2f}", fontsize=10)
        # pixel-wise scatter: model local energy vs ground-truth local energy
        s = slice(None, None, 37)
        ax[b, 3].scatter(mg.ravel()[s], mu.ravel()[s], s=3, alpha=0.25, color="#1f77b4", label="unguided")
        ax[b, 3].scatter(mg.ravel()[s], ms.ravel()[s], s=3, alpha=0.25, color="#b8399e", label="steered")
        hi_ = float(np.percentile(mg, 99.8))
        ax[b, 3].plot([0, hi_], [0, hi_], "k--", lw=1)
        ax[b, 3].set(xlim=(0, hi_), ylim=(0, hi_ * 1.6), xlabel="GT local energy", ylabel="model local energy")
        if b == 0:
            ax[b, 3].legend(fontsize=9, markerscale=3, frameon=False)
    fig.suptitle("Where the energy is placed: local band-energy maps and pixel-wise agreement",
                 fontsize=12)
    for ext in (".pdf", ".png"):
        fig.savefig(a.out + "_placement" + ext, dpi=200)
    plt.close(fig)

    # ---------- (C) the decomposition: spatial placement vs frame amplitude ----------
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.3), constrained_layout=True)
    labs, pooled_u, pooled_s, frame_u, frame_s = [], [], [], [], []
    for lo, hi in BANDS:
        G = np.stack([band_energy_map(gt[j], lo, hi) for j in range(len(gt))])
        U = np.stack([band_energy_map(un[j], lo, hi) for j in range(len(un))])
        S = np.stack([band_energy_map(st[j], lo, hi) for j in range(len(st))])
        labs.append(f"[{lo},{hi})")
        pooled_u.append(np.corrcoef(U.ravel(), G.ravel())[0, 1])
        pooled_s.append(np.corrcoef(S.ravel(), G.ravel())[0, 1])
        frame_u.append([np.corrcoef(U[j].ravel(), G[j].ravel())[0, 1] for j in range(len(G))])
        frame_s.append([np.corrcoef(S[j].ravel(), G[j].ravel())[0, 1] for j in range(len(G))])
        if (lo, hi) == BANDS[a.band]:
            amp_g, amp_u, amp_s = G.mean((1, 2)), U.mean((1, 2)), S.mean((1, 2))
    x = np.arange(len(labs)); w = 0.35
    ax[0].bar(x - w / 2, pooled_u, w, color="#1f77b4", label="unguided")
    ax[0].bar(x + w / 2, pooled_s, w, color="#b8399e", label="steered")
    ax[0].set(xticks=x, ylim=(0, 1.05), ylabel="correlation",
              title="pooled placement\n(all frames stacked together)")
    ax[0].set_xticklabels(labs); ax[0].legend(frameon=False, fontsize=9)
    for b in range(len(labs)):
        ax[1].scatter(np.full(len(frame_u[b]), b - w / 2) + np.random.default_rng(0).uniform(-.1, .1, len(frame_u[b])),
                      frame_u[b], s=5, color="#1f77b4", alpha=.4)
        ax[1].scatter(np.full(len(frame_s[b]), b + w / 2) + np.random.default_rng(1).uniform(-.1, .1, len(frame_s[b])),
                      frame_s[b], s=5, color="#b8399e", alpha=.4)
        ax[1].plot([b - w, b], [np.median(frame_u[b])] * 2, "k-", lw=2)
        ax[1].plot([b, b + w], [np.median(frame_s[b])] * 2, "k-", lw=2)
    ax[1].set(xticks=x, ylim=(0, 1.05), title="per-frame placement\n(each frame vs its own truth)")
    ax[1].set_xticklabels(labs)
    lo, hi = BANDS[a.band]
    ax[2].scatter(amp_g, amp_u, s=14, color="#1f77b4", label=f"unguided  r={np.corrcoef(amp_u, amp_g)[0,1]:.2f}")
    ax[2].scatter(amp_g, amp_s, s=14, color="#b8399e", label=f"steered   r={np.corrcoef(amp_s, amp_g)[0,1]:.2f}")
    lim = float(max(amp_g.max(), amp_u.max(), amp_s.max()))
    ax[2].plot([0, lim], [0, lim], "k--", lw=1)
    ax[2].set(xlabel="GT frame band energy", ylabel="model frame band energy",
              title=f"frame amplitude, $k\\in[{lo},{hi})$\n(what steering actually destroys)")
    ax[2].legend(frameon=False, fontsize=9)
    fig.suptitle("The pooled metric conflates two channels: steering preserves WHERE the energy goes "
                 "within a frame, and destroys HOW MUCH each frame gets", fontsize=12)
    for ext in (".pdf", ".png"):
        fig.savefig(a.out + "_decomposition" + ext, dpi=200)
    plt.close(fig)

    # ---------- numbers for the text ----------
    print(f"n samples = {len(gt)}")
    for lab, x in (("unguided", un), ("steered", st)):
        line = []
        for lo, hi in BANDS:
            mg = np.stack([band_energy_map(gt[j], lo, hi) for j in range(len(gt))])
            mm = np.stack([band_energy_map(x[j], lo, hi) for j in range(len(x))])
            line.append(f"[{lo},{hi}) place {np.corrcoef(mm.ravel(), mg.ravel())[0, 1]:.3f}")
        print(f"  {lab:<9} " + "  ".join(line))
    print(f"figures -> {a.out}_recon.pdf/.png and {a.out}_placement.pdf/.png")


if __name__ == "__main__":
    main()
