#!/usr/bin/env python
"""Figure: how each model reconstructs the SAME three frames (quiet / typical / energetic).

Rows = the three held-out frames chosen at the p10 / p50 / p90 of ground-truth fine-scale energy
(the ones in plot_frame_variability). Columns = ground truth, observation, and one column per
model/strategy. A final column carries that frame's spectra (all rows overlaid on the GT), and a
second figure gives the ratio E(k)/E_GT(k) per frame.

Reads STORED reconstructions - no inference.

Usage:
    JAX_PLATFORMS=cpu python plotting/plot_frames_by_model.py \
        --rows "base0__K3__none,r1k-449__K3__none,base0__K3__rewardv2,r1k-449__K3__rewardv2"
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
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "font.size": 11})
import numpy as np                       # noqa: E402

DIVERGING = "RdBu_r"
SIG = 4.7988
FDIR = "base_results/fields/re1000"
PALETTE = ["#1f77b4", "#0f9e78", "#b8399e", "#d4770a", "#7a4bd0", "#c22f4f"]
NICE = {"base0": "base", "r1k-449": "r1k fine-tune", "pr1k-549": "pr1k", "st1k-599": "st1k"}
SNICE = {"none": "unguided", "rewardv2": "+ v2 dial", "reward": "+ v1 dial", "tapered": "+ tapered dial",
         "all3v2": "+ all dials", "v3cal": "+ v3 dial", "residual": "+ pde dial"}


def load(name):
    return np.load(f"{FDIR}/{name}.npz")["x"].astype(np.float32)[..., 1] * SIG


def radial_spectrum(w):
    n = w.shape[-1]
    p = np.abs(np.fft.fft2(w)) ** 2
    k = np.fft.fftfreq(n, d=1.0 / n)
    kr = np.rint(np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)).astype(int)
    return np.bincount(kr.ravel(), weights=p.ravel(), minlength=n)[: n // 2]


def label(row):
    m, _, sg = row.split("__")
    return f"{NICE.get(m, m)} {SNICE.get(sg, sg)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="base0__K3__none,r1k-449__K3__none,"
                                      "base0__K3__rewardv2,r1k-449__K3__rewardv2")
    ap.add_argument("--out", default="plotting/figs/frames_by_model")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = [r for r in a.rows.split(",") if os.path.exists(f"{FDIR}/{r}.npz")]
    if not rows:
        sys.exit("none of the requested rows are stored")

    gt, lr = load("GT"), load("LR")
    Sg = np.stack([radial_spectrum(x) for x in gt])
    fine = Sg[:, 32:96].sum(1)
    order = np.argsort(fine)
    idx = [order[int(0.10 * len(order))], order[len(order) // 2], order[int(0.90 * len(order))]]
    flab = ["quiet (p10)", "typical (p50)", "energetic (p90)"]
    recs = {r: load(r) for r in rows}

    # ---- fields ----
    ncol = 2 + len(rows)
    fig, ax = plt.subplots(3, ncol, figsize=(2.35 * ncol, 7.6), constrained_layout=True)
    for r_, i in enumerate(idx):
        vm = float(np.abs(gt[i]).max())
        panels = [("ground truth", gt[i]), ("observation", lr[i])] + \
                 [(label(r), recs[r][i]) for r in rows]
        for c, (lab, img) in enumerate(panels):
            ax[r_, c].imshow(img, cmap=DIVERGING, vmin=-vm, vmax=vm)
            ax[r_, c].set_xticks([]); ax[r_, c].set_yticks([])
            if r_ == 0:
                ax[r_, c].set_title(lab, fontsize=10)
        ax[r_, 0].set_ylabel(f"{flab[r_]}\n{fine[i] / fine.mean():.2f}$\\times$ mean fine energy",
                             fontsize=9.5)
    fig.suptitle("The same three held-out frames, reconstructed by each model", fontsize=12.5)
    for ext in (".pdf", ".png"):
        fig.savefig(a.out + "_fields" + ext, dpi=200)
    plt.close(fig)

    # ---- spectra, absolute and ratio ----
    k = np.arange(1, 128)
    fig, ax = plt.subplots(2, 3, figsize=(14.0, 7.4), constrained_layout=True)
    for c, i in enumerate(idx):
        ax[0, c].loglog(k, Sg[i, 1:128], color="#000000", lw=2.6, label="GROUND TRUTH", zorder=9)
        ax[1, c].axhline(1, color="#000000", lw=2.0, ls="--", zorder=9,
                         label="GROUND TRUTH (=1)" if c == 0 else None)
        for j, r in enumerate(rows):
            S = radial_spectrum(recs[r][i])
            ax[0, c].loglog(k, S[1:128], color=PALETTE[j % len(PALETTE)], lw=1.7,
                            label=label(r) if c == 0 else None)
            ax[1, c].semilogx(k, S[1:128] / Sg[i, 1:128], color=PALETTE[j % len(PALETTE)], lw=1.7)
        for rr in (0, 1):
            ax[rr, c].axvspan(16, 32, color="#f8e8d0", zorder=0)
            ax[rr, c].axvspan(32, 96, color="#e3edf3", zorder=0)
        ax[0, c].set_title(f"{flab[c]} — {fine[i] / fine.mean():.2f}$\\times$ mean")
        ax[1, c].set(yscale="log", ylim=(0.05, 4), xlabel="wavenumber $k$")
    ax[0, 0].set_ylabel("$E(k)$"); ax[1, 0].set_ylabel("$E(k)\\,/\\,E_{GT}(k)$")
    ax[0, 0].legend(fontsize=7.6, loc="lower left")
    fig.suptitle("Spectra of those same frames: the unguided models track each frame's own level; "
                 "the dial pushes every frame to the ensemble target", fontsize=12.5)
    for ext in (".pdf", ".png"):
        fig.savefig(a.out + "_spectra" + ext, dpi=200)
    plt.close(fig)

    print(f"frames: {[int(i) for i in idx]}  (fine energy "
          f"{[round(float(fine[i] / fine.mean()), 2) for i in idx]}x mean)")
    print(f"{'row':<24} " + "  ".join(f"{l:>16}" for l in flab))
    for r in rows:
        vals = []
        for i in idx:
            S = radial_spectrum(recs[r][i])
            vals.append(f"{S[32:96].sum() / Sg[i, 32:96].sum():.2f} of its own GT")
        print(f"  {label(r):<22} " + "  ".join(f"{v:>16}" for v in vals))
    print(f"figures -> {a.out}_fields.pdf and {a.out}_spectra.pdf")


if __name__ == "__main__":
    main()
