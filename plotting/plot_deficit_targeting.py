#!/usr/bin/env python
"""Does the sampling guidance put its energy WHERE the model is actually short?

For each held-out frame, using sigma=6 smoothed local band-energy maps in a chosen band:
    deficit  D = map(ground truth) - map(unguided)      -> where the model is short
    addition A = map(guided)       - map(unguided)      -> where the guidance put energy
and reports corr(A, D): +1 = the guidance targets the deficit exactly, 0 = it adds energy
without regard to where it is missing, negative = it adds where the model was already fine.

Also reports how much of the deficit is actually closed, and the resulting map correlation
with ground truth, so gains in placement can be attributed.

    JAX_PLATFORMS=cpu python plotting/plot_deficit_targeting.py --model base0
"""
import argparse, os, sys, glob
os.environ.setdefault("JAX_PLATFORMS", "cpu")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import matplotlib                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "font.size": 11})
import numpy as np                       # noqa: E402

SIG = 4.7988
LAB = {"none": "unguided", "rewardv2": "v2 dose dial", "tapered": "taper", "all3v2": "all dials",
       "v6gate": "taper+scaling+gate", "v7bandgate": "per-band gate", "reward": "v1 dial"}
COL = {"rewardv2": "#b8399e", "tapered": "#7a4bd0", "all3v2": "#d4770a",
       "v6gate": "#0f9e78", "v7bandgate": "#17a2a2", "reward": "#c22f4f"}


def bmap(w, lo, hi, s=6.0):
    n = w.shape[-1]
    fy = np.fft.fftfreq(n) * n
    km = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
    g = np.exp(-2.0 * (np.pi * s) ** 2 * ((fy[:, None] / n) ** 2 + (fy[None, :] / n) ** 2))
    bp = np.real(np.fft.ifft2(np.fft.fft2(w) * ((km >= lo) & (km < hi))))
    return np.real(np.fft.ifft2(np.fft.fft2(bp ** 2) * g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="1000")
    ap.add_argument("--model", default="base0")
    ap.add_argument("--band", default="32,96")
    ap.add_argument("--configs", default="rewardv2,tapered,all3v2,v6gate,v7bandgate")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.band.split(","))
    fdir = f"base_results/fields/re{a.regime}"
    have = {os.path.basename(f)[:-4] for f in glob.glob(f"{fdir}/*.npz")}
    cfgs = [c for c in a.configs.split(",") if f"{a.model}__K3__{c}" in have]
    if not cfgs:
        sys.exit(f"no stored fields for {a.model} in {fdir}")
    out = a.out or f"plotting/figs/deficit_{a.model}_re{a.regime}"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def load(n): return np.load(f"{fdir}/{n}.npz")["x"].astype(np.float32)[..., 1] * SIG
    gt, un = load("GT"), load(f"{a.model}__K3__none")
    G = np.stack([bmap(x, lo, hi) for x in gt])
    U = np.stack([bmap(x, lo, hi) for x in un])
    D = G - U                                        # the deficit map, per frame

    res = {}
    for c in cfgs:
        Y = np.stack([bmap(x, lo, hi) for x in load(f"{a.model}__K3__{c}")])
        A = Y - U
        res[c] = dict(
            targ=np.array([np.corrcoef(A[i].ravel(), D[i].ravel())[0, 1] for i in range(len(G))]),
            closed=np.array([1 - np.abs(G[i] - Y[i]).sum() / np.abs(D[i]).sum() for i in range(len(G))]),
            mapcorr=np.array([np.corrcoef(Y[i].ravel(), G[i].ravel())[0, 1] for i in range(len(G))]),
            A=A, Y=Y)
    base_corr = np.array([np.corrcoef(U[i].ravel(), G[i].ravel())[0, 1] for i in range(len(G))])

    fig, ax = plt.subplots(2, 2, figsize=(12.8, 9.6), constrained_layout=True)
    rng = np.random.default_rng(0)
    for i, c in enumerate(cfgs):
        t = res[c]["targ"]
        ax[0, 0].scatter(np.full(len(t), i) + rng.uniform(-.18, .18, len(t)), t, s=13,
                         color=COL.get(c, "k"), alpha=.6)
        ax[0, 0].plot([i - .3, i + .3], [np.median(t)] * 2, color="k", lw=2.2)
    ax[0, 0].axhline(0, color="k", ls=":", lw=1)
    ax[0, 0].set(xticks=range(len(cfgs)), ylim=(-0.2, 1.0),
                 ylabel="corr(added energy, deficit)", title="(a) does the guidance target the deficit?")
    ax[0, 0].set_xticklabels([LAB.get(c, c) for c in cfgs], rotation=18, ha="right", fontsize=9)

    for i, c in enumerate(cfgs):
        f = res[c]["closed"]
        ax[0, 1].scatter(np.full(len(f), i) + rng.uniform(-.18, .18, len(f)), f, s=13,
                         color=COL.get(c, "k"), alpha=.6)
        ax[0, 1].plot([i - .3, i + .3], [np.median(f)] * 2, color="k", lw=2.2)
    ax[0, 1].axhline(0, color="k", ls=":", lw=1)
    ax[0, 1].set(xticks=range(len(cfgs)), ylabel="fraction of the deficit closed",
                 title="(b) how much of the shortfall is recovered?")
    ax[0, 1].set_xticklabels([LAB.get(c, c) for c in cfgs], rotation=18, ha="right", fontsize=9)

    ax[1, 0].axhline(np.median(base_corr), color="#1f77b4", ls="--", lw=1.6, label="unguided")
    for i, c in enumerate(cfgs):
        m = res[c]["mapcorr"]
        ax[1, 0].scatter(np.full(len(m), i) + rng.uniform(-.18, .18, len(m)), m, s=13,
                         color=COL.get(c, "k"), alpha=.6)
        ax[1, 0].plot([i - .3, i + .3], [np.median(m)] * 2, color="k", lw=2.2)
    ax[1, 0].set(xticks=range(len(cfgs)), ylabel="map correlation with ground truth",
                 title="(c) resulting spatial agreement")
    ax[1, 0].set_xticklabels([LAB.get(c, c) for c in cfgs], rotation=18, ha="right", fontsize=9)
    ax[1, 0].legend(fontsize=9)

    j = 0
    c0 = cfgs[-1]
    vmax = float(np.percentile(np.abs(D[j]), 99))
    ax[1, 1].scatter(D[j].ravel()[::29], res[c0]["A"][j].ravel()[::29], s=3, alpha=.25,
                     color=COL.get(c0, "k"))
    ax[1, 1].plot([-vmax, vmax], [-vmax, vmax], "k--", lw=1)
    ax[1, 1].set(xlabel="local deficit (GT $-$ unguided)", ylabel="local energy added",
                 title=f"(d) one frame, {LAB.get(c0, c0)}\nr={res[c0]['targ'][j]:.2f}")

    fig.suptitle(f"Re={a.regime}, {a.model}: is the guidance aimed at the missing energy? "
                 f"(band $k\\in[{lo},{hi})$)", fontsize=13)
    for ext in (".pdf", ".png"):
        fig.savefig(out + ext, dpi=200)
    plt.close(fig)

    print(f"{a.model} @ Re={a.regime}, band [{lo},{hi}) — median over {len(G)} frames")
    print(f"  {'config':<22} {'targeting r':>12} {'deficit closed':>15} {'map corr':>9} "
          f"(unguided {np.median(base_corr):.3f})")
    for c in cfgs:
        print(f"  {LAB.get(c, c):<22} {np.median(res[c]['targ']):>12.3f} "
              f"{np.median(res[c]['closed']):>14.2f} {np.median(res[c]['mapcorr']):>9.3f}")
    print(f"figure -> {out}.pdf")


if __name__ == "__main__":
    main()
