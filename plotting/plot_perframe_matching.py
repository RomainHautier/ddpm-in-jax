#!/usr/bin/env python
"""Per-frame diagnostics for any set of stored/graded configurations at one regime.

    (a) energy matching: each frame's reconstructed fine-scale energy against its OWN ground
        truth's, with the identity line - the picture of conditional vs marginal dose;
    (b) ratio curves: each frame's ratio, ordered by how energetic that frame is, with the
        +-20% tolerance band;
    (c) per-frame placement distributions;
    (d) placement against energy ratio - does over-dosing a frame cost its placement?

Reads the audit stores (per-frame arrays), no inference.

    JAX_PLATFORMS=cpu python plotting/plot_perframe_matching.py --model base0
"""
import argparse, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import matplotlib                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "font.size": 11})
import numpy as np                       # noqa: E402

LAB = {"none": "unguided", "rewardv2": "v2 dose dial", "all3v2": "all dials (v2)",
       "v6gate": "taper+scaling+gate", "v5taperscale": "taper+scaling", "v4scaled": "scaling only"}
COL = {"none": "#1f77b4", "rewardv2": "#b8399e", "all3v2": "#d4770a",
       "v6gate": "#0f9e78", "v5taperscale": "#7a4bd0", "v4scaled": "#c22f4f"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="1000")
    ap.add_argument("--model", default="base0")
    ap.add_argument("--configs", default="none,rewardv2,v6gate")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = a.regime
    store = "base_results/re1000_audit.npz" if R == "1000" else f"base_results/regime_audit_re{R}.npz"
    A = np.load(store, allow_pickle=True)
    out = a.out or f"plotting/figs/perframe_{a.model}_re{R}"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    gt = np.asarray(A[f"{R}|GT||psEb"])[:, 3:5].sum(1)           # each frame's true fine energy
    cfgs = [c for c in a.configs.split(",") if f"{R}|{a.model}|K3|{c}||ps_ret_paired" in A.files]
    if not cfgs:
        sys.exit("no per-frame data for those configurations")
    order = np.argsort(gt)

    fig, ax = plt.subplots(2, 2, figsize=(12.6, 9.6), constrained_layout=True)

    # (a) energy matching
    lim = float(gt.max()) * 1.15
    ax[0, 0].plot([0, lim], [0, lim], "k--", lw=1.2, label="perfect (identity)")
    for c in cfgs:
        r = np.asarray(A[f"{R}|{a.model}|K3|{c}||ps_ret_paired"])
        ax[0, 0].scatter(gt, r * gt, s=16, color=COL.get(c, "k"), alpha=.75, label=LAB.get(c, c))
    ax[0, 0].set(xlabel="frame's TRUE fine-scale energy", ylabel="frame's RECONSTRUCTED energy",
                 xlim=(0, lim), ylim=(0, lim), title="(a) per-frame energy matching")
    ax[0, 0].legend(fontsize=8.5)

    # (b) ratio curves ordered by frame energy
    ax[0, 1].axhspan(0.8, 1.2, color="#e3edf3", zorder=0, label="$\\pm$20% tolerance")
    ax[0, 1].axhline(1, color="k", ls="--", lw=1.2)
    x = gt[order] / gt.mean()
    for c in cfgs:
        r = np.asarray(A[f"{R}|{a.model}|K3|{c}||ps_ret_paired"])[order]
        ax[0, 1].plot(x, r, "o-", color=COL.get(c, "k"), lw=1, ms=3.5, alpha=.8, label=LAB.get(c, c))
    ax[0, 1].set(xlabel="frame's fine-scale energy / mean", ylabel="reconstructed / true",
                 yscale="log", title="(b) per-frame ratio, ordered by frame energy")
    ax[0, 1].legend(fontsize=8.5)

    # (c) per-frame placement
    for i, c in enumerate(cfgs):
        p = np.asarray(A[f"{R}|{a.model}|K3|{c}||ps_place"])
        ax[1, 0].scatter(np.full(len(p), i) + np.random.default_rng(0).uniform(-.18, .18, len(p)),
                         p, s=13, color=COL.get(c, "k"), alpha=.6)
        ax[1, 0].plot([i - .3, i + .3], [np.median(p)] * 2, color="k", lw=2.2)
    ax[1, 0].set(xticks=range(len(cfgs)), ylabel="per-frame placement correlation",
                 title="(c) placement, frame by frame")
    ax[1, 0].set_xticklabels([LAB.get(c, c) for c in cfgs], rotation=18, ha="right", fontsize=9)

    # (d) placement vs energy ratio
    for c in cfgs:
        r = np.asarray(A[f"{R}|{a.model}|K3|{c}||ps_ret_paired"])
        p = np.asarray(A[f"{R}|{a.model}|K3|{c}||ps_place"])
        rr = np.corrcoef(np.log(r), p)[0, 1]
        ax[1, 1].scatter(r, p, s=14, color=COL.get(c, "k"), alpha=.65,
                         label=f"{LAB.get(c, c)}  r={rr:+.2f}")
    ax[1, 1].axvline(1, color="k", ls="--", lw=1.2)
    ax[1, 1].set(xscale="log", xlabel="frame's energy ratio (reconstructed / true)",
                 ylabel="that frame's placement", title="(d) does mis-dosing a frame cost placement?")
    ax[1, 1].legend(fontsize=8.5)

    fig.suptitle(f"Re={R}, {a.model}: per-frame behaviour under each sampling configuration", fontsize=13)
    for ext in (".pdf", ".png"):
        fig.savefig(out + ext, dpi=200)
    plt.close(fig)
    print(f"{a.model} @ Re={R}")
    for c in cfgs:
        r = np.asarray(A[f"{R}|{a.model}|K3|{c}||ps_ret_paired"])
        p = np.asarray(A[f"{R}|{a.model}|K3|{c}||ps_place"])
        print(f"  {LAB.get(c, c):<22} in-band {np.mean((r > .8) & (r < 1.2)) * 100:>3.0f}%  "
              f"ratio {np.median(r):.2f} [{np.percentile(r, 10):.2f},{np.percentile(r, 90):.2f}]  "
              f"place {np.median(p):.3f}  slope {np.polyfit(np.log(gt), np.log(r * gt), 1)[0]:.3f}")
    print(f"figure -> {out}.pdf")


if __name__ == "__main__":
    main()
