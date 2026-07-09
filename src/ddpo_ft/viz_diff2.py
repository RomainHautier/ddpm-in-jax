"""Where does DDPO change the reconstruction? Per frame:
  1) DDPO recon (context)
  2) DDPO - base on its OWN scale (±99th pct) — the CHANGE PATTERN (visible now)
  3) |DDPO - base| magnitude heatmap — WHERE it changes (bright = big change)
  4) GT - base on its own scale — the TRUE missing structure, for spatial comparison
Title per row: spatial correlation corr(DDPO-base, GT-base) — >0 means DDPO's changes point
toward the real deficit (adding energy where it's actually missing).

    python -m src.ddpo_ft.viz_diff2 <ckpt.pkl> [--seq 36] [--frames 3] [--t_start 100]
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


def main(ckpt, seq=36, frames=3, t_start=100, seed=1, out="monitoring/ddpo_ckpts/viz_diff2.png"):
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
    print(f"reconstructing {frames} frames of seq {seq} ...", flush=True)

    key = jax.random.PRNGKey(seed)
    def recon(params):
        k1, k2 = jax.random.split(key)
        x_start = sqrt_ab * xin + sqrt_1m * jax.random.normal(k1, xin.shape)
        return np.asarray(sampler(params, x_start, k2))[..., 1] * STD
    base, ddpo, gtf = recon(base_params), recon(ddpo_params), xgt[..., 1] * STD
    dd, gd = ddpo - base, gtf - base                                # DDPO change, true missing

    fig, axes = plt.subplots(frames, 5, figsize=(16.5, 3.3 * frames))
    axes = np.atleast_2d(axes)
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    for r in range(frames):
        eb, ed = base[r] - gtf[r], ddpo[r] - gtf[r]                 # base error, DDPO error (vs GT)
        red = np.abs(eb) - np.abs(ed)                              # >0 where DDPO is CLOSER to GT
        vd = dd[r]                                                  # DDPO - base (the change)
        S = np.percentile(np.abs(eb), 99)                          # shared error scale
        pct_better = 100.0 * float((red > 0).mean())
        corr_rd = float(np.corrcoef(red.ravel(), np.abs(gd[r]).ravel())[0, 1])   # does help land on the deficit?
        panels = [(f"base − GT  (error, RMS {rms(eb):.2f})", eb, "RdBu_r", -S, S),
                  (f"DDPO − GT  (error, RMS {rms(ed):.2f})", ed, "RdBu_r", -S, S),
                  (f"error reduction  |b−GT|−|d−GT|\n(green=DDPO closer; {pct_better:.0f}% px better)", red, "PiYG", -0.5 * S, 0.5 * S),
                  (f"DDPO − base (the change)  corr↔deficit={corr_rd:+.2f}", vd, "RdBu_r",
                   -np.percentile(np.abs(vd), 99), np.percentile(np.abs(vd), 99)),
                  ("GT − base  (true deficit)", gd[r], "RdBu_r", -np.percentile(np.abs(gd[r]), 99), np.percentile(np.abs(gd[r]), 99))]
        for c, (name, data, cmap, vmn, vmx) in enumerate(panels):
            a = axes[r, c]
            a.imshow(data, cmap=cmap, vmin=vmn, vmax=vmx)
            a.set_xticks([]); a.set_yticks([]); a.set_title(name, fontsize=8.5)
        axes[r, 0].set_ylabel(f"frame {idx[r]}", fontsize=9)
    # aggregate
    eb_all, ed_all = (base - gtf), (ddpo - gtf)
    print(f"RMS(base−GT)={rms(eb_all):.3f}  RMS(DDPO−GT)={rms(ed_all):.3f}  "
          f"({100*(1-rms(ed_all)/rms(eb_all)):+.1f}% error)", flush=True)
    fig.suptitle(f"Re=1000 seq {seq} — does DDPO shrink the base−GT residual? ({os.path.basename(ckpt)})\n"
                 f"cols 1-2 same scale (smaller=better); green in col 3 = DDPO moved toward GT", y=1.01, fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=115, bbox_inches="tight")
    mean_corr = np.mean([np.corrcoef(dd[r].ravel(), gd[r].ravel())[0, 1] for r in range(frames)])
    print(f"saved {out}  | mean corr(DDPO−base, GT−base) = {mean_corr:+.3f}", flush=True)

    # ---- vorticity distribution (PDF) : base vs DDPO vs GT ----
    allb, alld, allg = base.ravel(), ddpo.ravel(), gtf.ravel()
    vmax = np.percentile(np.abs(allg), 99.9)
    bins = np.linspace(-vmax, vmax, 140); cen = 0.5 * (bins[1:] + bins[:-1])
    qs = np.linspace(0, 1, 501)
    w1 = lambda a: float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(allg, qs))))
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, data, c in [("base", allb, "#9498a0"), ("DDPO", alld, "#3ca951"), ("GT", allg, "k")]:
        h, _ = np.histogram(data, bins, density=True)
        a1.plot(cen, h, color=c, lw=1.8, label=name)
        a2.semilogy(cen, h + 1e-8, color=c, lw=1.8, label=name)
    a1.set_title("vorticity PDF (core)", fontsize=10); a1.set_xlabel("vorticity ω"); a1.legend(frameon=False)
    a2.set_title("vorticity PDF (log-y — the intermittent tails)", fontsize=10); a2.set_xlabel("vorticity ω"); a2.legend(frameon=False)
    fig2.suptitle(f"Vorticity distribution — W1(base,GT)={w1(allb):.3f}  W1(DDPO,GT)={w1(alld):.3f}  "
                  f"(lower=closer to GT)", y=1.02, fontsize=10)
    plt.tight_layout()
    pdfout = out.replace("viz_diff2", "viz_vortpdf")
    plt.savefig(pdfout, dpi=115, bbox_inches="tight")
    print(f"saved {pdfout}  | W1(base,GT)={w1(allb):.3f}  W1(DDPO,GT)={w1(alld):.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seq", type=int, default=36)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--t_start", type=int, default=100)
    main(**vars(ap.parse_args()))
