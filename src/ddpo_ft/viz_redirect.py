"""Was the reconstruction error REDIRECTED toward GT — and does the new model beat plain DDPO at it?

Three-way error comparison on the same frames / same sampling noise:
  1) GT vorticity                     (context)
  2) NEW recon                        (the reconstruction itself)
  3) base − GT error                  (what needs fixing; shared scale with col 4)
  4) NEW − GT error                   (did it shrink / move?)
  5) error-reduction map NEW          |base−GT| − |NEW−GT|  (green = NEW closer to GT)
  6) error-reduction map CTRL         same for the plain-DDPO control — the fair baseline

Redirection numbers per row + aggregate:
  corr(NEW−base, GT−base)   does what NEW *adds* land on the true deficit?  (vs CTRL's value)
  % pixels improved, RMS errors.

    python -m src.ddpo_ft.viz_redirect <ckpt_new.pkl> --ckpt_ctrl <ckpt.pkl> [--seq 36] [--frames 4]
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


def main(ckpt_new, ckpt_ctrl=None, seq=36, frames=4, t_start=100, seed=1,
         gt="flow-data/kf_2d_re1000_256_40seed.npy", label_new="align", label_ctrl="plainDDPO",
         out="monitoring/ddpo_ckpts/viz_redirect.png"):
    ddpm, base_params, _ = build_base_ddpm()
    new_params = pickle.load(open(ckpt_new, "rb"))["params"]
    ctrl_params = pickle.load(open(ckpt_ctrl, "rb"))["params"] if ckpt_ctrl else None
    sampler = make_sampler(ddpm.unet, ddpm.alpha_bar, ddpm.beta_schedule, t_start, temp=1.0)
    ab = ddpm.alpha_bar
    sa, s1 = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))

    s = load_sequence(gt, seq)
    inp = build_triplets(sparse_nnfill_degrade(s, seq), MEAN, STD)
    xgt = build_triplets(s, MEAN, STD)
    idx = np.linspace(len(inp) // 4, len(inp) - 1, frames).astype(int)
    xin = jnp.asarray(inp[idx])
    gtf = np.asarray(xgt[idx])[..., 1] * STD
    print(f"reconstructing {frames} frames of seq {seq} (base / {label_new}"
          f"{' / ' + label_ctrl if ctrl_params else ''}) ...", flush=True)

    key = jax.random.PRNGKey(seed)
    def recon(params):                            # same key -> same start noise for all models
        k1, k2 = jax.random.split(key)
        x_start = sa * xin + s1 * jax.random.normal(k1, xin.shape)
        return np.asarray(sampler(params, x_start, k2))[..., 1] * STD
    base = recon(base_params)
    new = recon(new_params)
    ctrl = recon(ctrl_params) if ctrl_params else None

    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    corr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    ncol = 6 if ctrl is not None else 5
    fig, axes = plt.subplots(frames, ncol, figsize=(2.9 * ncol + 1.5, 3.1 * frames))
    axes = np.atleast_2d(axes)
    for r in range(frames):
        eb, en = base[r] - gtf[r], new[r] - gtf[r]
        red_n = np.abs(eb) - np.abs(en)                             # >0 = NEW closer to GT
        S = np.percentile(np.abs(eb), 99)
        V = np.percentile(np.abs(gtf[r]), 99)
        c_new = corr(new[r] - base[r], gtf[r] - base[r])            # redirection: change vs deficit
        pct_n = 100.0 * float((red_n > 0).mean())
        panels = [(f"GT", gtf[r], "RdBu_r", -V, V),
                  (f"{label_new} recon", new[r], "RdBu_r", -V, V),
                  (f"base − GT  (RMS {rms(eb):.2f})", eb, "RdBu_r", -S, S),
                  (f"{label_new} − GT  (RMS {rms(en):.2f})\ncorr(change,deficit)={c_new:+.2f}", en, "RdBu_r", -S, S),
                  (f"error reduction ({label_new})\ngreen = closer to GT  ({pct_n:.0f}% px)", red_n, "PiYG", -0.5 * S, 0.5 * S)]
        if ctrl is not None:
            ec = ctrl[r] - gtf[r]
            red_c = np.abs(eb) - np.abs(ec)
            c_ctrl = corr(ctrl[r] - base[r], gtf[r] - base[r])
            pct_c = 100.0 * float((red_c > 0).mean())
            panels.append((f"error reduction ({label_ctrl})\nRMS {rms(ec):.2f}  corr={c_ctrl:+.2f}  ({pct_c:.0f}% px)",
                           red_c, "PiYG", -0.5 * S, 0.5 * S))
        for c, (name, data, cmap, vmn, vmx) in enumerate(panels):
            a = axes[r, c]
            a.imshow(data, cmap=cmap, vmin=vmn, vmax=vmx)
            a.set_xticks([]); a.set_yticks([]); a.set_title(name, fontsize=8)
        axes[r, 0].set_ylabel(f"frame {idx[r]}", fontsize=9)

    # aggregates
    eb_all, en_all = base - gtf, new - gtf
    cn = np.mean([corr(new[r] - base[r], gtf[r] - base[r]) for r in range(frames)])
    line = (f"RMS: base {rms(eb_all):.3f}  {label_new} {rms(en_all):.3f} "
            f"({100*(1-rms(en_all)/rms(eb_all)):+.1f}%)  | corr({label_new} change, deficit) = {cn:+.3f}")
    if ctrl is not None:
        ec_all = ctrl - gtf
        cc = np.mean([corr(ctrl[r] - base[r], gtf[r] - base[r]) for r in range(frames)])
        line += (f"\nRMS: {label_ctrl} {rms(ec_all):.3f} ({100*(1-rms(ec_all)/rms(eb_all)):+.1f}%)  "
                 f"| corr({label_ctrl} change, deficit) = {cc:+.3f}")
    print(line, flush=True)
    fig.suptitle(f"seq {seq} — is the error redirected toward GT? ({os.path.basename(ckpt_new)})\n" + line,
                 y=1.02, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(out, dpi=115, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_new")
    ap.add_argument("--ckpt_ctrl", type=str, default=None)
    ap.add_argument("--seq", type=int, default=36)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--t_start", type=int, default=100)
    ap.add_argument("--gt", type=str, default="flow-data/kf_2d_re1000_256_40seed.npy")
    ap.add_argument("--label_new", type=str, default="align")
    ap.add_argument("--label_ctrl", type=str, default="plainDDPO")
    ap.add_argument("--out", type=str, default="monitoring/ddpo_ckpts/viz_redirect.png")
    main(**vars(ap.parse_args()))
