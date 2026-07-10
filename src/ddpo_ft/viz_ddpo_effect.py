"""What does the DDPO finetune actually CHANGE vs base? Per frame, difference maps:
  E_hik(base) | E_hik(DDPO) | E_hik(GT) | DDPO added (Eddpo−Ebase) | GT deficit (Egt−Ebase)
  | Δresidual (|r_ddpo|−|r_base|) | Δerror (|ddpo−GT|−|base−GT|, blue=better)
The test of "is the finetune sensible": does the DDPO-ADDED energy (col 4) match the GT DEFICIT
(col 5)? Titles give corr(added, deficit) and per-frame MSE base→DDPO. Single chain (K=1).

    python -m src.ddpo_ft.viz_ddpo_effect <ckpt.pkl> [--seq 36] [--frames 4] [--grid_factor 4]
                                           [--re 1000] [--gt <path>]
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
from viz_energy import local_hik_energy           # noqa: E402
from src.physics_guidance import make_ns_residual  # noqa: E402
from src.sequence_inference import (              # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence, sparse_nnfill_degrade)

MEAN, STD, N = 0.0, 4.7988, 256
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"


def _c(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main(ckpt, seq=36, frames=4, t_start=100, grid_factor=4, seed=1, re=1000, gt=None, kcut=32, sigma=6.0,
         out=None):
    out = out or f"monitoring/ab_pdelocal/ddpo_effect_re{re}.png"
    ddpm, base_params, _ = build_base_ddpm()
    ddpo = pickle.load(open(ckpt, "rb"))["params"]
    sampler = make_sampler(ddpm.unet, ddpm.alpha_bar, ddpm.beta_schedule, t_start, temp=1.0)
    resid = jax.jit(make_ns_residual(n=N, re=float(re)))
    ab = ddpm.alpha_bar
    sa, s1 = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))

    s = load_sequence(gt or GT_PATH, seq)
    degraded = grid_downsample_degrade(s, grid_factor) if grid_factor else sparse_nnfill_degrade(s, seq)
    inp = build_triplets(degraded, MEAN, STD)
    gtt = build_triplets(s, MEAN, STD)
    idx = np.linspace(len(inp) // 4, len(inp) - 1, frames).astype(int)
    xin, xgt = jnp.asarray(inp[idx]), np.asarray(gtt[idx])
    print(f"reconstructing base & DDPO, {frames} frames seq {seq} (re={re}, grid={grid_factor}) ...", flush=True)

    key = jax.random.PRNGKey(seed)
    def one(p):
        nonlocal key; key, k1, k2 = jax.random.split(key, 3)
        return np.asarray(sampler(p, sa * xin + s1 * jax.random.normal(k1, xin.shape), k2))
    b0, d0 = one(base_params), one(ddpo)

    wg, wb, wd = xgt[..., 1] * STD, b0[..., 1] * STD, d0[..., 1] * STD
    Eb, Ed, Eg = (local_hik_energy(w, kcut, sigma) for w in (wb, wd, wg))
    Rb = np.abs(np.asarray(resid(jnp.asarray(b0) * STD))); Rd = np.abs(np.asarray(resid(jnp.asarray(d0) * STD)))
    added = Ed - Eb                                        # what DDPO added (high-k energy)
    deficit = Eg - Eb                                      # what was actually missing
    dresid = Rd - Rb                                       # residual change
    eb, ed = np.abs(wb - wg), np.abs(wd - wg)
    derr = ed - eb                                         # error change (<0 = DDPO closer to GT)

    ca = _c(added, deficit)                                # aggregate: does added match deficit?
    print(f"corr(DDPO-added, GT-deficit) = {ca:+.3f}   "
          f"MSE base {(eb**2).mean():.4f} -> DDPO {(ed**2).mean():.4f}   "
          f"mean|resid| base {Rb.mean():.2f} -> DDPO {Rd.mean():.2f}", flush=True)

    fig, axes = plt.subplots(frames, 7, figsize=(19.5, 2.8 * frames))
    axes = np.atleast_2d(axes)
    for r in range(frames):
        vH = np.percentile(np.concatenate([Eb[r], Ed[r], Eg[r]]).ravel(), 99)
        vD = np.percentile(np.abs(np.concatenate([added[r], deficit[r]]).ravel()), 99)
        vR = np.percentile(np.abs(dresid[r]), 99); vE = np.percentile(np.abs(derr[r]), 99)
        cf = _c(added[r], deficit[r]); mb = (eb[r] ** 2).mean(); md = (ed[r] ** 2).mean()
        cells = [("E_hik(base)", Eb[r], "viridis", 0, vH), ("E_hik(DDPO)", Ed[r], "viridis", 0, vH),
                 ("E_hik(GT)", Eg[r], "viridis", 0, vH),
                 (f"DDPO added  ρ(·,def)={cf:+.2f}", added[r], "RdBu_r", -vD, vD),
                 ("GT deficit (target)", deficit[r], "RdBu_r", -vD, vD),
                 ("Δresidual (red=worse)", dresid[r], "RdBu_r", -vR, vR),
                 (f"Δerror (blue=better) {md-mb:+.4f}", derr[r], "RdBu", -vE, vE)]
        for c, (nm, dat, cm, vmn, vmx) in enumerate(cells):
            a = axes[r, c]; a.imshow(dat, cmap=cm, vmin=vmn, vmax=vmx)
            a.set_xticks([]); a.set_yticks([]); a.set_title(nm, fontsize=8)
        axes[r, 0].set_ylabel(f"frame {idx[r]}", fontsize=9)
    fig.suptitle(f"Re={re} grid-{grid_factor}× — what the DDPO finetune changes vs base (seq {seq})   "
                 f"[corr(added,deficit)={ca:+.2f}  MSE {(eb**2).mean():.4f}→{(ed**2).mean():.4f}  "
                 f"resid {Rb.mean():.1f}→{Rd.mean():.1f}]", y=1.005, fontsize=10.5)
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seq", type=int, default=36)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--t_start", type=int, default=100)
    ap.add_argument("--grid_factor", type=int, default=4)
    ap.add_argument("--re", type=int, default=1000)
    ap.add_argument("--gt", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    main(**vars(ap.parse_args()))
