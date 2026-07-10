"""At what input resolution does DDPO start to PAY OFF? Sweep clean-grid downsampling factors and
track the two placement signals that decide whether the reward's added energy / residual-cleanup
lands in the right places:

  energy placement   corr(E_hik(recon), E_hik(GT))       added spectral energy lands correctly?
  PDE placement      corr(|resid(recon)|, |resid(GT)|)   residual structure sits on the real
                                                         dynamical features (not own artifact)?

When both cross ~0.6, the model already knows WHERE — so the spectral/PDE reward becomes additive
(fixes amount) instead of adding structure in the wrong places (the 1024-pt dead end).

Base model, single chain t_start=100, same noise. Grid factors -> point counts bracket 256..7396.

    python -m src.ddpo_ft.diag_resolution_sweep [--seqs 32,36] [--frames 8] [--factors 16,10,8,6,5,4,3]
"""
import argparse
import os
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
from diag_resolution import grid_downsample_degrade  # noqa: E402
from src.physics_guidance import make_ns_residual  # noqa: E402
from src.rewards import make_spectrum_fn          # noqa: E402
from src.sequence_inference import build_triplets, load_sequence, sparse_nnfill_degrade  # noqa: E402

MEAN, STD, N, HIK0 = 0.0, 4.7988, 256, 32
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"


def _corr(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def main(seqs="32,36", frames=8, factors="16,10,8,6,5,4,3", t_start=100, seed=1, kcut=32, sigma=6.0,
         out="monitoring/ab_pdelocal/diag_resolution_sweep.png"):
    seqs = [int(s) for s in seqs.split(",")]
    factors = [int(f) for f in factors.split(",")]
    ddpm, base_params, _ = build_base_ddpm()
    sampler = make_sampler(ddpm.unet, ddpm.alpha_bar, ddpm.beta_schedule, t_start, temp=1.0)
    spec_fn = make_spectrum_fn(N)
    resid = jax.jit(make_ns_residual(n=N, re=1000.0))
    ab = ddpm.alpha_bar
    sa, s1 = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))
    key = jax.random.PRNGKey(seed)

    # GT + reference fields once
    gt_list, seq_objs = [], []
    for s in seqs:
        seq = load_sequence(GT_PATH, s); seq_objs.append((s, seq))
        gt = build_triplets(seq, MEAN, STD)
        idx = np.linspace(len(gt) // 4, len(gt) - 1, frames).astype(int)
        gt_list.append((gt[idx], idx))
    xgt = np.concatenate([g for g, _ in gt_list])
    Ehik_gt = local_hik_energy(xgt[..., 1] * STD, kcut, sigma)
    R_gt = np.abs(np.asarray(resid(jnp.asarray(xgt) * STD)))
    E_gt = np.asarray(spec_fn(jnp.asarray(xgt)))

    def recon(x_cond):
        nonlocal key
        key, k1, k2 = jax.random.split(key, 3)
        xc = jnp.asarray(x_cond)
        return np.asarray(sampler(base_params, sa * xc + s1 * jax.random.normal(k1, xc.shape), k2))

    def build_input(degrade_fn):
        out = []
        for (s, seq), (_, idx) in zip(seq_objs, gt_list):
            d = degrade_fn(seq, s)
            out.append(build_triplets(d, MEAN, STD)[idx])
        return np.concatenate(out)

    # each entry: (label, npts, degrade_fn)
    configs = []
    for f in factors:
        _, npts = grid_downsample_degrade(seq_objs[0][1], f)
        configs.append((f"grid{f}x", npts, lambda seq, s, ff=f: grid_downsample_degrade(seq, ff)[0]))
    configs.append(("rand1024", 1024, lambda seq, s: sparse_nnfill_degrade(seq, s)))
    configs.sort(key=lambda c: c[1])

    print(f"\nseqs {seqs} x {frames} = {len(xgt)} frames | base model, t_start={t_start}", flush=True)
    print(f"{'config':<11}{'#pts':>7}{'%':>7}{'E-place':>9}{'PDE-place':>10}{'resid<->err':>12}"
          f"{'hik_ret':>9}{'residual':>10}{'MSE':>9}", flush=True)
    rows = []
    for label, npts, degrade in configs:
        x0 = recon(build_input(degrade))
        Ehik_r = local_hik_energy(x0[..., 1] * STD, kcut, sigma)
        R_r = np.abs(np.asarray(resid(jnp.asarray(x0) * STD)))
        err = np.abs((x0 - xgt)[..., 1] * STD)
        E_r = np.asarray(spec_fn(x0))
        e_place = _corr(Ehik_r, Ehik_gt)
        pde_place = _corr(R_r, R_gt)                                  # residual sits where GT's does
        r_err = _corr(R_r, err)
        hik = float((E_r[:, HIK0:].sum(-1) / E_gt[:, HIK0:].sum(-1)).mean())
        mse = float(((x0 - xgt) ** 2).mean())
        rows.append(dict(label=label, npts=npts, e_place=e_place, pde_place=pde_place, hik=hik, mse=mse))
        print(f"{label:<11}{npts:>7}{100*npts/(N*N):>6.2f}%{e_place:>9.3f}{pde_place:>10.3f}"
              f"{r_err:>12.3f}{hik:>9.3f}{R_r.mean():>10.1f}{mse:>9.4f}", flush=True)

    grid = [r for r in rows if r["label"].startswith("grid")]
    xs = [r["npts"] for r in grid]
    ep = [r["e_place"] for r in grid]; pp = [r["pde_place"] for r in grid]

    def cross(ys, thr=0.6):
        for i in range(1, len(ys)):
            if ys[i - 1] < thr <= ys[i]:
                x0_, x1_ = xs[i - 1], xs[i]; y0_, y1_ = ys[i - 1], ys[i]
                return int(x0_ + (thr - y0_) / (y1_ - y0_) * (x1_ - x0_))
        return None
    xe, xp = cross(ep), cross(pp)
    print(f"\n  energy placement crosses 0.6 at ~{xe} points" if xe else "\n  energy placement < 0.6 across sweep")
    print(f"  PDE   placement crosses 0.6 at ~{xp} points" if xp else "  PDE placement < 0.6 across sweep")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ep, "o-", lw=2, label="energy placement  E_hik↔GT")
    ax.plot(xs, pp, "s-", lw=2, color="#c94f4f", label="PDE placement  |resid|↔|resid_GT|")
    r1024 = [r for r in rows if r["label"] == "rand1024"][0]
    ax.scatter([1024], [r1024["e_place"]], marker="*", s=140, color="#3ca951", zorder=5,
               label=f"random-1024 (real task) E-place={r1024['e_place']:.2f}")
    ax.axhline(0.6, color="gray", lw=0.9, ls="--", label="~0.6 'DDPO pays off' threshold")
    for r in grid:
        ax.annotate(r["label"], (r["npts"], r["e_place"]), fontsize=7, xytext=(3, 5), textcoords="offset points")
    ax.set_xscale("log", base=2); ax.set_xlabel("# observed points (of 65536)"); ax.set_ylabel("placement corr")
    ax.set_title("Placement vs input resolution — where does DDPO start to pay off?", fontsize=11)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.25); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=str, default="32,36")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--factors", type=str, default="16,10,8,6,5,4,3")
    ap.add_argument("--t_start", type=int, default=100)
    main(**vars(ap.parse_args()))
