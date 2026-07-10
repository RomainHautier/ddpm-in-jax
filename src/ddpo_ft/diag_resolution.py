"""Does the placement ceiling lift with input resolution? Same energy-placement / redirection
analysis, swept over CLEAN regular-grid downsampling factors (deterministic every-N-pixel + NN-fill),
which brackets the current 1024-random-point regime:

    factor 4  -> 64x64 = 4096 pts (6.25%, 4x DENSER than current)
    factor 8  -> 32x32 = 1024 pts (1.56%, same COUNT as current random mask, but gridded)
    factor 16 -> 16x16 =  256 pts (0.39%, 4x SPARSER)
    [ref] random-1024 (the real task) for calibration -> should reproduce placement ~0.44

Base model, single chain t_start=100, same sampling noise. Per factor, vs GT:
  placement corr  E_hik(recon) <-> E_hik(GT)   THE ceiling metric (0.44 at random-1024)
  redirection     corr(recon - input, GT - input)   does the model's correction point at GT?
  hik retention, PDE residual, MSE.

    python -m src.ddpo_ft.diag_resolution [--seqs 32,36] [--frames 8]
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
from scipy.ndimage import distance_transform_edt  # noqa: E402

from eval_ddpo import make_sampler                # noqa: E402
from train_claude import build_base_ddpm          # noqa: E402
from viz_energy import local_hik_energy           # noqa: E402
from src.rewards import make_residual_loss, make_spectrum_fn  # noqa: E402
from src.sequence_inference import build_triplets, load_sequence, sparse_nnfill_degrade  # noqa: E402

MEAN, STD, N, HIK0 = 0.0, 4.7988, 256, 32
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"


def grid_downsample_degrade(seq, factor):
    """Deterministic regular-grid downsample: keep every `factor`-th pixel in each dim, NN-fill rest.
    A clean, reproducible degradation anchor (vs the random-collocation task mask)."""
    H, W = seq.shape[-2], seq.shape[-1]
    mask = np.zeros((H, W), bool)
    mask[::factor, ::factor] = True
    _, ind = distance_transform_edt(~mask, return_indices=True)
    return np.stack([f[ind[0], ind[1]] for f in seq]), int(mask.sum())


def _corr(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def main(seqs="32,36", frames=8, t_start=100, seed=1, kcut=32, sigma=6.0,
         out="monitoring/ab_pdelocal/diag_resolution.png"):
    seqs = [int(s) for s in seqs.split(",")]
    ddpm, base_params, _ = build_base_ddpm()
    sampler = make_sampler(ddpm.unet, ddpm.alpha_bar, ddpm.beta_schedule, t_start, temp=1.0)
    spec_fn = make_spectrum_fn(N)
    resid_fn = make_residual_loss(n=N, re=1000.0, std=STD, mean=MEAN)
    ab = ddpm.alpha_bar
    sa, s1 = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))
    key = jax.random.PRNGKey(seed)

    # collect GT + the per-degradation inputs for the chosen frames across seqs
    gt_list, frame_idx = [], []
    degr = {f"grid{f}x": [] for f in (4, 8, 16)}
    degr["rand1024"] = []
    npts = {}
    for s in seqs:
        seq = load_sequence(GT_PATH, s)
        gt = build_triplets(seq, MEAN, STD)
        idx = np.linspace(len(gt) // 4, len(gt) - 1, frames).astype(int)
        gt_list.append(gt[idx]); frame_idx.append(idx)
        for f in (4, 8, 16):
            d, n = grid_downsample_degrade(seq, f); npts[f"grid{f}x"] = n
            degr[f"grid{f}x"].append(build_triplets(d, MEAN, STD)[idx])
        degr["rand1024"].append(build_triplets(sparse_nnfill_degrade(seq, s), MEAN, STD)[idx])
    xgt = np.concatenate(gt_list); npts["rand1024"] = 1024
    E_gt = np.asarray(spec_fn(jnp.asarray(xgt)))
    Ehik_gt = local_hik_energy(xgt[..., 1] * STD, kcut, sigma)

    def recon(x_cond):
        nonlocal key
        key, k1, k2 = jax.random.split(key, 3)
        xc = jnp.asarray(x_cond)
        return np.asarray(sampler(base_params, sa * xc + s1 * jax.random.normal(k1, xc.shape), k2))

    order = ["grid4x", "grid8x", "rand1024", "grid16x"]
    rows = {}
    print(f"\nseqs {seqs} x {frames} frames = {len(xgt)} frames | base model, t_start={t_start}", flush=True)
    print(f"{'degradation':<12}{'#pts':>7}{'%':>7}{'placement':>11}{'redirect':>10}{'hik_ret':>9}"
          f"{'residual':>10}{'MSE':>9}   {'inp_place':>10}{'inp_hik':>8}")
    for name in order:
        xin = np.concatenate(degr[name])
        x0 = recon(xin)
        E_r = np.asarray(spec_fn(x0)); Ehik_r = local_hik_energy(x0[..., 1] * STD, kcut, sigma)
        Ehik_in = local_hik_energy(xin[..., 1] * STD, kcut, sigma)
        place = _corr(Ehik_r, Ehik_gt)                                   # THE ceiling metric
        redir = _corr((x0 - xin)[..., 1], (xgt - xin)[..., 1])           # correction points at GT?
        hik = float((E_r[:, HIK0:].sum(-1) / E_gt[:, HIK0:].sum(-1)).mean())
        resid = float(np.asarray(resid_fn(jnp.asarray(x0))).mean())
        mse = float(((x0 - xgt) ** 2).mean())
        inp_place = _corr(Ehik_in, Ehik_gt)                              # input's own placement
        inp_hik = float((np.asarray(spec_fn(jnp.asarray(xin)))[:, HIK0:].sum(-1)
                         / E_gt[:, HIK0:].sum(-1)).mean())
        rows[name] = dict(place=place, redir=redir, hik=hik, resid=resid, mse=mse,
                          inp_place=inp_place, inp_hik=inp_hik, npts=npts[name])
        print(f"{name:<12}{npts[name]:>7}{100*npts[name]/(N*N):>6.2f}%{place:>11.3f}{redir:>10.3f}"
              f"{hik:>9.3f}{resid:>10.1f}{mse:>9.4f}   {inp_place:>10.3f}{inp_hik:>8.2f}", flush=True)

    # figure: placement (recon & input) vs #points
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = [rows[n]["npts"] for n in order]
    ax[0].plot(xs, [rows[n]["place"] for n in order], "o-", lw=1.8, label="recon placement (E_hik↔GT)")
    ax[0].plot(xs, [rows[n]["inp_place"] for n in order], "s--", color="#c94f4f", lw=1.5, label="input placement")
    ax[0].axhline(0.441, color="gray", lw=0.8, ls=":", label="ref 0.44 (random-1024)")
    for n in order:
        ax[0].annotate(n, (rows[n]["npts"], rows[n]["place"]), fontsize=7, xytext=(3, 4), textcoords="offset points")
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("# observed points"); ax[0].set_ylabel("placement corr")
    ax[0].set_title("energy placement vs input resolution", fontsize=10); ax[0].legend(fontsize=8, frameon=False)
    ax[0].grid(alpha=.25)
    ax[1].plot(xs, [rows[n]["hik"] for n in order], "o-", lw=1.8, label="recon hik_ret")
    ax[1].plot(xs, [rows[n]["mse"] for n in order], "^-", color="#3ca951", lw=1.5, label="recon MSE")
    ax[1].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("# observed points")
    ax[1].set_title("retention & MSE vs input resolution", fontsize=10); ax[1].legend(fontsize=8, frameon=False)
    ax[1].grid(alpha=.25)
    fig.suptitle("Re=1000 base model — reconstruction quality vs clean-grid input resolution", y=1.02, fontsize=10)
    plt.tight_layout(); plt.savefig(out, dpi=115, bbox_inches="tight")
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=str, default="32,36")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--t_start", type=int, default=100)
    main(**vars(ap.parse_args()))
