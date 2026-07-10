"""Two questions at once, over the Re=1000 val(32-35)+test(36-39) sequences:

  (Q2) Does the reconstruction actually IMPROVE on the NN-filled INPUT, or just stay physical?
       -> the NN-filled sparse input is included as a baseline "model" row.
  (Q3) Does the MULTI-PHASE denoiser (K=3, S=[150,100,50]) beat the single chain (t_start=100)
       for the best DDPO model (align)?

Rows compared vs GT: input (NN-fill) | base 1-chain | align 1-chain | align K=3 multi-phase.
Metrics: hi-k energy retention R(k>=32), MSE, PDE residual, effective resolution k*.
Same sampling noise (seed) across models; frames evenly subsampled per sequence (--n_per_seq).

    python -m src.ddpo_ft.eval_multiphase <align_ckpt.pkl> [--n_per_seq 32] [--n_samples 1]
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

from eval_ddpo import make_sampler, eff_resolution  # noqa: E402
from train_claude import build_base_ddpm          # noqa: E402
from src.rewards import make_residual_loss, make_spectrum_fn  # noqa: E402
from src.sequence_inference import (               # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence, sparse_nnfill_degrade)

MEAN, STD, N, HIK0 = 0.0, 4.7988, 256, 32
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"
VAL, TEST = [32, 33, 34, 35], [36, 37, 38, 39]
S_MULTI = [150, 100, 50]                          # K=3 multi-phase schedule (inference_config.yaml)
T_SINGLE = 100


def evaluate(align_ckpt, n_per_seq=32, n_samples=1, batch=24, seed=0, grid_factor=None,
             re=1000, gt_path=None, val=None, test=None):
    gt_path = gt_path or GT_PATH
    val = val if val is not None else VAL
    test = test if test is not None else TEST
    degrade = (lambda seq, s: grid_downsample_degrade(seq, grid_factor)) if grid_factor \
        else sparse_nnfill_degrade
    ddpm, base_params, _ = build_base_ddpm()
    align_params = pickle.load(open(align_ckpt, "rb"))["params"]
    ab = ddpm.alpha_bar
    spec_fn = make_spectrum_fn(N)
    resid_fn = make_residual_loss(n=N, re=float(re), std=STD, mean=MEAN)
    # jitted samplers for every noise level we touch (single chain + the 3 multi-phase levels)
    samplers = {t: make_sampler(ddpm.unet, ab, ddpm.beta_schedule, t, temp=1.0)
                for t in sorted({T_SINGLE, *S_MULTI})}
    key = jax.random.PRNGKey(seed)

    def noise_to(x, t, k):
        return float(jnp.sqrt(ab[t])) * x + float(jnp.sqrt(1.0 - ab[t])) * jax.random.normal(k, x.shape)

    def single_chain(params, x_cond):
        nonlocal key
        key, k1, k2 = jax.random.split(key, 3)
        return np.asarray(samplers[T_SINGLE](params, noise_to(x_cond, T_SINGLE, k1), k2))

    def multiphase(params, x_cond):                # K=3 renoise/denoise refinement, from the input
        nonlocal key
        x = x_cond
        for Sj in S_MULTI:
            key, k1, k2 = jax.random.split(key, 3)
            x = jnp.asarray(samplers[Sj](params, noise_to(x, Sj, k1), k2))
        return np.asarray(x)

    def batched(fn, x_cond):
        return np.concatenate([fn(x_cond[i:i + batch]) for i in range(0, len(x_cond), batch)])

    def metrics(x0, xgt, E_gt):
        E_r = np.asarray(spec_fn(x0))
        hik = E_r[:, HIK0:].sum(-1) / E_gt[:, HIK0:].sum(-1)
        mse = ((x0 - xgt) ** 2).reshape(len(x0), -1).mean(-1)
        resid = np.asarray(resid_fn(jnp.asarray(x0)))
        return dict(hik=hik.mean(), hik_sd=hik.std(), mse=mse.mean(), resid=resid.mean(),
                    kstar=eff_resolution(E_r.mean(0), E_gt.mean(0)), spec=E_r.mean(0))

    results, spectra = {}, {}
    for split, seqs in ((f"val{val}", val), (f"test{test}", test)):
        xin, xgt = [], []
        for s in seqs:
            seq = load_sequence(gt_path, s)
            inp = build_triplets(degrade(seq, s), MEAN, STD)
            gt = build_triplets(seq, MEAN, STD)
            idx = np.linspace(0, len(inp) - 1, n_per_seq).astype(int)
            xin.append(inp[idx]); xgt.append(gt[idx])
        xin = np.repeat(np.concatenate(xin), n_samples, 0)
        xgt = np.repeat(np.concatenate(xgt), n_samples, 0)
        xin_j = jnp.asarray(xin)
        E_gt = np.asarray(spec_fn(jnp.asarray(xgt)))
        print(f"\n[{split}] {len(xin)} frames — reconstructing input/base/align-1chain/align-K3 ...", flush=True)

        row = {"gt_resid": float(np.asarray(resid_fn(jnp.asarray(xgt))).mean()),
               "input": metrics(xin, xgt, E_gt),                                  # NN-fill baseline
               "base_1chain": metrics(batched(lambda x: single_chain(base_params, x), xin_j), xgt, E_gt),
               "align_1chain": metrics(batched(lambda x: single_chain(align_params, x), xin_j), xgt, E_gt),
               "align_K3": metrics(batched(lambda x: multiphase(align_params, x), xin_j), xgt, E_gt)}
        results[split] = row
        spectra[split] = {k: row[k]["spec"] for k in ("input", "base_1chain", "align_1chain", "align_K3")}
        spectra[split]["gt"] = E_gt.mean(0)

    # ---- table ----
    print(f"\n{'split / model':<20}{'hik_ret':>13}{'MSE':>10}{'residual':>11}{'k*':>6}")
    for split in results:
        r = results[split]
        print(f"{split:<20}{'':>13}{'':>10}{r['gt_resid']:>9.1f}gt{'':>6}")
        for name in ("input", "base_1chain", "align_1chain", "align_K3"):
            m = r[name]
            print(f"  {name:<18}{m['hik']:>8.3f}±{m['hik_sd']:>3.2f}{m['mse']:>10.4f}{m['resid']:>11.1f}{m['kstar']:>6}")

    # ---- spectrum figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    kx = np.arange(1, 96)
    col = {"input": "#c94f4f", "base_1chain": "#9498a0", "align_1chain": "#3ca951", "align_K3": "#1f6fd0"}
    for j, split in enumerate(spectra):
        a = axes[j]; sp = spectra[split]
        a.plot(kx, sp["gt"][1:96], "k-", lw=2.2, label="GT")
        for name in ("input", "base_1chain", "align_1chain", "align_K3"):
            a.plot(kx, sp[name][1:96], color=col[name], lw=1.5,
                   ls="--" if name in ("input", "base_1chain") else "-", label=name)
        a.axvspan(HIK0, 95, color="#3ca951", alpha=0.06); a.axvline(HIK0, color="gray", lw=0.6, ls=":")
        a.set_xscale("log"); a.set_yscale("log"); a.set_title(f"enstrophy spectrum — {split}", fontsize=10)
        a.set_xlabel("wavenumber k"); a.set_ylabel("E(k)"); a.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Re={re} — NN-input vs base vs model(1-chain) vs model(K=3 multi-phase) "
                 f"({os.path.basename(align_ckpt)})", y=1.02)
    plt.tight_layout()
    figpath = f"monitoring/ab_pdelocal/eval_multiphase_re{re}.png"
    plt.savefig(figpath, dpi=115, bbox_inches="tight")
    print(f"\nsaved {figpath}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("align_ckpt")
    ap.add_argument("--n_per_seq", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--grid_factor", type=int, default=None, help="grid-N input instead of random-1024")
    ap.add_argument("--re", type=int, default=1000)
    ap.add_argument("--gt", type=str, default=None, help="GT .npy (default Re=1000)")
    ap.add_argument("--val", type=str, default=None, help="comma-list val seq ids")
    ap.add_argument("--test", type=str, default=None, help="comma-list test seq ids")
    a = ap.parse_args()
    _lst = lambda s: [int(x) for x in s.split(",")] if s else None
    evaluate(a.align_ckpt, n_per_seq=a.n_per_seq, n_samples=a.n_samples, grid_factor=a.grid_factor,
             re=a.re, gt_path=a.gt, val=_lst(a.val), test=_lst(a.test))
