"""Disentangle the two levers: MODEL (base vs DDPO) x SAMPLING (K=1 vs K=3). Does K=3 on the BASE
model already lift high-k energy and its placement toward GT, or does that need DDPO?

For all four (base-K1, base-K3, DDPO-K1, DDPO-K3) at grid-4×: hik_ret (amount), placement E_hik<->GT
(where), mean residual. Plus a 6-curve enstrophy spectrum (GT, input, and the four) so base-K3 sits
next to DDPO-K3.

    python -m src.ddpo_ft.diag_k3_disentangle <ddpo_ckpt.pkl> [--seq 0] [--frames 8] [--re 2000]
                                               [--gt <path>] [--grid_factor 4]
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
from src.rewards import make_spectrum_fn          # noqa: E402
from src.sequence_inference import (              # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence, sparse_nnfill_degrade)

MEAN, STD, N, HIK0 = 0.0, 4.7988, 256, 32
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"
S_MULTI, T_SINGLE = [150, 100, 50], 100


def main(ckpt, seqs="0", frames=8, re=2000, gt=None, grid_factor=4, seed=1, kcut=32, sigma=6.0,
         out=None):
    out = out or f"monitoring/ab_pdelocal/k3_disentangle_re{re}.png"
    seqs = [int(x) for x in str(seqs).split(",")]
    ddpm, base_params, _ = build_base_ddpm()
    ddpo = pickle.load(open(ckpt, "rb"))["params"]
    ab = ddpm.alpha_bar
    spec_fn = make_spectrum_fn(N)
    resid = jax.jit(make_ns_residual(n=N, re=float(re)))
    samplers = {t: make_sampler(ddpm.unet, ab, ddpm.beta_schedule, t, temp=1.0)
                for t in sorted({T_SINGLE, *S_MULTI})}
    key = jax.random.PRNGKey(seed)

    xin, xgt = [], []
    for s in seqs:
        seq = load_sequence(gt or GT_PATH, s)
        deg = grid_downsample_degrade(seq, grid_factor) if grid_factor else sparse_nnfill_degrade(seq, s)
        a, g = build_triplets(deg, MEAN, STD), build_triplets(seq, MEAN, STD)
        idx = np.linspace(len(g) // 4, len(g) - 1, frames).astype(int)
        xin.append(a[idx]); xgt.append(g[idx])
    xin, xgt = jnp.asarray(np.concatenate(xin)), np.concatenate(xgt)
    print(f"reconstructing base/DDPO x K1/K3, {len(xgt)} frames (re={re}, grid={grid_factor}) ...", flush=True)

    def noise_to(x, t, k):
        return float(jnp.sqrt(ab[t])) * x + float(jnp.sqrt(1.0 - ab[t])) * jax.random.normal(k, x.shape)

    def one(p):
        nonlocal key; key, k1, k2 = jax.random.split(key, 3)
        return np.asarray(samplers[T_SINGLE](p, noise_to(xin, T_SINGLE, k1), k2))

    def k3(p):
        nonlocal key; xc = xin
        for Sj in S_MULTI:
            key, k1, k2 = jax.random.split(key, 3)
            xc = jnp.asarray(samplers[Sj](p, noise_to(xc, Sj, k1), k2))
        return np.asarray(xc)

    Eg = np.asarray(spec_fn(jnp.asarray(xgt))); Ehg = local_hik_energy(xgt[..., 1] * STD, kcut, sigma)
    recons = {"base K=1": one(base_params), "base K=3": k3(base_params),
              "DDPO K=1": one(ddpo), "DDPO K=3": k3(ddpo)}
    spectra = {"GT": Eg.mean(0), "input": np.asarray(spec_fn(xin)).mean(0)}
    print(f"\n{'variant':<10}{'hik_ret':>9}{'placement':>11}{'MSE':>9}{'mean|resid|':>12}")
    for nm, x0 in recons.items():
        Er = np.asarray(spec_fn(x0)); Eh = local_hik_energy(x0[..., 1] * STD, kcut, sigma)
        hik = float((Er[:, HIK0:].sum(-1) / Eg[:, HIK0:].sum(-1)).mean())
        pl = float(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1])
        rs = float(np.abs(np.asarray(resid(jnp.asarray(x0) * STD))).mean())
        mse = float(((x0 - xgt) ** 2).mean())
        spectra[nm] = Er.mean(0)
        print(f"{nm:<10}{hik:>9.3f}{pl:>11.3f}{mse:>9.4f}{rs:>12.2f}", flush=True)
    print(f"{'GT':<10}{1.000:>9.3f}{1.000:>11.3f}{0.0:>9.4f}"
          f"{float(np.abs(np.asarray(resid(jnp.asarray(xgt) * STD))).mean()):>12.2f}", flush=True)

    kx = np.arange(1, 96)
    col = {"GT": "k", "input": "#c94f4f", "base K=1": "#9498a0", "base K=3": "#5b6169",
           "DDPO K=1": "#3ca951", "DDPO K=3": "#1f6fd0"}
    sty = {"GT": "-", "input": ":", "base K=1": "--", "base K=3": "-", "DDPO K=1": "--", "DDPO K=3": "-"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for nm in ("GT", "input", "base K=1", "base K=3", "DDPO K=1", "DDPO K=3"):
        ax.plot(kx, spectra[nm][1:96], color=col[nm], lw=2.4 if nm == "GT" else 1.6, ls=sty[nm], label=nm)
    ax.axvspan(HIK0, 95, color="#3ca951", alpha=0.06); ax.axvline(HIK0, color="gray", lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("wavenumber k"); ax.set_ylabel("E(k)")
    ax.set_title(f"Re={re} grid-{grid_factor}× — base vs DDPO × K=1 vs K=3 (does K=3 alone lift high-k?)",
                 fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seqs", type=str, default="0")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--re", type=int, default=2000)
    ap.add_argument("--gt", type=str, default=None)
    ap.add_argument("--grid_factor", type=int, default=4)
    ap.add_argument("--out", type=str, default=None)
    main(**vars(ap.parse_args()))
