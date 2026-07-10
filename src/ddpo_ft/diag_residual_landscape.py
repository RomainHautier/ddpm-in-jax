"""Does reducing the recon's PDE residual move it TOWARD GT, or toward a smoother-than-GT (lower
-residual) solution? Two probes on GT / base / DDPO reconstructions at grid-4×:

  (1) residual vs low-pass cutoff kc: smooth each field (keep k<kc) and measure its NS residual.
      If GT's residual DROPS as it is smoothed, GT is NOT the residual minimum -> a naive residual
      -minimizing reward over-smooths PAST GT. Where the recon's curve sits shows whether its excess
      residual is removable toward the GT floor.
  (2) residual POWER SPECTRUM: radial |FFT(residual field)|^2 for GT vs recon. If the recon has excess
      residual power at HIGH k that GT lacks, the excess is broadband speckle (artifact) — removable
      without eroding GT structure; a spectral/anti-speckle residual term is the lever.

    python -m src.ddpo_ft.diag_residual_landscape <ddpo_ckpt.pkl> [--seqs 32,36] [--frames 6]
                                                  [--re 1000] [--gt <path>] [--grid_factor 4]
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
from src.physics_guidance import make_ns_residual  # noqa: E402
from src.sequence_inference import (              # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence, sparse_nnfill_degrade)

MEAN, STD, N = 0.0, 4.7988, 256
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"
_k = np.fft.fftfreq(N) * N
_KMAG = np.sqrt(_k[:, None] ** 2 + _k[None, :] ** 2)
_KR = np.round(_KMAG).astype(int)


def lowpass(triplet, kc):
    """Keep radial wavenumbers < kc in each of the 3 channels. triplet: (B, N, N, 3)."""
    F = np.fft.fft2(triplet, axes=(1, 2))
    F *= (_KMAG < kc)[None, :, :, None]
    return np.real(np.fft.ifft2(F, axes=(1, 2)))


def radial_power(field2d):
    """Radially-averaged power spectrum of a (B, N, N) field: mean over B of |FFT|^2 binned by |k|."""
    P = np.abs(np.fft.fft2(field2d, axes=(1, 2))) ** 2
    out = np.zeros(96)
    for b in range(P.shape[0]):
        tb = np.bincount(_KR.ravel(), P[b].ravel(), minlength=N)[:96]
        nb = np.maximum(np.bincount(_KR.ravel(), minlength=N)[:96], 1)
        out += tb / nb
    return out / P.shape[0]


def main(ckpt, seqs="32,36", frames=6, t_start=100, re=1000, gt=None, grid_factor=4, seed=1,
         out=None):
    out = out or f"monitoring/ab_pdelocal/residual_landscape_re{re}.png"
    seqs = [int(x) for x in str(seqs).split(",")]
    ddpm, base_params, _ = build_base_ddpm()
    ddpo = pickle.load(open(ckpt, "rb"))["params"]
    sampler = make_sampler(ddpm.unet, ddpm.alpha_bar, ddpm.beta_schedule, t_start, temp=1.0)
    resid = jax.jit(make_ns_residual(n=N, re=float(re)))
    ab = ddpm.alpha_bar
    sa, s1 = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))

    xin, xgt = [], []
    for s in seqs:
        seq = load_sequence(gt or GT_PATH, s)
        deg = grid_downsample_degrade(seq, grid_factor) if grid_factor else sparse_nnfill_degrade(seq, s)
        a, g = build_triplets(deg, MEAN, STD), build_triplets(seq, MEAN, STD)
        idx = np.linspace(len(g) // 4, len(g) - 1, frames).astype(int)
        xin.append(a[idx]); xgt.append(g[idx])
    xin, xgt = jnp.asarray(np.concatenate(xin)), np.concatenate(xgt)
    key = jax.random.PRNGKey(seed)
    def one(p):
        nonlocal key; key, k1, k2 = jax.random.split(key, 3)
        return np.asarray(sampler(p, sa * xin + s1 * jax.random.normal(k1, xin.shape), k2))
    fields = {"GT": np.asarray(xgt), "base": one(base_params), "DDPO": one(ddpo)}
    R = lambda t: float(np.abs(np.asarray(resid(jnp.asarray(t) * STD))).mean())

    # (1) residual vs low-pass cutoff
    kcs = [8, 12, 16, 24, 32, 48, 64, 96, 200]                  # 200 = full field
    print(f"\n(1) mean |residual| vs low-pass cutoff kc  (re={re}, {len(xgt)} frames)")
    print(f"{'kc':>6}" + "".join(f"{n:>9}" for n in fields))
    curves = {n: [] for n in fields}
    for kc in kcs:
        row = {n: R(lowpass(fields[n], kc) if kc < N else fields[n]) for n in fields}
        for n in fields:
            curves[n].append(row[n])
        print(f"{('full' if kc >= N else kc):>6}" + "".join(f"{row[n]:>9.2f}" for n in fields))

    # (2) residual power spectrum (full fields)
    Rp = {n: radial_power(np.abs(np.asarray(resid(jnp.asarray(fields[n]) * STD)))) for n in fields}

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
    kx = [kc if kc < N else 128 for kc in kcs]
    col = {"GT": "k", "base": "#9498a0", "DDPO": "#3ca951"}
    for n in fields:
        ax[0].plot(kx, curves[n], "o-", color=col[n], lw=1.9, label=n)
    ax[0].axhline(curves["GT"][-1], color="k", lw=0.7, ls=":")
    ax[0].set_xlabel("low-pass cutoff kc (128 = full field)"); ax[0].set_ylabel("mean |NS residual|")
    ax[0].set_title("residual vs smoothing — does smoothing lower residual below GT?", fontsize=10)
    ax[0].legend(fontsize=9, frameon=False); ax[0].grid(alpha=.25)
    kk = np.arange(1, 96)
    for n in fields:
        ax[1].plot(kk, Rp[n][1:96], color=col[n], lw=1.9, label=n)
    ax[1].axvspan(32, 95, color="#3ca951", alpha=0.06); ax[1].axvline(32, color="gray", lw=0.6, ls=":")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("wavenumber k")
    ax[1].set_ylabel("residual power |FFT(resid)|²")
    ax[1].set_title("residual power spectrum — is the recon excess high-k speckle?", fontsize=10)
    ax[1].legend(fontsize=9, frameon=False); ax[1].grid(alpha=.25)
    fig.suptitle(f"Re={re} grid-{grid_factor}× — PDE residual landscape (GT floor={curves['GT'][-1]:.2f}, "
                 f"base={curves['base'][-1]:.2f}, DDPO={curves['DDPO'][-1]:.2f})", y=1.02, fontsize=11)
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seqs", type=str, default="32,36")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--re", type=int, default=1000)
    ap.add_argument("--gt", type=str, default=None)
    ap.add_argument("--grid_factor", type=int, default=4)
    ap.add_argument("--out", type=str, default=None)
    main(**vars(ap.parse_args()))
