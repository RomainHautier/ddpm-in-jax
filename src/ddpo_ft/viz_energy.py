"""Is LOCAL high-k energy a GT-free proxy for WHERE the reconstruction is over-smoothed?

Companion to viz_pde.py. The PDE residual localizes WRONG-structure error (spurious/misplaced energy).
This asks the opposite question: can the spatial map of small-scale (high-k) energy tell us WHERE the
model has left structure OUT (the over-smoothing / missing-energy error)?

Local high-k energy density  E_hik(x) = gaussblur( |highpass_{k>=kcut}(x)|^2 ):
per pixel, how much fine-scale vorticity energy sits in its neighbourhood.

Per frame, columns:
  1) |base − GT|         the actual error (needs GT) — what we'd want to localize
  2) E_hik(base)         recon's own local high-k energy (NO GT) — candidate GT-free map
  3) E_hik(DDPO)         after finetuning (did DDPO move energy toward GT's placement?)
  4) E_hik(GT)           the target field — where the fine energy actually belongs

Correlations printed / titled:
  * corr(E_hik_base, |err|)      does high local energy sit where the error is? (energy IS structure)
  * corr(deficit, |err|)         deficit = relu(E_hik_GT − E_hik_base): does the MISSING energy (needs
                                 GT) localize the error? — i.e. is over-smoothing the dominant error?
  * placement corr base↔GT, DDPO↔GT:  does the recon put its fine energy in the RIGHT PLACES (phase),
                                 and does DDPO improve that? (the phase-blindness probe, spatially)

    python -m src.ddpo_ft.viz_energy <ckpt.pkl> [--seq 36] [--frames 3] [--kcut 32] [--sigma 6] [--re 1000]
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

import jax                                          # noqa: E402
import jax.numpy as jnp                             # noqa: E402
import numpy as np                                  # noqa: E402
import matplotlib.pyplot as plt                     # noqa: E402

from eval_ddpo import make_sampler                  # noqa: E402
from train_claude import build_base_ddpm            # noqa: E402
from src.sequence_inference import (               # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence, sparse_nnfill_degrade)

MEAN, STD = 0.0, 4.7988
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"


def local_hik_energy(w, kcut=32, sigma=6.0):
    """Local high-k energy density of a vorticity field w (..., H, W), all numpy.
    highpass keeps radial wavenumbers >= kcut (index units), square, then Gaussian-blur to a
    per-pixel neighbourhood density. Returns (..., H, W) >= 0."""
    H, W = w.shape[-2:]
    fy = np.fft.fftfreq(H) * H                       # integer wavenumber index
    fx = np.fft.fftfreq(W) * W
    kmag = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    hp_mask = (kmag >= kcut).astype(np.float32)
    F = np.fft.fft2(w)
    hp = np.real(np.fft.ifft2(F * hp_mask))          # small-scale part of the field
    e = hp ** 2                                       # local energy
    # Gaussian blur in Fourier space -> smooth local density (dependency-free)
    g = np.exp(-2.0 * (np.pi * sigma) ** 2 * ((fy[:, None] / H) ** 2 + (fx[None, :] / W) ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(e) * g))


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main(ckpt, seq=36, frames=3, t_start=100, re=1000, kcut=32, sigma=6.0, seed=1,
         out="monitoring/ddpo_ckpts/viz_energy.png", grid_factor=None, gt=None):
    ddpm, base_params, _ = build_base_ddpm()
    ddpo_params = pickle.load(open(ckpt, "rb"))["params"]
    sampler = make_sampler(ddpm.unet, ddpm.alpha_bar, ddpm.beta_schedule, t_start, temp=1.0)
    ab = ddpm.alpha_bar
    sqrt_ab, sqrt_1m = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))

    s = load_sequence(gt or GT_PATH, seq)
    degraded = grid_downsample_degrade(s, grid_factor) if grid_factor else sparse_nnfill_degrade(s, seq)
    inp = build_triplets(degraded, MEAN, STD)
    gt = build_triplets(s, MEAN, STD)
    idx = np.linspace(len(inp) // 4, len(inp) - 1, frames).astype(int)
    xin, xgt = jnp.asarray(inp[idx]), np.asarray(gt[idx])
    print(f"reconstructing {frames} frames of seq {seq} + local high-k energy (kcut={kcut}, sigma={sigma}) ...",
          flush=True)

    key = jax.random.PRNGKey(seed)
    def recon(params):
        k1, k2 = jax.random.split(key)
        x_start = sqrt_ab * xin + sqrt_1m * jax.random.normal(k1, xin.shape)
        return np.asarray(sampler(params, x_start, k2))          # (B,256,256,3) full triplet
    b0, d0 = recon(base_params), recon(ddpo_params)

    # middle-frame vorticity (denormalized), same channel viz_pde uses for the error
    wb, wd, wg = b0[..., 1] * STD, d0[..., 1] * STD, xgt[..., 1] * STD
    Eb = local_hik_energy(wb, kcut, sigma)
    Ed = local_hik_energy(wd, kcut, sigma)
    Eg = local_hik_energy(wg, kcut, sigma)
    err_base = np.abs(wb - wg)                                    # actual pointwise error
    deficit = np.maximum(Eg - Eb, 0.0)                           # missing energy (needs GT)

    fig, axes = plt.subplots(frames, 4, figsize=(13.5, 3.4 * frames))
    axes = np.atleast_2d(axes)
    for r in range(frames):
        c_e = _corr(Eb[r], err_base[r])
        c_def = _corr(deficit[r], err_base[r])
        c_place = _corr(Eb[r], Eg[r])
        vE = np.percentile(err_base[r], 99)
        vH = np.percentile(np.concatenate([Eb[r], Ed[r], Eg[r]]).ravel(), 99)
        panels = [(f"|base − GT|  (actual error)", err_base[r], vE, "magma"),
                  (f"E_hik(base)  corr(·,err)={c_e:+.2f}", Eb[r], vH, "viridis"),
                  (f"E_hik(DDPO)  place↔GT={_corr(Ed[r], Eg[r]):+.2f}", Ed[r], vH, "viridis"),
                  (f"E_hik(GT)  base place↔GT={c_place:+.2f}", Eg[r], vH, "viridis")]
        for c, (name, data, vmx, cm) in enumerate(panels):
            a = axes[r, c]
            a.imshow(data, cmap=cm, vmin=0, vmax=vmx)
            a.set_xticks([]); a.set_yticks([]); a.set_title(name, fontsize=8.5)
        axes[r, 0].set_ylabel(f"frame {idx[r]}\ncorr(deficit,err)={c_def:+.2f}", fontsize=8)

    # aggregate
    ce = _corr(Eb, err_base)
    cdef = _corr(deficit, err_base)
    cpb = _corr(Eb, Eg)                                           # base placement vs GT
    cpd = _corr(Ed, Eg)                                           # DDPO placement vs GT
    frac_missing = float(Eb.sum() / (Eg.sum() + 1e-12))          # how much hik energy base retains
    print(f"corr(E_hik_base, |err|)      = {ce:+.3f}   (energy sits where structure is, not where error is)", flush=True)
    print(f"corr(deficit=GT−base, |err|) = {cdef:+.3f}   (>0 => over-smoothing/missing-energy localizes the error)", flush=True)
    print(f"placement corr  base↔GT = {cpb:+.3f}   DDPO↔GT = {cpd:+.3f}   (Δ={cpd - cpb:+.3f}, does DDPO fix WHERE?)", flush=True)
    print(f"  => this is corr(E_hik_model, E_hik_GT). POLARITY IS OPPOSITE TO THE RESIDUAL PROBE:", flush=True)
    print(f"     energy is ADDED, so HIGH corr with GT is GOOD (model puts energy in GT's places -> adding fills", flush=True)
    print(f"     the right spots). {cpb:+.2f} = only partial, and DDPO leaves it flat => it adds energy but not WHERE.", flush=True)
    print(f"high-k energy retained: base/GT = {frac_missing:.3f}", flush=True)
    fig.suptitle(f"Re={re} seq {seq} — is LOCAL high-k energy a GT-free localizer of over-smoothing? "
                 f"({os.path.basename(ckpt)})\ncorr(deficit,err)={cdef:+.2f}  "
                 f"placement base↔GT={cpb:+.2f} → DDPO↔GT={cpd:+.2f}", y=1.01, fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=115, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seq", type=int, default=36)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--t_start", type=int, default=100)
    ap.add_argument("--re", type=int, default=1000)
    ap.add_argument("--kcut", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=6.0)
    ap.add_argument("--out", type=str, default="monitoring/ddpo_ckpts/viz_energy.png")
    ap.add_argument("--grid_factor", type=int, default=None, help="grid-N input instead of random-1024")
    ap.add_argument("--gt", type=str, default=None, help="GT .npy (default Re=1000); e.g. Re=2000 path")
    main(**vars(ap.parse_args()))
