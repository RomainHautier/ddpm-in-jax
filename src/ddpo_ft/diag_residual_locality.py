"""How LOCAL should a PDE-residual reward be? Spatial-scale diagnostics of the residual field,
its spatial gradient, and its relation to HIGH-SHEAR (strain-rate) regions.

Reconstructs base-model frames (Re=1000 sparse task) and measures, on the residual field R:
  (1) correlation length  ell_R  (radial autocorrelation 1/e)   -> pixel-span of a residual structure
                                                                   = the MINIMUM meaningful patch size
  (2) spatial gradient |grad R|  -> where the residual CHANGES fast (thin sharp edges); its own
      correlation length ell_gradR (< ell_R)                     -> the finest useful targeting scale
  (3) SHEAR relation: strain-rate sigma_s from the velocity field (high sigma_s = shear/filament
      regions). corr(sigma_s, R^2), corr(sigma_s, |grad R|), and enrichment of R^2 / |grad R| in the
      top-decile-shear pixels                                    -> is "target high shear" ~ "target residual"?
  (4) patch-scale: variance of R^2 retained by a PxP block-mean, and fraction of total R^2 in the
      worst-10% of PxP blocks, vs P in {1,2,4,8,16,32}           -> the practical reward patch size

    python -m src.ddpo_ft.diag_residual_locality <ckpt.pkl> [--seq 36] [--frames 6] [--re 1000]
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
from src.physics_guidance import make_ns_residual   # noqa: E402
from src.sequence_inference import build_triplets, load_sequence, sparse_nnfill_degrade  # noqa: E402

MEAN, STD = 0.0, 4.7988
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"


def strain_rate(w, L=2 * np.pi):
    """Strain-rate magnitude sigma_s = sqrt((u_x - v_y)^2 + (v_x + u_y)^2) from vorticity w (H,W),
    via streamfunction (psi_hat = w_hat / k^2). High sigma_s = shear/strain (filament) regions."""
    H, W = w.shape
    k = np.fft.fftfreq(W) * W * (2 * np.pi / L)
    kx, ky = k[None, :], k[:, None]
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0
    wh = np.fft.fft2(w)
    psih = wh / k2
    psih[0, 0] = 0.0
    u = np.real(np.fft.ifft2(1j * ky * psih))
    v = np.real(np.fft.ifft2(-1j * kx * psih))
    uh, vh = np.fft.fft2(u), np.fft.fft2(v)
    ux = np.real(np.fft.ifft2(1j * kx * uh)); uy = np.real(np.fft.ifft2(1j * ky * uh))
    vx = np.real(np.fft.ifft2(1j * kx * vh)); vy = np.real(np.fft.ifft2(1j * ky * vh))
    return np.sqrt((ux - vy) ** 2 + (vx + uy) ** 2)


def radial_autocorr(field):
    """Radially-averaged spatial autocorrelation C(r), normalized C(0)=1. r index in pixels."""
    f = field - field.mean()
    F = np.fft.fft2(f)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F) ** 2))) / f.size
    ac /= ac.max()
    H, W = field.shape
    yy, xx = np.indices((H, W))
    rr = np.round(np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)).astype(int)
    return np.bincount(rr.ravel(), ac.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)


def corr_length(field, thr=np.exp(-1)):
    C = radial_autocorr(field)
    below = np.where(C < thr)[0]
    return (int(below[0]) if len(below) else len(C)), C


def grad_mag(field):
    gy, gx = np.gradient(field)
    return np.sqrt(gx ** 2 + gy ** 2)


def var_retained(field, P):
    """Fraction of field variance captured by a PxP block-mean (coarsening to scale P)."""
    H, W = field.shape
    Hc, Wc = H // P * P, W // P * P
    f = field[:Hc, :Wc]
    bm = f.reshape(Hc // P, P, Wc // P, P).mean(axis=(1, 3))
    up = np.repeat(np.repeat(bm, P, 0), P, 1)
    return float(1 - ((f - up) ** 2).mean() / (f.var() + 1e-20))


def patch_conc(field, P, frac=0.1):
    """Fraction of total sum(field) held by the worst-`frac` PxP blocks."""
    H, W = field.shape
    Hc, Wc = H // P * P, W // P * P
    b = field[:Hc, :Wc].reshape(Hc // P, P, Wc // P, P).sum(axis=(1, 3)).ravel()
    b = np.sort(b)[::-1]
    ntop = max(1, int(len(b) * frac))
    return float(b[:ntop].sum() / (b.sum() + 1e-20))


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main(ckpt, seq=36, frames=6, t_start=100, re=1000, seed=1,
         out="monitoring/ddpo_ckpts/diag_residual_locality"):
    ddpm, base_params, _ = build_base_ddpm()
    _ = pickle.load(open(ckpt, "rb"))["params"]                # (base is what matters for reward design)
    sampler = make_sampler(ddpm.unet, ddpm.alpha_bar, ddpm.beta_schedule, t_start, temp=1.0)
    resid = jax.jit(make_ns_residual(n=256, re=float(re)))
    ab = ddpm.alpha_bar
    sqrt_ab, sqrt_1m = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))

    s = load_sequence(GT_PATH, seq)
    inp = build_triplets(sparse_nnfill_degrade(s, seq), MEAN, STD)
    idx = np.linspace(len(inp) // 4, len(inp) - 1, frames).astype(int)
    xin = jnp.asarray(inp[idx])
    print(f"reconstructing {frames} frames of seq {seq} (base model) for residual-locality diagnostics ...", flush=True)
    k = jax.random.PRNGKey(seed); k1, k2 = jax.random.split(k)
    x0 = np.asarray(sampler(base_params, sqrt_ab * xin + sqrt_1m * jax.random.normal(k1, xin.shape), k2))

    R = np.asarray(resid(jnp.asarray(x0) * STD))                # signed residual field (frames,256,256)
    R2 = R ** 2
    gR = np.stack([grad_mag(R[i]) for i in range(frames)])      # |grad R|
    W = x0[..., 1] * STD                                        # middle-frame vorticity (physical)
    Sh = np.stack([strain_rate(W[i]) for i in range(frames)])   # strain-rate (shear)

    # ---- (1)/(2) correlation lengths ----
    lR = np.mean([corr_length(R[i])[0] for i in range(frames)])
    lR2 = np.mean([corr_length(R2[i])[0] for i in range(frames)])
    lgR = np.mean([corr_length(gR[i])[0] for i in range(frames)])
    lSh = np.mean([corr_length(Sh[i])[0] for i in range(frames)])
    print(f"\n[1/2] correlation lengths (1/e, pixels):  R={lR:.1f}  R^2={lR2:.1f}  |gradR|={lgR:.1f}  strain={lSh:.1f}")
    print(f"      -> residual structures span ~{lR:.0f}px; its gradient ~{lgR:.0f}px (finest scale); "
          f"shear is broader (~{lSh:.0f}px)")

    # ---- (3) shear relation ----
    cSR2 = _corr(Sh, R2); cSgR = _corr(Sh, gR)
    hi = Sh >= np.percentile(Sh, 90, axis=(-2, -1), keepdims=True)     # top-decile shear per frame
    enR2 = float(R2[hi].mean() / R2.mean()); engR = float(gR[hi].mean() / gR.mean())
    print(f"\n[3] SHEAR relation:  corr(strain, R^2)={cSR2:+.3f}   corr(strain, |gradR|)={cSgR:+.3f}")
    print(f"    enrichment in top-10% shear:  R^2 x{enR2:.2f}   |gradR| x{engR:.2f}   "
          f"(>1 => residual/its-gradient concentrate in shear)")

    # ---- (4) patch-scale curves ----
    Ps = [1, 2, 4, 8, 16, 32]
    vr = [np.mean([var_retained(R2[i], P) for i in range(frames)]) for P in Ps]
    pc = [np.mean([patch_conc(R2[i], P) for i in range(frames)]) for P in Ps]
    print(f"\n[4] patch scale P:        " + "  ".join(f"{P:>5}" for P in Ps))
    print(f"    var(R^2) retained:    " + "  ".join(f"{v:5.2f}" for v in vr) + "   (fraction of residual structure kept by PxP block-mean)")
    print(f"    R^2 in worst-10% blk: " + "  ".join(f"{v:5.2f}" for v in pc) + "   (concentration; ~flat => coherent at that scale)")

    # ---- figures ----
    fig, axes = plt.subplots(min(frames, 4), 4, figsize=(14, 3.4 * min(frames, 4)))
    axes = np.atleast_2d(axes)
    for r in range(min(frames, 4)):
        panels = [("vorticity ω", W[r], "RdBu_r", None),
                  ("R² (residual²)", R2[r], "magma", 99),
                  ("|∇R| (residual gradient)", gR[r], "magma", 99),
                  ("strain σ_s (shear)", Sh[r], "viridis", 99)]
        for c, (nm, d, cm, pc_) in enumerate(panels):
            a = axes[r, c]
            if pc_:
                a.imshow(d, cmap=cm, vmin=0, vmax=np.percentile(d, pc_))
            else:
                m = np.percentile(np.abs(d), 99); a.imshow(d, cmap=cm, vmin=-m, vmax=m)
            a.set_xticks([]); a.set_yticks([]); a.set_title(nm, fontsize=9)
        axes[r, 0].set_ylabel(f"frame {idx[r]}", fontsize=9)
    fig.suptitle(f"Re={re} seq {seq} — residual locality: ℓ_R≈{lR:.0f}px, ℓ_∇R≈{lgR:.0f}px | "
                 f"corr(strain,R²)={cSR2:+.2f}, R² ×{enR2:.1f} in top-10% shear", y=1.005, fontsize=10)
    plt.tight_layout(); plt.savefig(f"{out}_maps.png", dpi=115, bbox_inches="tight")
    print(f"\nsaved {out}_maps.png", flush=True)

    fig2, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    CR = radial_autocorr(R2[0]); Cg = radial_autocorr(gR[0]); CS = radial_autocorr(Sh[0])
    rr = np.arange(min(60, len(CR)))
    ax[0].plot(rr, CR[:len(rr)], label=f"R² (ℓ≈{lR2:.0f})", lw=1.8)
    ax[0].plot(rr, Cg[:len(rr)], label=f"|∇R| (ℓ≈{lgR:.0f})", lw=1.8)
    ax[0].plot(rr, CS[:len(rr)], label=f"strain (ℓ≈{lSh:.0f})", lw=1.8, ls="--")
    ax[0].axhline(np.exp(-1), color="k", lw=0.6, ls=":"); ax[0].set_xlabel("lag r (pixels)")
    ax[0].set_ylabel("autocorrelation"); ax[0].set_title("spatial autocorrelation → correlation length", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False); ax[0].grid(alpha=.25)
    ax[1].plot(Ps, vr, "o-", label="var(R²) retained by PxP block")
    ax[1].plot(Ps, pc, "s-", label="R² in worst-10% blocks")
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("patch size P (pixels)"); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("patch-scale: how much residual survives coarsening", fontsize=10)
    ax[1].legend(fontsize=8, frameon=False); ax[1].grid(alpha=.25)
    fig2.suptitle(f"Re={re} seq {seq} — reward-granularity diagnostic", y=1.02, fontsize=10)
    plt.tight_layout(); plt.savefig(f"{out}_scales.png", dpi=115, bbox_inches="tight")
    print(f"saved {out}_scales.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seq", type=int, default=36)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--t_start", type=int, default=100)
    ap.add_argument("--re", type=int, default=1000)
    main(**vars(ap.parse_args()))
