"""Skeptic's view of x0-guidance: what does it REALLY do vs unguided (lam=0)? Per frame:
  GT | unguided | guided | |resid(unguided)| | |resid(guided)| | Δ|resid| (blue=down,red=up) | E_hik(GT)
Plus an enstrophy spectrum (GT/unguided/guided) and printed metrics that specifically test for a
spatial trade-off: residual at HIGH-ENERGY locations (top-decile |w| and E_hik), corr(Δresid, energy),
PDE residual placement corr(|resid|,|resid_GT|), retention, MSE.

    python -m src.ddpo_ft.viz_guided_effect <ckpt.pkl> [--lam 30] [--seqs 32,36] [--frames 4]
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

from diag_guided_residual import make_guided_sampler  # noqa: E402
from train_claude import build_base_ddpm          # noqa: E402
from viz_energy import local_hik_energy           # noqa: E402
from src.physics_guidance import make_dx_func, make_ns_residual  # noqa: E402
from src.rewards import make_spectrum_fn          # noqa: E402
from src.sequence_inference import (              # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence, sparse_nnfill_degrade)

MEAN, STD, N, HIK0 = 0.0, 4.7988, 256, 32


def _c(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def _topdec(field, weight):
    """mean of `field` over the top-10% pixels of `weight`, per frame, averaged."""
    out = []
    for i in range(len(field)):
        thr = np.percentile(weight[i], 90)
        out.append(field[i][weight[i] >= thr].mean())
    return float(np.mean(out))


S_MULTI = [150, 100, 50]


def main(ckpt, lam=30.0, seqs="32,36", frames=4, t_start=100, re=1000, gt=None, grid_factor=4, seed=1,
         out=None, k3=False):
    out = out or f"monitoring/ab_pdelocal/guided_effect_re{re}{'_k3' if k3 else ''}.png"
    seqs = [int(x) for x in str(seqs).split(",")]
    ddpm, _, _ = build_base_ddpm()
    params = pickle.load(open(ckpt, "rb"))["params"]
    ab = ddpm.alpha_bar
    dx_func = make_dx_func(n=N, re=float(re), std=STD, mean=MEAN)
    resid = jax.jit(make_ns_residual(n=N, re=float(re)))
    spec_fn = make_spectrum_fn(N)
    sa, s1 = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))

    xin, xgt = [], []
    for s in seqs:
        seq = load_sequence(gt or "flow-data/kf_2d_re1000_256_40seed.npy", s)
        deg = grid_downsample_degrade(seq, grid_factor) if grid_factor else sparse_nnfill_degrade(seq, s)
        a, g = build_triplets(deg, MEAN, STD), build_triplets(seq, MEAN, STD)
        idx = np.linspace(len(g) // 4, len(g) - 1, frames).astype(int)
        xin.append(a[idx]); xgt.append(g[idx])
    xin, xgt = jnp.asarray(np.concatenate(xin)), np.concatenate(xgt)

    levels = sorted({t_start, *S_MULTI}) if k3 else [t_start]
    sU = {t: make_guided_sampler(ddpm.unet, ab, ddpm.beta_schedule, t, dx_func, 0.0) for t in levels}
    sG = {t: make_guided_sampler(ddpm.unet, ab, ddpm.beta_schedule, t, dx_func, lam) for t in levels}
    key = jax.random.PRNGKey(seed)

    def noise_to(x, t, kk):
        return float(jnp.sqrt(ab[t])) * x + float(jnp.sqrt(1.0 - ab[t])) * jax.random.normal(kk, x.shape)

    def recon(sdict):
        """K=1 single chain, or K=3 renoise/denoise. FIXED per-pass keys (fold_in) so unguided and
        guided share identical noise — the only difference is the guidance."""
        if k3:
            xc = xin
            for j, Sj in enumerate(S_MULTI):
                xc = jnp.asarray(sdict[Sj](params, noise_to(xc, Sj, jax.random.fold_in(key, 10 + j)),
                                           jax.random.fold_in(key, 20 + j)))
            return np.asarray(xc)
        xn = noise_to(xin, t_start, jax.random.fold_in(key, 10))
        return np.asarray(sdict[t_start](params, xn, jax.random.fold_in(key, 20)))
    U, G = recon(sU), recon(sG)

    wg, wu, wgd = xgt[..., 1] * STD, U[..., 1] * STD, G[..., 1] * STD
    Ru = np.abs(np.asarray(resid(jnp.asarray(U) * STD)))
    Rgd = np.abs(np.asarray(resid(jnp.asarray(G) * STD)))
    Rg = np.abs(np.asarray(resid(jnp.asarray(xgt) * STD)))
    dR = Rgd - Ru                                                  # residual change (guided - unguided)
    Ehg = local_hik_energy(wg, HIK0, 6.0); Eu = local_hik_energy(wu, HIK0, 6.0); Eg = local_hik_energy(wgd, HIK0, 6.0)
    absw_u = np.abs(wu)                                            # vorticity magnitude (energy structures)
    Egt = np.asarray(spec_fn(jnp.asarray(xgt)))

    hik = lambda X: float((np.asarray(spec_fn(X))[:, HIK0:].sum(-1) / Egt[:, HIK0:].sum(-1)).mean())
    print(f"\nx0-guidance effect (re={re}, lam={lam}, {len(xgt)} frames):")
    print(f"  mean|resid|      unguided {Ru.mean():.2f}   guided {Rgd.mean():.2f}   GT {Rg.mean():.2f}")
    print(f"  resid @ top-10% |w|   unguided {_topdec(Ru, absw_u):.2f}   guided {_topdec(Rgd, absw_u):.2f}"
          f"   (TRADE-OFF CHECK: did residual rise at the vortices?)")
    print(f"  resid @ top-10% E_hik unguided {_topdec(Ru, Eu):.2f}   guided {_topdec(Rgd, Eu):.2f}"
          f"   (did residual rise at the fine structures?)")
    print(f"  corr(Δresid, |w|)   = {_c(dR, absw_u):+.3f}   corr(Δresid, E_hik) = {_c(dR, Eu):+.3f}"
          f"   (>0 => residual INCREASED where energy is high)")
    print(f"  PDE placement corr(|resid|,|resid_GT|)  unguided {_c(Ru, Rg):+.3f}   guided {_c(Rgd, Rg):+.3f}")
    print(f"  hik_ret  unguided {hik(U):.3f}   guided {hik(G):.3f}   |   MSE  unguided {((U-xgt)**2).mean():.4f}"
          f"   guided {((G-xgt)**2).mean():.4f}", flush=True)

    fig, axes = plt.subplots(frames, 7, figsize=(19.5, 2.8 * frames))
    axes = np.atleast_2d(axes)
    for r in range(frames):
        vv = np.percentile(np.abs(wg[r]), 99)
        vR = np.percentile(np.concatenate([Ru[r], Rgd[r]]).ravel(), 99)
        vD = np.percentile(np.abs(dR[r]), 99); vH = np.percentile(Ehg[r], 99)
        cells = [("GT ω", wg[r], "RdBu_r", -vv, vv), ("unguided", wu[r], "RdBu_r", -vv, vv),
                 (f"guided λ={lam:.0f}", wgd[r], "RdBu_r", -vv, vv),
                 ("|resid| unguided", Ru[r], "inferno", 0, vR), ("|resid| guided", Rgd[r], "inferno", 0, vR),
                 ("Δ|resid| (blue=down)", dR[r], "RdBu_r", -vD, vD),
                 ("E_hik(GT)  energy loc", Ehg[r], "viridis", 0, vH)]
        for c, (nm, d, cm, vmn, vmx) in enumerate(cells):
            a = axes[r, c]; a.imshow(d, cmap=cm, vmin=vmn, vmax=vmx)
            a.set_xticks([]); a.set_yticks([]); a.set_title(nm, fontsize=8)
        axes[r, 0].set_ylabel(f"frame {r}", fontsize=9)
    fig.suptitle(f"Re={re} grid-{grid_factor}× {'K=3 ' if k3 else 'K=1 '}— x0-guidance (λ={lam:.0f}) vs unguided: residual, energy, "
                 f"trade-off check", y=1.005, fontsize=11)
    plt.tight_layout(); plt.savefig(out, dpi=118, bbox_inches="tight")
    print(f"saved {out}", flush=True)

    # spectrum
    fig2, ax = plt.subplots(figsize=(8.5, 5))
    kx = np.arange(1, 96)
    for nm, X, c, ls in (("GT", xgt, "k", "-"), ("unguided", U, "#9498a0", "--"), (f"guided λ={lam:.0f}", G, "#1f6fd0", "-")):
        ax.plot(kx, np.asarray(spec_fn(jnp.asarray(X))).mean(0)[1:96], color=c, lw=2 if nm == "GT" else 1.7, ls=ls, label=nm)
    ax.axvspan(HIK0, 95, color="#3ca951", alpha=0.06); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("E(k)"); ax.legend(fontsize=9, frameon=False)
    ax.set_title(f"Re={re} {'K=3' if k3 else 'K=1'} enstrophy spectrum — guided vs unguided vs GT", fontsize=11)
    plt.tight_layout(); plt.savefig(out.replace(".png", "_spec.png"), dpi=118, bbox_inches="tight")
    print(f"saved {out.replace('.png', '_spec.png')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--seqs", type=str, default="32,36")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--t_start", type=int, default=100)
    ap.add_argument("--re", type=int, default=1000)
    ap.add_argument("--gt", type=str, default=None)
    ap.add_argument("--grid_factor", type=int, default=4)
    ap.add_argument("--k3", action="store_true", help="use K=3 multi-phase recon instead of single chain")
    ap.add_argument("--out", type=str, default=None)
    main(**vars(ap.parse_args()))
