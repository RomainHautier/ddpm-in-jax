"""The missing cross: {K=1, K=3} x {guidance off, x0-guidance lam} for the DDPO model at grid-4x.
Does linear physics guidance still reduce the residual under K=3 multi-phase, and does K=3's amplitude
overshoot interact with it? Reports retention / residual / MSE / placement for all four.

    python -m src.ddpo_ft.diag_k3_guided <ddpo_ckpt.pkl> --re 1000 [--lam 3] [--seqs 32,36] [--frames 6]
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

from diag_guided_residual import make_guided_sampler  # noqa: E402
from train_claude import build_base_ddpm          # noqa: E402
from viz_energy import local_hik_energy           # noqa: E402
from src.physics_guidance import make_dx_func, make_ns_residual  # noqa: E402
from src.rewards import make_spectrum_fn          # noqa: E402
from src.sequence_inference import (              # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence, sparse_nnfill_degrade)

MEAN, STD, N, HIK0 = 0.0, 4.7988, 256, 32
S_MULTI, T_SINGLE = [150, 100, 50], 100


def main(ckpt, re=1000, gt=None, lam=3.0, seqs="32,36", frames=6, grid_factor=4, seed=1):
    seqs = [int(x) for x in str(seqs).split(",")]
    ddpm, _, _ = build_base_ddpm()
    P = pickle.load(open(ckpt, "rb"))["params"]
    ab = ddpm.alpha_bar
    dx = make_dx_func(n=N, re=float(re), std=STD, mean=MEAN)
    resid = jax.jit(make_ns_residual(n=N, re=float(re)))
    spec_fn = make_spectrum_fn(N)
    # guided + unguided single-chain samplers at every level we touch
    sU = {t: make_guided_sampler(ddpm.unet, ab, ddpm.beta_schedule, t, dx, 0.0) for t in sorted({T_SINGLE, *S_MULTI})}
    sG = {t: make_guided_sampler(ddpm.unet, ab, ddpm.beta_schedule, t, dx, lam) for t in sorted({T_SINGLE, *S_MULTI})}

    xin, xgt = [], []
    for s in seqs:
        seq = load_sequence(gt or "flow-data/kf_2d_re1000_256_40seed.npy", s)
        deg = grid_downsample_degrade(seq, grid_factor) if grid_factor else sparse_nnfill_degrade(seq, s)
        a, g = build_triplets(deg, MEAN, STD), build_triplets(seq, MEAN, STD)
        idx = np.linspace(len(g) // 4, len(g) - 1, frames).astype(int)
        xin.append(a[idx]); xgt.append(g[idx])
    xin, xgt = jnp.asarray(np.concatenate(xin)), np.concatenate(xgt)
    E_gt = np.asarray(spec_fn(jnp.asarray(xgt))); Ehg = local_hik_energy(xgt[..., 1] * STD, HIK0, 6.0)
    Rg = np.abs(np.asarray(resid(jnp.asarray(xgt) * STD)))
    key = jax.random.PRNGKey(seed)

    def noise(x, t, k):
        return float(jnp.sqrt(ab[t])) * x + float(jnp.sqrt(1.0 - ab[t])) * jax.random.normal(k, x.shape)

    def k1(S):
        nonlocal key; key, a, b = jax.random.split(key, 3)
        return np.asarray(S[T_SINGLE](P, noise(xin, T_SINGLE, a), b))

    def k3(S):
        nonlocal key; xc = xin
        for Sj in S_MULTI:
            key, a, b = jax.random.split(key, 3)
            xc = jnp.asarray(S[Sj](P, noise(xc, Sj, a), b))
        return np.asarray(xc)

    print(f"\n=== Re={re} grid-{grid_factor}x — DDPO x {{K=1,K=3}} x {{guide off, lam={lam}}}  "
          f"({len(xgt)} frames)  GT residual {Rg.mean():.2f} ===")
    print(f"{'variant':<14}{'hik_ret':>9}{'residual':>10}{'MSE':>9}{'placement':>11}")
    for nm, x0 in (("K=1 off", k1(sU)), (f"K=1 lam{lam:.0f}", k1(sG)),
                   ("K=3 off", k3(sU)), (f"K=3 lam{lam:.0f}", k3(sG))):
        E_r = np.asarray(spec_fn(x0)); Eh = local_hik_energy(x0[..., 1] * STD, HIK0, 6.0)
        hik = float((E_r[:, HIK0:].sum(-1) / E_gt[:, HIK0:].sum(-1)).mean())
        R = float(np.abs(np.asarray(resid(jnp.asarray(x0) * STD))).mean())
        mse = float(((x0 - xgt) ** 2).mean()); pl = float(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1])
        print(f"{nm:<14}{hik:>9.3f}{R:>10.2f}{mse:>9.4f}{pl:>11.3f}", flush=True)
    print(f"{'GT':<14}{1.0:>9.3f}{Rg.mean():>10.2f}{0.0:>9.4f}{1.0:>11.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--re", type=int, default=1000)
    ap.add_argument("--gt", type=str, default=None)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--seqs", type=str, default="32,36")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--grid_factor", type=int, default=4)
    main(**vars(ap.parse_args()))
