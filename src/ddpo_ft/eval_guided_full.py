"""Full val+test evaluation of the 2x2 matrix { base | finetuned } x { unguided | x0-guided lam } at
grid-4x, per regime. Reports the metrics that matter: hi-k retention, PDE residual (mean|R|), MSE,
energy placement E_hik<->GT, effective resolution k*. Shows whether adding physics guidance moves
the residual / retention for base vs the DDPO model, in-dist and OOD.

    python -m src.ddpo_ft.eval_guided_full <ddpo_ckpt.pkl> --re 1000 --gt <path>
        --val 32,33,34,35 --test 36,37,38,39 --grid_factor 4 --lam 3 --n_per_seq 8
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
from eval_ddpo import eff_resolution              # noqa: E402
from train_claude import build_base_ddpm          # noqa: E402
from viz_energy import local_hik_energy           # noqa: E402
from src.physics_guidance import make_dx_func, make_ns_residual  # noqa: E402
from src.rewards import make_spectrum_fn          # noqa: E402
from src.sequence_inference import (              # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence, sparse_nnfill_degrade)

MEAN, STD, N, HIK0 = 0.0, 4.7988, 256, 32


def main(ddpo_ckpt, re=1000, gt=None, val="32,33,34,35", test="36,37,38,39", grid_factor=4,
         lam=3.0, n_per_seq=8, batch=16, t_start=100, seed=0):
    gt = gt or "flow-data/kf_2d_re1000_256_40seed.npy"
    seqs = [int(x) for x in (val + "," + test).split(",")]
    ddpm, base_params, _ = build_base_ddpm()
    ddpo = pickle.load(open(ddpo_ckpt, "rb"))["params"]
    ab = ddpm.alpha_bar
    dx = make_dx_func(n=N, re=float(re), std=STD, mean=MEAN)
    resid = jax.jit(make_ns_residual(n=N, re=float(re)))
    spec_fn = make_spectrum_fn(N)
    sa, s1 = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))
    sU = make_guided_sampler(ddpm.unet, ab, ddpm.beta_schedule, t_start, dx, 0.0)
    sG = make_guided_sampler(ddpm.unet, ab, ddpm.beta_schedule, t_start, dx, lam)

    xin, xgt = [], []
    for s in seqs:
        seq = load_sequence(gt, s)
        deg = grid_downsample_degrade(seq, grid_factor) if grid_factor else sparse_nnfill_degrade(seq, s)
        a, g = build_triplets(deg, MEAN, STD), build_triplets(seq, MEAN, STD)
        idx = np.linspace(0, len(g) - 1, n_per_seq).astype(int)
        xin.append(a[idx]); xgt.append(g[idx])
    xin, xgt = np.concatenate(xin), np.concatenate(xgt)
    E_gt = np.asarray(spec_fn(jnp.asarray(xgt))); Ehg = local_hik_energy(xgt[..., 1] * STD, HIK0, 6.0)
    Rg = np.abs(np.asarray(resid(jnp.asarray(xgt) * STD)))
    key = jax.random.PRNGKey(seed)
    print(f"\n=== Re={re} grid-{grid_factor}x  {len(xgt)} frames (seqs {seqs}, {n_per_seq}/seq)  "
          f"guidance lam={lam}  |  GT: residual {Rg.mean():.2f} ===", flush=True)
    print(f"{'model':<10}{'guide':>7}{'hik_ret':>9}{'residual':>10}{'MSE':>9}{'placement':>11}{'k*':>5}", flush=True)

    def recon(smp, params):
        outs = []
        nonlocal key
        for i in range(0, len(xin), batch):
            xc = jnp.asarray(xin[i:i + batch])
            key, k1, k2 = jax.random.split(key, 3)
            outs.append(np.asarray(smp(params, sa * xc + s1 * jax.random.normal(k1, xc.shape), k2)))
        return np.concatenate(outs)

    for mname, params in (("base", base_params), ("DDPO", ddpo)):
        for gname, smp in (("off", sU), (f"lam{lam:.0f}", sG)):
            x0 = recon(smp, params)
            E_r = np.asarray(spec_fn(x0)); Eh = local_hik_energy(x0[..., 1] * STD, HIK0, 6.0)
            hik = float((E_r[:, HIK0:].sum(-1) / E_gt[:, HIK0:].sum(-1)).mean())
            R = float(np.abs(np.asarray(resid(jnp.asarray(x0) * STD))).mean())
            mse = float(((x0 - xgt) ** 2).mean())
            pl = float(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1])
            ks = eff_resolution(E_r.mean(0), E_gt.mean(0))
            print(f"{mname:<10}{gname:>7}{hik:>9.3f}{R:>10.2f}{mse:>9.4f}{pl:>11.3f}{ks:>5}", flush=True)
    print(f"{'GT':<10}{'-':>7}{1.0:>9.3f}{Rg.mean():>10.2f}{0.0:>9.4f}{1.0:>11.3f}{'-':>5}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ddpo_ckpt")
    ap.add_argument("--re", type=int, default=1000)
    ap.add_argument("--gt", type=str, default=None)
    ap.add_argument("--val", type=str, default="32,33,34,35")
    ap.add_argument("--test", type=str, default="36,37,38,39")
    ap.add_argument("--grid_factor", type=int, default=4)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--n_per_seq", type=int, default=8)
    main(**vars(ap.parse_args()))
