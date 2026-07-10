"""Did the K=3 multi-phase denoiser improve the SPATIAL correlations (placement of energy, locality
of PDE residual) — or only the scalar residual/MSE? Compares base-1chain / align-1chain / align-K3
on exactly the metrics from viz_energy + viz_pde:

  placement    corr(E_hik(recon), E_hik(GT))        the ~0.44 ceiling (energy in the right places?)
  E_hik<->err  corr(E_hik(recon), |err|)
  resid<->err  corr(|resid(recon)|, |err|)          does the residual localize the true error?
  resid_own    corr(|resid(GT)|, |resid(recon)|)    is the model residual its own artifact (low=good)?
  scalars      mean|resid|, RMS err, hik_ret

    python -m src.ddpo_ft.diag_multiphase_spatial <align_ckpt.pkl> [--seqs 32,36] [--frames 8]
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

from eval_ddpo import make_sampler                # noqa: E402
from train_claude import build_base_ddpm          # noqa: E402
from viz_energy import local_hik_energy           # noqa: E402
from src.physics_guidance import make_ns_residual  # noqa: E402
from src.rewards import make_spectrum_fn          # noqa: E402
from src.sequence_inference import build_triplets, load_sequence, sparse_nnfill_degrade  # noqa: E402

MEAN, STD, N, HIK0 = 0.0, 4.7988, 256, 32
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"
S_MULTI, T_SINGLE = [150, 100, 50], 100


def _corr(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def main(align_ckpt, seqs="32,36", frames=8, seed=1, kcut=32, sigma=6.0):
    seqs = [int(s) for s in seqs.split(",")]
    ddpm, base_params, _ = build_base_ddpm()
    align_params = pickle.load(open(align_ckpt, "rb"))["params"]
    ab = ddpm.alpha_bar
    spec_fn = make_spectrum_fn(N)
    resid = jax.jit(make_ns_residual(n=N, re=1000.0))
    samplers = {t: make_sampler(ddpm.unet, ab, ddpm.beta_schedule, t, temp=1.0)
                for t in sorted({T_SINGLE, *S_MULTI})}
    key = jax.random.PRNGKey(seed)

    def noise_to(x, t, k):
        return float(jnp.sqrt(ab[t])) * x + float(jnp.sqrt(1.0 - ab[t])) * jax.random.normal(k, x.shape)

    def one(params, x):
        nonlocal key; key, k1, k2 = jax.random.split(key, 3)
        return np.asarray(samplers[T_SINGLE](params, noise_to(x, T_SINGLE, k1), k2))

    def k3(params, x):
        nonlocal key; xc = x
        for Sj in S_MULTI:
            key, k1, k2 = jax.random.split(key, 3)
            xc = jnp.asarray(samplers[Sj](params, noise_to(xc, Sj, k1), k2))
        return np.asarray(xc)

    xin, xgt = [], []
    for s in seqs:
        seq = load_sequence(GT_PATH, s)
        inp = build_triplets(sparse_nnfill_degrade(seq, s), MEAN, STD)
        gt = build_triplets(seq, MEAN, STD)
        idx = np.linspace(len(inp) // 4, len(inp) - 1, frames).astype(int)
        xin.append(inp[idx]); xgt.append(gt[idx])
    xin, xgt = np.concatenate(xin), np.concatenate(xgt)
    xin_j = jnp.asarray(xin)
    print(f"\nseqs {seqs} x {frames} = {len(xin)} frames | align={os.path.basename(align_ckpt)}", flush=True)

    Ehik_gt = local_hik_energy(xgt[..., 1] * STD, kcut, sigma)
    R_gt = np.abs(np.asarray(resid(jnp.asarray(xgt) * STD)))
    E_gt = np.asarray(spec_fn(jnp.asarray(xgt)))

    recons = {"base_1chain": one(base_params, xin_j),
              "align_1chain": one(align_params, xin_j),
              "align_K3": k3(align_params, xin_j)}

    print(f"{'model':<15}{'placement':>11}{'E_hik<->err':>13}{'resid<->err':>13}{'resid_own':>11}"
          f"{'mean|res|':>11}{'RMS_err':>9}{'hik_ret':>9}")
    for name, x0 in recons.items():
        wr = x0[..., 1] * STD; wg = xgt[..., 1] * STD
        Ehik_r = local_hik_energy(wr, kcut, sigma)
        R_r = np.abs(np.asarray(resid(jnp.asarray(x0) * STD)))
        err = np.abs(wr - wg)
        E_r = np.asarray(spec_fn(x0))
        placement = _corr(Ehik_r, Ehik_gt)
        e_err = _corr(Ehik_r, err)
        r_err = _corr(R_r, err)
        r_own = _corr(R_gt, R_r)
        hik = float((E_r[:, HIK0:].sum(-1) / E_gt[:, HIK0:].sum(-1)).mean())
        print(f"{name:<15}{placement:>11.3f}{e_err:>13.3f}{r_err:>13.3f}{r_own:>11.3f}"
              f"{R_r.mean():>11.2f}{float(np.sqrt((err**2).mean())):>9.3f}{hik:>9.3f}", flush=True)
    print(f"{'GT (ref)':<15}{'1.000':>11}{'':>13}{'':>13}{'':>11}{R_gt.mean():>11.2f}{'0':>9}{'1.000':>9}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("align_ckpt")
    ap.add_argument("--seqs", type=str, default="32,36")
    ap.add_argument("--frames", type=int, default=8)
    main(**vars(ap.parse_args()))
