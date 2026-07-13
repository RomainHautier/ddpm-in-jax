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
S_MULTI = [150, 100, 50]


def main(ddpo_ckpt, re=1000, gt=None, val="32,33,34,35", test="36,37,38,39", grid_factor=4,
         lam=3.0, n_per_seq=8, batch=16, t_start=100, seed=0, k3=False,
         base_ddim_init=False, ddim_steps=20, ddim_t_start=100):
    gt = gt or "flow-data/kf_2d_re1000_256_40seed.npy"
    seqs = [int(x) for x in (val + "," + test).split(",")]
    ddpm, base_params, _ = build_base_ddpm()
    ddpo = pickle.load(open(ddpo_ckpt, "rb"))["params"]
    ab = ddpm.alpha_bar
    dx = make_dx_func(n=N, re=float(re), std=STD, mean=MEAN)
    resid = jax.jit(make_ns_residual(n=N, re=float(re)))
    spec_fn = make_spectrum_fn(N)
    sa, s1 = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))
    levels = sorted({t_start, *S_MULTI}) if k3 else [t_start]
    sU = {t: make_guided_sampler(ddpm.unet, ab, ddpm.beta_schedule, t, dx, 0.0) for t in levels}
    sG = {t: make_guided_sampler(ddpm.unet, ab, ddpm.beta_schedule, t, dx, lam) for t in levels}

    xin, xgt = [], []
    for s in seqs:
        seq = load_sequence(gt, s)
        deg = grid_downsample_degrade(seq, grid_factor) if grid_factor else sparse_nnfill_degrade(seq, s)
        a, g = build_triplets(deg, MEAN, STD), build_triplets(seq, MEAN, STD)
        idx = np.linspace(0, len(g) - 1, n_per_seq).astype(int)
        xin.append(a[idx]); xgt.append(g[idx])
    xin, xgt = np.concatenate(xin), np.concatenate(xgt)

    # base-DDIM init eval: transform the inputs with the SAME frozen-base deterministic DDIM pass used
    # at training time (train_claude --base_ddim_init), so ddiminit-trained models are evaluated on the
    # input distribution they were finetuned for. Applied to base AND DDPO rows (base = proper control).
    if base_ddim_init:
        from ppo_claude import build_ddim_denoiser
        _ddim = build_ddim_denoiser(ddpm.unet, ab, ddim_t_start, ddim_steps)
        _sa, _s1 = float(jnp.sqrt(ab[ddim_t_start])), float(jnp.sqrt(1.0 - ab[ddim_t_start]))
        _dkey = jax.random.PRNGKey(12345)
        outs = []
        for i in range(0, len(xin), batch):
            xb = jnp.asarray(xin[i:i + batch], dtype=jnp.float32)
            xs = _sa * xb + _s1 * jax.random.normal(jax.random.fold_in(_dkey, i), xb.shape)
            outs.append(np.asarray(_ddim(base_params, xs)))
        xin = np.concatenate(outs, axis=0)
        print(f"BASE-DDIM INIT eval: inputs replaced with frozen-base DDIM reconstruction "
              f"(SDEdit t={ddim_t_start}, {ddim_steps} steps, eta=0) -> {xin.shape}", flush=True)
    E_gt = np.asarray(spec_fn(jnp.asarray(xgt))); Ehg = local_hik_energy(xgt[..., 1] * STD, HIK0, 6.0)
    Rg = np.abs(np.asarray(resid(jnp.asarray(xgt) * STD)))
    key = jax.random.PRNGKey(seed)
    print(f"\n=== Re={re} grid-{grid_factor}x  K={3 if k3 else 1}{'  DDIM-INIT' if base_ddim_init else ''}  "
          f"{len(xgt)} frames (seqs {seqs}, {n_per_seq}/seq)  "
          f"guidance lam={lam}  |  GT: residual {Rg.mean():.2f} ===", flush=True)
    print(f"{'model':<10}{'guide':>7}{'hik_ret':>9}{'residual':>10}{'MSE':>9}{'placement':>11}{'k*':>5}", flush=True)

    def recon(sdict, params):
        """Returns {K: recon}. K=1: single SDEdit chain from t_start. k3 mode: renoise/denoise cascade
        over S_MULTI, capturing the state after chain 2 (= the K=2 result, same noise) AND chain 3 —
        so K=2 is exactly the intermediate of the K=3 cascade, not a separate run."""
        outs = {}
        nonlocal key
        for i in range(0, len(xin), batch):
            xc = jnp.asarray(xin[i:i + batch])
            if k3:
                for j, Sj in enumerate(S_MULTI):
                    saj, s1j = float(jnp.sqrt(ab[Sj])), float(jnp.sqrt(1.0 - ab[Sj]))
                    key, k1, k2 = jax.random.split(key, 3)
                    xc = sdict[Sj](params, saj * xc + s1j * jax.random.normal(k1, xc.shape), k2)
                    if j + 1 >= 2:                                  # end of chain 2 -> K=2, chain 3 -> K=3
                        outs.setdefault(j + 1, []).append(np.asarray(xc))
            else:
                key, k1, k2 = jax.random.split(key, 3)
                outs.setdefault(1, []).append(
                    np.asarray(sdict[t_start](params, sa * xc + s1 * jax.random.normal(k1, xc.shape), k2)))
        return {K: np.concatenate(v) for K, v in outs.items()}

    for mname, params in (("base", base_params), ("DDPO", ddpo)):
        for gname, smp in (("off", sU), (f"lam{lam:.0f}", sG)):
            for K, x0 in sorted(recon(smp, params).items()):
                E_r = np.asarray(spec_fn(x0)); Eh = local_hik_energy(x0[..., 1] * STD, HIK0, 6.0)
                hik = float((E_r[:, HIK0:].sum(-1) / E_gt[:, HIK0:].sum(-1)).mean())
                R = float(np.abs(np.asarray(resid(jnp.asarray(x0) * STD))).mean())
                mse = float(((x0 - xgt) ** 2).mean())
                pl = float(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1])
                ks = eff_resolution(E_r.mean(0), E_gt.mean(0))
                tag = f"{mname} K{K}" if k3 else mname
                print(f"{tag:<10}{gname:>7}{hik:>9.3f}{R:>10.2f}{mse:>9.4f}{pl:>11.3f}{ks:>5}", flush=True)
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
    ap.add_argument("--k3", action="store_true", help="K=3 multi-phase recon instead of single SDEdit chain")
    ap.add_argument("--base_ddim_init", action="store_true",
                    help="pre-transform inputs with the frozen-base DDIM reconstruction (matches training)")
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--ddim_t_start", type=int, default=100)
    main(**vars(ap.parse_args()))
