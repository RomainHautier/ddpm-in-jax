"""Physics-guided sampling to attack the residual DIRECTLY (no retraining): at each reverse step,
subtract lambda * grad(mean residual^2) from the sample (BaratiLab linear guidance). This computes
the exact residual-descent direction — including the coordinated multi-frame change that fixes the
k<8 temporal-balance error — which policy-gradient sampling could never find by chance.

Sweeps lambda on a finetuned model at grid-4×; reports residual (mean|R| + k<8 band power), MSE,
hi-k retention, energy placement. Finds whether guidance lowers the residual toward GT without
wrecking the reconstruction.

    python -m src.ddpo_ft.diag_guided_residual <ckpt.pkl> [--seqs 32,36] [--frames 6] [--re 1000]
                                                [--gt <path>] [--grid_factor 4] [--lams 0,10,30,100,200]
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

from ppo_claude import policy_mean_std            # noqa: E402
from train_claude import build_base_ddpm          # noqa: E402
from viz_energy import local_hik_energy           # noqa: E402
from src.physics_guidance import make_dx_func, make_ns_residual  # noqa: E402
from src.rewards import make_spectrum_fn          # noqa: E402
from src.sequence_inference import (              # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence, sparse_nnfill_degrade)

MEAN, STD, N, HIK0 = 0.0, 4.7988, 256, 32
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"
_k = np.fft.fftfreq(N) * N
_LOWK = jnp.asarray((np.sqrt(_k[:, None] ** 2 + _k[None, :] ** 2) < 8).astype(np.float32))


def make_guided_sampler(unet, alpha_bar, beta_schedule, t_start, dx_func, lam, temp=1.0,
                        guidance_t_max=50, clip=0.15):
    """Reverse chain with linear physics guidance, applied ONLY for t <= guidance_t_max (the low-noise
    late steps where the residual gradient is meaningful; guiding the early noisy steps diverges).
    Per-step guidance is clipped to +-clip to prevent runaway. lam=0 -> plain sampler."""
    ts = jnp.arange(t_start, 0, -1, dtype=jnp.int32)

    def sample(params, x_start, key):
        def step(carry, t):
            x, k = carry
            dx = jnp.where(t <= guidance_t_max, jnp.clip(lam * dx_func(x), -clip, clip), 0.0)
            k, sk = jax.random.split(k)
            m, s = policy_mean_std(unet, params, x, t, alpha_bar, beta_schedule, None, 0.0, temp)
            z = jnp.where(t > 1, jax.random.normal(sk, x.shape), jnp.zeros_like(x))
            return (m + s * z - (dx if lam > 0 else 0.0), k), None
        (x0, _), _ = jax.lax.scan(step, (x_start, key), ts)
        return x0
    return jax.jit(sample)


def main(ckpt, seqs="32,36", frames=6, t_start=100, re=1000, gt=None, grid_factor=4, seed=1,
         lams="0,10,30,100,200"):
    lams = [float(x) for x in lams.split(",")]
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
        seq = load_sequence(gt or GT_PATH, s)
        deg = grid_downsample_degrade(seq, grid_factor) if grid_factor else sparse_nnfill_degrade(seq, s)
        a, g = build_triplets(deg, MEAN, STD), build_triplets(seq, MEAN, STD)
        idx = np.linspace(len(g) // 4, len(g) - 1, frames).astype(int)
        xin.append(a[idx]); xgt.append(g[idx])
    xin, xgt = jnp.asarray(np.concatenate(xin)), np.concatenate(xgt)
    Eg = np.asarray(spec_fn(jnp.asarray(xgt))); Ehg = local_hik_energy(xgt[..., 1] * STD, HIK0 - 0, 6.0)
    Rgs = np.asarray(resid(jnp.asarray(xgt) * STD)); Rg = np.abs(Rgs)   # signed for spectrum, abs for mean
    lowk_gt = float((np.abs(np.fft.fft2(Rgs)) ** 2 * np.asarray(_LOWK)[None]).sum((1, 2)).mean() / float(_LOWK.sum()))
    print(f"\nguided-sampling sweep (re={re}, {len(xgt)} frames)  |  GT: mean|R|={Rg.mean():.2f}  k<8={lowk_gt:.2e}")
    print(f"{'lambda':>7}{'mean|R|':>9}{'k<8 power':>11}{'MSE':>9}{'hik_ret':>9}{'placement':>11}")
    for lam in lams:
        smp = make_guided_sampler(ddpm.unet, ab, ddpm.beta_schedule, t_start, dx_func, lam)
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        x0 = np.asarray(smp(params, sa * xin + s1 * jax.random.normal(k1, xin.shape), k2))
        Rs = np.asarray(resid(jnp.asarray(x0) * STD)); R = np.abs(Rs)
        lowk = float((np.abs(np.fft.fft2(Rs)) ** 2 * np.asarray(_LOWK)[None]).sum((1, 2)).mean() / float(_LOWK.sum()))
        Er = np.asarray(spec_fn(x0)); Eh = local_hik_energy(x0[..., 1] * STD, HIK0 - 0, 6.0)
        hik = float((Er[:, HIK0:].sum(-1) / Eg[:, HIK0:].sum(-1)).mean())
        pl = float(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1])
        mse = float(((x0 - xgt) ** 2).mean())
        print(f"{lam:>7.0f}{R.mean():>9.2f}{lowk:>11.2e}{mse:>9.4f}{hik:>9.3f}{pl:>11.3f}", flush=True)
    print(f"{'GT':>7}{Rg.mean():>9.2f}{lowk_gt:>11.2e}{0.0:>9.4f}{1.0:>9.3f}{1.0:>11.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seqs", type=str, default="32,36")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--t_start", type=int, default=100)
    ap.add_argument("--re", type=int, default=1000)
    ap.add_argument("--gt", type=str, default=None)
    ap.add_argument("--grid_factor", type=int, default=4)
    ap.add_argument("--lams", type=str, default="0,10,30,100,200")
    main(**vars(ap.parse_args()))
