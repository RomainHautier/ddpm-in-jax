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


def make_guided_sampler(unet, alpha_bar, beta_schedule, t_start, dx_func, lam, temp=1.0):
    """Reverse chain with x0-PREDICTED physics guidance (DPS-style). At each step we form the model's
    clean estimate x0_hat = (x_t - sqrt(1-abar)*eps)/sqrt(abar) and take the residual gradient on THAT
    (physically meaningful at every noise level -> stable, applied throughout the chain, no clipping
    needed). lam=0 -> plain sampler."""
    ts = jnp.arange(t_start, 0, -1, dtype=jnp.int32)

    def sample(params, x_start, key):
        def step(carry, t):
            x, k = carry
            B = x.shape[0]
            eps = unet.apply({"params": params}, x, jnp.full((B,), t, jnp.int32), train=False)
            ab_t = alpha_bar[t]
            alpha_t = 1.0 - beta_schedule[t]
            x0_hat = (x - jnp.sqrt(1.0 - ab_t) * eps) / jnp.sqrt(ab_t)     # clean estimate
            dx = lam * dx_func(x0_hat) if lam > 0 else 0.0                 # residual grad on the CLEAN field
            mean = (1.0 / jnp.sqrt(alpha_t)) * (x - (1.0 - alpha_t) / jnp.sqrt(1.0 - ab_t) * eps)
            k, sk = jax.random.split(k)
            z = jnp.where(t > 1, jax.random.normal(sk, x.shape), jnp.zeros_like(x))
            return (mean + temp * jnp.sqrt(beta_schedule[t]) * z - dx, k), None
        (x0, _), _ = jax.lax.scan(step, (x_start, key), ts)
        return x0
    return jax.jit(sample)


def make_strided_guided_sampler(unet, alpha_bar, t_start, n_steps, dx_func, lam):
    """Accelerated reverse chain: eta=0 DDIM over a strided subsequence t_start -> ~0 in `n_steps`
    steps (same schedule construction as ppo_claude.build_ddim_denoiser), with x0-predicted guidance
    applied in x0 space at each visited step. Deterministic — `key` accepted for signature
    compatibility with make_guided_sampler but unused. NOTE: guidance fires n_steps times (vs t_start
    times in the stepwise sampler), so the same lam gives a proportionally weaker total nudge."""
    raw = [int(round(t_start - i * (t_start - 1) / max(n_steps - 1, 1))) for i in range(n_steps)]
    seq = sorted({t for t in raw if t >= 1}, reverse=True)
    t_cur = jnp.asarray(seq[:-1], dtype=jnp.int32)
    t_nxt = jnp.asarray(seq[1:], dtype=jnp.int32)
    t_last = int(seq[-1])

    def _x0g(params, x, t):
        eps = unet.apply({"params": params}, x, jnp.full((x.shape[0],), t, jnp.int32), train=False)
        x0 = (x - jnp.sqrt(1.0 - alpha_bar[t]) * eps) / jnp.sqrt(alpha_bar[t])
        if lam > 0:
            x0 = x0 - lam * dx_func(x0)
        return x0, eps

    def sample(params, x_start, key=None):
        def step(x, ts_pair):
            tc, tn = ts_pair
            x0, eps = _x0g(params, x, tc)
            return jnp.sqrt(alpha_bar[tn]) * x0 + jnp.sqrt(1.0 - alpha_bar[tn]) * eps, None
        x, _ = jax.lax.scan(step, x_start, (t_cur, t_nxt))
        x0, _ = _x0g(params, x, t_last)
        return x0
    return jax.jit(sample)


def make_spec_brake_grad(log_spec_ref, kband=(32, 96), n=256):
    """Gradient of the ONE-SIDED spectral hinge: penalizes log-spectrum energy ABOVE the reward
    anchor in `kband` only — a soft brake on tail overshoot during deep amplified travel. Uses the
    same anchor the reward trains against (no per-sample GT; OOD-safe via extrapolated anchors).
    PER-SAMPLE normalization (mean over shells, SUM over batch) so the gradient scale is independent
    of batch size — the original batch-mean form made mu effectively scale with 16/B and diverged at
    small batches (caught 2026-07-14). Validated at t=350: per-sample mu ~ 70-230 (plateau; == the
    old batch-16 mu 1100-3700; unstable above ~700 == old ~1e4) brings k[32,96) from 1.5-2.1x down
    to ~0.95 WITHOUT draining the mid-band fill (tail and mid-band amplification are separable).
    Apply on x0_hat: x0 -= mu * brake(x0)."""
    from src.rewards import make_spectrum_fn
    spec = make_spectrum_fn(n)
    lref = jnp.asarray(log_spec_ref)

    def dist(x):
        lE = jnp.log(spec(x)[:, kband[0]:kband[1]] + 1e-12)
        return (jnp.maximum(lE - lref[kband[0]:kband[1]], 0.0) ** 2).mean(axis=1).sum()
    return jax.grad(dist)


def make_kchain_ddim_sampler(unet, alpha_bar, chain_starts, n_steps, dx_func, lam,
                             eta=1.0, temp=1.0, return_stages=False, stride=None,
                             brake_func=None, mu=0.0):
    # INFERENCE TEMPERATURE (measured 2026-07-19): colder is MONOTONICALLY better on every metric.
    # On the temp2.5-trained OOD model: temp 1.00 -> ret 0.906, 0.70 -> 0.934, 0.30 -> 0.958 (k*87),
    # free (no retraining). Training wants wide noise (exploration); inference wants narrow noise —
    # fresh noise is white and later steps partially smooth it, so each renoise/denoise cycle costs
    # fine structure. Deploying a temp2.5-trained policy at temp 0.3 is a DELIBERATE train/test
    # mismatch: the learned means assume heavy noise will follow, and we do not supply it.
    # eta IS A DIFFERENT KNOB: it enters BOTH sigma AND the mean via sqrt(1-ab_n-sigma^2), so eta=0
    # is variance-PRESERVING and deterministic, whereas eta=1 with small temp is variance-DEFICIENT.
    # They are not the same limit: eta=0 -> ret 0.551 (k*59) vs temp 0.30 -> 0.958 (k*87).
    """Inference-side mirror of ppo_claude.build_ddim_rollout: stochastic-DDIM (Song eq.16) K-chain
    sampler — chain j from chain_starts[j] (descending), schedules from kchain_schedules (n_steps
    total budget split proportionally), deterministic x0-prediction per chain, renoise (forward q)
    between chains, reward x0 from the last chain. Optional x0-predicted guidance: subtract
    lam*dx(x0_hat) inside every step's mean and at each chain-final prediction. temp=1.0/eta=1.0 =
    the training probe's eval convention. sample(params, x_start, key) with x_start noised to
    chain_starts[0]."""
    from ppo_claude import kchain_schedules
    starts = [int(s) for s in chain_starts]
    scheds = kchain_schedules(starts, n_steps, stride)
    pairs = [(jnp.asarray(s[:-1], dtype=jnp.int32), jnp.asarray(s[1:], dtype=jnp.int32), int(s[-1]))
             for s in scheds]
    renoise_coef = [(float(jnp.sqrt(alpha_bar[S])), float(jnp.sqrt(1.0 - alpha_bar[S])))
                    for S in starts[1:]]

    def _x0hat(params, x, t):
        eps = unet.apply({"params": params}, x, jnp.full((x.shape[0],), t, jnp.int32), train=False)
        x0 = (x - jnp.sqrt(1.0 - alpha_bar[t]) * eps) / jnp.sqrt(alpha_bar[t])
        if lam > 0:
            x0 = x0 - lam * dx_func(x0)
        if brake_func is not None and mu > 0:
            x0 = x0 - mu * brake_func(x0)                     # spectral hinge-brake (see make_spec_brake_grad)
        return x0, eps

    def sample(params, x_start, key):
        def step(carry, ts):
            x, k = carry
            tc, tn = ts
            k, sk = jax.random.split(k)
            x0, eps = _x0hat(params, x, tc)
            ab_c, ab_n = alpha_bar[tc], alpha_bar[tn]
            sigma = eta * jnp.sqrt((1.0 - ab_n) / (1.0 - ab_c)) * jnp.sqrt(1.0 - ab_c / ab_n)
            mean = jnp.sqrt(ab_n) * x0 + jnp.sqrt(1.0 - ab_n - sigma ** 2) * eps
            return (mean + temp * sigma * jax.random.normal(sk, x.shape), k), None

        x = x_start
        stages = []
        for j, (tc, tn, tl) in enumerate(pairs):
            key, k_scan, k_re = jax.random.split(key, 3)
            (x_low, _), _ = jax.lax.scan(step, (x, k_scan), (tc, tn))
            x0, _ = _x0hat(params, x_low, tl)
            stages.append(x0)                                  # output of chain j+1 (before renoise)
            if j < len(pairs) - 1:
                sa, s1 = renoise_coef[j]
                x = sa * x0 + s1 * jax.random.normal(k_re, x0.shape)
        return tuple(stages) if return_stages else x0
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
