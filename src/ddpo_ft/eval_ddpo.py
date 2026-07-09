"""Comprehensive GT-paired evaluation of a DDPO checkpoint vs the frozen base, on the real Re=1000
validation+test sequences (32-39). Reports the full picture — hi-k retention, MSE, PDE residual,
total enstrophy (energy), effective resolution k* — plus a spectrum overlay figure (GT / base /
DDPO) and per-scale retention. Answers 'does it actually work': is the high-k deficit filled with
REAL energy (retention up, spectrum matches GT, MSE/residual not worse)?

    python -m src.ddpo_ft.eval_ddpo <ckpt.pkl> [--t_start 100] [--n_per_seq 12] [--n_samples 2]

Lean sampler (carries only x, no trajectory) so we can batch large without the DDPO memory cost.
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

from ppo_claude import policy_mean_std            # noqa: E402
from train_claude import build_base_ddpm          # noqa: E402
from src.rewards import make_residual_loss, make_spectrum_fn  # noqa: E402
from src.sequence_inference import build_triplets, load_sequence, sparse_nnfill_degrade  # noqa: E402

MEAN, STD, N, HIK0 = 0.0, 4.7988, 256, 32
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"
VAL, TEST = [32, 33, 34, 35], [36, 37, 38, 39]
FIGDIR = "monitoring/ddpo_ckpts"


def make_sampler(unet, alpha_bar, beta_schedule, t_start, temp=1.0):
    """Lean deterministic-chain sampler: returns x0 only (carries x, no trajectory stacking)."""
    ts = jnp.arange(t_start, 0, -1, dtype=jnp.int32)

    def sample(params, x_start, key):
        def step(carry, t):
            x, k = carry
            k, sk = jax.random.split(k)
            mean, std = policy_mean_std(unet, params, x, t, alpha_bar, beta_schedule, None, 0.0, temp)
            z = jnp.where(t > 1, jax.random.normal(sk, x.shape), jnp.zeros_like(x))
            return (mean + std * z, k), None
        (x0, _), _ = jax.lax.scan(step, (x_start, key), ts)
        return x0
    return jax.jit(sample)


def radial_spectrum(x, spec_fn):
    return np.asarray(spec_fn(x))                 # (B, 128) enstrophy spectrum of middle frame


def eff_resolution(E_recon, E_gt):
    """Smallest k where recon spectrum drops below half of GT — 'reconstructs structure down to k*'."""
    r = E_recon[1:96] / (E_gt[1:96] + 1e-20)
    below = np.where(r < 0.5)[0]
    return int(below[0] + 1) if len(below) else 95


def evaluate(ckpt_path, t_start=100, n_per_seq=12, n_samples=2, batch=24, seed=0,
             re=1000, gt_path=None, val=None, test=None):
    gt_path = gt_path or GT_PATH
    val = val if val is not None else VAL
    test = test if test is not None else TEST
    ddpm, base_params, _ = build_base_ddpm()
    ddpo_params = pickle.load(open(ckpt_path, "rb"))["params"]
    print(f"eval {ckpt_path} | re={re} gt={gt_path} | t_start={t_start} val{val}+test{test} "
          f"n/seq={n_per_seq} samples={n_samples}", flush=True)

    sampler = make_sampler(ddpm.unet, ddpm.alpha_bar, ddpm.beta_schedule, t_start, temp=1.0)
    spec_fn = make_spectrum_fn(N)
    resid_fn = make_residual_loss(n=N, re=float(re), std=STD, mean=MEAN)
    ab = ddpm.alpha_bar
    sqrt_ab, sqrt_1m = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))
    key = jax.random.PRNGKey(seed)

    def recon_batch(params, x_cond):
        outs = []
        for i in range(0, len(x_cond), batch):
            xc = x_cond[i:i + batch]
            nonlocal key
            key, k1, k2 = jax.random.split(key, 3)
            x_start = sqrt_ab * xc + sqrt_1m * jax.random.normal(k1, xc.shape)
            outs.append(np.asarray(sampler(params, x_start, k2)))
        return np.concatenate(outs)

    def collect(seqs):
        xin, xgt = [], []
        for s in seqs:
            seq = load_sequence(gt_path, s)
            inp = build_triplets(sparse_nnfill_degrade(seq, s), MEAN, STD)
            gt = build_triplets(seq, MEAN, STD)
            idx = np.linspace(0, len(inp) - 1, n_per_seq).astype(int)
            xin.append(inp[idx]); xgt.append(gt[idx])
        xin = np.repeat(np.concatenate(xin), n_samples, axis=0)
        xgt = np.repeat(np.concatenate(xgt), n_samples, axis=0)
        return jnp.asarray(xin), jnp.asarray(xgt)

    results = {}
    spectra = {}
    for split, seqs in ((f"val{val}", val), (f"test{test}", test)):
        xin, xgt = collect(seqs)
        E_gt = radial_spectrum(xgt, spec_fn)
        row = {"gt_resid": float(np.asarray(resid_fn(xgt)).mean()), "gt_enst": float((np.asarray(xgt)[..., 1] ** 2).mean())}
        for name, params in (("base", base_params), ("ddpo", ddpo_params)):
            x0 = recon_batch(params, xin)
            E_r = radial_spectrum(x0, spec_fn)
            hik = E_r[:, HIK0:].sum(-1) / E_gt[:, HIK0:].sum(-1)
            mse = ((x0 - np.asarray(xgt)) ** 2).reshape(len(x0), -1).mean(-1)
            resid = np.asarray(resid_fn(jnp.asarray(x0)))
            enst = np.asarray(x0)[..., 1] ** 2
            row[name] = {"hik": hik.mean(), "hik_sd": hik.std(), "mse": mse.mean(),
                         "resid": resid.mean(), "enst_ret": enst.mean() / row["gt_enst"],
                         "kstar": eff_resolution(E_r.mean(0), E_gt.mean(0))}
            spectra.setdefault(split, {})[name] = E_r.mean(0)
        spectra[split]["gt"] = E_gt.mean(0)
        results[split] = row

    # ---- table ----
    print(f"\n{'split / model':<18}{'hik_ret':>12}{'MSE':>10}{'residual':>11}{'enst_ret':>10}{'k*':>6}")
    for split in results:
        r = results[split]
        print(f"{split:<18}{'':>12}{'':>10}{r['gt_resid']:>9.1f}gt{'':>10}{'':>6}")
        for name in ("base", "ddpo"):
            m = r[name]
            print(f"  {name:<16}{m['hik']:>8.3f}±{m['hik_sd']:>3.2f}{m['mse']:>10.4f}{m['resid']:>11.1f}"
                  f"{m['enst_ret']:>10.3f}{m['kstar']:>6}")
        b, d = r["base"], r["ddpo"]
        print(f"  {'Δ(ddpo-base)':<16}{d['hik'] - b['hik']:>+11.3f}{d['mse'] - b['mse']:>+10.4f}"
              f"{d['resid'] - b['resid']:>+11.1f}{d['enst_ret'] - b['enst_ret']:>+10.3f}{d['kstar'] - b['kstar']:>+6}")

    # ---- figure: spectrum overlay + per-scale retention ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    kx = np.arange(1, 96)
    for j, split in enumerate(spectra):
        sp = spectra[split]
        a = axes[0, j]
        a.plot(kx, sp["gt"][1:96], "k-", lw=2.2, label="GT")
        a.plot(kx, sp["base"][1:96], color="#9498a0", lw=1.5, ls="--", label="base recon")
        a.plot(kx, sp["ddpo"][1:96], color="#3ca951", lw=1.7, label="DDPO recon")
        a.axvspan(HIK0, 95, color="#3ca951", alpha=0.06); a.axvline(HIK0, color="gray", lw=0.6, ls=":")
        a.set_xscale("log"); a.set_yscale("log"); a.set_title(f"enstrophy spectrum — {split}", fontsize=10)
        a.set_xlabel("wavenumber k"); a.set_ylabel("E(k)"); a.legend(fontsize=8, frameon=False)
        b = axes[1, j]
        b.plot(kx, sp["base"][1:96] / (sp["gt"][1:96] + 1e-20), color="#9498a0", lw=1.5, ls="--", label="base/GT")
        b.plot(kx, sp["ddpo"][1:96] / (sp["gt"][1:96] + 1e-20), color="#3ca951", lw=1.7, label="DDPO/GT")
        b.axhline(1.0, color="k", lw=0.8); b.axvspan(HIK0, 95, color="#3ca951", alpha=0.06)
        b.axvline(HIK0, color="gray", lw=0.6, ls=":"); b.set_xscale("log"); b.set_ylim(0, 1.6)
        b.set_title(f"per-scale retention E_recon/E_GT — {split}", fontsize=10)
        b.set_xlabel("wavenumber k"); b.set_ylabel("retention"); b.legend(fontsize=8, frameon=False)
    fig.suptitle(f"DDPO {os.path.basename(ckpt_path)} vs base — Re={re} val+test (shaded = hi-k reward band)", y=1.01)
    plt.tight_layout()
    figpath = os.path.join(FIGDIR, f"eval_spectrum_re{re}.png")
    plt.savefig(figpath, dpi=110, bbox_inches="tight")
    print(f"\nsaved {figpath}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--t_start", type=int, default=100)
    ap.add_argument("--n_per_seq", type=int, default=12)
    ap.add_argument("--n_samples", type=int, default=2)
    ap.add_argument("--re", type=int, default=1000)
    ap.add_argument("--gt", type=str, default=None, help="GT .npy (defaults to Re=1000 path)")
    ap.add_argument("--val", type=str, default=None, help="comma-list of val seq ids")
    ap.add_argument("--test", type=str, default=None, help="comma-list of test seq ids")
    a = ap.parse_args()
    _lst = lambda s: [int(x) for x in s.split(",")] if s else None
    evaluate(a.ckpt, t_start=a.t_start, n_per_seq=a.n_per_seq, n_samples=a.n_samples,
             re=a.re, gt_path=a.gt, val=_lst(a.val), test=_lst(a.test))
