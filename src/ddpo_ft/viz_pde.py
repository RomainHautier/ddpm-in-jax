"""Is the PDE residual FIELD a GT-free proxy for WHERE the reconstruction is wrong?

Per frame, columns:
  1) |base − GT|           the actual error (needs GT) — what we'd want to localize
  2) |NS residual(base)|   the residual field (NO GT needed) — candidate proxy
  3) |NS residual(DDPO)|   residual after finetuning (did it drop where error was?)
  4) |NS residual(GT)|     GT's own residual (the discretization floor — even perfect has this)
Title per row: corr(|residual_base|, |base−GT|) — >0 means the residual lights up where the error is,
i.e. it could steer localized improvement without ground truth.

    python -m src.ddpo_ft.viz_pde <ckpt.pkl> [--seq 36] [--frames 3] [--t_start 100] [--re 1000]
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

from eval_ddpo import make_sampler                # noqa: E402
from train_claude import build_base_ddpm          # noqa: E402
from src.physics_guidance import make_ns_residual  # noqa: E402
from src.sequence_inference import (               # noqa: E402
    build_triplets, grid_downsample_degrade, load_sequence, sparse_nnfill_degrade)

MEAN, STD = 0.0, 4.7988
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"


def main(ckpt, seq=36, frames=3, t_start=100, re=1000, seed=1, out="monitoring/ddpo_ckpts/viz_pde.png",
         grid_factor=None, gt=None):
    ddpm, base_params, _ = build_base_ddpm()
    ddpo_params = pickle.load(open(ckpt, "rb"))["params"]
    sampler = make_sampler(ddpm.unet, ddpm.alpha_bar, ddpm.beta_schedule, t_start, temp=1.0)
    resid = jax.jit(make_ns_residual(n=256, re=float(re)))     # w_phys (...,256,256,3) -> (...,256,256)
    ab = ddpm.alpha_bar
    sqrt_ab, sqrt_1m = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))

    s = load_sequence(gt or GT_PATH, seq)
    degraded = grid_downsample_degrade(s, grid_factor) if grid_factor else sparse_nnfill_degrade(s, seq)
    inp = build_triplets(degraded, MEAN, STD)
    gt = build_triplets(s, MEAN, STD)
    idx = np.linspace(len(inp) // 4, len(inp) - 1, frames).astype(int)
    xin, xgt = jnp.asarray(inp[idx]), np.asarray(gt[idx])
    print(f"reconstructing {frames} frames of seq {seq} + residual fields (re={re}) ...", flush=True)

    key = jax.random.PRNGKey(seed)
    def recon(params):
        k1, k2 = jax.random.split(key)
        x_start = sqrt_ab * xin + sqrt_1m * jax.random.normal(k1, xin.shape)
        return np.asarray(sampler(params, x_start, k2))        # (B,256,256,3) full triplet
    b0, d0 = recon(base_params), recon(ddpo_params)
    Rf = lambda x0: np.abs(np.asarray(resid(jnp.asarray(x0) * STD)))   # residual FIELD magnitude
    R_base, R_ddpo, R_gt = Rf(b0), Rf(d0), Rf(np.asarray(xgt))
    err_base = np.abs((b0 - np.asarray(xgt))[..., 1] * STD)     # middle-frame error magnitude
    err_ddpo = np.abs((d0 - np.asarray(xgt))[..., 1] * STD)

    fig, axes = plt.subplots(frames, 4, figsize=(13.5, 3.4 * frames))
    axes = np.atleast_2d(axes)
    for r in range(frames):
        cR = float(np.corrcoef(R_base[r].ravel(), err_base[r].ravel())[0, 1])
        cGB = float(np.corrcoef(R_gt[r].ravel(), R_base[r].ravel())[0, 1])   # model resid vs GT's own floor
        cGD = float(np.corrcoef(R_gt[r].ravel(), R_ddpo[r].ravel())[0, 1])
        vE = np.percentile(err_base[r], 99)
        vR = np.percentile(np.concatenate([R_base[r], R_gt[r], R_ddpo[r]]).ravel(), 99)
        panels = [(f"|base − GT|  (the actual error)", err_base[r], vE),
                  (f"|NS residual(base)|  ρ(·,err)={cR:+.2f}  ρ(·,GTresid)={cGB:+.2f}", R_base[r], vR),
                  (f"|NS residual(DDPO)|  ρ(·,GTresid)={cGD:+.2f}", R_ddpo[r], vR),
                  (f"|NS residual(GT)|  (floor)", R_gt[r], vR)]
        for c, (name, data, vmx) in enumerate(panels):
            a = axes[r, c]
            a.imshow(data, cmap="magma", vmin=0, vmax=vmx)
            a.set_xticks([]); a.set_yticks([]); a.set_title(name, fontsize=8.5)
        axes[r, 0].set_ylabel(f"frame {idx[r]}", fontsize=9)
    # aggregate correlations
    cb = float(np.corrcoef(R_base.ravel(), err_base.ravel())[0, 1])
    cd = float(np.corrcoef(R_ddpo.ravel(), err_ddpo.ravel())[0, 1])
    cgb = float(np.corrcoef(R_gt.ravel(), R_base.ravel())[0, 1])
    cgd = float(np.corrcoef(R_gt.ravel(), R_ddpo.ravel())[0, 1])
    print(f"corr(|resid_base|, |err_base|) = {cb:+.3f}   corr(|resid_ddpo|, |err_ddpo|) = {cd:+.3f}", flush=True)
    print(f"corr(|resid_GT|, |resid_base|) = {cgb:+.3f}   corr(|resid_GT|, |resid_ddpo|) = {cgd:+.3f}", flush=True)
    print(f"  => LOW corr means the model residual is its OWN artifact (NOT the GT's irreducible floor),", flush=True)
    print(f"     so penalizing model residual moves the recon TOWARD GT rather than eroding GT-encoded structure.", flush=True)
    print(f"mean |resid|:  base {R_base.mean():.2f}  DDPO {R_ddpo.mean():.2f}  GT {R_gt.mean():.2f}", flush=True)
    fig.suptitle(f"Re={re} seq {seq} — PDE residual as GT-free target? "
                 f"({os.path.basename(ckpt)})\nρ(resid_base,err)={cb:+.2f} (localizes error)   "
                 f"ρ(resid_GT,resid_base)={cgb:+.2f} (LOW => targeting resid ⇒ toward GT, not fighting GT's floor)",
                 y=1.01, fontsize=9.5)
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
    ap.add_argument("--out", type=str, default="monitoring/ddpo_ckpts/viz_pde.png")
    ap.add_argument("--grid_factor", type=int, default=None, help="grid-N input instead of random-1024")
    ap.add_argument("--gt", type=str, default=None, help="GT .npy (default Re=1000)")
    main(**vars(ap.parse_args()))
