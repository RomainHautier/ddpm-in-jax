"""Is the (resolved, large-scale) STRAIN FIELD a GT-free anchor for WHERE high-k energy belongs?

Premise: filaments (the missing high-k energy) are CREATED by large-scale strain, and the large
scales ARE constrained by the sparse input. If so, the recon's own strain field predicts where GT
puts its fine energy — a GT-free placement target the phase-blind spectral rewards lack.

Per model (GT / base / DDPO-control), on held-out frames:
  [P1 premise]   corr(sigma_recon, sigma_GT)      is the strain field actually resolved?
  [P2 headroom]  corr(sigma_base,  E_hik_GT)      does the recon's strain predict GT's energy
                                                  placement BETTER than the recon's own energy does
                                                  (placement corr ~0.44)? -> headroom for steering
  [S self]       corr(E_hik_x, sigma_x)           energy-follows-strain self-consistency; GT's value
                                                  is the reward reference p_ref
  [W ow]         corr(E_hik_x, relu(W_x))         Okubo-Weiss (strain-dominance) variant
  [A align]      <cos^2(grad-omega, compressive strain axis)>_{|grad w|^2-weighted}
                                                  Batchelor alignment; GT's value = a_ref; gap to
                                                  recon = signal for an orientation reward

    python -m src.ddpo_ft.diag_strain <ddpo_ckpt.pkl> [--seq 36] [--frames 6]
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
from viz_energy import local_hik_energy             # noqa: E402
from src.sequence_inference import build_triplets, load_sequence, sparse_nnfill_degrade  # noqa: E402

MEAN, STD = 0.0, 4.7988
GT_PATH = "flow-data/kf_2d_re1000_256_40seed.npy"


def strain_fields(w, L=2 * np.pi):
    """From vorticity w (H,W): strain magnitude sigma, compressive-eigenvector angle theta_c,
    Okubo-Weiss W = sigma^2 - w^2, and grad-omega (mag^2, angle). Spectral (periodic) derivatives."""
    H, Wd = w.shape
    k = np.fft.fftfreq(Wd) * Wd * (2 * np.pi / L)
    kx, ky = k[None, :], k[:, None]
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0
    wh = np.fft.fft2(w)
    psih = wh / k2
    psih[0, 0] = 0.0
    u = np.real(np.fft.ifft2(1j * ky * psih))          # u = psi_y
    v = np.real(np.fft.ifft2(-1j * kx * psih))         # v = -psi_x
    uh, vh = np.fft.fft2(u), np.fft.fft2(v)
    ux = np.real(np.fft.ifft2(1j * kx * uh)); uy = np.real(np.fft.ifft2(1j * ky * uh))
    vx = np.real(np.fft.ifft2(1j * kx * vh)); vy = np.real(np.fft.ifft2(1j * ky * vh))
    s_n, s_s = ux - vy, vx + uy                        # normal / shear strain
    sigma = np.sqrt(s_n ** 2 + s_s ** 2)
    theta_c = 0.5 * np.arctan2(s_s, s_n) + np.pi / 2   # compressive axis (extensional + 90 deg)
    W_ow = sigma ** 2 - w ** 2
    wx = np.real(np.fft.ifft2(1j * kx * wh)); wy = np.real(np.fft.ifft2(1j * ky * wh))
    g2 = wx ** 2 + wy ** 2
    theta_g = np.arctan2(wy, wx)
    return sigma, theta_c, W_ow, g2, theta_g


def alignment(w):
    """|grad w|^2-weighted <cos^2(theta_g - theta_c)>. 0.5 = random; Batchelor predicts > 0.5."""
    _, theta_c, _, g2, theta_g = strain_fields(w)
    c2 = np.cos(theta_g - theta_c) ** 2
    return float((g2 * c2).sum() / (g2.sum() + 1e-20))


def _corr(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def main(ckpt, seq=36, frames=6, t_start=100, seed=1, kcut=32, sigma_blur=6.0,
         out="monitoring/ddpo_ckpts/diag_strain.png"):
    ddpm, base_params, _ = build_base_ddpm()
    ddpo_params = pickle.load(open(ckpt, "rb"))["params"]
    sampler = make_sampler(ddpm.unet, ddpm.alpha_bar, ddpm.beta_schedule, t_start, temp=1.0)
    ab = ddpm.alpha_bar
    sa, s1 = float(jnp.sqrt(ab[t_start])), float(jnp.sqrt(1.0 - ab[t_start]))

    s = load_sequence(GT_PATH, seq)
    inp = build_triplets(sparse_nnfill_degrade(s, seq), MEAN, STD)
    gt = build_triplets(s, MEAN, STD)
    idx = np.linspace(len(inp) // 4, len(inp) - 1, frames).astype(int)
    xin, xgt = jnp.asarray(inp[idx]), np.asarray(gt[idx])
    print(f"reconstructing {frames} frames of seq {seq} for strain-placement diagnostics ...", flush=True)
    key = jax.random.PRNGKey(seed)
    def recon(params):
        k1, k2 = jax.random.split(key)
        return np.asarray(sampler(params, sa * xin + s1 * jax.random.normal(k1, xin.shape), k2))
    b0, d0 = recon(base_params), recon(ddpo_params)

    fields = {"gt": xgt[..., 1] * STD, "base": b0[..., 1] * STD, "ddpo": d0[..., 1] * STD}
    E, SIG, WOW, AL = {}, {}, {}, {}
    for name, W in fields.items():
        E[name] = local_hik_energy(W, kcut, sigma_blur)
        sig, wow = [], []
        for i in range(frames):
            sg, _, wo, _, _ = strain_fields(W[i])
            sig.append(sg); wow.append(np.maximum(wo, 0.0))
        SIG[name], WOW[name] = np.stack(sig), np.stack(wow)
        AL[name] = float(np.mean([alignment(W[i]) for i in range(frames)]))

    print("\n[P1 premise] is the strain field resolved by the recon?")
    print(f"  corr(sigma_base, sigma_GT) = {_corr(SIG['base'], SIG['gt']):+.3f}   "
          f"corr(sigma_ddpo, sigma_GT) = {_corr(SIG['ddpo'], SIG['gt']):+.3f}")

    print("\n[P2 headroom] does the recon's OWN strain predict GT's energy placement?")
    p_sb_eg = _corr(SIG["base"], E["gt"]); p_wb_eg = _corr(WOW["base"], E["gt"])
    p_eb_eg = _corr(E["base"], E["gt"])
    print(f"  corr(sigma_base, E_hik_GT) = {p_sb_eg:+.3f}   corr(W+_base, E_hik_GT) = {p_wb_eg:+.3f}")
    print(f"  vs current placement corr(E_hik_base, E_hik_GT) = {p_eb_eg:+.3f}")
    print(f"  -> headroom = {max(p_sb_eg, p_wb_eg) - p_eb_eg:+.3f}  (>0 => strain knows more about GT's "
          f"placement than the recon's energy currently does)")

    print("\n[S self-consistency] corr(E_hik_x, sigma_x)  — reward candidate #1 (push recon toward GT's value)")
    for name in ("gt", "base", "ddpo"):
        print(f"  {name:<5} corr(E,sigma) = {_corr(E[name], SIG[name]):+.3f}   "
              f"corr(E,W+) = {_corr(E[name], WOW[name]):+.3f}")

    print("\n[A alignment] <cos^2(grad-w, compressive axis)>  (0.5 = random) — reward candidate #2")
    for name in ("gt", "base", "ddpo"):
        print(f"  {name:<5} alignment = {AL[name]:.4f}")
    print(f"  gap GT-base = {AL['gt'] - AL['base']:+.4f}  GT-ddpo = {AL['gt'] - AL['ddpo']:+.4f}")

    # figure: one frame's maps
    r = 0
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for j, name in enumerate(("gt", "base", "ddpo")):
        a = ax[0, j]; a.imshow(E[name][r], cmap="viridis", vmin=0, vmax=np.percentile(E["gt"][r], 99))
        a.set_title(f"E_hik({name})", fontsize=9); a.set_xticks([]); a.set_yticks([])
        b = ax[1, j]; b.imshow(SIG[name][r], cmap="magma", vmin=0, vmax=np.percentile(SIG["gt"][r], 99))
        b.set_title(f"strain sigma({name})  corr(E,sig)={_corr(E[name][r], SIG[name][r]):+.2f}", fontsize=9)
        b.set_xticks([]); b.set_yticks([])
    fig.suptitle(f"seq {seq} — strain as GT-free placement anchor | "
                 f"corr(sig_base,E_GT)={p_sb_eg:+.2f} vs placement {p_eb_eg:+.2f}", y=1.0, fontsize=10)
    plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seq", type=int, default=36)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--t_start", type=int, default=100)
    main(**vars(ap.parse_args()))
