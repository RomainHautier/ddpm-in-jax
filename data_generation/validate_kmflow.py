"""Validate a generated Kolmogorov-flow dataset against reference full-resolution data.

Generated trajectories use random initial conditions and chaotic dynamics, so they will
NOT match a reference set frame-by-frame. Validation is therefore STATISTICAL:

  1. Global stats: mean (≈0) and std (≈4.78 for Re=1000 kf_2d).
  2. Vorticity value distribution (PDF overlay).
  3. Radially-averaged energy & enstrophy spectra (the key turbulence fingerprint).
  4. Per-frame std over time (stationarity / spin-up check).

Pure numpy + matplotlib — runs anywhere (no PyTorch / GPU needed), so it can validate a
GPU-generated .npy against the local kf_2d_re1000_256_40seed.npy on this VM.

Both inputs must be shape (n_seqs, n_frames, H, W).

Example:
    python data_generation/validate_kmflow.py \
        --generated kf_2d_re1000_256_40seed_REGEN.npy \
        --reference flow-data/kf_2d_re1000_256_40seed.npy \
        --out monitoring/validation_re1000.png
"""
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sample_frames(path, n_frames, rng):
    """Memory-safe: mmap the array and pull n_frames random (seq, t) slices."""
    arr = np.load(path, mmap_mode="r")
    assert arr.ndim == 4, f"{path}: expected (n_seqs, n_frames, H, W), got {arr.shape}"
    n_seq, n_t = arr.shape[0], arr.shape[1]
    n_frames = min(n_frames, n_seq * n_t)
    flat_idx = rng.choice(n_seq * n_t, size=n_frames, replace=False)
    frames = np.stack([np.asarray(arr[i // n_t, i % n_t], dtype=np.float64) for i in flat_idx])
    return frames, arr.shape


def radial_spectrum(frames):
    """Average radially-binned energy & enstrophy spectra over a stack of (H,W) fields."""
    n = frames.shape[-1]
    k = np.fft.fftfreq(n, d=1.0 / n)  # integer wavenumbers
    KX, KY = np.meshgrid(k, k, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2)
    ksq = kmag ** 2
    ksq[0, 0] = 1.0
    kbins = np.arange(0.5, n // 2 + 1, 1.0)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    masks = [(kmag >= kbins[i]) & (kmag < kbins[i + 1]) for i in range(len(kvals))]

    ens = np.zeros(len(kvals))
    ene = np.zeros(len(kvals))
    for f in frames:
        psd = np.abs(np.fft.fft2(f)) ** 2 / (n ** 2)  # enstrophy density |w_hat|^2
        epsd = psd / ksq  # energy density |u_hat|^2 = |w_hat|^2 / |k|^2
        for i, m in enumerate(masks):
            ens[i] += psd[m].sum()
            ene[i] += epsd[m].sum()
    return kvals, ene / len(frames), ens / len(frames)


def per_frame_std(path, max_seqs=4):
    arr = np.load(path, mmap_mode="r")
    seqs = min(max_seqs, arr.shape[0])
    return np.stack([np.asarray(arr[s]).std(axis=(-1, -2)) for s in range(seqs)]).mean(0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generated", required=True, help="generated dataset .npy (n_seqs,n_frames,H,W)")
    p.add_argument("--reference", required=True, help="reference full-res dataset .npy")
    p.add_argument("--n-spectrum-frames", type=int, default=200, help="random frames per dataset for spectra")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="kmflow_validation.png", help="output comparison figure")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    gen, gshape = sample_frames(args.generated, args.n_spectrum_frames, rng)
    ref, rshape = sample_frames(args.reference, args.n_spectrum_frames, rng)

    print(f"generated {gshape}: mean={gen.mean():+.5f}  std={gen.std():.5f}")
    print(f"reference {rshape}: mean={ref.mean():+.5f}  std={ref.std():.5f}")
    print(f"std ratio gen/ref = {gen.std() / ref.std():.4f}  (want ~1.0)")

    kv, gene, gens = radial_spectrum(gen)
    _, rene, rens = radial_spectrum(ref)
    # spectral agreement metric: mean abs log10 ratio of energy spectra (lower = better)
    valid = (rene > 0) & (gene > 0)
    spec_err = float(np.abs(np.log10(gene[valid] / rene[valid])).mean())
    print(f"energy-spectrum mean |log10 ratio| = {spec_err:.3f}  (0 = identical; <~0.2 is close)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (1) vorticity PDF
    ax = axes[0, 0]
    lo, hi = np.percentile(np.concatenate([gen.ravel(), ref.ravel()]), [0.1, 99.9])
    bins = np.linspace(lo, hi, 120)
    ax.hist(ref.ravel(), bins=bins, density=True, histtype="step", color="C0", label="reference")
    ax.hist(gen.ravel(), bins=bins, density=True, histtype="step", color="C3", label="generated")
    ax.set_yscale("log"); ax.set_xlabel("vorticity"); ax.set_ylabel("pdf"); ax.set_title("vorticity distribution"); ax.legend()

    # (2) energy spectrum
    ax = axes[0, 1]
    ax.loglog(kv, rene, color="C0", label="reference")
    ax.loglog(kv, gene, color="C3", label="generated")
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("E(k)"); ax.set_title("energy spectrum"); ax.legend(); ax.grid(alpha=0.3, which="both")

    # (3) enstrophy spectrum
    ax = axes[1, 0]
    ax.loglog(kv, rens, color="C0", label="reference")
    ax.loglog(kv, gens, color="C3", label="generated")
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("enstrophy(k)"); ax.set_title("enstrophy spectrum"); ax.legend(); ax.grid(alpha=0.3, which="both")

    # (4) per-frame std over time (stationarity)
    ax = axes[1, 1]
    ax.plot(per_frame_std(args.reference), color="C0", label="reference")
    ax.plot(per_frame_std(args.generated), color="C3", label="generated")
    ax.set_xlabel("frame index"); ax.set_ylabel("per-frame std"); ax.set_title("stationarity over time"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(
        f"kmflow validation  |  std ratio={gen.std()/ref.std():.3f}  |  energy-spec |log10 ratio|={spec_err:.3f}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
