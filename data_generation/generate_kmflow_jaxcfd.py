"""Generate 2D Kolmogorov-flow vorticity datasets with jax-cfd, on TPU/GPU/CPU.

Uses Kochkov et al.'s `google/jax-cfd` spectral solver (the authoritative, validated
Kolmogorov-flow implementation) instead of a hand-rolled solver. Pure JAX, so it runs
natively on a TPU. Reproduces the Shu et al. `kf_2d_re1000_256_40seed.npy` setup:

  - vorticity NS on (0, 2*pi)^2, periodic, forcing f = -4 cos(4 y) - 0.1 omega
    (jax-cfd's ForcedNavierStokes2D: Kolmogorov forcing at k=4 + linear drag 0.1)
  - Crank-Nicolson RK4 time stepping, CFL=0.5
  - DNS at --dns-res, spun up to STEADY STATE (kf_2d frame-0 already shows the k=4
    peak & std~4.6, so a long spin-up is required), then 320 frames recorded at dt=1/32
  - each recorded frame spectrally downsampled to --out-res (e.g. 2048 -> 256)

Requires a jax-cfd env (separate from the JAX/TPU training venv to avoid version clashes):
    python3 -m venv ~/venv-jaxcfd && source ~/venv-jaxcfd/bin/activate
    pip install jax-cfd && pip install "jax[tpu]==0.10.1"

Cost note: exact 2048^2 is very expensive (~hours/sequence even on TPU). 512-1024^2
downsampled to 256^2 is a strong, far cheaper approximation (the 256^2 output only keeps
modes up to k=128, which a 512^2 DNS already over-resolves).

Example (practical Re=1000, 1024^2 DNS -> 256^2, 40 sequences):
    python generate_kmflow_jaxcfd.py --re 1000 --n-samples 40 --dns-res 1024 --out-res 256 \
        --spinup-time 40 --record-frames 320 --record-dt 0.03125 --seed 0 \
        --out kf_re1000_256_jaxcfd.npy
"""
import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral


def fourier_downsample(wh, n, m):
    """Truncate an rfft2 vorticity field (n x n grid) to an m x m real field by keeping
    the lowest wavenumbers. Returns physical-space vorticity (m, m). Memory-light: the
    full DNS frame is never materialized in host memory."""
    if m >= n:
        return jnp.fft.irfftn(wh, s=(n, n))
    h = m // 2
    wh = wh[:, : h + 1]                                  # rfft axis: keep 0..m/2
    wh = jnp.concatenate([wh[: h + 1], wh[n - h + 1 :]], axis=0)  # full axis: low |freq|, m rows
    return jnp.fft.irfftn(wh, s=(m, m)) * (m / n) ** 2  # rescale for grid-size change


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--re", type=float, default=1000.0)
    p.add_argument("--n-samples", type=int, default=40)
    p.add_argument("--dns-res", type=int, default=1024, help="DNS resolution (power of 2)")
    p.add_argument("--out-res", type=int, default=256, help="downsampled output resolution")
    p.add_argument("--spinup-time", type=float, default=40.0, help="time integrated to reach steady state before recording")
    p.add_argument("--record-frames", type=int, default=320)
    p.add_argument("--record-dt", type=float, default=1.0 / 32.0, help="time between recorded frames")
    p.add_argument("--max-velocity", type=float, default=7.0, help="IC max velocity (jax-cfd default)")
    p.add_argument("--peak-wavenumber", type=int, default=4, help="IC peak wavenumber")
    p.add_argument("--cfl", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="kmflow_jaxcfd.npy")
    args = p.parse_args()

    visc = 1.0 / args.re
    n = args.dns_res
    grid = grids.Grid((n, n), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    dt = cfd.equations.stable_time_step(args.max_velocity, args.cfl, visc, grid)
    spinup_steps = int(args.spinup_time / dt)
    inner = max(1, round(args.record_dt / dt))
    print(
        f"Re={args.re} dns={n} -> out={args.out_res} | dt={dt:.3e} | spinup={spinup_steps} steps "
        f"| {inner} steps/frame x {args.record_frames} frames | devices={jax.device_count()}",
        flush=True,
    )

    eq = spectral.equations.ForcedNavierStokes2D(visc, grid, smooth=True)
    step = spectral.time_stepping.crank_nicolson_rk4(eq, dt)
    spin_fn = jax.jit(cfd.funcutils.repeated(step, spinup_steps))
    rec_fn = jax.jit(cfd.funcutils.repeated(step, inner))
    down_fn = jax.jit(lambda wh: fourier_downsample(wh, n, args.out_res))

    seqs = []
    for s in range(args.n_samples):
        t0 = time.time()
        v0 = cfd.initial_conditions.filtered_velocity_field(
            jax.random.PRNGKey(args.seed + s), grid, args.max_velocity, args.peak_wavenumber
        )
        wh = jnp.fft.rfftn(cfd.finite_differences.curl_2d(v0).data)
        wh = spin_fn(wh)
        frames = []
        for _ in range(args.record_frames):
            wh = rec_fn(wh)
            frames.append(np.asarray(down_fn(wh)))
        seq = np.stack(frames)
        seqs.append(seq)
        print(f"  seq {s + 1}/{args.n_samples}: std={seq.std():.3f} ({time.time() - t0:.1f}s)", flush=True)

    data = np.stack(seqs).astype(np.float32)
    np.save(args.out, data)
    print(f"Saved {args.out}  shape={data.shape}  mean={data.mean():.4f}  std={data.std():.4f}", flush=True)
    print("Validate vs kf_2d with: python data_generation/validate_kmflow.py --generated <out> --reference flow-data/kf_2d_re1000_256_40seed.npy", flush=True)


if __name__ == "__main__":
    main()
