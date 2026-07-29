"""Generate 2D Navier-Stokes vorticity data matching the MNO / FNO datasets, in pure JAX (TPU).

Reproduces the data-generation setup of the Zenodo record 7495555 ("Learning Dissipative
Dynamics in Chaotic Systems", Li et al. 2022) — the `2D_NS_Re{40,500,5000}.npy` files — which
were generated with the FNO pseudo-spectral solver (NOT the kf_2d Kolmogorov setup this repo's
model trains on). This is a faithful JAX port of that vorticity-form solver so it runs on TPU.

Equation (vorticity form, periodic torus, here mapped to (0, 2*pi)^2 with integer wavenumbers):

    d(omega)/dt + (u . grad) omega = nu * laplacian(omega) + f

with velocity from the streamfunction (laplacian(psi) = -omega, u = d psi/dy, v = -d psi/dx),
Crank-Nicolson on the linear (viscous) term + explicit nonlinear advection, and 2/3-rule
dealiasing. Initial conditions are a Gaussian random field N(0, tau^(alpha-1)(-lap+tau^2 I)^(-alpha)),
tau=7, alpha=2.5 — identical to the FNO generator.

Three forcing modes (`--forcing`), all with nu = 1/Re:
  * mno  (default): f = -4*cos(4*y), linear drag 0.01 -> the calibrated match to the Zenodo
                    2D_NS Kolmogorov data (k=4 forcing, tiny drag; see forcing_field()).
  * kf            : f = -4*cos(4*y), linear drag 0.1  -> this repo's kf_2d forcing (k=4 + drag),
                    for cross-checking against the training data.
  * fno           : f = 0.1*(sin(x+y) + cos(x+y)), NO drag -> the original FNO low-k forcing;
                    kept for reference only (std~1.5, does NOT match the MNO NS data).

The MNO arrays are float64, shape (n_traj, 501, res, res) (frame 0 = initial condition, then
500 recorded steps), Re/res/n_traj: 40/64/200, 500/64/1000, 5000/128/100.

  -- CALIBRATION --
The Zenodo authors did NOT publish their exact `dt` and recording interval. The forcing, domain,
viscosity (nu=1/Re), IC, dtype, and shape above are pinned from the data + the FNO solver, but the
TIME SPACING between frames is not. Calibrate `--dt` / `--record-dt` against a downloaded array:
match (a) per-frame std (Re500 download ~4.78, identical to kf_2d), (b) the k~1-peaked spectrum, and (c) the
frame-to-frame autocorrelation, using `validate_kmflow.py`. Defaults below are a starting point,
not the published values.

Runs on TPU/GPU/CPU. float64 is the faithful dtype but is slow/emulated on TPU; use --dtype
float32 for speed (the diffusion model normalizes to float32 anyway).

Example (Re=500 set, calibrated forcing; still calibrate dt/record-dt against the download):
    python data_generation/generate_fno_ns_jax.py --forcing mno --re 500 --res 64 \
        --n-samples 1000 --record-steps 501 --dt 1e-3 --record-dt 0.34 --seed 0 \
        --dtype float32 --out 2D_NS_Re500_regen.npy
"""
import argparse

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P


def gaussian_random_field(key, batch, n, tau=7.0, alpha=2.5):
    """Sample `batch` real vorticity fields ~ N(0, sigma^2 (-lap + tau^2 I)^(-alpha)) on the
    periodic (0, 2*pi)^2 torus with integer wavenumbers. Matches the FNO GaussianRF."""
    sigma = tau ** (0.5 * (2 * alpha - 2.0))
    k = jnp.fft.fftfreq(n, d=1.0 / n)               # integer wavenumbers 0,1,..,-1
    kx = k[:, None]
    ky = k[None, :]
    ksq = kx ** 2 + ky ** 2
    sqrt_eig = (n ** 2) * sigma * (ksq + tau ** 2) ** (-alpha / 2.0)  # (n**2) offsets ifft2's 1/n**2
    sqrt_eig = sqrt_eig.at[0, 0].set(0.0)            # zero-mean field
    kr, ki = jax.random.split(key)
    noise = jax.random.normal(kr, (batch, n, n)) + 1j * jax.random.normal(ki, (batch, n, n))
    return jnp.fft.ifft2(sqrt_eig[None] * noise).real


def build_operators(n, dtype):
    """Wavenumber grids and the forcing meshgrid coordinates on (0, 2*pi)^2."""
    k = jnp.fft.fftfreq(n, d=1.0 / n).astype(dtype)
    kx = k[:, None] * jnp.ones((1, n), dtype)
    ky = k[None, :] * jnp.ones((n, 1), dtype)
    ksq = kx ** 2 + ky ** 2
    ksq_nonzero = ksq.at[0, 0].set(1.0)             # avoid /0 in the Poisson solve
    k_max = n // 2
    cutoff = (2.0 / 3.0) * k_max
    dealias = ((jnp.abs(kx) <= cutoff) & (jnp.abs(ky) <= cutoff)).astype(dtype)
    coord = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False, dtype=dtype)
    xx, yy = jnp.meshgrid(coord, coord, indexing="ij")
    return kx, ky, ksq, ksq_nonzero, dealias, xx, yy


def forcing_field(forcing, xx, yy):
    """Return (f, drag): the vorticity forcing field and the linear-drag coefficient.

    Calibrated against the downloaded 2D_NS_Re500.npy: its steady std is ~4.78 (identical to
    kf_2d) and its vorticity spectrum peaks at k=1 with a strong k=4 component. That is k=4
    Kolmogorov forcing at amplitude 4 with NO drag — removing the drag lets the 2D inverse
    cascade condense energy at k=1. The original FNO low-k forcing (0.1(sin+cos)) gives std~1.5
    and does NOT match, so it's kept only for reference.
    """
    if forcing == "mno":      # Zenodo 2D_NS Kolmogorov data: k=4 forcing, small drag (calibrated)
        return -4.0 * jnp.cos(4.0 * yy), 0.01
    if forcing == "kf":       # this repo's kf_2d training-data forcing: k=4 + linear drag 0.1
        return -4.0 * jnp.cos(4.0 * yy), 0.1
    if forcing == "fno":      # original FNO low-k forcing — does NOT match the MNO NS data
        return 0.1 * (jnp.sin(xx + yy) + jnp.cos(xx + yy)), 0.0
    raise ValueError(f"unknown forcing '{forcing}'")


def spectral_downsample_hat(w_h, dns, out):
    """Truncate a full-fft2 vorticity field (..., dns, dns) to physical (..., out, out) by
    keeping the lowest wavenumbers — the standard way a 2048^2 DNS frame is reduced to 256^2.
    The (out/dns)^2 factor compensates the change in ifft2's 1/N^2 normalization."""
    if out >= dns:
        return jnp.fft.ifft2(w_h, axes=(-2, -1)).real
    wsh = jnp.fft.fftshift(w_h, axes=(-2, -1))
    c, h = dns // 2, out // 2
    wsh = wsh[..., c - h : c + h, c - h : c + h]
    return jnp.fft.ifft2(jnp.fft.ifftshift(wsh, axes=(-2, -1)), axes=(-2, -1)).real * (out / dns) ** 2


def make_integrator(n, visc, dt, drag, f_h, dealias, kx, ky, ksq, ksq_nonzero,
                    spinup_steps, record_steps, record_every, out_res=None):
    """Build a jitted batched integrator. DNS runs at resolution n; each recorded frame is
    spectrally downsampled to out_res (<= n): (B, n, n) IC -> (record_steps, B, out, out)."""
    out_res = out_res or n
    lin = visc * ksq + drag
    denom = 1.0 + 0.5 * dt * lin
    num_w = 1.0 - 0.5 * dt * lin

    def step(w_h, _):
        psi_h = w_h / ksq_nonzero
        u = jnp.fft.ifft2(1j * ky * psi_h, axes=(-2, -1)).real
        v = jnp.fft.ifft2(-1j * kx * psi_h, axes=(-2, -1)).real
        wx = jnp.fft.ifft2(1j * kx * w_h, axes=(-2, -1)).real
        wy = jnp.fft.ifft2(1j * ky * w_h, axes=(-2, -1)).real
        adv_h = jnp.fft.fft2(u * wx + v * wy, axes=(-2, -1)) * dealias
        w_h = (num_w * w_h + dt * (-adv_h + f_h)) / denom   # CN linear + explicit nonlinear/forcing
        return w_h, None

    @jax.jit
    def integrate(w0):
        w_h = jnp.fft.fft2(w0, axes=(-2, -1))
        if spinup_steps:
            w_h, _ = lax.scan(step, w_h, None, length=spinup_steps)

        def rec_body(w_h, _):
            w_h, _ = lax.scan(step, w_h, None, length=record_every)
            return w_h, spectral_downsample_hat(w_h, n, out_res)

        frame0 = spectral_downsample_hat(w_h, n, out_res)        # frame 0 = IC (post-spinup)
        w_h, rest = lax.scan(rec_body, w_h, None, length=record_steps - 1)
        return jnp.concatenate([frame0[None], rest], axis=0)     # (record_steps, B, out, out)

    return integrate


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forcing", choices=["mno", "kf", "fno"], default="mno",
                   help="mno = Zenodo 2D_NS Kolmogorov (k=4, no drag); kf = kf_2d (k=4 + drag)")
    p.add_argument("--re", type=float, default=500.0, help="Reynolds number; nu = 1/Re")
    p.add_argument("--drag", type=float, default=None, help="override the forcing's linear-drag coefficient")
    p.add_argument("--res", type=int, default=64, help="grid resolution when not using DNS downsample")
    p.add_argument("--dns-res", type=int, default=None, help="DNS resolution (e.g. 2048); defaults to --res")
    p.add_argument("--out-res", type=int, default=None, help="output resolution after spectral downsample (e.g. 256); defaults to --res")
    p.add_argument("--n-samples", type=int, default=1000, help="number of trajectories")
    p.add_argument("--record-steps", type=int, default=501, help="frames per trajectory incl. IC at t=0")
    p.add_argument("--dt", type=float, default=5e-4, help="integration timestep (reduce if unstable at high Re)")
    p.add_argument("--record-dt", type=float, default=0.2, help="time between recorded frames (CALIBRATE vs download autocorr)")
    p.add_argument("--spinup-time", type=float, default=0.0, help="time discarded before recording (MNO records from the IC at t=0)")
    p.add_argument("--batch", type=int, default=10, help="trajectories integrated at once (device memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--out", type=str, default="2D_NS_regen.npy")
    args = p.parse_args()

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    np_dtype = np.float64 if args.dtype == "float64" else np.float32

    dns = args.dns_res or args.res          # DNS resolution
    out_res = args.out_res or args.res      # recorded/output resolution after downsample
    visc = 1.0 / args.re
    record_every = max(1, round(args.record_dt / args.dt))
    spinup_steps = round(args.spinup_time / args.dt)

    kx, ky, ksq, ksq_nonzero, dealias, xx, yy = build_operators(dns, dtype)
    f, drag = forcing_field(args.forcing, xx, yy)
    if args.drag is not None:
        drag = args.drag
    f_h = jnp.fft.fft2(f)
    integrate = make_integrator(
        dns, visc, args.dt, drag, f_h, dealias, kx, ky, ksq, ksq_nonzero,
        spinup_steps, args.record_steps, record_every, out_res=out_res,
    )

    print(
        f"forcing={args.forcing} Re={args.re} (nu={visc:.3e}) DNS={dns}->out={out_res} | dt={args.dt} "
        f"record_every={record_every} steps x {args.record_steps} frames | spinup={spinup_steps} "
        f"| n_traj={args.n_samples} dtype={args.dtype} | devices={jax.device_count()}",
        flush=True,
    )

    # Data-parallel: shard the trajectory batch across all chips so each runs an independent
    # DNS (the per-trajectory FFTs need no cross-device communication -> ~n_devices speedup).
    n_devices = jax.device_count()
    mesh = jax.make_mesh((n_devices,), ("batch",))
    batch_shard = NamedSharding(mesh, P("batch"))

    # Write straight to a disk-backed array so a large dataset never has to fit in RAM.
    out = np.lib.format.open_memmap(
        args.out, mode="w+", dtype=np_dtype, shape=(args.n_samples, args.record_steps, out_res, out_res)
    )
    done = 0
    while done < args.n_samples:
        b = min(args.batch, args.n_samples - done)
        key = jax.random.PRNGKey(args.seed + done)            # distinct IC per trajectory block
        w0 = gaussian_random_field(key, b, dns).astype(dtype)
        if b % n_devices == 0:                                # shard only when it divides evenly
            w0 = jax.device_put(w0, batch_shard)
        frames = np.asarray(integrate(w0))                    # (record_steps, b, out_res, out_res)
        out[done : done + b] = np.transpose(frames, (1, 0, 2, 3)).astype(np_dtype)
        done += b
        print(f"  {done}/{args.n_samples} trajectories  (last-batch std={frames.std():.3f})", flush=True)
    out.flush()
    print(f"Saved {args.out}  shape={out.shape}  dtype={out.dtype}  mean={out.mean():.4f}  std={out.std():.4f}", flush=True)
    print(
        "CALIBRATE dt/record-dt vs the download:\n"
        "  python data_generation/validate_kmflow.py --generated "
        f"{args.out} --reference flow-data/2D_NS_Re{int(args.re)}.npy",
        flush=True,
    )


if __name__ == "__main__":
    main()
