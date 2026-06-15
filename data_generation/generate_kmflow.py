"""Generate 2D Kolmogorov-flow vorticity datasets at arbitrary Reynolds number.

Pseudo-spectral solver (FFT, Crank-Nicolson for linear terms) for the 2D
vorticity transport equation on a periodic [0, 2*pi)^2 box:

    d(omega)/dt + (u . grad) omega = (1/Re) laplacian(omega) + f
    f(x) = -4 cos(4 * y) - 0.1 * omega        (Kolmogorov forcing + linear drag)

with velocity recovered from vorticity via the streamfunction
(laplacian(psi) = -omega, u = d(psi)/dy, v = -d(psi)/dx). Initial conditions are
drawn from a Gaussian random field with covariance sigma^2 (-laplacian + tau^2 I)^(-alpha),
sigma = tau^(0.5*(2*alpha - 2)); for tau=7, alpha=2.5 this is N(0, 7^(3/2)(-lap+49I)^(-5/2)),
matching Shu, Li & Farimani (2023) / the FNO data generator of Li et al.

This reproduces the data-generation step that is NOT shipped in the BaratiLab
"Diffusion-based-Fluid-Super-resolution" repo (that repo only ships a PDE *residual*
and an FNO *neural network*, neither of which integrates the PDE forward in time).

Requires PyTorch (GPU strongly recommended). It will NOT run on the JAX/TPU VM.

Output: a .npy of shape (n_samples, record_steps, res, res), float32 — same layout
as `kf_2d_re1000_256_40seed.npy`.

Example (reproduce the Re=1000 setup, 40 trajectories, 256^2, 320 frames):
    python generate_kmflow.py --re 1000 --n-samples 40 --res 256 \
        --record-steps 320 --record-dt 0.03125 --dt 1e-3 --spinup 4.0 \
        --seed 0 --out kf_2d_re1000_256_40seed.npy

Change `--re` to simulate other Reynolds numbers (reduce `--dt` if it blows up).
"""
import argparse
import math

import numpy as np
import torch


def gaussian_random_field(batch, n, tau, alpha, device, dtype):
    """Sample `batch` real vorticity fields ~ N(0, sigma^2 (-lap + tau^2 I)^(-alpha))
    on the periodic [0, 2*pi)^2 torus with integer wavenumbers."""
    sigma = tau ** (0.5 * (2 * alpha - 2.0))  # 2D: dim=2

    k = torch.fft.fftfreq(n, d=1.0 / n, device=device)  # integer wavenumbers 0,1,..,-1
    kx = k.view(n, 1).expand(n, n)
    ky = k.view(1, n).expand(n, n)
    ksq = kx ** 2 + ky ** 2

    # sqrt of the covariance eigenvalues; (n**2) compensates ifft2's 1/n**2 normalization
    sqrt_eig = (n ** 2) * sigma * (ksq + tau ** 2) ** (-alpha / 2.0)
    sqrt_eig[0, 0] = 0.0  # zero-mean field

    noise = torch.randn(batch, n, n, dtype=dtype, device=device) + 1j * torch.randn(
        batch, n, n, dtype=dtype, device=device
    )
    w0 = torch.fft.ifft2(sqrt_eig[None] * noise).real
    return w0.to(dtype)


def navier_stokes_2d(w0, re, dt, n_steps, record_every, record_steps, spinup_steps):
    """Integrate the Kolmogorov-flow vorticity equation. w0: (batch, n, n).
    Returns recorded vorticity snapshots: (batch, record_steps, n, n)."""
    device, dtype = w0.device, w0.dtype
    batch, n, _ = w0.shape
    visc = 1.0 / re

    # wavenumbers on [0, 2*pi)^2
    k = torch.fft.fftfreq(n, d=1.0 / n, device=device)
    kx = k.view(n, 1).expand(n, n)
    ky = k.view(1, n).expand(n, n)
    ksq = (kx ** 2 + ky ** 2).to(dtype)
    ksq_nonzero = ksq.clone()
    ksq_nonzero[0, 0] = 1.0  # avoid division by zero in the Poisson solve

    # 2/3-rule dealiasing mask
    k_max = n // 2
    cutoff = (2.0 / 3.0) * k_max
    dealias = ((kx.abs() <= cutoff) & (ky.abs() <= cutoff)).to(dtype)

    # steady Kolmogorov forcing f = -4 cos(4 y); the -0.1*omega drag is linear -> implicit
    x = torch.linspace(0, 2 * math.pi, n + 1, device=device, dtype=dtype)[:-1]
    yy = x.view(1, n).expand(n, n)
    f = -4.0 * torch.cos(4.0 * yy)
    f_h = torch.fft.fft2(f)[None]

    # linear (implicit) operator: viscous diffusion + linear drag
    lin = visc * ksq + 0.1
    denom = 1.0 + 0.5 * dt * lin
    num_w = 1.0 - 0.5 * dt * lin

    w_h = torch.fft.fft2(w0)
    out = torch.empty(batch, record_steps, n, n, dtype=dtype, device=device)
    rec = 0
    total = spinup_steps + n_steps
    for step in range(total):
        # streamfunction and velocity
        psi_h = w_h / ksq_nonzero
        u = torch.fft.ifft2(1j * ky * psi_h).real
        v = torch.fft.ifft2(-1j * kx * psi_h).real
        wx = torch.fft.ifft2(1j * kx * w_h).real
        wy = torch.fft.ifft2(1j * ky * w_h).real
        advection_h = torch.fft.fft2(u * wx + v * wy) * dealias

        # Crank-Nicolson: explicit nonlinear + forcing, implicit linear
        w_h = (num_w * w_h + dt * (-advection_h + f_h)) / denom

        if step >= spinup_steps and (step - spinup_steps) % record_every == 0 and rec < record_steps:
            out[:, rec] = torch.fft.ifft2(w_h).real
            rec += 1
    return out[:, :rec]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--re", type=float, default=1000.0, help="Reynolds number")
    p.add_argument("--n-samples", type=int, default=40, help="number of trajectories")
    p.add_argument("--res", type=int, default=256, help="grid resolution (power of 2)")
    p.add_argument("--record-steps", type=int, default=320, help="frames recorded per trajectory")
    p.add_argument("--record-dt", type=float, default=1.0 / 32.0, help="time between recorded frames")
    p.add_argument("--dt", type=float, default=1e-3, help="integration timestep (reduce if unstable)")
    p.add_argument("--spinup", type=float, default=4.0, help="time discarded before recording")
    p.add_argument("--batch", type=int, default=8, help="trajectories integrated at once (GPU memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--out", type=str, default="kmflow_generated.npy")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    record_every = max(1, round(args.record_dt / args.dt))
    n_steps = record_every * args.record_steps
    spinup_steps = round(args.spinup / args.dt)
    print(
        f"Re={args.re}  res={args.res}  dt={args.dt}  record_every={record_every} steps"
        f"  n_steps={n_steps}  spinup={spinup_steps} steps  device={device}  dtype={dtype}",
        flush=True,
    )

    all_traj = []
    done = 0
    while done < args.n_samples:
        b = min(args.batch, args.n_samples - done)
        w0 = gaussian_random_field(b, args.res, tau=7.0, alpha=2.5, device=device, dtype=dtype)
        traj = navier_stokes_2d(
            w0, args.re, args.dt, n_steps, record_every, args.record_steps, spinup_steps
        )
        all_traj.append(traj.float().cpu().numpy())
        done += b
        print(f"  {done}/{args.n_samples} trajectories done", flush=True)

    data = np.concatenate(all_traj, axis=0).astype(np.float32)
    np.save(args.out, data)
    print(
        f"Saved {args.out}  shape={data.shape}  mean={data.mean():.5f}  std={data.std():.5f}",
        flush=True,
    )
    print("Validation: for Re=1000 the published kf_2d std is ~4.78 — compare against it.", flush=True)


if __name__ == "__main__":
    main()
