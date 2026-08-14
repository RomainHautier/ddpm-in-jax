"""Sampling-time physics guidance — faithful JAX port of BaratiLab's "Linear" variant.

Reference: BaratiLab/Diffusion-based-Fluid-Super-resolution, runners/rs256_guided_diffusion.py
(`voriticity_residual`) and functions/denoising_step.py (`ddpm_steps`/`ddim_steps`). No retraining,
no architecture change: at each reverse step subtract dx = lambda * grad_w(mean residual^2) / std
from the sample (their `physical_gradient_func`), with the residual evaluated on the de-normalized
(physical) field.

PDE residual on a 3-frame vorticity triplet (omega_{t-1}, omega_t, omega_{t+1}), matching theirs:

    residual = wt + (advection - (1/Re) * lap(omega) + 0.1*omega) - f,    f = -4 cos(4 y)

- wt = (omega_{t+1} - omega_{t-1}) / (2 dt): finite-diff time derivative (the only FD term).
- advection u.grad(omega): spectral derivatives on the (0, 2*pi)^2 torus, integer wavenumbers,
  velocity from the streamfunction (lap psi = -omega). NO 2/3 dealiasing (matching their residual).
- the +0.1*omega drag and the -4 cos(4 y) forcing are the kf_2d source terms.

The guidance gradient is autodiff (jax.grad), exactly as their torch.autograd.grad. For OOD
super-resolution, build with the TARGET flow's Re so the guidance injects the correct physics.
Grid/forcing conventions are identical to data_generation/generate_fno_ns_jax.build_operators, so
the residual is consistent with the flows the OOD data was generated from.
"""
import jax
import jax.numpy as jnp


def _operators(n, dtype):
    k = jnp.fft.fftfreq(n, d=1.0 / n).astype(dtype)         # integer wavenumbers on (0, 2*pi)
    kx = k[:, None] * jnp.ones((1, n), dtype)
    ky = k[None, :] * jnp.ones((n, 1), dtype)
    ksq = kx ** 2 + ky ** 2
    ksq_nz = ksq.at[0, 0].set(1.0)                          # avoid /0 in the Poisson solve
    coord = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False, dtype=dtype)
    _, yy = jnp.meshgrid(coord, coord, indexing="ij")
    return kx, ky, ksq, ksq_nz, yy


def make_ns_residual(n=256, re=1000.0, dt=1.0 / 32.0, dtype=jnp.float32):
    """Return residual(w) -> (..., n, n): the NS vorticity residual at the middle frame of a
    PHYSICAL (..., n, n, 3) triplet. Faithful port of BaratiLab's voriticity_residual()."""
    kx, ky, ksq, ksq_nz, yy = _operators(n, dtype)
    visc = 1.0 / re
    forcing = -4.0 * jnp.cos(4.0 * yy)

    def residual(w):
        wt = (w[..., 2] - w[..., 0]) / (2.0 * dt)            # finite-diff time derivative
        wm = w[..., 1]                                       # middle frame
        wh = jnp.fft.fft2(wm, axes=(-2, -1))
        psih = wh / ksq_nz
        u = jnp.fft.ifft2(1j * ky * psih, axes=(-2, -1)).real
        v = jnp.fft.ifft2(-1j * kx * psih, axes=(-2, -1)).real
        wx = jnp.fft.ifft2(1j * kx * wh, axes=(-2, -1)).real
        wy = jnp.fft.ifft2(1j * ky * wh, axes=(-2, -1)).real
        wlap = jnp.fft.ifft2(-ksq * wh, axes=(-2, -1)).real
        advection = u * wx + v * wy
        return wt + (advection - visc * wlap + 0.1 * wm) - forcing
    return residual


def make_dx_func(n=256, re=1000.0, dt=1.0 / 32.0, std=4.7988, mean=0.0, dtype=jnp.float32):
    """Return a jitted dx(x_norm) -> (..., n, n, 3): the RAW residual gradient
    grad_w(mean residual^2) / std on the de-normalized field (BaratiLab's
    voriticity_residual(...)[0] / scale, WITHOUT lambda). This single quantity serves both
    guidance paradigms — apply the paradigm at the call site, not here:
      - LEARNED: feed it to the model as `condRes` (matches training, which used lambda=1).
      - LINEAR : subtract `lambda * dx` from the sample after each reverse step."""
    residual = make_ns_residual(n, re, dt, dtype)

    def residual_loss(w):
        return jnp.mean(residual(w) ** 2)                   # their (residual**2).mean()

    grad_w = jax.grad(residual_loss)

    def dx(x_norm):
        w = x_norm * std + mean                             # scaler.inverse(x)
        return grad_w(w) / std                              # dw / scaler.scale()

    return jax.jit(dx)


def make_field_func(n=256, re=1000.0, dt=1.0 / 32.0, std=4.7988, mean=0.0, dtype=jnp.float32):
    """Return a jitted field(x_norm) -> (..., n, n, 3): the RAW NS residual FIELD (NOT its
    gradient), normalized by `std` and broadcast across the 3 triplet channels, to be fed to
    the model as `condRes`.

    This is the ENS ("Error-Conditioned Neural Solvers", arXiv:2606.27354) signal: the network
    sees the *spatial error structure* directly and learns the correction, rather than being
    handed the residual-*minimization direction* that `make_dx_func` provides (the linear-guidance
    gradient). The NS residual is a single middle-frame map; we tile it to 3 channels so it drops
    into the existing `cond_in` (in_ch=3) with no architecture change.

    Normalization: divide by the data scaler `std` (same scale as the normalized input) to keep the
    input O(1)-ish. This is the obvious knob to sweep if the field signal proves too hot/cold."""
    residual = make_ns_residual(n, re, dt, dtype)

    def field(x_norm):
        w = x_norm * std + mean                             # scaler.inverse(x)
        r = residual(w) / std                               # (..., n, n), same scale as input
        return jnp.repeat(r[..., None], 3, axis=-1)         # (..., n, n, 3) — tile to cond_in channels

    return jax.jit(field)


def make_field_func_visc(n=256, dt=1.0 / 32.0, std=4.7988, mean=0.0, dtype=jnp.float32):
    """Viscosity-parameterized variant of make_field_func: field(x_norm, visc) where visc is a
    TRACED scalar (1/Re), so ONE compiled rollout/loss serves every regime with visc passed per
    batch — instead of nine re-baked closures. Same ENS residual-field signal, same /std
    normalization, same 3-channel tiling for `cond_in`. Deliberately NOT jitted here: it is traced
    inside the pmapped policy functions that consume it."""
    kx, ky, ksq, ksq_nz, yy = _operators(n, dtype)
    forcing = -4.0 * jnp.cos(4.0 * yy)

    def field(x_norm, visc):
        w = x_norm * std + mean
        wt = (w[..., 2] - w[..., 0]) / (2.0 * dt)
        wm = w[..., 1]
        wh = jnp.fft.fft2(wm, axes=(-2, -1))
        psih = wh / ksq_nz
        u = jnp.fft.ifft2(1j * ky * psih, axes=(-2, -1)).real
        v = jnp.fft.ifft2(-1j * kx * psih, axes=(-2, -1)).real
        wx = jnp.fft.ifft2(1j * kx * wh, axes=(-2, -1)).real
        wy = jnp.fft.ifft2(1j * ky * wh, axes=(-2, -1)).real
        wlap = jnp.fft.ifft2(-ksq * wh, axes=(-2, -1)).real
        r = wt + (u * wx + v * wy - visc * wlap + 0.1 * wm) - forcing
        return jnp.repeat((r / std)[..., None], 3, axis=-1)
    return field


def make_cond_func(signal="gradient", **kw):
    """Select the LEARNED-conditioning signal fed to the adapter as `condRes`:
      - 'gradient' -> make_dx_func  : residual-minimization direction (original / linear-guidance grad).
      - 'field'    -> make_field_func: raw residual field (ENS-style, arXiv:2606.27354).
    kw are forwarded (n, re, dt, std, mean, dtype)."""
    if signal == "gradient":
        return make_dx_func(**kw)
    if signal == "field":
        return make_field_func(**kw)
    raise ValueError(f"unknown cond_signal {signal!r} (expected 'gradient' or 'field')")


def make_residual_loss(n=256, re=1000.0, dt=1.0 / 32.0, std=4.7988, mean=0.0, dtype=jnp.float32):
    """Return a jitted fn(x_norm) -> (...,): the PER-SAMPLE mean PDE residual^2 on the
    de-normalized field (the scalar the guidance minimizes). Use to log how the equation
    residual evolves over the denoising steps, for guided AND baseline (lambda=0) runs."""
    residual = make_ns_residual(n, re, dt, dtype)

    def loss(x_norm):
        w = x_norm * std + mean
        return jnp.mean(residual(w) ** 2, axis=(-2, -1))    # mean over space, per sample
    return jax.jit(loss)
