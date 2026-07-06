"""DDPO reward functions for physics-aware finetuning of the SR diffusion model.

Design principles (see base_results/reward_calibration.ipynb for the calibration study):

1. REFERENCE-FREE w.r.t. paired GT. Every reward compares the sample against *regime-level
   statistics* (target spectrum, total enstrophy, vorticity quantiles) or against the PDE itself
   (NS residual). No per-frame GT is needed at reward time. This is what lets the same rewards
   drive finetuning toward regimes where no paired HR data exists, and it prevents the reward
   from collapsing back into the MSE objective whose high-k blindness we are trying to fix.

2. PER-SAMPLE SCALARS on normalized triplets. All fns map x (..., N, N, 3) -> (...,), operating
   on the model's normalized output (mean 0, std 4.7988 for the Re=1000 base), the same
   convention as physics_guidance and the inference pkls. DDPO needs only scalar rewards (no
   gradients through the reward), but everything here is jax-differentiable anyway.

3. DISTANCES, not rewards. Each factory returns d(x) >= 0, lower = better. `make_ddpo_reward`
   combines them into a single reward r = -sum_i w_i * d_i / s_i, with scales s_i taken from the
   calibration run (per-component spread over a reference population) so the weights w_i are
   comparable knobs. DDPO's per-batch advantage normalization removes any remaining affine slack.

Spectral conventions match docs/energy_spectrum_metric.md and lambda_sweep_re1000.spectrum_fn
verbatim: integer fftfreq wavenumbers on (0,2pi)^2, round() shell binning, vorticity-power
(enstrophy) spectrum of the normalized MIDDLE frame, shells k = 0..KMAX-1 (KMAX = N/2).
Shells k >= ~96 are excluded by default (GT energy -> 0 there; log-ratios blow up).
"""
import json

import jax
import jax.numpy as jnp
import numpy as np

from src.physics_guidance import make_residual_loss


# ---------------------------------------------------------------- spectrum machinery

def _shell_index(n, dtype=jnp.int32):
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    kr = jnp.round(jnp.sqrt(k[:, None] ** 2 + k[None, :] ** 2))
    return kr.astype(dtype).ravel()


def make_spectrum_fn(n=256):
    """Return spectrum(x) -> (..., n//2): vorticity-power spectrum of the middle frame of a
    normalized triplet (..., n, n, 3). Identical convention to lambda_sweep_re1000.spectrum_fn."""
    kr = _shell_index(n)
    kmax = n // 2

    def spectrum(x):
        p = jnp.abs(jnp.fft.fft2(x[..., 1], axes=(-2, -1))) ** 2
        batch = p.shape[:-2]
        p = p.reshape((-1, n * n))
        e = jax.vmap(lambda q: jax.ops.segment_sum(q, kr, num_segments=n))(p)
        return e[:, :kmax].reshape(batch + (kmax,))

    return jax.jit(spectrum)


# ---------------------------------------------------------------- regime statistics

def compute_regime_stats(fields_norm, n=256, n_quantiles=257, batch=64):
    """Precompute the reference statistics a regime's rewards are anchored to, from ANY sample of
    that regime's HR fields (train split, a generated run at the target Re, ...).

    fields_norm: (M, n, n) NORMALIZED vorticity frames (numpy ok). Returns a plain dict of numpy
    arrays (json/npz-friendly):
      spec_ref       (n//2,)  arithmetic-mean vorticity-power spectrum E_ref(k)
      log_spec_ref   (n//2,)  mean of log E(k) — the GEOMETRIC-mean anchor. Frame spectra are
                              roughly log-normal per shell, so the arithmetic mean sits above the
                              typical frame and a slightly *blurred* frame can look closer to it
                              than GT does (calibration finding); the geometric mean puts the
                              typical frame at ~zero distance.
      enstrophy_ref  ()       mean total enstrophy  mean_x(w^2)
      quantiles_ref  (Q,)     mean vorticity quantiles (W1 anchor)
    """
    fields_norm = np.asarray(fields_norm, np.float32)
    spec_fn = make_spectrum_fn(n)
    qs = np.linspace(0.0, 1.0, n_quantiles)
    specs, quants, enst = [], [], []
    for i in range(0, len(fields_norm), batch):
        f = fields_norm[i : i + batch]
        tri = np.repeat(f[..., None], 3, axis=-1)          # middle-frame convention
        specs.append(np.asarray(spec_fn(tri)))
        quants.append(np.quantile(f.reshape(len(f), -1), qs, axis=1).T)
        enst.append((f ** 2).mean(axis=(1, 2)))
    specs = np.concatenate(specs)
    return {
        "spec_ref": specs.mean(axis=0),
        "log_spec_ref": np.log(specs + 1e-20).mean(axis=0),
        "enstrophy_ref": float(np.concatenate(enst).mean()),
        "quantiles_ref": np.concatenate(quants).mean(axis=0),
    }


def save_regime_stats(stats, path):
    np.savez(path, **stats)


def load_regime_stats(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- distance components

def make_spectrum_distance(spec_ref, kband=(1, 96), n=256, log_ref=None, rel_floor=1e-6):
    """d_spec(x): mean squared log-ratio of the sample's spectrum to the regime reference over the
    shell band [kband[0], kband[1]). Log-space so every decade of the enstrophy cascade counts —
    this is the component that *sees* the high-k deficit MSE is blind to.

    log_ref: per-shell mean log-spectrum (stats['log_spec_ref']) — preferred anchor (geometric
    mean; see compute_regime_stats). Falls back to log(spec_ref).
    rel_floor: sample shells are floored at rel_floor * reference so a hard-zeroed shell
    contributes a bounded (log rel_floor)^2 instead of an epsilon-driven blowup."""
    spec_fn = make_spectrum_fn(n)
    lo, hi = kband
    lref = jnp.asarray((log_ref if log_ref is not None else np.log(spec_ref + 1e-20))[lo:hi],
                       jnp.float32)
    floor = jnp.exp(lref) * rel_floor

    def d(x):
        e = jnp.maximum(spec_fn(x)[..., lo:hi], floor)
        return jnp.mean((jnp.log(e) - lref) ** 2, axis=-1)

    return jax.jit(d)


def make_energy_distance(enstrophy_ref):
    """d_E(x): squared log-ratio of total enstrophy (mean w^2 of the middle frame) to the regime
    reference. The 'energy conservation' term — scale-blind, so it complements d_spec (which fixes
    the distribution across scales) by pinning the overall level."""
    log_ref = jnp.log(jnp.float32(enstrophy_ref))

    def d(x):
        e = jnp.mean(x[..., 1] ** 2, axis=(-2, -1))
        return (jnp.log(e + 1e-20) - log_ref) ** 2

    return jax.jit(d)


def make_vorticity_w1_distance(quantiles_ref):
    """d_W1(x): 1-Wasserstein distance between the sample's pointwise vorticity distribution and
    the regime reference, via quantile matching on the sorted middle frame. Sensitive to the
    PDF tails (extreme-vorticity filaments) that over-smoothing clips."""
    q_ref = jnp.asarray(quantiles_ref, jnp.float32)
    nq = q_ref.shape[0]

    def d(x):
        v = x[..., 1].reshape(x.shape[:-3] + (-1,))
        v = jnp.sort(v, axis=-1)
        idx = jnp.round(jnp.linspace(0, v.shape[-1] - 1, nq)).astype(jnp.int32)
        return jnp.mean(jnp.abs(v[..., idx] - q_ref), axis=-1)

    return jax.jit(d)


def make_pde_residual_distance(n=256, re=1000.0, dt=1.0 / 32.0, std=4.7988, mean=0.0,
                               residual_ref=None):
    """d_pde(x): mean NS residual^2 of the de-normalized triplet (physics_guidance conventions,
    evaluated at the TARGET regime's Re). If residual_ref is given (the GT residual level for the
    regime — nonzero: ~12.9 at Re=1000 from discretization), returns the squared log-ratio to it
    instead of the raw value, so 'as consistent as the data itself' is the optimum rather than an
    unattainable 0. The only component that checks *dynamics* (phases), not statistics — it is
    what catches spectrum-matched noise."""
    loss_fn = make_residual_loss(n=n, re=re, dt=dt, std=std, mean=mean)
    if residual_ref is None:
        return loss_fn
    log_ref = jnp.log(jnp.float32(residual_ref))

    def d(x):
        return (jnp.log(loss_fn(x) + 1e-20) - log_ref) ** 2

    return jax.jit(d)


# ---------------------------------------------------------------- combined DDPO reward

DEFAULT_WEIGHTS = {"spec": 1.0, "energy": 1.0, "w1": 1.0, "pde": 1.0}


def make_ddpo_reward(stats, re, weights=None, scales=None, kband=(1, 96), highk_band=(32, 96),
                     n=256, dt=1.0 / 32.0, std=4.7988, mean=0.0, residual_ref=None):
    """Build the combined DDPO reward for one regime.

    stats: dict from compute_regime_stats (or load_regime_stats) for the TARGET regime.
    scales: per-component normalization s_i (std of d_i over a reference population, from the
            calibration notebook); None -> 1.0 each (raw distances).
    Returns reward(x) -> (r, components): r = -sum w_i d_i / s_i (higher = better), components a
    dict of the raw per-sample distances for logging."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    scales = scales or {}
    lref = stats.get("log_spec_ref")
    fns = {
        "spec": make_spectrum_distance(stats["spec_ref"], kband=kband, n=n, log_ref=lref),
        "spec_highk": make_spectrum_distance(stats["spec_ref"], kband=highk_band, n=n, log_ref=lref),
        "energy": make_energy_distance(stats["enstrophy_ref"]),
        "w1": make_vorticity_w1_distance(stats["quantiles_ref"]),
        "pde": make_pde_residual_distance(n=n, re=re, dt=dt, std=std, mean=mean,
                                          residual_ref=residual_ref),
    }

    def reward(x):
        components = {k: f(x) for k, f in fns.items()}
        r = -sum(weights[k] * components[k] / scales.get(k, 1.0)
                 for k in weights if weights.get(k, 0.0) != 0.0)
        return r, components

    return reward


def load_calibration(path):
    """Load the scales/weights json written by the calibration notebook."""
    with open(path) as f:
        return json.load(f)
