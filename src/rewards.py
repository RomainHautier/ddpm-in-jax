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

from src.physics_guidance import make_ns_residual, make_residual_loss


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
                               residual_ref=None, hinge=False):
    """d_pde(x): mean NS residual^2 of the de-normalized triplet (physics_guidance conventions,
    evaluated at the TARGET regime's Re). If residual_ref is given (the GT residual level for the
    regime — nonzero: ~12.9 at Re=1000 from discretization), returns the squared log-ratio to it
    instead of the raw value, so 'as consistent as the data itself' is the optimum rather than an
    unattainable 0. The only component that checks *dynamics* (phases), not statistics — it is
    what catches spectrum-matched noise.

    hinge=True: ONE-SIDED penalty — only residual ABOVE the floor is penalized,
    max(0, ln(pde) - ln(ref))^2. This removes pde's resistance to a sample carrying MORE small-scale
    energy (which, if physical, keeps the residual near the floor); it still fires on residual that
    rises above the floor (un-physical / wrong-phase / noise). Intended to let DDPO add legitimate
    high-k energy the two-sided form was fighting (the spec-vs-pde plateau)."""
    loss_fn = make_residual_loss(n=n, re=re, dt=dt, std=std, mean=mean)
    if residual_ref is None:
        return loss_fn
    log_ref = jnp.log(jnp.float32(residual_ref))

    def d(x):
        lr = jnp.log(loss_fn(x) + 1e-20) - log_ref
        return jnp.maximum(lr, 0.0) ** 2 if hinge else lr ** 2

    return jax.jit(d)


def make_pde_local(n=256, re=1000.0, dt=1.0 / 32.0, std=4.7988, mean=0.0, frac=0.1,
                   residual_ref=None, patch=1):
    """d_pde_local(x): the SPATIALLY-LOCALIZED NS residual — magnitude-weighted mean residual^2 over
    the worst-violating `frac` fraction of the field (default top 10%), instead of the whole-field
    mean. Concentrates the physics penalty where the equation is most violated, so DDPO preferentially
    fixes those regions (a GT-free localizer — the residual field correlates ~+0.23 with the actual
    reconstruction error; see viz_pde). Because it targets the highest-residual sites, it drives
    cleanup of spurious / wrong-phase structure (which raises the residual), complementary to the
    spectral terms that add the missing high-k energy. residual_ref -> squared log-ratio to a floor.

    patch (P): pool residual^2 into PxP block-means (block residual DENSITY) before selecting the
    worst `frac`, so the target is the worst PxP REGIONS rather than isolated pixels. P=1 = pixel
    (default, back-compat). The residual's correlation length is ~3px (see diag_residual_locality),
    so P=2..4 matches its natural scale and averages out single-pixel discretization noise; P>=8
    over-coarsens (a 2x2 block keeps ~76% of the residual structure, 8x8 only ~25%)."""
    residual = make_ns_residual(n=n, re=re, dt=dt)

    def _block_mean(a, P):                                   # (..., n, n) -> (..., n//P, n//P) block-mean
        s = a.shape[:-2] + (a.shape[-2] // P, P, a.shape[-1] // P, P)
        return a.reshape(s).mean(axis=(-3, -1))

    def metric(x_norm):
        w = x_norm * std + mean
        r2 = residual(w) ** 2                                # (..., n, n) per-pixel residual^2
        if patch > 1:
            r2 = _block_mean(r2, patch)                      # (..., n/P, n/P) per-region residual density
        thresh = jnp.quantile(r2, 1.0 - frac, axis=(-2, -1), keepdims=True)
        mask = r2 >= thresh
        return jnp.sum(r2 * mask, axis=(-2, -1)) / (jnp.sum(mask, axis=(-2, -1)) + 1e-8)

    if residual_ref is None:
        return jax.jit(metric)
    log_ref = jnp.log(jnp.float32(residual_ref))

    def d(x):
        return (jnp.log(metric(x) + 1e-20) - log_ref) ** 2

    return jax.jit(d)


def make_alignment_distance(n=256, std=4.7988, mean=0.0, align_ref=0.2887, align_scale=0.105,
                            L=2 * np.pi):
    """d_align(x): squared normalized gap between the sample's small-scale ORIENTATION statistic and
    the GT reference. a(x) = <cos^2(angle(grad omega, compressive strain axis))> weighted by
    |grad omega|^2, from the middle frame (spectral derivatives, streamfunction velocity).

    Physics (Batchelor filament dynamics): vorticity filaments are stretched along the extensional
    strain axis, so their gradients lock to a preferred orientation relative to the LOCAL strain
    eigenframe — and the strain field is LARGE-SCALE, i.e. resolved by the recon (corr ~0.82 to GT,
    diag_strain). GT sits at a(x) ~ 0.289 (strongly organized; isotropic = 0.5); the base recon at
    ~0.394 — the high-k energy DDPO adds is orientation-RANDOM speckle. Penalizing the gap demands
    the added energy form coherent, strain-consistent filaments: a GT-free structural constraint the
    spectral terms are blind to (they fix HOW MUCH energy; this fixes HOW it is organized).
    align_ref/align_scale from diag_strain at Re=1000 (ref = GT value, scale = |base - GT| so the
    base recon scores d ~ 1)."""
    k = np.fft.fftfreq(n) * n * (2 * np.pi / L)
    KX, KY = jnp.asarray(k[None, :]), jnp.asarray(k[:, None])
    K2 = (KX ** 2 + KY ** 2).at[0, 0].set(1.0)

    def metric(x_norm):
        w = x_norm[..., 1] * std + mean                    # middle frame (B, n, n) physical vorticity
        wh = jnp.fft.fft2(w)
        psih = (wh / K2).at[..., 0, 0].set(0.0)            # streamfunction
        u = jnp.real(jnp.fft.ifft2(1j * KY * psih))        # u = psi_y
        v = jnp.real(jnp.fft.ifft2(-1j * KX * psih))       # v = -psi_x
        uh, vh = jnp.fft.fft2(u), jnp.fft.fft2(v)
        ux = jnp.real(jnp.fft.ifft2(1j * KX * uh)); uy = jnp.real(jnp.fft.ifft2(1j * KY * uh))
        vx = jnp.real(jnp.fft.ifft2(1j * KX * vh)); vy = jnp.real(jnp.fft.ifft2(1j * KY * vh))
        theta_c = 0.5 * jnp.arctan2(vx + uy, ux - vy) + jnp.pi / 2   # compressive strain axis
        wx = jnp.real(jnp.fft.ifft2(1j * KX * wh)); wy = jnp.real(jnp.fft.ifft2(1j * KY * wh))
        g2 = wx ** 2 + wy ** 2                             # |grad omega|^2 weight
        c2 = jnp.cos(jnp.arctan2(wy, wx) - theta_c) ** 2
        return jnp.sum(g2 * c2, axis=(-2, -1)) / (jnp.sum(g2, axis=(-2, -1)) + 1e-20)

    def d(x):
        return ((metric(x) - align_ref) / align_scale) ** 2

    return jax.jit(d)


def make_spec_residual_distance(n=256, re=1000.0, dt=1.0 / 32.0, std=4.7988, mean=0.0,
                                kband=(32, 96), ref_power=None):
    """d_spec_residual(x): mean POWER of the NS-residual field in the band [kband).

    HONEST STATUS (measured 2026-07-09, grid-4x Re=1000): the recon's high-k (k>=32) residual power
    is AT/BELOW GT's (base 1.26e6 vs GT 2.27e6) — there is NO high-k speckle exceeding GT, so the
    default band is NOT the recon's problem. The recon's residual excess is at LARGE scales (k<8,
    2x GT power): a temporal-spatial BALANCE error (the wt term fails to cancel advection by ~2%,
    vs GT's 10^4 cancellation) — attack that via the plain `pde` term (Parseval-dominated by k<8),
    not this component. Kept for band-limited residual experiments (kband is configurable).

    ref_power (GT band residual-power floor) -> hinged squared log-ratio: penalize only ABOVE the
    floor. None -> raw mean band residual power."""
    residual = make_ns_residual(n=n, re=re, dt=dt)
    k = np.fft.fftfreq(n) * n
    kmag = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    mask = jnp.asarray(((kmag >= kband[0]) & (kmag < kband[1])).astype(np.float32))
    denom = float(mask.sum())

    def metric(x_norm):
        w = x_norm * std + mean
        R = residual(w)                                      # (..., n, n) residual field
        P = jnp.abs(jnp.fft.fft2(R)) ** 2                    # residual power
        return jnp.sum(P * mask, axis=(-2, -1)) / denom      # (...,) mean high-k residual power

    if ref_power is None:
        return jax.jit(metric)
    log_ref = jnp.log(jnp.float32(ref_power))

    def d(x):
        return jnp.maximum(jnp.log(metric(x) + 1e-8) - log_ref, 0.0) ** 2   # hinge above GT floor

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
