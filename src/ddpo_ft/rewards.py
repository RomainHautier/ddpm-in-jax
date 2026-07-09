# --- original design notes (kept from the stub; answered in the module docstring below) ---
# 1. Reward model specification  2. reward components as standalone functions
# - weighting: fixed weights first, then a secondary (anti-hacking) objective rather than pure
#   reward-max; weights may vary across regimes.  - must be jittable (denoising is jitted).
#   - class so attributes (components) can be added/removed easily.
"""Standalone reward-component builders for DDPO finetuning.

Each `component_*` returns a JITTED function  x -> (B,)  mapping a batch of normalized vorticity
triplets  x : (B, N, N, 3)  to a per-sample DISTANCE (>= 0, lower = better). They delegate to the
calibrated implementations in `src/rewards.py`, so the reward DDPO optimizes and the reward the
calibration study validated can never drift apart (single source of truth).

Answers to the stub questions:
  * Class vs functions: components are standalone jitted FACTORIES here; assemble them in your
    Reward CLASS (build once, hold weights/scales, expose __call__(x0) -> (reward, components)).
    `combine()` below is the minimal version of that formula.
  * Jittable: yes -- each builder returns a jitted closure; call it every PPO step on the batch.
  * Weighting: fixed weights first (calibration §6); `scales` from reward_calibration.json put
    components on comparable footing. Learning weights is a later rung.
  * Anti-hacking secondary objective: correct instinct, but it is a KL-to-base penalty in the PPO
    LOSS, not part of the reward. Two separate anti-hack layers: `pde`/`pde_lr` inside the reward
    (physics check), KL-to-base in the loss (stay near the base manifold).

Design contract (see docs/ddpo_reward_math.pdf + reward_calibration for the why):
  * Distances on the FINAL denoised sample x0 only. No trajectory, no paired GT.
  * Combine as  r = - sum_i w_i * d_i / s_i  (higher = better).
  * The per-input advantage baseline is NOT here -- it belongs in the PPO loop, which groups the K
    samples per input; that grouping cancels the regime-anchor offset (the 0.11 -> -0.74 jump).
    Keep the reward a pure function of x.

Anchors come from a regime-stats dict (src.rewards.compute_regime_stats / load_regime_stats):
    spec_ref (K,) · log_spec_ref (K,) · enstrophy_ref () · quantiles_ref (Q,)
plus, for the regime's Re, an optional PDE residual floor `residual_ref` (enables d_pde_lr).

Note: this directory (`src/ddpo-ft`) has a hyphen and is NOT an importable package. Run as a script
from the repo root, or rename to `ddpo_ft` to `import` it. The sys.path shim makes
`from src.rewards import ...` work when this file is executed directly.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.rewards import (                     # noqa: E402  (path shim must precede import)
    load_calibration,
    load_regime_stats,
    make_energy_distance,
    make_pde_residual_distance,
    make_spectrum_distance,
    make_vorticity_w1_distance,
)

STD, MEAN, DT, N = 4.7988, 0.0, 1.0 / 32.0, 256
FULL_BAND, HIGHK_BAND = (1, 96), (32, 96)


# ---------------------------------------------------------------- component builders
# Each returns a jitted  fn(x) -> (B,)  per-sample distance. Build ONCE (they close over the anchor
# and jit-compile on first call), then call every PPO step on the batch of samples.

def component_spectrum(stats, kband=FULL_BAND, n=N):
    """d_spec: mean squared log-ratio of the sample's enstrophy spectrum to the regime anchor over
    the full resolved cascade [1, 96). Steering signal (sees the hi-k deficit MSE is blind to).
    Uses the geometric-mean anchor (log_spec_ref) when present."""
    return make_spectrum_distance(stats["spec_ref"], kband=kband, n=n, log_ref=stats.get("log_spec_ref"))


def component_spectrum_highk(stats, highk_band=HIGHK_BAND, n=N):
    """d_spec_highk: same, restricted to the deficit band k >= 32. Sharpest steering signal for the
    small-scale (high-wavenumber energy) reconstruction the base model over-smooths."""
    return make_spectrum_distance(stats["spec_ref"], kband=highk_band, n=n, log_ref=stats.get("log_spec_ref"))


def component_energy(stats):
    """d_energy: squared log-ratio of total enstrophy (mean_x x_mid^2) to the regime anchor. Scale-
    blind level guard; weak in-dist (dominated by the Re-invariant forcing peak). Small weight."""
    return make_energy_distance(stats["enstrophy_ref"])


def component_vorticity_w1(stats):
    """d_w1: 1-Wasserstein distance between the sample's pointwise vorticity CDF and the regime
    anchor, via quantile (inverse-CDF) matching. Weakest sensor on this task; drop candidate, or
    swap for a tail-weighted / |grad omega| variant later."""
    return make_vorticity_w1_distance(stats["quantiles_ref"])


def component_pde_residual(re, residual_ref=None, n=N, std=STD, mean=MEAN, dt=DT):
    """d_pde / d_pde_lr: mean NS residual^2 at the regime's Re (physics/dynamics check; the only
    anti-hack component -- fires on spectrum-matched noise). Pass `residual_ref` (the regime's GT
    residual floor) for the log-ratio form d_pde_lr, which is what you want: raw d_pde rewards
    smoothing, the log-ratio makes the GT floor the optimum. Leave None only for diagnostics."""
    return make_pde_residual_distance(n=n, re=float(re), dt=dt, std=std, mean=mean, residual_ref=residual_ref)


# ---------------------------------------------------------------- registry + assembly
# Name -> builder, so a Reward class can select components straight from a config/kwarg. `pde` needs
# `re` (+ optional residual_ref), so it is wrapped at assembly time in build_components.

_STAT_BUILDERS = {
    "spec": component_spectrum,
    "spec_highk": component_spectrum_highk,
    "energy": component_energy,
    "w1": component_vorticity_w1,
}
COMPONENT_NAMES = (*_STAT_BUILDERS.keys(), "pde")


def build_components(stats, re, names=COMPONENT_NAMES, residual_ref=None, n=N,
                     std=STD, mean=MEAN, dt=DT):
    """Build the requested components into a {name: jitted_fn(x)->(B,)} dict.

    stats: regime-stats dict for the TARGET regime (load_regime_stats / compute_regime_stats).
    re: the target regime's Reynolds number (for the PDE residual).
    names: which components to include (subset of COMPONENT_NAMES).
    residual_ref: the regime's GT residual floor -> enables d_pde_lr (recommended). None -> raw pde.

    The natural entry point for a Reward class: pass a config, get the component fns, then hold
    weights/scales alongside and combine. Keeps 'which components' a pure data decision.
    """
    fns = {}
    for nm in names:
        if nm in _STAT_BUILDERS:
            fns[nm] = _STAT_BUILDERS[nm](stats)
        elif nm == "pde":
            fns[nm] = component_pde_residual(re, residual_ref=residual_ref, n=n, std=std, mean=mean, dt=dt)
        else:
            raise ValueError(f"unknown reward component {nm!r} (known: {COMPONENT_NAMES})")
    return fns


def combine(component_fns, weights, scales=None):
    """Fold components into a single reward fn(x) -> (reward (B,), components dict).

    reward = - sum_i w_i * d_i / s_i   (higher = better).  scales s_i default to 1.0 (raw distances);
    pass the per-Re `scales` from reward_calibration.json to make weights comparable. Components with
    weight 0 are still evaluated and returned (for logging) but do not enter the reward.

    Mirrors src.rewards.make_ddpo_reward; provided here so a Reward class can build on the same
    formula. The returned `components` dict is what you log to watch for reward hacking.
    """
    scales = scales or {}

    def reward(x):
        components = {k: f(x) for k, f in component_fns.items()}
        r = -sum(weights.get(k, 0.0) * components[k] / scales.get(k, 1.0)
                 for k in component_fns if weights.get(k, 0.0) != 0.0)
        return r, components

    return reward


__all__ = [
    "component_spectrum", "component_spectrum_highk", "component_energy",
    "component_vorticity_w1", "component_pde_residual",
    "build_components", "combine", "COMPONENT_NAMES",
    "load_regime_stats", "load_calibration",
]
