"""Reference reward implementation for DDPO finetuning (Claude reference — sibling to rewards.py).

Self-contained "as it should be" version: the five standalone jitted component builders, a `Reward`
class that assembles them from a config and exposes `__call__(x0) -> (reward, components)`, and the
per-input advantage helper the PPO loop needs. Delegates the actual math to the CALIBRATED
`src/rewards.py` factories, so this and the calibration study never diverge.

Contract (docs/ddpo_reward_math.pdf, reward_calibration):
  * distances on the FINAL sample x0 : (B, N, N, 3), per-sample (B,), >= 0, lower = better.
  * reward = - sum_i w_i d_i / s_i  (higher = better); components dict returned for hacking-watch.
  * the per-input advantage baseline is applied by the PPO loop (`per_input_advantage`), NOT baked
    into the reward — it needs the K-samples-per-input grouping that cancels the anchor offset.

Run-as-script from repo root (the hyphen in `src/ddpo-ft` blocks `import`); the sys.path shim makes
`from src.rewards import ...` resolve.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402

from src.rewards import (                         # noqa: E402
    load_calibration,
    load_regime_stats,
    make_alignment_distance,
    make_energy_distance,
    make_pde_local,
    make_pde_residual_distance,
    make_spec_residual_distance,
    make_spectrum_distance,
    make_vorticity_w1_distance,
)

# GT high-k (k>=32) residual-power floors, measured from the regime GT (diag: kf_re*.npy).
SPEC_RESID_FLOOR = {1000: 1.1377e6, 2000: 3.1050e6, 500: 4.0e5}

STD, MEAN, DT, N = 4.7988, 0.0, 1.0 / 32.0, 256
FULL_BAND, HIGHK_BAND = (1, 96), (32, 96)
DEFAULT_WEIGHTS = {"spec": 0.5, "spec_highk": 1.0, "energy": 0.25, "w1": 0.25, "pde": 1.0,
                   "pde_local": 0.0, "align": 0.0, "spec_residual": 0.0}


# ---------------------------------------------------------------- component builders
# Each returns a jitted fn(x) -> (B,) per-sample distance. Build ONCE, call every PPO step.

def component_spectrum(stats, kband=FULL_BAND, n=N):
    """d_spec: mean squared log-ratio of the sample enstrophy spectrum to the regime anchor over
    [1, 96). Steering signal (geometric-mean anchor via log_spec_ref)."""
    return make_spectrum_distance(stats["spec_ref"], kband=kband, n=n, log_ref=stats.get("log_spec_ref"))


def component_spectrum_highk(stats, highk_band=HIGHK_BAND, n=N):
    """d_spec_highk: same over the deficit band k >= 32 — the sharpest small-scale steering signal."""
    return make_spectrum_distance(stats["spec_ref"], kband=highk_band, n=n, log_ref=stats.get("log_spec_ref"))


def component_energy(stats):
    """d_energy: squared log-ratio of total enstrophy to the anchor. Scale-blind level guard."""
    return make_energy_distance(stats["enstrophy_ref"])


def component_vorticity_w1(stats):
    """d_w1: 1-Wasserstein (inverse-CDF) distance of the vorticity marginal to the anchor."""
    return make_vorticity_w1_distance(stats["quantiles_ref"])


def component_pde_residual(re, residual_ref=None, n=N, std=STD, mean=MEAN, dt=DT, hinge=False):
    """d_pde / d_pde_lr: NS residual^2 at the regime Re — the anti-hack (dynamics) term. Pass
    residual_ref (the regime GT residual floor) for the log-ratio form d_pde_lr (recommended).
    hinge=True: one-sided (penalize residual only ABOVE the floor) — lets DDPO add legitimate
    high-k energy without pde resisting it (the spec-vs-pde plateau fix)."""
    return make_pde_residual_distance(n=n, re=float(re), dt=dt, std=std, mean=mean,
                                      residual_ref=residual_ref, hinge=hinge)


def component_pde_local(re, frac=0.1, residual_ref=None, patch=1, n=N, std=STD, mean=MEAN, dt=DT):
    """d_pde_local: SPATIALLY-LOCALIZED residual — magnitude-weighted mean residual^2 over the
    worst-violating `frac` of the field, concentrating the physics penalty where the equation is most
    violated (GT-free error localizer, ~+0.23 corr with the true error). Drives cleanup of
    spurious/wrong-phase structure; complements the spectral terms (which add missing high-k energy).
    patch (P): pool into PxP block-means before selecting the worst `frac` -> target worst REGIONS not
    pixels. P=1 pixel (default); P=2..4 matches the residual's ~3px correlation length (diag)."""
    return make_pde_local(n=n, re=float(re), dt=dt, std=std, mean=mean, frac=frac,
                          residual_ref=residual_ref, patch=patch)


def component_spec_residual(re, ref_power=None, kband=HIGHK_BAND, n=N, std=STD, mean=MEAN, dt=DT):
    """d_spec_residual: high-k (k>=32) NS-residual POWER above the GT floor — kills the residual
    speckle DDPO adds (physically-inconsistent fine structure) while the spectral energy terms keep
    the enstrophy. ref_power defaults to the measured GT floor for `re`. GT is the residual minimum
    so this pulls toward the physical solution, not below it (see diag_residual_landscape)."""
    ref = ref_power if ref_power is not None else SPEC_RESID_FLOOR.get(int(re))
    return make_spec_residual_distance(n=n, re=float(re), dt=dt, std=std, mean=mean,
                                       kband=kband, ref_power=ref)


def component_align(align_ref=0.2887, align_scale=0.105, n=N, std=STD, mean=MEAN):
    """d_align: small-scale ORIENTATION statistic vs the GT reference — demands that added high-k
    energy form coherent strain-locked filaments (GT ~0.289, base recon ~0.394, isotropic 0.5) rather
    than orientation-random speckle. GT-free at reward time; ref/scale measured once from owned
    Re=1000 GT (diag_strain). Complements the spectral terms: they fix amount, this fixes structure."""
    return make_alignment_distance(n=n, std=std, mean=mean, align_ref=align_ref, align_scale=align_scale)


_STAT_BUILDERS = {
    "spec": component_spectrum,
    "spec_highk": component_spectrum_highk,
    "energy": component_energy,
    "w1": component_vorticity_w1,
}
COMPONENT_NAMES = (*_STAT_BUILDERS.keys(), "pde", "pde_local", "align", "spec_residual")


def build_components(stats, re, names=COMPONENT_NAMES, residual_ref=None, n=N,
                     std=STD, mean=MEAN, dt=DT, pde_hinge=False, pde_local_frac=0.1, pde_local_patch=1):
    """{name: jitted fn(x)->(B,)} for the requested components. `pde`/`pde_local` need `re`.
    pde_hinge=True -> one-sided pde. pde_local -> localized (worst-`frac`) residual; pde_local_patch
    (P) targets worst PxP regions instead of pixels (P=1 default; P=2..4 matches the ~3px scale)."""
    fns = {}
    for nm in names:
        if nm in _STAT_BUILDERS:
            fns[nm] = _STAT_BUILDERS[nm](stats)
        elif nm == "pde":
            fns[nm] = component_pde_residual(re, residual_ref=residual_ref, n=n, std=std, mean=mean,
                                             dt=dt, hinge=pde_hinge)
        elif nm == "pde_local":
            fns[nm] = component_pde_local(re, frac=pde_local_frac, residual_ref=residual_ref,
                                          patch=pde_local_patch, n=n, std=std, mean=mean, dt=dt)
        elif nm == "align":
            fns[nm] = component_align(n=n, std=std, mean=mean)
        elif nm == "spec_residual":
            fns[nm] = component_spec_residual(re, n=n, std=std, mean=mean, dt=dt)
        else:
            raise ValueError(f"unknown reward component {nm!r} (known: {COMPONENT_NAMES})")
    return fns


# ---------------------------------------------------------------- the Reward class

class Reward:
    """Assembled DDPO reward for one regime.

    reward(x0) -> (r, components):  r = - sum_i w_i d_i / s_i  (higher = better),  components a dict
    of the raw per-sample distances (log these to watch for reward hacking). Build once; jitted
    components recompile on first call, then run on the batch each PPO step.

    Args:
      stats: regime-stats dict (load_regime_stats / compute_regime_stats) for the TARGET regime.
      re: target regime Reynolds number (PDE residual).
      weights: {name: w_i}; defaults to the calibrated starting weights.
      scales: {name: s_i} per-Re normalization (reward_calibration.json). None -> 1.0 each.
      names: which components to include.
      residual_ref: regime GT residual floor -> enables d_pde_lr (recommended).
    """

    def __init__(self, stats, re, weights=None, scales=None, names=COMPONENT_NAMES,
                 residual_ref=None, n=N, std=STD, mean=MEAN, dt=DT, pde_hinge=False, pde_local_frac=0.1,
                 pde_local_patch=1):
        self.re = float(re)
        self.names = tuple(names)
        self.component_fns = build_components(stats, re, names=names, residual_ref=residual_ref,
                                              n=n, std=std, mean=mean, dt=dt, pde_hinge=pde_hinge,
                                              pde_local_frac=pde_local_frac, pde_local_patch=pde_local_patch)
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.scales = dict(scales or {})

    def __call__(self, x0):
        components = {k: f(x0) for k, f in self.component_fns.items()}
        r = -sum(self.weights.get(k, 0.0) * components[k] / self.scales.get(k, 1.0)
                 for k in self.component_fns if self.weights.get(k, 0.0) != 0.0)
        return r, components

    @classmethod
    def from_calibration(cls, stats_path, calib_path, re, names=COMPONENT_NAMES, weights=None,
                         pde_hinge=False, scales_re=None, residual_ref=None,
                         pde_local_frac=0.1, pde_local_patch=1):
        """Build from the on-disk artifacts: a regime_stats_re{Re}.npz and reward_calibration.json.
        Pulls the per-Re `scales` and the regime `residual_ref` from the calibration json, and uses
        d_pde_lr (log-ratio) mode. `weights` overrides the json/default weights if given.
        pde_hinge=True -> one-sided pde (penalize residual only above the floor).

        OOD / extrapolation overrides (so a run can touch ZERO target-regime data):
          scales_re: pull the per-component `scales` from THIS regime instead of `re` (e.g. reuse the
                     owned Re=1000 normalizers for an extrapolated Re=2000 run).
          residual_ref: override the pde floor (e.g. an extrapolated floor) instead of the json's
                     measured `re` value. If the stats npz carries its own `residual_ref`, that is
                     used when this arg is None."""
        stats = load_regime_stats(stats_path)
        calib = load_calibration(calib_path)
        regime = calib["regimes"][str(int(re))]
        sc_regime = calib["regimes"][str(int(scales_re))] if scales_re is not None else regime
        scales = dict(sc_regime.get("scales") or {})
        # in log-ratio (pde_lr) mode the pde scale to use is the json's "pde_lr" entry
        if "pde_lr" in scales and "pde" in names:
            scales = {**scales, "pde": scales["pde_lr"]}
        # residual floor: explicit override > stats-file's own > json regime value
        r_ref = residual_ref
        if r_ref is None and "residual_ref" in stats:
            r_ref = float(stats["residual_ref"])
        if r_ref is None:
            r_ref = regime.get("residual_ref")
        return cls(stats, re, weights=weights or calib.get("weights"),
                   scales=scales, names=names, residual_ref=r_ref,
                   std=calib.get("std", STD), mean=calib.get("mean", MEAN), pde_hinge=pde_hinge,
                   pde_local_frac=pde_local_frac, pde_local_patch=pde_local_patch)


# ---------------------------------------------------------------- per-input advantage (PPO helper)

def per_input_advantage(rewards, group_size, eps=1e-8, normalize_std=True):
    """DDPO/GRPO advantage: center (and optionally scale) each input's group of K samples.

    rewards: (B,) with B = n_inputs * group_size, laid out **input-major**:
        [ input0 sample0..K-1,  input1 sample0..K-1,  ... ]   (see rollout: tile each input K times).
    Returns A: (B,) with A_ij = (r_ij - mean_k r_ik) / (std_k r_ik + eps).

    The per-input mean subtraction is the load-bearing step: it cancels the regime-anchor offset
    (the 0.11 -> -0.74 correlation jump), so the advantage reflects sample quality, not which input
    it came from. Dividing by the group std equalizes the effective learning rate across inputs.
    Treat the result as a CONSTANT in the loss (stop_gradient in the PPO update).
    """
    xp = jnp if isinstance(rewards, jnp.ndarray) else np
    r = rewards.reshape(-1, group_size)                       # (n_inputs, K)
    centered = r - r.mean(axis=1, keepdims=True)
    if normalize_std:
        centered = centered / (r.std(axis=1, keepdims=True) + eps)
    return centered.reshape(-1)


__all__ = [
    "component_spectrum", "component_spectrum_highk", "component_energy",
    "component_vorticity_w1", "component_pde_residual", "build_components",
    "Reward", "per_input_advantage", "COMPONENT_NAMES", "DEFAULT_WEIGHTS",
]
