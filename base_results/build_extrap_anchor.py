"""Build an EXTRAPOLATED Re=2000 reward anchor from {Re=500, Re=1000} ONLY — no Re=2000 data touched.

Emits base_results/regime_stats_re2000_extrap.npz in the exact regime_stats format
(spec_ref, log_spec_ref, enstrophy_ref, quantiles_ref) so train_claude can point at it directly.

Method (from base_results/anchor_study — commit 1a91dbf):
  * spectrum: fit logE(k) = logA - alpha*log(k/kf) - (k/k_d)^p on Re=500 & 1000; hold {logA,alpha,p}
    universal (mean of the two), shift k_d by the 2D enstrophy-cascade law k_d ~ Re^(1/2). Low-k
    (forcing, k<=kf) copied from the OWNED Re=1000 anchor.
  * enstrophy: SATURATED (drag-controlled, peak-dominated) -> hold at the Re=1000 level, do NOT grow.
  * vorticity PDF: universal normalized shape (quantiles / sqrt(enstrophy)) averaged over 500 & 1000,
    rescaled by sqrt(enstrophy_ext).

Prints the gap to the measured regime_stats_re2000.npz (for honesty — NOT used to build anything).
"""
import os
import json
import numpy as np
from scipy.optimize import curve_fit

os.chdir("/home/rhautier/ddpm-jax")
N, SIG, KF, KMAX = 256, 4.7988, 4, 128
_k = np.fft.fftfreq(N, 1.0 / N)
_KR = np.round(np.sqrt(_k[:, None] ** 2 + _k[None, :] ** 2)).astype(int).ravel()
QS = np.linspace(0.0, 1.0, 257)
REG = {500: "flow-data/kf_re500_256_20seed.npy", 1000: "flow-data/kf_2d_re1000_256_40seed.npy"}
BAND, HIK = slice(1, 96), slice(32, 96)


def frame_spectrum(f):
    P = (np.abs(np.fft.fft2(f)) ** 2).ravel()
    return np.bincount(_KR, P, minlength=N)[:KMAX]


def collect(path, n_seqs=18, f0=40, f1=312, fstride=4):
    arr = np.load(path, mmap_mode="r")
    S, Q, E = [], [], []
    for s in range(min(n_seqs, arr.shape[0])):
        for t in range(f0, min(f1, arr.shape[1]), fstride):
            f = np.asarray(arr[s, t], np.float32) / SIG
            S.append(frame_spectrum(f)); Q.append(np.quantile(f.ravel(), QS)); E.append(float((f ** 2).mean()))
    return np.asarray(S), np.asarray(Q), np.asarray(E)


def anchor(S, Q, E):
    return dict(logspec=np.log(S + 1e-20).mean(0), spec=S.mean(0), quant=Q.mean(0), enst=float(np.mean(E)))


print("collecting {500,1000} per-frame stats (NO Re=2000) ...", flush=True)
REF = {}
for re, path in REG.items():
    S, Q, E = collect(path)
    REF[re] = anchor(S, Q, E)
    print(f"  Re={re}: {len(S)} frames | enstrophy={REF[re]['enst']:.4f}", flush=True)


def model(k, logA, alpha, kd, p):
    return logA - alpha * np.log(k / KF) - (k / kd) ** p


KFIT = np.arange(KF + 2, 110)
def fit_regime(re):
    y = np.log(REF[re]["spec"][KFIT] + 1e-30)
    p0 = [np.log(REF[re]["spec"][KF + 2] + 1e-20), 1.0, 40.0, 2.0]
    popt, _ = curve_fit(model, KFIT.astype(float), y, p0=p0, maxfev=20000,
                        bounds=([-40, 0, 5, 0.5], [40, 4, 300, 6]))
    return dict(logA=popt[0], alpha=popt[1], kd=popt[2], p=popt[3])


FIT = {re: fit_regime(re) for re in (500, 1000)}
for re in (500, 1000):
    f = FIT[re]; print(f"  Re={re:>4}: logA={f['logA']:.2f} alpha={f['alpha']:.2f} k_d={f['kd']:.1f} p={f['p']:.2f}", flush=True)

# universal shape params + theory k_d ~ Re^0.5 (a-priori, no Re=2000)
alpha_ext = np.mean([FIT[500]["alpha"], FIT[1000]["alpha"]])
logA_ext = np.mean([FIT[500]["logA"], FIT[1000]["logA"]])
p_ext = np.mean([FIT[500]["p"], FIT[1000]["p"]])
kd2000 = FIT[1000]["kd"] * (2000 / 1000) ** 0.5
print(f"\n  k_d(2000) theory (Re^0.5) = {kd2000:.1f}  (from k_d(1000)={FIT[1000]['kd']:.1f})", flush=True)


def build_logspec(logA, alpha, kd, p):
    lg = np.full(KMAX, -40.0)
    kk = np.arange(KF + 1, KMAX)
    lg[kk] = model(kk.astype(float), logA, alpha, kd, p)
    lg[1:KF + 1] = REF[1000]["logspec"][1:KF + 1]     # forcing low-k from OWNED Re=1000
    return lg


log_spec_ref = build_logspec(logA_ext, alpha_ext, kd2000, p_ext).astype(np.float32)
spec_ref = np.exp(log_spec_ref).astype(np.float32)
enstrophy_ref = np.float32(REF[1000]["enst"])          # saturated -> hold at Re=1000
shape_ext = 0.5 * (REF[500]["quant"] / np.sqrt(REF[500]["enst"]) + REF[1000]["quant"] / np.sqrt(REF[1000]["enst"]))
quantiles_ref = (shape_ext * np.sqrt(float(enstrophy_ref))).astype(np.float32)

# ---- extrapolated PDE residual floor (owned Re=1000 floor scaled by hi-k enstrophy ratio) ----
# The floor is concave in Re (500:1.56, 1000:17.46 measured) so a 2-pt power law overshoots.
# Physically the residual floor tracks small-scale (hi-k) enstrophy content, which we HAVE
# extrapolated -> scale the OWNED Re=1000 floor by the extrapolated hi-k enstrophy ratio.
RESID_REF_RE1000 = 17.455413818359375          # owned, from reward_calibration.json Re=1000
hik = slice(32, 96)
hik_ratio = float(np.exp(log_spec_ref)[hik].sum() / np.exp(REF[1000]["logspec"])[hik].sum())
residual_ref_ext = np.float32(RESID_REF_RE1000 * hik_ratio)
print(f"\n  hi-k enstrophy ratio (extrap2000 / Re1000) = {hik_ratio:.3f}  "
      f"-> residual_ref_ext = {float(residual_ref_ext):.2f}", flush=True)

out = "base_results/regime_stats_re2000_extrap.npz"
np.savez(out, spec_ref=spec_ref, log_spec_ref=log_spec_ref,
         enstrophy_ref=enstrophy_ref, quantiles_ref=quantiles_ref, residual_ref=residual_ref_ext)
print(f"SAVED {out}  keys=['spec_ref','log_spec_ref','enstrophy_ref','quantiles_ref','residual_ref']", flush=True)

# ---- honesty check vs MEASURED Re=2000 (loaded only to report the gap) ----
def dspec(a, b, band): return float(np.mean((a[band] - b[band]) ** 2))
def w1(a, b): return float(np.mean(np.abs(a - b)))
try:
    m = np.load("base_results/regime_stats_re2000.npz")
    print("\n--- gap: EXTRAPOLATED vs MEASURED Re=2000 (measured only used for this check) ---")
    print(f"  d_spec (log anchor)  full[1,96) = {dspec(log_spec_ref, m['log_spec_ref'], BAND):.4f}   "
          f"hi-k[32,96) = {dspec(log_spec_ref, m['log_spec_ref'], HIK):.4f}")
    print(f"  enstrophy_ref  extrap={float(enstrophy_ref):.4f}  measured={float(m['enstrophy_ref']):.4f}  "
          f"(ratio {float(enstrophy_ref)/float(m['enstrophy_ref']):.3f})")
    print(f"  vorticity-PDF  W1(extrap, measured) = {w1(quantiles_ref, m['quantiles_ref']):.4f}")
    mc = json.load(open("base_results/reward_calibration.json"))["regimes"]["2000"] if os.path.exists(
        "base_results/reward_calibration.json") else None
    if mc:
        print(f"  residual_ref   extrap={float(residual_ref_ext):.2f}  measured={mc['residual_ref']:.2f}  "
              f"(ratio {float(residual_ref_ext)/mc['residual_ref']:.3f})   [naive powerlaw would ~overshoot 3x]")
    # naive baseline: just reuse Re=1000 anchor
    r1 = np.load("base_results/regime_stats_re1000.npz")
    print(f"  [naive use-Re1000 baseline]  d_spec hi-k = {dspec(r1['log_spec_ref'], m['log_spec_ref'], HIK):.4f}  "
          f"W1 = {w1(r1['quantiles_ref'], m['quantiles_ref']):.4f}")
except FileNotFoundError:
    print("  (measured Re=2000 file absent — skipping gap check)")
print("\nDONE")
