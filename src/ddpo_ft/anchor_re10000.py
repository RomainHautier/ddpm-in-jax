"""Blind GT-free anchor for Re=10000, built from Re=500 + Re=1000 ONLY.

LIMITATION (documented, not worked around): residual_ref is deliberately NOT supplied. The
calibration json's residual_ref lives in a different normalisation from the eval's mean|R|
(json Re=1000 = 17.46 vs eval GT mean|R| = 1.06), and its 500->1000 ratio implies an exponent
of 3.48 that extrapolates absurdly. Rather than fabricate a number in units we cannot verify,
the reward falls back to the Re=1000 entry (coherent with --scales_re 1000). Consequence: the
PDE floor is NOT regime-adapted for Re=10000. pde weight is 1.0 against spec_highk 3.0, so the
spectral terms dominate, but this is a real caveat on the pde component.

CONSTRUCTION INPUTS (strict):
  - Re=500 and Re=1000 full fields: spectrum-model fits (logA, alpha, k_d, p) + the empirical
    k_d ~ Re^q exponent measured BETWEEN them.
  - Nothing else. No Re=2000, no Re=10000 (neither GT nor low-res observations, which cannot see
    past their Nyquist k=32 anyway — verified 2026-07-20).
NOT USED: any Re=10000 field. A single report-only grading line at the end reads GT purely to
record how good the blind anchor turned out; it is computed AFTER the npz is written and feeds
back into nothing.

Method (matches the Re=2000 recipe that worked, minus the parts that did not):
  1. fit E(k) = A (k/k_f)^-alpha exp(-(k/k_d)^p) on Re=500 and Re=1000
  2. extrapolate: k_d(target) = k_d(1000) * (Re/1000)^q ; alpha/logA/p = mean of the two regimes
  3. UP-direction leave-one-regime-out bias correction (500->1000). The bias is direction-
     dependent (UP under-predicts, DOWN over-predicts) so the two-sided mean used for Re=2000
     cancels the very effect we need; we extrapolate UP, so we calibrate on UP.
  4. NO tail generosity: unvalidated (+0.014, inside +-0.04 seed noise) and it would compound the
     overshoot the UP correction already produces. Only evidence-backed components are kept.
  NO grid cap: the known regimes show no numerical floor below k=125, so a cap cannot be
  justified from data, and the extrapolation under-shoots the tail anyway.
"""
import numpy as np
from scipy.optimize import curve_fit
import os
os.chdir('/home/rhautier/ddpm-jax')
N, SIG, KF, KMAX = 256, 4.7988, 4, 128
_k = np.fft.fftfreq(N, 1.0/N)
_KR = np.round(np.sqrt(_k[:,None]**2 + _k[None,:]**2)).astype(int).ravel()
BANDS = [(1,10),(10,20),(20,32),(32,64),(64,96)]

def spec(path, seqs=range(0,10), fstride=8):
    a = np.load(path, mmap_mode='r'); S = []
    for s in seqs:
        if s >= a.shape[0]: continue
        for t in range(40, min(312, a.shape[1]), fstride):
            f = np.asarray(a[s,t], np.float32)/SIG
            S.append(np.bincount(_KR, (np.abs(np.fft.fft2(f))**2).ravel(), minlength=N)[:KMAX])
    return np.asarray(S).mean(0)

def model(k, logA, alpha, kd, p): return logA - alpha*np.log(k/KF) - (k/kd)**p
KFIT = np.arange(KF+2, 110)
def fit(E):
    y = np.log(E[KFIT] + 1e-30)
    o,_ = curve_fit(model, KFIT.astype(float), y, p0=[y[0],1.,40.,2.], maxfev=40000,
                    bounds=([-40,0,5,0.5],[40,4,400,6]))
    return dict(zip(('logA','alpha','kd','p'), o))

# ---- CONSTRUCTION: Re=500 and Re=1000 only ----------------------------------
E500, E1000 = spec('flow-data/kf_re500_256_20seed.npy'), spec('flow-data/kf_2d_re1000_256_40seed.npy')
F500, F1000 = fit(E500), fit(E1000)
q = np.log(F1000['kd']/F500['kd'])/np.log(2.0)
alpha_e = 0.5*(F500['alpha']+F1000['alpha'])
logA_e  = 0.5*(F500['logA'] +F1000['logA'])
p_e     = 0.5*(F500['p']    +F1000['p'])
print('CONSTRUCTION (blind — Re=500 + Re=1000 only)', flush=True)
print(f"  Re=500 : logA={F500['logA']:.3f} alpha={F500['alpha']:.3f} k_d={F500['kd']:.1f} p={F500['p']:.3f}", flush=True)
print(f"  Re=1000: logA={F1000['logA']:.3f} alpha={F1000['alpha']:.3f} k_d={F1000['kd']:.1f} p={F1000['p']:.3f}", flush=True)
print(f"  empirical exponent q={q:.3f}  (k_d ~ Re^q)", flush=True)

def build(kd, logA=logA_e, alpha=alpha_e, p=p_e, low_src=E1000):
    lg = np.full(KMAX, -40.0); kk = np.arange(KF+1, KMAX)
    lg[kk] = model(kk.astype(float), logA, alpha, kd, p)
    lg[0:KF+1] = np.log(low_src[0:KF+1] + 1e-30)     # forcing scales from the known regime
    return lg

# two-sided LOO bias correction (500<->1000 only)
def loo(target_E, target_true_fit, src_fit):
    kd_pred = src_fit['kd'] * (2.0 if target_true_fit is F1000 else 0.5)**q
    lg = build(kd_pred, src_fit['logA'], src_fit['alpha'], src_fit['p'], target_E)
    return np.array([float(np.exp(lg)[a:b].sum()/target_E[a:b].sum()) for a,b in BANDS])
r_up   = loo(E1000, F1000, F500)     # 500 -> 1000  (UP — the direction we actually use)
r_down = loo(E500,  F500,  F1000)    # 1000 -> 500  (DOWN — opposite regime, opposite bias)
print(f"  LOO bias (500->1000, UP):   " + ' '.join(f'{v:.3f}' for v in r_up), flush=True)
print(f"  LOO bias (1000->500, DOWN): " + ' '.join(f'{v:.3f}' for v in r_down), flush=True)
print(f"  two-sided geometric mean:   " + ' '.join(f'{v:.3f}' for v in 1.0/np.sqrt(r_up*r_down)), flush=True)
# The bias is DIRECTION-DEPENDENT: extrapolating UP under-predicts (0.65-0.94), DOWN over-predicts
# (1.01-1.43). The two-sided mean cancels precisely the effect we need. We extrapolate UP, so we
# use the UP bias. CAVEAT: calibrated on a 2x jump, applied to a 10x jump -> a LOWER BOUND on the
# correction if the bias grows with extrapolation distance (unmeasurable with only two regimes).
CORR = 1.0/r_up
print(f"  UP-direction correction used: " + ' '.join(f'{v:.3f}' for v in CORR), flush=True)

kd_10k = F1000['kd'] * (10000/1000.)**q
lg = build(kd_10k)
kk = np.arange(KMAX, dtype=float)
band_centers = np.array([5.5, 15., 26., 48., 80.])   # one per BAND
corr_k = np.interp(kk, band_centers, CORR, left=CORR[0], right=CORR[-1]); corr_k[:20] = 1.0
lg = lg + np.log(corr_k + 1e-30)
# TAIL GENEROSITY: DROPPED. It was carried through every winning Re=2000 run, but its measured
# benefit there was +0.014 — inside the +-0.04 seed-noise floor, i.e. never validated. Here it
# would compound an overshoot (the UP-direction correction already lifts the tail by ~1.5x), so
# the evidence-backed choice is to omit it and keep only components with measured support.
GEN = 1.0
lg = lg + np.log(np.interp(kk, [20,32,96], [1.0, GEN, GEN]))
print(f"  predicted k_d(10000) = {kd_10k:.1f}  (no grid cap — unjustifiable from data)", flush=True)

out = dict(np.load('base_results/regime_stats_re2000_extrap.npz'))   # template for the other keys
out['log_spec_ref'] = lg.astype(np.float32)
out['spec_ref'] = np.exp(lg).astype(np.float32)
np.savez('base_results/regime_stats_re10000_blind.npz', **out)
print('\nWROTE base_results/regime_stats_re10000_blind.npz  (construction complete, GT never read)', flush=True)

# ---- REPORT ONLY: grade the finished anchor. Feeds back into nothing. --------
E10k = spec('flow-data/kf_re10000_256_40seed.npy')
r = [float(np.exp(lg)[a:b].sum()/E10k[a:b].sum()) for a,b in BANDS]
print('\n[REPORT ONLY — after the fact] blind anchor / true Re=10000:', flush=True)
print('  ' + '  '.join(f'k[{a},{b})={v:.3f}' for (a,b),v in zip(BANDS,r)), flush=True)
print(f"  hi-k [32,96) = {float(np.exp(lg)[32:96].sum()/E10k[32:96].sum()):.3f}", flush=True)
print(f"  true fitted k_d(10000) = {fit(E10k)['kd']:.1f} vs predicted {kd_10k:.1f}", flush=True)
