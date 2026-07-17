"""Observation-constrained GT-free anchor for Re=2000.
HARD CONSTRAINT: target regime contributes ONLY 4x-subsampled (64x64) pixels.
Pipeline: (1) measure the observation transfer T(k)=E_coarse/E_fine on Re=500/1000 (full fields
allowed there); (2) fit the spectrum model to the T-corrected observed band k in [6,30] of the
target with alpha/p priors from the known regimes, k_d FREE; (3) leave-one-regime-out validation
(pretend Re=1000 is the target, use only ITS lowres + Re=500 priors/T) -> per-band bias correction
+ error bar; (4) build corrected Re=2000 anchor npz. GT of Re=2000 used ONLY for final reporting."""
import os
import numpy as np
from scipy.optimize import curve_fit, least_squares
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SP = "/tmp/claude-2001/-home-rhautier-ddpm-jax/24ff525a-722f-4c10-b5d9-663bc835ccb1/scratchpad"
os.chdir("/home/rhautier/ddpm-jax")
N, NC, SIG, KF, KMAX = 256, 64, 4.7988, 4, 128
_k = np.fft.fftfreq(N, 1.0/N)
_KR = np.round(np.sqrt(_k[:,None]**2 + _k[None,:]**2)).astype(int).ravel()
_kc = np.fft.fftfreq(NC, 1.0/NC)
_KRC = np.round(np.sqrt(_kc[:,None]**2 + _kc[None,:]**2)).astype(int).ravel()
REG = {500:"flow-data/kf_re500_256_20seed.npy", 1000:"flow-data/kf_2d_re1000_256_40seed.npy", 2000:"flow-data/kf_re2000_256_20seed.npy"}

def pools(path, n_seqs=12, fstride=8):
    """fine spectra (KMAX,) and coarse (subsampled 4x) spectra (33,) per frame."""
    arr = np.load(path, mmap_mode="r"); Sf, Sc = [], []
    for s in range(min(n_seqs, arr.shape[0])):
        for t in range(40, min(312, arr.shape[1]), fstride):
            f = np.asarray(arr[s,t], np.float32)/SIG
            Pf = (np.abs(np.fft.fft2(f))**2).ravel()
            Sf.append(np.bincount(_KR, Pf, minlength=N)[:KMAX])
            fl = f[::4, ::4]
            Pc = (np.abs(np.fft.fft2(fl))**2).ravel()
            Sc.append(np.bincount(_KRC, Pc, minlength=NC)[:33])
    return np.asarray(Sf), np.asarray(Sc)

print("streaming spectra ...", flush=True)
POOL = {re: pools(p) for re, p in REG.items()}
Ef = {re: POOL[re][0].mean(0) for re in REG}     # fine truth (NOT usable for 2000 in construction)
Ec = {re: POOL[re][1].mean(0) for re in REG}     # coarse observed (usable everywhere)

# (1) observation transfer on known regimes
KOBS = np.arange(6, 31)
T500  = Ec[500][KOBS]  / Ef[500][KOBS]
T1000 = Ec[1000][KOBS] / Ef[1000][KOBS]
print(f"T(k) 500 vs 1000 agreement: max rel diff {np.max(np.abs(T500/T1000 - 1)):.3f}", flush=True)

def model(k, logA, alpha, kd, p): return logA - alpha*np.log(k/KF) - (k/kd)**p
KFIT_FULL = np.arange(KF+2, 110)
def fit_full(E):
    y = np.log(E[KFIT_FULL] + 1e-30)
    popt,_ = curve_fit(model, KFIT_FULL.astype(float), y, p0=[y[0],1.0,40.0,2.0], maxfev=20000,
                       bounds=([-40,0,5,0.5],[40,4,300,6]))
    return popt

def obs_fit(E_obs_corr, prior, w_alpha=8.0, w_p=4.0):
    """fit model to corrected observed band with Gaussian priors on alpha/p; logA, kd free."""
    y = np.log(E_obs_corr + 1e-30)
    def resid(th):
        logA, alpha, kd, p = th
        r = model(KOBS.astype(float), logA, alpha, kd, p) - y
        return np.concatenate([r, [w_alpha*(alpha-prior['alpha']), w_p*(p-prior['p'])]])
    th0 = [y[0]+prior['alpha']*np.log(KOBS[0]/KF), prior['alpha'], 45.0, prior['p']]
    out = least_squares(resid, th0, bounds=([-40,0,5,0.5],[40,4,300,6]), max_nfev=20000)
    return dict(zip(('logA','alpha','kd','p'), out.x))

def build_anchor(fit, low_src):
    lg = np.full(KMAX, -40.0)
    kk = np.arange(KF+1, KMAX)
    lg[kk] = model(kk.astype(float), fit['logA'], fit['alpha'], fit['kd'], fit['p'])
    lg[0:KF+1] = np.log(low_src[0:KF+1] + 1e-30)     # forcing scales from observations
    return lg

BANDS = [(20,32),(32,64),(64,96)]
def band_ratios(lg, E_true):
    E = np.exp(lg)
    return [float(E[a:b].sum()/E_true[a:b].sum()) for a,b in BANDS]

# (3) leave-one-out: target=1000 using ONLY its lowres + 500's T and priors
p500 = fit_full(Ef[500]); prior500 = dict(alpha=p500[1], p=p500[3])
obs1000_corr = Ec[1000][KOBS] / T500                       # T from 500 only
fit_loo = obs_fit(obs1000_corr, prior500)
# low-k from observations too (coarse spectrum /T at low k): correct k<=KF via T extension
Tlow500 = Ec[500][0:KF+1] / Ef[500][0:KF+1]
low1000_obs = Ec[1000][0:KF+1] / Tlow500
lg_loo = build_anchor(fit_loo, low1000_obs)
r_loo = band_ratios(lg_loo, Ef[1000])
print(f"\nLOO (target=Re1000, only lowres+Re500): kd={fit_loo['kd']:.1f} (true-fit {fit_full(Ef[1000])[2]:.1f})", flush=True)
print(f"  band ratios pred/true: [20,32)={r_loo[0]:.3f}  [32,64)={r_loo[1]:.3f}  [64,96)={r_loo[2]:.3f}", flush=True)
CORR = 1.0/np.array(r_loo)                                 # per-band multiplicative bias correction

# reverse LOO: target=500 using 1000's T/priors (sanity on correction stability)
p1000 = fit_full(Ef[1000]); prior1000 = dict(alpha=p1000[1], p=p1000[3])
obs500_corr = Ec[500][KOBS] / T1000
fit_loo2 = obs_fit(obs500_corr, prior1000)
Tlow1000 = Ec[1000][0:KF+1] / Ef[1000][0:KF+1]
lg_loo2 = build_anchor(fit_loo2, Ec[500][0:KF+1]/Tlow1000)
r_loo2 = band_ratios(lg_loo2, Ef[500])
print(f"reverse LOO (target=Re500): bands {r_loo2[0]:.3f} {r_loo2[1]:.3f} {r_loo2[2]:.3f}", flush=True)

# (4) TARGET Re=2000: only its lowres; T and priors averaged from BOTH known regimes
Tavg = 0.5*(T500 + T1000)
prior_avg = dict(alpha=0.5*(prior500['alpha']+prior1000['alpha']), p=0.5*(prior500['p']+prior1000['p']))
obs2000_corr = Ec[2000][KOBS] / Tavg
fit2000 = obs_fit(obs2000_corr, prior_avg)
Tlow = 0.5*(Tlow500 + Tlow1000)
low2000_obs = Ec[2000][0:KF+1] / Tlow
lg2000 = build_anchor(fit2000, low2000_obs)
print(f"\nTARGET Re=2000 obs-fit: logA={fit2000['logA']:.2f} alpha={fit2000['alpha']:.3f} "
      f"kd={fit2000['kd']:.1f} p={fit2000['p']:.3f}", flush=True)
# apply LOO bias correction to the tail bands (piecewise-smooth: interpolate corr across k)
kk = np.arange(KMAX, dtype=float)
band_centers = np.array([26.0, 48.0, 80.0])
corr_k = np.interp(kk, band_centers, CORR, left=CORR[0], right=CORR[-1])
corr_k[:20] = 1.0
lg2000_corr = lg2000 + np.log(corr_k + 1e-30)

# ---- REPORTING ONLY: compare against true Re2000 (not used in construction)
E2 = Ef[2000]
old = np.load('base_results/regime_stats_re2000_extrap.npz')['log_spec_ref']
rows = [('shipped extrapolated anchor', old), ('obs-fit (uncorrected)', lg2000), ('obs-fit + LOO correction', lg2000_corr)]
print(f"\n[REPORT ONLY] anchor/GT band ratios vs true Re=2000:")
print(f"{'variant':<30}{'[20,32)':>9}{'[32,64)':>9}{'[64,96)':>9}")
for nm, lg in rows:
    r = band_ratios(np.asarray(lg, float), E2)
    print(f"{nm:<30}{r[0]:>9.3f}{r[1]:>9.3f}{r[2]:>9.3f}", flush=True)

# save the corrected anchor npz (spectrum replaced; other refs carried from shipped extrap)
oldnpz = dict(np.load('base_results/regime_stats_re2000_extrap.npz'))
oldnpz['log_spec_ref'] = lg2000_corr.astype(np.float32)
oldnpz['spec_ref'] = np.exp(lg2000_corr).astype(np.float32)
np.savez('base_results/regime_stats_re2000_obsfit.npz', **oldnpz)
print("\nsaved base_results/regime_stats_re2000_obsfit.npz", flush=True)

# figure
kx = np.arange(1, 110)
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5))
axes[0].plot(kx, E2[1:110], 'k-', lw=2.2, label='true Re=2000 GT (report only)')
axes[0].plot(kx, np.exp(old[1:110]), '--', color='#8a8578', lw=1.6, label='shipped extrapolation')
axes[0].plot(kx, np.exp(lg2000_corr[1:110]), '-', color='#28658a', lw=1.9, label='obs-fit + LOO correction (GT-free)')
axes[0].set_xscale('log'); axes[0].set_yscale('log'); axes[0].set_xlabel('k'); axes[0].set_ylabel('E(k)')
axes[0].axvline(31, color='#a05a12', lw=1, ls=':')
axes[0].text(31, axes[0].get_ylim()[0]*3, ' observation Nyquist', fontsize=8, color='#a05a12')
axes[0].set_title('GT-free anchor from low-res observations only'); axes[0].legend(frameon=False, fontsize=8.5)
for nm, lg, c, ls in [('shipped extrapolation', old, '#8a8578', '--'),
                      ('obs-fit uncorrected', lg2000, '#c9a227', ':'),
                      ('obs-fit + LOO correction', lg2000_corr, '#28658a', '-')]:
    axes[1].plot(kx, np.exp(np.asarray(lg,float)[1:110])/E2[1:110], ls, color=c, lw=1.8, label=nm)
axes[1].axhline(1, color='k', lw=1); axes[1].set_xscale('log'); axes[1].set_ylim(0.4, 1.6)
axes[1].set_xlabel('k'); axes[1].set_ylabel('anchor / GT'); axes[1].set_title('ratio to truth (report only)')
axes[1].legend(frameon=False, fontsize=8.5)
plt.tight_layout(); plt.savefig(f'{SP}/fig_anchor_obsfit.png', dpi=112, bbox_inches='tight')
print('figure saved', flush=True)

# ================= V2: kd gets a scaling-law prior; observed band adjusts, not determines =====
print("\n================ V2 (kd prior from scaling law) ================", flush=True)
def obs_fit2(E_obs_corr, prior, kd_prior, w_alpha=8.0, w_p=4.0, w_kd=6.0):
    y = np.log(E_obs_corr + 1e-30)
    def resid(th):
        logA, alpha, kd, p = th
        r = model(KOBS.astype(float), logA, alpha, kd, p) - y
        return np.concatenate([r, [w_alpha*(alpha-prior['alpha']), w_p*(p-prior['p']),
                                   w_kd*np.log(kd/kd_prior)]])
    th0 = [y[0]+prior['alpha']*np.log(KOBS[0]/KF), prior['alpha'], kd_prior, prior['p']]
    out = least_squares(resid, th0, bounds=([-40,0,5,0.5],[40,4,300,6]), max_nfev=20000)
    return dict(zip(('logA','alpha','kd','p'), out.x))

# LOO target=1000: kd prior from Re500 kd scaled by THEORY q=0.5 (only one known regime)
kd_prior_1000 = p500[2] * (1000/500)**0.5
fit_loo_v2 = obs_fit2(obs1000_corr, prior500, kd_prior_1000)
lg_loo_v2 = build_anchor(fit_loo_v2, low1000_obs)
r_loo_v2 = band_ratios(lg_loo_v2, Ef[1000])
print(f"LOO v2 (target=1000): kd_prior={kd_prior_1000:.1f} -> kd={fit_loo_v2['kd']:.1f} (true 34.0)")
print(f"  bands pred/true: {r_loo_v2[0]:.3f} {r_loo_v2[1]:.3f} {r_loo_v2[2]:.3f}")
# reverse LOO target=500: kd prior from 1000 scaled DOWN by theory
kd_prior_500 = p1000[2] * (500/1000)**0.5
fit_loo2_v2 = obs_fit2(obs500_corr, prior1000, kd_prior_500)
lg_loo2_v2 = build_anchor(fit_loo2_v2, Ec[500][0:KF+1]/Tlow1000)
r_loo2_v2 = band_ratios(lg_loo2_v2, Ef[500])
print(f"reverse LOO v2 (target=500): kd_prior={kd_prior_500:.1f} -> kd={fit_loo2_v2['kd']:.1f} (true 23.1)")
print(f"  bands pred/true: {r_loo2_v2[0]:.3f} {r_loo2_v2[1]:.3f} {r_loo2_v2[2]:.3f}")
CORR2 = 1.0/np.sqrt(np.array(r_loo_v2)*np.array(r_loo2_v2))    # geometric-mean bias of BOTH directions
print(f"  two-sided LOO correction (geom mean): {CORR2[0]:.3f} {CORR2[1]:.3f} {CORR2[2]:.3f}")

# TARGET 2000: kd prior from empirical exponent measured on 500->1000
q_emp = np.log(p1000[2]/p500[2]) / np.log(2.0)
kd_prior_2000 = p1000[2] * 2.0**q_emp
fit2000_v2 = obs_fit2(obs2000_corr, prior_avg, kd_prior_2000)
lg2000_v2 = build_anchor(fit2000_v2, low2000_obs)
corr_k2 = np.interp(kk, band_centers, CORR2, left=CORR2[0], right=CORR2[-1]); corr_k2[:20] = 1.0
lg2000_v2c = lg2000_v2 + np.log(corr_k2 + 1e-30)
print(f"\nTARGET 2000 v2: q_emp={q_emp:.3f} kd_prior={kd_prior_2000:.1f} -> kd={fit2000_v2['kd']:.1f} "
      f"(true-fit 55.6) alpha={fit2000_v2['alpha']:.3f} p={fit2000_v2['p']:.3f}")
GEN = 1.2   # deliberate generosity variant (in-dist anchors sit ~1.2x above their GT tail)
lg2000_v2g = lg2000_v2c.copy()
gen_k = np.interp(kk, [20, 32, 96], [1.0, GEN, GEN]); lg2000_v2g += np.log(gen_k)
print(f"\n[REPORT ONLY] v2 anchors vs true Re=2000:")
print(f"{'variant':<34}{'[20,32)':>9}{'[32,64)':>9}{'[64,96)':>9}")
for nm, lg in [('shipped extrapolated', old), ('v2 obs-fit + kd prior', lg2000_v2),
               ('v2 + two-sided LOO corr', lg2000_v2c), ('v2 + LOO corr + 1.2x generosity', lg2000_v2g),
               ('(in-dist Re1000 anchor/its GT)', None)]:
    if lg is None:
        A1 = np.load('base_results/regime_stats_re1000.npz')['spec_ref']
        r = [float(A1[a:b].sum()/Ef[1000][a:b].sum()) for a,b in BANDS]
    else:
        r = band_ratios(np.asarray(lg, float), E2)
    print(f"{nm:<34}{r[0]:>9.3f}{r[1]:>9.3f}{r[2]:>9.3f}", flush=True)

oldnpz2 = dict(np.load('base_results/regime_stats_re2000_extrap.npz'))
oldnpz2['log_spec_ref'] = lg2000_v2c.astype(np.float32); oldnpz2['spec_ref'] = np.exp(lg2000_v2c).astype(np.float32)
np.savez('base_results/regime_stats_re2000_obsfit.npz', **oldnpz2)
oldnpz2['log_spec_ref'] = lg2000_v2g.astype(np.float32); oldnpz2['spec_ref'] = np.exp(lg2000_v2g).astype(np.float32)
np.savez('base_results/regime_stats_re2000_obsfit_gen12.npz', **oldnpz2)
print("saved obsfit (v2+corr) and obsfit_gen12 npz variants")

fig, ax = plt.subplots(figsize=(8.5, 5.2))
for nm, lg, c, ls, lw in [('shipped extrapolation', old, '#8a8578', '--', 1.6),
                          ('v2 obs-fit+prior+LOO corr (GT-free)', lg2000_v2c, '#28658a', '-', 2.0),
                          ('v2 + 1.2x generosity (GT-free)', lg2000_v2g, '#a05a12', '-', 1.7)]:
    ax.plot(kx, np.exp(np.asarray(lg,float)[1:110])/E2[1:110], ls, color=c, lw=lw, label=nm)
A1 = np.load('base_results/regime_stats_re1000.npz')['spec_ref']
ax.plot(kx, A1[1:110]/Ef[1000][1:110], ':', color='#3ca951', lw=1.4, label='in-dist Re1000 anchor/GT (the profile that works)')
ax.axhline(1, color='k', lw=1); ax.set_xscale('log'); ax.set_ylim(0.4, 1.8)
ax.axvline(31, color='#a05a12', lw=1, ls=':', alpha=.6)
ax.set_xlabel('k'); ax.set_ylabel('anchor / GT')
ax.set_title('GT-free Re=2000 anchors (only low-res target samples used) — ratio to truth, report-only')
ax.legend(frameon=False, fontsize=8.5)
plt.tight_layout(); plt.savefig(f'{SP}/fig_anchor_obsfit_v2.png', dpi=112, bbox_inches='tight')
print('v2 figure saved', flush=True)
