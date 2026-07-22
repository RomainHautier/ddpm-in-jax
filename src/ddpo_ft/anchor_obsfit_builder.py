"""FROZEN anchor-construction PROCEDURE (parameterized) — protocol v1.3.
Rebuilds the observation-constrained spectral anchor FROM THE DEPLOYMENT DATA GENERATION:
the target regime contributes ONLY the LR (4x-subsampled) fields of the given sequences.

Recipe (validated 2026-07-17/22, unchanged): spectrum model E(k)=A(k/kf)^-a exp(-(k/kd)^p) fitted
on Re=500+1000 full data; observation transfer T(k) measured on those refs; target obs-band
(T-corrected, k in [6,30]) fitted with alpha/p priors and a kd prior from the measured Re-scaling
exponent; two-sided LOO bias correction (500<->1000). residual_ref via the Re-scaling law
(b=2.705). The npz stores an OBS FINGERPRINT (the LR band spectrum + source path + seqs) so any
future run can verify the anchor matches its training data generation (staleness check).

Usage: python -m src.ddpo_ft.anchor_obsfit_builder <target_gt_path> <re> <seq0,seq1,...> <out.npz>
"""
import sys, os
import numpy as np
from scipy.optimize import curve_fit, least_squares
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
N, NC, SIG, KF, KMAX = 256, 64, 4.7988, 4, 128
_k = np.fft.fftfreq(N, 1.0/N)
_KR = np.round(np.sqrt(_k[:,None]**2+_k[None,:]**2)).astype(int).ravel()
_kc = np.fft.fftfreq(NC, 1.0/NC)
_KRC = np.round(np.sqrt(_kc[:,None]**2+_kc[None,:]**2)).astype(int).ravel()
REFS = {500:'flow-data/kf_re500_256_20seed.npy', 1000:'flow-data/kf_2d_re1000_256_40seed.npy'}
KOBS = np.arange(6, 31)

def _frames(T, fstride):
    # full-length sequences: skip the 40-frame transient window; short (already-strided) files: all frames
    return range(0, T) if T < 40 else range(40, min(312, T), fstride)

def fine_spec(path, seqs, fstride=8):
    a = np.load(path, mmap_mode='r'); S = []
    for s in seqs:
        for t in _frames(a.shape[1], fstride):
            f = np.asarray(a[s,t], np.float32)/SIG
            S.append(np.bincount(_KR, (np.abs(np.fft.fft2(f))**2).ravel(), minlength=N)[:KMAX])
    return np.asarray(S).mean(0)

def coarse_spec(path, seqs, fstride=8):
    a = np.load(path, mmap_mode='r'); S = []
    for s in seqs:
        for t in _frames(a.shape[1], fstride):
            fl = (np.asarray(a[s,t], np.float32)/SIG)[::4, ::4]
            S.append(np.bincount(_KRC, (np.abs(np.fft.fft2(fl))**2).ravel(), minlength=NC)[:33])
    return np.asarray(S).mean(0)

def model(k, logA, alpha, kd, p): return logA - alpha*np.log(k/KF) - (k/kd)**p
KFIT = np.arange(KF+2, 110)
def fit_full(E):
    y = np.log(E[KFIT] + 1e-30)
    o,_ = curve_fit(model, KFIT.astype(float), y, p0=[y[0],1.,40.,2.], maxfev=40000,
                    bounds=([-40,0,5,0.5],[40,4,400,6]))
    return dict(zip(('logA','alpha','kd','p'), o))

def obs_fit(E_obs_corr, prior, kd_prior, w_a=8., w_p=4., w_kd=6.):
    y = np.log(E_obs_corr + 1e-30)
    def resid(th):
        logA, alpha, kd, p = th
        return np.concatenate([model(KOBS.astype(float), logA, alpha, kd, p) - y,
                               [w_a*(alpha-prior['alpha']), w_p*(p-prior['p']), w_kd*np.log(kd/kd_prior)]])
    th0 = [y[0]+prior['alpha']*np.log(KOBS[0]/KF), prior['alpha'], kd_prior, prior['p']]
    return dict(zip(('logA','alpha','kd','p'),
        least_squares(resid, th0, bounds=([-40,0,5,0.5],[40,4,400,6]), max_nfev=40000).x))

def build(target_path, re_t, target_seqs, out_path):
    E500, E1000 = fine_spec(REFS[500], range(0,10)), fine_spec(REFS[1000], range(0,10))
    C500, C1000 = coarse_spec(REFS[500], range(0,10)), coarse_spec(REFS[1000], range(0,10))
    F500, F1000 = fit_full(E500), fit_full(E1000)
    T = 0.5*(C500[KOBS]/E500[KOBS] + C1000[KOBS]/E1000[KOBS])
    Tlow = 0.5*(C500[0:KF+1]/E500[0:KF+1] + C1000[0:KF+1]/E1000[0:KF+1])
    prior = dict(alpha=0.5*(F500['alpha']+F1000['alpha']), p=0.5*(F500['p']+F1000['p']))
    q = np.log(F1000['kd']/F500['kd'])/np.log(2.0)
    # two-sided LOO correction, measured on refs (unchanged from the validated recipe)
    def loo(tE, tC, src_fit, ratio):
        f = obs_fit(tC[KOBS]/ (C500[KOBS]/E500[KOBS] if ratio=='up' else C1000[KOBS]/E1000[KOBS]),
                    dict(alpha=src_fit['alpha'], p=src_fit['p']), src_fit['kd']*(2.0 if ratio=='up' else 0.5)**q)
        lg = np.full(KMAX,-40.0); kk = np.arange(KF+1,KMAX)
        lg[kk] = model(kk.astype(float), f['logA'], f['alpha'], f['kd'], f['p'])
        lg[0:KF+1] = np.log(tC[0:KF+1]/Tlow + 1e-30)
        BANDS = [(20,32),(32,64),(64,96)]
        return np.array([float(np.exp(lg)[a:b].sum()/tE[a:b].sum()) for a,b in BANDS])
    r_up, r_dn = loo(E1000, C1000, F500, 'up'), loo(E500, C500, F1000, 'dn')
    CORR = 1.0/np.sqrt(r_up*r_dn)
    # TARGET: obs from the DEPLOYMENT data generation
    Ct = coarse_spec(target_path, target_seqs)
    kd_prior = F1000['kd']*(re_t/1000.0)**q
    ft = obs_fit(Ct[KOBS]/T, prior, kd_prior)
    lg = np.full(KMAX,-40.0); kk = np.arange(KF+1,KMAX)
    lg[kk] = model(kk.astype(float), ft['logA'], ft['alpha'], ft['kd'], ft['p'])
    lg[0:KF+1] = np.log(Ct[0:KF+1]/Tlow + 1e-30)
    kx = np.arange(KMAX, dtype=float)
    corr_k = np.interp(kx, [26.,48.,80.], CORR, left=CORR[0], right=CORR[-1]); corr_k[:20] = 1.0
    lg = lg + np.log(corr_k + 1e-30)
    # residual_ref by the frozen Re-scaling law
    b = 2.705; r1000 = 10.218
    res_ref = r1000*(re_t/1000.0)**b
    out = dict(log_spec_ref=lg.astype(np.float32), spec_ref=np.exp(lg).astype(np.float32),
               residual_ref=np.float32(res_ref),
               obs_lr_fingerprint=Ct.astype(np.float32),                  # staleness check data
               obs_source=np.bytes_(f"{target_path}|seqs={list(target_seqs)}"),
               kd_fit=np.float32(ft['kd']), kd_prior=np.float32(kd_prior))
    # carry enstrophy/quantiles from the Re=1000 hold/universal-shape convention
    ref = np.load('base_results/regime_stats_re1000.npz')
    out['enstrophy_ref'] = ref['enstrophy_ref']; out['quantiles_ref'] = ref['quantiles_ref']
    np.savez(out_path, **out)
    print(f"anchor -> {out_path}")
    print(f"  kd: prior {kd_prior:.1f} -> fit {ft['kd']:.1f} | alpha {ft['alpha']:.3f} | residual_ref {res_ref:.2f}")
    print(f"  obs fingerprint stored ({target_path}, seqs {list(target_seqs)})")

def verify_freshness(anchor_path, target_path, target_seqs, tol=0.05):
    """GT-free PRE-FLIGHT: does the anchor's stored observation fingerprint match the LR spectrum
    of the data generation about to be trained on? Band ratio in k[10,30) beyond `tol` -> STALE."""
    d = np.load(anchor_path)
    if 'obs_lr_fingerprint' not in d:
        print(f"PRE-FLIGHT: {anchor_path} has NO fingerprint (pre-v1.3 artifact) — treat as STALE"); return False
    Ct = coarse_spec(target_path, target_seqs)
    fp = d['obs_lr_fingerprint']
    ratio = float(fp[10:30].sum()/Ct[10:30].sum())
    ok = abs(ratio-1) <= tol
    print(f"PRE-FLIGHT: anchor-obs / training-LR band ratio = {ratio:.3f} -> {'OK' if ok else 'STALE'}")
    return ok

if __name__ == '__main__':
    tgt, re_t, seqs, out = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(',')], sys.argv[4]
    build(tgt, re_t, seqs, out)

REF_LAG1 = (0.95, 0.9985)   # lag-1 frame corr of dt=1/32 generations: re1000 base 0.9907, old re2000 0.9864
def temporal_compat_check(path, seqs):
    """v1.4 pre-flight (GT-free, uses LR-resolution reads only): is the file's frame spacing
    compatible with the base model's training dt? The 2026-07 40-seed generation is ~80x finer
    (lag-1 corr 0.99999 vs 0.986-0.991) -> triplets are near-static, conditioning off-distribution,
    and dt-based residuals are in the wrong convention."""
    a = np.load(path, mmap_mode='r'); cs = []
    t0 = min(100, a.shape[1] - 2)
    for s in seqs[:4]:
        x = np.asarray(a[s,t0], np.float32)[::4,::4].ravel()
        y = np.asarray(a[s,t0+1], np.float32)[::4,::4].ravel()
        cs.append(float(np.corrcoef(x, y)[0,1]))
    c = float(np.mean(cs))
    ok = REF_LAG1[0] <= c <= REF_LAG1[1]
    print(f"PRE-FLIGHT (temporal): LR lag-1 corr = {c:.5f} -> "
          f"{'OK (dt-compatible)' if ok else 'INCOMPATIBLE frame spacing (fine-dt generation?)'}")
    return ok
