"""Robustness of the anchor extrapolation laws.  --cutoff: is the kd ceiling a GRID property (independent of the Re=10000 file) or an artefact of it?
Measures, per regime, where the spectrum actually falls off a cliff. If that k is the same in
every regime, the ceiling is the 256^2 grid and survives dropping Re=10000."""
import os, sys
os.chdir('/home/rhautier/ddpm-jax'); sys.path[:0]=['.', 'src/ddpo_ft']
import numpy as np
from anchor_obsfit_builder import fine_spec
REG = {500:('flow-data/kf_re500_256_20seed.npy',range(0,10)),
 1000:('flow-data/kf_2d_re1000_256_40seed.npy',range(0,10)),
 1500:('flow-data/generated/gen_fnons_re1500_kf_1024to256_20seq.npy',range(0,10)),
 2000:('flow-data/kf_re2000_256_40seed.npy',range(0,10)),
 3000:('flow-data/generated/gen_fnons_re3000_kf_1024to256_20seq.npy',range(0,10)),
 4000:('flow-data/generated/gen_fnons_re4000_kf_1024to256_20seq.npy',range(0,10)),
 5000:('flow-data/generated/gen_fnons_re5000_kf_1024to256_20seq.npy',range(0,10)),
 10000:('flow-data/kf_re10000_256_40seed.npy',range(0,10))}
out={}
for Re,(p,s) in (REG.items() if '--cutoff' in sys.argv else []):
    E = fine_spec(p, s); out[Re]=E
    k = np.arange(1,128); e = E[1:128]
    sl = np.gradient(np.log(e), np.log(k))          # local log-log slope
    kcut = k[60:][np.argmin(sl[60:])]               # steepest fall above k=60
    # last k where E is still above 1e-4 of its k=10 value
    kend = k[e > 1e-4*e[9]][-1] if (e > 1e-4*e[9]).any() else -1
    print(f"Re={Re:<6} steepest-fall k={kcut:3d} (slope {sl[60:].min():.1f})   "
          f"k where E<1e-4*E(10): {kend:3d}   E(120)/E(10)={e[119]/e[9]:.2e}", flush=True)
if out:
    np.savez('/home/rhautier/ddpm-jax/base_results/fine_spectra_8regimes.npz',
             **{f'E{r}':v for r,v in out.items()})
    print("CUTOFF CHECK DONE")

# --- appended: robustness of the LOO verdicts (2026-07-31) -------------------
# Two challenges to the 2026-07-30 conclusions: (1) drop the suspect Re=10000 regime and refit
# every law; (2) test whether the residuals track DATASET FAMILY (real sims vs fnons-generated)
# rather than Re. Run: python src/ddpo_ft/anchor_law_robustness.py --laws
if '--laws' in sys.argv:
    from scipy.optimize import curve_fit
    d = np.load('base_results/anchor_law_loo.npz')
    Res, T, kd, al, pp = d['Res'].astype(float), d['T'], d['kd'], d['alpha'], d['p']
    FAM = np.array(['real' if r in (500, 1000, 2000, 10000) else 'gen' for r in Res])

    def loo_power(Rs, v):
        e = []
        for i in range(len(Rs)):
            m = np.arange(len(Rs)) != i
            s, c = np.polyfit(np.log(Rs[m]), np.log(v[m]), 1)
            e.append(abs(np.exp(c + s*np.log(Rs[i]))/v[i] - 1))
        return np.array(e)

    def sat(R, kmax, R0, q):
        return kmax*(1 - np.exp(-(R/R0)**q))

    for tag, mask in (("WITH Re=10000", Res > 0), ("WITHOUT Re=10000", Res < 9000)):
        R_, T_, K_, A_, P_ = Res[mask], T[mask], kd[mask], al[mask], pp[mask]
        print(f"\n=== {tag} (n={mask.sum()}) ===")
        for name, v in (('T', T_), ('alpha', A_), ('p', P_)):
            lo = loo_power(R_, v); fx = np.abs(0.5*(v[0]+v[1])/v - 1)
            print(f"  {name:>5}: LOO {np.median(lo)*100:5.1f}%  vs fixed {np.median(fx)*100:5.1f}%")
        lp = loo_power(R_, K_)
        ls = [abs(sat(R_[i], *curve_fit(sat, R_[np.arange(len(R_)) != i], K_[np.arange(len(R_)) != i],
                                        p0=[90, 2000, 1.0], maxfev=40000)[0])/K_[i] - 1)
              for i in range(len(R_))]
        pr, _ = curve_fit(sat, R_, K_, p0=[90, 2000, 1.0], maxfev=40000)
        print(f"     kd: LOO power {np.median(lp)*100:5.1f}%  vs saturating {np.median(ls)*100:5.1f}%"
              f"  | sat kd_max={pr[0]:.1f} R0={pr[1]:.0f} q={pr[2]:.2f}")
        print(f"      T: power-law exponent {np.polyfit(np.log(R_), np.log(T_), 1)[0]:+.4f}")

    print("\n=== FAMILY CONFOUND (real 40seed/20seed sims vs fnons 1024->256 generated) ===")
    for q, v in (('T', T), ('kd', kd), ('alpha', al), ('p', pp)):
        s, c = np.polyfit(np.log(Res), np.log(v), 1)
        r = np.log(v) - (c + s*np.log(Res))
        off = np.exp(r[FAM == 'gen'].mean() - r[FAM == 'real'].mean())
        print(f"  {q:>5}: 8-pt exponent {s:+.3f} | gen/real offset {off:.2f}x | residual scatter {r.std():.3f}")
    for f in ('real', 'gen'):
        m = FAM == f
        for q, v in (('T', T), ('kd', kd), ('p', pp)):
            s, c = np.polyfit(np.log(Res[m]), np.log(v[m]), 1)
            err = np.abs(np.exp(c + s*np.log(Res[m]))/v[m] - 1)
            print(f"  {f:>4} {q:>3}: exponent {s:+.3f}  in-sample |err| median {np.median(err)*100:4.1f}%")
    print("ROBUSTNESS CHECK DONE")
