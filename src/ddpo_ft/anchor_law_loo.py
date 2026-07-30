"""LEAVE-ONE-REFERENCE-OUT validation of the anchor's extrapolation laws.

The anchor is built from four quantities extrapolated in Re: the transfer function T, the
dissipation wavenumber kd, and the spectral shape parameters alpha and p. Currently only kd is
extrapolated (2-point law from Re=500/1000); T is a FIXED reference average and alpha/p are pinned
at the reference mean. This measures each quantity at 8 Reynolds numbers, fits a power law on 7,
predicts the 8th, and reports the error — the honest GT-free test, since a deployment target's own
data never enters its own prediction.
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np
from anchor_obsfit_builder import fine_spec, coarse_spec, fit_full, KOBS
REGIMES = {
  500:  ('flow-data/kf_re500_256_20seed.npy',              range(0,10)),
  1000: ('flow-data/kf_2d_re1000_256_40seed.npy',          range(0,10)),
  1500: ('flow-data/generated/gen_fnons_re1500_kf_1024to256_20seq.npy', range(0,10)),
  2000: ('flow-data/kf_re2000_256_40seed.npy',             range(0,10)),
  3000: ('flow-data/generated/gen_fnons_re3000_kf_1024to256_20seq.npy', range(0,10)),
  4000: ('flow-data/generated/gen_fnons_re4000_kf_1024to256_20seq.npy', range(0,10)),
  5000: ('flow-data/generated/gen_fnons_re5000_kf_1024to256_20seq.npy', range(0,10)),
  10000:('flow-data/kf_re10000_256_40seed.npy',            range(0,10)),
}
print("measuring per-regime truth (HR spectrum fit + transfer function)...", flush=True)
M = {}
for Re,(p,seqs) in REGIMES.items():
    E = fine_spec(p, seqs); C = coarse_spec(p, seqs)
    f = fit_full(E)
    M[Re] = dict(alpha=f['alpha'], p=f['p'], kd=f['kd'], T=float((C[KOBS]/E[KOBS]).mean()))
    print(f"  Re={Re:<6} alpha={f['alpha']:.3f}  p={f['p']:.3f}  kd={f['kd']:.1f}  T={M[Re]['T']:.5f}",
          flush=True)
Res = np.array(sorted(M))
print("\nLEAVE-ONE-OUT: fit log(q) ~ log(Re) on 7 regimes, predict the 8th")
print(f"{'quantity':>6} | {'median |err|':>12} | {'max |err|':>10} | per-regime errors")
summary = {}
for q in ('T','kd','alpha','p'):
    vals = np.array([M[r][q] for r in Res])
    errs = []
    for i,r in enumerate(Res):
        keep = np.arange(len(Res)) != i
        sl,ic = np.polyfit(np.log(Res[keep]), np.log(vals[keep]), 1)
        pred = np.exp(ic + sl*np.log(r))
        errs.append(abs(pred/vals[i] - 1))
    errs = np.array(errs); summary[q] = errs
    print(f"{q:>6} | {np.median(errs)*100:11.1f}% | {errs.max()*100:9.1f}% | "
          + " ".join(f"{r}:{e*100:.1f}%" for r,e in zip(Res,errs)))
print("\nWHAT THE CURRENT BUILDER DOES INSTEAD (error if you DON'T extrapolate):")
for q in ('T','alpha','p'):
    vals = np.array([M[r][q] for r in Res])
    ref = 0.5*(M[500][q]+M[1000][q])          # the fixed reference average the builder uses
    err = np.abs(ref/vals - 1)
    print(f"  {q:>5}: fixed at the Re=500/1000 mean -> median error {np.median(err)*100:.1f}%, "
          f"max {err.max()*100:.1f}% (at Re={Res[err.argmax()]})")
print("\n=> extrapolating a quantity is worth doing when its LOO error is much smaller than its"
      " fixed-value error.")
for q in ('T','alpha','p'):
    vals = np.array([M[q_] for q_ in []]) if False else np.array([M[r][q] for r in Res])
    ref = 0.5*(M[500][q]+M[1000][q])
    fixed = np.median(np.abs(ref/vals - 1))
    print(f"  {q:>5}: fixed {fixed*100:5.1f}%  vs  extrapolated {np.median(summary[q])*100:5.1f}%  "
          f"-> {'WORTH IT' if np.median(summary[q]) < 0.6*fixed else 'no clear gain'}")
np.savez('base_results/anchor_law_loo.npz', Res=Res,
         **{q: np.array([M[r][q] for r in Res]) for q in ('T','kd','alpha','p')})
print("\nLAW LOO COMPLETE", flush=True)
