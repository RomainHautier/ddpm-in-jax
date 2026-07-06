"""Two studies for the DDPO OOD reward-anchor question:
  (1) HOW FEW IS FEW  — anchor-convergence: error of a spectrum/PDF/enstrophy anchor built from N
      frames vs a large reference, per regime. Tells us the minimum simulation length.
  (2) EXTRAPOLATION  — fit the enstrophy-spectrum model on Re=500 & 1000, extrapolate to Re=2000
      (k_d ~ Re^1/2), compare to the true Re=2000 anchor. Extrapolation is BEYOND the bracket.
Memory-frugal: stream frames, keep only per-frame stats (spectra/quantiles/enstrophy). numpy/CPU.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SP = "/tmp/claude-2001/-home-rhautier-ddpm-jax/24ff525a-722f-4c10-b5d9-663bc835ccb1/scratchpad"
os.chdir("/home/rhautier/ddpm-jax")
N, SIG, KF, KMAX = 256, 4.7988, 4, 128
rng = np.random.default_rng(0)

_k = np.fft.fftfreq(N, 1.0 / N)
_KR = np.round(np.sqrt(_k[:, None] ** 2 + _k[None, :] ** 2)).astype(int).ravel()
QS = np.linspace(0.0, 1.0, 257)
REG = {500: "flow-data/kf_re500_256_20seed.npy",
       1000: "flow-data/kf_2d_re1000_256_40seed.npy",
       2000: "flow-data/kf_re2000_256_20seed.npy"}


def frame_spectrum(f):                      # f: normalized (N,N) -> enstrophy spectrum (KMAX,)
    P = (np.abs(np.fft.fft2(f)) ** 2).ravel()
    return np.bincount(_KR, P, minlength=N)[:KMAX]


def collect(path, n_seqs, f0=40, f1=312, fstride=4):
    """Stream frames; return per-frame (spectra[M,KMAX], quantiles[M,257], enstrophy[M])."""
    arr = np.load(path, mmap_mode="r")
    n_seqs = min(n_seqs, arr.shape[0])
    S, Q, E = [], [], []
    for s in range(n_seqs):
        for t in range(f0, min(f1, arr.shape[1]), fstride):
            f = np.asarray(arr[s, t], np.float32) / SIG
            S.append(frame_spectrum(f)); Q.append(np.quantile(f.ravel(), QS)); E.append(float((f ** 2).mean()))
    return np.asarray(S), np.asarray(Q), np.asarray(E)


print("collecting per-frame stats (streaming) ...", flush=True)
DATA = {}
for re, path in REG.items():
    S, Q, E = collect(path, n_seqs=18)
    DATA[re] = dict(S=S, Q=Q, E=E)
    print(f"  Re={re}: {len(S)} frames | enstrophy={E.mean():.4f}", flush=True)

# ---------- reference anchors (full pool) ----------
def anchor(S, Q, E):
    return dict(logspec=np.log(S + 1e-20).mean(0),   # geometric-mean spectrum (the used anchor)
                spec=S.mean(0), quant=Q.mean(0), enst=float(np.mean(E)))
REF = {re: anchor(**{k: DATA[re][k] for k in "SQE"} if False else DATA[re]) for re in REG}
# fix kw:
REF = {re: anchor(DATA[re]["S"], DATA[re]["Q"], DATA[re]["E"]) for re in REG}

BAND = slice(1, 96); HIK = slice(32, 96)
def dspec(logE, logref, band):              # d_spec metric: mean sq log-ratio over band
    return float(np.mean((logE[band] - logref[band]) ** 2))
def w1(q, qref):
    return float(np.mean(np.abs(q - qref)))

# ==================================================================== STUDY 1: how few is few
print("\n[1] anchor convergence ...", flush=True)
NS = [5, 10, 20, 40, 80, 160, 320]
conv = {re: {"spec": [], "spec_hik": [], "enst": [], "w1": []} for re in REG}
for re in REG:
    S, Q, E = DATA[re]["S"], DATA[re]["Q"], DATA[re]["E"]
    M = len(S); half = M // 2
    refS, refQ, refE = S[half:], Q[half:], E[half:]                     # disjoint reference
    ref_log = np.log(refS + 1e-20).mean(0); ref_q = refQ.mean(0); ref_e = float(refE.mean())
    poolS, poolQ, poolE = S[:half], Q[:half], E[:half]
    for n in NS:
        if n > len(poolS): break
        es, eh, ee, ew = [], [], [], []
        for _ in range(20):
            idx = rng.choice(len(poolS), n, replace=False)
            lg = np.log(poolS[idx] + 1e-20).mean(0)
            es.append(dspec(lg, ref_log, BAND)); eh.append(dspec(lg, ref_log, HIK))
            ee.append(abs(np.log(poolE[idx].mean() / ref_e)))
            ew.append(w1(poolQ[idx].mean(0), ref_q))
        conv[re]["spec"].append((n, np.mean(es), np.std(es)))
        conv[re]["spec_hik"].append((n, np.mean(eh), np.std(eh)))
        conv[re]["enst"].append((n, np.mean(ee), np.std(ee)))
        conv[re]["w1"].append((n, np.mean(ew), np.std(ew)))
    print(f"  Re={re}: converged (full-pool anchor uses {M} frames)", flush=True)

# in-dist deficit magnitude to compare against (recon d_spec_highk ~1.4 vs floor ~0.24 from calib)
DEFICIT_HIK = 1.4    # the signal the anchor must resolve, from reward_calibration §4

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
col = {500: "#4269d0", 1000: "#3ca951", 2000: "#ff725c"}
for re in REG:
    a = np.array(conv[re]["spec_hik"]); ax[0].errorbar(a[:, 0], a[:, 1], yerr=a[:, 2], fmt="o-", color=col[re], label=f"Re={re}", capsize=2)
    b = np.array(conv[re]["w1"]);       ax[1].errorbar(b[:, 0], b[:, 1], yerr=b[:, 2], fmt="o-", color=col[re], label=f"Re={re}", capsize=2)
    c = np.array(conv[re]["enst"]);     ax[2].errorbar(c[:, 0], c[:, 1], yerr=c[:, 2], fmt="o-", color=col[re], label=f"Re={re}", capsize=2)
ax[0].axhline(0.1 * DEFICIT_HIK, ls="--", color="k", lw=0.8, label="10% of in-dist deficit")
for a, t, yl in zip(ax, ["hi-k spectrum anchor error (d_spec_highk)", "vorticity-PDF anchor error (W1)", "enstrophy anchor |log-ratio| error"],
                    ["error", "error", "error"]):
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("# frames in anchor"); a.set_ylabel(yl); a.set_title(t, fontsize=10); a.legend(fontsize=8, frameon=False); a.grid(alpha=.25)
fig.suptitle("STUDY 1 — how few frames for a reliable reward anchor (disjoint reference)", y=1.02)
plt.tight_layout(); plt.savefig(f"{SP}/study1_convergence.png", dpi=110, bbox_inches="tight")
print("  saved study1_convergence.png")

# print the knee: smallest N with hi-k error < 10% deficit
print("\n  hi-k spectrum anchor error vs N frames:")
print("   N   " + "  ".join(f"Re{re}" for re in REG))
for i, n in enumerate(NS):
    row = []
    for re in REG:
        a = conv[re]["spec_hik"]
        row.append(f"{a[i][1]:.3f}" if i < len(a) else "  -  ")
    print(f"  {n:>3}  " + "  ".join(f"{v:>6}" for v in row))
print(f"  (compare to in-dist deficit d_spec_highk ~ {DEFICIT_HIK}; 10% = {0.1*DEFICIT_HIK})")

# ==================================================================== STUDY 2: extrapolation
print("\n[2] extrapolation {500,1000} -> 2000 ...", flush=True)

def model(k, logA, alpha, kd, p):           # enstrophy-cascade + dissipation cutoff (log space)
    return logA - alpha * np.log(k / KF) - (k / kd) ** p

KFIT = np.arange(KF + 2, 110)               # fit above forcing, below the noisy far tail
def fit_regime(re):
    y = np.log(REF[re]["spec"][KFIT] + 1e-30)
    p0 = [np.log(REF[re]["spec"][KF + 2] + 1e-20), 1.0, 40.0, 2.0]
    popt, _ = curve_fit(model, KFIT.astype(float), y, p0=p0, maxfev=20000,
                        bounds=([-40, 0, 5, 0.5], [40, 4, 300, 6]))
    return dict(logA=popt[0], alpha=popt[1], kd=popt[2], p=popt[3])

FIT = {re: fit_regime(re) for re in (500, 1000, 2000)}
for re in (500, 1000, 2000):
    f = FIT[re]; print(f"  Re={re:>4}: logA={f['logA']:.2f}  alpha={f['alpha']:.2f}  k_d={f['kd']:.1f}  p={f['p']:.2f}")

# k_d scaling check + extrapolation to 2000
kd500, kd1000 = FIT[500]["kd"], FIT[1000]["kd"]
q_fit = np.log(kd1000 / kd500) / np.log(1000 / 500)                 # empirical exponent from 2 pts
kd2000_theory = kd1000 * (2000 / 1000) ** 0.5                       # theory: k_d ~ Re^0.5
kd2000_fit = kd1000 * (2000 / 1000) ** q_fit                       # empirical exponent
print(f"\n  k_d: 500->{kd500:.1f}  1000->{kd1000:.1f}  (ratio {kd1000/kd500:.3f}, sqrt2={np.sqrt(2):.3f})")
print(f"  empirical exponent q={q_fit:.3f} (theory 0.5)")
print(f"  k_d(2000): theory={kd2000_theory:.1f}  empirical={kd2000_fit:.1f}  TRUE(fit)={FIT[2000]['kd']:.1f}")

# extrapolated params: hold alpha, logA universal (mean of 500&1000); shift only k_d
alpha_ext = np.mean([FIT[500]["alpha"], FIT[1000]["alpha"]])
logA_ext = np.mean([FIT[500]["logA"], FIT[1000]["logA"]])
p_ext = np.mean([FIT[500]["p"], FIT[1000]["p"]])
kEV = np.arange(1, 96)
def build(logA, alpha, kd, p):
    lg = np.full(KMAX, -40.0)
    kk = np.arange(KF + 1, KMAX)
    lg[kk] = model(kk.astype(float), logA, alpha, kd, p)
    lg[1:KF + 1] = REF[1000]["logspec"][1:KF + 1]                   # low-k (forcing) from known regime
    return lg
logE_ext_theory = build(logA_ext, alpha_ext, kd2000_theory, p_ext)
logE_ext_fit = build(logA_ext, alpha_ext, kd2000_fit, p_ext)
logE_true2000 = REF[2000]["logspec"]
logE_1000 = REF[1000]["logspec"]                                    # naive baseline: use Re1000 anchor

def band_err(lg, band): return dspec(lg, logE_true2000, band)
print("\n  spectrum match to TRUE Re=2000 (d_spec, lower=better):")
print(f"                       full[1,96)   hi-k[32,96)")
for name, lg in [("extrapolated (theory k_d)", logE_ext_theory),
                 ("extrapolated (fit k_d)", logE_ext_fit),
                 ("naive: use Re=1000 anchor", logE_1000)]:
    print(f"   {name:<26} {band_err(lg, BAND):>8.3f}   {band_err(lg, HIK):>8.3f}")

# enstrophy extrapolation (integral of spectrum) & PDF (universal shape x sqrt(enst))
def enst_from_log(lg): return float(np.exp(lg[1:96]).sum() / (N ** 4) * (N ** 2))  # approx rel scale
# simpler: compare enstrophy anchors directly (measured), and predict via sqrt scaling of std
enst_true = {re: REF[re]["enst"] for re in REG}
# PDF: normalized shape from 500&1000 (divide quantiles by std=sqrt(enst)), extrapolate scale
def norm_shape(re): return REF[re]["quant"] / np.sqrt(REF[re]["enst"])
shape_ext = 0.5 * (norm_shape(500) + norm_shape(1000))
print(f"\n  enstrophy(level) across Re: 500={enst_true[500]:.4f} 1000={enst_true[1000]:.4f} "
      f"2000={enst_true[2000]:.4f}  -> SATURATES (drag-controlled), so hold ~constant, do NOT extrapolate multiplicatively")
enst_geom = enst_true[1000] * (enst_true[1000] / enst_true[500])          # WRONG: geometric growth
enst_hold = enst_true[1000]                                              # RIGHT: saturated -> hold constant
for name, ee in [("geometric (overshoots)", enst_geom), ("hold-constant (saturated)", enst_hold)]:
    qx = shape_ext * np.sqrt(ee)
    print(f"    enst_ext={ee:.4f} [{name:<24}]  W1(PDF ext, true2000)={w1(qx, REF[2000]['quant']):.4f}")
print(f"    (naive use-1000 PDF: W1={w1(REF[1000]['quant'], REF[2000]['quant']):.4f})")
q_ext = shape_ext * np.sqrt(enst_hold)

# ---------- figure: spectra overlay + ratio ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
kx = np.arange(1, 96)
ax[0].plot(kx, np.exp(REF[500]["logspec"][1:96]), color="#4269d0", lw=1.3, label="Re=500 (fit input)")
ax[0].plot(kx, np.exp(REF[1000]["logspec"][1:96]), color="#3ca951", lw=1.3, label="Re=1000 (fit input)")
ax[0].plot(kx, np.exp(logE_true2000[1:96]), color="k", lw=2.2, label="Re=2000 TRUE")
ax[0].plot(kx, np.exp(logE_ext_theory[1:96]), color="#ff725c", lw=1.6, ls="--", label="Re=2000 extrapolated")
ax[0].plot(kx, np.exp(logE_1000[1:96]), color="#9498a0", lw=1.2, ls=":", label="Re=2000 naive(use 1000)")
ax[0].axvline(32, color="gray", lw=0.6, ls=":"); ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("wavenumber k"); ax[0].set_ylabel("enstrophy spectrum E(k)"); ax[0].legend(fontsize=8, frameon=False)
ax[0].set_title("STUDY 2 — extrapolate {500,1000} -> 2000 spectrum", fontsize=10)
ax[1].plot(kx, np.exp(logE_ext_theory[1:96] - logE_true2000[1:96]), color="#ff725c", lw=1.6, label="extrapolated / true")
ax[1].plot(kx, np.exp(logE_1000[1:96] - logE_true2000[1:96]), color="#9498a0", lw=1.2, ls=":", label="naive(1000) / true")
ax[1].axhline(1, color="k", lw=0.8); ax[1].axvspan(32, 95, color="#ff725c", alpha=0.06)
ax[1].axvline(32, color="gray", lw=0.6, ls=":"); ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("wavenumber k"); ax[1].set_ylabel("ratio to true Re=2000"); ax[1].legend(fontsize=8, frameon=False)
ax[1].set_title("per-shell ratio (1.0 = perfect); shaded = hi-k reward band", fontsize=10)
plt.tight_layout(); plt.savefig(f"{SP}/study2_extrapolation.png", dpi=110, bbox_inches="tight")
print("  saved study2_extrapolation.png")
print("\nDONE")
