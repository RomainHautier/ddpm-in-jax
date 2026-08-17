"""Mining the multi-day setpoint-ablation sweep with information that did not exist when it ran.

The sweep scored 648 (model x rung x regime) cells blind on the EXTRAPOLATED obs-fit anchors and
audited 96 of them against ground truth. Since then, the oracle conditioning runs required
building MEASURED anchors (regime_stats_re{R}_measured_train.npz) for every regime. Joining the
three answers questions the original ablation could not:

1. ANCHOR BIAS, exactly: band-energy ratio (extrapolated anchor / measured anchor) per regime —
   how hot the deployed anchor ran, regime by regime, in the exact band the blind score uses.
2. CALIBRATION: audited cells have both a blind score and a graded retention. If blind mis-
   predicts ret differently per regime, does multiplying by the anchor-bias ratio collapse the
   regime dependence? (I.e. was the anchor the ONLY thing wrong with the blind score?)
   Also: the implied set-point — the blind value at which ret=1 per regime, raw and corrected.
3. GUARD COUNTERFACTUALS: with saved blind+proxy for every cell, replay selection per arm with
   the guard off / at 0.15 / 0.20 — which picks change, and (where audited) how they graded.

CPU-only, reads saved npz files. Caveat carried from the sweep design: blind scores were
computed on the anchor's source pool (selection seqs), audits on held-out seqs — pool noise is
part of the scatter.
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np

d = np.load('base_results/setpoint_ablation.npz', allow_pickle=True)
REGS = [1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
OBSFIT = {R: (f'base_results/regime_stats_re{R}_obsfit_gen.npz' if R != 2000
              else 'base_results/regime_stats_re2000_obsfit_newgen.npz') for R in REGS}
RUNG_ORDER = ['K1-50x12', 'K1-75x20', 'K1-100x20', 'K2x50', 'K3x86', 'K4x110', 'K5x140',
              'K6x170', 'K7x200']
ARMS = {'without_re500': (0.784, 0.764, 0.804), 'with_re500': (0.810, 0.770, 0.850)}

# ---- 1. anchor bias in the blind band ----
print("=" * 78)
print("1. EXTRAPOLATED-ANCHOR BIAS vs the measured anchor (band-energy ratios)")
print(f"{'regime':>7} {'E_extrap/E_meas [10,96)':>24} {'[32,96) (reward band)':>22}")
bias = {}
for R in REGS:
    Ae = np.load(OBSFIT[R])['spec_ref']
    Am = np.load(f'base_results/regime_stats_re{R}_measured_train.npz')['spec_ref']
    b10 = float(Ae[10:96].sum() / Am[10:96].sum())
    b32 = float(Ae[32:96].sum() / Am[32:96].sum())
    bias[R] = b10
    print(f"{R:>7} {b10:>24.3f} {b32:>22.3f}")

# ---- 2. blind -> ret calibration, raw and bias-corrected ----
print("\n" + "=" * 78)
print("2. CALIBRATION: graded retention vs blind score (96 audited cells)")
rows = []
for k in d.files:
    if not k.startswith('A|') or not k.endswith('||ret'):
        continue
    _, R, rung, mdl, _ = k.split('|', 4)
    sk = f'S|{R}|{rung}|{mdl}||blind'
    if sk in d.files:
        rows.append(dict(R=int(R), rung=rung, mdl=mdl, blind=float(d[sk]),
                         ret=float(d[k]), place=float(d[f'A|{R}|{rung}|{mdl}||place'])))
print(f"joined cells: {len(rows)}")
print(f"{'regime':>7} {'n':>3} {'corr(blind,ret)':>16} {'ret at blind=0.81':>18} "
      f"{'implied blind@ret=1':>20} {'corrected blind@ret=1':>22}")
for R in REGS:
    rr = [r for r in rows if r['R'] == R]
    if len(rr) < 4:
        continue
    b = np.array([r['blind'] for r in rr]); t = np.array([r['ret'] for r in rr])
    o = np.argsort(b)
    c = float(np.corrcoef(b, t)[0, 1])
    ret_at = float(np.interp(0.81, b[o], t[o]))
    if (t.min() < 1.0 < t.max()):
        o2 = np.argsort(t)
        b_at1 = float(np.interp(1.0, t[o2], b[o2]))
        s = f"{b_at1:>20.3f} {b_at1 / bias[R]:>22.3f}"
    else:
        b_at1 = np.nan
        s = f"{'ret<1 everywhere':>20} {'-':>22}"
    print(f"{R:>7} {len(rr):>3} {c:>16.3f} {ret_at:>18.3f} {s}")
allb = np.array([r['blind'] for r in rows]); allt = np.array([r['ret'] for r in rows])
allc = np.array([r['blind'] / bias[r['R']] for r in rows])
print(f"\nPOOLED corr(blind, ret):           {np.corrcoef(allb, allt)[0,1]:+.3f}")
print(f"POOLED corr(corrected blind, ret): {np.corrcoef(allc, allt)[0,1]:+.3f}")
# per-regime spread of the ret that the SAME raw blind value implies
print("Spread of ret at blind=0.81 across regimes = how non-transferable the raw band is.")

# ---- 3. guard counterfactuals ----
print("\n" + "=" * 78)
print("3. GUARD COUNTERFACTUALS (replayed selection from saved blind+proxy)")
for arm, (sp, lo, hi) in ARMS.items():
    print(f"\n--- arm {arm}: band [{lo},{hi}], set-point {sp} ---")
    for R in REGS:
        cells = {}
        for k in d.files:
            if k.startswith(f'S|{R}|') and k.endswith('||blind'):
                _, _, rung, mdl, _ = k.split('|', 4)
                pk = f'S|{R}|{rung}|{mdl}||proxy'
                cells[(rung, mdl)] = (float(d[k]), float(d[pk]) if pk in d.files else np.nan)
        def pick(guard):
            inb = {c: v for c, v in cells.items() if lo <= v[0] <= hi
                   and (guard is None or not (v[1] > guard))}
            pool = inb if inb else cells
            tag = 'IN-BAND' if inb else 'fallback'
            best = min(pool, key=lambda c: (abs(pool[c][0] - sp) if inb
                                            else abs(pool[c][0] - np.clip(pool[c][0], lo, hi)),
                                            RUNG_ORDER.index(c[0])))
            return best, tag
        p_none, t_none = pick(None)
        p_020, t_020 = pick(0.20)
        p_015, t_015 = pick(0.15)
        def gr(c):
            k = f'A|{R}|{c[0]}|{c[1]}||ret'
            return f"ret={float(d[k]):.3f}" if k in d.files else "unaudited"
        line = f"  Re={R}: no-guard {p_none[0]}|{p_none[1]} ({t_none}, {gr(p_none)})"
        if p_020 != p_none:
            line += f"  | guard.20 -> {p_020[0]}|{p_020[1]} ({t_020}, {gr(p_020)})"
        else:
            line += "  | guard.20 same"
        if p_015 != p_020:
            line += f"  | guard.15 -> {p_015[0]}|{p_015[1]} ({t_015}, {gr(p_015)})"
        print(line)

print("\nMINING COMPLETE")
