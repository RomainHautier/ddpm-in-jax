"""FAITHFUL replay of the setpoint-ablation selection — exact shipped semantics — plus the
blind-only variant (no structure guard), so the report can state per regime: which inference
configuration the extrapolated anchor's blind score alone would have chosen.

Semantics copied verbatim from setpoint_ablation.py:
  band  = cells with lo <= blind <= hi
  guard = keep cells with proxy >= proxy(K1-50x12, base) - 0.20      [shipped rule only]
  none survive -> NO-BAND fallback: nearest blind to the set-point among ALL cells
  else nearest set-point; ties within 0.02 -> shallower rung, then less-trained model
Validated below against every pick printed in the original run's log before being used.

Outputs base_results/ablation_choice_replay.json for the report builder.
"""
import os, sys, json
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np

d = np.load('base_results/setpoint_ablation.npz', allow_pickle=True)
SEL, AUD = {}, {}
for f in d.files:
    if f.startswith('S|'):
        k, fld = f[2:].split('||'); SEL.setdefault(k, {})[fld] = float(d[f])
    elif f.startswith('A|'):
        k, fld = f[2:].split('||'); AUD.setdefault(k, {})[fld] = float(d[f])

RUNG_ORDER = ['K1-50x12', 'K1-75x20', 'K1-100x20', 'K2x50', 'K3x86', 'K4x110', 'K5x140',
              'K6x170', 'K7x200']
ARMS = {'without_re500': (0.784, 0.764, 0.804), 'with_re500': (0.810, 0.770, 0.850)}
PROXY_THR, TIE = 0.20, 0.02
REGS = [1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000]


def _train_amount(name):
    if name == 'base':
        return -1
    digits = ''.join(c for c in name if c.isdigit())
    return int(digits) if digits else 10**6


def select(R, sp, lo, hi, use_guard):
    cells = [k for k in SEL if k.startswith(f'{R}|') and not k.endswith('_picks')]
    band = [k for k in cells if lo <= SEL[k]['blind'] <= hi]
    if use_guard:
        ref = SEL[f'{R}|K1-50x12|base']['proxy']
        surv = [k for k in band if SEL[k]['proxy'] >= ref - PROXY_THR]
    else:
        surv = band
    if not surv:
        near = min(cells, key=lambda k: abs(SEL[k]['blind'] - sp))
        return 'NO-BAND', near
    dmin = min(abs(SEL[k]['blind'] - sp) for k in surv)
    tied = [k for k in surv if abs(SEL[k]['blind'] - sp) <= dmin + TIE]
    tied.sort(key=lambda k: (RUNG_ORDER.index(k.split('|')[1]), _train_amount(k.split('|')[2])))
    return 'IN-BAND', tied[0]


# ---- validation: the guarded replay must reproduce every pick the original run printed ----
LOGGED = {  # regime -> arm -> (label, 'rung|model'), transcribed from the run log
    1500: {'without_re500': ('NO-BAND', 'K3x86|r1k-449'), 'with_re500': ('NO-BAND', 'K3x86|r1k-149')},
    2000: {'without_re500': ('IN-BAND', 'K3x86|0149'),   'with_re500': ('IN-BAND', 'K3x86|0149')},
    3000: {'without_re500': ('NO-BAND', 'K4x110|r1k-149'), 'with_re500': ('NO-BAND', 'K5x140|r1k-449')},
    4000: {'without_re500': ('NO-BAND', 'K5x140|r1k-449'), 'with_re500': ('NO-BAND', 'K5x140|r1k-449')},
    5000: {'without_re500': ('NO-BAND', 'K5x140|r1k-449'), 'with_re500': ('NO-BAND', 'K5x140|r1k-449')},
    6000: {'without_re500': ('NO-BAND', 'K5x140|r1k-149'), 'with_re500': ('NO-BAND', 'K5x140|r1k-149')},
    7000: {'without_re500': ('NO-BAND', 'K5x140|r1k-149'), 'with_re500': ('NO-BAND', 'K5x140|r1k-149')},
    8000: {'without_re500': ('NO-BAND', 'K5x140|r1k-149'), 'with_re500': ('NO-BAND', 'K5x140|r1k-149')},
}
mismatch = 0
for R in REGS:
    for arm, (sp, lo, hi) in ARMS.items():
        lab, k = select(R, sp, lo, hi, use_guard=True)
        got = (lab, k.split('|', 1)[1])
        if got != LOGGED[R][arm]:
            mismatch += 1
            print(f"VALIDATION MISMATCH Re={R} {arm}: replay {got} vs logged {LOGGED[R][arm]}")
if mismatch:
    sys.exit(f"{mismatch} mismatches — replay semantics NOT faithful, do not build the report")
print("VALIDATION OK: guarded replay reproduces all 16 logged picks exactly\n")

out = {}
for R in REGS:
    out[R] = {}
    for arm, (sp, lo, hi) in ARMS.items():
        for mode, guard in (('shipped', True), ('blind_only', False)):
            lab, k = select(R, sp, lo, hi, use_guard=guard)
            _, rung, mdl = k.split('|')
            cell = dict(label=lab, rung=rung, model=mdl, blind=SEL[k]['blind'],
                        proxy=SEL[k]['proxy'])
            a = AUD.get(f'{R}|{rung}|{mdl}')
            if a:
                cell.update(ret=a['ret'], place=a['place'], kstar=a['kstar'])
            out[R][f'{arm}|{mode}'] = cell
            g = 'guarded' if guard else 'blind-only'
            gr = f" ret={cell.get('ret', float('nan')):.3f}" if a else " (unaudited)"
            print(f"Re={R} [{arm:>13}] {g:>10}: {lab:>7} {rung}|{mdl} "
                  f"blind={cell['blind']:.3f}{gr}")
json.dump(out, open('base_results/ablation_choice_replay.json', 'w'), indent=1)
print("\nREPLAY COMPLETE -> base_results/ablation_choice_replay.json")
