"""Rebuild base_results/setpoint_ablation.npz from the run log.

The disk filled mid-save and truncated the archive. All cell values were also printed to the log,
so this reconstructs the file from there. The log briefly had TWO writers (a stale duplicate
process, since killed), so parsing is deliberately strict:
  - selection/audit lines must match exact-format regexes with a known cascade/model vocabulary
    and values in physical range;
  - cells attach to the most recent NEW-format section header ('3 models x 9 rungs'); sections
    from the stale process ('1 models') are skipped;
  - any regime whose reconstructed cell count exceeds the possible maximum is dropped entirely
    (the resume run then recomputes it from scratch, which is always safe).
Missing cells are harmless: the resumed run recomputes exactly what is absent.
"""
import re, sys
import numpy as np

LOG = 'monitoring/ab_pdelocal/setpoint_ablation.log'
CASC = {'K1-50x12', 'K1-75x20', 'K1-100x20', 'K2x50', 'K3x86', 'K4x110', 'K5x140', 'K6x170', 'K7x200'}
MODELS = {'base', 'r1k-149', 'r1k-449'} | {f'{i:04d}' for i in range(0, 600, 50)}
sel_re = re.compile(r'^  (K[0-9x\-]+)\s+(\S+)\s+blind=([01]\.\d{3}) proxy=([01]\.\d{3})\s*$')
aud_re = re.compile(r'^    (K[0-9x\-]+)\s+(\S+)\s+ret=(\d\.\d{3}) place=([01]\.\d{3}) k\*=(\d+)\s*$')
hdr_re = re.compile(r'^=== Re=(\d+): SELECTION on seqs .* (\d) models x 9 rungs ===')

SEL, AUD = {}, {}
regime, live = None, False
for line in open(LOG, errors='replace'):
    m = hdr_re.match(line)
    if m:
        regime, live = m.group(1), (m.group(2) == '3')
        continue
    if not live or regime is None:
        continue
    m = sel_re.match(line)
    if m and m.group(1) in CASC and m.group(2) in MODELS:
        b, p = float(m.group(3)), float(m.group(4))
        if 0 <= b <= 2 and 0 <= p <= 1.01:
            SEL[f'{regime}|{m.group(1)}|{m.group(2)}'] = dict(blind=b, proxy=p)
        continue
    m = aud_re.match(line)
    if m and m.group(1) in CASC and m.group(2) in MODELS:
        r, pl = float(m.group(3)), float(m.group(4))
        if 0 <= r <= 8 and 0 <= pl <= 1.01:
            AUD[f'{regime}|{m.group(1)}|{m.group(2)}'] = dict(
                ret=r, place=pl, lowk=np.nan, kstar=int(m.group(5)))

import collections
cs = collections.Counter(k.split('|')[0] for k in SEL)
ca = collections.Counter(k.split('|')[0] for k in AUD)
for R, n in list(cs.items()):
    if n > 27:
        print(f'regime {R}: {n} selection cells > 27 possible -> dropping regime (will recompute)')
        for k in [k for k in SEL if k.startswith(f'{R}|')]:
            del SEL[k]
        for k in [k for k in AUD if k.startswith(f'{R}|')]:
            del AUD[k]
print('reconstructed selection cells per regime:', dict(sorted(collections.Counter(
    k.split('|')[0] for k in SEL).items())))
print('reconstructed audit cells per regime:', dict(sorted(collections.Counter(
    k.split('|')[0] for k in AUD).items())))
np.savez('base_results/setpoint_ablation.npz',
         sel_keys=np.array(list(SEL)), aud_keys=np.array(list(AUD)), picks=np.array([]),
         **{f'S|{k}||{f}': np.float32(v) for k, d in SEL.items() for f, v in d.items()},
         **{f'A|{k}||{f}': np.float32(v) for k, d in AUD.items() for f, v in d.items()})
print('rebuilt npz written')
