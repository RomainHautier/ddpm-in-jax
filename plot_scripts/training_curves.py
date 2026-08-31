"""DDPO training curves: reward, its raw component distances, and the GT retention probe.

Parses the training logs directly, so any run can be compared with any other. Note the total
reward R is NOT comparable across runs with different weight sets (a run carrying align=2.0
adds a term the matched runs do not have); the per-component RAW distances below it are
comparable, because they are the unweighted, unnormalised d_i.

  python plot_scripts/training_curves.py --regime 1000
"""
import os, sys, re, argparse
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts')
import numpy as np, style
import matplotlib.pyplot as plt

CONFIG = dict(
    runs={
        1000: [('monitoring/match_re1000_ft.log', 'matched (measured anchor, no align)', '#0f9e78'),
               ('monitoring/ddpo_re1000_newpool.log', 'newpool = r1k-449 (obsfit, align 2.0)', '#28658a'),
               ('monitoring/ddpo_re1000_temp20.log', 'temp20 (seqs 32,33)', '#9aa198')],
        2000: [('monitoring/match_re2000_ft.log', 'matched (measured anchor, no align)', '#0f9e78'),
               ('monitoring/re2000_realstats_ft.log', 'realstats (measured, [32,96))', '#28658a'),
               ('monitoring/ddpo_re2000_newpool.log', 'newpool (obsfit)', '#9aa198')],
        8000: [('monitoring/match_re8000_ft.log', 'matched (measured anchor, no align)', '#0f9e78'),
               ('monitoring/re8000_realstats_ft.log', 'realstats (temp 2.5)', '#28658a'),
               ('monitoring/re8000_t35fresh_ft.log', 't35fresh (temp 3.5)', '#9aa198')]},
    smooth=9, out='plotting/figs/training_curves_re{R}.pdf')
p = argparse.ArgumentParser(); p.add_argument('--regime', type=int, default=1000); p.add_argument('--out')
a = p.parse_args()
R = a.regime
if a.out: CONFIG['out'] = a.out


def parse(path):
    it, comp, rew, probe = [], {}, [], []
    if not os.path.exists(path): return None
    for L in open(path, errors='ignore'):
        m = re.match(r'\[(\d+)\]\s+R=(-?[\d.]+)', L)
        if m:
            it.append(int(m.group(1))); rew.append(float(m.group(2)))
            for k, v in re.findall(r'\b(\w+)=(-?[\d.eE+]+)', L):
                if k in ('R', 'gstd', 'loss'): continue
                comp.setdefault(k, []).append(float(v))
        g = re.search(r'\[GTeval iter=(\d+)\]\s+hik_ret=([\d.]+)', L)
        if g: probe.append((int(g.group(1)), float(g.group(2))))
    b = re.search(r'GT-probe base hik_ret = ([\d.]+)', open(path, errors='ignore').read())
    return dict(it=np.array(it), rew=np.array(rew), comp={k: np.array(v) for k, v in comp.items()},
                probe=np.array(probe), base=float(b.group(1)) if b else np.nan)


def smooth(y, w):
    if len(y) < w: return y
    return np.convolve(y, np.ones(w) / w, mode='valid')


runs = [(lab, c, parse(pth)) for pth, lab, c in CONFIG['runs'][R]]
runs = [(l, c, d) for l, c, d in runs if d is not None and len(d['it'])]
style.apply()
PANELS = [('rew', 'total reward R   (composition differs between runs)'),
          ('spec_highk', f'd_spec_highk   the heavy term'),
          ('spec', 'd_spec   [1,96)'),
          ('pde', 'd_pde   residual'),
          ('energy', 'd_energy   enstrophy'),
          ('probe', 'GT probe: fine-band retention E[32,96)/GT')]
fig, AX = plt.subplots(2, 3, figsize=(16, 8))
for ax, (key, title) in zip(AX.ravel(), PANELS):
    for lab, c, d in runs:
        if key == 'rew':
            x, y = d['it'], d['rew']
        elif key == 'probe':
            if not len(d['probe']): continue
            x, y = d['probe'][:, 0], d['probe'][:, 1]
            ax.axhline(d['base'], color=c, ls=':', lw=1.2)
        else:
            if key not in d['comp']: continue
            x, y = d['it'][:len(d['comp'][key])], d['comp'][key]
        w = 1 if key == 'probe' else CONFIG['smooth']
        ys = smooth(y, w); xs = x[w - 1:] if w > 1 else x
        ax.plot(xs, ys, color=c, lw=1.7, label=lab)
    ax.set_title(title, fontsize=style.TITLE_FS); ax.set_xlabel('outer iteration')
    if key == 'probe':
        ax.plot([], [], color=style.INK, ls=':', label='each run\'s base model')
        ax.set_ylabel('retention')
    ax.legend(fontsize=style.LEG_FS, loc='best')
fig.suptitle(f'DDPO training at Re={R} — reward, raw component distances, and the GT probe'
             f'   (curves smoothed over {CONFIG["smooth"]} iterations)', fontsize=12, y=.98)
fig.tight_layout(rect=[0, 0, 1, .955])
out = CONFIG['out'].format(R=R)
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=145)
print('wrote', out)
for lab, c, d in runs:
    n = len(d['it'])
    print(f"  {lab:<44} {n:>4} iters | R final {np.mean(d['rew'][-20:]):+.2f}"
          f" | spec_highk {np.mean(d['comp'].get('spec_highk', [np.nan])[-20:]):.3f}"
          f" | probe {d['probe'][-1][1] if len(d['probe']) else float('nan'):.3f} (base {d['base']:.3f})")
