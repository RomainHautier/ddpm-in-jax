"""Total energy, the counterpart to the fine-band dose figure.

The fine band [32,96) carries about 1% of the ground truth's energy, so a model can miss half of
it and still hold ~99.4% of the total. This shows the same per-triplet analysis applied to TOTAL
energy instead, alongside where the energy actually lives.

  JAX_PLATFORMS=cpu python plot_scripts/total_energy.py
"""
import os, sys, argparse
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts')
import numpy as np, style
import matplotlib.pyplot as plt

CONFIG = dict(
    store='base_results/re1000_audit.npz', regime=1000, tol=0.05,
    rows=[('base0', 'none', 'base, unguided', '#9aa198'),
          ('base0', 'mop0.2_3', 'base + B dial', '#c22f4f'),
          ('base0', 'v6gate', 'base + gated dial', '#0f9e78'),
          ('mt1k-0499', 'mo0', 'matched fine-tune', '#28658a')],
    out='plotting/figs/total_energy.pdf')
p = argparse.ArgumentParser(); p.add_argument('--out'); a = p.parse_args()
if a.out: CONFIG['out'] = a.out
A = np.load(CONFIG['store'], allow_pickle=True); R = CONFIG['regime']
g = lambda m, sg, f: (np.asarray(A[f'{R}|{m}|K3|{sg}||{f}'])
                      if f'{R}|{m}|K3|{sg}||{f}' in A.files else None)
ROWS = [r for r in CONFIG['rows'] if g(r[0], r[1], 'psEb') is not None]
gtE = np.asarray(A[f'{R}|GT||psEb'])            # (n, 5) per-triplet band energies
Eg = np.asarray(A[f'{R}|GT||E'])
BL, TOT = style.BAND_LABELS, gtE.sum(1)
style.apply()
fig, AX = plt.subplots(1, 3, figsize=(16.5, 4.7))

# --- 1. where the energy actually is -----------------------------------------------------------
share = [Eg[lo:hi].sum() / Eg[1:].sum() * 100 for lo, hi in style.BANDS]
b = AX[0].bar(range(5), share, color=['#9aa198'] * 3 + ['#c22f4f'] * 2)
for i, v in enumerate(share):
    AX[0].text(i, v + 1.2, f'{v:.2f}%', ha='center', fontsize=8.5)
AX[0].set_xticks(range(5)); AX[0].set_xticklabels(BL, fontsize=8.5)
AX[0].set_ylabel('share of the ground truth\'s total energy (%)'); AX[0].set_ylim(0, 70)
AX[0].set_title(f'the fine band [32,96) is {sum(share[3:]):.2f}% of the total', fontsize=style.TITLE_FS)

# --- 2. per-triplet TOTAL energy ---------------------------------------------------------------
lo, hi = TOT.min() * .93, TOT.max() * 1.07
AX[1].fill_between([lo, hi], [lo * .95, hi * .95], [lo * 1.05, hi * 1.05], color='k', alpha=.07, lw=0)
AX[1].plot([lo, hi], [lo, hi], color=style.GT_COLOR, lw=2, zorder=5, label='perfect')
for m, sg, lab, c in ROWS:
    y = g(m, sg, 'psEb').sum(1)
    sl = np.polyfit(np.log(TOT), np.log(y), 1)[0]
    AX[1].scatter(TOT, y, s=11, color=c, alpha=.6, lw=0, label=f'{lab}  (slope {sl:.2f})')
AX[1].set_xscale('log'); AX[1].set_yscale('log'); AX[1].set_xlim(lo, hi); AX[1].set_ylim(lo, hi)
AX[1].set_xlabel("that triplet's own total GT energy"); AX[1].set_ylabel('reconstructed total energy')
AX[1].set_title('per-triplet TOTAL energy (shaded = within +-5%)', fontsize=style.TITLE_FS)
AX[1].legend(fontsize=style.LEG_FS, loc='upper left')

# --- 3. the ratio distribution -----------------------------------------------------------------
bins = np.linspace(-0.20, 0.12, 46)
for m, sg, lab, c in ROWS:
    r = g(m, sg, 'psEb').sum(1) / TOT
    w5 = np.mean(np.abs(r - 1) < 0.05) * 100
    w10 = np.mean(np.abs(r - 1) < 0.10) * 100
    AX[2].hist(np.log(r), bins=bins, histtype='step', lw=1.9, color=c,
               label=f'{lab} — {w5:.0f}% / {w10:.0f}% within ±5% / ±10%')
AX[2].axvspan(np.log(0.90), np.log(1.10), color='k', alpha=.045, lw=0)   # the wider tolerance
AX[2].axvspan(np.log(0.95), np.log(1.05), color='k', alpha=.07, lw=0)
AX[2].axvline(0, color=style.GT_COLOR, lw=2)
tk = np.array([0.85, 0.90, 0.95, 1.0, 1.05, 1.10])
AX[2].set_xticks(np.log(tk)); AX[2].set_xticklabels([f'{t:g}' for t in tk])
AX[2].set_xlabel('reconstructed / true TOTAL energy'); AX[2].set_ylabel('triplets')
AX[2].set_title('per-triplet total-energy agreement (bands: ±5%, ±10%)', fontsize=style.TITLE_FS)
AX[2].legend(fontsize=style.LEG_FS, loc='upper left')
fig.suptitle(f'Total energy, Re={R}, {len(TOT)} held-out triplets — the same analysis as the fine '
             'band, on the quantity that carries the energy', fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0, 1, .94])
os.makedirs(os.path.dirname(CONFIG['out']), exist_ok=True)
fig.savefig(CONFIG['out'], bbox_inches='tight')
fig.savefig(CONFIG['out'].replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
print('wrote', CONFIG['out'])
for m, sg, lab, c in ROWS:
    y = g(m, sg, 'psEb').sum(1); r = y / TOT
    print(f"  {lab:<24} total {y.sum()/TOT.sum():.4f}  median {np.median(r):.4f}  "
          f"within5% {np.mean(np.abs(r-1)<.05)*100:>3.0f}%  within10% {np.mean(np.abs(r-1)<.10)*100:>3.0f}%  "
          f"slope {np.polyfit(np.log(TOT),np.log(y),1)[0]:.2f}")
