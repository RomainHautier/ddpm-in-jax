"""In-distribution (Re=1000) comparison ladder: base, base+steering, fine-tune, fine-tune+steering.

Two figures, matching the two ways of judging the same reconstructions:
  indist_ensemble.*  - the ensemble view (mean spectra, ratio to GT, per-band retention/placement)
  indist_perframe.*  - the per-triplet view (dose against each triplet's own truth, agreement)

  python plot_scripts/indist_ladder.py [--out-dir plotting/figs]
"""
import os, sys, argparse
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts')
import numpy as np, style
import matplotlib.pyplot as plt

CONFIG = dict(
    store='base_results/re1000_audit.npz', regime=1000,
    # The matched comparison only: base and the matched Re=1000 fine-tune, each with and without
    # the matched dial (tapered). Colour = model, linestyle = dial, as in the other Re=1000
    # figures. The v2 dose dial and the gated stack optimise different objectives and are not
    # part of this comparison.
    rows=[('base0', 'mop0.2_0', 'base, unguided', '#28658a', '--'),
          ('base0', 'tapp0.2_3', 'base + dial', '#28658a', '-'),
          ('mt1k-0499', 'mop0.2_0', 'fine-tune, unguided', '#c22f4f', '--'),
          ('mt1k-0499', 'tapp0.2_3', 'fine-tune + dial', '#c22f4f', '-')],
    tol=0.2, out_dir='plotting/figs',
)
p = argparse.ArgumentParser(); p.add_argument('--out-dir')
p.add_argument('--band', default='32,96', help="the band the per-triplet dose is measured over, "
               "e.g. 16,96 - must align with the stored band edges 1/5/16/32/64/96")
a = p.parse_args()
if a.out_dir: CONFIG['out_dir'] = a.out_dir
BLO, BHI = (int(x) for x in a.band.split(','))
EDGES = [1, 5, 16, 32, 64, 96]
COLS = [i for i in range(5) if EDGES[i] >= BLO and EDGES[i + 1] <= BHI]
assert COLS, f'band {a.band} does not align with the stored edges {EDGES}'
BAND = f'[{BLO},{BHI})'
SUF = '' if (BLO, BHI) == (32, 96) else f'_k{BLO}'
print(f'per-triplet dose measured over {BAND}  (psEb columns {COLS})')

A = np.load(CONFIG['store'], allow_pickle=True)
R = CONFIG['regime']
g = lambda m, sg, f: (np.asarray(A[f'{R}|{m}|K3|{sg}||{f}'])
                      if f'{R}|{m}|K3|{sg}||{f}' in A.files else None)
ROWS = [r for r in CONFIG['rows'] if g(r[0], r[1], 'ret') is not None]
Eg = np.asarray(A[f'{R}|GT||E']); k = np.arange(1, 128)
style.apply()

# ================= FIGURE 1: the ensemble view =================
fig, AX = plt.subplots(1, 3, figsize=(16.5, 4.6))
ax = AX[0]
ax.loglog(k, Eg[1:128], color=style.GT_COLOR, lw=style.GT_LW, label='ground truth', zorder=5)
for m, sg, lab, c, ls in ROWS:
    E = g(m, sg, 'E')
    if E is not None: ax.loglog(k, np.asarray(E)[1:128], ls, color=c, lw=1.7, label=lab)
style.shade_bands(ax); ax.axvline(96, color=style.INK, lw=1, ls=':')
ax.set_xlabel('wavenumber k'); ax.set_ylabel('E(k)')
ax.set_title('mean vorticity spectrum', fontsize=style.TITLE_FS); ax.legend(fontsize=style.LEG_FS, loc='lower left')

ax = AX[1]
ax.axhline(1, color=style.GT_COLOR, lw=style.GT_LW, zorder=5, label='ground truth')
for m, sg, lab, c, ls in ROWS:
    E = g(m, sg, 'E')
    if E is not None:
        ax.semilogx(k, np.asarray(E)[1:128] / Eg[1:128], ls, color=c, lw=1.7, label=lab)
style.shade_bands(ax); ax.axvline(96, color=style.INK, lw=1, ls=':')
ax.set_ylim(0, 2.2); ax.set_xlabel('wavenumber k'); ax.set_ylabel('E(k) / E_GT(k)')
ax.set_title('ratio to ground truth', fontsize=style.TITLE_FS); ax.legend(fontsize=style.LEG_FS, loc='upper left')

ax = AX[2]  # hatch = unguided, solid = dialled (colour still separates the models)
w = 0.8 / len(ROWS)
for j, (m, sg, lab, c, ls) in enumerate(ROWS):
    eb = g(m, sg, 'Eb')
    if eb is None: continue
    hh = '///' if ls == '--' else None
    ax.bar(np.arange(5) + (j - (len(ROWS) - 1) / 2) * w, eb, w, color=c, label=lab,
           alpha=.45 if hh else .95, hatch=hh, edgecolor=c, lw=.9)
ax.axhline(1, color=style.GT_COLOR, lw=2.2, zorder=5)
ax.set_xticks(range(5)); ax.set_xticklabels(style.BAND_LABELS, fontsize=8)
ax.set_ylabel('band energy / GT'); ax.set_ylim(0, 1.55)
ax.set_title('per-band energy retention', fontsize=style.TITLE_FS); ax.legend(fontsize=style.LEG_FS, loc='upper left', ncol=2)
fig.suptitle(f'In distribution (Re={R}) — the ensemble view: every steered and fine-tuned '
             'configuration matches the mean spectrum', fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0, 1, .95])
o1 = os.path.join(CONFIG['out_dir'], f'indist_ensemble{SUF}.pdf')
fig.savefig(o1, bbox_inches='tight'); fig.savefig(o1.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)

# ================= FIGURE 2: the per-triplet view =================
gtE = A[f'{R}|GT||psEb']; t = gtE[:, COLS].sum(1)
fig, AX = plt.subplots(1, 3, figsize=(16.5, 4.6))
ax = AX[0]
lo, hi = t.min() * .8, t.max() * 1.25
ax.fill_between([lo, hi], [lo * .8, hi * .8], [lo * 1.2, hi * 1.2], color='k', alpha=.07, lw=0)
ax.plot([lo, hi], [lo, hi], color=style.GT_COLOR, lw=2.2, zorder=5, label='perfect (slope 1)')
for m, sg, lab, c, ls in ROWS:
    # From the stored per-band energies for the CHOSEN band. ps_ret_paired is [32,96) ONLY, so
    # using it here plotted ret_[32,96) * E_GT[16,96) on a --band 16,96 figure - two different
    # bands multiplied together, and the reported slopes were wrong. The histogram panel already
    # recomputes from psEb; this now matches it.
    pE = g(m, sg, 'psEb')
    if pE is None: continue
    P = pE[:, COLS].sum(1)
    sl = np.polyfit(np.log(t), np.log(P), 1)[0]
    # marker separates unguided from dialled; colour still separates the models
    mk, sz, al = ('o', 7, .35) if ls == '--' else ('^', 13, .6)
    ax.scatter(t, P, s=sz, marker=mk, color=c, alpha=al, lw=0,
               label=f'{lab}  (slope {sl:.2f})')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo * .55, hi * 2.6)
ax.set_xlabel(f"that triplet's own GT energy in {BAND}"); ax.set_ylabel(f'reconstructed energy in {BAND}')
ax.set_title('per-triplet dose', fontsize=style.TITLE_FS); ax.legend(fontsize=style.LEG_FS, loc='upper left')

ax = AX[1]
bins = np.linspace(-0.9, 1.4, 46)
for m, sg, lab, c, ls in ROWS:
    # recomputed for the CHOSEN band from the stored per-band energies, so the histogram and the
    # scatter always describe the same quantity (pst_ret / ps_ret_paired are [32,96) only)
    pE = g(m, sg, 'psEb')
    if pE is None: continue
    pr = pE[:, COLS].sum(1) / t
    ib = np.mean(np.abs(pr - 1) < CONFIG['tol']) * 100
    ax.hist(np.log(pr), bins=bins, histtype='step', lw=1.9, color=c, label=f'{lab} — {ib:.0f}%')
ax.axvspan(np.log(.8), np.log(1.2), color='k', alpha=.07, lw=0)
ax.axvline(0, color=style.GT_COLOR, lw=2.2)
tk = np.array([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
ax.set_xticks(np.log(tk)); ax.set_xticklabels([f'{x:g}x' for x in tk])
ax.set_xlabel(f"reconstructed / that triplet's own truth, over {BAND}"); ax.set_ylabel('triplets')
ax.set_title('per-triplet agreement (share within +-20%)', fontsize=style.TITLE_FS); ax.legend(fontsize=style.LEG_FS)

ax = AX[2]
for j, (m, sg, lab, c, ls) in enumerate(ROWS):
    ps = g(m, sg, 'ps_bp')
    x = np.arange(5) + (j - (len(ROWS) - 1) / 2) * w
    if ps is None:
        ax.bar(x, np.zeros(5), w, color=c, alpha=.25, hatch='//', edgecolor=c, label=lab + ' (pending)')
    else:
        hh = '///' if ls == '--' else None
        ax.bar(x, np.median(ps, 0), w, color=c, label=lab,
               alpha=.45 if hh else .95, hatch=hh, edgecolor=c, lw=.9)
ax.set_xticks(range(5)); ax.set_xticklabels(style.BAND_LABELS, fontsize=8)
ax.set_ylim(0.5, 1.02); ax.set_ylabel('placement per triplet (median)')
ax.set_title('per-triplet placement by band', fontsize=style.TITLE_FS); ax.legend(fontsize=style.LEG_FS, loc='lower left')
fig.suptitle(f'In distribution (Re={R}) — the per-triplet view over {BAND}: the same configurations are far apart',
             fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0, 1, .95])
o2 = os.path.join(CONFIG['out_dir'], f'indist_perframe{SUF}.pdf')
fig.savefig(o2, bbox_inches='tight'); fig.savefig(o2.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
print('wrote', o1, 'and', o2)
