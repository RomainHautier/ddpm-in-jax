"""The sampling chain as a dose control: downward transfer fixed without any guidance.

r8kp02 (trained at Re=8000) and mt2k (Re=2000) carried DOWN to Re=1000 overshoot the fine band
under the standard K3 chain, and no guidance strength fixes it without destroying the physics.
Shortening the chain removes the surplus at the source: each re-noising pass regenerates fine
scale energy, so the ladder of chain configurations is a monotone dose dial with no gradient
anywhere in it.

  JAX_PLATFORMS=cpu python plot_scripts/chain_dose.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

CHAINS = [('K3', 'K3 [150,100,50]'), ('K150-100', '[150,100]'), ('K150', '[150]'),
          ('K100', '[100]'), ('K50', '[50]')]
MODELS = [('r8kp02-0599', 'Re=8000 ft at Re=1000', 'D'),
          ('mt2k-0599', 'Re=2000 ft at Re=1000', '^')]
S = np.load('base_results/re1000_audit.npz', allow_pickle=True)


def get(m, ck, f):
    # K3 rows are the unguided dial rows; chain-probe rows carry the fc tag
    for sg in (('mop0.2_0',) if ck == 'K3' else ('fcp0.2_0',)):
        k = f'1000|{m}|{ck}|{sg}||{f}'
        if f == 'inband':
            for ff in ('pst_ret', 'ps_ret_paired'):
                kk = f'1000|{m}|{ck}|{sg}||{ff}'
                if kk in S.files: return float(np.mean(np.abs(np.asarray(S[kk]) - 1) < .2)) * 100
            return np.nan
        if k in S.files: return float(S[k])
    return np.nan


def make():
    style.apply(style.BASE_FS, 150)
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 4.2), constrained_layout=True)
    x = np.arange(len(CHAINS))
    for f, lab, i, ref in (('ret', 'retention $E_{[32,96)}/E_{\\mathrm{GT}}$', 0, 1),
                           ('inband', '% within $\\pm20\\%$ of own truth', 1, None),
                           ('resid_ratio', 'PDE residual / ground truth', 2, 1)):
        a = ax[i]
        for m, ml, mk in MODELS:
            y = [get(m, ck, f) for ck, _ in CHAINS]
            a.plot(x, y, '-', color=style.MODEL_COLOR[m], lw=1.9, marker=mk, ms=5,
                   label=ml if i == 0 else None)
        if ref: a.axhline(ref, color=style.GT_COLOR, lw=1.5)
        a.set_xticks(x); a.set_xticklabels([l for _, l in CHAINS], fontsize=style.LEG_FS - 1,
                                           rotation=15)
        a.set_title(lab, fontsize=style.TITLE_FS)
        a.set_xlabel('sampling chain')
    ax[0].set_ylim(0, 4); ax[1].set_ylim(0, 100); ax[2].set_ylim(0, 3.2)
    ax[0].legend(fontsize=style.LEG_FS, loc='upper right')
    fig.suptitle('Shortening the chain removes surplus energy at the source: downward transfer '
                 'without any guidance (Re=1000 validation pool)', fontsize=style.SUP_FS)
    for d, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        os.makedirs(d, exist_ok=True)
        p = f'{d}/chain_dose.{ext}'; fig.savefig(p); print('  written', p)


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    make()
