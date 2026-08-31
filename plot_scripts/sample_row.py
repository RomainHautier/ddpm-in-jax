"""One reconstruction, every model, side by side — the same triplet at a given regime.

Panels: ground truth, then the base model and each fine-tune, all UNGUIDED, so the differences
are what the weights alone produce. Shared colour scale across the row, set from the ground truth,
so the panels are directly comparable rather than each normalised to itself.

  JAX_PLATFORMS=cpu python plot_scripts/sample_row.py --re 2000 [--idx 12] [--no-gt]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())          # so `src` resolves after the chdir
import numpy as np
import matplotlib.pyplot as plt
import style

SIG = 4.7988
CONFIG = dict(re=2000, idx=None, panel=2.9, fontsize=style.BASE_FS, dpi=170,
              outdir='docs/figs_overleaf', pngdir='plotting/figs')
# base, the Re=1000 fine-tune, and the model trained AT THIS REGIME. Cross-regime transfer of the
# 2000/8000 fine-tunes is deliberately excluded: showing mt8k at Re=2000 (or mt2k at Re=8000) mixes
# a transfer question into a figure about local training. The Re=8000 slot uses the REPAIRED run
# (pde weight 0.2); the matched mt8k drains the large scales and is not a usable model.
HOME = {2000: ('mt2k-0599', 'Re=2000 fine-tune\n(trained here)'),
        8000: ('r8kp02-0599', 'Re=8000 fine-tune\n(trained here, repaired)')}


def models_for(R):
    rows = [('base0', 'base'), ('mt1k-0499', 'Re=1000 fine-tune')]
    if R in HOME: rows.append(HOME[R])
    return rows


def load(R, name):
    p = f'base_results/fields/re{R}/{name}.npz'
    return np.load(p)['x'].astype(np.float32) if os.path.exists(p) else None


def align_gt(R, gt, n):
    """The stored GT is the FULL audit pool (seqs 8-19 OOD, 120 triplets); the model fields are
    the TEST pool only (12-19, 80). Indexing both with the same integer pairs a reconstruction
    with a DIFFERENT snapshot's truth. Mask the GT by the stored per-triplet seq id."""
    if len(gt) == n: return gt
    S = np.load('base_results/re1000_audit.npz' if R == 1000
                else f'base_results/regime_audit_re{R}.npz', allow_pickle=True)
    k = f'{R}|EVAL||seq'
    assert k in S.files, f'no EVAL mask at Re={R}; cannot align {len(gt)} GT to {n} fields'
    sq = np.asarray(S[k]); m = sq >= 12
    assert m.sum() == n, f'mask gives {m.sum()} triplets, fields have {n}'
    return gt[m]


def make(c):
    style.apply(c['fontsize'], c['dpi'])
    R = c['re']
    gt = load(R, 'GT')
    fields = [(lab, load(R, f'{m}__K3__mop0.2_0')) for m, lab in models_for(R)]
    fields = [(l, f) for l, f in fields if f is not None]
    gt = align_gt(R, gt, fields[0][1].shape[0])
    # verify the pairing rather than trust it: a correctly paired reconstruction correlates
    # strongly with its own truth at the large scales, a mispaired one does not
    _b = fields[0][1]
    _c = float(np.corrcoef(gt[:, ..., 1].reshape(len(gt), -1).mean(0),
                           _b[:, ..., 1].reshape(len(_b), -1).mean(0))[0, 1])
    _per = np.array([np.corrcoef(gt[j, ..., 1].ravel(), _b[j, ..., 1].ravel())[0, 1]
                     for j in range(min(len(gt), 20))])
    print(f'  alignment check: median per-triplet corr(GT, base recon) = {np.median(_per):.3f} '
          f'(expect >0.8; ~0 means mispaired)')
    assert np.median(_per) > 0.5, 'GT and reconstructions are NOT aligned'
    # pick the triplet whose GT fine-band energy is closest to the pool median: a representative
    # sample, not a flattering one
    i = c['idx']
    if i is None:
        from src.rewards import make_spectrum_fn
        import jax.numpy as jnp
        E = np.asarray(make_spectrum_fn(256)(jnp.asarray(gt)))[:, 32:96].sum(1)
        i = int(np.argsort(E)[len(E) // 2])
    panels = ([('ground truth', gt)] if c['show_gt'] else []) + fields
    n = len(panels)
    fig, ax = plt.subplots(1, n, figsize=(c['panel'] * n, c['panel'] * 1.16),
                           constrained_layout=True)
    w_gt = gt[i, ..., 1] * SIG
    v = float(np.percentile(np.abs(w_gt), 99.5))
    for a, (lab, f) in zip(np.atleast_1d(ax), panels):
        a.imshow(f[i, ..., 1] * SIG, cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
        a.set_title(lab, fontsize=style.TITLE_FS)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(True)
    fig.suptitle(f'The same reconstruction at Re={R} (triplet {i}, middle frame), '
                 'every model unguided', fontsize=style.SUP_FS)
    name = f'sample_row_re{R}'
    for d, ext in ((c['outdir'], 'pdf'), (c['pngdir'], 'png')):
        os.makedirs(d, exist_ok=True)
        p = f'{d}/{name}.{ext}'; fig.savefig(p); print('  written', p)
    plt.close(fig)
    print(f'  triplet {i} of {len(gt)}, colour scale +-{v:.2f} (99.5th pct of |GT vorticity|)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--re', type=int, default=2000); ap.add_argument('--idx', type=int)
    ap.add_argument('--no-gt', action='store_true')
    a = ap.parse_args(); c = dict(CONFIG)
    c['re'] = a.re; c['idx'] = a.idx; c['show_gt'] = not a.no_gt
    make(c)
