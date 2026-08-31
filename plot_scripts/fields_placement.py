"""Where does the energy go? Analyses on STORED held-out fields (base_results/fields/re1000/*.npz,
written by the audit; no inference needed). Per sample and per band:
  - map correlation of smoothed band-energy maps vs ground truth (the placement metric, per frame)
  - overlap of the top-q% high-energy pixels with the ground truth's (Jaccard) - 'are the hot
    spots in the right places?'
  - spatial cross-correlation peak offset (is structure shifted?)
Edit CONFIG (rows, bands, top-q, smoothing) or pass --rows/--out. Fields are float16 normalized
vorticity triplets (n,256,256,3); channel 1 is the middle frame; x SIG for physical units.
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

CONFIG = dict(
    fdir='base_results/fields/re1000', SIG=4.7988,
    rows=['recon', 'base0__K3__none', 'base0__K3__rewardv2', 'r1k-449__K3__none', 'r1k-449__K3__rewardv2'],
    bands=[(16, 32), (32, 64), (64, 96)], smooth_sigma=6.0, top_q=5.0,
    panel_w=4.6, panel_h=4.2, fontsize=10, dpi=140, out='monitoring/figs/fields_placement.pdf',
)
N = 256
fy = np.fft.fftfreq(N) * N
KM = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)


def band_energy_map(w, lo, hi, sigma):
    """smoothed local energy of the [lo,hi) band, per frame; w (n,256,256) physical vorticity"""
    F = np.fft.fft2(w)
    bp = np.real(np.fft.ifft2(F * ((KM >= lo) & (KM < hi))))
    g = np.exp(-2.0 * (np.pi * sigma) ** 2 * ((fy[:, None] / N) ** 2 + (fy[None, :] / N) ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(bp ** 2) * g))


def load(fdir, name, SIG):
    return np.load(f'{fdir}/{name}.npz')['x'].astype(np.float32)[..., 1] * SIG


def analyse(c):
    gt = load(c['fdir'], 'GT', c['SIG'])
    res = {}
    for r in c['rows']:
        y = load(c['fdir'], r, c['SIG'])
        # ALIGNMENT GUARD: see deficit_panel.py
        assert len(y) == len(gt), f'{r}: {len(y)} triplets vs GT {len(gt)} - mask the GT first'
        per_band = {}
        for lo, hi in c['bands']:
            mg, my = band_energy_map(gt, lo, hi, c['smooth_sigma']), band_energy_map(y, lo, hi, c['smooth_sigma'])
            corr = np.array([np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(my, mg)])
            q = 100 - c['top_q']
            jac = []
            for a, b in zip(my, mg):
                ha, hb = a > np.percentile(a, q), b > np.percentile(b, q)
                jac.append((ha & hb).sum() / max((ha | hb).sum(), 1))
            # cross-correlation peak offset of the band-energy maps (pixels)
            off = []
            for a, b in zip(my, mg):
                xc = np.real(np.fft.ifft2(np.fft.fft2(a - a.mean()) * np.conj(np.fft.fft2(b - b.mean()))))
                i, j = np.unravel_index(np.argmax(xc), xc.shape)
                off.append(np.hypot((i + N // 2) % N - N // 2, (j + N // 2) % N - N // 2))
            per_band[(lo, hi)] = dict(corr=corr, jaccard=np.array(jac), offset=np.array(off))
        res[r] = per_band
    return res


def make_figure(c, res):
    style.apply(c['fontsize'], c['dpi'])
    nb = len(c['bands'])
    fig, axes = plt.subplots(2, nb, figsize=(c['panel_w'] * nb, c['panel_h'] * 2), constrained_layout=True, squeeze=False)
    rng = np.random.default_rng(0)
    for bi, (lo, hi) in enumerate(c['bands']):
        for stat, ax in (('corr', axes[0][bi]), ('jaccard', axes[1][bi])):
            for i, r in enumerate(c['rows']):
                v = res[r][(lo, hi)][stat]; col = style.PALETTE[i % len(style.PALETTE)]
                ax.scatter(np.full(len(v), i) + rng.uniform(-0.2, 0.2, len(v)), v, s=7, color=col, alpha=0.45)
                ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=style.INK, lw=2)
            ax.set_xticks(range(len(c['rows']))); ax.set_xticklabels([r.replace('__', ' ') for r in c['rows']], rotation=50, ha='right', fontsize=c['fontsize'] - 2.5)
            what = 'map correlation' if stat == 'corr' else f"top-{c['top_q']:g}% hot-spot overlap (Jaccard)"
            ax.set_title(f"{what} [{lo},{hi})")
    fig.suptitle('Where the energy goes vs ground truth — per held-out sample (dots), medians (bars)', fontsize=c['fontsize'] + 2)
    fig.savefig(c['out']); plt.close(fig); print('written', c['out'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rows'); ap.add_argument('--out'); ap.add_argument('--top-q', type=float)
    a = ap.parse_args(); c = dict(CONFIG)
    if a.rows: c['rows'] = a.rows.split(',')
    if a.out: c['out'] = a.out
    if a.top_q: c['top_q'] = a.top_q
    have = {os.path.basename(f)[:-4] for f in glob.glob(f"{c['fdir']}/*.npz")}
    c['rows'] = [r for r in c['rows'] if r in have]
    if not c['rows']: sys.exit(f"no stored fields yet in {c['fdir']} (the audit's backfill pass writes them)")
    res = analyse(c)
    for r in c['rows']:
        print(r, ' | '.join(f"[{lo},{hi}) corr {np.median(v['corr']):.2f} jac {np.median(v['jaccard']):.2f} off {np.median(v['offset']):.1f}px" for (lo, hi), v in res[r].items()))
    make_figure(c, res)
