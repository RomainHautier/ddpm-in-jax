"""CAN THE REGIMES BE TOLD APART FROM THE LOW-RESOLUTION DATA ALONE?

The question behind it: is there enough regime information in a coarse observation to CONDITION a
reconstruction model on, so one network could serve all regimes? Uses ONLY the observed 64x64
files (flow-data/observed/) - the exact information a deployed model would see.

Per sequence, three cheap observables:
  1. spectral tail ratio  E_lr[18,32) / E_lr[6,12)   - how much of the observable band's energy
     sits near the observation Nyquist (dissipation range encroaching = higher Re)
  2. temporal decorrelation  lag-8 autocorrelation of the coarse field (faster eddies = lower)
  3. coarse enstrophy  mean square vorticity at 64^2 (weakly Re-dependent)
Separability is scored by leave-one-sequence-out nearest-centroid classification over the 9
regimes, features standardised; plus the confusion structure (adjacent-regime mixups are expected
and acceptable for conditioning - a smooth conditioning variable does not need exact class labels).
Output: base_results/lr_separability.npz + figures monitoring/figs/fig_lrsep_*.png
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'savefig.dpi': 118, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#d8d6cd', 'grid.linewidth': 0.5,
                     'legend.frameon': False})
FIG = 'monitoring/figs'
REGS = [1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
NC = 64
kc = np.fft.fftfreq(NC, 1.0 / NC)
KRC = np.round(np.sqrt(kc[:, None] ** 2 + kc[None, :] ** 2)).astype(int).ravel()

def lr_spec(frames):
    S = [np.bincount(KRC, (np.abs(np.fft.fft2(f)) ** 2).ravel(), minlength=NC)[:33]
         for f in frames]
    return np.asarray(S).mean(0)

feats, labels, spectra = [], [], {}
snap = {}
for R in REGS:
    a = np.load(f'flow-data/observed/re{R}_obs.npy', mmap_mode='r')
    n_seq = a.shape[0]
    per = []
    for s in range(n_seq):
        fr = np.asarray(a[s, ::20], np.float32)          # 16 frames spread over the sequence
        E = lr_spec(fr)
        tail = E[18:32].sum() / E[6:12].sum()
        x, y = a[s, 100].astype(np.float32), a[s, 108].astype(np.float32)   # lag-8 frames
        x = x - x.mean(); y = y - y.mean()
        lag8 = float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
        ens = float((fr ** 2).mean())
        feats.append([np.log(tail), lag8, np.log(ens)]); labels.append(R); per.append(E)
    spectra[R] = np.mean(per, 0)
    snap[R] = np.asarray(a[0, 100], np.float32)
    print(f'Re={R}: {n_seq} sequences', flush=True)

X = np.array(feats); y = np.array(labels)
mu, sd = X.mean(0), X.std(0)
Xs = (X - mu) / sd
# leave-one-out nearest-centroid
pred = []
for i in range(len(Xs)):
    keep = np.arange(len(Xs)) != i
    cents = {R: Xs[keep][y[keep] == R].mean(0) for R in REGS}
    pred.append(min(cents, key=lambda R: np.linalg.norm(Xs[i] - cents[R])))
pred = np.array(pred)
acc = float((pred == y).mean())
adj = float(np.mean([abs(REGS.index(p) - REGS.index(t)) <= 1 for p, t in zip(pred, y)]))
rank = np.array([REGS.index(v) for v in y])
rankp = np.array([REGS.index(v) for v in pred])
rho = np.corrcoef(rank, rankp)[0, 1]
print(f'\nLeave-one-sequence-out nearest-centroid over 9 regimes:')
print(f'  exact accuracy      : {acc:.1%}   (chance 11.1%)')
print(f'  within one regime   : {adj:.1%}')
print(f'  rank correlation    : {rho:+.3f}')
conf = np.zeros((9, 9), int)
for t, p in zip(y, pred):
    conf[REGS.index(t), REGS.index(p)] += 1
np.savez('base_results/lr_separability.npz', X=X, y=y, pred=pred, conf=conf,
         regimes=np.array(REGS), acc=acc, adj=adj)

INK = '#232823'
fig, axes = plt.subplots(1, 9, figsize=(17.2, 2.3), constrained_layout=True)
vm = max(np.abs(snap[R]).max() for R in REGS)
for ax, R in zip(axes, REGS):
    ax.imshow(snap[R], cmap='RdBu_r', vmin=-vm, vmax=vm)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(f'Re={R}', fontsize=9)
fig.suptitle('The observed 64x64 fields, one snapshot per regime - the eye can barely tell them apart',
             fontsize=11)
fig.savefig(f'{FIG}/fig_lrsep_fields.png'); plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.3), constrained_layout=True)
kk = np.arange(1, 33)
cm = plt.cm.viridis(np.linspace(0.05, 0.95, 9))
for c, R in zip(cm, REGS):
    a1.loglog(kk, spectra[R][1:33], '-', color=c, lw=1.8, label=f'{R}')
    a2.semilogx(kk, spectra[R][1:33] / spectra[1000][1:33], '-', color=c, lw=1.8)
a1.set(xlabel='wavenumber k', ylabel='observed spectrum E(k)',
       title='Low-resolution spectra, all 9 regimes')
a1.legend(fontsize=7, ncol=2, title='Re')
a2.axhline(1, color=INK, lw=1.2, ls='--')
a2.set(xlabel='wavenumber k', ylabel='ratio to Re=1000',
       title='Ratio to Re=1000: the tail separates the regimes')
fig.savefig(f'{FIG}/fig_lrsep_spectra.png'); plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.3), constrained_layout=True)
for c, R in zip(cm, REGS):
    m = y == R
    a1.scatter(X[m, 0], X[m, 1], s=22, color=c, label=f'{R}', edgecolors='white', linewidths=0.5)
a1.set(xlabel='log spectral tail ratio  E[18,32)/E[6,12)', ylabel='lag-8 temporal correlation',
       title='Each point is one SEQUENCE, coloured by regime')
a1.legend(fontsize=7, ncol=2, title='Re')
im = a2.imshow(conf, cmap='Blues')
a2.set_xticks(range(9)); a2.set_xticklabels(REGS, rotation=45, fontsize=7)
a2.set_yticks(range(9)); a2.set_yticklabels(REGS, fontsize=7)
a2.set(xlabel='predicted regime', ylabel='true regime',
       title=f'Confusion (leave-one-out): exact {acc:.0%}, within-one {adj:.0%}')
a2.grid(False)
for i in range(9):
    for j in range(9):
        if conf[i, j]:
            a2.text(j, i, conf[i, j], ha='center', va='center', fontsize=7,
                    color='white' if conf[i, j] > conf.max() / 2 else INK)
fig.savefig(f'{FIG}/fig_lrsep_features.png'); plt.close(fig)
print('figures written', flush=True)
print('LR SEPARABILITY COMPLETE', flush=True)
