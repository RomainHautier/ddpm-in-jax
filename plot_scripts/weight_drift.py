"""How far did each fine-tune move from the base model?

Relative L2 displacement ||theta - theta_base|| / ||theta_base||, globally and per parameter
block, for every fine-tune. The claim under test is that a regime further off the base model's
manifold demands a larger weight change.

  JAX_PLATFORMS=cpu python plot_scripts/weight_drift.py
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, jax
import matplotlib.pyplot as plt
import style

BASE = '/tmp/ema_ckpts/ema_base_0299.pkl'
# EMA weights are what every evaluation samples from, so the drift must be measured on those
CKPTS = [('mt1k-0499', 1000, 'monitoring/ddpo_re1000_match_ckpts/ddpo_re1000_iter0499.pkl'),
         ('mt2k-0599', 2000, 'monitoring/ddpo_re2000_match_ckpts/ddpo_re1000_iter0599.pkl'),
         ('mt8k-0549', 8000, 'monitoring/ddpo_re8000_match_ckpts/ddpo_re1000_iter0549.pkl'),
         ('r8kp02-0599', 8000, 'monitoring/ddpo_re8000_pdew02_ckpts/ddpo_re1000_iter0599.pkl')]


def flat(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return np.concatenate([np.asarray(l, np.float64).ravel() for l in leaves])


def params_of(d):
    return d.get('ema_params') if d.get('ema_params') is not None else d['params']


def blocks(tree, prefix=''):
    """flatten to {path: array} so drift can be attributed to parts of the network"""
    out = {}
    fd = jax.tree_util.tree_flatten_with_path(tree)[0]
    for path, leaf in fd:
        key = '/'.join(str(getattr(p, 'key', getattr(p, 'idx', p))) for p in path)
        out[key] = np.asarray(leaf, np.float64).ravel()
    return out


DIRS = [('mt1k-0499', 1000, 'monitoring/ddpo_re1000_match_ckpts'),
        ('mt2k-0599', 2000, 'monitoring/ddpo_re2000_match_ckpts'),
        ('mt8k-0549', 8000, 'monitoring/ddpo_re8000_match_ckpts'),
        ('r8kp02-0599', 8000, 'monitoring/ddpo_re8000_pdew02_ckpts')]

b = params_of(pickle.load(open(BASE, 'rb')))
fb, bb = flat(b), blocks(b)
nb = np.linalg.norm(fb)
print(f'base: {fb.size:,} parameters, ||theta_base|| = {nb:.2f}\n')
print(f'{"model":14s}{"trained at":>11s}{"rel L2":>10s}{"abs L2":>11s}{"cos angle":>11s}'
      f'{"max |dw|":>10s}')
rows = []
for name, R, p in CKPTS:
    if not os.path.exists(p): continue
    t = params_of(pickle.load(open(p, 'rb')))
    ft = flat(t); d = ft - fb
    rel = np.linalg.norm(d) / nb
    cos = float(ft @ fb / (np.linalg.norm(ft) * nb))
    rows.append((name, R, rel, bb, blocks(t)))
    print(f'{name:14s}{R:11d}{rel:10.5f}{np.linalg.norm(d):11.3f}{cos:11.6f}'
          f'{np.abs(d).max():10.4f}')

# ---- per-block drift, so the change can be located in the network ----
print('\nper-block relative drift (top 12 by magnitude, first model):')
name, R, rel, bb0, tb0 = rows[0]
per = {k: np.linalg.norm(tb0[k] - bb0[k]) / (np.linalg.norm(bb0[k]) + 1e-12) for k in bb0}
for k, v in sorted(per.items(), key=lambda kv: -kv[1])[:12]:
    print(f'   {v:8.5f}  {k}')

# ---- drift against TRAINING ITERATION: the endpoint comparison above is confounded, because
# the runs stopped at 499/599/549/599 iterations and displacement accumulates with training.
import glob, re as _re
traj = {}
for name, R, d in DIRS:
    pts = []
    for f in sorted(glob.glob(f'{d}/ddpo_re1000_iter*.pkl')):
        it = int(_re.search(r'iter(\d+)', f).group(1))
        if it % 100 != 99 and it != 49: continue          # every ~100 iters, keep it cheap
        try: t = params_of(pickle.load(open(f, 'rb')))
        except Exception: continue
        pts.append((it + 1, np.linalg.norm(flat(t) - fb) / nb))
    traj[name] = (R, np.array(pts))
    print(f'  {name:14s} ' + '  '.join(f'{i}:{v:.4f}' for i, v in pts))

style.apply()
fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
a = ax[0]
names = [r[0] for r in rows]; rels = [r[2] for r in rows]
cols = [style.MODEL_COLOR.get(n, '#5d645d') for n in names]
a.bar(range(len(rows)), rels, color=cols, alpha=.9)
a.set_xticks(range(len(rows)))
a.set_xticklabels([f'{n}\n(Re={r[1]})' for n, r in zip(names, rows)], fontsize=style.LEG_FS - 1)
a.set_ylabel('$\\|\\theta-\\theta_{\\mathrm{base}}\\|_2\\,/\\,\\|\\theta_{\\mathrm{base}}\\|_2$')
a.set_title('total weight displacement from the base model', fontsize=style.TITLE_FS)

b_ = ax[1]
for name, (R, pts) in traj.items():
    if not len(pts): continue
    b_.plot(pts[:, 0], pts[:, 1], '-', color=style.MODEL_COLOR.get(name, '#5d645d'),
            lw=1.9, marker='o', ms=3.5, label=f'{name} (Re={R})')
b_.axvline(500, color='#9aa198', lw=1, ls='--')
b_.text(505, b_.get_ylim()[0], ' compare here', fontsize=style.LEG_FS - 1.5, color='#5d645d',
        va='bottom')
b_.set_xlabel('training iteration'); b_.set_ylabel('relative displacement')
b_.set_title('displacement accumulates with training, so compare at\nequal iteration count',
             fontsize=style.TITLE_FS)
b_.legend(fontsize=style.LEG_FS - 1)
fig.suptitle('How far each fine-tune moved from the base model', fontsize=style.SUP_FS)
for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
    os.makedirs(d_, exist_ok=True)
    q = f'{d_}/weight_drift.{ext}'; fig.savefig(q); print('\n  written', q)
