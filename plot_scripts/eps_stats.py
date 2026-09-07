"""Quantify the per-step corrections of the finetunes: magnitude, alignment, spectral split, and
how much of the strongest model's correction is explained by the weaker ones.

  JAX_PLATFORMS=cpu python plot_scripts/eps_stats.py
"""
import os, sys, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'src/ddpo_ft')
os.environ.setdefault('BASE_CKPT', '/tmp/ema_ckpts/ema_base_0299.pkl')
import numpy as np

TS, DRAWS, NSAMP = [150, 100, 50], 2, 6
FT = [('mt1k', 'monitoring/ddpo_re1000_match_ckpts/ddpo_re1000_iter0499.pkl'),
      ('mt2k', 'monitoring/ddpo_re2000_match_ckpts/ddpo_re1000_iter0599.pkl'),
      ('r8k', 'monitoring/ddpo_re8000_pdew02_ckpts/ddpo_re1000_iter0599.pkl')]


def params_of(d): return d.get('ema_params') if d.get('ema_params') is not None else d['params']


def main():
    import jax, jax.numpy as jnp
    from train_claude import build_base_ddpm
    ddpm, bp, _ = build_base_ddpm(); ab = ddpm.alpha_bar
    fts = {n: params_of(pickle.load(open(p, 'rb'))) for n, p in FT}
    x0 = jnp.asarray(np.load('base_results/fields/re1000/recon.npz')['x'].astype(np.float32)[:NSAMP])
    rng = np.random.default_rng(0)
    # spectral split masks (middle frame FFT shells)
    fy = np.fft.fftfreq(256) * 256
    km = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
    lo_m, hi_m = (km < 32), (km >= 32)

    acc = {k: [] for k in ('n_mt2k', 'n_r8k', 'c12', 'c18', 'c28', 'expl', 'hi_mt1k', 'hi_mt2k',
                           'hi_r8k', 'supp12', 'supp18')}
    for t in TS:
        for dr in range(DRAWS):
            eps = jnp.asarray(rng.normal(size=x0.shape).astype(np.float32))
            xt = float(np.sqrt(ab[t])) * x0 + float(np.sqrt(1 - ab[t])) * eps
            ta = jnp.full((NSAMP,), t, jnp.int32)
            eb = np.asarray(ddpm.unet.apply({'params': bp}, xt, ta, train=False))
            d = {n: np.asarray(ddpm.unet.apply({'params': p}, xt, ta, train=False)) - eb
                 for n, p in fts.items()}
            for j in range(NSAMP):                     # PER-SAMPLE statistics
                v = {n: d[n][j].ravel() for n in d}
                nrm = {n: np.linalg.norm(v[n]) for n in v}
                acc['n_mt2k'].append(nrm['mt2k'] / nrm['mt1k'])
                acc['n_r8k'].append(nrm['r8k'] / nrm['mt1k'])
                cs = lambda a, b: float(v[a] @ v[b] / (nrm[a] * nrm[b]))
                acc['c12'].append(cs('mt2k', 'mt1k')); acc['c18'].append(cs('r8k', 'mt1k'))
                acc['c28'].append(cs('r8k', 'mt2k'))
                # least squares: how much of r8k's correction lies in span(mt1k, mt2k)
                A = np.stack([v['mt1k'], v['mt2k']], 1)
                coef, *_ = np.linalg.lstsq(A, v['r8k'], rcond=None)
                acc['expl'].append(1 - np.linalg.norm(v['r8k'] - A @ coef) ** 2 / nrm['r8k'] ** 2)
                # spectral split of each correction (middle frame): fraction of power at k>=32
                for n in d:
                    F2 = np.abs(np.fft.fft2(d[n][j][..., 1])) ** 2
                    acc[f'hi_{n}'].append(F2[hi_m].sum() / F2.sum())
                # support overlap: correlation of |correction| magnitude maps (where, not what sign)
                m1 = np.abs(d['mt1k'][j][..., 1]).ravel()
                acc['supp12'].append(np.corrcoef(np.abs(d['mt2k'][j][..., 1]).ravel(), m1)[0, 1])
                acc['supp18'].append(np.corrcoef(np.abs(d['r8k'][j][..., 1]).ravel(), m1)[0, 1])
    f = lambda k: f'{np.mean(acc[k]):.2f} $\\pm$ {np.std(acc[k]):.2f}'
    ft = lambda k: f'{np.mean(acc[k]):.2f} ± {np.std(acc[k]):.2f}'
    print(f'per-sample statistics over {len(acc["c12"])} (sample, t, draw) cells, t in {TS}:')
    print(f'  |d_mt2k|/|d_mt1k|          {ft("n_mt2k")}')
    print(f'  |d_r8k|/|d_mt1k|           {ft("n_r8k")}')
    print(f'  cos(mt2k, mt1k)            {ft("c12")}')
    print(f'  cos(r8k, mt1k)             {ft("c18")}')
    print(f'  cos(r8k, mt2k)             {ft("c28")}')
    print(f'  r8k explained by span(mt1k,mt2k)  {ft("expl")}   (fraction of variance)')
    print(f'  support overlap |d| maps: mt2k-mt1k {ft("supp12")}, r8k-mt1k {ft("supp18")}')
    print(f'  fraction of correction power at k>=32: mt1k {ft("hi_mt1k")}, mt2k {ft("hi_mt2k")}, r8k {ft("hi_r8k")}')
    rows = [
        ['$\\|\\Delta\\epsilon\\|$ vs $Re{=}1000$ ft', f('n_mt2k'), f('n_r8k')],
        ['cosine with $Re{=}1000$ ft', f('c12'), f('c18')],
        ['cosine $Re{=}8000$ vs $Re{=}2000$', '', f('c28')],
        ['variance explained by span of the other two', '', f('expl')],
        ['support overlap of $|\\Delta\\epsilon|$ with $Re{=}1000$ ft', f('supp12'), f('supp18')],
        ['fraction of correction power at $k\\geq32$', f('hi_mt2k'), f('hi_r8k')],
    ]
    tx = ["\\begin{table}[t]", "  \\centering",
          "  \\caption{Statistics of the single-step correction "
          "$\\Delta\\epsilon=\\epsilon_{ft}-\\epsilon_{base}$ on identical noised inputs from the "
          "$Re{=}1000$ test pool, per sample, over $t\\in\\{150,100,50\\}$ and two noise draws "
          "($" + str(len(acc['c12'])) + "$ cells). The $Re{=}1000$ finetune's correction carries a "
          "$k\\geq32$ power fraction of " + f('hi_mt1k') + ".}",
          "  \\label{tab:eps-correction-stats}",
          "  \\begin{tabular}{lrr}", "    \\toprule",
          "    & $Re{=}2000$ ft & $Re{=}8000$ ft \\\\", "    \\midrule"]
    for r in rows: tx.append("    " + " & ".join(r) + " \\\\")
    tx += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    open('docs/tables/eps_correction_stats.tex', 'w').write("\n".join(tx))
    print('\n  written docs/tables/eps_correction_stats.tex')


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args(); main()
