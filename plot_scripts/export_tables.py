"""Emit the result tables as LaTeX (booktabs) into docs/tables/.

Every number is read live from the result stores, so re-running this after new cells land
refreshes the tables. Edit the ROWS lists below to change what appears.

  JAX_PLATFORMS=cpu python plot_scripts/export_tables.py
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
OUT = 'docs/tables'; os.makedirs(OUT, exist_ok=True)
A1 = np.load('base_results/re1000_audit.npz', allow_pickle=True)
GT1 = np.asarray(A1['1000|GT||psEb'])


def cell(store, m, sg, f):
    k = f"{'1000' if store is A1 else store['re']}|{m}|K3|{sg}||{f}"
    return np.asarray(store[k]) if k in store.files else None


def tex(name, caption, label, header, rows, note=None):
    ncol = len(header)
    s = ["\\begin{table}[t]", "  \\centering", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
         "  \\begin{tabular}{l" + "r" * (ncol - 1) + "}", "    \\toprule",
         "    " + " & ".join(header) + " \\\\", "    \\midrule"]
    for r in rows:
        if r[0] == "\\midrule":
            s.append("    \\midrule"); continue
        s.append("    " + " & ".join(r) + " \\\\")
    s += ["    \\bottomrule", "  \\end{tabular}"]
    if note: s.append(f"  \\vspace{{2pt}}\\\\ \\footnotesize {note}")
    s += ["\\end{table}", ""]
    open(f'{OUT}/{name}.tex', 'w').write("\n".join(s))
    print(f"  {OUT}/{name}.tex")


# ---------------- 1. the Re=1000 headline ----------------
HEAD = [('base0', 'none', 'base model, unguided'),
        ('mt1k-0499', 'mo0', 'matched fine-tune, unguided'),
        ('base0', 'mop0.2_3', 'base + matched dial ($\\lambda=3$)'),
        ('mt1k-0499', 'mop0.2_3', 'fine-tune + matched dial'),
        ('base0', 'v6gate', 'base + target-gated dial')]
rows = []
T = GT1[:, 3] + GT1[:, 4]
for m, sg, lab in HEAD:
    if cell(A1, m, sg, 'ret') is None: continue
    pr = cell(A1, m, sg, 'ps_ret_paired')
    q = cell(A1, m, sg, 'pst_ret'); q = q if q is not None else pr
    pp = cell(A1, m, sg, 'pst_place'); pp = pp if pp is not None else cell(A1, m, sg, 'ps_place')
    rows.append([lab, f"{float(cell(A1,m,sg,'ret')):.3f}", f"{float(cell(A1,m,sg,'lowk')):.3f}",
                 f"{np.median(pp):.3f}", f"{float(cell(A1,m,sg,'resid_ratio')):.2f}",
                 f"{float(cell(A1,m,sg,'mse')):.2f}",
                 f"{np.mean(np.abs(q-1)<0.2)*100:.0f}\\%",
                 f"{np.polyfit(np.log(T), np.log(pr*T), 1)[0]:.2f}"])
tex('re1000_headline',
    'In-distribution performance at $\\mathrm{Re}=1000$ on 120 held-out triplets. Retention is '
    'band energy $E[32,96)$ relative to ground truth; in-band is the share of triplets within '
    '$\\pm20\\%$ of their own truth; slope is $\\mathrm{d}\\log E/\\mathrm{d}\\log E_{\\mathrm{GT}}$ '
    'per triplet ($1$ = tracks each triplet, $0$ = identical dose for all).',
    'tab:re1000-headline',
    ['configuration', 'ret.', 'low-$k$', 'place.', 'resid./GT', 'MSE', 'in-band', 'slope'], rows)

# ---------------- 2. per band ----------------
BL = ['$[1,5)$', '$[5,16)$', '$[16,32)$', '$[32,64)$', '$[64,96)$']
BR = [('base0', 'none', 'base model, unguided'), ('mt1k-0499', 'mo0', 'matched fine-tune, unguided')]
rows = [[lab] + [f"{x:.2f}" for x in cell(A1, m, sg, 'Eb')] for m, sg, lab in BR]
tex('re1000_perband_retention',
    'Per-band energy retention at $\\mathrm{Re}=1000$: band energy divided by the ground truth\'s.',
    'tab:re1000-perband-ret', ['configuration'] + BL, rows)
Z = np.load('base_results/placement_vs_k.npz', allow_pickle=True) if \
    os.path.exists('base_results/placement_vs_k.npz') else None
rows = []
for m, sg, lab in BR:
    v = cell(A1, m, sg, 'ps_bp')
    if v is None: continue
    rows.append([lab] + [f"{np.median(v[:, i]):.3f}" for i in range(5)])
if rows:
    tex('re1000_perband_placement',
        'Per-band placement at $\\mathrm{Re}=1000$: correlation of the local band-energy map with '
        'the ground truth\'s, computed per triplet and aggregated (median).',
        'tab:re1000-perband-place', ['configuration'] + BL, rows)

# ---------------- 3. total energy ----------------
Eg = np.asarray(A1['1000|GT||E']); tot = Eg[1:].sum()
TR = [('base0', 'none', 'base model, unguided'),
      ('mt1k-0499', 'mo0', 'matched fine-tune, unguided'),
      ('base0', 'v6gate', 'base + target-gated dial')]
rows = []
for m, sg, lab in TR:
    E = cell(A1, m, sg, 'E'); pE = cell(A1, m, sg, 'psEb')
    if E is None: continue
    r = pE.sum(1) / GT1.sum(1)
    rows.append([lab, f"{E[1:].sum()/tot:.4f}", f"{np.abs(E[1:]-Eg[1:]).sum()/tot*100:.2f}\\%",
                 f"{np.mean(np.abs(r-1)<0.05)*100:.0f}\\%", f"{np.mean(np.abs(r-1)<0.10)*100:.0f}\\%",
                 f"{np.abs(r-1).max()*100:.1f}\\%", f"{E[32:96].sum()/Eg[32:96].sum():.3f}"])
tex('re1000_total_energy',
    'Total energy at $\\mathrm{Re}=1000$. The fine band $[32,96)$ carries only $1.07\\%$ of the '
    'ground truth\'s energy, so large changes there are small changes in the total.',
    'tab:re1000-total',
    ['configuration', 'total/GT', 'misalloc.', 'within $5\\%$', 'within $10\\%$', 'worst', 'fine band'], rows)

# ---------------- 4. the transfer ----------------
REGS = [1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
rows = []
for R in REGS:
    p = 'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz'
    if not os.path.exists(p): continue
    S = np.load(p, allow_pickle=True); S = {'files': S.files, 'd': S, 're': R}
    def c(m, sg, f):
        k = f'{R}|{m}|K3|{sg}||{f}'
        return np.asarray(S['d'][k]) if k in S['files'] else None
    gt = c('GT', '', 'psEb')
    gt = np.asarray(S['d'][f'{R}|GT||psEb']) if f'{R}|GT||psEb' in S['files'] else None
    row = [str(R)]
    for m, sg in (('base0', 'mop0.2_0'), ('base0', 'mop0.2_3'),
                  ('mt1k-0499', 'mop0.2_0'), ('mt1k-0499', 'mop0.2_3')):
        v = c(m, sg, 'ret')
        pr = c(m, sg, 'pst_ret'); pr = pr if pr is not None else c(m, sg, 'ps_ret_paired')
        row += ([f"{float(v):.2f}", f"{np.mean(np.abs(pr-1)<0.2)*100:.0f}\\%"]
                if v is not None and pr is not None else ['--', '--'])
    rows.append(row)
if rows:
    tex('transfer_bdial',
        'Transport of the matched dial ($\\lambda=3$, chosen on validation from anchor-relative '
        'statistics alone) across regimes. Each pair is retention and the share of triplets within '
        '$\\pm20\\%$ of their own truth.',
        'tab:transfer',
        ['$\\mathrm{Re}$', 'base ret.', 'in-band', 'base+dial', 'in-band',
         'ft ret.', 'in-band', 'ft+dial', 'in-band'], rows)
print(f"\n-> {OUT}/  (\\input{{...}} these, or paste directly)")

# ---------------- 5. the full transfer (placement, residual, MSE) ----------------
CFG = [('base0', 'mop0.2_0', 'base'), ('mt1k-0499', 'mop0.2_0', 'ft'),
       ('base0', 'mop0.2_3', 'base+dial'), ('mt1k-0499', 'mop0.2_3', 'ft+dial')]
for metric, fld, fmt, cap in [
    ('placement', 'pst_place', '{:.3f}',
     'Per-triplet placement (median) transported across regimes.'),
    ('residual', 'resid_ratio', '{:.2f}',
     'PDE residual relative to the ground truth\'s, transported across regimes. '
     'Values below $1$ mean the sample is too smooth.'),
    ('mse', 'mse', '{:.2f}', 'Reconstruction MSE transported across regimes.')]:
    rows = []
    for R in REGS:
        p = 'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz'
        if not os.path.exists(p): continue
        S = np.load(p, allow_pickle=True); F = set(S.files)
        row = [str(R)]
        for m, sg, _ in CFG:
            k = f'{R}|{m}|K3|{sg}||{fld}'
            k2 = f'{R}|{m}|K3|{sg}||' + ('ps_place' if fld == 'pst_place' else fld)
            v = S[k] if k in F else (S[k2] if k2 in F else None)
            row.append('--' if v is None else
                       fmt.format(float(np.median(np.asarray(v))) if np.asarray(v).ndim else float(v)))
        rows.append(row)
    if rows:
        tex(f'transfer_{metric}', cap + ' Dial at $\\lambda=3$.', f'tab:transfer-{metric}',
            ['$\\mathrm{Re}$'] + [c[2].replace('+', '$+$') for c in CFG], rows)

# ---------------- 6. transfer_full: every config x every regime, all metrics ----------------
rows = []
for R in REGS:
    p = 'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz'
    if not os.path.exists(p): continue
    S = np.load(p, allow_pickle=True); F = set(S.files)
    for j, (m, sg, lab) in enumerate(CFG):
        g = lambda f: (np.asarray(S[f'{R}|{m}|K3|{sg}||{f}'])
                       if f'{R}|{m}|K3|{sg}||{f}' in F else None)
        if g('ret') is None: continue
        pr = g('pst_ret'); pr = pr if pr is not None else g('ps_ret_paired')
        pp = g('pst_place'); pp = pp if pp is not None else g('ps_place')
        rows.append([('\\multirow{4}{*}{%d}' % R) if j == 0 else '',
                     lab.replace('+', '$+$'), f"{float(g('ret')):.3f}",
                     f"{np.mean(np.abs(pr-1)<0.2)*100:.0f}\\%" if pr is not None else '--',
                     f"{float(g('lowk')):.3f}",
                     f"{np.median(pp):.3f}" if pp is not None else '--',
                     f"{float(g('resid_ratio')):.2f}", f"{float(g('mse')):.2f}"])
    if rows: rows.append(['\\midrule'] + [''] * 7)
if rows and rows[-1][0] == '\\midrule': rows.pop()
tex('transfer_full',
    'Full transfer grid: the base model and the matched $\\mathrm{Re}=1000$ fine-tune, unguided '
    'and with the matched dial at $\\lambda=3$, evaluated at all nine Reynolds numbers on 120 '
    'held-out triplets each. In-band is the share of triplets within $\\pm20\\%$ of their own '
    'ground truth; residual is relative to the ground truth\'s.',
    'tab:transfer-full',
    ['$\\mathrm{Re}$', 'configuration', 'ret.', 'in-band', 'low-$k$', 'place.', 'resid.', 'MSE'],
    rows, note='Requires \\texttt{multirow}.')

# ---------------- 7. per-band retention, four configs, every regime ----------------
BL5 = ['$[1,5)$', '$[5,16)$', '$[16,32)$', '$[32,64)$', '$[64,96)$']
rows = []
for R in REGS:
    p = 'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz'
    if not os.path.exists(p): continue
    S = np.load(p, allow_pickle=True); F = set(S.files)
    for j, (m, sg, lab) in enumerate(CFG):
        k = f'{R}|{m}|K3|{sg}||Eb'
        if k not in F: continue
        Eb = np.asarray(S[k])[:5]
        rk, mk = f'{R}|{m}|K3|{sg}||resid_ratio', f'{R}|{m}|K3|{sg}||mse'
        rr = f'{float(S[rk]):.2f}' if rk in F else '--'
        ms = f'{float(S[mk]):.2f}' if mk in F else '--'
        rows.append([('\\multirow{4}{*}{%d}' % R) if j == 0 else '',
                     lab.replace('+', '$+$')] + [f'{v:.3f}' for v in Eb] + [rr, ms])
    if rows: rows.append(['\\midrule'] + [''] * 8)
if rows and rows[-1][0] == '\\midrule': rows.pop()
tex('perband_all_regimes',
    'Per-band energy retention $E_{\\mathrm{band}}/E_{\\mathrm{GT,band}}$ for the base model and '
    'the matched $\\mathrm{Re}=1000$ fine-tune, each with and without the matched dial '
    '($\\lambda=3$), at every regime. The two large-scale bands are untouched by either mechanism; '
    '$[16,32)$ straddles the coarse Nyquist $k=32$ and is never fully recovered. The final two '
    'columns are whole-field quantities, not per-band: the PDE residual relative to the ground '
    "truth (values below $1$ mean the sample is too smooth) and the reconstruction MSE.",
    'tab:perband-all', ['$\\mathrm{Re}$', 'configuration'] + BL5
    + ['resid./GT', 'MSE'], rows,
    note='Requires \\texttt{multirow}.')


# ---------------- 8. transfer with the CURRENT dial (tapered) + the gated dial ----------------
CFG2 = [('base0', 'mop0.2_0', 'base'), ('base0', 'tapp0.2_3', 'base$+$dial'),
        ('mt1k-0499', 'mop0.2_0', 'ft'), ('mt1k-0499', 'tapp0.2_3', 'ft$+$dial'),
        ('base0', 'v7bandgate', 'base$+$gate'), ('mt1k-0499', 'v7bandgate', 'ft$+$gate')]
rows = []
for R in REGS:
    p_ = 'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz'
    if not os.path.exists(p_): continue
    S = np.load(p_, allow_pickle=True); F = set(S.files)
    for j, (m, sg, lab) in enumerate(CFG2):
        if f'{R}|{m}|K3|{sg}||ret' not in F: continue
        g = lambda f: np.asarray(S[f'{R}|{m}|K3|{sg}||{f}'])
        pr = None
        for kk in ('pst_ret', 'ps_ret_paired'):
            if f'{R}|{m}|K3|{sg}||{kk}' in F: pr = g(kk); break
        pp = None
        for kk in ('pst_place', 'ps_place'):
            if f'{R}|{m}|K3|{sg}||{kk}' in F: pp = g(kk); break
        rows.append([('\\multirow{6}{*}{%d}' % R) if j == 0 else '', lab,
                     f'{float(g("ret")):.3f}',
                     f'{np.mean(np.abs(pr-1)<0.2)*100:.0f}\\%' if pr is not None else '--',
                     f'{float(g("lowk")):.3f}',
                     f'{np.median(pp):.3f}' if pp is not None else '--',
                     f'{float(g("resid_ratio")):.2f}', f'{float(g("mse")):.2f}'])
    if rows: rows.append(['\\midrule'] + [''] * 7)
if rows and rows[-1][0] == '\\midrule': rows.pop()
tex('transfer_all_dials',
    'Out-of-distribution transfer of the $\\mathrm{Re}=1000$ pair. The matched dial '
    '($\\lambda=3$, tapered band edge) is the fine-tune\'s own reward one scalar apart, so those '
    'rows carry no objective confound; the target-gated dial optimises a different objective and is '
    'shown as the best steering available rather than as a matched comparison.',
    'tab:transfer-all-dials',
    ['$\\mathrm{Re}$', 'configuration', 'ret.', 'in-band', 'low-$k$', 'place.', 'resid.', 'MSE'],
    rows, note='Requires \\texttt{multirow}.')

# ---------------- 9. MSE and PDE residual under upward generalisation ----------------
def _g(R, m, sg, f):
    p = 'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz'
    if not os.path.exists(p): return None
    S = np.load(p, allow_pickle=True); k = f'{R}|{m}|K3|{sg}||{f}'
    return float(S[k]) if k in S.files else None


UP = [('base0', 'mop0.2_0', 'base'), ('base0', 'tapp0.2_3', 'base $+$ dial'),
      ('mt1k-0499', 'mop0.2_0', 'ft'), ('mt1k-0499', 'tapp0.2_3', 'ft $+$ dial')]
rows = []
for R in REGS:
    if _g(R, 'base0', 'mop0.2_0', 'mse') is None: continue
    r = [str(R)]
    for m, sg, _ in UP:
        mse, rr = _g(R, m, sg, 'mse'), _g(R, m, sg, 'resid_ratio')
        r += ([f'{mse:.2f}', f'{rr:.2f}'] if mse is not None else ['--', '--'])
    rows.append(r)
tex('mse_resid_upward',
    'Reconstruction MSE and PDE residual (relative to that regime\'s ground truth) for the '
    '$\\mathrm{Re}=1000$ models carried upward. A residual below $1$ means the sample is smoother '
    'than the true flow; it falls monotonically with $\\mathrm{Re}$ for every configuration, so the '
    'reconstructions become progressively over-smoothed. MSE rises monotonically, and every '
    'configuration that adds energy pays for it.',
    'tab:mse-resid-upward',
    ['$\\mathrm{Re}$'] + [c for _, _, l in UP for c in (f'{l} MSE', 'resid.')], rows)

# in-regime rows for the models trained at 2000 / 8000
rows = []
for m, R, lab in (('mt2k-0599', 2000, 'Re=2000 fine-tune'),
                  ('r8kp02-0599', 8000, 'Re=8000 fine-tune')):
    d = next((t for t in ('tapp0.2_3', 'mop0.2_3') if _g(R, m, t, 'mse') is not None), None)
    rows.append([lab, str(R), f"{_g(R, m, 'mop0.2_0', 'mse'):.2f}",
                 f"{_g(R, m, 'mop0.2_0', 'resid_ratio'):.2f}",
                 f'{_g(R, m, d, "mse"):.2f}' if d else '--',
                 f'{_g(R, m, d, "resid_ratio"):.2f}' if d else '--'])
    rows.append([f'\\quad base at that regime', str(R), f"{_g(R, 'base0', 'mop0.2_0', 'mse'):.2f}",
                 f"{_g(R, 'base0', 'mop0.2_0', 'resid_ratio'):.2f}",
                 f"{_g(R, 'base0', 'tapp0.2_3', 'mse'):.2f}",
                 f"{_g(R, 'base0', 'tapp0.2_3', 'resid_ratio'):.2f}"])
tex('mse_resid_inregime',
    'The same two quantities for the models trained at $\\mathrm{Re}=2000$ and $8000$, evaluated '
    'in their own regime only, each against the base model there.',
    'tab:mse-resid-inregime',
    ['model', '$\\mathrm{Re}$', 'MSE', 'resid.', '$+$dial MSE', '$+$dial resid.'], rows)

# ---------------- 10. the specialists across regimes, unguided + tapered dial ----------------
SPEC = [('mt1k-0499', 'Re=1000 ft'), ('mt2k-0599', 'Re=2000 ft'), ('r8kp02-0599', 'Re=8000 ft')]
rows = []
for R in REGS:
    p_ = 'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz'
    if not os.path.exists(p_): continue
    S = np.load(p_, allow_pickle=True); F = set(S.files)
    r = [str(R)]
    for m, _ in SPEC:
        for sg in ('mop0.2_0', 'tapp0.2_3'):
            k = f'{R}|{m}|K3|{sg}||ret'
            if k not in F: r += ['--', '--']; continue
            pr = None
            for ff in ('pst_ret', 'ps_ret_paired'):
                if f'{R}|{m}|K3|{sg}||{ff}' in F: pr = np.asarray(S[f'{R}|{m}|K3|{sg}||{ff}']); break
            r += [f'{float(S[k]):.3f}',
                  f'{np.mean(np.abs(pr-1)<0.2)*100:.0f}\\%' if pr is not None else '--']
    rows.append(r)
tex('specialist_transfer',
    'Every fine-tune carried across every regime, unguided and with the tapered dial at '
    '$\\lambda=3$. Each pair of columns is retention over $[32,96)$ and the share of triplets '
    'within $\\pm20\\%$ of their own truth. Each model peaks at its training regime; upward of it '
    'the model undershoots by a near-constant factor, downward it overshoots.',
    'tab:specialist-transfer',
    ['$\\mathrm{Re}$'] + [c for _, l in SPEC for c in (f'{l}', 'in-b.', '$+$dial', 'in-b.')], rows)
