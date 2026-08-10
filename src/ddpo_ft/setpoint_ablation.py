"""WITH / WITHOUT-Re=500 ABLATION of the engraved selection rule.

Two arms, identical except for what the calibration knew:
  without_re500: set-point 0.784, band [0.764, 0.804]   (Re=1000 calibration alone)
  with_re500   : set-point 0.810, band [0.770, 0.850]   (both ground-truth regimes)
Rule per arm: blind-score all cells on the anchor's source pool -> keep band -> proxy guard
(veto if map-correlation falls >0.20 below the shallowest rung's) -> nearest set-point,
ties (within 0.02) -> shallower rung, then earlier checkpoint / base first.

Cells: at Re=2000 the full grid (base + the 12 new-pool checkpoints) x 9 rungs; at the other
eight generated regimes no native finetune exists, so base x 9 rungs.

PHASE A (selection) is GT-free: source-pool observations only.
PHASE B (audit) grades every BASE cell and every distinct pick on HELD-OUT sequences against
ground truth - after-the-fact verification on research data, exactly as all prior audits.
The oracle (ground-truth-best base cell per regime) comes free from grading all base cells.

Output: base_results/setpoint_ablation.npz
"""
import os, sys, glob, pickle, re as _re
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
LADDER = [('K1-50x12', [50], 12), ('K1-75x20', [75], 20), ('K1-100x20', [100], 20),
          ('K2x50', [100, 75], 50), ('K3x86', [150, 100, 50], 86),
          ('K4x110', [200, 150, 100, 50], 110), ('K5x140', [250, 200, 150, 100, 50], 140),
          ('K6x170', [300, 250, 200, 150, 100, 50], 170),
          ('K7x200', [350, 300, 250, 200, 150, 100, 50], 200)]
RUNG_ORDER = [c for c, _, _ in LADDER]
ARMS = {'without_re500': (0.784, 0.764, 0.804), 'with_re500': (0.810, 0.770, 0.850)}
PROXY_THR, TIE = 0.20, 0.02
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {}
for R in (1500, 3000, 4000, 5000, 6000, 7000, 8000):
    REGIMES[R] = dict(gt=GEN.format(R), anchor=f'base_results/regime_stats_re{R}_obsfit_gen.npz',
                      ck=None)
REGIMES[2000] = dict(gt=GEN.format(2000), anchor='base_results/regime_stats_re2000_obsfit_newgen.npz',
                     ck='monitoring/ddpo_re2000_newpool_ckpts')
REGIMES[10000] = dict(gt=GEN.format(10000), anchor='base_results/regime_stats_re10000_obsfit_newgen.npz',
                      ck=None)
PER_SRC, PER_HELD = 8, 10

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))


def batched(fn, x, seed, bs=16):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(x), bs):
        o.append(np.asarray(fn(jnp.asarray(x[i:i + bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)


def pool(gt, seqs, n_per, with_gt=False):
    xg, xl = [], []
    for s in seqs:
        q = load_sequence(gt, s)
        l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(l) - 1, n_per).astype(int)
        xl.append(l[i2])
        if with_gt:
            xg.append(build_triplets(q, MEAN, SIG)[i2])
    return (np.concatenate(xg) if with_gt else None), np.concatenate(xl)


SEL, AUD = {}, {}
for R, c in REGIMES.items():
    d = np.load(c['anchor']); A = d['spec_ref']
    src = d['obs_source'].item().decode().split('|seqs=')[1]
    src_seqs = eval(src)
    held = [s for s in range(20) if s not in src_seqs]
    dx = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    models = [('base', base_params)]
    if c['ck']:
        models += [(_re.search(r'iter(\d+)', p).group(1), pickle.load(open(p, 'rb'))['params'])
                   for p in sorted(glob.glob(f"{c['ck']}/*_iter*.pkl"))]
    # PHASE A - selection pool (anchor source, GT-free)
    _, xl = pool(c['gt'], src_seqs, PER_SRC)
    recon = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    Eh_rc = local_hik_energy(recon[..., 1] * SIG, HIK0, 6.0)
    print(f"\n=== Re={R}: SELECTION on seqs {src_seqs} ({len(xl)} triplets), "
          f"{len(models)} models x 9 rungs ===", flush=True)
    ref_proxy = None
    for cname, starts, steps in LADDER:
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=0.30)
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        for mname, P in models:
            y = batched(lambda xb, kk: smp(P, sa * xb + s1 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
            E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
            Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
            b = float(E[10:96].sum() / A[10:96].sum())
            px = float(np.corrcoef(Eh.ravel(), Eh_rc.ravel())[0, 1])
            if ref_proxy is None:
                ref_proxy = px if mname == 'base' else ref_proxy
            SEL[f'{R}|{cname}|{mname}'] = dict(blind=b, proxy=px)
            print(f"  {cname:<11} {mname:<6} blind={b:.3f} proxy={px:.3f}", flush=True)
    # apply both arms
    cells = [k for k in SEL if k.startswith(f'{R}|')]
    ref = SEL[f'{R}|K1-50x12|base']['proxy']
    picks = {}
    for arm, (sp, lo, hi) in ARMS.items():
        band = [k for k in cells if lo <= SEL[k]['blind'] <= hi]
        surv = [k for k in band if SEL[k]['proxy'] >= ref - PROXY_THR]
        if not surv:
            near = min(cells, key=lambda k: abs(SEL[k]['blind'] - sp))
            picks[arm] = ('NO_CELL_IN_BAND', near)
            print(f"  [{arm}] NO cell in band {lo}-{hi}; nearest {near.split('|',1)[1]} "
                  f"(blind {SEL[near]['blind']:.3f})", flush=True)
            continue
        dmin = min(abs(SEL[k]['blind'] - sp) for k in surv)
        tied = [k for k in surv if abs(SEL[k]['blind'] - sp) <= dmin + TIE]
        tied.sort(key=lambda k: (RUNG_ORDER.index(k.split('|')[1]),
                                 -1 if k.split('|')[2] == 'base' else int(k.split('|')[2])))
        picks[arm] = ('OK', tied[0])
        print(f"  [{arm}] pick: {tied[0].split('|',1)[1]} "
              f"(blind {SEL[tied[0]]['blind']:.3f}, {len(surv)} in band)", flush=True)
    SEL[f'{R}|_picks'] = picks
    # PHASE B - audit pool (held-out, GT used to grade)
    xg, xl2 = pool(c['gt'], held, PER_HELD, with_gt=True)
    recon2 = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl2, 500)
    Eg = np.asarray(spec_fn(jnp.asarray(xg))); E_gt = Eg.mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    to_grade = [(cn, 'base') for cn in RUNG_ORDER]
    for arm, (st, k) in picks.items():
        cn, mn = k.split('|')[1], k.split('|')[2]
        if (cn, mn) not in to_grade:
            to_grade.append((cn, mn))
    print(f"  AUDIT on seqs {held[:3]}..{held[-1]} ({len(xg)} triplets): "
          f"{len(to_grade)} cells", flush=True)
    mp = dict(models)
    for cname, mname in to_grade:
        starts, steps = next((s, st) for c2, s, st in LADDER if c2 == cname)
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=0.30)
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        y = batched(lambda xb, kk: smp(mp[mname], sa * xb + s1 * jax.random.normal(
            jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon2, 900)
        E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
        Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
        AUD[f'{R}|{cname}|{mname}'] = dict(
            ret=float(E[HIK0:96].sum() / E_gt[HIK0:96].sum()),
            place=float(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1]),
            lowk=float(E[1:5].sum() / E_gt[1:5].sum()), kstar=int(eff_resolution(E, E_gt)))
        r = AUD[f'{R}|{cname}|{mname}']
        print(f"    {cname:<11} {mname:<6} ret={r['ret']:.3f} place={r['place']:.3f} "
              f"k*={r['kstar']}", flush=True)
    np.savez('base_results/setpoint_ablation.npz',
             sel_keys=np.array([k for k in SEL if not k.endswith('_picks')]),
             aud_keys=np.array(list(AUD)),
             picks=np.array([f"{R2}|{a}|{st}|{k}" for R2 in REGIMES
                             for a, (st, k) in SEL.get(f'{R2}|_picks', {}).items()]),
             **{f'S|{k}||{f}': np.float32(v) for k, d2 in SEL.items()
                if not k.endswith('_picks') for f, v in d2.items()},
             **{f'A|{k}||{f}': np.float32(v) for k, d2 in AUD.items() for f, v in d2.items()})

print("\n===== ABLATION VERDICT =====", flush=True)
for R in REGIMES:
    pk = SEL.get(f'{R}|_picks', {})
    base_cells = [(cn, AUD[f'{R}|{cn}|base']) for cn in RUNG_ORDER if f'{R}|{cn}|base' in AUD]
    oracle = min(base_cells, key=lambda t: abs(t[1]['ret'] - 1.0))
    line = f"Re={R}: oracle(base)={oracle[0]} ret {oracle[1]['ret']:.3f}"
    for arm in ARMS:
        st, k = pk[arm]
        cn, mn = k.split('|')[1], k.split('|')[2]
        a = AUD.get(f'{R}|{cn}|{mn}')
        line += f" | {arm}: {cn}|{mn} ret {a['ret']:.3f}" + (" (NO-BAND)" if st != 'OK' else "")
    print(line, flush=True)
print("ABLATION COMPLETE", flush=True)
