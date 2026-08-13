"""GRADE THE MULTI-REGIME (oracle-anchored) MODEL: every regime, across the cascade ladder.

PHASE ORDER (user's priority): the MEAN-DOSE TEST runs first - the multi-regime model at the
single fixed depth K2x50 (exactly the training policy's chain [100,75]x50) across all nine
regimes, printed as one row with its verdict, before any other cell is graded. If the
unconditioned model learned one compromise dose, that row tilts around the middle of the training
family (overshoot at Re=1000-2000, sweet spot ~3000-4000, undershoot at 7000-8000). If it learned
to read the input's regime signal, retention sits near 1 across the row.

Then: the rest of the ladder (single-config-everywhere vs depth-compensation question), the
EMA-vs-raw check, and the ablation-winning baseline cells re-run on the same pools.

Pools are DISJOINT from multi-regime training (train used gen seqs 0-7 and Re=1000 seqs 20-27):
gen regimes grade on seqs 8-19, Re=1000 on its standard held-out 4-19. GT grading is legitimate
here: the model is the ORACLE variant (measured GT anchors, labelled NOT DEPLOYABLE) - this is
an audit of what one shared model can do, not a deployment selection.

Every cell also records its blind score against the regime's obs-fit anchor, so if depth must
vary per regime we can check post-hoc whether the engraved GT-free rule finds this model's depth.

Resume-aware: rerun after a crash and computed cells are skipped.
Output: base_results/multiregime_grade.npz  (+ per-cell mean spectra for plots)
"""
import os, sys, glob, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn, make_residual_loss
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
LADDER = {'K1-50x12': ([50], 12), 'K2x50': ([100, 75], 50), 'K3x86': ([150, 100, 50], 86),
          'K4x110': ([200, 150, 100, 50], 110), 'K5x140': ([250, 200, 150, 100, 50], 140),
          'K6x170': ([300, 250, 200, 150, 100, 50], 170)}
MEANDOSE_RUNG = 'K2x50'                 # == the training policy depth
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {
    1000: dict(gt='flow-data/kf_2d_re1000_256_40seed.npy', seqs=list(range(4, 20)), per=8,
               anchor='base_results/regime_stats_re1000_obsfit.npz'),
    **{R: dict(gt=GEN.format(R), seqs=list(range(8, 20)), per=10,
               anchor=f'base_results/regime_stats_re{R}_obsfit_gen.npz')
       for R in (1500, 3000, 4000, 5000, 6000, 7000, 8000)},
    2000: dict(gt=GEN.format(2000), seqs=list(range(8, 20)), per=10,
               anchor='base_results/regime_stats_re2000_obsfit_newgen.npz'),
}
ORDER = sorted(REGIMES)
REST_RUNGS = {R: ['K1-50x12', 'K3x86', 'K4x110', 'K5x140'] + (['K6x170'] if R >= 6000 else [])
              for R in ORDER}
EMA_RUNGS = ['K2x50', 'K3x86']          # training-adjacent depths only, raw-vs-EMA check
# the ablation's winning cell per regime, re-run on THIS pool for a like-for-like baseline
R1K = 'monitoring/ddpo_re1000_newpool_ckpts/ddpo_re1000_iter{:04d}.pkl'
BASELINE = {1000: ('K3x86', 'r1k-149', R1K.format(149)),
            1500: ('K3x86', 'r1k-449', R1K.format(449)),
            2000: ('K3x86', 'native-149', 'monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter0149.pkl'),
            3000: ('K4x110', 'r1k-149', R1K.format(149)),
            4000: ('K5x140', 'r1k-449', R1K.format(449)),
            5000: ('K5x140', 'r1k-449', R1K.format(449)),
            6000: ('K5x140', 'r1k-149', R1K.format(149)),
            7000: ('K5x140', 'r1k-149', R1K.format(149)),
            8000: ('K5x140', 'r1k-149', R1K.format(149))}

mr_ckpts = sorted(glob.glob('monitoring/ddpo_multiregime_gt_ckpts/ddpo_multi_iter*.pkl'))
assert mr_ckpts, 'no multi-regime checkpoints found'
MR_PATH = mr_ckpts[-1]
mr = pickle.load(open(MR_PATH, 'rb'))
print(f"multi-regime checkpoint: {MR_PATH} (iter {mr['iter']})", flush=True)
MR = {'mr': mr['params']}
if 'ema_params' in mr:
    MR['mr-ema'] = mr['ema_params']

OUTP = 'base_results/multiregime_grade.npz'
OUT = {}
if os.path.exists(OUTP):
    old = np.load(OUTP, allow_pickle=True)
    OUT = {k: old[k] for k in old.files}
    print(f"resume: {len([k for k in OUT if k.endswith('||ret')])} cells already graded", flush=True)

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
BASE_CK = {}      # baseline params cache


def batched(fn, x, seed, bs=16):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(x), bs):
        o.append(np.asarray(fn(jnp.asarray(x[i:i + bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)


# ---- build all pools + regime stats up front (kept in memory; xg/xl dropped after use) ----
POOL = {}
for R in ORDER:
    cfg = REGIMES[R]
    xg, xl = [], []
    for s in cfg['seqs']:
        q = load_sequence(cfg['gt'], s)
        g = build_triplets(q, MEAN, SIG)
        l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, cfg['per']).astype(int)
        xg.append(g[i2]); xl.append(l[i2])
    xg, xl = np.concatenate(xg), np.concatenate(xl)
    resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
    d = dict(
        dx=make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN), resid_fn=resid_fn,
        E_gt=np.asarray(spec_fn(jnp.asarray(xg))).mean(0),
        Ehg=local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0),
        rg=float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                                 for i in range(0, len(xg), 32)]).mean()),
        A=np.load(cfg['anchor'])['spec_ref'], n=len(xg))
    d['recon'] = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    POOL[R] = d
    print(f"  Re={R}: {d['n']} val triplets (seqs {cfg['seqs'][0]}..{cfg['seqs'][-1]})", flush=True)


def measure(R, y, key, label):
    d = POOL[R]
    E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
    ry = float(np.concatenate([np.asarray(d['resid_fn'](jnp.asarray(y[i:i + 32]))).ravel()
                               for i in range(0, len(y), 32)]).mean())
    Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
    vals = dict(ret=E[HIK0:96].sum() / d['E_gt'][HIK0:96].sum(),
                place=np.corrcoef(Eh.ravel(), d['Ehg'].ravel())[0, 1],
                lowk=E[1:5].sum() / d['E_gt'][1:5].sum(), kstar=eff_resolution(E, d['E_gt']),
                resid_ratio=ry / d['rg'], blind=E[10:96].sum() / d['A'][10:96].sum())
    for f, v in vals.items():
        OUT[f'{key}||{f}'] = np.float32(v)
    OUT[f'{key}||E'] = E.astype(np.float32)
    OUT[f'{R}|GT||E'] = d['E_gt'].astype(np.float32)
    print(f"    Re={R} {label:<28} ret={vals['ret']:.3f}  place={vals['place']:.3f}  "
          f"lowk={vals['lowk']:.3f}  k*={vals['kstar']}  resid={vals['resid_ratio']:.2f}xGT  "
          f"blind={vals['blind']:.3f}", flush=True)
    return vals


def run_cell(R, nm, cname):
    key = f'{R}|{nm}|{cname}'
    if f'{key}||ret' in OUT:
        return dict(ret=float(OUT[f'{key}||ret']))
    starts, steps = LADDER[cname]
    smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, POOL[R]['dx'], 3.0, temp=0.30)
    sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
    if nm in MR:
        P = MR[nm]
    else:
        path = BASELINE[R][2]
        if path not in BASE_CK:
            BASE_CK[path] = pickle.load(open(path, 'rb'))['params']
        P = BASE_CK[path]
    y = batched(lambda xb, kk: smp(P, sa * xb + s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), POOL[R]['recon'], 700)
    vals = measure(R, y, key, f'{nm} @ {cname}')
    np.savez(OUTP, **OUT)
    return vals


# ---- PHASE 1: THE MEAN-DOSE TEST (fixed training-policy depth, all nine regimes) ----
print(f"\n===== PHASE 1: MEAN-DOSE TEST - mr @ {MEANDOSE_RUNG} across all regimes =====", flush=True)
row = {}
for R in ORDER:
    if f'{R}|recon|-||ret' not in OUT:
        measure(R, POOL[R]['recon'], f'{R}|recon|-', 'base recon (DDIM20)')
    row[R] = run_cell(R, 'mr', MEANDOSE_RUNG)
print("\nMEAN-DOSE ROW (retention at fixed depth " + MEANDOSE_RUNG + "):", flush=True)
print("  " + "  ".join(f"Re{R}:{row[R]['ret']:.3f}" for R in ORDER), flush=True)
near = np.mean([row[R]['ret'] for R in (1000, 1500, 2000)])
mid = np.mean([row[R]['ret'] for R in (3000, 4000, 5000)])
far = np.mean([row[R]['ret'] for R in (6000, 7000, 8000)])
print(f"  near(1000-2000)={near:.3f}  mid(3000-5000)={mid:.3f}  far(6000-8000)={far:.3f}", flush=True)
print("  READ: flat near 1 everywhere = input-dependent dose; near>1 with far<<1 tilting around "
      "the middle = compromise (mean) dose", flush=True)

# ---- PHASE 2: rest of the ladder (does one config work, or must depth compensate?) ----
print("\n===== PHASE 2: full ladder =====", flush=True)
for R in ORDER:
    for c in REST_RUNGS[R]:
        run_cell(R, 'mr', c)

# ---- PHASE 3: EMA-vs-raw + like-for-like baselines ----
print("\n===== PHASE 3: EMA check + ablation-winner baselines on the same pools =====", flush=True)
for R in ORDER:
    if 'mr-ema' in MR:
        for c in EMA_RUNGS:
            run_cell(R, 'mr-ema', c)
    bl_rung, bl_name, _ = BASELINE[R]
    run_cell(R, bl_name, bl_rung)

np.savez(OUTP, **OUT)
print("\nMULTIREGIME GRADE COMPLETE", flush=True)
