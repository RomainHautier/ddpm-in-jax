"""GRADE THE MULTI-REGIME (oracle-anchored) MODEL: every regime, across the cascade ladder.

The questions this answers, per the user's framing:
1. Mean-dose or input-dependent? If the unconditioned model learned one compromise dose, the
   retention profile across regimes at a fixed depth will tilt around the middle of the training
   family (overshoot at Re=1000-2000, sweet spot ~3000-4000, undershoot at 7000-8000). If it
   learned to read the input's regime signal, retention sits near 1 across the row at one depth.
2. Does a SINGLE inference config work across regimes, or does depth still have to compensate?
   The full ladder K1..K5 (K6 added at the three farthest regimes) is graded everywhere.
3. If depth must vary: would the engraved blind rule have found it for this model? Every cell's
   blind score against the regime's obs-fit anchor is recorded alongside the GT grade.

Pools are DISJOINT from multi-regime training (train used gen seqs 0-7 and Re=1000 seqs 20-27):
gen regimes grade on seqs 8-19, Re=1000 on its standard held-out 4-19. GT grading is legitimate
here: the model is the ORACLE variant (measured GT anchors, labelled NOT DEPLOYABLE) - this is
an audit of what one shared model can do, not a deployment selection.

Baselines re-run on the SAME pools: the ablation's winning cell per regime (borrowed r1k
checkpoints; the native ckpt-149 at Re=2000; the verified healthy cell at Re=1000) so the
multi-regime model vs specialist comparison shares sequences, triplets and noise seeds.

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
MR_RUNGS = {R: ['K1-50x12', 'K2x50', 'K3x86', 'K4x110', 'K5x140']
               + (['K6x170'] if R >= 6000 else []) for R in REGIMES}
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
DONE = {}
if os.path.exists(OUTP):
    old = np.load(OUTP, allow_pickle=True)
    DONE = {k: old[k] for k in old.files}
    print(f"resume: {len([k for k in DONE if k.endswith('||ret')])} cells already graded", flush=True)

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


OUT = dict(DONE)
for R in sorted(REGIMES):
    cfg = REGIMES[R]
    # regime work list: (model label, params-getter, rung)
    work = [('mr', c) for c in MR_RUNGS[R]]
    if 'mr-ema' in MR:
        work += [('mr-ema', c) for c in EMA_RUNGS]
    bl_rung, bl_name, bl_path = BASELINE[R]
    work += [(bl_name, bl_rung)]
    work = [(nm, c) for nm, c in work if f'{R}|{nm}|{c}||ret' not in OUT]
    if not work and f'{R}|recon|-||ret' in OUT:
        print(f"=== Re={R}: all cells already graded ===", flush=True)
        continue

    xg, xl = [], []
    for s in cfg['seqs']:
        q = load_sequence(cfg['gt'], s)
        g = build_triplets(q, MEAN, SIG)
        l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, cfg['per']).astype(int)
        xg.append(g[i2]); xl.append(l[i2])
    xg, xl = np.concatenate(xg), np.concatenate(xl)
    print(f"\n=== Re={R}: VAL seqs {cfg['seqs'][0]}..{cfg['seqs'][-1]} ({len(xg)} triplets), "
          f"{len(work)} cells ===", flush=True)

    dx = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
    E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                               for i in range(0, len(xg), 32)]).mean())
    A = np.load(cfg['anchor'])['spec_ref']
    recon = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)

    def measure(y, key, label):
        E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
        ry = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                                   for i in range(0, len(y), 32)]).mean())
        Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
        vals = dict(ret=E[HIK0:96].sum() / E_gt[HIK0:96].sum(),
                    place=np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1],
                    lowk=E[1:5].sum() / E_gt[1:5].sum(), kstar=eff_resolution(E, E_gt),
                    resid_ratio=ry / rg, blind=E[10:96].sum() / A[10:96].sum())
        for f, v in vals.items():
            OUT[f'{key}||{f}'] = np.float32(v)
        OUT[f'{key}||E'] = E.astype(np.float32)
        OUT[f'{R}|GT||E'] = E_gt.astype(np.float32)
        print(f"    {label:<28} ret={vals['ret']:.3f}  place={vals['place']:.3f}  "
              f"lowk={vals['lowk']:.3f}  k*={vals['kstar']}  resid={vals['resid_ratio']:.2f}xGT  "
              f"blind={vals['blind']:.3f}", flush=True)

    if f'{R}|recon|-||ret' not in OUT:
        measure(recon, f'{R}|recon|-', 'base recon (DDIM20)')
    by_cfg = {}
    for nm, c in work:
        by_cfg.setdefault(c, []).append(nm)
    for cname, names in by_cfg.items():     # config-outer: each cascade compiles once per regime
        starts, steps = LADDER[cname]
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=0.30)
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        for nm in names:
            if nm in MR:
                P = MR[nm]
            else:
                if bl_path not in BASE_CK:
                    BASE_CK[bl_path] = pickle.load(open(bl_path, 'rb'))['params']
                P = BASE_CK[bl_path]
            y = batched(lambda xb, kk: smp(P, sa * xb + s1 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
            measure(y, f'{R}|{nm}|{cname}', f'{nm} @ {cname}')
        np.savez(OUTP, **OUT)
    np.savez(OUTP, **OUT)

print("\nMULTIREGIME GRADE COMPLETE", flush=True)
