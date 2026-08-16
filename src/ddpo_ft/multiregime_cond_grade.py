"""GRADE THE CONDITIONED MULTI-REGIME MODELS: the mean-dose acceptance test, then the ladder.

Companion to multiregime_grade.py (which graded the UNCONDITIONED multi-regime model and the
baselines) — writes into the SAME npz so every model shares pools, noise seeds and keys.

Models: both conditioned arms (from-zero and adapter-warm-start), each as final / best-probe /
EMA-of-final. Both trajectories oscillated (unlike the smooth unconditioned run), so the final
checkpoint is not presumed best; best-probe rows were iter 1259 (from-zero) and iter 1169
(adapter) — probe-based selection uses training-pool reward only, no validation data.

PHASE 1 — the acceptance test: every variant at the fixed training depth K2x50 across all nine
regimes, printed against the unconditioned row (0.797 -> 0.281). Conditioning works if and only
if a conditioned row is FLAT near 1 where that row decayed.
PHASE 2 — the ladder for the best phase-1 variant (best mean |1-ret| over the row), to see how
depth interacts with whatever dose behaviour the conditioning learned.

Sampling matches every previously graded cell (lambda=3 regime-correct guidance, temp 0.30,
same seeds) plus the condRes input at every eps prediction (cond-aware kchain sampler).
"""
import os, sys, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.models.model import ConditionalUnet
from src.physics_guidance import make_field_func_visc
from src.rewards import make_spectrum_fn, make_residual_loss
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
LADDER = {'K1-50x12': ([50], 12), 'K2x50': ([100, 75], 50), 'K3x86': ([150, 100, 50], 86),
          'K4x110': ([200, 150, 100, 50], 110), 'K5x140': ([250, 200, 150, 100, 50], 140),
          'K6x170': ([300, 250, 200, 150, 100, 50], 170)}
MEANDOSE_RUNG = 'K2x50'
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
CZ = 'monitoring/ddpo_multiregime_cond_gt_ckpts/ddpo_multicond_iter{:04d}.pkl'
CA = 'monitoring/ddpo_multiregime_cond_gt_ckpts_adapter/ddpo_multicond_iter{:04d}.pkl'
MODELS = {                      # name -> (ckpt path, param key)
    'cz-final': (CZ.format(1349), 'params'),
    'cz-1259':  (CZ.format(1259), 'params'),
    'cz-ema':   (CZ.format(1349), 'ema_params'),
    'ca-final': (CA.format(1349), 'params'),
    'ca-1169':  (CA.format(1169), 'params'),
    'ca-ema':   (CA.format(1349), 'ema_params'),
}
REST_RUNGS = ['K1-50x12', 'K3x86', 'K4x110', 'K5x140']       # phase 2, best variant only (+K6 far)

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
u = ddpm.unet
cond_unet = ConditionalUnet(ch=u.ch, ch_mult=u.ch_mult, out_ch=u.out_ch, in_ch=u.in_ch,
                            n_resnet_blocks=u.n_resnet_blocks, dropout_p=u.dropout_p,
                            freq_dim=u.freq_dim)
FIELD = make_field_func_visc(n=256, std=SIG, mean=MEAN)
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
PARAMS = {nm: pickle.load(open(p, 'rb'))[key] for nm, (p, key) in MODELS.items()}

OUTP = 'base_results/multiregime_grade.npz'
OUT = {}
if os.path.exists(OUTP):
    old = np.load(OUTP, allow_pickle=True)
    OUT = {k: old[k] for k in old.files}
    print(f"resume/merge: {len([k for k in OUT if k.endswith('||ret')])} cells already in file",
          flush=True)


def batched(fn, x, seed, bs=16):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(x), bs):
        o.append(np.asarray(fn(jnp.asarray(x[i:i + bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)


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
    print(f"  Re={R}: {d['n']} val triplets", flush=True)


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
    print(f"    Re={R} {label:<24} ret={vals['ret']:.3f}  place={vals['place']:.3f}  "
          f"lowk={vals['lowk']:.3f}  k*={vals['kstar']}  resid={vals['resid_ratio']:.2f}xGT  "
          f"blind={vals['blind']:.3f}", flush=True)
    return vals


def run_cell(R, nm, cname):
    key = f'{R}|{nm}|{cname}'
    if f'{key}||ret' in OUT:
        return dict(ret=float(OUT[f'{key}||ret']))
    starts, steps = LADDER[cname]
    smp = make_kchain_ddim_sampler(cond_unet, ab, starts, steps, POOL[R]['dx'], 3.0, temp=0.30,
                                   cond_fn=FIELD, cond_visc=jnp.float32(1.0 / R))
    sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
    y = batched(lambda xb, kk: smp(PARAMS[nm], sa * xb + s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), POOL[R]['recon'], 700)
    vals = measure(R, y, key, f'{nm} @ {cname}')
    np.savez(OUTP, **OUT)
    return vals


# ---- PHASE 1: the acceptance test ----
print(f"\n===== PHASE 1: MEAN-DOSE ACCEPTANCE TEST — all variants @ {MEANDOSE_RUNG} =====", flush=True)
uncond_row = {R: float(OUT[f'{R}|mr|{MEANDOSE_RUNG}||ret']) for R in ORDER}
print("  UNCOND ref: " + "  ".join(f"Re{R}:{uncond_row[R]:.3f}" for R in ORDER), flush=True)
rows = {}
for nm in MODELS:
    rows[nm] = {R: run_cell(R, nm, MEANDOSE_RUNG) for R in ORDER}
    print(f"\n  {nm} ROW: " + "  ".join(f"Re{R}:{rows[nm][R]['ret']:.3f}" for R in ORDER), flush=True)
    near = np.mean([rows[nm][R]['ret'] for R in (1000, 1500, 2000)])
    mid = np.mean([rows[nm][R]['ret'] for R in (3000, 4000, 5000)])
    far = np.mean([rows[nm][R]['ret'] for R in (6000, 7000, 8000)])
    flat = np.mean([abs(1.0 - rows[nm][R]['ret']) for R in ORDER])
    print(f"  {nm}: near={near:.3f}  mid={mid:.3f}  far={far:.3f}  mean|1-ret|={flat:.3f}", flush=True)

best = min(rows, key=lambda nm: np.mean([abs(1.0 - rows[nm][R]['ret']) for R in ORDER]))
print(f"\nBEST VARIANT BY ROW FLATNESS: {best}", flush=True)

# ---- PHASE 2: ladder for the best variant ----
print(f"\n===== PHASE 2: ladder for {best} =====", flush=True)
for R in ORDER:
    for c in REST_RUNGS + (['K6x170'] if R >= 6000 else []):
        run_cell(R, best, c)

np.savez(OUTP, **OUT)
print("\nMULTIREGIME-COND GRADE COMPLETE", flush=True)
