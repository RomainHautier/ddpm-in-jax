"""GRADE THE FiLM MULTI-REGIME MODEL: the single-config mean-dose acceptance test.

Third conditioning arm. Same user-scoped test as the residual-feedback arms: ONE fixed
inference config (the training depth K2x50); across the nine regimes only the conditioning
changes — here the scalar regime code log(Re/1000)/log 8 fed through the FiLM pathway.
Writes into the SAME npz as every previous variant (shared pools, seeds, keys), so the row
prints directly against the unconditioned reference (0.797 -> 0.281) and the six
residual-conditioned rows that all decayed in lockstep with it.

Variants: final / best-probe (chosen automatically from probe_history.json by highest mean
probe reward across regimes — training-pool information only) / EMA-of-final.
"""
import os, sys, json, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.models.model import FiLMUnet
from src.rewards import make_spectrum_fn, make_residual_loss
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution
from psample import pbatched as batched     # all-chip sampling; PSAMPLE=0 restores serial

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
MEANDOSE_RUNG, STARTS, STEPS = 'K2x50', [100, 75], 50
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
DIR = 'monitoring/ddpo_multiregime_film_gt_ckpts'


def RECODE(x, visc):
    code = jnp.log(1.0 / (1000.0 * visc)) / jnp.log(8.0)
    return jnp.full((x.shape[0],), 1.0, jnp.float32) * code


hist = json.load(open(f'{DIR}/probe_history.json'))
mean_probe = {row['iter']: np.mean([row[str(R)] for R in ORDER]) for row in hist}
best_iter = max(mean_probe, key=mean_probe.get)
print(f"best-probe checkpoint: iter {best_iter} (mean probe {mean_probe[best_iter]:.2f}); "
      f"final iter {hist[-1]['iter']} (mean probe {mean_probe[hist[-1]['iter']]:.2f})", flush=True)
CK = DIR + '/ddpo_multifilm_iter{:04d}.pkl'
MODELS = {'film-final': (CK.format(hist[-1]['iter']), 'params'),
          'film-ema': (CK.format(hist[-1]['iter']), 'ema_params')}
if best_iter != hist[-1]['iter']:
    MODELS[f'film-{best_iter}'] = (CK.format(best_iter), 'params')

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
u = ddpm.unet
film_unet = FiLMUnet(ch=u.ch, ch_mult=u.ch_mult, out_ch=u.out_ch, in_ch=u.in_ch,
                     n_resnet_blocks=u.n_resnet_blocks, dropout_p=u.dropout_p,
                     freq_dim=u.freq_dim)
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


def run_cell(R, nm):
    key = f'{R}|{nm}|{MEANDOSE_RUNG}'
    if f'{key}||ret' in OUT:
        return dict(ret=float(OUT[f'{key}||ret']))
    d = POOL[R]
    smp = make_kchain_ddim_sampler(film_unet, ab, STARTS, STEPS, d['dx'], 3.0, temp=0.30,
                                   cond_fn=RECODE, cond_visc=jnp.float32(1.0 / R))
    sa, s1 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
    y = batched(lambda xb, kk: smp(PARAMS[nm], sa * xb + s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), d['recon'], 700)
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
    print(f"    Re={R} {nm:<12} @ {MEANDOSE_RUNG}  ret={vals['ret']:.3f}  "
          f"place={vals['place']:.3f}  k*={vals['kstar']}  blind={vals['blind']:.3f}", flush=True)
    np.savez(OUTP, **OUT)
    return vals


print(f"\n===== FiLM MEAN-DOSE ACCEPTANCE TEST @ {MEANDOSE_RUNG} =====", flush=True)
uncond_row = {R: float(OUT[f'{R}|mr|{MEANDOSE_RUNG}||ret']) for R in ORDER}
print("  UNCOND ref: " + "  ".join(f"Re{R}:{uncond_row[R]:.3f}" for R in ORDER), flush=True)
for nm in MODELS:
    row = {R: run_cell(R, nm) for R in ORDER}
    print(f"\n  {nm} ROW: " + "  ".join(f"Re{R}:{row[R]['ret']:.3f}" for R in ORDER), flush=True)
    near = np.mean([row[R]['ret'] for R in (1000, 1500, 2000)])
    mid = np.mean([row[R]['ret'] for R in (3000, 4000, 5000)])
    far = np.mean([row[R]['ret'] for R in (6000, 7000, 8000)])
    flat = np.mean([abs(1.0 - row[R]['ret']) for R in ORDER])
    print(f"  {nm}: near={near:.3f}  mid={mid:.3f}  far={far:.3f}  mean|1-ret|={flat:.3f}",
          flush=True)
np.savez(OUTP, **OUT)
print("\nMULTIREGIME-FILM GRADE COMPLETE", flush=True)
