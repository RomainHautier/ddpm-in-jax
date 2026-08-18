"""MODEL CARDS: the full stat battery for every deployable checkpoint, across every regime
and the relevant cascade depths — the model-centric view the user asked for.

Models: the shared base; the Re=1000-targeted fine-tune (checkpoints 149 and 449 — the two the
anchor ever chooses); the Re=2000-targeted fine-tune (149 and 599). One backbone throughout.
Regimes: all nine (1000-8000). Depths: K3/K4/K5, the range the anchor's choices span.

Per cell: band retention E[32,96), low-k retention E[1,5), high-k placement correlation vs
ground truth, PDE residual / GT floor, effective resolution k*, blind score vs the regime's
extrapolated anchor — plus ONE reconstruction frame saved per cell, and per regime a GT frame
and base-recon frame, so the report can show reconstructions next to the numbers.

Pools: the standard validation splits (gen regimes seqs 8-19, Re=1000 seqs 4-19) — identical
to multiregime_grade.npz, into which results are merged (resume-aware; existing cells skipped).
Fields go to base_results/model_cards_fields.npz. All-chip sampling (PSAMPLE=0 for serial).
"""
import os, sys, pickle
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
from psample import pbatched as batched

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
LADDER = {'K3x86': ([150, 100, 50], 86), 'K4x110': ([200, 150, 100, 50], 110),
          'K5x140': ([250, 200, 150, 100, 50], 140)}
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
R1K = 'monitoring/ddpo_re1000_newpool_ckpts/ddpo_re1000_iter{:04d}.pkl'
R2K = 'monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter{:04d}.pkl'
# TRACK-A NOMINATION (user 2026-08-18): this sweep assumes the TRUE statistics of every regime
# are known but NO samples exist — so the fine-tuned checkpoints entering it are chosen as the
# best match to their home regime's true statistics (|1-ret| minimal, low-k healthy; placement
# and MSE are sample-paired and were NEVER consulted): r1k-449 (ret 0.999 @K3, lowk 0.970) and
# re2k-599 (ret 0.968 @K3, lowk 0.948; ckpt-549 is the strict lowk>=0.95 alternative). The
# anchor-picked 149s belong to the extrapolated-anchor track (the Anchor's Choice page), not
# here; their cells stay in the npz from the earlier pass.
MODELS = {'base': None, 'r1k-449': R1K.format(449), 're2k-599': R2K.format(599)}

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
PARAMS = {nm: (base_params if p is None else pickle.load(open(p, 'rb'))['params'])
          for nm, p in MODELS.items()}

OUTP = 'base_results/multiregime_grade.npz'
FLDP = 'base_results/model_cards_fields.npz'
OUT, FLD = {}, {}
if os.path.exists(OUTP):
    old = np.load(OUTP, allow_pickle=True); OUT = {k: old[k] for k in old.files}
    print(f"merge: {len([k for k in OUT if k.endswith('||ret')])} cells already in file", flush=True)
if os.path.exists(FLDP):
    old = np.load(FLDP, allow_pickle=True); FLD = {k: old[k] for k in old.files}

for R in ORDER:
    cfg = REGIMES[R]
    todo = [(nm, c) for c in LADDER for nm in MODELS if f'{R}|{nm}|{c}||mse' not in OUT]
    if not todo and f'{R}|GT' in FLD:
        print(f"=== Re={R}: complete ===", flush=True); continue
    xg, xl = [], []
    for s in cfg['seqs']:
        q = load_sequence(cfg['gt'], s)
        g = build_triplets(q, MEAN, SIG)
        l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, cfg['per']).astype(int)
        xg.append(g[i2]); xl.append(l[i2])
    xg, xl = np.concatenate(xg), np.concatenate(xl)
    resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
    E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                               for i in range(0, len(xg), 32)]).mean())
    A = np.load(cfg['anchor'])['spec_ref']
    dx = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    recon = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    FLD[f'{R}|GT'] = (xg[0, ..., 1] * SIG).astype(np.float32)
    FLD[f'{R}|recon'] = (recon[0, ..., 1] * SIG).astype(np.float32)
    OUT[f'{R}|GT||E'] = E_gt.astype(np.float32)
    OUT[f'{R}|GT||resid'] = np.float32(rg)
    print(f"\n=== Re={R}: {len(xg)} val triplets, {len(todo)} cells ===", flush=True)
    for cname, (starts, steps) in LADDER.items():
        names = [nm for nm, c in todo if c == cname]
        if not names:
            continue
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=0.30)
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        for nm in names:
            y = batched(lambda xb, kk: smp(PARAMS[nm], sa * xb + s1 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
            E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
            ry = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                                       for i in range(0, len(y), 32)]).mean())
            Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
            key = f'{R}|{nm}|{cname}'
            vals = dict(ret=E[HIK0:96].sum() / E_gt[HIK0:96].sum(),
                        place=np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1],
                        lowk=E[1:5].sum() / E_gt[1:5].sum(), kstar=eff_resolution(E, E_gt),
                        resid_ratio=ry / rg, blind=E[10:96].sum() / A[10:96].sum(),
                        mse=np.mean((y[..., 1] - xg[..., 1]) ** 2) * SIG ** 2)
            for f, v in vals.items():
                OUT[f'{key}||{f}'] = np.float32(v)
            OUT[f'{key}||E'] = E.astype(np.float32)
            FLD[key] = (y[0, ..., 1] * SIG).astype(np.float32)
            print(f"    {nm:<9} @ {cname}  ret={vals['ret']:.3f}  lowk={vals['lowk']:.3f}  "
                  f"place={vals['place']:.3f}  resid={vals['resid_ratio']:.2f}xGT  "
                  f"mse={vals['mse']:.2f}  k*={vals['kstar']}  blind={vals['blind']:.3f}",
                  flush=True)
            np.savez(OUTP, **OUT)
            np.savez_compressed(FLDP, **FLD)
print("\nMODEL CARDS GRADE COMPLETE", flush=True)
