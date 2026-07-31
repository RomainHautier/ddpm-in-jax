"""CROSS-REGIME DECOMPOSITION — every model on every regime, one frozen inference config.

The question: when you point a finetuned model at a regime it was not trained for, what happens to
(a) the PDE residual, (b) energy retention, (c) placement, and (d) the energy budget BAND BY BAND?

Design:
- Abscissa = the regime being inferred on (1000 ... 10000). One curve per model.
- ONE inference config everywhere (the frozen deployment cascade K3 [150,100,50] x86, lam3,
  itemp 0.30) so the curves isolate the MODEL, not the config.
- The PDE residual is Re-dependent, so the GT residual is measured on the same triplets and
  reported as a floor; the meaningful quantity across regimes is resid/resid_GT.
- Independent units are SEQUENCES (adjacent triplets are ~98% correlated), so every error bar is a
  bootstrap over sequences, and every regime uses as many sequences as its file allows.
- Per-triplet band energies are saved so any band decomposition can be recut without re-running.

Output: base_results/crossregime_decomposition.npz  (+ a readable log on stdout)
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

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
BAND_EDGES = [1, 5, 10, 16, 24, 32, 48, 64, 80, 96, 128]
PER_SEQ_GEN = int(os.environ.get("PER_SEQ", 40))
SMOKE = os.environ.get("SMOKE", "0") == "1"

MODELS = {
    'in-dist Re=1000': 'monitoring/ddpo_re1000_k2_s100-75_ddim50_ddiminit_hk10_emabase_ckpts/ddpo_re1000_iter0299.pkl',
    'Re=2000 model':   'monitoring/ddpo_re2000_dt32_ckpts/ddpo_re1000_iter0599.pkl',
    'Re=10000 model':  'monitoring/ddpo_re10000_dt32_ckpts/ddpo_re1000_iter0549.pkl',
}
# seqs: the sequences this regime's file makes available for TEST (training pools excluded).
# gen files hold 20 sequences x many frames; the dt32 files hold 40 sequences x 2 triplets.
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {
 1000:  dict(gt='flow-data/kf_2d_re1000_256_40seed.npy', seqs=list(range(4, 20)), per=PER_SEQ_GEN,
             anchor='base_results/regime_stats_re1000.npz', note='legacy measured-stats anchor'),
 1500:  dict(gt=GEN.format(1500), seqs=list(range(4, 20)), per=PER_SEQ_GEN,
             anchor='base_results/regime_stats_re1500_obsfit_gen.npz', note=''),
 2000:  dict(gt='flow-data/kf_re2000_256_40seed_dt32.npy', seqs=[s for s in range(40) if s > 4], per=None,
             anchor='base_results/regime_stats_re2000_obsfit_v3.npz', note='dt32 file ceiling'),
 3000:  dict(gt=GEN.format(3000), seqs=list(range(4, 20)), per=PER_SEQ_GEN,
             anchor='base_results/regime_stats_re3000_obsfit_gen.npz', note=''),
 4000:  dict(gt=GEN.format(4000), seqs=list(range(4, 20)), per=PER_SEQ_GEN,
             anchor='base_results/regime_stats_re4000_obsfit_gen.npz', note=''),
 5000:  dict(gt=GEN.format(5000), seqs=list(range(4, 20)), per=PER_SEQ_GEN,
             anchor='base_results/regime_stats_re5000_obsfit_gen.npz', note=''),
 10000: dict(gt='flow-data/kf_re10000_256_40seed_dt32.npy', seqs=[s for s in range(40) if not 20 <= s <= 24],
             per=None, anchor='base_results/regime_stats_re10000_obsfit_dt32.npz',
             note='dt32 file ceiling; this dataset is under suspicion (weak tail for its Re)'),
}
if SMOKE:
    for c in REGIMES.values():
        c['seqs'] = c['seqs'][:3]
        if c['per']:
            c['per'] = 4
    REGIMES = {k: v for k, v in REGIMES.items() if k in (1500, 2000)}

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sat, s1t = float(jnp.sqrt(ab[150])), float(jnp.sqrt(1.0 - ab[150]))
PARAMS = {k: pickle.load(open(v, 'rb'))['params'] for k, v in MODELS.items()}


def batched(fn, x, seed, bs=16):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(x), bs):
        o.append(np.asarray(fn(jnp.asarray(x[i:i + bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)


def band_sums(E_all):
    """(n_triplets, 128) spectra -> (n_triplets, n_bands) energy in each band."""
    return np.stack([E_all[:, a:b].sum(1) for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:])], 1)


OUT, FIELDS = {}, {}
for R, cfg in REGIMES.items():
    xg, xl, sid = [], [], []
    for s in cfg['seqs']:
        seq = load_sequence(cfg['gt'], s)
        g = build_triplets(seq, MEAN, SIG)
        l = build_triplets(grid_downsample_degrade(seq, 4), MEAN, SIG)
        if cfg['per']:
            i2 = np.linspace(0, len(g) - 1, cfg['per']).astype(int)
            g, l = g[i2], l[i2]
        xg.append(g); xl.append(l); sid += [s] * len(g)
    xg = np.concatenate(xg); xl = np.concatenate(xl); sid = np.array(sid)
    us = np.unique(sid); idx_by_seq = {s: np.where(sid == s)[0] for s in us}
    rng = np.random.default_rng(0)
    boot_sets = [np.concatenate([idx_by_seq[s] for s in rng.choice(us, len(us), replace=True)])
                 for _ in range(400)]
    print(f"\n=== Re={R}: {len(us)} sequences, {len(xg)} triplets"
          f"{' | ' + cfg['note'] if cfg['note'] else ''} ===", flush=True)

    dx = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
    Eg_all = np.asarray(spec_fn(jnp.asarray(xg))); E_gt = Eg_all.mean(0)
    Bg = band_sums(Eg_all)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    rg_all = np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                             for i in range(0, len(xg), 32)])
    A = np.load(cfg['anchor'])['spec_ref']
    print(f"  {'GROUND TRUTH (floor)':<24} resid={rg_all.mean():.1f}", flush=True)

    recon = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)

    def report(y, nm):
        Ey_all = np.asarray(spec_fn(jnp.asarray(y))); E = Ey_all.mean(0)
        By = band_sums(Ey_all)
        ry = np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                             for i in range(0, len(y), 32)])
        Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
        pl = float(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1])
        ret = float(E[HIK0:96].sum() / E_gt[HIK0:96].sum())
        lo = float(E[1:5].sum() / E_gt[1:5].sum())
        bret, bres, bpl, bband = [], [], [], []
        for m in boot_sets:
            bret.append(Ey_all[m][:, HIK0:96].sum(1).mean() / Eg_all[m][:, HIK0:96].sum(1).mean())
            bres.append(ry[m].mean() / rg_all[m].mean())
            bpl.append(np.corrcoef(Eh[m].ravel(), Ehg[m].ravel())[0, 1])
            bband.append(By[m].mean(0) / Bg[m].mean(0))
        bband = np.array(bband)
        OUT[f'{R}|{nm}'] = dict(E=E, E_gt=E_gt, band=By.mean(0) / Bg.mean(0), band_sd=bband.std(0),
                                ret=ret, ret_sd=np.std(bret), lowk=lo, place=pl,
                                place_sd=np.std(bpl), resid=ry.mean(), resid_gt=rg_all.mean(),
                                resid_ratio=ry.mean() / rg_all.mean(), resid_ratio_sd=np.std(bres),
                                kstar=eff_resolution(E, E_gt),
                                blind=float(E[10:96].sum() / A[10:96].sum()),
                                n_seq=len(us), n_trip=len(xg))
        print(f"  {nm:<24} ret={ret:.3f}+-{np.std(bret):.3f}  place={pl:.3f}+-{np.std(bpl):.3f}  "
              f"resid={ry.mean():.1f} ({ry.mean()/rg_all.mean():.2f}x GT +-{np.std(bres):.2f})  "
              f"lowk={lo:.3f}  k*={eff_resolution(E, E_gt)}", flush=True)

    report(recon, 'base recon')
    FIELDS[f'{R}|GT'] = xg[:2, ..., 1] * SIG
    FIELDS[f'{R}|base recon'] = recon[:2, ..., 1] * SIG
    # ONE sampler per regime: make_kchain_ddim_sampler returns a fresh jax.jit, so building it
    # inside the model loop would recompile the 86-step cascade for every model (4x the compile bill).
    smp = make_kchain_ddim_sampler(ddpm.unet, ab, [150, 100, 50], 86, dx, 3.0, temp=0.30)
    for nm in MODELS:
        y = batched(lambda xb, kk: smp(PARAMS[nm], sat * xb + s1t * jax.random.normal(
            jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
        report(y, nm)
        FIELDS[f'{R}|{nm}'] = y[:2, ..., 1] * SIG

    np.savez('base_results/crossregime_decomposition.npz',
             band_edges=np.array(BAND_EDGES), keys=np.array(list(OUT)),
             **{f'{k}||{f}': v for k, d in OUT.items() for f, v in d.items()})
    np.savez_compressed('base_results/crossregime_fields.npz', **FIELDS)
print("\nDECOMPOSITION COMPLETE", flush=True)
