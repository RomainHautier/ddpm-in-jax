"""AUTO-TUNED DIAL VALIDATION (user-approved 2026-08-22): the dial as a self-calibrating
procedure. Per (model, regime): evaluate the stat-match score (band energy vs the MEASURED
target statistics — no samples) on a small probe pool at lambda in {0,2,4,8,16,32}; choose
lambda* = argmin |statmatch - 1| (smallest-lambda tie-break); then grade the auto-chosen cell
on the full val pool. If auto-tuned cells grade like the best swept cells, the dial needs no
sweep in deployment: hand it statistics, it sets itself. Models: re2k-149 + pr2k-549, all
nine regimes, K3x86. Keys '{R}|{model}|AUTO' (+ '||lstar') in steering_full_grid.npz.
"""
import os, sys, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from functools import partial
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn, make_residual_loss, make_spectrum_distance
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution
from psample import pbatched

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
STARTS, STEPS = [150, 100, 50], 86
LGRID = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {
    1000: dict(gt='flow-data/kf_2d_re1000_256_40seed.npy', seqs=list(range(4, 20)), per=8,
               anchor='base_results/regime_stats_re1000_measured_train.npz'),
    **{R: dict(gt=GEN.format(R), seqs=list(range(8, 20)), per=10,
               anchor=f'base_results/regime_stats_re{R}_measured_train.npz')
       for R in (1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000)},
}
R2K = 'monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter0149.pkl'
PRK = 'monitoring/ddpo_re2000_placereward_ckpts/ddpo_re1000_iter0549.pkl'
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
MODELS = {'re2k-149': pickle.load(open(R2K, 'rb'))['params'],
          'pr2k-549': pickle.load(open(PRK, 'rb'))['params']}
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sa3, s13 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
B16 = partial(pbatched, per_dev=16)


def make_anchor_dose_dx(stats):
    lref = stats.get('log_spec_ref')
    d1 = make_spectrum_distance(stats['spec_ref'], kband=(1, 96), n=N, log_ref=lref)
    d2 = make_spectrum_distance(stats['spec_ref'], kband=(32, 96), n=N, log_ref=lref)
    def loss(x):
        return jnp.sum(0.5 * d1(x) + 3.0 * d2(x))
    return jax.jit(jax.grad(loss))


OUT = {}
OUTP = 'base_results/steering_full_grid.npz'
if os.path.exists(OUTP):
    old = np.load(OUTP, allow_pickle=True); OUT = {k: old[k] for k in old.files}

for R, c in REGIMES.items():
    if all(f'{R}|{nm}|AUTOv2||ret' in OUT for nm in MODELS):
        continue
    d = np.load(c['anchor'])
    stats = {k: d[k] for k in d.files}
    A = stats['spec_ref']
    dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    dx_dose = make_anchor_dose_dx(stats)
    xg, xl = [], []
    for s in c['seqs']:
        q = load_sequence(c['gt'], s)
        g = build_triplets(q, MEAN, SIG)
        l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, c['per']).astype(int)
        xg.append(g[i2]); xl.append(l[i2])
    xg, xl = np.concatenate(xg), np.concatenate(xl)
    resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
    E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                               for i in range(0, len(xg), 32)]).mean())
    recon = B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    probe = recon[:32]
    print(f"\n=== Re={R}: auto-tune ===", flush=True)
    for nm, P in MODELS.items():
        if f'{R}|{nm}|AUTO||ret' in OUT:
            continue
        best = (None, 1e9)
        for ls in LGRID:
            dx = dx_pde if ls == 0 else jax.jit(lambda x, _l=ls: dx_pde(x) + (_l / 3.0) * dx_dose(x))
            smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx, 3.0, temp=0.30)
            yb = np.asarray(smp(P, sa3 * jnp.asarray(probe) + s13 * jax.random.normal(
                jax.random.PRNGKey(11), probe.shape), jax.random.PRNGKey(12)))
            # v2 meter (user 2026-08-24): score the probe on [32,96) — the SAME band as the
            # graded deliverable. The old [10,96) aggregate is mid-band dominated and reads
            # 'matched' while the fine band still overshoots (the Re=1000 under-dial). The
            # DIAL'S LOSS IS UNTOUCHED: it keeps pulling over the whole band.
            sm = float(np.asarray(spec_fn(jnp.asarray(yb))).mean(0)[32:96].sum() / A[32:96].sum())
            print(f"  {nm} probe ls={ls:<4g} statmatch={sm:.3f}", flush=True)
            if abs(sm - 1) < best[1] - 1e-6:
                best = (ls, abs(sm - 1))
        lstar = best[0]
        dx = dx_pde if lstar == 0 else jax.jit(lambda x, _l=lstar: dx_pde(x) + (_l / 3.0) * dx_dose(x))
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx, 3.0, temp=0.30)
        y = B16(lambda xb, kk: smp(P, sa3 * xb + s13 * jax.random.normal(
            jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
        E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
        ry = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                                   for i in range(0, len(y), 32)]).mean())
        Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
        key = f'{R}|{nm}|AUTOv2'
        vals = dict(ret=E[HIK0:96].sum() / E_gt[HIK0:96].sum(),
                    place=np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1],
                    lowk=E[1:5].sum() / E_gt[1:5].sum(), kstar=eff_resolution(E, E_gt),
                    resid_ratio=ry / rg, lstar=float(lstar),
                    mse=np.mean((y[..., 1] - xg[..., 1]) ** 2) * SIG ** 2)
        for f, vv in vals.items():
            OUT[f'{key}||{f}'] = np.float32(vv)
        OUT[f'{key}||E'] = E.astype(np.float32)
        np.savez(OUTP, **OUT)
        print(f"  {nm} AUTO ls*={lstar:g}  ret={vals['ret']:.3f} place={vals['place']:.3f}",
              flush=True)
    jax.clear_caches()
print("\nAUTO DIAL COMPLETE", flush=True)
