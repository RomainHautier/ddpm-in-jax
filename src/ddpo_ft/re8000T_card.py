"""REAL-STATS SPECIALIST: nominate + card the Re=2000 fine-tune trained against MEASURED
statistics (the Track-A twin of the extrapolated-anchor-trained newpool run; identical recipe,
only the anchor differs). Two phases:

PHASE 1 (nomination, statistics only): all 12 checkpoints at K3 on the Re=2000 val pool —
pick argmin |1-ret| subject to lowk >= 0.94 (the same rule that nominated 449/599; placement
and MSE never consulted).
PHASE 2 (card): the nominee across all nine regimes x K3/K4/K5, full battery + a field frame,
into the shared npz under 'rs8kT-<iter>' — directly comparable to the re2k-599 card, isolating
what the anchor's extrapolation error cost IN TRAINING.
"""
import os, sys, glob, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from functools import partial
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn, make_residual_loss
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution
from psample import pbatched

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
CKS = sorted(glob.glob('monitoring/ddpo_re8000_rs_t35fresh_ckpts/ddpo_re1000_iter*.pkl'))
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
B16 = partial(pbatched, per_dev=16)


def pool(R):
    c = REGIMES[R]
    xg, xl = [], []
    for s in c['seqs']:
        q = load_sequence(c['gt'], s)
        g = build_triplets(q, MEAN, SIG)
        l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, c['per']).astype(int)
        xg.append(g[i2]); xl.append(l[i2])
    return np.concatenate(xg), np.concatenate(xl)


OUTP = 'base_results/multiregime_grade.npz'
FLDP = 'base_results/model_cards_fields.npz'
OUT, FLD = {}, {}
for pth, D in ((OUTP, OUT), (FLDP, FLD)):
    if os.path.exists(pth):
        old = np.load(pth, allow_pickle=True)
        D.update({k: old[k] for k in old.files})

# ---- PHASE 1: nomination at home (statistics only) ----
xg, xl = pool(8000)
E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
recon = B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
    jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
dx2k = make_dx_func(n=N, re=8000.0, std=SIG, mean=MEAN)
starts, steps = LADDER['K3x86']
smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx2k, 3.0, temp=0.30)
sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
print("=== PHASE 1: re8000 realstats checkpoints @ K3, Re=8000 val (statistics only) ===", flush=True)
scores = {}
for p in CKS:
    it = int(p.split('iter')[1][:4])
    P = pickle.load(open(p, 'rb'))['params']
    y = B16(lambda xb, kk: smp(P, sa * xb + s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
    E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
    ret = float(E[HIK0:96].sum() / E_gt[HIK0:96].sum())
    lowk = float(E[1:5].sum() / E_gt[1:5].sum())
    scores[it] = (ret, lowk)
    print(f"  iter{it:04d}: ret={ret:.3f} lowk={lowk:.3f}", flush=True)
ok = {it: rl for it, rl in scores.items() if rl[1] >= 0.94}
nominee = min(ok or scores, key=lambda it: abs(1 - (ok or scores)[it][0]))
print(f"NOMINEE (stats only): iter{nominee:04d} ret={scores[nominee][0]:.3f} "
      f"lowk={scores[nominee][1]:.3f}", flush=True)
NM = f'rs8kT-{nominee}'
P_nom = pickle.load(open([p for p in CKS if f'iter{nominee:04d}' in p][0], 'rb'))['params']

# ---- PHASE 2: card the nominee ----
for R in sorted(REGIMES):
    if all(f'{R}|{NM}|{c}||mse' in OUT for c in LADDER):
        continue
    xg, xl = pool(R)
    dx = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
    E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                               for i in range(0, len(xg), 32)]).mean())
    A = np.load(REGIMES[R]['anchor'])['spec_ref']
    recon = B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    print(f"\n=== Re={R}: carding {NM} ===", flush=True)
    for cname, (starts, steps) in LADDER.items():
        if f'{R}|{NM}|{cname}||mse' in OUT:
            continue
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=0.30)
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        y = B16(lambda xb, kk: smp(P_nom, sa * xb + s1 * jax.random.normal(
            jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
        E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
        ry = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                                   for i in range(0, len(y), 32)]).mean())
        Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
        vals = dict(ret=E[HIK0:96].sum() / E_gt[HIK0:96].sum(),
                    place=np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1],
                    lowk=E[1:5].sum() / E_gt[1:5].sum(), kstar=eff_resolution(E, E_gt),
                    resid_ratio=ry / rg, blind=E[10:96].sum() / A[10:96].sum(),
                    mse=np.mean((y[..., 1] - xg[..., 1]) ** 2) * SIG ** 2)
        for f, vv in vals.items():
            OUT[f'{R}|{NM}|{cname}||{f}'] = np.float32(vv)
        OUT[f'{R}|{NM}|{cname}||E'] = E.astype(np.float32)
        FLD[f'{R}|{NM}|{cname}'] = (y[0, ..., 1] * SIG).astype(np.float32)
        np.savez(OUTP, **OUT)
        np.savez_compressed(FLDP, **FLD)
        print(f"  {NM} @ {cname}  ret={vals['ret']:.3f}  place={vals['place']:.3f}  "
              f"lowk={vals['lowk']:.3f}  resid={vals['resid_ratio']:.2f}xGT  "
              f"mse={vals['mse']:.2f}  k*={vals['kstar']}", flush=True)
print("\nREALSTATS CARD COMPLETE", flush=True)
