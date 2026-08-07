"""Grade every checkpoint of the Re=1000 new-pool run — the calibration that de-circularises R6.

Re=1000 is the one regime whose ground truth is legitimately ours, so this is where the stopping
rule gets calibrated. Three questions:
  1. WHERE IS THE GT OPTIMUM, and what does the BLIND anchor score read there? That reading is the
     set-point the OOD runs must target. The current band [0.798, 0.858] came from the two OOD
     models' own readings — calibrated on the very models it judges. This replaces it with a
     reading from a model whose health is verified against truth.
  2. DOES THE ENLARGED POOL OVERFIT? Grade on the TRAINING sequences and on held-out val: if the
     train optimum sits far later than the val optimum, the pool is too small. (Old runs: 8
     triplets / ~4 independent states. This run: 2544 / ~48.)
  3. DOES R3.2's BLIND PICK match the GT-best checkpoint?
Deployment convention: K3[150,100,50] x86, lam3, itemp 0.30. Errors bootstrap over sequences.
Test sequences 36-39 are NOT touched here — they stay sealed for the final selection.
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
GT = 'flow-data/kf_re500_256_20seed.npy'
ANCHOR = 'base_results/regime_stats_re500_obsfit.npz'
CKDIR = 'monitoring/ddpo_re500_newpool_ckpts'
POOLS = {'train': list(range(0, 8)), 'val': [8, 9, 10, 11, 12, 13]}
PER_SEQ = 20

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sat, s1t = float(jnp.sqrt(ab[150])), float(jnp.sqrt(1.0 - ab[150]))
A = np.load(ANCHOR)['spec_ref']
dx = make_dx_func(n=N, re=500.0, std=SIG, mean=MEAN)
smp = make_kchain_ddim_sampler(ddpm.unet, ab, [150, 100, 50], 86, dx, 3.0, temp=0.30)


def batched(fn, x, seed, bs=16):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(x), bs):
        o.append(np.asarray(fn(jnp.asarray(x[i:i + bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)


DATA = {}
for tag, seqs in POOLS.items():
    xg, xl, sid = [], [], []
    for s in seqs:
        q = load_sequence(GT, s)
        g = build_triplets(q, MEAN, SIG); l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, PER_SEQ).astype(int)
        xg.append(g[i2]); xl.append(l[i2]); sid += [s] * PER_SEQ
    xg = np.concatenate(xg); xl = np.concatenate(xl); sid = np.array(sid)
    recon = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    Eg = np.asarray(spec_fn(jnp.asarray(xg)))
    DATA[tag] = dict(recon=recon, sid=sid, Eg=Eg, E_gt=Eg.mean(0),
                     Ehg=local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0))
    print(f"{tag}: seqs {seqs} x {PER_SEQ} = {len(xg)} triplets", flush=True)


def grade(params, tag):
    d = DATA[tag]
    y = batched(lambda xb, kk: smp(params, sat * xb + s1t * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), d['recon'], 700)
    Ey = np.asarray(spec_fn(jnp.asarray(y))); E = Ey.mean(0)
    Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
    us = np.unique(d['sid']); rng = np.random.default_rng(0)
    idx = {s: np.where(d['sid'] == s)[0] for s in us}
    bs = []
    for _ in range(300):
        m = np.concatenate([idx[s] for s in rng.choice(us, len(us), replace=True)])
        bs.append(Ey[m][:, HIK0:96].sum(1).mean() / d['Eg'][m][:, HIK0:96].sum(1).mean())
    return dict(ret=float(E[HIK0:96].sum() / d['E_gt'][HIK0:96].sum()), ret_sd=float(np.std(bs)),
                place=float(np.corrcoef(Eh.ravel(), d['Ehg'].ravel())[0, 1]),
                lowk=float(E[1:5].sum() / d['E_gt'][1:5].sum()),
                kstar=int(eff_resolution(E, d['E_gt'])),
                blind=float(E[10:96].sum() / A[10:96].sum()))


cks = sorted(glob.glob(f'{CKDIR}/*_iter*.pkl'))
print(f"\n{len(cks)} checkpoints + the un-finetuned base\n", flush=True)
rows = []
for label, params in ([('base', base_params)] +
                      [(_re.search(r'iter(\d+)', c).group(1), pickle.load(open(c, 'rb'))['params'])
                       for c in cks]):
    r = {t: grade(params, t) for t in POOLS}
    rows.append((label, r))
    print(f"  {label:<8} train ret={r['train']['ret']:.3f} blind={r['train']['blind']:.3f} | "
          f"val ret={r['val']['ret']:.3f}+-{r['val']['ret_sd']:.3f} place={r['val']['place']:.3f} "
          f"k*={r['val']['kstar']:>2} lowk={r['val']['lowk']:.3f}", flush=True)

ft = [(l, r) for l, r in rows if l != 'base']
best_val = min(ft, key=lambda t: abs(t[1]['val']['ret'] - 1.0))
best_trn = min(ft, key=lambda t: abs(t[1]['train']['ret'] - 1.0))
r32 = min(ft, key=lambda t: abs(t[1]['train']['blind'] - 1.0))
print(f"\n1. GT OPTIMUM on val: iter {best_val[0]} (ret {best_val[1]['val']['ret']:.3f})")
print(f"   => BLIND anchor score there = {best_val[1]['train']['blind']:.3f}   <-- THE SET-POINT")
print(f"      (the band inherited from the OOD models was [0.798, 0.858])")
print(f"2. OVERFITTING: train optimum iter {best_trn[0]} vs val optimum iter {best_val[0]} -> "
      f"{'AGREE' if best_trn[0] == best_val[0] else 'DIFFER'}")
print(f"3. R3.2 blind pick: iter {r32[0]} -> val ret {r32[1]['val']['ret']:.3f} "
      f"({'MATCHES' if r32[0] == best_val[0] else 'differs from'} the GT optimum)")
np.savez('base_results/re500_newpool_grading.npz', labels=np.array([l for l, _ in rows]),
         **{f'{l}|{t}|{k}': np.float32(v) for l, r in rows for t, m in r.items() for k, v in m.items()})
print("\nRE1000 GRADING COMPLETE", flush=True)
