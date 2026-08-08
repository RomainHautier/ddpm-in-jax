"""LADDER EXTENSION: cascades DEEPER than K4, at the two regimes whose ground truth we own.

WHY. The 84-cell sweep found Re=500's optimum at K4x110 — the DEEPEST rung available — reaching
only ret 0.924. An optimum sitting on the boundary of the search space is not an optimum, it is a
truncation: the same failure shape as freezing the cascade, one level up. If a deeper rung carries
Re=500 to ~1.0, its blind reading moves and the measured 0.159 set-point gap changes with it.
Deeper = more chains restarting from HIGHER noise levels, i.e. more generative freedom per pass.
Re=1000 is included as a control: its optimum is interior (K3), so deeper rungs should only
overshoot further — if they do not, something is wrong with the dose picture.

ORIGINAL DOCSTRING FOLLOWS.
CASCADE x CHECKPOINT sweep at the two regimes whose ground truth we own.

WHY. Every grading so far froze the inference cascade at K3[150,100,50]x86 and varied only the
checkpoint. But the cascade sets the ENERGY DOSE and the checkpoint only modulates it, so we varied
the weaker term and froze the dominant one. At Re=500 that produced a degenerate result: no
checkpoint reached retention ~1.0 (best 0.744), so "the blind score of a healthy model" — the
set-point — could not be read there at all, and the apparent Re=500-vs-Re=1000 set-point difference
was comparing a healthy model against an unhealthy one.

WHAT THIS ANSWERS.
  1. Does a HEALTHY combination (ret ~ 1.0) exist at Re=500? If yes we finally get a second valid
     set-point and can test whether it is regime-invariant. If no, that is a real limit.
  2. Which variable dominates — cascade or checkpoint? If the cascade does, checkpoint selection
     (R3.2, early stopping) matters far less than we have been assuming.
  3. What the joint blind rule should be, chosen on Re=500/1000 ONLY — never on OOD data.

GT USE. Re=500 and Re=1000 are the two regimes whose ground truth is legitimately ours; it is used
here to GRADE. The blind score is computed alongside so the two can be compared. No OOD regime is
touched.

Output: base_results/cascade_ckpt_sweep_deep.npz
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
LADDER = [('K5x140', [250, 200, 150, 100, 50], 140),
          ('K6x170', [300, 250, 200, 150, 100, 50], 170),
          ('K7x200', [350, 300, 250, 200, 150, 100, 50], 200)]

# val pools, disjoint from every training pool; test sequences stay sealed.
CASES = {
    500:  dict(gt='flow-data/kf_re500_256_20seed.npy',      val=[8, 9, 10, 11, 12, 13],
               anchor='base_results/regime_stats_re500_obsfit.npz',
               ck='monitoring/ddpo_re500_newpool_ckpts',  train=list(range(0, 8))),
    1000: dict(gt='flow-data/kf_2d_re1000_256_40seed.npy',  val=[32, 33, 34, 35],
               anchor='base_results/regime_stats_re1000_obsfit.npz',
               ck='monitoring/ddpo_re1000_newpool_ckpts', train=list(range(20, 28))),
}
PER_SEQ = 12          # val grading
PER_SEQ_TR = 8        # blind score, on the anchor's own source pool
# subset of checkpoints: the sweep is 6 cascades x N models x 2 regimes, so N is kept modest
CKPT_SUBSET = ['0049', '0149', '0249', '0349', '0449', '0549']

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))

def batched(fn, x, seed, bs=16):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(x), bs):
        o.append(np.asarray(fn(jnp.asarray(x[i:i + bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)

def pool(gt, seqs, n_per):
    xg, xl, sid = [], [], []
    for s in seqs:
        q = load_sequence(gt, s)
        g = build_triplets(q, MEAN, SIG); l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, n_per).astype(int)
        xg.append(g[i2]); xl.append(l[i2]); sid += [s] * n_per
    return np.concatenate(xg), np.concatenate(xl), np.array(sid)

OUT = {}
for R, c in CASES.items():
    A = np.load(c['anchor'])['spec_ref']
    dx = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    xg, xl, sid = pool(c['gt'], c['val'], PER_SEQ)              # GT grading pool
    _, xl_tr, _ = pool(c['gt'], c['train'], PER_SEQ_TR)         # blind-score pool (anchor's source)
    rec_v = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    rec_t = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl_tr, 500)
    Eg = np.asarray(spec_fn(jnp.asarray(xg))); E_gt = Eg.mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    us = np.unique(sid); idx = {s: np.where(sid == s)[0] for s in us}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[s] for s in rng.choice(us, len(us), replace=True)]) for _ in range(200)]
    models = [('base', base_params)] + [
        (t, pickle.load(open(p, 'rb'))['params'])
        for t in CKPT_SUBSET
        for p in glob.glob(f"{c['ck']}/*_iter{t}.pkl")]
    print(f"\n=== Re={R}: {len(LADDER)} cascades x {len(models)} models "
          f"| val seqs {c['val']} ({len(xg)} triplets) | blind on train seqs {c['train']} ===",
          flush=True)
    print(f"  {'cascade':<11} {'model':<6} {'ret':>7} {'+-':>6} {'place':>7} {'k*':>4} "
          f"{'lowk':>6} {'blind':>7}", flush=True)
    for cname, starts, steps in LADDER:
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=0.30)
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        for mname, P in models:
            y = batched(lambda xb, kk: smp(P, sa * xb + s1 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), rec_v, 700)
            Ey = np.asarray(spec_fn(jnp.asarray(y))); E = Ey.mean(0)
            Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
            bs = [Ey[m][:, HIK0:96].sum(1).mean() / Eg[m][:, HIK0:96].sum(1).mean() for m in boots]
            yt = batched(lambda xb, kk: smp(P, sa * xb + s1 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), rec_t, 700)
            Et = np.asarray(spec_fn(jnp.asarray(yt))).mean(0)
            r = dict(ret=float(E[HIK0:96].sum() / E_gt[HIK0:96].sum()), ret_sd=float(np.std(bs)),
                     place=float(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1]),
                     lowk=float(E[1:5].sum() / E_gt[1:5].sum()),
                     kstar=int(eff_resolution(E, E_gt)),
                     blind=float(Et[10:96].sum() / A[10:96].sum()))
            OUT[f'{R}|{cname}|{mname}'] = r
            print(f"  {cname:<11} {mname:<6} {r['ret']:>7.3f} {r['ret_sd']:>6.3f} "
                  f"{r['place']:>7.3f} {r['kstar']:>4} {r['lowk']:>6.3f} {r['blind']:>7.3f}",
                  flush=True)
        np.savez('base_results/cascade_ckpt_sweep_deep.npz', keys=np.array(list(OUT)),
                 **{f'{k}||{f}': np.float32(v) for k, d in OUT.items() for f, v in d.items()})
print("\nCASCADE x CHECKPOINT SWEEP COMPLETE", flush=True)
