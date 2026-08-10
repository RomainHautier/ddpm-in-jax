"""Can placement be read WITHOUT ground truth? Test of the base-reconstruction proxy.

MOTIVATION. The 126-cell sweep showed retention alone cannot define a healthy model (12 Re=500
cells within 10% of ret=1.0 span true placement 0.869 down to 0.575, one with k*=1). But true
placement needs GT, so it can never be part of the OOD deployment rule. Candidate GT-free proxy:

    proxy = corr( model's local hi-k energy map , BASE RECONSTRUCTION's local hi-k energy map )

The base recon is available at every regime from LR alone, and its own map correlates with truth
0.91 (Re=1000) to 0.70 (Re=10000). A model that has redistributed fine structure to the wrong
places should decorrelate from the base's hedge just as it decorrelates from truth.

DESIGN. 42 cells re-run: at each GT regime, three rungs spanning under/at/over the optimum, all 7
models, val pool only, SAME seeds as the sweep (recon 500, sampler 700) so the fields are identical
to what the sweep graded. For each cell compute true placement (corr with GT map — validation only)
and the proxy (corr with base-recon map — GT-free). The deliverable is the proxy-vs-truth
correlation across cells; per-regime and pooled.

GT ROLE: grading the proxy, never feeding it. Deployment would use the proxy alone.
Output: base_results/placement_proxy_test.npz
"""
import os, sys, glob, pickle, re as _re
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
RUNGS = {
    500:  [('K3x86', [150, 100, 50], 86), ('K5x140', [250, 200, 150, 100, 50], 140),
           ('K7x200', [350, 300, 250, 200, 150, 100, 50], 200)],
    1000: [('K1-100x20', [100], 20), ('K3x86', [150, 100, 50], 86),
           ('K5x140', [250, 200, 150, 100, 50], 140)],
}
CASES = {
    500:  dict(gt='flow-data/kf_re500_256_20seed.npy', val=[8, 9, 10, 11, 12, 13],
               ck='monitoring/ddpo_re500_newpool_ckpts'),
    1000: dict(gt='flow-data/kf_2d_re1000_256_40seed.npy', val=[32, 33, 34, 35],
               ck='monitoring/ddpo_re1000_newpool_ckpts'),
}
PER_SEQ = 12
CKPT_SUBSET = ['0049', '0149', '0249', '0349', '0449', '0549']

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))


def batched(fn, x, seed, bs=16):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(x), bs):
        o.append(np.asarray(fn(jnp.asarray(x[i:i + bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)


OUT = {}
for R, c in CASES.items():
    xg, xl = [], []
    for s in c['val']:
        q = load_sequence(c['gt'], s)
        g = build_triplets(q, MEAN, SIG); l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, PER_SEQ).astype(int)
        xg.append(g[i2]); xl.append(l[i2])
    xg = np.concatenate(xg); xl = np.concatenate(xl)
    recon = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    Eh_gt = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)        # GT map — VALIDATION ONLY
    Eh_rc = local_hik_energy(recon[..., 1] * SIG, HIK0, 6.0)     # base-recon map — GT-FREE
    ref = float(np.corrcoef(Eh_rc.ravel(), Eh_gt.ravel())[0, 1])
    print(f"\n=== Re={R}: base-recon map vs truth corr = {ref:.3f} "
          f"(the proxy's own ceiling) ===", flush=True)
    models = [('base', base_params)] + [
        (t, pickle.load(open(p, 'rb'))['params'])
        for t in CKPT_SUBSET for p in glob.glob(f"{c['ck']}/*_iter{t}.pkl")]
    dx = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    print(f"  {'cascade':<11} {'model':<6} {'true place':>11} {'proxy':>8}", flush=True)
    for cname, starts, steps in RUNGS[R]:
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=0.30)
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        for mname, P in models:
            y = batched(lambda xb, kk: smp(P, sa * xb + s1 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
            Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
            tp = float(np.corrcoef(Eh.ravel(), Eh_gt.ravel())[0, 1])
            px = float(np.corrcoef(Eh.ravel(), Eh_rc.ravel())[0, 1])
            OUT[f'{R}|{cname}|{mname}'] = dict(true_place=tp, proxy=px)
            print(f"  {cname:<11} {mname:<6} {tp:>11.3f} {px:>8.3f}", flush=True)
    np.savez('base_results/placement_proxy_test.npz', keys=np.array(list(OUT)),
             **{f'{k}||{f}': np.float32(v) for k, d in OUT.items() for f, v in d.items()})

print("\n=== VERDICT ===", flush=True)
for R in CASES:
    cs = [k for k in OUT if k.startswith(f'{R}|')]
    t = np.array([OUT[k]['true_place'] for k in cs]); p = np.array([OUT[k]['proxy'] for k in cs])
    print(f"  Re={R}: proxy-vs-truth r = {np.corrcoef(p, t)[0, 1]:+.3f} over {len(cs)} cells "
          f"(true placement range {t.min():.3f}-{t.max():.3f})", flush=True)
allc = list(OUT)
t = np.array([OUT[k]['true_place'] for k in allc]); p = np.array([OUT[k]['proxy'] for k in allc])
print(f"  POOLED: r = {np.corrcoef(p, t)[0, 1]:+.3f} over {len(allc)} cells", flush=True)
print("PROXY TEST COMPLETE", flush=True)
