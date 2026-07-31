"""Blind ladder scores on the ANCHOR'S OWN SOURCE POOL — the pool the protocol actually specifies.

Why this exists: crossregime_adapted.py scored the ladder on the TEST pool for the three regimes
that have a native model (Re=1000/2000/10000). That is the wrong pool. The blind score has a fixed
denominator (the anchor), so nothing cancels and it is pool-dependent: the Re=2000 native model
reads 0.858 on its source pool and 1.043 on the test pool — a 21% shift, larger than the whole
set-point band. Selection must therefore happen on the same pool the band was derived on, which is
the anchor's own source sequences (read from each anchor's stored fingerprint).

This recomputes ONLY the blind scores (cheap: 4 source sequences per regime). Grading against
ground truth stays where it belongs, on the disjoint test pool, and is unchanged.
"""
import os, sys, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from src.rewards import make_spectrum_fn
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence

MEAN, SIG, N = 0.0, 4.7988, 256
PER_SEQ = 15                       # matches blind_select_636.py: 4 seqs x 15 = 60 triplets
SETPOINT = (0.798, 0.858)
CFGS = {'K4[200,150,100,50]x110': ([200, 150, 100, 50], 110),
        'K3[150,100,50]x86':      ([150, 100, 50], 86),
        'K2[100,75]x50':          ([100, 75], 50),
        'K1[100]x20':             ([100], 20),
        'K1[75]x20':              ([75], 20),
        'K1[50]x12':              ([50], 12)}
MODELS = {
    'in-dist Re=1000': 'monitoring/ddpo_re1000_k2_s100-75_ddim50_ddiminit_hk10_emabase_ckpts/ddpo_re1000_iter0299.pkl',
    'Re=2000 model':   'monitoring/ddpo_re2000_dt32_ckpts/ddpo_re1000_iter0599.pkl',
    'Re=10000 model':  'monitoring/ddpo_re10000_dt32_ckpts/ddpo_re1000_iter0549.pkl',
}
NATIVE = {2000: 'Re=2000 model', 10000: 'Re=10000 model'}
# Re=1000 is EXCLUDED on purpose: its anchor is the legacy measured-stats one, built from ground
# truth with no observation fingerprint. It is a different instrument and the band never applied.
CASES = {2000:  'base_results/regime_stats_re2000_obsfit_v3.npz',
         10000: 'base_results/regime_stats_re10000_obsfit_dt32.npz'}

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
PARAMS = {k: pickle.load(open(v, 'rb'))['params'] for k, v in MODELS.items()}


def batched(fn, x, seed, bs=16):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(x), bs):
        o.append(np.asarray(fn(jnp.asarray(x[i:i + bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)


OUT = {}
for R, apath in CASES.items():
    d = np.load(apath)
    src = d['obs_source'].item().decode()
    path, seqs = src.split('|seqs=')
    seqs = eval(seqs)
    A = d['spec_ref']
    print(f"\n=== Re={R}: anchor source pool = {path} seqs {seqs} ===", flush=True)
    xl = []
    for s in seqs:
        l = build_triplets(grid_downsample_degrade(load_sequence(path, s), 4), MEAN, SIG)
        xl.append(l[np.linspace(0, len(l) - 1, min(PER_SEQ, len(l))).astype(int)])
    xl = np.concatenate(xl)
    print(f"  {len(xl)} triplets from the anchor's own sequences", flush=True)
    dx = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    recon = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    b0 = float(np.asarray(spec_fn(jnp.asarray(recon))).mean(0)[10:96].sum() / A[10:96].sum())
    print(f"  {'base recon (no finetuning)':<34} blind={b0:.3f}"
          f"{'  <- ALREADY IN BAND' if SETPOINT[0] <= b0 <= SETPOINT[1] else ''}", flush=True)
    for cname, (starts, steps) in CFGS.items():
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=0.30)
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        for nm in MODELS:
            if nm == NATIVE[R]:
                continue
            y = batched(lambda xb, kk: smp(PARAMS[nm], sa * xb + s1 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
            b = float(np.asarray(spec_fn(jnp.asarray(y))).mean(0)[10:96].sum() / A[10:96].sum())
            OUT[f'{R}|{nm}|{cname}'] = b
            print(f"  {nm + ' @ ' + cname:<46} blind={b:.3f}"
                  f"{'  <- IN BAND' if SETPOINT[0] <= b <= SETPOINT[1] else ''}", flush=True)
    for nm in MODELS:
        if nm == NATIVE[R]:
            continue
        rung = [(c, OUT[f'{R}|{nm}|{c}']) for c in CFGS if f'{R}|{nm}|{c}' in OUT]
        best = min(rung, key=lambda t: abs(t[1] - np.clip(t[1], *SETPOINT)))
        print(f"  => BLIND PICK (source pool) {nm} @ Re={R}: {best[0]} (blind {best[1]:.3f})",
              flush=True)
np.savez('base_results/blind_picks_source_pool.npz',
         **{k: np.float32(v) for k, v in OUT.items()})
print("\nSOURCE-POOL BLIND SCORES COMPLETE", flush=True)
