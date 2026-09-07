"""Grade the pure-gate Re=1000 checkpoint ladder on the CANONICAL validation split.

Unguided production sampler (K3 [150,100,50], 86 steps, temp 0.30, lam 0), EMA params,
validation sequences 28-33 x 20 triplets. Per checkpoint: mean fine-band retention,
per-triplet in-band share and slope, and the low/mid band retentions - the columns the
pure-gate reward is supposed to win. Test sequences 34-39 are NOT touched; the selected
checkpoint gets ONE test evaluation through the audit machinery afterwards.

  python src/ddpo_ft/grade_puregate_re1k.py            # TPU
"""
import os, sys, glob, pickle, re as _re
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
RE = int(os.environ.get('GRADE_RE', '1000'))
if RE == 1000:
    GT = 'flow-data/kf_2d_re1000_256_40seed.npy'
    VAL_SEQS, PER_SEQ = [28, 29, 30, 31, 32, 33], 20
else:
    # generated regimes: canonical validation pool (nomination) is seqs 8-11
    GT = f'flow-data/generated/gen_fnons_re{RE}_kf_1024to256_20seq.npy'
    VAL_SEQS, PER_SEQ = [8, 9, 10, 11], 10
CKDIR = os.environ.get('GRADE_CKDIR', f'monitoring/ddpo_re{RE}_puregate_ckpts')
BEST_OUT = os.environ.get('GRADE_BEST_OUT')
ITERS = [int(x) for x in os.environ.get('GRADE_ITERS',
         '59,119,179,239,299,359,419,479,539,599').split(',')]

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sat, s1t = float(jnp.sqrt(ab[150])), float(jnp.sqrt(1.0 - ab[150]))
dx = make_dx_func(n=N, re=float(RE), std=SIG, mean=MEAN)
smp = make_kchain_ddim_sampler(ddpm.unet, ab, [150, 100, 50], 86, dx, 0.0, temp=0.30)


def batched(fn, x, seed, bs=16):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(x), bs):
        o.append(np.asarray(fn(jnp.asarray(x[i:i + bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)


xg, xl = [], []
for s in VAL_SEQS:
    q = load_sequence(GT, s)
    g = build_triplets(q, MEAN, SIG); l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
    i2 = np.linspace(0, len(g) - 1, PER_SEQ).astype(int)
    xg.append(g[i2]); xl.append(l[i2])
xg = np.concatenate(xg); xl = np.concatenate(xl)
recon = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
    jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
Eg = np.asarray(spec_fn(jnp.asarray(xg)))
print(f"val pool: seqs {VAL_SEQS} x {PER_SEQ} = {len(xg)} triplets", flush=True)

BANDS = [(1, 7), (7, 16), (16, 32), (32, 96)]
BEST = (None, -1.0, 9e9)
print('iter '.ljust(6) + ''.join(f'{f"[{a},{b})":>8}' for a, b in BANDS)
      + '   in-band  slope', flush=True)
for it in ITERS:
    p = f'{CKDIR}/ddpo_re1000_iter{it:04d}.pkl'
    if not os.path.exists(p):
        print(f'{it:<6} MISSING'); continue
    ck = pickle.load(open(p, 'rb'))
    params = ck.get('ema_params') or ck['params']
    y = batched(lambda xb, kk: smp(params, sat * xb + s1t * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
    Ey = np.asarray(spec_fn(jnp.asarray(y)))
    d = Ey[:, 32:96].sum(1); g = Eg[:, 32:96].sum(1); r = d / g
    inb = np.mean(np.abs(r - 1) < 0.2) * 100
    sl = np.polyfit(np.log(g), np.log(d), 1)[0]
    b = ''.join(f'{Ey[:, a:b_].sum() / Eg[:, a:b_].sum():8.3f}' for a, b_ in BANDS)
    print(f'{it:<6}' + b + f'   {inb:5.0f}%  {sl:5.2f}', flush=True)
    key = (round(inb), -abs(Ey[:, 32:96].sum() / Eg[:, 32:96].sum() - 1))
    if BEST[0] is None or key > (BEST[1], -BEST[2]):
        BEST = (it, round(inb), abs(Ey[:, 32:96].sum() / Eg[:, 32:96].sum() - 1))
print(f'BEST ITER {BEST[0]} (in-band {BEST[1]}%, |ret-1| {BEST[2]:.3f})', flush=True)
if BEST_OUT: open(BEST_OUT, 'w').write(str(BEST[0]))
print('GRADE DONE', flush=True)
