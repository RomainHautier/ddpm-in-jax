"""ACROSS-TRIPLET CONSISTENCY (user 2026-08-27): reconstructing a full sequence with a sliding
window means each interior frame is reconstructed THREE times - once as the leading frame of a
triplet, once as the middle, once as the trailing. Do those three estimates agree, and does
fine-tuning or sampling guidance change how much they agree?

The audit pools subsampled triplets (linspace), so no frame appears twice there; this run uses
CONSECUTIVE stride-1 triplets over a contiguous stretch of held-out sequence, which is what a real
sequence reconstruction would do.

Frame i of a sequence appears in triplet i-2 at channel 2, triplet i-1 at channel 1, triplet i at
channel 0. Comparing those three isolates POSITION. Because sampling is stochastic they will differ
even in principle, so every configuration is also run a second time with a different sampling key:
that gives the noise floor (same triplet, same position, different noise) against which the
positional disagreement is measured. A ratio near 1 means position does not matter; above 1 means
the model treats a frame differently depending on where it sits in the window.

The dose dial reads the MIDDLE frame's spectrum only (make_spectrum_fn indexes x[..., 1]) and
applies the resulting gradient to all three channels, so guidance is expected to raise the ratio.

Writes base_results/overlap_consistency.npz and fields/overlap/ for offline analysis.
"""
import os, sys, pickle
os.chdir('/home/rhautier/ddpm-jax')
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from functools import partial
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from src.rewards import make_spectrum_fn, make_spectrum_distance
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from psample import pbatched

MEAN, SIG, N, HIK0, R = 0.0, 4.7988, 256, 32, 1000
GT = 'flow-data/kf_2d_re1000_256_40seed.npy'
SEQS = [int(x) for x in os.environ.get('OC_SEQS', '34,35').split(',')]
NFR = int(os.environ.get('OC_NFRAMES', '26'))          # consecutive frames -> NFR-2 triplets each
STARTS, STEPS = [150, 100, 50], 86
MODELS = os.environ.get('OC_MODELS', 'base0,r1k-449').split(',')
STRATS = os.environ.get('OC_STRATS', 'none,rewardv2,v6gate').split(',')
KEYS = [700, 911]                                      # second key = the stochastic noise floor
CKPT = {'r1k-449': 'monitoring/ddpo_re1000_newpool_ckpts/ddpo_re1000_iter0449.pkl',
        're2k-149': 'monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter0149.pkl'}

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sa3, s13 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
B16 = partial(pbatched, per_dev=16)
stats = np.load(f'base_results/regime_stats_re{R}_measured_train.npz')
ref, lref = stats['spec_ref'], stats['log_spec_ref']
dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)

# --- contiguous stride-1 triplets, so consecutive windows overlap by two frames ---------------
xg, xl, owner = [], [], []
for s in SEQS:
    q = load_sequence(GT, s)[:NFR]
    xg.append(build_triplets(q, MEAN, SIG))
    xl.append(build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG))
    owner.append(np.full(len(xg[-1]), s))
xg, xl, owner = np.concatenate(xg), np.concatenate(xl), np.concatenate(owner)
print(f"{len(xg)} consecutive triplets from sequences {SEQS} ({NFR} frames each)", flush=True)

recon = np.asarray(B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
    jax.random.fold_in(kk, 1), xb.shape)), xl, 500))

# per-sample scale for the gate: recon fine-band energy over a fixed calibration constant (GT-free)
obs = np.asarray(spec_fn(jnp.asarray(recon)))[:, 32:96].sum(1)
C_R = float(obs.mean())
SCALE = (obs / C_R).astype(np.float32)

d1 = make_spectrum_distance(ref, kband=(1, 96), n=N, log_ref=lref)
d2 = make_spectrum_distance(ref, kband=(32, 96), n=N, log_ref=lref)
dm = make_spectrum_distance(ref, kband=(16, 32), n=N, log_ref=lref)
dose2 = jax.jit(jax.grad(lambda x: jnp.sum(0.5 * d1(x) + 3.0 * dm(x) + 3.0 * d2(x))))

FDIR = 'base_results/fields/overlap'; os.makedirs(FDIR, exist_ok=True)
OUTP = 'base_results/overlap_consistency.npz'
OUT = {k: v for k, v in np.load(OUTP, allow_pickle=True).items()} if os.path.exists(OUTP) else {}
if 'GT' not in OUT:
    np.savez_compressed(f'{FDIR}/GT.npz', x=xg.astype(np.float16))
    np.savez(f'{FDIR}/index.npz', seqs=np.array(SEQS), nframes=NFR, owner=owner,
             note='CONSECUTIVE stride-1 triplets; triplet j holds frames j, j+1, j+2 of its sequence')
    OUT['GT'] = np.float32(1)


def gated_dx(sl):
    """taper + per-sample scaling + target gate, exactly as v6_gated builds it"""
    sc = jnp.asarray(SCALE[sl])
    g1 = make_spectrum_distance(ref, kband=(1, 96), n=N, log_ref=lref, per_sample_scale=sc)
    gm = make_spectrum_distance(ref, kband=(16, 32), n=N, log_ref=lref, per_sample_scale=sc)
    _w = np.ones(64, np.float32); _w[48:] = 0.5 * (1 + np.cos(np.pi * np.arange(16) / 16))
    _wj, _lr32, _sh = jnp.asarray(_w), jnp.asarray(lref[32:96], jnp.float32), jnp.log(sc)[:, None]

    def gt_(x, _w=_wj, _l=_lr32, _s=_sh):
        e = jnp.maximum(spec_fn(x)[..., 32:96], jnp.exp(_l) * 1e-6)
        return jnp.sum(_w * (jnp.log(e) - (_l[None, :] + _s)) ** 2, axis=-1) / jnp.sum(_w)

    dose = jax.jit(jax.grad(lambda x: jnp.sum(0.5 * g1(x) + 3.0 * gm(x) + 3.0 * gt_(x))))
    tgt = jnp.asarray(ref[32:96].sum()) * sc

    def dx(x, _d=dose, _t=tgt):
        e = spec_fn(x)[..., 32:96].sum(-1)
        gate = jnp.minimum(1.0, jnp.abs(jnp.log(jnp.maximum(e, 1e-8 * _t) / _t)) / 0.2)
        return dx_pde(x) + (8.0 / 3.0) * gate[:, None, None, None] * _d(x)
    return jax.jit(dx)


for m in MODELS:
    P = base_params if m == 'base0' else pickle.load(open(CKPT[m], 'rb'))['params']
    for sg in STRATS:
        for key in KEYS:
            row = f'{m}|{sg}|k{key}'
            fpath = f"{FDIR}/{row.replace('|', '__')}.npz"
            if os.path.exists(fpath) and not os.environ.get('OC_FORCE'):
                print(f"  {row:<26} already done", flush=True); continue
            ys = []
            for i in range(0, len(recon), 64):
                sl = slice(i, min(i + 64, len(recon)))
                if sg == 'none':
                    dx, lam = dx_pde, 0.0
                elif sg == 'rewardv2':
                    dx, lam = jax.jit(lambda x: (8.0 / 3.0) * dose2(x)), 3.0
                elif sg == 'v6gate':
                    dx, lam = gated_dx(sl), 3.0
                else:
                    raise SystemExit(f'unknown strategy {sg}')
                smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx, lam, temp=0.30)
                xb = recon[sl]
                k0 = jax.random.PRNGKey(key)
                ys.append(np.asarray(smp(P, sa3 * jnp.asarray(xb) + s13 * jax.random.normal(
                    jax.random.fold_in(k0, i), xb.shape), jax.random.fold_in(k0, i + 1))))
                jax.clear_caches()
            y = np.concatenate(ys)
            np.savez_compressed(fpath, x=y.astype(np.float16))
            mse = float(np.mean((y[..., 1] - xg[..., 1]) ** 2) * SIG ** 2)
            OUT[f'{row}||mse'] = np.float32(mse)
            np.savez(OUTP, **OUT)
            print(f"  {row:<26} done   mid-frame MSE {mse:.3f}", flush=True)
    jax.clear_caches()
print("OVERLAP CONSISTENCY COMPLETE", flush=True)
