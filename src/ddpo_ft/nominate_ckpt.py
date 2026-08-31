"""CHECKPOINT NOMINATION - the frozen rule, on HELD-OUT VALIDATION sequences.

Rule (unchanged from the runs that nominated 449 / 599 / 799; placement and MSE never consulted):

    ret  = E_model[32,96) / E_GT[32,96)        on the validation pool, K3 chain
    lowk = E_model[1,5)   / E_GT[1,5)
    ok       = {iter : lowk >= 0.94}           reject checkpoints that drained the large scales
    nominee  = argmin |1 - ret| over ok        closest to the correct fine-band dose

No inference-chain sweep: K3 ([150,100,50], 86 steps) only, the configuration every audit uses.

VALIDATION POOLS are disjoint from both training and the audit/test pool:
    Re=1000   train 20-27   validation 28-33   test 34-39
    Re>=1500  train 0-7     validation 8-11    test 12-19
(sequence-to-sequence correlation in these files is ~0.05, i.e. noise, so a disjoint index range
is a genuinely independent sample.)

  NOM_RE=1000 NOM_DIR=monitoring/ddpo_re1000_match_ckpts python -m src.ddpo_ft.nominate_ckpt
"""
import os, sys, glob, json, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from functools import partial
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from src.rewards import make_spectrum_fn
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from psample import pbatched

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
STARTS, STEPS = [150, 100, 50], 86          # K3 - no chain sweep
R = int(os.environ['NOM_RE'])
CKDIR = os.environ['NOM_DIR']
VAL = {1000: ('flow-data/kf_2d_re1000_256_40seed.npy', [28, 29, 30, 31, 32, 33], 20)}
for _r in (1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000):
    VAL[_r] = (f'flow-data/generated/gen_fnons_re{_r}_kf_1024to256_20seq.npy', [8, 9, 10, 11], 30)
GT, SEQS, PER = VAL[R]
if os.environ.get('NOM_SEQS'):
    SEQS = [int(x) for x in os.environ['NOM_SEQS'].split(',')]

assert jax.devices()[0].platform == 'tpu', f'ABORT: on {jax.devices()[0].platform}, not tpu'
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sa, s1 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
B16 = partial(pbatched, per_dev=16)

xg, xl = [], []
for s in SEQS:
    q = load_sequence(GT, s)
    g = build_triplets(q, MEAN, SIG); l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
    i2 = np.linspace(0, len(g) - 1, PER).astype(int)
    xg.append(g[i2]); xl.append(l[i2])
xg, xl = np.concatenate(xg), np.concatenate(xl)
E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
print(f"=== NOMINATION Re={R} | validation seqs {SEQS} | {len(xg)} triplets | K3 only ===", flush=True)
recon = np.asarray(B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
    jax.random.fold_in(kk, 1), xb.shape)), xl, 500))
dxp = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dxp, 0.0, temp=0.30)

CKS = sorted(glob.glob(f'{CKDIR}/ddpo_re1000_iter*.pkl'))
scores = {}
for p in CKS:
    it = int(p.split('iter')[1][:4])
    P = pickle.load(open(p, 'rb'))['params']
    y = np.asarray(B16(lambda xb, kk: smp(P, sa * xb + s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700))
    E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
    ret = float(E[HIK0:96].sum() / E_gt[HIK0:96].sum())
    lowk = float(E[1:5].sum() / E_gt[1:5].sum())
    scores[it] = (ret, lowk)
    print(f"  iter{it:04d}: ret={ret:.3f} lowk={lowk:.3f}"
          + ("" if lowk >= 0.94 else "   [rejected: low-k drain]"), flush=True)
    jax.clear_caches()
ok = {it: rl for it, rl in scores.items() if rl[1] >= 0.94}
nominee = min(ok or scores, key=lambda it: abs(1 - (ok or scores)[it][0]))
print(f"NOMINEE: iter{nominee:04d}  ret={scores[nominee][0]:.3f} lowk={scores[nominee][1]:.3f}"
      + ("" if ok else "   (NO checkpoint met lowk>=0.94 - fell back to the full set)"), flush=True)
out = dict(re=R, ckpt_dir=CKDIR, nominee=int(nominee), rule='argmin|1-ret| s.t. lowk>=0.94',
           band=[HIK0, 96], chain='K3[150,100,50]x86', val_seqs=SEQS, n_triplets=int(len(xg)),
           scores={str(k): dict(ret=v[0], lowk=v[1]) for k, v in sorted(scores.items())})
with open(f'{CKDIR}/nomination.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"  -> {CKDIR}/nomination.json", flush=True)
print(f"NOMINATED_CKPT={CKDIR}/ddpo_re1000_iter{nominee:04d}.pkl", flush=True)
