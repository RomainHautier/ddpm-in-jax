"""MATCHED-OBJECTIVE COMPARISON (user 2026-08-27): every earlier comparison of the fine-tune
against a steering dial confounds two things - the optimisation MECHANISM (per-step gradient at the
clean estimate vs terminal-sample policy gradient) and the OBJECTIVE (the fine-tune optimises five
weighted components against one anchor; the plain dial optimises three spectral terms against
another). This run removes the objective difference.

It reads a checkpoint's own config.json and rebuilds THE SAME Reward object the trainer used -
identical component set, identical weights, identical calibration scales, identical anchor file -
then steers the BASE model with the gradient of that reward. What remains between
`base + matched dial` and `the fine-tune, unguided` is mechanism alone.

The overall guidance strength has no counterpart in training (a policy gradient has a learning rate
instead), so it is swept rather than tuned: every lambda is reported, not the best one.

  MO_CKPT=monitoring/ddpo_re1000_newpool_ckpts MO_LAMS=1,3,8,20 python -m src.ddpo_ft.matched_objective
"""
import os, sys, json, pickle
os.chdir('/home/rhautier/ddpm-jax')
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from functools import partial
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from rewards_claude import Reward
from src.rewards import make_spectrum_fn, make_residual_loss, load_regime_stats
from src.physics_guidance import make_dx_func   # the always-on residual gradient the
                                               # ORIGINAL dials used (MO_PDE_OLD=1)
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from psample import pbatched

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
R = int(os.environ.get('MO_RE', '1000'))
# the same held-out pools every audit uses, so these rows drop straight into the existing stores
if R == 1000:
    GT = 'flow-data/kf_2d_re1000_256_40seed.npy'
    SEQS, PER = [34, 35, 36, 37, 38, 39], 20
else:
    GT = f'flow-data/generated/gen_fnons_re{R}_kf_1024to256_20seq.npy'
    # seqs 8-11 are the held-out VALIDATION pool used for checkpoint nomination, so the test pool
    # is 12-19. per=10 keeps the same linspace frames as the 8-19 audits, so these 80 triplets are
    # exactly the 12-19 subset of the stored 120 - existing rows stay comparable by masking.
    SEQS, PER = list(range(12, 20)), 10

# pool override, for choosing the guidance strength on VALIDATION instead of on the test pool
if os.environ.get('MO_SEQS'):
    SEQS = [int(x) for x in os.environ['MO_SEQS'].split(',')]
    PER = int(os.environ.get('MO_PER', PER))
    print(f"  POOL OVERRIDE: seqs {SEQS} per {PER}", flush=True)
STARTS, STEPS = [150, 100, 50], 86
# MO_STARTS='150,100' / MO_STARTS='100' etc: fewer or gentler chains. The K3 ladder re-noises and
# re-sharpens three times, which ADDS fine-scale energy each pass - the right thing for an
# under-shooting model, the wrong thing for one that already overshoots (r8kp02 carried down to
# Re=1000). NOTE rows sampled with a different chain are NOT comparable to K3 rows; tag them.
if os.environ.get('MO_STARTS'):
    STARTS = [int(x) for x in os.environ['MO_STARTS'].split(',')]
    print(f"  CHAIN OVERRIDE: STARTS={STARTS} x {STEPS} steps", flush=True)
BANDS = [(1, 5), (5, 16), (16, 32), (32, 64), (64, 96)]
CKDIR = os.environ.get('MO_CKPT', 'monitoring/ddpo_re1000_newpool_ckpts')
import glob as _glob
CKPT = os.environ.get('MO_CKPT_FILE') or sorted(_glob.glob(f'{CKDIR}/*.pkl'))[-1]
LAMS = [float(x) for x in os.environ.get('MO_LAMS', '1,3,8,20').split(',')]
MODELS = os.environ.get('MO_MODELS', 'base0,r1k-449').split(',')
TAG = os.environ.get('MO_TAG', 'mo')
if float(os.environ.get('MO_PDE_W', '1')) != 1.0:
    TAG = f"{TAG}p{os.environ['MO_PDE_W']}_"
if os.environ.get('MO_PDE_HINGE', '1') == '0':
    TAG = f'{TAG}h0_'
if os.environ.get('MO_PDE_OLD') == '1':
    TAG = f'{TAG}op_'
OUTP = 'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz'
FDIR = f'base_results/fields/re{R}'

# MO_W_CKPT: take the dial's REWARD WEIGHTS from a different checkpoint's config.json than the
# one supplying the PARAMS. The dial is "matched" to whichever objective it is built from, so a
# model trained on a different reward (e.g. the gated-dose fine-tune) would otherwise get a
# different dial and the guided rows would not be comparable. Pin all models to one objective to
# make the guided comparison like-for-like. Also required for rewards whose terms need a
# per-sample scale, which a dial cannot supply.
WCKDIR = os.environ.get('MO_W_CKPT', CKDIR)
cfg = json.load(open(f'{WCKDIR}/config.json'))
rw = cfg['reward']
W, SC = rw['weights'], rw.get('scales', {})
# ANCHOR: default to the ONE anchor every other experiment on disk was graded against
# (regime_stats_re{R}_measured_train), so this row is directly comparable to all of them
# and nothing already computed has to be re-run. MO_STATS overrides it, for the
# anchor-sensitivity row only.
STATS_PATH = os.environ.get('MO_STATS',
                            f'base_results/regime_stats_re{R}_measured_train.npz')
TRAIN_STATS = rw['stats']        # what this checkpoint was actually trained against
HIGHK = tuple(rw.get('highk_band', (32, 96)))
# MO_PDE_W scales the PDE weight in the DIAL only (training is untouched). The spectral
# terms inject energy that is spectrally right but not divergence-consistent; raising the
# residual weight is the direct counterweight.
_pdew = float(os.environ.get('MO_PDE_W', '1'))
if _pdew != 1.0:
    W = dict(W); W['pde'] = W.get('pde', 1.0) * _pdew
NAMES = tuple(k for k, v in W.items() if v)
print(f"matching the objective of {CKDIR}", flush=True)
print(f"  anchor      {STATS_PATH}", flush=True)
if os.path.abspath(STATS_PATH) != os.path.abspath(TRAIN_STATS):
    _a = np.load(STATS_PATH)["spec_ref"]; _b = np.load(TRAIN_STATS)["spec_ref"]
    _r = ", ".join(f"[{lo},{hi}) {_a[lo:hi].sum() / _b[lo:hi].sum():.3f}"
                   for lo, hi in ((16, 32), (32, 96), (64, 96)))
    print(f"  NOTE      checkpoint trained against {os.path.basename(TRAIN_STATS)}; "
          f"this anchor / that one = {_r}", flush=True)
print(f"  components  {', '.join(f'{k}={W[k]}' for k in NAMES)}", flush=True)
if _pdew != 1.0:
    print(f"  PDE WEIGHT x{_pdew}", flush=True)
print(f"  highk band  {HIGHK}", flush=True)
print(f"  scales      {', '.join(f'{k}={SC.get(k, 1.0):.4g}' for k in NAMES)}", flush=True)

stats = load_regime_stats(STATS_PATH)
# pde_hinge is TRAINING's choice: one-sided, so the residual term does not resist the
# spectral term while the policy learns. As a per-step dial that removes the only brake on
# the spectral push - and below the floor the term is exactly zero. MO_PDE_HINGE=0 makes it
# two-sided so it restrains in both directions.
_hinge = os.environ.get('MO_PDE_HINGE', '1') != '0'
# MO_TAPER=16 gives the spectrum terms a raised-cosine roll-off over their top 16 shells (full
# weight to k=80, zero at k=96), the v7 gate's edge. The training reward uses a HARD band edge and
# needs no taper - the plain fine-tune shows no artifact at k=96 - but the same reward applied as a
# per-step gradient at x-hat-0 rings against that rectangular window: a wobble at k=91-95 and a step
# at 96 (v7_bandgate.py records the extreme form, a 69x spike at k=95).
_taper = int(os.environ.get('MO_TAPER', '0'))
# MO_PDE_OLD=1 restores the physics term the ORIGINAL dials used: make_dx_func, the gradient of
# mean(residual^2), which is always active and always pushes the residual DOWN. The reward's own
# pde component is a hinged log-ratio, deliberately built so it does NOT resist the spectral term
# while the policy trains - which as a per-step dial leaves the spectral push unopposed, and is
# exactly zero while the sample sits below the floor.
_pde_old = os.environ.get('MO_PDE_OLD') == '1'
print(f"  highk_taper {_taper}   (0 = hard band edge)")
print(f"  pde_hinge   {_hinge}   old-style pde term: {_pde_old}", flush=True)
if _pde_old:
    # spectral terms only in the reward; the residual is handled by the always-on gradient below
    _names = tuple(n for n in NAMES if n != 'pde')
    _w_pde = W.get('pde', 1.0) / SC.get('pde', 1.0)      # same effective weight as the reward's
    rfn = Reward(stats, R, weights=W, scales=SC, names=_names, highk_band=HIGHK,
                 residual_ref=stats.get('residual_ref'))
    _dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    _g_spec = jax.grad(lambda x: -jnp.sum(rfn(x)[0]))
    dx_matched = jax.jit(lambda x: _g_spec(x) + _w_pde * _dx_pde(x))
    print(f"  spectral terms {_names}  +  {_w_pde:.2f} x always-on residual gradient", flush=True)
else:
    rfn = Reward(stats, R, weights=W, scales=SC, names=NAMES, highk_band=HIGHK,
                 pde_hinge=_hinge, residual_ref=stats.get('residual_ref'),
                 highk_taper=_taper)
    # the trainer maximises r = -sum w_i d_i / s_i, so the loss to descend is -r
    dx_matched = jax.jit(jax.grad(lambda x: -jnp.sum(rfn(x)[0])))

# a job that starts while the previous one is still releasing /dev/accel* falls back to CPU
# silently and then runs ~50x slower; refuse rather than waste hours
if not os.environ.get('MO_ALLOW_CPU') and jax.devices()[0].platform != 'tpu':
    raise SystemExit(f'ABORT: jax is on {jax.devices()[0].platform}, not tpu')
print(f'devices: {len(jax.devices())} x {jax.devices()[0].platform}', flush=True)

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sa3, s13 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
B16 = partial(pbatched, per_dev=16)
resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))

xg, xl = [], []
for s in SEQS:
    q = load_sequence(GT, s)
    g = build_triplets(q, MEAN, SIG); l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
    i2 = np.linspace(0, len(g) - 1, PER).astype(int)
    xg.append(g[i2]); xl.append(l[i2])
xg, xl = np.concatenate(xg), np.concatenate(xl)
E_gt_s = np.asarray(spec_fn(jnp.asarray(xg))); E_gt = E_gt_s.mean(0)
Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                           for i in range(0, len(xg), 32)]).mean())
recon = np.asarray(B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
    jax.random.fold_in(kk, 1), xb.shape)), xl, 500))

S = {k: v for k, v in np.load(OUTP, allow_pickle=True).items()} if os.path.exists(OUTP) else {}
P_FT = pickle.load(open(CKPT, 'rb'))['params']

# The sampler bakes lam in and takes the params as an ARGUMENT, so one compilation serves every
# model and every chunk at a given lam. Building it inside the chunk loop (and clearing the cache
# after each chunk) forced a fresh XLA compile of the 258-step scan for all 16 model x lam x chunk
# combinations - which was the bulk of the ~13 min/cell, not the sampling itself.
for lam in LAMS:
    smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx_matched, lam, temp=0.30)
    for m in MODELS:
        P = base_params if m == 'base0' else P_FT
        _ck = 'K3' if STARTS == [150, 100, 50] else 'K' + '-'.join(str(t) for t in STARTS)
        row = f'{m}|{_ck}|{TAG}{lam:g}'
        if f'{R}|{row}||ret' in S and not os.environ.get('MO_FORCE'):
            print(f"  {row:<26} already done", flush=True); continue
        ys = []
        for i in range(0, len(recon), 64):
            sl = slice(i, min(i + 64, len(recon)))
            xb = recon[sl]
            k0 = jax.random.PRNGKey(700)
            ys.append(np.asarray(smp(P, sa3 * jnp.asarray(xb) + s13 * jax.random.normal(
                jax.random.fold_in(k0, i), xb.shape), jax.random.fold_in(k0, i + 1))))
        y = np.concatenate(ys)
        os.makedirs(FDIR, exist_ok=True)
        np.savez_compressed(f"{FDIR}/{row.replace('|', '__')}.npz", x=y.astype(np.float16))
        E_s = np.asarray(spec_fn(jnp.asarray(y))); E = E_s.mean(0)
        ps_ret = E_s[:, 32:96].sum(1) / E_gt_s[:, 32:96].sum(1)
        Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
        ps_place = np.array([np.corrcoef(Eh[j].ravel(), Ehg[j].ravel())[0, 1] for j in range(len(y))])
        ry = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(y[j:j + 32]))).ravel()
                                   for j in range(0, len(y), 32)]).mean())
        S[f'{R}|{row}||E'] = E.astype(np.float32)
        S[f'{R}|{row}||Eb'] = np.array([E[lo:hi].sum() / E_gt[lo:hi].sum() for lo, hi in BANDS], np.float32)
        S[f'{R}|{row}||ret'] = np.float32(E[32:96].sum() / E_gt[32:96].sum())
        S[f'{R}|{row}||lowk'] = np.float32(E[1:5].sum() / E_gt[1:5].sum())
        S[f'{R}|{row}||place'] = np.float32(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1])
        S[f'{R}|{row}||resid_ratio'] = np.float32(ry / rg)
        # GT-FREE quantities, for choosing hyper-parameters without touching ground truth.
        # Measured against the ANCHOR (the assumed-known ensemble statistics) or the anchor's own
        # residual floor - never the held-out truth. Placement and MSE have no GT-free counterpart
        # and must stay out of any selection rule.
        _aref = np.asarray(stats['spec_ref'])          # the MEASURED anchor spectrum
        S[f'{R}|{row}||blind_ret'] = np.float32(E[32:96].sum() / _aref[32:96].sum())
        S[f'{R}|{row}||blind_lowk'] = np.float32(E[1:5].sum() / _aref[1:5].sum())
        S[f'{R}|{row}||resid_abs'] = np.float32(ry)
        S[f'{R}|{row}||blind_resid'] = np.float32(ry / float(stats['residual_ref']))
        S[f'{R}|{row}||mse'] = np.float32(np.mean((y[..., 1] - xg[..., 1]) ** 2) * SIG ** 2)
        S[f'{R}|{row}||ps_ret_paired'] = ps_ret.astype(np.float32)
        S[f'{R}|{row}||ps_place'] = ps_place.astype(np.float32)
        S[f'{R}|{row}||psEb'] = np.stack([E_s[:, lo:hi].sum(1) for lo, hi in BANDS], 1).astype(np.float32)
        S[f'{R}|{row}||ps_mse'] = (np.mean((y[..., 1] - xg[..., 1]) ** 2, axis=(1, 2)) * SIG ** 2).astype(np.float32)
        np.savez(OUTP, **S)
        print(f"  {row:<26} ret={float(S[f'{R}|{row}||ret']):.3f} place={np.median(ps_place):.3f} "
              f"blind[ret={float(S[f'{R}|{row}||blind_ret']):.3f} "
              f"lowk={float(S[f'{R}|{row}||blind_lowk']):.3f} "
              f"res={float(S[f'{R}|{row}||blind_resid']):.2f}] "
              f"| resid={ry / rg:.2f} mse={float(S[f'{R}|{row}||mse']):.2f} "
              f"in-band={np.mean(np.abs(ps_ret - 1) < .2) * 100:.0f}%", flush=True)
    del smp
    jax.clear_caches()          # one compiled sampler retained at a time, freed when lam changes
print("MATCHED OBJECTIVE COMPLETE", flush=True)
