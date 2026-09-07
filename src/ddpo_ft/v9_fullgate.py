"""PER-BAND TARGET GATE (v7, 2026-08-27): v6 gates on the AGGREGATE [32,96) energy, so once the
fine band as a whole arrives the guidance shuts off for every term - leaving the mid band short
(0.91 vs the v2 dial's 0.97 at Re=1000). Here each band gets its OWN gate: the mid-band term
stops when [16,32) arrives, the hi-k term when [32,96) does. ORIGINAL DOC: TARGET-GATED (v6, 2026-08-26): the in-regime blow-up mechanism is OVERSHOOT WITHIN
THE CHAIN - samples that reach their target early keep receiving full guidance pressure for the
remaining steps, which accumulates into a narrow spectral spike. Fix: gate each sample's dose
gradient by how far it still is from its own target: gate_i = min(1, |log(E_i/target_i)| / 0.2)
- full strength beyond 20% error, fading linearly to zero AT the target, two-sided. On top of
taper + per-sample scaling. ORIGINAL DOC: COMBINED v5. The isolated runs showed the
two refinements are complementary: per-sample scaling doubles per-frame dose accuracy (38%->78%
in-band) but raises the dial's pressure, which the HARD band edge turns into a 69x spike at k=95
(residual 20x, placement 0.45); the taper removes that mechanism (peak ratio 1.1) but leaves
per-frame dose untouched. This runs them together. ORIGINAL DOC: THE SCALED DIAL (v4, user 2026-08-26): the dose dial matches MARGINAL statistics, so per-frame
dose scatters (38% of frames within +-20% of their own GT fine-band energy, vs 78% for the plain
fine-tune). Fix: scale the anchor PER SAMPLE by that frame's predicted level,
    s_i = (base recon's fine-band energy of sample i) / C_R,
where C_R is a fixed per-regime calibration constant = the mean recon fine-band energy over a pool
(NO ground truth anywhere; the recon is a deterministic function of the LR input). The recon's
fine-band energy tracks the GT's at log-r 0.92-0.98 at every regime; the ratio cancels its
systematic under-estimate. A fixed constant (not the batch mean) is essential: batch-mean
normalization degrades the predictor from 93% to 78% within +-20%.
Graded on the metric that matters: PER-SAMPLE in-band rate, alongside the ensemble numbers.
Cells: Re=1000 (base0, r1k-449; held-out seqs 34-39) and Re=5000/8000 (re2k-149, st8k-599).
Keys '{R}|{m}|K3|v9fullgate'."""
import os, sys, pickle
os.chdir('/home/rhautier/ddpm-jax')
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
from psample import pbatched

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
STARTS, STEPS = [150, 100, 50], 86
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
_ALL = {1000: ('flow-data/kf_2d_re1000_256_40seed.npy', [34, 35, 36, 37, 38, 39], 20,
               'base_results/re1000_audit.npz'),
        **{R: (GEN.format(R), list(range(8, 20)), 10, f'base_results/regime_audit_re{R}.npz')
           for R in (1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000)}}
DB = float(os.environ.get('GATE_DEADBAND', '0.2'))
TAG0 = 'v9fullgate'
_REGS = [int(r) for r in os.environ.get('GATE_REGIMES', '1000,5000,8000').split(',')]
_MODELS = os.environ.get('GATE_MODELS', 'base0,r1k-449').split(',')
# GATE_OOD_SEQS/GATE_OOD_PER repin the R>=1500 test pool. The default here (seqs 8-19) predates the
# validation split: seqs 8-11 are now the held-out pool used for checkpoint nomination, and
# matched_objective tests on 12-19 only. Any row meant to sit beside a matched_objective row must
# be run with GATE_OOD_SEQS=12,...,19 GATE_OOD_PER=10 or the two are on different ground truth.
if os.environ.get('GATE_OOD_SEQS'):
    _os = [int(x) for x in os.environ['GATE_OOD_SEQS'].split(',')]
    _op = int(os.environ.get('GATE_OOD_PER', '10'))
    for _r in list(_ALL):
        if _r >= 1500:
            _g, _, _, _st = _ALL[_r]; _ALL[_r] = (_g, _os, _op, _st)
    print(f"  OOD POOL OVERRIDE: seqs {_os} per {_op}", flush=True)
CFG = {R: (*_ALL[R], tuple(_MODELS)) for R in _REGS}
CKPT = {'r1k-449': 'monitoring/ddpo_re1000_newpool_ckpts/ddpo_re1000_iter0449.pkl',
        're2k-149': 'monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter0149.pkl',
        'rs8kkl-799': 'monitoring/ddpo_re8000_rs_kl3_ckpts/ddpo_re1000_iter0799.pkl',
        'st8k-599': 'monitoring/ddpo_re8000_steeredtrain_ckpts/ddpo_re1000_iter0599.pkl',
        'pr2k-549': 'monitoring/ddpo_re2000_placereward_ckpts/ddpo_re1000_iter0549.pkl'}
# GATE_CKPTS='name:path,...' registers checkpoints trained after this script was written
# (e.g. the matched set) so the ORIGINAL dials can be re-run against them.
for _s in filter(None, os.environ.get('GATE_CKPTS', '').split(',')):
    _p = _s.split(':'); CKPT[_p[0]] = _p[1]
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sa3, s13 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
B16 = partial(pbatched, per_dev=16)

for R, (gt_path, seqs, per, store_path, models) in CFG.items():
    S = {k: v for k, v in np.load(store_path, allow_pickle=True).items()}
    if all(f'{R}|{m}|K3|{TAG0}||ret' in S for m in models) and not os.environ.get('GATE_FORCE'):
        continue
    d = np.load(f'base_results/regime_stats_re{R}_measured_train.npz')
    ref, lref = d['spec_ref'], d.get('log_spec_ref')
    dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
    xg, xl = [], []
    for s in seqs:
        q = load_sequence(gt_path, s)
        g = build_triplets(q, MEAN, SIG); l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, per).astype(int); xg.append(g[i2]); xl.append(l[i2])
    xg, xl = np.concatenate(xg), np.concatenate(xl)
    E_gt_s = np.asarray(spec_fn(jnp.asarray(xg)))                      # per-sample GT spectra
    E_gt = E_gt_s.mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i+32]))).ravel() for i in range(0, len(xg), 32)]).mean())
    recon = np.asarray(B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500))
    # --- the conditional scale: recon fine-band energy / fixed calibration constant (GT-free) ---
    E_rc = np.asarray(spec_fn(jnp.asarray(recon)))
    obs = E_rc[:, 32:96].sum(1)
    C_R = float(obs.mean())                       # calibration constant for this regime
    # dispersion (beta) by VARIANCE MATCHING - Track-A legal: needs the flow's per-sample spectral
    # spread (an ensemble statistic) and our own recons, never paired samples. In-regime beta~1;
    # OOD the recon over-disperses (beta 0.77 at Re=8000) so the plain ratio over-corrects.
    E_gt_ps = E_gt_s[:, 32:96].sum(1)             # stands in for the assumed spread statistic
    beta = float(np.std(np.log(E_gt_ps)) / np.std(np.log(obs)))
    SCALES = {TAG0: (obs / C_R).astype(np.float32)}
    # [1,7) anchor: each sample's own recon large scales are the target (recon preserves
    # them at ~0.99), tighter deadband because the losses chased are 2-5%.
    LREF_LL = np.log(np.maximum(E_rc[:, 1:7], 1e-20)).astype(np.float32)
    T_LL = E_rc[:, 1:7].sum(1).astype(np.float32)
    _sc = SCALES[TAG0]
    print(f"\n=== Re={R}: C_R={C_R:.4e}, scale p10/50/90 = "
          f"{np.percentile(_sc,10):.2f}/{np.percentile(_sc,50):.2f}/{np.percentile(_sc,90):.2f} ===", flush=True)
    tgt_s = E_gt_s[:, 32:96].sum(1)
    for _t, _s in SCALES.items():
        _p = E_gt[32:96].sum() * _s
        print(f"    predictor check [{_t}]: median |err| {np.median(np.abs(_p-tgt_s)/tgt_s)*100:.1f}%, "
              f"within20% {np.mean(np.abs(_p/tgt_s-1)<0.2)*100:.0f}%", flush=True)
    for m in models:
        for tag, scale in SCALES.items():
            key = f'{R}|{m}|K3|{tag}'
            if f'{key}||ret' in S and not os.environ.get('GATE_FORCE'):
                continue
            P = base_params if m == 'base0' else pickle.load(open(CKPT[m], 'rb'))['params']
            ys = []
            for i in range(0, len(recon), 64):
                sl = slice(i, min(i + 64, len(recon)))
                sc = jnp.asarray(scale[sl])
                d1 = make_spectrum_distance(ref, kband=(1, 96), n=N, log_ref=lref, per_sample_scale=sc)
                # TAPERED hi-k term (full weight to k=80, cosine roll-off to zero AT 96; the
                # reference is never consulted past 96) with the per-sample log shift applied.
                _w = np.ones(64, np.float32); _w[48:] = 0.5 * (1 + np.cos(np.pi * np.arange(16) / 16))
                _wj = jnp.asarray(_w)
                _lr32 = jnp.asarray((lref if lref is not None else np.log(ref + 1e-20))[32:96], jnp.float32)
                _ls_shift = jnp.log(sc)[:, None]
                def d2(x, _w=_wj, _l=_lr32, _sh=_ls_shift):
                    e = jnp.maximum(spec_fn(x)[..., 32:96], jnp.exp(_l) * 1e-6)
                    return jnp.sum(_w * (jnp.log(e) - (_l[None, :] + _sh)) ** 2, axis=-1) / jnp.sum(_w)
                dm = make_spectrum_distance(ref, kband=(16, 32), n=N, log_ref=lref, per_sample_scale=sc)
                # LOW-BAND term (v8): the deficit starts near k=7, below the reach of the mid and
                # hi terms. Push k in [7,16) with a cosine ramp-in over the first three shells so
                # the lower edge is tapered, gated per sample like the other bands.
                _wl = np.ones(9, np.float32)
                _wl[:3] = 0.5 * (1 - np.cos(np.pi * (np.arange(3) + 1) / 3))
                _wlj = jnp.asarray(_wl)
                _lr7 = jnp.asarray((lref if lref is not None else np.log(ref + 1e-20))[7:16], jnp.float32)
                def dl(x, _w=_wlj, _l=_lr7, _sh=_ls_shift):
                    e = jnp.maximum(spec_fn(x)[..., 7:16], jnp.exp(_l) * 1e-6)
                    return jnp.sum(_w * (jnp.log(e) - (_l[None, :] + _sh)) ** 2, axis=-1) / jnp.sum(_w)
                _lll = jnp.asarray(LREF_LL[sl])
                def dll(x, _l=_lll):
                    e = jnp.maximum(spec_fn(x)[..., 1:7], jnp.exp(_l) * 1e-6)
                    return jnp.mean((jnp.log(e) - _l) ** 2, axis=-1)
                dose = jax.jit(jax.grad(lambda x: jnp.sum(0.5 * d1(x) + 3.0 * dm(x) + 3.0 * d2(x))))
                # target gate: per-sample fine-band energy of the CURRENT clean estimate vs its
                # own scaled target; strength fades to zero at the target (deadband 20%)
                # PER-BAND gates: each term keeps its own deadband, so a band that has arrived
                # stops being pushed while the others continue.
                _t_mid = jnp.asarray(ref[16:32].sum()) * sc
                _t_hi = jnp.asarray(ref[32:96].sum()) * sc
                _t_low = jnp.asarray(ref[7:16].sum()) * sc
                _g_bb = jax.jit(jax.grad(lambda x: jnp.sum(0.5 * d1(x))))
                _g_mid = jax.jit(jax.grad(lambda x: jnp.sum(3.0 * dm(x))))
                _g_hi = jax.jit(jax.grad(lambda x: jnp.sum(3.0 * d2(x))))
                _g_low = jax.jit(jax.grad(lambda x: jnp.sum(2.0 * dl(x))))
                _g_ll = jax.jit(jax.grad(lambda x: jnp.sum(2.0 * dll(x))))
                _t_ll = jnp.asarray(T_LL[sl])
                def _gate(e, t):
                    return jnp.minimum(1.0, jnp.abs(jnp.log(jnp.maximum(e, 1e-8 * t) / t)) / DB)[:, None, None, None]
                def _gate005(e, t):
                    return jnp.minimum(1.0, jnp.abs(jnp.log(jnp.maximum(e, 1e-8 * t) / t)) / 0.05)[:, None, None, None]
                def dx(x, _tm=_t_mid, _th=_t_hi, _tl=_t_low, _tll=_t_ll):
                    S_ = spec_fn(x)
                    gm = _gate(S_[..., 16:32].sum(-1), _tm)
                    gh = _gate(S_[..., 32:96].sum(-1), _th)
                    gl = _gate(S_[..., 7:16].sum(-1), _tl)
                    gll = _gate005(S_[..., 1:7].sum(-1), _tll)
                    return dx_pde(x) + (8.0 / 3.0) * (_g_bb(x) + gm * _g_mid(x)
                                                      + gh * _g_hi(x) + gl * _g_low(x)
                                                      + gll * _g_ll(x))
                dx = jax.jit(dx)
                smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx, 3.0, temp=0.30)
                xb = recon[sl]
                k0 = jax.random.PRNGKey(700)
                ys.append(np.asarray(smp(P, sa3 * jnp.asarray(xb) + s13 * jax.random.normal(
                    jax.random.fold_in(k0, i), xb.shape), jax.random.fold_in(k0, i + 1))))
                jax.clear_caches()
            y = np.concatenate(ys)
            if os.environ.get('SAVE_FIELDS'):     # store reconstructions so per-band placement
                fd = f'base_results/fields/re{R}'  # and deficit analyses can be done offline
                os.makedirs(fd, exist_ok=True)
                np.savez_compressed(f'{fd}/{m}__K3__{tag}.npz', x=y.astype(np.float16))
                print(f"    saved fields -> {fd}/{m}__K3__{tag}.npz", flush=True)
            E_s = np.asarray(spec_fn(jnp.asarray(y)))
            E = E_s.mean(0)
            ps_ret = E_s[:, 32:96].sum(1) / E_gt_s[:, 32:96].sum(1)
            Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
            ps_place = np.array([np.corrcoef(Eh[j].ravel(), Ehg[j].ravel())[0, 1] for j in range(len(y))])
            ry = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(y[j:j + 32]))).ravel()
                                       for j in range(0, len(y), 32)]).mean())
            vals = dict(ret=E[32:96].sum() / E_gt[32:96].sum(),
                        place=np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1],
                        lowk=E[1:5].sum() / E_gt[1:5].sum(), resid_ratio=ry / rg,
                        mse=np.mean((y[..., 1] - xg[..., 1]) ** 2) * SIG ** 2,
                        inband=float(np.mean((ps_ret > 0.8) & (ps_ret < 1.2))))
            for f, vv in vals.items():
                S[f'{key}||{f}'] = np.float32(vv)
            S[f'{key}||E'] = E.astype(np.float32)
            S[f'{key}||Eb'] = np.array([E[a:b].sum() / E_gt[a:b].sum()
                                        for a, b in ((1, 5), (5, 16), (16, 32), (32, 64), (64, 96))], np.float32)
            S[f'{key}||ps_ret_paired'] = ps_ret.astype(np.float32)
            S[f'{key}||ps_place'] = ps_place.astype(np.float32)
            S[f'{key}||ps_mse'] = (np.mean((y[..., 1] - xg[..., 1]) ** 2, axis=(1, 2)) * SIG ** 2).astype(np.float32)
            # store fields + per-band arrays so every metric can be backfilled offline later
            _fd = f'base_results/fields/re{R}'; os.makedirs(_fd, exist_ok=True)
            np.savez_compressed(f"{_fd}/{m}__K3__{tag}.npz", x=np.asarray(y, np.float16))
            S[f'{key}||psEb'] = np.stack([E_s[:, a:b].sum(1) for a, b in
                                          ((1, 5), (5, 16), (16, 32), (32, 64), (64, 96))], 1).astype(np.float32)
            np.savez(store_path, **S)
            print(f"  {m:<10} {tag:<9} ret={vals['ret']:.3f} place={vals['place']:.3f} "
                  f"resid={vals['resid_ratio']:.2f} mse={vals['mse']:.2f} | PER-SAMPLE in-band "
                  f"{vals['inband'] * 100:.0f}% [p10 {np.percentile(ps_ret, 10):.2f}, "
                  f"p90 {np.percentile(ps_ret, 90):.2f}]", flush=True)
print("\nV7 BANDGATE COMPLETE", flush=True)
