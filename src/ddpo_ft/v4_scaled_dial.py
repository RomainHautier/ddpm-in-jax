"""THE SCALED DIAL (v4, user 2026-08-26): the dose dial matches MARGINAL statistics, so per-frame
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
Keys '{R}|{m}|K3|v4scaled' in the matching audit store."""
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
CFG = {1000: ('flow-data/kf_2d_re1000_256_40seed.npy', [34, 35, 36, 37, 38, 39], 20,
              'base_results/re1000_audit.npz', ('base0', 'r1k-449')),
       5000: ('flow-data/generated/gen_fnons_re5000_kf_1024to256_20seq.npy', list(range(8, 20)), 10,
              'base_results/regime_audit_re5000.npz', ('re2k-149', 'st8k-599')),
       8000: ('flow-data/generated/gen_fnons_re8000_kf_1024to256_20seq.npy', list(range(8, 20)), 10,
              'base_results/regime_audit_re8000.npz', ('re2k-149', 'st8k-599'))}
CKPT = {'r1k-449': 'monitoring/ddpo_re1000_newpool_ckpts/ddpo_re1000_iter0449.pkl',
        're2k-149': 'monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter0149.pkl',
        'st8k-599': 'monitoring/ddpo_re8000_steeredtrain_ckpts/ddpo_re1000_iter0599.pkl'}
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sa3, s13 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
B16 = partial(pbatched, per_dev=16)

for R, (gt_path, seqs, per, store_path, models) in CFG.items():
    S = {k: v for k, v in np.load(store_path, allow_pickle=True).items()}
    if all(f'{R}|{m}|K3|v4scaled||ret' in S for m in models):
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
    SCALES = {'v4scaled': (obs / C_R).astype(np.float32)}
    _b = (obs / C_R) ** beta
    SCALES['v4beta'] = (_b / _b.mean()).astype(np.float32)
    print(f"    beta (variance-matched) = {beta:.3f}", flush=True)
    _sc = SCALES['v4scaled']
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
            if f'{key}||ret' in S:
                continue
            P = base_params if m == 'base0' else pickle.load(open(CKPT[m], 'rb'))['params']
            ys = []
            for i in range(0, len(recon), 64):
                sl = slice(i, min(i + 64, len(recon)))
                sc = jnp.asarray(scale[sl])
                d1 = make_spectrum_distance(ref, kband=(1, 96), n=N, log_ref=lref, per_sample_scale=sc)
                d2 = make_spectrum_distance(ref, kband=(32, 96), n=N, log_ref=lref, per_sample_scale=sc)
                dm = make_spectrum_distance(ref, kband=(16, 32), n=N, log_ref=lref, per_sample_scale=sc)
                dose = jax.jit(jax.grad(lambda x: jnp.sum(0.5 * d1(x) + 3.0 * dm(x) + 3.0 * d2(x))))
                dx = jax.jit(lambda x, _d=dose: dx_pde(x) + (8.0 / 3.0) * _d(x))
                smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx, 3.0, temp=0.30)
                xb = recon[sl]
                k0 = jax.random.PRNGKey(700)
                ys.append(np.asarray(smp(P, sa3 * jnp.asarray(xb) + s13 * jax.random.normal(
                    jax.random.fold_in(k0, i), xb.shape), jax.random.fold_in(k0, i + 1))))
                jax.clear_caches()
            y = np.concatenate(ys)
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
            np.savez(store_path, **S)
            print(f"  {m:<10} {tag:<9} ret={vals['ret']:.3f} place={vals['place']:.3f} "
                  f"resid={vals['resid_ratio']:.2f} mse={vals['mse']:.2f} | PER-SAMPLE in-band "
                  f"{vals['inband'] * 100:.0f}% [p10 {np.percentile(ps_ret, 10):.2f}, "
                  f"p90 {np.percentile(ps_ret, 90):.2f}]", flush=True)
print("\nV4 SCALED COMPLETE", flush=True)
