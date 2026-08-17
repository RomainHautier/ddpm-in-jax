"""SWEEP EXTENSION: transport the Re=2000-TARGETED fine-tune across the research regimes.

Gap it closes (user, 2026-08-17): the setpoint ablation never ran the Re=2000-targeted
checkpoints anywhere but Re=2000 — regimes 3000-8000 saw only the base and the two
Re=1000-targeted checkpoints. Yet the Re=2000 fine-tune learned a HOTTER dose (its home
requirement is larger), so transported outward it plausibly needs shallower cascades and may
grade better than the Re=1000 transfers. This extension gives the anchor that candidate.

Same machinery as the ablation: blind score on the regime's extrapolated obs-fit anchor over
its source sequences; cells within/near the engraved band audited against held-out ground
truth. Two checkpoints: 0149 (the conservative early one the rule favours) and 0599 (the
hottest, retention-best at home). Depths K2..K7 (the K1s never move the blind score).

Output: base_results/setpoint_ablation_re2k.npz, same S|/A| key format as the ablation, so
ablation_choice_replay.py-style analyses can merge both files. Resume-aware.
"""
import os, sys, pickle
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
from psample import pbatched as batched     # all-chip sampling; PSAMPLE=0 restores serial

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
LADDER = [('K2x50', [100, 75], 50), ('K3x86', [150, 100, 50], 86),
          ('K4x110', [200, 150, 100, 50], 110), ('K5x140', [250, 200, 150, 100, 50], 140),
          ('K6x170', [300, 250, 200, 150, 100, 50], 170),
          ('K7x200', [350, 300, 250, 200, 150, 100, 50], 200)]
BAND_LO, BAND_HI, MARGIN = 0.764, 0.850, 0.05   # audit anything within/near either arm's band
PER_SRC, PER_HELD = 8, 10
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {R: dict(gt=GEN.format(R), anchor=f'base_results/regime_stats_re{R}_obsfit_gen.npz')
           for R in (1500, 3000, 4000, 5000, 6000, 7000, 8000)}
CK = 'monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter{:04d}.pkl'
MODELS = {'re2k-149': CK.format(149), 're2k-599': CK.format(599)}

OUTP = 'base_results/setpoint_ablation_re2k.npz'
OUT = {}
if os.path.exists(OUTP):
    old = np.load(OUTP, allow_pickle=True)
    OUT = {k: old[k] for k in old.files}
    print(f"resume: {len(OUT)} keys preloaded", flush=True)

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
PARAMS = {k: pickle.load(open(v, 'rb'))['params'] for k, v in MODELS.items()}


def pool(gt, seqs, n_per, with_gt=False):
    xg, xl = [], []
    for s in seqs:
        q = load_sequence(gt, s)
        l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(l) - 1, n_per).astype(int)
        xl.append(l[i2])
        if with_gt:
            xg.append(build_triplets(q, MEAN, SIG)[i2])
    return (np.concatenate(xg) if with_gt else None), np.concatenate(xl)


for R, c in REGIMES.items():
    dnpz = np.load(c['anchor']); A = dnpz['spec_ref']
    src_seqs = eval(dnpz['obs_source'].item().decode().split('|seqs=')[1])
    held = [s for s in range(20) if s not in src_seqs]
    dx = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    # PHASE A - blind selection scores on the anchor's source pool
    _, xl = pool(c['gt'], src_seqs, PER_SRC)
    recon = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    Eh_rc = local_hik_energy(recon[..., 1] * SIG, HIK0, 6.0)
    print(f"\n=== Re={R}: re2k transfer, SELECTION on seqs {src_seqs} ({len(xl)} triplets) ===",
          flush=True)
    for cname, starts, steps in LADDER:
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=0.30)
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        for mname, P in PARAMS.items():
            if f'S|{R}|{cname}|{mname}||blind' in OUT:
                continue
            y = batched(lambda xb, kk: smp(P, sa * xb + s1 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
            E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
            Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
            OUT[f'S|{R}|{cname}|{mname}||blind'] = np.float32(E[10:96].sum() / A[10:96].sum())
            OUT[f'S|{R}|{cname}|{mname}||proxy'] = np.float32(
                np.corrcoef(Eh.ravel(), Eh_rc.ravel())[0, 1])
            print(f"  {cname:<8} {mname:<9} blind={float(OUT[f'S|{R}|{cname}|{mname}||blind']):.3f}"
                  f" proxy={float(OUT[f'S|{R}|{cname}|{mname}||proxy']):.3f}", flush=True)
            np.savez(OUTP, **OUT)
    # PHASE B - audit every cell within/near the band on the held-out pool
    to_grade = [(cn, m) for cn, _, _ in LADDER for m in MODELS
                if BAND_LO - MARGIN <= float(OUT[f'S|{R}|{cn}|{m}||blind']) <= BAND_HI + MARGIN
                and f'A|{R}|{cn}|{m}||ret' not in OUT]
    if not to_grade:
        print(f"  AUDIT: nothing new to grade", flush=True)
        continue
    xg, xl2 = pool(c['gt'], held, PER_HELD, with_gt=True)
    recon2 = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl2, 500)
    E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    print(f"  AUDIT on {len(held)} held-out seqs ({len(xg)} triplets): {len(to_grade)} cells",
          flush=True)
    for cname, mname in to_grade:
        starts, steps = next((s, st) for cn, s, st in LADDER if cn == cname)
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=0.30)
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        y = batched(lambda xb, kk: smp(PARAMS[mname], sa * xb + s1 * jax.random.normal(
            jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon2, 700)
        E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
        Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
        OUT[f'A|{R}|{cname}|{mname}||ret'] = np.float32(E[HIK0:96].sum() / E_gt[HIK0:96].sum())
        OUT[f'A|{R}|{cname}|{mname}||place'] = np.float32(
            np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1])
        OUT[f'A|{R}|{cname}|{mname}||kstar'] = np.float32(eff_resolution(E, E_gt))
        print(f"    {cname:<8} {mname:<9} ret={float(OUT[f'A|{R}|{cname}|{mname}||ret']):.3f} "
              f"place={float(OUT[f'A|{R}|{cname}|{mname}||place']):.3f} "
              f"k*={int(OUT[f'A|{R}|{cname}|{mname}||kstar'])}", flush=True)
        np.savez(OUTP, **OUT)

print("\nRE2K TRANSFER EXTENSION COMPLETE", flush=True)
