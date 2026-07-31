"""CROSS-REGIME DECOMPOSITION, TUNED ARM — the same models with the inference procedure adapted.

Companion to crossregime_decomposition.py (which runs the FROZEN config everywhere). Together they
answer: what does tuning the inference procedure buy, per regime, per band, and in the PDE residual?

- Re=1500/3000/4000/5000: the blind picks already selected in stage 2 are re-run here so their
  spectra, band energies, PDE residuals and fields are saved (the stage-2/3 logs kept only summary
  numbers). Selection was GT-free; this only re-measures what it chose.
- Re=1000/2000/10000: no blind ladder search was ever run for these, so the FULL 6-rung ladder is
  run for each NON-NATIVE model (the native model at its own regime needs no adaptation), the blind
  score is computed per rung, and the same band rule picks one. Every rung is saved, so the ladder
  itself is plottable.

Output: base_results/crossregime_adapted.npz + crossregime_adapted_fields.npz
"""
import os, sys, re as _re, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn, make_residual_loss
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
BAND_EDGES = [1, 5, 10, 16, 24, 32, 48, 64, 80, 96, 128]
SETPOINT = (0.798, 0.858)          # the blind band used by stage 2, unchanged
CFGS = {'K4[200,150,100,50]x110': ([200, 150, 100, 50], 110),
        'K3[150,100,50]x86':      ([150, 100, 50], 86),
        'K2[100,75]x50':          ([100, 75], 50),
        'K1[100]x20':             ([100], 20),
        'K1[75]x20':              ([75], 20),
        'K1[50]x12':              ([50], 12)}
LADDER = list(CFGS)
MODELS = {
    'in-dist Re=1000': 'monitoring/ddpo_re1000_k2_s100-75_ddim50_ddiminit_hk10_emabase_ckpts/ddpo_re1000_iter0299.pkl',
    'Re=2000 model':   'monitoring/ddpo_re2000_dt32_ckpts/ddpo_re1000_iter0599.pkl',
    'Re=10000 model':  'monitoring/ddpo_re10000_dt32_ckpts/ddpo_re1000_iter0549.pkl',
}
NATIVE = {1000: 'in-dist Re=1000', 2000: 'Re=2000 model', 10000: 'Re=10000 model'}
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {
 1000:  dict(gt='flow-data/kf_2d_re1000_256_40seed.npy', seqs=list(range(4, 20)), per=40,
             anchor='base_results/regime_stats_re1000.npz'),
 1500:  dict(gt=GEN.format(1500), seqs=list(range(4, 20)), per=40,
             anchor='base_results/regime_stats_re1500_obsfit_gen.npz'),
 2000:  dict(gt='flow-data/kf_re2000_256_40seed_dt32.npy', seqs=[s for s in range(40) if s > 4],
             per=None, anchor='base_results/regime_stats_re2000_obsfit_v3.npz'),
 3000:  dict(gt=GEN.format(3000), seqs=list(range(4, 20)), per=40,
             anchor='base_results/regime_stats_re3000_obsfit_gen.npz'),
 4000:  dict(gt=GEN.format(4000), seqs=list(range(4, 20)), per=40,
             anchor='base_results/regime_stats_re4000_obsfit_gen.npz'),
 5000:  dict(gt=GEN.format(5000), seqs=list(range(4, 20)), per=40,
             anchor='base_results/regime_stats_re5000_obsfit_gen.npz'),
 10000: dict(gt='flow-data/kf_re10000_256_40seed_dt32.npy',
             seqs=[s for s in range(40) if not 20 <= s <= 24], per=None,
             anchor='base_results/regime_stats_re10000_obsfit_dt32.npz'),
}

# ---- stage-2 blind picks for the four unseen regimes (parsed, not re-derived) ----
picks, cur_R, cur_m = {}, None, None
for line in open('monitoring/ab_pdelocal/blind_select_636.log'):
    m = _re.search(r'=== re(\d+) \(unseen\)', line) or _re.search(r'=== Re=(\d+) BLIND SELECTION', line)
    if m:
        cur_R = int(m.group(1)); continue
    m = _re.match(r'  (\S.*):\s*$', line)
    if m and m.group(1) in MODELS:
        cur_m = m.group(1); continue
    m = _re.search(r'=> BLIND PICK \(band .*\): (\S+)', line)
    if m and cur_R and cur_m:
        picks[(cur_R, cur_m)] = m.group(1)
print(f"parsed {len(picks)} stage-2 blind picks", flush=True)

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


def band_sums(E_all):
    return np.stack([E_all[:, a:b].sum(1) for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:])], 1)


OUT, FIELDS = {}, {}
for R, cfg in REGIMES.items():
    xg, xl, sid = [], [], []
    for s in cfg['seqs']:
        seq = load_sequence(cfg['gt'], s)
        g = build_triplets(seq, MEAN, SIG)
        l = build_triplets(grid_downsample_degrade(seq, 4), MEAN, SIG)
        if cfg['per']:
            i2 = np.linspace(0, len(g) - 1, cfg['per']).astype(int); g, l = g[i2], l[i2]
        xg.append(g); xl.append(l); sid += [s] * len(g)
    xg = np.concatenate(xg); xl = np.concatenate(xl); sid = np.array(sid)
    us = np.unique(sid); idx_by_seq = {s: np.where(sid == s)[0] for s in us}
    rng = np.random.default_rng(0)
    boot = [np.concatenate([idx_by_seq[s] for s in rng.choice(us, len(us), replace=True)])
            for _ in range(400)]
    print(f"\n=== Re={R}: {len(us)} sequences, {len(xg)} triplets ===", flush=True)

    dx = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
    Eg_all = np.asarray(spec_fn(jnp.asarray(xg))); E_gt = Eg_all.mean(0); Bg = band_sums(Eg_all)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    rg = np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                         for i in range(0, len(xg), 32)])
    A = np.load(cfg['anchor'])['spec_ref']
    recon = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)

    def measure(y, key, label):
        Ey = np.asarray(spec_fn(jnp.asarray(y))); E = Ey.mean(0); By = band_sums(Ey)
        ry = np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                             for i in range(0, len(y), 32)])
        Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
        br, bres, bb = [], [], []
        for m in boot:
            br.append(Ey[m][:, HIK0:96].sum(1).mean() / Eg_all[m][:, HIK0:96].sum(1).mean())
            bres.append(ry[m].mean() / rg[m].mean())
            bb.append(By[m].mean(0) / Bg[m].mean(0))
        blind = float(E[10:96].sum() / A[10:96].sum())
        OUT[key] = dict(E=E, E_gt=E_gt, band=By.mean(0) / Bg.mean(0), band_sd=np.array(bb).std(0),
                        ret=float(E[HIK0:96].sum() / E_gt[HIK0:96].sum()), ret_sd=np.std(br),
                        lowk=float(E[1:5].sum() / E_gt[1:5].sum()),
                        place=float(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1]),
                        resid=ry.mean(), resid_gt=rg.mean(), resid_ratio=ry.mean() / rg.mean(),
                        resid_ratio_sd=np.std(bres), kstar=eff_resolution(E, E_gt), blind=blind,
                        n_seq=len(us), n_trip=len(xg))
        print(f"    {label:<46} ret={OUT[key]['ret']:.3f}  place={OUT[key]['place']:.3f}  "
              f"resid={ry.mean():.1f} ({ry.mean()/rg.mean():.2f}x GT)  k*={OUT[key]['kstar']}  "
              f"blind={blind:.3f}", flush=True)
        return blind

    todo = {}          # config name -> [model names needing it]
    if R in NATIVE:    # full ladder for the non-native models
        for nm in MODELS:
            if nm != NATIVE[R]:
                for c in LADDER:
                    todo.setdefault(c, []).append(nm)
    else:              # the four unseen regimes: only the stage-2 picks
        for nm in MODELS:
            c = picks.get((R, nm))
            if c:
                todo.setdefault(c, []).append(nm)
    # config-outer so each cascade compiles ONCE per regime
    for cname, names in todo.items():
        starts, steps = CFGS[cname]
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=0.30)
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        for nm in names:
            y = batched(lambda xb, kk: smp(PARAMS[nm], sa * xb + s1 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
            measure(y, f'{R}|{nm}|{cname}', f'{nm} @ {cname}')
            FIELDS[f'{R}|{nm}|{cname}'] = y[:2, ..., 1] * SIG
    # apply the same band rule to whatever ladder was run here
    if R in NATIVE:
        for nm in MODELS:
            if nm == NATIVE[R]:
                continue
            rung = [(c, OUT[f'{R}|{nm}|{c}']['blind']) for c in LADDER if f'{R}|{nm}|{c}' in OUT]
            best = min(rung, key=lambda t: abs(t[1] - np.clip(t[1], *SETPOINT)))
            print(f"  BLIND PICK  {nm} @ Re={R}: {best[0]} (blind {best[1]:.3f}, "
                  f"band {SETPOINT})", flush=True)
    np.savez('base_results/crossregime_adapted.npz', band_edges=np.array(BAND_EDGES),
             keys=np.array(list(OUT)),
             **{f'{k}||{f}': v for k, d in OUT.items() for f, v in d.items()})
    np.savez_compressed('base_results/crossregime_adapted_fields.npz', **FIELDS)
print("\nADAPTED ARM COMPLETE", flush=True)
