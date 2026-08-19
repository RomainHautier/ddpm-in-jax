"""PLACEMENT-DIAL PILOT (user 2026-08-19): a third, PLACED steering gradient — data-consistency
with the sample's OWN low-res input, GT-free.

Motivation (measured): the LR observation's local band-energy map correlates 0.7-0.86 with
ground truth in [32,64) — ABOVE its own Nyquist (fine-scale activity condenses on the resolved
strain/shear zones) — and deep cascades destroy exactly that inherited signal because no reward
or guidance term is placed. This dial adds mu * grad of a map-consistency loss:
    L_place(x) = sum_b || nmap_b(x) - nmap_b(LR_x) ||^2  over bands [16,32),[32,64)
with nmap_b = the Gaussian-smoothed local band-energy map, per-map mean-normalized so the term
constrains WHERE energy sits, not how much (leaving dose to the spectral dial). Differentiable
end-to-end; the LR reference map is fixed per sample (no gradient through it).

Grid: re2k-599 and re2k-149 at K3x86 and K5x140 (the depth that hurts placement most),
mu in {0, 1, 3, 10}, combined with the standard lam_pde=3 (spectral dial off, isolate the
effect), at Re=5000/8000. Keys '{R}|{nm}|{cfg}mu{v}' in steering_pilot.npz.
"""
import os, sys, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from functools import partial
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn, make_residual_loss
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution
from psample import pbatched

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
DEPTHS = {'K3x86': ([150, 100, 50], 86), 'K5x140': ([250, 200, 150, 100, 50], 140)}
MUS = [0.0, 1.0, 3.0, 10.0]
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {5000: dict(gt=GEN.format(5000), seqs=list(range(8, 20)), per=10,
                      anchor='base_results/regime_stats_re5000_obsfit_gen.npz'),
           8000: dict(gt=GEN.format(8000), seqs=list(range(8, 20)), per=10,
                      anchor='base_results/regime_stats_re8000_obsfit_gen.npz')}
R2K = 'monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter{:04d}.pkl'
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
MODELS = {'re2k-599': pickle.load(open(R2K.format(599), 'rb'))['params'],
          're2k-149': pickle.load(open(R2K.format(149), 'rb'))['params']}
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
B16 = partial(pbatched, per_dev=16)

# band machinery in jnp (differentiable)
fy = np.fft.fftfreq(N) * N
kmag = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
gsm = jnp.asarray(np.exp(-2.0 * (np.pi * 6.0) ** 2 *
                         ((fy[:, None] / N) ** 2 + (fy[None, :] / N) ** 2)))
PB = [jnp.asarray(((kmag >= lo) & (kmag < hi)).astype(np.float32)) for lo, hi in
      [(16, 32), (32, 64)]]


def nmaps(w):
    """w (B,H,W) physical -> list of per-band mean-normalized smoothed local energy maps."""
    F = jnp.fft.fft2(w)
    out = []
    for m in PB:
        bp = jnp.real(jnp.fft.ifft2(F * m))
        e = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(bp ** 2) * gsm))
        out.append(e / (jnp.mean(e, axis=(-2, -1), keepdims=True) + 1e-12))
    return out


def make_place_dx(lr_ref_maps):
    """grad of sum_b ||nmap_b(x) - ref_b||^2 wrt the normalized triplet x. refs fixed."""
    refs = [jax.lax.stop_gradient(r) for r in lr_ref_maps]
    def loss(x):
        ms = nmaps(x[..., 1] * SIG)
        return sum(jnp.sum((m - r) ** 2) for m, r in zip(ms, refs)) / (N * N)
    return jax.grad(loss)


OUT = {}
OUTP = 'base_results/steering_pilot.npz'
if os.path.exists(OUTP):
    old = np.load(OUTP, allow_pickle=True); OUT = {k: old[k] for k in old.files}

for R, c in REGIMES.items():
    dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    xg, xl = [], []
    for s in c['seqs']:
        q = load_sequence(c['gt'], s)
        g = build_triplets(q, MEAN, SIG)
        l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, c['per']).astype(int)
        xg.append(g[i2]); xl.append(l[i2])
    xg, xl = np.concatenate(xg), np.concatenate(xl)
    resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
    E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                               for i in range(0, len(xg), 32)]).mean())
    A = np.load(c['anchor'])['spec_ref']
    recon = B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    print(f"\n=== Re={R}: placement-dial grid ===", flush=True)
    for cname, (starts, steps) in DEPTHS.items():
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        for mu in MUS:
            for nm, P in MODELS.items():
                key = f'{R}|{nm}|{cname}mu{mu:g}'
                if f'{key}||ret' in OUT:
                    continue
                # per-chunk sampler: the LR reference maps are baked per chunk (fixed refs)
                def run_chunk(xb_lr, xb_rc, kk):
                    if mu == 0.0:
                        dx = dx_pde
                    else:
                        pd = make_place_dx(nmaps(jnp.asarray(xb_lr[..., 1]) * SIG))
                        dx = lambda x: dx_pde(x) + (mu / 3.0) * pd(x)
                    smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0,
                                                   temp=0.30)
                    return smp(P, sa * jnp.asarray(xb_rc) + s1 * jax.random.normal(
                        jax.random.fold_in(kk, 1), xb_rc.shape), jax.random.fold_in(kk, 2))
                # chunked manually (sampler recompiles per chunk shape only once; refs traced in)
                ys, k0 = [], jax.random.PRNGKey(700)
                B = 64
                for i in range(0, len(recon), B):
                    ys.append(np.asarray(run_chunk(xl[i:i + B], recon[i:i + B],
                                                   jax.random.fold_in(k0, i))))
                y = np.concatenate(ys)
                E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
                ry = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                                           for i in range(0, len(y), 32)]).mean())
                Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
                vals = dict(ret=E[HIK0:96].sum() / E_gt[HIK0:96].sum(),
                            place=np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1],
                            lowk=E[1:5].sum() / E_gt[1:5].sum(), kstar=eff_resolution(E, E_gt),
                            resid_ratio=ry / rg,
                            mse=np.mean((y[..., 1] - xg[..., 1]) ** 2) * SIG ** 2)
                for f, vv in vals.items():
                    OUT[f'{key}||{f}'] = np.float32(vv)
                OUT[f'{key}||E'] = E.astype(np.float32)
                np.savez(OUTP, **OUT)
                print(f"  {nm:<9} {cname} mu={mu:<4g} ret={vals['ret']:.3f} "
                      f"place={vals['place']:.3f} resid={vals['resid_ratio']:.2f}xGT "
                      f"mse={vals['mse']:.2f} k*={vals['kstar']}", flush=True)
print("\nPLACE DIAL COMPLETE", flush=True)
