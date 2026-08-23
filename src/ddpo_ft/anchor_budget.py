"""THE ANCHOR BUDGET (user 2026-08-23): how many high-res samples of a target flow are needed
to estimate its statistics (spectrum + vorticity distribution) well enough to fine-tune on?
Per regime: bootstrap n decorrelated frames (spacing >=10, the measured decorrelation gap)
from the train pool, compare the n-sample estimate to the full-pool reference:
  - band-energy errors |Ê/E_ref - 1| for [1,5), [10,32), [32,96)  (the [32,96) error passes
    ~one-for-one into trained dose, per the anchor-bias analysis)
  - max |log Ê(k) - log E(k)| over k in [10,96)
  - vorticity decile error (max over deciles, normalized by the pool std)
Also: the correlated-sequence variant (n CONSECUTIVE frames from one sequence) to measure the
effective-sample-size penalty, and n=1 = a single triplet (3 consecutive frames averaged).
CPU-only. Output: base_results/anchor_budget.npz + printed verdicts.
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax.numpy as jnp
from src.rewards import make_spectrum_fn
from src.sequence_inference import build_triplets, load_sequence

MEAN, SIG, N = 0.0, 4.7988, 256
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {1000: ('flow-data/kf_2d_re1000_256_40seed.npy', list(range(0, 16))),
           **{R: (GEN.format(R), list(range(0, 8)))
              for R in (1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000)}}
spec_fn = make_spectrum_fn(N)
BANDS = [(1, 5), (10, 32), (32, 96)]
NS = [1, 2, 4, 8, 16, 32, 64, 128]
B = 300
rng = np.random.default_rng(0)
OUT = {}

for R, (path, seqs) in REGIMES.items():
    frames, seq_id = [], []
    for si, s in enumerate(seqs):
        q = load_sequence(path, s)
        g = build_triplets(q, MEAN, SIG)
        frames.append(g[:, :, :, 1])          # middle frame, normalized
        seq_id.append(np.full(len(g), si))
    frames = np.concatenate(frames); seq_id = np.concatenate(seq_id)
    # per-frame spectra (chunked)
    S = np.concatenate([np.asarray(spec_fn(jnp.asarray(frames[i:i + 64][..., None]
                        .repeat(3, -1)))) for i in range(0, len(frames), 64)])
    W = (frames * SIG).reshape(len(frames), -1)
    # decorrelated pool: every 10th frame within each sequence
    keep = np.concatenate([np.where(seq_id == si)[0][::10] for si in range(len(seqs))])
    Sd, Wd = S[keep], W[keep]
    ref_spec = S.mean(0)
    ref_dec = np.percentile((frames * SIG).ravel()[::37], np.arange(10, 100, 10))
    wstd = (frames * SIG).std()
    print(f"\n=== Re={R}: pool {len(frames)} frames, {len(Sd)} decorrelated ===", flush=True)
    for n in NS:
        if n > len(Sd): break
        errs = {f'b{lo}_{hi}': [] for lo, hi in BANDS}
        errs.update(maxlog=[], dec=[])
        for b in range(B):
            idx = rng.choice(len(Sd), n, replace=False)
            Es = Sd[idx].mean(0)
            for lo, hi in BANDS:
                errs[f'b{lo}_{hi}'].append(abs(Es[lo:hi].sum() / ref_spec[lo:hi].sum() - 1))
            errs['maxlog'].append(np.abs(np.log(Es[10:96] + 1e-12)
                                         - np.log(ref_spec[10:96] + 1e-12)).max())
            dq = np.percentile(Wd[idx].ravel(), np.arange(10, 100, 10))
            errs['dec'].append(np.abs(dq - ref_dec).max() / wstd)
        for k2, v in errs.items():
            OUT[f'{R}|n{n}|{k2}|p50'] = np.float32(np.median(v))
            OUT[f'{R}|n{n}|{k2}|p90'] = np.float32(np.percentile(v, 90))
        print(f"  n={n:>3}: hi-band err p50={np.median(errs['b32_96']):.3f} "
              f"p90={np.percentile(errs['b32_96'], 90):.3f}  "
              f"deciles p90={np.percentile(errs['dec'], 90):.3f}", flush=True)
    # correlated-sequence variant: n consecutive frames from one sequence
    for n in (8, 32):
        errs = []
        for b in range(B):
            si = rng.integers(len(seqs))
            rows = np.where(seq_id == si)[0]
            if len(rows) < n: continue
            i0 = rng.integers(len(rows) - n + 1)
            Es = S[rows[i0:i0 + n]].mean(0)
            errs.append(abs(Es[32:96].sum() / ref_spec[32:96].sum() - 1))
        OUT[f'{R}|corr{n}|b32_96|p90'] = np.float32(np.percentile(errs, 90))
        print(f"  {n} CONSECUTIVE frames: hi-band err p90={np.percentile(errs, 90):.3f} "
              f"(vs decorrelated p90={float(OUT[f'{R}|n{n}|b32_96|p90']):.3f})", flush=True)
np.savez('base_results/anchor_budget.npz', **OUT)
print("\nANCHOR BUDGET COMPLETE", flush=True)
