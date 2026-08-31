"""Mark which evaluation triplets fall inside the spin-up transient the ANCHOR excludes.

The reward/steering anchors are built by anchor_obsfit_builder._frames, which for these
320-frame sequences takes range(40, 312, 8) - it deliberately drops the first 40 frames as
transient. The audits, however, sample triplets with linspace over the WHOLE sequence, so the
evaluation population contains frames the anchor was built to exclude. Those frames are ~10-12%
quieter in the fine band, which biases every ensemble number and disproportionately hurts
configurations that dose per sample.

This stores, per regime, the triplet index of every evaluation sample and a boolean mask for the
post-transient subset, so any metric can be recomputed on the consistent population offline -
no re-inference. Keys '{R}|EVAL||idx', '{R}|EVAL||seq', '{R}|EVAL||post_transient'.

  JAX_PLATFORMS=cpu python -m src.ddpo_ft.add_eval_mask
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.')
import numpy as np

TRANSIENT = 40                      # frames dropped by anchor_obsfit_builder._frames
NTRIP = 318                         # triplets per 320-frame sequence
RECIPE = {1000: ([34, 35, 36, 37, 38, 39], 20)}
for R in (1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000):
    RECIPE[R] = (list(range(8, 20)), 10)

for R, (seqs, per) in RECIPE.items():
    p = 'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz'
    if not os.path.exists(p):
        print(f"  Re={R}: no store"); continue
    S = {k: v for k, v in np.load(p, allow_pickle=True).items()}
    idx = np.concatenate([np.linspace(0, NTRIP - 1, per).astype(int) for _ in seqs])
    sq = np.repeat(seqs, per)
    keep = idx >= TRANSIENT
    # a stored per-sample array must have exactly this length, or the recipe here is wrong
    lens = {v.shape[0] for k, v in S.items() if k.startswith(f'{R}|') and k.endswith('||ps_place')}
    ok = (not lens) or lens == {len(idx)}
    S[f'{R}|EVAL||idx'] = idx.astype(np.int32)
    S[f'{R}|EVAL||seq'] = sq.astype(np.int32)
    S[f'{R}|EVAL||post_transient'] = keep
    np.savez(p, **S)
    print(f"  Re={R:<5} {len(idx)} triplets, {keep.sum()} post-transient ({keep.mean()*100:.0f}%)"
          f"   per-sample array lengths {sorted(lens) or '(none)'}  "
          f"{'OK' if ok else '<-- LENGTH MISMATCH, recipe is wrong'}", flush=True)
print("EVAL MASK COMPLETE", flush=True)
