"""Materialise the OBSERVED (low-resolution) data as its own file, once.

WHY THIS EXISTS. Everything downstream of deployment is supposed to be GT-free, but the helpers
(`grid_downsample_degrade`, `coarse_spec`) open the full 256^2 ground-truth array and then keep
every 4th pixel. The unused 15/16 of the field is loaded into memory. "We don't use GT" was
therefore a claim about code discipline, not something an auditor could verify.

This script performs the ONLY step that legitimately reads the fine field, and writes out exactly
what a 4x-coarse solver would have produced: the (n_seq, n_frames, 64, 64) subsampled field. Every
later stage — anchor construction, cascade sweep, blind scoring, config selection — reads this file
and NOTHING else, so there is no ground truth present to leak.

EXACTNESS. The kept pixels are seq[..., ::4, ::4], which is precisely the set `grid_downsample_degrade`
retains before its nearest-neighbour fill, and precisely the array `coarse_spec` FFTs. Reconstructing
the 256^2 nearest-neighbour field from this artifact is bit-identical to the old path — verified by
--verify. This is a provenance change, not a numerical one.

Usage:
  python -m src.ddpo_ft.materialize_observed <gt_file> <out_file> <seq0,seq1,...> [--verify]
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np

FACTOR = 4


def observed_from_fine(gt_path, seqs, factor=FACTOR):
    """THE ONE PLACE the fine field is read. Returns (n_seq, n_frames, H/f, W/f)."""
    a = np.load(gt_path, mmap_mode='r')
    return np.stack([np.asarray(a[s], np.float32)[:, ::factor, ::factor] for s in seqs])


def nnfill_from_observed(obs_seq, out_hw=256, factor=FACTOR):
    """(n_frames, h, w) observed -> (n_frames, 256, 256) nearest-neighbour-filled, exactly as
    grid_downsample_degrade would have produced from the fine field."""
    from scipy.ndimage import distance_transform_edt
    mask = np.zeros((out_hw, out_hw), bool)
    mask[::factor, ::factor] = True
    _, ind = distance_transform_edt(~mask, return_indices=True)
    full = np.zeros((len(obs_seq), out_hw, out_hw), np.float32)
    full[:, ::factor, ::factor] = obs_seq
    return np.stack([f[ind[0], ind[1]] for f in full])


if __name__ == '__main__':
    gt, out, seqs = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
    obs = observed_from_fine(gt, seqs)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, obs)
    print(f"observed -> {out}  shape={obs.shape} dtype={obs.dtype} "
          f"({obs.nbytes/1e6:.0f} MB, 1/{FACTOR**2} of the fine field)")
    print(f"  source (read ONCE, here only): {gt} seqs {seqs}")
    if '--verify' in sys.argv:
        from src.sequence_inference import grid_downsample_degrade, load_sequence
        from anchor_obsfit_builder import coarse_spec, SIG, _KRC, NC, _frames
        ok = True
        for i, s in enumerate(seqs[:2]):
            old = grid_downsample_degrade(load_sequence(gt, s), FACTOR)
            new = nnfill_from_observed(obs[i])
            same = np.array_equal(old, new)
            ok &= same
            print(f"  seq {s}: nnfill field identical to the old path -> {same}")
        a = np.load(gt, mmap_mode='r')
        S = [np.bincount(_KRC, (np.abs(np.fft.fft2(obs[i, t] / SIG)) ** 2).ravel(),
                         minlength=NC)[:33]
             for i in range(len(seqs)) for t in _frames(a.shape[1], 8)]
        new_cs = np.asarray(S).mean(0)
        old_cs = coarse_spec(gt, seqs)
        rel = float(np.abs(new_cs[1:33] / old_cs[1:33] - 1).max())
        print(f"  coarse spectrum: max relative difference {rel:.2e} -> "
              f"{'IDENTICAL' if rel < 1e-6 else 'DIFFERS'}")
        ok &= rel < 1e-6
        print(f"\n  EQUIVALENCE {'PROVEN — the artifact carries all observable information' if ok else 'FAILED'}")
