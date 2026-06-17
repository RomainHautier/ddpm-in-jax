# Generated Kolmogorov-flow datasets

Synthetic 2D Kolmogorov-flow vorticity datasets produced by the generators in
`data_generation/`, used to test the DDPM reconstruction model on flows it was **not**
trained on (different Reynolds number / independent simulation of the training regime).

**GCS location:** `gs://ddpm-thesis-rh/flow-data/generated/`

| File | Shape | Dtype | Size | Re | What it is |
|---|---|---|---|---|---|
| `jaxcfd_re2000_1024to256_40seq.npy` | `(40, 320, 256, 256)` | `float32` | 3.35 GB | 2000 | **Out-of-distribution** test set. 40 independent trajectories at Re=2000 (model was trained at Re≈1000). |
| `jaxcfd_re1000_1024to256_2seq.npy` | `(2, 320, 256, 256)` | `float32` | 167 MB | 1000 | **In-distribution** sanity check. 2 trajectories at the training Reynolds number, independently simulated. |

## Array layout

Each file is a NumPy `.npy` (load with `np.load(path)`), axis order:

```
[ sequence , frame , height , width ]
   n_seq      320     256      256
```

- The stored field is **scalar vorticity** ω(x, y) on the periodic box (0, 2π)².
- Frames are recorded at Δt = 1/32 time units, 320 frames per trajectory.
- Values are raw (un-normalized) vorticity. The model consumes
  `(ω − mean) / std`; the reference Re=1000 `kf_2d` set has std ≈ 4.7988.
  Compute per-file stats with `arr.mean()` / `arr.std()` before use.
- Triplet convention used downstream: reconstruction triplet `i` corresponds to raw
  frames `(i, i+1, i+2)`, so triplet `i`'s center is raw frame `i+1`.

## How they were generated

Both files were produced with `data_generation/generate_kmflow_jaxcfd.py` — Kochkov
et al.'s validated `jax-cfd` spectral solver for the 2D vorticity equation on (0, 2π)²,
periodic, with Kolmogorov forcing `f = -4 cos(4y) - 0.1·ω` (forcing wavenumber k=4 +
linear drag 0.1). Crank–Nicolson RK4 time stepping, CFL = 0.5.

- **DNS resolution 1024², spectrally downsampled to 256²** (the `1024to256` in the name).
  The 256² output keeps modes up to k=128, which a 1024² DNS over-resolves.
- **Spun up to statistically steady state** (~40 time units) before recording, because the
  reference `kf_2d` data is recorded in steady state (frame 0 already shows the k=4 forcing
  peak and std ≈ 4.6, not a smooth initial condition).
- Initial conditions: `jax-cfd` `filtered_velocity_field`, max velocity 7, peak wavenumber 4.
- Per-sequence reproducible seed: `PRNGKey(seed + sequence_index)`, base `seed = 0`.

Reproduce (needs a `jax-cfd` env, see the script header):

```bash
# Re=2000 OOD set (40 sequences)
python data_generation/generate_kmflow_jaxcfd.py --re 2000 --n-samples 40 \
    --dns-res 1024 --out-res 256 --spinup-time 40 --record-frames 320 \
    --record-dt 0.03125 --seed 0 --out jaxcfd_re2000_1024to256_40seq.npy

# Re=1000 in-distribution check (2 sequences)
python data_generation/generate_kmflow_jaxcfd.py --re 1000 --n-samples 2 \
    --dns-res 1024 --out-res 256 --spinup-time 40 --record-frames 320 \
    --record-dt 0.03125 --seed 0 --out jaxcfd_re1000_1024to256_2seq.npy
```

A second, PyTorch-based generator (`generate_kmflow.py`) and a statistical validator
(`validate_kmflow.py`, compares generated stats / spectra against `kf_2d`) are also in
`data_generation/`.

## Relation to the training / reference data

`gs://ddpm-thesis-rh/flow-data/kf_2d_re1000_256_40seed.npy` is the original Shu et al.
reference set (Re=1000, 256², 40 seeds, recorded in steady state) the DDPM was trained on.
The generated files above reproduce that pipeline with `jax-cfd` so the model can be tested
on (a) the same regime from an independent solver and (b) a higher, unseen Reynolds number.
