# Inference scripts — structure & overlap overview

Three scripts in `src/` run inference with the trained DDPM. They are **the same
core algorithm** (SDEdit-style reconstruction) wrapped in three different
data/batching/output harnesses. This document maps how each is built, the methods
each pulls in, and where they overlap — as a reference before consolidating them.

## The shared algorithm (SDEdit reconstruction)

Every script does the same thing at its heart:

1. Take an input field `x_g` (clean GT, or a degraded version of it).
2. Forward-noise it to a chosen timestep `t_start = S`:
   `x_T = sqrt(ᾱ_S)·x_g + sqrt(1−ᾱ_S)·ε`
3. Run the reverse diffusion from `S` back to `0` to get a reconstruction `x_0`.
4. (Optionally) feed `x_0` back in as the new `x_g` and repeat `K` times with a
   schedule of horizons `S = [S_1, …, S_K]`.

The reverse step is the standard DDPM posterior mean + noise:
`x_{t-1} = (1/√α_t)·(x_t − (1−α_t)/√(1−ᾱ_t)·ε_θ(x_t,t)) + √β_t·z`.

This exact step is implemented **twice** (see Overlaps §1).

---

## Script-by-script anatomy

### 1. `src/plain_ddpm_inference.py` — baseline (no degradation)
- **Config block:** `plain_diffusion`
- **Question it answers:** how faithfully does the model reconstruct *clean* GT as a
  function of the noise horizon `S`? (a baseline/upper-bound, not a real task)
- **Entry:** `run_inference(cfgs)`
- **Flow:**
  1. `load_checkpoint` → `DDPM(cfgs[0])`
  2. mesh over all devices; replicate params; shard each batch (`data_sharding`)
  3. `load_dataset(cfgs[0], max_test_samples=n, n_devices=…)` → test frames
  4. for each batch, **for each `s` in `S`** (a *sweep*, independent): one
     `ddpm.sample(x_g=ims, t_start=s)`; compute `mse(gt, recon)` per image
  5. pickle → `base_results/plain_ddpm_reconstructions_baseline.pkl`,
     upload via **raw `subprocess` gcloud**
- **Sampler:** `DDPM.sample` (model.py)
- **MSE:** `utils.mse` → `(mean, per_pixel)`, stored per `S`
- **Note:** `S` here is a list of horizons to compare, **not** an iterative schedule.

### 2. `src/inference.py` — sparse super-resolution (single frames)
- **Config blocks:** `sparse_diffusion` + `super_resolution` + `nn_fill`
- **Question it answers:** can the model super-resolve a *degraded* frame? Task 1 =
  low-res bicubic upsample; Task 2 = sparse sample + nearest-neighbour fill.
- **Entry:** `run_inference(cfgs)` ← **same name as plain's function**
- **Flow:**
  1. `load_checkpoint` → `DDPM`
  2. `load_dataset(cfgs[0], n_devices=1)` then `.shuffle(seed).take(n)` —
     **single device, one image at a time**
  3. per image: `lr_input` (task 1) or `sparsify_input` (task 2) → a **list** of
     seeded degraded inputs
  4. per input: **`K` iterations**, each `ddpm.sample(x_g, t_start=S[j])`, feeding
     the output back as the next `x_g`
  5. pickle → `monitoring/sparse_reconstructions/reconstructions_task{1,2}_*.pkl`,
     upload via `save_results_to_gcs`
- **Sampler:** `DDPM.sample` (model.py)
- **MSE:** none — deferred to notebooks
- **Note:** the only **unsharded / single-image** script — slow on TPU.

### 3. `src/sequence_inference.py` — full-sequence reconstruction
- **Config block:** `sequence_diffusion`
- **Question it answers:** reconstruct an entire flow *sequence* (video), frame by
  frame as `(t,t+1,t+2)` triplets, from a degraded input — the production task,
  built for TPU throughput.
- **Entries:** `run_sequence_inference(cfgs)` → `reconstruct_sequence(…)`
- **Flow:**
  1. `load_checkpoint` → `DDPM`; mesh; replicate params; build
     `make_batched_sampler(ddpm, data_sharding)` (its **own** batched sampler)
  2. resolve which sequences to run (`seq_idxs` = `"all"` / explicit list, else
     `test_set` last-N, else single `seq_idx`)
  3. per sequence (`reconstruct_sequence`):
     - `load_sequence(gt_path, seq_idx)` (mmap npy / npz)
     - degrade: `sparse_nnfill_degrade(gt)` **or** load a pre-degraded npz
     - `build_triplets` on both input and GT (normalize + stack 3 frames)
     - pad frame count to a multiple of `B`, split into chunks
     - per chunk: **`K` iterations**, each `sampler(params, x_g, key, S[j])`
     - per-frame MSE inline `((gt−pred)**2).mean()`
  4. pickle per sequence → `monitoring/sequence_reconstructions/…seq{i}.pkl`,
     upload via `save_results_to_gcs`
- **Sampler:** **its own** `make_batched_sampler` (duplicate of `DDPM.sample`)
- **Helpers it owns:** `load_sequence`, `num_sequences`, `sparse_nnfill_degrade`,
  `build_triplets`, `make_batched_sampler`, `reconstruct_sequence`

---

## Method/dependency map

| Helper | plain | inference | sequence |
|---|:--:|:--:|:--:|
| `DDPM.sample` (model.py) | ✅ | ✅ | — (uses own) |
| `make_batched_sampler` (dup of sample) | — | — | ✅ |
| `load_checkpoint` | ✅ | ✅ | ✅ |
| `load_dataset` (tf pipeline) | ✅ | ✅ | — |
| `load_sequence` / `build_triplets` (numpy) | — | — | ✅ |
| `mse` (util) | ✅ | — | inline |
| `sparsify_input` / `lr_input` | — | ✅ | — |
| `sparse_nnfill_degrade` | — | — | ✅ |
| `save_results_to_gcs` | ❌ raw subprocess | ✅ | ✅ |
| two-yaml `__main__` loader | ✅ | ✅ | ✅ |
| manual mesh / sharding | ✅ | — | ✅ |

## Overlaps & friction (consolidation targets)

1. **Reverse-diffusion step written twice.** `DDPM.sample`'s `jit_denoise_step`
   (model.py) and `make_batched_sampler`'s `denoise_step` (sequence_inference.py)
   are identical except `jnp.array([t])` vs `jnp.full((b,), t)`. Two places to keep
   in sync.
2. **`inference.py` is unsharded.** It is essentially a multi-seed, single-frame
   version of the sequence loop but shares none of its batching/sharding.
3. **Two data paths to the same triplets.** `load_dataset`/`build_samples` (tf) vs
   `load_sequence`/`build_triplets` (numpy) — `build_triplets` re-implements the
   normalization in `build_samples`.
4. **Boilerplate ×3:** device print + checkpoint load + `DDPM(cfg)` + mesh +
   results-dict + pickle + upload, repeated in each — and GCS upload done two ways.
5. **Config sprawl + name collisions.** Four parallel config blocks each redeclare
   `checkpoint`/`S`; `plain` and `inference` both define `run_inference`; MSE is
   computed three inconsistent ways (util / none / inline).

## Where outputs are consumed (downstream)
- `monitoring/sequence_reconstructions/*.pkl` → `viz/visualise_sequences.py`,
  `base_results/re2000/re2000_analysis.ipynb`
- `monitoring/sparse_reconstructions/*.pkl` → `viz/compute_tsne.py`, notebooks
- `base_results/…baseline.pkl` → baseline notebooks

## One-line summary
> Same SDEdit reconstruction loop in all three; they diverge only on **input
> degradation** (none / LR / sparse-NN-fill), **data unit** (frames vs sequences),
> and **batching** (sharded vs single-image). A single sampler + bootstrap + save
> helper + a degradation registry would collapse them to one core and three thin
> entrypoints.
