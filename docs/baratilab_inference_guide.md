# Running BaratiLab / Shu et al. pretrained checkpoints (GPU) — handoff guide

**Goal.** Run BaratiLab's pretrained diffusion model on the Kolmogorov-flow sparse-reconstruction task,
dump its predictions, and push them to GCS so the JAX side can compare against our model in physical units.

**Why GPU.** Their code is CUDA-native PyTorch (`.to('cuda')` throughout `functions/denoising_step.py`
and `runners/rs256_guided_diffusion.py`). Our main box is a **TPU VM (no GPU)**. Measured on CPU there:
**0.3 s per model call**, and their sampling is ~240 steps/frame (the **conditional does 2 calls/step**,
classifier-free `w=3.0`) over 4 seqs × 318 frames = 1272 frames:

| scope | baseline | conditional |
|---|---|---|
| 1 frame | ~70 s | ~2.5 min |
| **full test set (1272 frames)** | **~25 h** | **~55 h** |

Hence: GPU. On a modern GPU this is minutes, not hours.

---

## ✅ Already verified on the TPU box (don't re-litigate)

- Both checkpoints **load cleanly**: `0 missing, 0 unexpected` keys against their `Model` /
  `ConditionalModel`. No architecture guesswork needed.
- `baseline_ckpt.pth` → **3.47 M** params · `conditional_ckpt_new.pth` → **3.51 M** params.
- Checkpoint layout is a **5-element list**: `[state_dict, optimizer, epoch(279), step(100000), EMA]`.
  Their runner loads **`torch.load(ckpt_path)[-1]`** — i.e. the **EMA weights**. Use that.
- The conditional's extra modules (the physics-conditioning path):
  `emb_conv.0` (64,3,1,1) · `emb_conv.2` (64,64,3,3) · `combine_conv` (64,128,1,1).
- A forward pass runs and returns `(B,3,256,256)`.

**What we already have — don't redo:** the **baseline** model's predictions from a previous run,
`gs://ddpm-thesis-rh/baratilab_results.pkl` (keys `reference`, `input`, `pred_it2`; physical;
`(4,318,256,256,3)`). Our model already beats it (~10% lower MSE on 100% of frames; commit `858496a`,
`base_results/baseline/compare_models.ipynb`).

> ### 👉 Run the **CONDITIONAL** checkpoint. That's the one that's never been evaluated.
> It is the "Learned" physics-informed variant and the direct counterpart to our `grad_frozen60`.

---

## 1. Repo + environment

```bash
git clone https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution.git
cd Diffusion-based-Fluid-Super-resolution
git checkout main_v1        # IMPORTANT: main_v1, not main
```

Their tested env (newer torch works — verified on torch 2.13):
```
python 3.8 | PyTorch 1.7 + CUDA 10.1 | torchvision 0.8.2
numpy | tqdm | einops | matplotlib | tensorboard
```

## 2. Get the weights + data — **all already on our GCS, no figshare needed**

**Checkpoints → `./pretrained_weights/`** (keep these exact filenames — the configs reference them):
```bash
gsutil cp gs://ddpm-thesis-rh/checkpoints/ddpm/pretrained_weights/conditional_ckpt_new.pth ./pretrained_weights/
gsutil cp gs://ddpm-thesis-rh/checkpoints/ddpm/pretrained_weights/baseline_ckpt.pth ./pretrained_weights/   # optional
```

**Data → `./data/`**:
```bash
gsutil cp gs://ddpm-thesis-rh/flow-data/kf_2d_re1000_256_40seed.npy ./data/          # high-res GT
gsutil cp gs://ddpm-thesis-rh/flow-data/kmflow_sampled_data_irregnew.npz ./data/     # sparse input (6.3 GB)
```

<details><summary>figshare fallback (if GCS is unavailable)</summary>

- conditional ckpt: https://figshare.com/ndownloader/files/39184073 (rename → `conditional_ckpt_new.pth`)
- baseline ckpt: https://figshare.com/ndownloader/files/40320733
- high-res GT: https://figshare.com/ndownloader/files/39181919
- sparse input: https://figshare.com/ndownloader/files/39214622
</details>

Note: `stat_path: ./pretrained_weights/km256_stats.npz` appears in the config but the sparse-recons path
computes stats from the data itself (`load_recons_data` → `np.mean/std(ref_data[:-4])`), so a missing
`km256_stats.npz` should be harmless. If it errors, that's the culprit.

## 3. Config — `configs/kmflow_re1000_rs256_sparse_recons_conditional.yml`

Verify/set:
- `model.type: "conditional"` and `ckpt_path: "./pretrained_weights/conditional_ckpt_new.pth"`
- `data.data_dir` → `./data/kf_2d_re1000_256_40seed.npy`
- `data.sample_data_dir` → `./data/kmflow_sampled_data_irregnew.npz`
- `data.smoothing: True` (ships this way for the conditional — leave it)
- **`sampling.dump_arr: True`** ← REQUIRED; this is what writes the arrays we need
- `sampling.lambda_`: the released *baseline* config ships `0.` (Linear guidance OFF). For the
  conditional model the physics enters via the **conditioning input + classifier-free weight `w`
  (default 3.0 in `denoising_step.py`)**, not `lambda_`. Leave as shipped unless you deliberately want
  to stack Linear guidance on top.

## 4. Run

```bash
python main.py --config kmflow_re1000_rs256_sparse_recons_conditional.yml \
               --seed 1234 --sample_step 1 --t 240 --r 30
```

## 5. Sanity check before the long run (30 s, verified working)

```python
import sys, yaml, argparse, torch
from models.diffusion_new import Model, ConditionalModel
def d2n(d):
    ns=argparse.Namespace()
    for k,v in d.items(): setattr(ns,k, d2n(v) if isinstance(v,dict) else v)
    return ns
cfg = d2n(yaml.safe_load(open("configs/kmflow_re1000_rs256_sparse_recons_conditional.yml")))
m = ConditionalModel(cfg)
missing, unexpected = m.load_state_dict(
    torch.load("pretrained_weights/conditional_ckpt_new.pth", map_location="cpu")[-1], strict=False)
print(missing, unexpected)          # MUST both be empty
m.eval().cuda()
x, t, dx = torch.randn(1,3,256,256).cuda(), torch.tensor([100.]).cuda(), torch.randn(1,3,256,256).cuda()
with torch.no_grad(): print(m(x, t, dx).shape)   # -> (1, 3, 256, 256)
```
If `missing`/`unexpected` are non-empty, you loaded the wrong element — it must be **`[-1]` (EMA)**.

## 6. Output — what to collect

The dump lands under the config's `log_dir`
(`./experiments/kmflow_re1000_rs256_ddim_recons_conditional_log`). We need the arrays with keys:

```
reference        # ground truth,   physical, (4, 318, 256, 256, 3)
input            # sparse NN-fill, physical, (4, 318, 256, 256, 3)
pred_it0/1/2...  # predictions per SDEdit iteration, physical
```
This matches the existing `baratilab_results.pkl` exactly — **keep the same key names** so the comparison
notebook works unchanged. If the runner writes pieces, bundle them into one pickle with those keys.

## 7. Ship back

```bash
gsutil cp <your_dump>.pkl gs://ddpm-thesis-rh/baratilab_conditional_results.pkl
```
Use a **distinct name** — do **not** overwrite `baratilab_results.pkl` (the baseline). Then ping the
JAX-side instance; it pulls it and runs the physical-space comparison.

---

## Critical details for a like-for-like comparison

1. **Test split matches automatically.** `load_recons_data` slices `[-4:]` — the last 4 sequences —
   which are exactly our test seqs **36–39**. Don't change it.
2. **Their outputs are already physical** (the runner applies `scaler.inverse`). Ours are stored
   normalized → multiply by **std = 4.7988**. Physical MSE = normalized MSE × std², so the ranking is
   identical either way.
3. **Previously verified:** their `reference` == our de-normalized GT (max diff 0), and `input` is the
   same sparse+NN-fill. The *only* difference is the model.
4. **Triplet format** `(4,318,256,256,3)` = 4 seqs × 318 triplets × 256² × 3 consecutive frames; the
   **middle channel** is the reconstructed frame.
5. **They compute no energy spectrum** — only L2/RMSE and the PDE residual. All spectral/enstrophy
   analysis happens on our side.

## Known gotchas
- Branch **`main_v1`** (not `main`).
- **Load `[-1]` (EMA)**, not `[0]`.
- CUDA is hardcoded — needs a real GPU.
- Their `voriticity_residual` (sic) defaults to `re=1000.0, dt=1/32`; only relevant if `lambda_ > 0`.
- The conditional path in `guided_ddim_steps` calls the model **twice per step** — budget ~2× the baseline.
