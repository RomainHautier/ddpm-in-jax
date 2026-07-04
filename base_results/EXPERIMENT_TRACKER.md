# Experiment tracker — physics-diffusion for turbulent flow reconstruction

Living index of what we've run, on which data/sequences, where results live, and what's queued.
**Legend:** ✅ complete · 🟡 in progress · ⬜ planned / not started

Three guidance paradigms recur:
- **none** — base DDPM, plain SDEdit sparse reconstruction (K=3, S=[150,100,50]).
- **linear** — BaratiLab/Shu et al. sampling-time physics guidance: subtract `λ·∇_w[mean(residual²)]/std`
  each reverse step (no retraining). λ-swept. (Verified verbatim = our `make_dx_func`.)
- **learned** — trained `cond_*` adapter that ingests a residual signal as `condRes`. Two signals:
  `gradient` (`make_dx_func`, ∇ of mean squared residual — the linear signal) and `field`
  (`make_field_func`, the raw residual field, ENS-style arXiv:2606.27354).

All conditional models are frozen-base-or-full-finetune adapters on the **Re=1000** base. Model
normalization: mean 0, std 4.7988 (Re=1000 first-32-seq train split). Residual evaluated at the
**data's** Re.

---

## 1. Models / checkpoints  (`gs://ddpm-thesis-rh/checkpoints/ddpm/`)

| Model | cond signal | scope | Epochs | Checkpoint | Status |
|:---|:---|:---|---:|:---|:---:|
| Base DDPM (unconditional) | — | — | 300 | `ckpt_epoch_0299.pkl` | ✅ |
| Learned adapter (ref) | gradient | frozen | 30 | `conditioned_frozen_base/ckpt_epoch_0029.pkl` | ✅ |
| Learned adapter | gradient | frozen | 60 | `conditioned_frozen_base_60ep/ckpt_epoch_0059.pkl` | ✅ |
| Full-finetune | gradient | full | 60 | `conditioned_full_finetune/ckpt_epoch_0059.pkl` | ✅ |
| Field adapter (ENS) | field | frozen | 60 | `conditioned_field_cond_60ep/ckpt_epoch_0059.pkl` | ✅ |
| Field full-finetune (ENS) | field | full | 60 | `conditioned_field_full_finetune/ckpt_epoch_0059.pkl` | ✅ |

**Signal × scope matrix (all real kf_2d Re=1000, 60ep):**

| cond signal | frozen (partial) | full-finetune |
|:---|:---|:---|
| **gradient** (∇ mean residual²) | ✅ 30ep + 60ep | ✅ 60ep |
| **field** (ENS raw residual) | ✅ 60ep | ✅ 60ep |

## 2. Data assets (`gs://ddpm-thesis-rh/flow-data/`)

| Dataset | Re | Seqs×Frames | Dist | Path | Used as |
|:---|---:|:---|:---:|:---|:---|
| kf_2d_re1000_256_40seed.npy | 1000 | 40×320 | in | `flow-data/…` | train (32) + val (33-35... see below) + GT |
| generated_kf/kf_re500_256_20seed.npy | 500 | 20×320 | OOD | `flow-data/generated_kf/…` | OOD GT (seqs 0-7) |
| generated_kf/kf_re2000_256_20seed.npy | 2000 | 20×320 | OOD | `flow-data/generated_kf/…` | OOD GT (seqs 0-7) |
| kmflow_idx_lst.npz | — | 40×1024 | — | `flow-data/…` (327 KB, **on GCS**) | sparse-sample mask for degradation |

**Sequence splits (Re=1000 40-seq set):** train = 0–31, **val = 32–35**, **test = 36–39** (32/4/4).
All in-dist inference uses **val+test = seqs 32–39** (8 seqs). OOD sets use **seqs 0–7** (8 seqs).

**Degradation for reconstruction:** `sparse_nnfill` — keep the 1024 masked pixels (per `kmflow_idx_lst.npz`)
+ nearest-neighbour fill; the model reconstructs from that sparse input (BaratiLab task).

---

## 3. Results

### 3a. Base model (ckpt_0299) — no retraining ✅
| Experiment | Guidance | Data | Re | Where |
|:---|:---|:---|---:|:---|
| Sparse recon + λ-sweep | none/linear | real | 1000 | `results_summary.md` (λ=0: MSE 0.1426) |
| Linear-guidance λ-sweep | linear | real/gen | 500/1000/2000 | `lambda_sweep_*`, `lambda_sweep_metrics*.ipynb` |
| Cross-Re summary | none/linear | both | all | `cross_re_summary.ipynb` |
| BaratiLab comparison | — | — | 1000 | `baseline/compare_models.ipynb` |

**Key base finding:** raw MSE degrades with Re; the real failure is a **high-wavenumber energy deficit
that grows with Re** (over-smoothing) — an L2-objective limitation.

### 3b. In-dist learned-only sequence inference — Re=1000, seqs 32–39 (2544 frames) ✅

Task: sparse-recon (`sparse_nnfill`, K=3, S=[150,100,50]), `guidance_lambda=0` (no linear), `cond_strength=0`.
Reconstructions: `monitoring/sparse_reconstructions/sequence_reconstruction_indist_re1000_<tag>_seq{32-39}.pkl`.
Analysis: `learned_finetune_comparison.ipynb`. Residual = mean NS residual² (GT=12.9); `hi-k ret` = frac of
GT small-scale (k≥32) energy retained.

| model | signal | scope | MSE | residual | res/GT | hi-k ret |
|:---|:---|:---|--:|--:|--:|--:|
| GT | — | — | — | 12.9 | 1.00 | 1.00 |
| grad_frozen_30ep | gradient | frozen | 0.1436 | 28.9 | 2.24× | 0.37 |
| grad_frozen_60ep | gradient | frozen | 0.1436 | 28.3 | 2.20× | 0.37 |
| grad_full_60ep | gradient | full | 0.1418 | 31.9 | 2.48× | 0.36 |
| field_frozen_60ep | field | frozen | 0.1438 | 30.3 | 2.36× | 0.37 |
| field_full_60ep | field | full | **0.1411** | 28.7 | 2.23× | 0.35 |
| base (unconditional) | — | — | *(pkls done — add to table)* | | | |

**Finding (in-dist):** MSE flat (~0.142–0.144, floor). On PDE residual **none wins** — all ~2.2–2.5× GT;
**gradient-frozen best (2.20×), gradient full-finetune worst (2.48×)** (unfreezing *hurt* physics).
**Field (ENS) ≈ gradient.** hi-k ~0.36 everywhere — over-smoothing universal. ⇒ win (if any) not in-dist.

### 3c. OOD learned-only inference — Re=500 & Re=2000, seqs 0–7 🟡

Same task; conditional models run with `conditioning.inference.re = target Re` so the residual signal
encodes the target-Re physics (how the model is "told" the Re). Base runs unconditionally. Driver:
`run_ood_inference_queue.py` (resumable — skips tags already on GCS).
Reconstructions: `…/sequence_reconstruction_ood_re{500,2000}_<model>_seq{0-7}.pkl`.

| model | Re=500 | Re=2000 |
|:---|:---:|:---:|
| base (unconditional) | ✅ 8/8 | ⬜ |
| grad_frozen60 | ✅ 8/8 | ⬜ |
| grad_full60 | ✅ 8/8 | ⬜ |
| field_frozen60 | ✅ 8/8 | ⬜ |
| field_full60 | 🟡 4/8 (running) | ⬜ |

Also done: **plain base in-dist** (Re=1000, seqs 32–39) — the missing baseline for §3b.
**Pending analysis:** OOD comparison notebook (conditional vs plain base, per Re) — does the learned
residual mapping improve OOD generalisation? (Not yet built.)

---

## 4. In-flight / queue (TPU = one job at a time)
- 🟡 **now** — OOD inference (`run_ood_inference_queue.py`): finishing `field_full60@Re500`, then the
  5-job **Re=2000 block** (base + 4 conditional). Resumable across env resets (checks GCS).
- ⬜ **next** — OOD comparison notebook (per-Re: base vs 4 conditional, residual/MSE/spectrum/vorticity).
- ⬜ **free levers** (no retrain): `cond_strength>0` sweep; learned+linear together.
- ⬜ **later rungs** — explicit residual-loss finetune; DDPO.

## 5. Reproduce / rerun
- In-dist inference: `run_learned_inference_queue.py` (4 cond models, seqs 32–39).
- OOD + base inference: `run_ood_inference_queue.py` (base + 4 cond, Re 500/2000 seqs 0–7, + base in-dist).
- Analysis: `learned_finetune_comparison.ipynb` (in-dist). Both notebooks are CPU-only (numpy), stream
  pkls from `monitoring/sparse_reconstructions/` (mirror on GCS), skip missing → re-runnable anytime.
- **Env note:** `base_results/`, `monitoring/`, `flow-data/` are local scratch and get wiped on env
  reset — all recoverable from GCS. Reconstruction pkls mirror to `monitoring/sparse_reconstructions/`.

## 6. Metric conventions
`MSE` recon-vs-GT (normalized) · `residual` = mean NS residual² at the data's Re · `res/GT` = ×GT ·
`hi-k ret` = Σ_{k≥32} E(k) model / GT. Residual normalized at model `std=4.7988`.
