# Physics-conditioning (learned residual adapter) — open problems

Status: architecture + training wiring drafted in `src/models/model.py` (`ConditionalUnet`)
and `src/train_ddpm.py` (`build_params`, `build_optimizer`, `make_steps`). Config lives under
`conditioning.train.*` in `configs/config.yaml`. The config-key paths in the code are now
consistent. Remaining work below.

## Blockers — RESOLVED

1. **`load_checkpoint` 3-tuple unpack — DONE** (train_ddpm.py:50 now
   `pretrained_params, _, _ = load_checkpoint(...)`). ✅
2. **`DDPM.__init__` model branch — DONE** (model.py:274 picks `ConditionalUnet`/`Unet`). ✅

## Correctness / robustness

3. **`pretrained_ckpt` emptiness check is the wrong sentinel.** `build_params` uses
   `... is not None`, but the empty case is the string `""` (see `conditioning.inference`),
   which is `not None`. Use truthiness: `if cond and cfg['conditioning']['train']['pretrained_ckpt']:`.

4. **Verify merge + freeze actually work** before a long run:
   - after `{**init, **pretrained}`: assert every pretrained base key exists in init with matching
     shape (catches Flax auto-name drift). Extras should be exactly `cond_in/cond_hidden/cond_combine`.
   - after 1 train step: assert a base leaf (e.g. `Conv_0/kernel`) is unchanged and a `cond_` leaf moved.
   - confirm identity init: with merged params and `condRes=None`, conditional output == base `Unet` output.

5. **Conditional-only script.** `make_steps`/`train_step` always pass `condRes`, so `enabled: false`
   would still try to condition (and hit blocker 2). Either guard the `apply` on `cond_func is not None`
   (the `**kwargs` trick) to make the flag truly toggle both modes, or document this as the
   conditional trainer only. Currently `cond_func` is also built unconditionally (train_ddpm.py:221).

## Inference — `DDPM.sample`

6b. CFG formula direction — DONE (model.py:327, `(1+s)*eps_cond - s*eps_uncond`). ✅
6c. Branch on `enabled` (Python `if`, not `lax.cond`) — DONE (model.py:323-329). ✅
6a-`re`. `re` config path — DONE (model.py:301). ✅
6d. `std`/`mean` threaded as args, strength named `cond_strength` (no `w` clash) — DONE. ✅

6a-`cond_strength` — DONE. Config key added (`conditioning.inference.cond_strength: 0`, matches
    their inference `w=0`) and read with the correct path (model.py:302). ✅

    Reference (BaratiLab main_v1): inference `guidance_weight: 0.0` in all conditional configs
    (pure conditional; the `kwargs.get('w', 3.0)` default is overridden). Training uses no strength,
    only unconditional-dropout `p=0.1` (== our `conditioning.train.proba`).

OPTIONAL cleanup (not a crash): `cond_strength` (line 302), `re` (301) and `cond_func` (304) are
    read/built OUTSIDE the `if enabled` branch — wasted work when conditioning is disabled.
    Move them inside `if enabled:`. Also: at `cond_strength=0` the `eps_uncond` pass is dead weight
    (only needed if sweeping strength > 0).

7. **`eval_step`** currently runs unconditional (no `condRes`). Optionally pass
   `condRes=cond_func(noised)` (no dropout) for a like-for-like conditional eval.

## TODO — full-model re-training (nothing frozen)

Already supported via config: set `conditioning.train.freeze_base: false`. Then `build_optimizer`
skips the `multi_transform`/`set_to_zero` freeze and returns a plain `optax.adam` over ALL params,
so the base fine-tunes alongside the `cond_*` adapter. The freeze asserts in `train()` are guarded
by `freeze_base`, so they're skipped automatically.

To make it a first-class mode, still to consider:
- **LR**: full fine-tuning the whole UNet at lr=2e-4 (adapter LR) may be too high for the base —
  add a separate `full_finetune_lr` (or lower the base LR via a param-group / multi_transform with
  two adam instances) so the base drifts gently while the adapter learns faster.
- **run_name**: use a distinct one (e.g. `conditioned_full_finetune`) so it doesn't overwrite the
  frozen-base checkpoints.
- **Start point**: decide whether to still load the pretrained base (`pretrained_ckpt`) and fine-tune,
  or train the whole conditional model from scratch (no ckpt) — both work; former is cheaper.
- The residual probe + plot already work in this mode (guarded by `enabled`, not `freeze_base`).

## TODO — unify the inference / posterior-sampling code

Right now the reverse (posterior) sampling loop is reimplemented in several places that have
drifted apart: `DDPM.sample` (src/models/model.py), `make_batched_sampler` (src/sequence_inference.py),
and `plain_ddpm_inference.py`. Consolidate into ONE jitted posterior sampler with flags:
- **DDPM vs DDIM** sampling (stochastic ancestral vs deterministic DDIM; DDIM enables far fewer steps).
- **single vs multi (K-iteration) refine loop** — the outer noise->denoise repeat used in sequence
  reconstruction (K, S schedule) vs a one-shot chain; make K/S a parameter, K=1 the default.
- **learned + linear guidance** — already disentangled in `make_batched_sampler` (raw `dx`;
  `condRes` for learned, `lambda*dx` subtracted for linear); reuse that as the single source.
- **batched / sharded** core, with a thin sequence wrapper (chunking frames, per-chunk probe) on top.
Goal: one `denoise_step` + one driver, callers differ only by config/flags — no copy-pasted reverse
loops. Keep the static-flag pattern (Python bools resolved at trace time) so nothing re-traces per step.

## TODO — DDPO finetuning for generalization to unseen behaviours

Implement DDPO (Denoising Diffusion Policy Optimization, Black et al. 2023) to RL-finetune the
diffusion model: treat the reverse chain as a policy, reward samples by a task metric (e.g. LOW PDE
residual at the target Re, or spectrum match), and do policy-gradient updates over the denoising
trajectory. Aim: push the model to generalize to unseen regimes (OOD Re / flows) beyond what the
frozen-base + conditioning adapter achieves. Notes for later:
- Reward = `-make_residual_loss(...)` (per-sample, differentiable-free is fine — DDPO is RL, not
  backprop-through-sampling) at the target Re; optionally combine with energy-spectrum match.
- Needs trajectory logprobs from the (DDIM) sampler -> another reason to land the unified sampler first.
- Separate config block (`ddpo:`) + run_name; can finetune either the full model or just the adapter.

## Sequence inference — add the LEARNED (adapter) path (src/sequence_inference.py)

Today `make_batched_sampler.denoise_step` calls `model.apply(..., train=False)` with NO `condRes`
(line ~79), so a conditional checkpoint runs UNCONDITIONALLY. The only physics is the LINEAR
`dx`-subtraction (`dx = dx_func(xT)` subtracted after the step, lines ~101-104). Need to add the
learned path.

KEY IDEA — two ORTHOGONAL guidance switches, each resolved ONCE (static) and applied at a DIFFERENT
site. Don't tangle them into one nested if:
  - LEARNED (adapter): changes how `eps_pred` is computed, INSIDE `denoise_step` (before the reverse
    step). Switch = `learned` (config `conditioning.train.enabled`). Weight = `cond_strength`.
  - LINEAR (gradient descent): subtracts `dx` AFTER the reverse step, in `sample_batch`. Switch =
    `guided = dx_func is not None`. Weight baked into `dx_func`'s `lam`.
  They compose freely: {learned only, linear only, both, neither}.

Changes:
1. In `run_sequence_inference`: build a CONDITIONING function (separate from the linear `dx_func`):
   `cond_func = make_dx_func(n=256, re=cfgs[0]['conditioning']['train']['re'], std=std, mean=mean, lam=1.0)`
   only when `conditioning.train.enabled`. Pass it into `make_batched_sampler` (new `cond_func=None`
   arg, parallel to `dx_func`). The loaded `sd['checkpoint']` MUST be the conditional adapter ckpt.
2. In `make_batched_sampler` (already takes `config`): read ONCE, as static Python values:
   `learned = config['conditioning']['train']['enabled']`
   `cond_strength = config['conditioning']['inference']['cond_strength'] if learned else 0.0`
   (add `cond_func` param).
3. In `denoise_step` (jitted; `learned`/`cond_strength`/`cond_func` are closed-over statics, so the
   `if learned` resolves at trace time — no TracerBool issue):
       tt = jnp.full((b,), t)
       if learned:
           condRes = cond_func(xT)
           eps_c = model.apply({"params": params}, xT, tt, train=False, condRes=condRes)
           eps_u = model.apply({"params": params}, xT, tt, train=False)          # condRes=None
           eps_pred = (1 + cond_strength) * eps_c - cond_strength * eps_u
       else:
           eps_pred = model.apply({"params": params}, xT, tt, train=False)
   Leave the LINEAR `dx`-subtraction in `sample_batch` exactly as-is (guarded by `guided`).

Notes:
- `cond_strength` semantics match DDPM.sample: 0 = pure conditional, -1 = unconditional (cancels),
  >0 = CFG-amplified. For a clean adapter run use `guidance_lambda = 0` (linear off).
- For OOD at Re=X, both `conditioning.train.re` (learned) and `guidance_re` (linear, if used) should
  be X so the residual injects the target physics.
- The `n=256` in the two `make_dx_func` calls is hardcoded; fine for now (image_size=256).
