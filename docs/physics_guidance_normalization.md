# Physics-guidance normalization convention — consistency note

A reference for verifying that our JAX physics guidance matches BaratiLab's, specifically the
**normalization order** around the PDE-residual gradient. Use the checklist in §4 to self-check.

> **The convention, in one line:** `de-normalize → residual → derivative (w.r.t. physical field) →
> divide by scale`. The residual is **never** normalized before the derivative; the only
> normalization is `÷ scale` (= ÷std) applied to the gradient **afterwards**.

---

## 1. BaratiLab references (repo `main_v1`)

Both their sampling ("Linear") and training ("Learned"/conditional) paths use the same order.

**Residual + derivative** — `train_ddpm/functions/losses.py`
([`voriticity_residual`](https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/train_ddpm/functions/losses.py#L5-L52)),
mirrored in `runners/rs256_guided_diffusion.py`
([L199-L247](https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/runners/rs256_guided_diffusion.py#L199-L247)):
```python
def voriticity_residual(w, re=1000.0, dt=1/32):
    w.requires_grad_(True)                       # L9  — derivative is w.r.t. THIS field (physical)
    ...
    residual = wt + (advection - (1/re)*wlap + 0.1*w) - f
    residual_loss = (residual**2).mean()         # L51 — squared (L2) residual
    dw = torch.autograd.grad(residual_loss, w)[0] # L52 — ∂L/∂w_phys
    return dw
```

**Sampling (Linear)** — `runners/rs256_guided_diffusion.py`
[L394-L397](https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/runners/rs256_guided_diffusion.py#L394-L397):
```python
physical_gradient_func = lambda x: \
    voriticity_residual(scaler.inverse(x))[0] / scaler.scale() * self.config.sampling.lambda_
#                       ^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^
#                       de-normalize first      ÷ scale AFTER      × lambda (sampling only)
```
applied in `functions/denoising_step.py` (`ddpm_steps`/`ddim_steps`): `dx = dx_func(x)` on the
pre-step field, then `sample = posterior_mean + noise - dx` — **subtracted from the sample**.

**Training (Learned/conditional)** — `train_ddpm/functions/losses.py`
[L70-L90](https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/train_ddpm/functions/losses.py#L70-L90):
```python
x = x0*a.sqrt() + e*(1-a).sqrt()                 # noised sample, NORMALIZED space
if flag < p:                                     # p=0.1 classifier-free dropout
    output = model(x, t.float())
else:
    dx = voriticity_residual((x*x_scale + x_offset)) / x_scale   # L85 — de-norm → resid → grad → ÷scale
    output = model(x, t.float(), dx)             # dx is a CONDITIONING INPUT, not added to output
return (e - output).square().sum((1,2,3)).mean() # L88-90 — PLAIN noise L2; NO lambda in training
```

---

## 2. Our JAX references (this repo)

**Residual + derivative + ÷std** — [`src/physics_guidance.py:60-75`](../src/physics_guidance.py#L60-L75):
```python
def make_dx_func(n=256, re=1000.0, dt=1/32, std=4.7988, mean=0.0, lam=1.0, ...):
    residual = make_ns_residual(n, re, dt, dtype)        # operates on PHYSICAL field
    def residual_loss(w):
        return jnp.mean(residual(w) ** 2)                # == (residual**2).mean()
    grad_w = jax.grad(residual_loss)                     # ∂L/∂w_phys
    def dx(x_norm):
        w = x_norm * std + mean                          # de-normalize (= scaler.inverse)
        return lam * grad_w(w) / std                     # ÷ scale AFTER, × lambda
    return jax.jit(dx)
```
The residual formula itself ([`make_ns_residual`, :44-57](../src/physics_guidance.py#L44-L57)) matches
`voriticity_residual` term-for-term: `wt + (advection - (1/Re)*wlap + 0.1*w) - f`, `f = -4cos(4y)`,
no dealiasing, integer `fftfreq` wavenumbers on `(0,2π)²`.

**Application (Linear, sampling)** — [`src/sequence_inference.py:101-104`](../src/sequence_inference.py#L101-L104):
```python
dx = dx_func(xT) if (guided and ...) else None   # on the pre-step field
xT = denoise_step(params, xT, t, z)              # DDPM ancestral step
if dx is not None: xT = xT - dx                  # subtract from the SAMPLE
```

We implement the **Linear** variant only (no retraining); we do **not** train the conditional model.

---

## 3. The subtlety worth understanding (the `÷std`)

The derivative is taken w.r.t. the **physical** field `w`, giving `∂L/∂w_phys`. BaratiLab then divide
by `scale` (=std). Note this is **not** the chain-rule gradient w.r.t. the *normalized* variable:

```
∂L/∂x_norm = ∂L/∂w_phys · ∂w_phys/∂x_norm = ∂L/∂w_phys · std          (the "pure" normalized gradient)
BaratiLab dx (per unit λ) = ∂L/∂w_phys / std                          (their convention)
```

They differ by a factor `std²`. So `dx` is the *physical-space* gradient rescaled by `1/std`, **not**
the gradient of the loss w.r.t. the input the model/sampler actually steps in. This is a deliberate
scaling convention — it simply re-defines what a given `λ` means. **Because we copy `÷std` exactly,
our `λ` is in the same units as theirs** (so a `λ` from their configs would transfer; theirs ship with
`lambda_: 0.`, so there is no tuned value to copy — see `docs/` discussion). The important thing for
consistency is not which convention is "correct" but that **both sides use the same one**, which they
do.

---

## 4. Self-check checklist

Verify each holds in both BaratiLab and our code:

- [ ] Residual is computed on the **de-normalized (physical)** field (`scaler.inverse(x)` /
      `x*x_scale+x_offset` / `x_norm*std+mean`), **not** the normalized field.
- [ ] The autodiff derivative is taken **w.r.t. that physical field** (`requires_grad_(True)` on the
      physical `w` / `jax.grad(residual_loss)` of `residual_loss(w_phys)`).
- [ ] The objective is the **squared** residual, mean-reduced: `(residual**2).mean()`.
- [ ] The gradient is divided by **scale (=std)** *after* differentiation (`/ scaler.scale()` /
      `/ std`), and the residual is **never** normalized *before* the derivative.
- [ ] `λ` multiplies the gradient **only at sampling time**; there is **no `λ` in training**.
- [ ] At sampling, `dx` is **subtracted from the sample** (`sample = mean + noise - dx`), **not** added
      to the ε-prediction. (i.e. the noise-prediction loss stays `L2(e - eps_pred)`; physics is a
      separate step.)
- [ ] (Learned variant only — not implemented here) `dx` enters as a **conditioning input**
      `model(x, t, dx)` trained with plain `L2(e - output)` and `p=0.1` classifier-free dropout.

Conclusion of the current audit: **all boxes hold; our Linear implementation is faithful, including the
post-hoc `/std`.**
