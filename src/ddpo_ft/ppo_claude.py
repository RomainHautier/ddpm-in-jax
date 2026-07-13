"""Reference DDPO / PPO implementation for physics-reward finetuning (Claude reference).

Faithful DDPO (Black et al. 2023, "Training Diffusion Models with RL") on the SR diffusion base:
the denoising chain is treated as a multi-step MDP, the reward is terminal (r(x0) from the physics
Reward class), and the policy is the Gaussian reverse step. Optimized with a PPO clipped surrogate
over several inner epochs per collected batch, plus an optional KL-to-base penalty.

THE ALGORITHM (the two loops):
  OUTER (collect, frozen theta_old, no grad):
    1. rollout the reverse chain for a batch of (n_inputs * K) samples; STORE per step the state
       x_t, the ACTION taken a_t = x_{t-1}, and its log-prob log pi_{theta_old}(a_t | x_t).
    2. reward r(x0) -> per-INPUT advantages A = (r - mean_K)/std_K  (stop-gradient constant).
  INNER (optimize, current theta, several epochs on the SAME batch):
    3. recompute log pi_theta(a_t | x_t) for the STORED (x_t, a_t) — NO resampling; only the mean
       moves with theta (std = sqrt(beta_t) is fixed).
    4. ratio rho_t = exp(log pi_theta - log pi_{theta_old});  clipped surrogate
         L = -E_t[ min(rho_t A, clip(rho_t, 1-eps, 1+eps) A) ]  (+ kl_coef * KL(pi_theta||pi_base)).
    5. grad -> optimizer step. Repeat inner epochs.
  Then theta_old <- theta, resample. (Reusing the batch is why PPO is sample-efficient; the
  ratio+clip is what makes reusing off-policy data safe.)

WHAT MUST NOT HAPPEN: no backprop through sampling or through the reward. States, actions, old
log-probs and advantages are all CONSTANTS in the loss; the gradient flows only through the
recomputed log pi_theta(a_t | x_t).

Memory note: the trajectory (states+actions) is (T, B, N, N, 3). For T denoising steps this is
large — use few steps (SDEdit t_start ~ 50-150, or DDIM), a modest batch, and/or gradient
checkpointing. The rollout below stores the full trajectory for clarity.

Run as a script from repo root (hyphen dir blocks import); sys.path shim resolves both the repo and
the sibling `rewards_claude`.
"""
import os
import sys
from functools import partial

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax                                        # noqa: E402
import jax.numpy as jnp                           # noqa: E402
import numpy as np                                # noqa: E402
import optax                                      # noqa: E402

from rewards_claude import per_input_advantage    # noqa: E402
from src.physics_guidance import make_cond_func   # noqa: E402


# ============================================================ policy: the Gaussian reverse step

def gaussian_logprob(mean, std, x):
    """Per-sample log N(x; mean, std^2 I), summed over spatial+channel axes, batch axis KEPT.

    NB this fixes two bugs in model.calc_log_probs: scipy `scale` is the std (not std**2), and the
    sum must NOT collapse the batch (we need one log-prob per trajectory to form per-sample ratios).
    std is a scalar (sqrt(beta_t)); broadcasting handles it.
    """
    var = std ** 2
    logp = -0.5 * ((x - mean) ** 2 / var + jnp.log(2.0 * jnp.pi * var))
    return jnp.sum(logp, axis=(-3, -2, -1))       # (..., B) -> per-sample


def policy_mean_std(unet, params, x_t, t, alpha_bar, beta_schedule,
                    cond_func=None, cond_strength=0.0, temp=1.0):
    """The reverse-step policy pi(x_{t-1} | x_t, t): return (mean_theta(x_t, t), std_t).

    Identical DDPM posterior-mean formula as model.jit_denoise_step. Only `mean` depends on theta.
    `std = temp * sqrt(beta_t)`: `temp` is the SAMPLING TEMPERATURE — temp>1 widens the reverse-step
    Gaussian to make the K samples of an input more diverse (stronger advantage signal), applied
    CONSISTENTLY here (used by both rollout sampling and log-prob eval, so the PPO ratio stays valid).
    `t` is a Python/traced int; the batch shares it.
    """
    B = x_t.shape[0]
    t_arr = jnp.full((B,), t, dtype=jnp.int32)
    if cond_func is not None:
        condRes = cond_func(x_t)
        eps_c = unet.apply({"params": params}, x_t, t_arr, train=False, condRes=condRes)
        eps_u = unet.apply({"params": params}, x_t, t_arr, train=False)
        eps_pred = (1.0 + cond_strength) * eps_c - cond_strength * eps_u
    else:
        eps_pred = unet.apply({"params": params}, x_t, t_arr, train=False)
    alpha_bar_t = alpha_bar[t]
    alpha_t = 1.0 - beta_schedule[t]
    mean = (1.0 / jnp.sqrt(alpha_t)) * (x_t - (1.0 - alpha_t) / jnp.sqrt(1.0 - alpha_bar_t) * eps_pred)
    std = temp * jnp.sqrt(beta_schedule[t])
    return mean, std


# ============================================================ SDEdit start + collection (rollout)

def sdedit_start(x_cond, t_start, alpha_bar, key):
    """Noise the (normalized) conditioning field to timestep t_start: the SDEdit initial state x_T.
    x_cond: (B, N, N, 3) the sparse-nnfill input triplet, already tiled to B = n_inputs * K."""
    eps = jax.random.normal(key, x_cond.shape)
    return jnp.sqrt(alpha_bar[t_start]) * x_cond + jnp.sqrt(1.0 - alpha_bar[t_start]) * eps


def build_ddim_denoiser(unet, alpha_bar, t_start, n_steps):
    """Deterministic (eta=0) DDIM reverse chain t_start -> ~0 over `n_steps`, single chain — faithful to
    BaratiLab functions/denoising_step.ddim_steps (Hu et al-style accelerated schedule). Returns a jitted
    denoise(params, x_start) -> x0_pred, where x_start is the SDEdit-noised (to t_start) conditioning field
    and x0_pred is the model's reconstruction. For the base-DDIM finetuning init we run this with the FROZEN
    base params to seed the policy off a clean on-manifold reconstruction instead of the raw low-res."""
    raw = [int(round(t_start - i * (t_start - 1) / max(n_steps - 1, 1))) for i in range(n_steps)]
    seq = sorted({t for t in raw if t >= 1}, reverse=True)        # descending unique timesteps, >= 1
    t_cur = jnp.asarray(seq[:-1], dtype=jnp.int32)
    t_nxt = jnp.asarray(seq[1:], dtype=jnp.int32)
    t_last = int(seq[-1])

    def _x0(params, x, t):
        eps = unet.apply({"params": params}, x, jnp.full((x.shape[0],), t, jnp.int32), train=False)
        return (x - jnp.sqrt(1.0 - alpha_bar[t]) * eps) / jnp.sqrt(alpha_bar[t]), eps

    def denoise(params, x_start):
        def step(x, ts):
            tc, tn = ts
            x0, eps = _x0(params, x, tc)
            ab_n = alpha_bar[tn]
            return jnp.sqrt(ab_n) * x0 + jnp.sqrt(1.0 - ab_n) * eps, None   # eta=0 deterministic step
        x, _ = jax.lax.scan(step, x_start, (t_cur, t_nxt))
        x0, _ = _x0(params, x, t_last)                                      # final clean estimate
        return x0
    return jax.jit(denoise)


def build_rollout(unet, alpha_bar, beta_schedule, t_start, cond_func=None, cond_strength=0.0, temp=1.0):
    """Return a jitted rollout(params, x_start, key) -> (states, actions, logp_old, ts, x0) using
    lax.scan over the T=t_start reverse steps (no Python unrolling). Collection only (no grad);
    everything it returns is a constant for the loss. This is the ONLY place sampling happens.
    `temp` is the sampling temperature (see policy_mean_std)."""
    ts = jnp.arange(t_start, 0, -1, dtype=jnp.int32)       # [t_start, ..., 1]

    def rollout(params, x_start, key):
        def step(carry, t):
            x, k = carry
            k, sk = jax.random.split(k)
            mean, std = policy_mean_std(unet, params, x, t, alpha_bar, beta_schedule, cond_func, cond_strength, temp)
            z = jnp.where(t > 1, jax.random.normal(sk, x.shape), jnp.zeros_like(x))
            a = mean + std * z                             # sample x_{t-1}
            return (a, k), (x, a, gaussian_logprob(mean, std, a))
        (x0, _), (states, actions, logp) = jax.lax.scan(step, (x_start, key), ts)
        return states, actions, logp, x0                   # (T,B,...) stacked + final x0 (B,...)
    return rollout                                         # raw (caller jits or pmaps)


# ============================================================ PPO clipped surrogate loss

def build_loss(unet, alpha_bar, beta_schedule, t_start, clip_eps=0.2,
               cond_func=None, cond_strength=0.0, kl_coef=0.0, temp=1.0):
    """Return loss_fn(params, states, actions, logp_old, adv, base_params) -> scalar: the PPO
    clipped surrogate over the stored trajectory via lax.scan, with jax.checkpoint (remat) on the
    per-step UNet forward so backprop through T steps is memory-bounded (recompute-in-backward,
    without which T unrolled UNet activations would blow HBM).

    Gradient flows ONLY through the recomputed log pi_theta(a_t | x_t); adv/traj are constants.
    Optional KL-to-base (same-variance Gaussians): KL = ||mean_theta - mean_base||^2 / (2 std^2)."""
    ts = jnp.arange(t_start, 0, -1, dtype=jnp.int32)

    def loss_fn(params, states, actions, logp_old, adv, base_params):
        adv = jax.lax.stop_gradient(adv)

        def step(carry, d):
            x_t, a_t, lpo, t = d
            mean, std = policy_mean_std(unet, params, x_t, t, alpha_bar, beta_schedule, cond_func, cond_strength, temp)
            logp = gaussian_logprob(mean, std, a_t)                    # (B,) under current theta
            ratio = jnp.exp(logp - lpo)
            surr = jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv))
            kl = 0.0
            if kl_coef > 0.0:
                mean_b, _ = policy_mean_std(unet, base_params, x_t, t, alpha_bar, beta_schedule, cond_func, cond_strength, temp)
                kl = jnp.mean(jnp.sum((mean - mean_b) ** 2, axis=(-3, -2, -1)) / (2.0 * std ** 2))
            return carry, (surr, kl)

        _, (surrs, kls) = jax.lax.scan(jax.checkpoint(step), None, (states, actions, logp_old, ts))
        return -jnp.mean(surrs) + kl_coef * jnp.mean(kls)
    return loss_fn


# ============================================================ trainer (outer + inner loops)

class DDPOTrainer:
    """Drives DDPO finetuning of a DDPM policy against a physics Reward.

    ddpm: src.models.model.DDPM (provides .unet, .alpha_bar, .beta_schedule, .config).
    reward: rewards_claude.Reward (or any callable x0 -> (r, components)).
    optimizer: an optax optimizer (e.g. optax.adam(1e-5); consider a small lr + grad clipping).
    """

    def __init__(self, ddpm, params, reward, optimizer, group_size,
                 t_start=150, clip_eps=0.2, kl_coef=0.0, n_inner=4, seed=0,
                 cond_strength=0.0, sampling_temp=1.0, base_params=None, opt_state=None):
        # RESUME: pass `params`/`opt_state` from a checkpoint to continue, and `base_params` = the
        # ORIGINAL frozen base (ckpt_0299) so the KL anchor stays the true base, not the resumed model.
        self.ddpm = ddpm
        self.unet = ddpm.unet
        self.alpha_bar = ddpm.alpha_bar
        self.beta_schedule = ddpm.beta_schedule
        self.reward = reward
        self.optimizer = optimizer
        self.params = params
        self.base_params = base_params if base_params is not None else jax.tree_util.tree_map(lambda a: a, params)
        self.opt_state = opt_state if opt_state is not None else optimizer.init(params)
        self.group_size = group_size              # K samples per input
        self.t_start = t_start
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.n_inner = n_inner
        self.cond_strength = cond_strength
        self.key = jax.random.PRNGKey(seed)
        # conditioning signal (if the base model uses it); None for the plain unconditional base
        cfg = ddpm.config
        self.cond_func = None
        if cfg["conditioning"]["train"]["enabled"] or cfg["conditioning"]["inference"]["enabled"]:
            self.cond_func = make_cond_func(
                cfg["conditioning"]["train"].get("cond_signal", "gradient"),
                n=256, re=cfg["conditioning"]["train"]["re"])

        self.sampling_temp = sampling_temp
        # ---- DATA PARALLEL across all devices (pmap). Each device runs whole input-groups; the
        # gradient is averaged across devices (pmean) => one effective batch of n_dev * per_dev_B.
        self.n_dev = jax.local_device_count()
        self._sqrt_ab = float(jnp.sqrt(self.alpha_bar[t_start]))
        self._sqrt_1mab = float(jnp.sqrt(1.0 - self.alpha_bar[t_start]))
        rollout = build_rollout(self.unet, self.alpha_bar, self.beta_schedule,
                                t_start, self.cond_func, cond_strength, sampling_temp)
        loss_fn = build_loss(self.unet, self.alpha_bar, self.beta_schedule, t_start,
                             clip_eps, self.cond_func, cond_strength, kl_coef, sampling_temp)
        reward, K, optimizer = self.reward, group_size, optimizer

        def reward_adv(x0):                                   # per-device: reward + per-input advantage
            r, comps = reward(x0)
            return r, per_input_advantage(r, K), comps

        def update(params, opt_state, states, actions, logp_old, adv, base_params):
            loss, grads = jax.value_and_grad(loss_fn)(params, states, actions, logp_old, adv, base_params)
            grads = jax.lax.pmean(grads, "dev")              # data-parallel gradient average
            loss = jax.lax.pmean(loss, "dev")
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        self._p_rollout = jax.pmap(rollout, axis_name="dev")
        self._p_reward_adv = jax.pmap(reward_adv, axis_name="dev")
        self._p_update = jax.pmap(update, axis_name="dev")
        # single-device deterministic (temp=1) rollout for the live GT-retention probe
        self._eval_rollout = jax.jit(build_rollout(self.unet, self.alpha_bar, self.beta_schedule,
                                                   t_start, self.cond_func, cond_strength, 1.0))
        # replicate params / opt_state / frozen base across devices
        devs = jax.local_devices()
        self.params = jax.device_put_replicated(self.params, devs)
        self.base_params = jax.device_put_replicated(self.base_params, devs)
        self.opt_state = jax.device_put_replicated(self.opt_state, devs)

    def _next_key(self):
        self.key, k = jax.random.split(self.key)
        return k

    def probe_x0(self, x_cond, key, params=None):
        """Deterministic (temp=1) reconstruction of x_cond (B,N,N,3) with the CURRENT model (one
        replica) — for the live GT-retention probe. params=None uses self.params[0]; pass base
        params to get the base reference."""
        p = params if params is not None else jax.tree_util.tree_map(lambda a: a[0], self.params)
        k1, k2 = jax.random.split(key)
        x_start = self._sqrt_ab * x_cond + self._sqrt_1mab * jax.random.normal(k1, x_cond.shape)
        *_, x0 = self._eval_rollout(p, x_start, k2)
        return x0

    def train_iter(self, x_cond):
        """One OUTER iteration, DATA-PARALLEL. x_cond: (n_inputs, N,N,3) with n_inputs divisible by
        n_dev. Each device gets n_inputs/n_dev inputs (tiled K times), rolls out under theta_old,
        computes its own per-input advantage, and contributes to the pmean'd gradient => effective
        batch = n_inputs * K. Returns a metrics dict."""
        nd = self.n_dev
        ipd = x_cond.shape[0] // nd                                    # inputs per device
        xc = x_cond.reshape(nd, ipd, *x_cond.shape[1:])               # (nd, ipd, N,N,3)
        xc = jnp.repeat(xc, self.group_size, axis=1)                  # (nd, ipd*K, N,N,3) input-major/device
        noise = jax.random.normal(self._next_key(), xc.shape)
        x_start = self._sqrt_ab * xc + self._sqrt_1mab * noise        # SDEdit start, per device
        keys = jax.random.split(self._next_key(), nd)                 # (nd, 2) one rollout key per device
        states, actions, logp_old, x0 = self._p_rollout(self.params, x_start, keys)
        r, adv, comps = self._p_reward_adv(x0)                        # (nd, ipd*K) each
        losses = []
        for _ in range(self.n_inner):
            self.params, self.opt_state, loss = self._p_update(
                self.params, self.opt_state, states, actions, logp_old, adv, self.base_params)
            losses.append(float(loss[0]))                            # pmean'd -> identical per device
        r_np = np.asarray(r).reshape(-1); adv_np = np.asarray(adv).reshape(-1)
        return {
            "reward_mean": float(r_np.mean()), "reward_std": float(r_np.std()),
            "adv_abs_mean": float(np.abs(adv_np).mean()),
            "loss_first": losses[0], "loss_last": losses[-1],
            "components": {k: float(np.asarray(v).mean()) for k, v in comps.items()},
        }

    def train(self, input_iterator, n_outer, log_every=1):
        """input_iterator yields (n_inputs, N, N, 3) normalized sparse-input batches."""
        history = []
        for outer in range(n_outer):
            x_cond = next(input_iterator)
            m = self.train_iter(x_cond)
            history.append(m)
            if outer % log_every == 0:
                comp = "  ".join(f"{k}={v:.3f}" for k, v in m["components"].items())
                print(f"[{outer:04d}] R={m['reward_mean']:.3f}±{m['reward_std']:.3f} "
                      f"|A|={m['adv_abs_mean']:.2f} loss {m['loss_first']:.3f}->{m['loss_last']:.3f}  {comp}",
                      flush=True)
        return history


# ============================================================ wiring example (not executed here)
# from src.models.model import DDPM
# from src.utils import load_checkpoint
# from src.sequence_inference import load_sequence, sparse_nnfill_degrade, build_triplets
# from rewards_claude import Reward
# import yaml
#
# cfg = yaml.safe_load(open("configs/config.yaml"))
# ddpm = DDPM(cfg)
# params, _, _ = load_checkpoint("gs://ddpm-thesis-rh/checkpoints/ddpm/ckpt_epoch_0299.pkl")
# reward = Reward.from_calibration("base_results/regime_stats_re1000.npz",
#                                  "base_results/reward_calibration.json", re=1000)
# opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-5))
# trainer = DDPOTrainer(ddpm, params, reward, opt, group_size=8, t_start=150, kl_coef=0.01, n_inner=4)
#
# def input_iter():                       # yields (n_inputs, 256, 256, 3) normalized sparse triplets
#     while True:
#         seq = load_sequence("flow-data/kf_2d_re1000_256_40seed.npy", 32)
#         nn = sparse_nnfill_degrade(seq, 32)
#         tri = build_triplets(nn, mean=0.0, std=4.7988)       # (n, 256, 256, 3)
#         yield jnp.asarray(tri[:4])                            # 4 inputs -> 4*K samples
#
# trainer.train(input_iter(), n_outer=500)
