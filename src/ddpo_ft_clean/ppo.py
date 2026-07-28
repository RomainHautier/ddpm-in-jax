# Replicate the implementation of the DDPO paper.
# Probably a denoising re-implementation, storing the parameters of the previous
# distribution and some clipping

import functools

import jax
import jax.numpy as jnp
from jax.scipy import stats
import distrax
import optax
import numpy as np

## 1. Sampling denoising trajectories to then use as offline dataset: done
## 2. Function to calculate the advantage - done
## 2.bis. add assertions in the sample trajectories function to make sure that inputs and outputs
## are the right shape.
## 3. Function to calculate the loss - to do
## 4. Optimizer steps - to do
## 5. training loop - to do
## 6. initialisation of the class with - to do


class PPO():
    """This class implements proximal policy optimization to be used
    for downstream finetuning of a diffusion model (hence for DDPO).

    What this class needs to be usable:

        - Requires the following inputs:
            - Batch of low/res samples
            - params of the model

        - Run a batched sampling process (re-implement or take from a previous pre-defined sampling function)

        - Calculate the PPO loss
        - Update the parameters of the network K times in a batch of size B

        - How to shard and use all TPU chips available?
            --> to shard here probably is the first sampling process?
            --> when running the K optimisation steps, say one per 4 samples in a batch of 16, we can also shard that forward process
            The second forward process is not sampling, it is a log prob evaluation of the action under the policy. Then that we backprop
            through each policy chain, accumulate the gradients from the chips and then apply the opt step?


        Here are the functions required:

        - denoising sampling function (jitted): if re-using old function,
        add the possibility of storing log_probs when sampling.
        - loss function (jitted)
        - forward pass to evaluate the log probs under the new weights --> not sampling
        - optimiser step



        The class should init the model a single time, meaning that it should also
        have a fully inbuilt training step which then performs multiple epochs
    """

    def __init__(self, ppo_config: dict, ddpm, params, reward_fn):

        # Initialising the model architecture & the noise schedule. 
        self.ddpm = ddpm
        self.model = ddpm.unet
        self.alpha_bar = ddpm.alpha_bar
        self.beta_schedule = ddpm.beta_schedule

        # T denoising steps
        self.T = ppo_config['T']

        # Initialising the timesteps at which to denoise (currently only supports DDPM)
        self.t_vec = jnp.arange(self.T, 0, -1)

        # checking that denoising starts at the correct timestep
        assert self.T == self.t_vec[0], f"start t {self.t_vec[0]} vs self.T {self.T}"
        assert self.t_vec[-1] == 1, f"t_vec[-1] = {self.t_vec[-1]}"
        assert self.T < len(self.alpha_bar)

        # M samples to denoised per low-res input.
        self.M = ppo_config['M']

        # Add the reward function as a class attribute.
        self.reward_fn = reward_fn

        # Init the optimizer
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate = ppo_config['lr']))

        ## PPO optimizes the following objective
        ## sum_0^T ( p_theta(x_t-1|x_t) / p_theta_old(x_t-1|x_t) * delta log p_theta(x_t-1|x_t) * r(x_0, c) )
        ## using the log derivative trick:
        ## sum_0^T (  delta p_theta(x_t-1|x_t) / p_theta_old(x_t-1|x_t) * r(x_0, c) )
        ## The objective is minimized by taking the derivative of the above loss,
        ## with both p_theta_old and r(x_0, c) constants.

    @functools.partial(jax.jit, static_argnums=(0,))
    def sample_trajectories(self, params, xT, chain_keys):
        """This function samples full denoising chains on a batch of low resolution
        input using the current policy parameters.

        Args:
            - xT (jnp.array): (B, H, W, C) low-resolution sample to denoise
            - params (dict): model parameters
            - chain_key (B, key): per batch key.

        Returns:
            - x0 (jnp.array): (B, M, H, W, C)
            - x_in (jnp.array): (B, M, T, H, W, C) noisy sample fed as x_{t}} at each denoising step of the chain.
            - x_out (jnp.array): (B, M, T, H, W, C) denoised sample x_{t-1} sampled from p(x_{t-1}|x_t)
            - log_probs (jnp.array): (B, M, T) log-probabilities of the sample under the current posterior distribution.
        """


        def group_denoising_chain(xT_single, params, per_group_key):
            """ This function vmap's the denoising chain across the group size dimension,
            performing M denoising chains to obtain M denoised samples from the same low
            resolution input.
            """

            # Splitting the group key into a single key per m sample to denoise
            per_m_keys = jax.random.split(per_group_key, self.M)

            # Duplicating the low resolution sample xT_single (H, W, C) along the
            # M axis --> xT_m (M, H, W, C)
            xT_M = jnp.broadcast_to(xT_single, (self.M,) + xT_single.shape)

            def denoising_chain(xT, params, per_m_key):
                """ This function projects the single denoising_step_fn along the time
                axis to obtain a clean sample x_0.

                Args:
                    same as 'sample_trajectories' function

                Returns:
                    x_0 (jnp.array): (T,H,W,C) final clean denoised sample.
                    (x_in, xt_out, log_prob) (Tuple(jnp.array))): ((T,H,W,C), (T,H,W,C), (T,)) monitoring variables
                """

                chain_keys = jax.random.split(per_m_key, self.t_vec.shape[0])

                def denoising_step_fn(xt, step_args):
                    """This function performs a single denoising step given
                    an instance of the Unet model.
                    It was built to be used with jax.lax.scan() to denoise across T timesteps.

                    ### extensions:
                    - make it work for conditional and unconditional models
                    - allow the use of linear guidance within the denoising process --> though
                    this needs to be thought about, do we calc the log prob of the corrected sample
                    or the uncorrected one?
                    """

                    # Collect timestep related arguments.
                    t = step_args[0]
                    key = step_args[1]

                    # compute the next sample by:
                    # 1. Predicting noise: forward pass through the Unet
                    eps_pred = self.model.apply({"params": params}, xt[None], jnp.array([t]), train=False)[0]

                    # 2. Computing the new sample mean & std
                    alpha_bar_t = self.alpha_bar[t]
                    alpha_t = 1 - self.beta_schedule[t]

                    mean = (1 / jnp.sqrt(alpha_t)) * (
                        xt - (1 - alpha_t) / jnp.sqrt(1 - alpha_bar_t) * eps_pred)


                    # 3. Sampling a gaussian centered around the mean & get log probs of sample
                    is_last = (t == 1)

                    # 3.a. First sample from the gaussian, std = 0 if t = 0 because last step is deterministic in DDPM
                    sample_std = jnp.where(is_last, 0.0, jnp.sqrt(self.beta_schedule[t]))
                    xt_bwd = mean + sample_std * jax.random.normal(key, mean.shape)

                    # 3.b Secondly get the log probs
                    safe_std = jnp.where(is_last, 1.0, jnp.sqrt(self.beta_schedule[t]))
                    log_probs = distrax.Normal(mean, safe_std).log_prob(xt_bwd)
                    log_probs = jnp.where(is_last, 0.0, log_probs)
                    log_prob = jax.lax.stop_gradient(log_probs.sum(axis=(-3, -2, -1)))

                    return xt_bwd, (xt, xt_bwd, log_prob, is_last)

                x0, (x_in, x_out, log_prob, is_last) = jax.lax.scan(denoising_step_fn, xT, (self.t_vec, chain_keys))

                return x0, x_in, x_out, log_prob, is_last

            x0, x_in, x_out, log_prob, is_last = jax.vmap(denoising_chain, (0, None, 0))(xT_M, params, per_m_keys)


            def compute_advantage(rewards, eps = 1e-5):
                return (rewards - jnp.mean(rewards)) / (jnp.std(rewards) + 1e-5)

            rewards, comps = self.reward_fn(x0) # (M,)
            advantage = jax.lax.stop_gradient(compute_advantage(rewards))

            return x0, x_in, x_out, log_prob, is_last, advantage, comps

        return jax.vmap(group_denoising_chain, (0, None, 0))(xT, params, chain_keys)


    def compute_ratio_step(self, params, x_in, log_prob_old, x_out):
        
        B, M, T = x_in.shape[:-3]
        H, W, C = x_in.shape[-3:]
        xt = jnp.reshape(x_in, (B*M*T, H, W, C))
        xt_bwd = jnp.reshape(x_out, (B*M*T, H, W, C))
        tt = jnp.broadcast_to(self.t_vec, (B, M, T)).reshape(B*M*T)

        eps_pred = self.model.apply({"params": params}, xt, tt, train=False)

        alpha_bar_t = self.alpha_bar[tt]
        alpha_t = 1 - self.beta_schedule[tt]
        alpha_coef = ((1 - alpha_t) / jnp.sqrt(1 - alpha_bar_t))[:, None, None, None]

        mean = (1 / jnp.sqrt(alpha_t))[:, None, None, None] * (
            xt - alpha_coef * eps_pred)

        is_last = (tt == 1)
        safe_std = jnp.where(is_last, 1.0, jnp.sqrt(self.beta_schedule[tt]))[:, None, None, None]

        log_probs = distrax.Normal(mean, safe_std).log_prob(xt_bwd)
        log_prob_batched = jnp.sum(log_probs, axis=(-3, -2, -1))
        log_prob_batched = jnp.where(is_last, 0.0, log_prob_batched)
        log_prob_new = jnp.reshape(log_prob_batched, (B, M, T))
        
        ratio = jnp.exp(jnp.clip(log_prob_new - log_prob_old, -20.0, 20.0))

        return ratio


    def loss_fn(self, params, x_in, log_prob_old, x_out, advantage, mask, eps = 0.1):
        """ This function calculates the PPO objective.
        Log probs are potentially in batches so take the average over the batch?
        """

        ratio = self.compute_ratio_step(params, x_in, log_prob_old, x_out)
        adv = advantage[:, :, None]
        loss = ratio * adv
        clipped_loss = jnp.clip(ratio, 1-eps, 1+eps) * adv
        surrogate = jnp.minimum(loss, clipped_loss)

        # Decided to mean over all axes rather than summing like proposed in the DDPO paper.
        # This makes the magnitude of the loss invariant to chain length (T) - interesting in case
        # we do DDIM instead of DDPM. 
        per_chain_loss = (surrogate * mask).sum(-1) / mask.sum(-1)
        
        return -per_chain_loss.mean()

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_params_step(self, params, opt_state, x_in, log_prob, x_out, advantage, mask, eps = 0.1):
        """ This function updates the model parameters by differentiating through the PPO loss. 
        """
        
        loss_values, grads = jax.value_and_grad(self.loss_fn)(params, x_in, log_prob, x_out, advantage, mask, eps)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return loss_values, params, opt_state


    def train_step(self, x_LR, params, opt_state, n_inner, noise_key, batched_step_keys):
        """ This function performs one full step of PPO update.
            - Noises a batch of low-res inputs
            - Collect a batch of low-res to high-res trajectories, M samples per each low-res input in the batch,
            - Applied n steps of PPO optimisation
            - Returns the updated parameters, reward components & loss value. 
        """

        # 1.bis. To implement - perform a few DDIM steps on the low res inputs - jitted function and return B, H, W, C
        # partly denoised input to use downstream.

        # 1. Noise the low-res input
        xT = self.ddpm.forward_process(x_LR, self.T, noise_key)

        # 2. Rollout trajectories
        x0, x_in, x_out, log_prob, is_last, advantage, comps = self.sample_trajectories(params, xT, batched_step_keys)
        
        B, H, W, C = xT.shape

        assert x0.shape == (B, self.M, H, W, C)
        assert x_in.shape == (B, self.M, self.T, H, W, C)
        assert advantage.shape == (B, self.M)
        assert (is_last.sum(-1) == 1).all()
        assert np.isclose(advantage[0].mean(), 0, atol = 0.1) and np.isclose(advantage[0].std(), 1, atol = 0.1) 

        mask = 1 - is_last # invert the mask so that all but the last denoising step gradients are passed.

        # 3. for n_inner steps do:
        #       - compute log prob ratio by doing a forward pass through the updated model.
        #       - compute advantage
        #       - compute clipped loss
        #       - collect gradients
        #       - collect updates on parameters and new opt_state
        #       - apply the updates to the current parameters.
        for _ in range(n_inner):
            loss_values, params, opt_state = self.update_params_step(params, opt_state, x_in, log_prob, x_out, advantage, mask, eps = 0.1)


        return loss_values, params, opt_state, comps
