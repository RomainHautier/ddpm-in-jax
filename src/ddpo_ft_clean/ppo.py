# Replicate the implementation of the DDPO paper.
# Probably a denoising re-implementation, storing the parameters of the previous
# distribution and some clipping

import jax
import functools
from src.models.model import DDPM, jit_denoise_step
from jax.scipy import stats
from src.models.model import calc_log_probs
import distrax

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

    def __init__(self, ppo_config: dict, ddpm_config, weights):
        
        self.config = config
        ddpm = DDPM(config)
        self.model = ddpm.unet
        init_params = .... # load params from pretained checkpoint.

        self.alpha_bar = ddpm.alpha_bar
        self.beta_schedule = ddpm.beta_schedule

        ## PPO optimizes the following objective
        ## sum_0^T ( p_theta(x_t-1|x_t) / p_theta_old(x_t-1|x_t) * delta log p_theta(x_t-1|x_t) * r(x_0, c) )
        ## using the log derivative trick:
        ## sum_0^T (  delta p_theta(x_t-1|x_t) / p_theta_old(x_t-1|x_t) * r(x_0, c) )
        ## The objective is minimized by taking the derivative of the above loss,
        ## with both p_theta_old and r(x_0, c) constants.
    
    
        return model, params


    @functools.partial(jax.jit, static_argnums=(0,))
    def sample_trajectories(self, params, xT, chain_key):
        """This function samples full denoising chains on a batch of low resolution
        input using the current policy parameters.

        Args:
            - xT (jnp.array): (B, H, W, C) low-resolution sample to denoise
            - params (dict): model parameters
            - chain_key (1,): key to split along the batch and then time dimension for stochastic denoising.
        
        Returns:
            - x0 (jnp.array): (B, M, H, W, C)
            - x_in (jnp.array): (B, M, T, H, W, C) noisy sample fed as x_{t}} at each denoising step of the chain.
            - x_out (jnp.array): (B, M, T, H, W, C) denoised sample x_{t-1} sampled from p(x_{t-1}|x_t)
            - log_probs (jnp.array): (B, M, T) log-probabilities of the sample under the current posterior distribution.
        """

        # Initialising the timesteps at which to denoise (currently only supports DDPM)
        t_vec = jnp.arange(self.T)[::-1] ## if ddpm only!
        # Splitting the noise key per batch.
        per_batch_keys = jax.random.split(chain_key, xT.shape[0])
        

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

                chain_keys = jax.random.split(per_m_key, t_vec.shape[0])

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
                    is_last = (t == 0)

                    # 3.a. First sample from the gaussian, std = 0 if t = 0 because last step is deterministic in DDPM
                    sample_std = jnp.where(is_last, 0.0, jnp.sqrt(self.beta_schedule[t]))
                    xt_bwd = mean + sample_std * jax.random.normal(key, mean.shape)
                    
                    # 3.b Secondly get the log probs 
                    safe_std = jnp.where(is_last, 1.0, jnp.sqrt(self.beta_schedule[t]))
                    log_probs = distrax.Normal(mean, safe_std).log_prob(xt_bwd)
                    log_probs = jnp.where(is_last, 0.0, log_probs)
                    log_prob = log_probs.sum(axis=(-3,-2,-1))

                    return xt_bwd, (xt, xt_bwd, log_prob, is_last)
                
                x0, (x_in, x_out, log_prob, is_last) = jax.lax.scan(denoising_step_fn, xT, (t_vec, chain_keys))
                
                return x0, x_in, x_out, log_prob, is_last

            x0, x_in, x_out, log_prob, is_last = jax.vmap(denoising_chain, (0, None, 0))(xT_M, params, per_m_keys)


            def compute_advantage(rewards):
                mean_reward = jnp.mean(rewards)
                return rewards -mean_reward

            rewards = jax.vmap(compute_rewards)(x0) # (M,)
            advantage = compute_advantage(rewards)

            return x0, x_in, x_out, log_prob, is_last, advantage

        return jax.vmap(group_denoising_chain, (0, None, 0))(xT, params, per_batch_keys)
    

    def get_advantage(x0):
    """ This function computes the net advantage of denoised trajectories per 
    samples of the batch.

    The advantage is defined as r(x_0, c)_i - 1/n * sum_{i=0}^{n} r(x_0, c)_i 
    with i denoting the i'th denoised sample of sample b in batch of size B.
    
    Args:
        - x0 (jnp.array): (B,H,W,C)

    Returns:

    """

    @jax.jit
    def loss_fn(log_probs_old, log_probs, reward)
    """ This function calculates the PPO objective.
    Log probs are potentially in batches so take the average over the batch?
    
    """ 
        def loss(log_probs_old, log_probs, reward)
            ### the sum needs to be done along the batch axis
            return 1/log_probs.shape[0] * jnp.sum( jnp.exp(log_probs - log_probs_old) ) * reward

        return loss
    
    @jax.jit
    def forward_probs(x_ts, std):
    """ This function evaluates the log-probabilities under the updated policy.
    Gets a normal distribution centered at the previous sample and a pre-defined std.
    """


        return log_probs
    


            

    def train_step():

        sampling_std = config[std]
        
        for b in batch:

            xt, log_probs_old, rewards = sample()

            for k in range(K): ## number of inner gradient steps to take over a batch

                log_probs = forward_probs(xt[k], std)
                advantage = ## standardise rewards per low res sample
                loss = loss_fn(log_probs_old, log_probs, advantage)
                params = opt_stop(loss, params)



