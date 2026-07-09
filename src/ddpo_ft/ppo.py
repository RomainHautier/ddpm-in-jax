# Replicate the implementation of the DDPO paper.
# Probably a denoising re-implementation, storing the parameters of the previous
# distribution and some clipping

import jax
from src.models.model import DDPM, jit_denoise_step
from jax.scipy import stats
from src.models.model import calc_log_probs

class PPO():

    def __init__(self, ppo_config: dict, ddpm_config):
        
        self.config = config
        ddpm = DDPM(config)

    

    def sample(xT, params, T, key)
        
        log_probs = []
        
        for t in range(T, 0, -1):
            key, noise_key = jax.random.split(key)
            z = jax.random.normal(noise_key, shape=xT.shape) if t > 1 else jnp.zeros(shape=xT.shape)
            xt, log_prob = jit_denoise_step(params, xT, t, z)
            log_probs_old.append(log_prob)
            xT = xt
        
        return xt, log_probs

    def train_step(self, params, seed):

        
        key = jax.random.PRNGKey(seed)

        def loss_fn(xT, params, T, key):
            # Loss = pi / pi_old * -advantage * diff(log pi)
            x0, log_probs = sample(xT, params, T, key)
            
            # compute rewards

            # compute advantage
            

            return = - advantage * log_prob


        # comput loss and diff through the network based on advantage -
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # get update through optimizer
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_update(params, updates)

        
        return params



            




