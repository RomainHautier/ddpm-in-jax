import os
import sys
import copy
import pickle
import subprocess

# Make the script runnable directly from the terminal, from any cwd:
#   python src/ddpo_ft_clean/train_ddpo.py
# (adds the repo root + this dir to sys.path, and chdirs to the root so the
#  relative configs/... and checkpoint paths resolve)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path[:0] = [_ROOT, _HERE]
os.chdir(_ROOT)

import jax
import jax.numpy as jnp
import yaml
import optax
import distrax

from ppo import PPO
from src.models.model import DDPM

# Initialise a DDPM class, init the model and load the params to pass to the PPO class.

# The PPO base is the EMA shadow (mu=0.9999) of the 300-epoch unconditional retrain, epoch 299.
# Resolution order: $BASE_CKPT -> local working copy -> fetch from GCS (canonical).
GCS_CKPT = "gs://ddpm-thesis-rh/checkpoints/ddpm/base_ema_mu9999_300ep/ckpt_epoch_0299.pkl"
LOCAL_CKPT = "/tmp/ema_ckpts/ema_base_0299.pkl"


def load_base_params(ckpt_path=None):
    """Load the base checkpoint and return the EMA parameters (falls back to online `params`
    only if the checkpoint carries no EMA shadow — print line says which one you got)."""
    path = ckpt_path or os.environ.get("BASE_CKPT") or LOCAL_CKPT
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"checkpoint not found locally — fetching {GCS_CKPT}")
        subprocess.run(["gsutil", "-q", "-o", "GSUtil:check_hashes=never",
                        "cp", GCS_CKPT, path], check=True)
    with open(path, "rb") as f:
        ck = pickle.load(f)
    use_ema = "ema_params" in ck
    params = ck["ema_params"] if use_ema else ck["params"]
    print(f"base ckpt: {path} | epoch {ck.get('epoch', '?')} | "
          f"{'EMA weights (mu=' + str(ck.get('ema_rate', '?')) + ')' if use_ema else 'ONLINE weights (no EMA in ckpt!)'}")
    return params


def main():

    # yaml.safe_load parses a *stream*: passing the bare path string would just return the
    # string itself — open the files.
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    with open("configs/ppo_finetuning_cfg.yaml") as f:
        ppo_config = yaml.safe_load(f)

    im_size = config["data"]["image_size"]
    channels = config["model"]["in_ch"]

    # initialise the model — conditioning forced OFF so the plain Unet matches the
    # unconditional checkpoint (same convention as build_base_ddpm in the main repo)
    config = copy.deepcopy(config)
    config["conditioning"]["train"]["enabled"] = False
    config["conditioning"]["inference"]["enabled"] = False
    ddpm = DDPM(config)

    # load parameters from the EMA base checkpoint (env BASE_CKPT / local / GCS)
    params = load_base_params()

    ppo = PPO(ppo_config, ddpm, params)

    # --- Single sampling smoke test -------------------------------------------------------
    # Real DDPO seeds the chain SDEdit-style: xT = sqrt(ab[t0])*x_input + sqrt(1-ab[t0])*noise.
    # For a pure sampling-mechanics check, standard-normal xT is fine (chain length T=cfg T).
    B = 2
    sampling_key = jax.random.key(42)
    init_key, chain_key = jax.random.split(sampling_key)
    dummy_xT = jax.random.normal(init_key, shape=(B, im_size, im_size, channels))

    x0, x_in, x_out, log_prob, is_last, advantage = ppo.sample_trajectories(
        params, dummy_xT, chain_key)

    print(f"x0        {x0.shape}   (expect (B={B}, M={ppo.M}, {im_size}, {im_size}, {channels}))")
    print(f"x_in      {x_in.shape}   (expect (B, M, T={ppo.T}, H, W, C))")
    print(f"x_out     {x_out.shape}")
    print(f"log_prob  {log_prob.shape}   (expect (B, M, T))")
    print(f"is_last   {is_last.shape}  sum per chain = {is_last.sum(-1)[0, 0]} (expect 1: only t=0)")
    print(f"advantage {advantage.shape}   (expect (B, M)); per-group mean = "
          f"{jnp.abs(advantage.mean(-1)).max():.2e} (expect ~0 by construction)")
    print(f"sanity: x0 finite={bool(jnp.isfinite(x0).all())}  std={float(x0.std()):.3f} | "
          f"log_prob finite={bool(jnp.isfinite(log_prob).all())}  "
          f"mean per step={float(log_prob.mean()):.1f}")
    

    ### --- Building out the inner loop with n_inner gradient steps

    ## 1. Set up the optimizer 
    ## 2. Set up the loop over n_inner optimization steps.
    ## 3. Sample the log probs of the trajectory under the current set of parameters
    ## 4. Calculate the loss - sum the - ratio * advantage over all T timesteps
    ## 5. Get the gradients by autodiff the loss and apply an opt step
    ## 6. Overwrite updated params

    optimizer = optax.adam(learning_rate = ppo_config['lr'])
    n_inner = ppo_config['n_opt_steps']
    opt_state = optimiser.init(parameters)

    ### This loop runs n_inner times, and the gradient is aggregated loss from all B*M samples
    for _ in range(n_inner):

        loss_values, grads = jax.value_and_grad(ppo.loss_fn)( params, model, x_in, log_prob_old, x_out, t_vec, advantage, mask, eps = 0.1)
        updates, opt_state = optax.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)



if __name__ == "__main__":
    main()
