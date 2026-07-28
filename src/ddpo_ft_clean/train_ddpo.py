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
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ppo import PPO
from src.models.model import DDPM
from src.ddpo_ft.rewards_claude import Reward

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

    # Initialise the reward class
    reward_fn  = Reward.from_calibration("base_results/regime_stats_re1000.npz",
                                 "base_results/reward_calibration.json",
                                 re=1000, weights={"spec": 0.5, "spec_highk": 3.0, "energy": 0.1, "w1": 0.0, "pde": 1.0}, pde_hinge=True)

    # Initialise the PPO class
    ppo = PPO(ppo_config, ddpm, params, reward_fn)

    ### --- Building out the inner loop with n_inner gradient steps

    ## 1. Set up the optimizer 
    ## 2. Set up the loop over n_inner optimization steps.
    ## 3. Sample the log probs of the trajectory under the current set of parameters
    ## 4. Calculate the loss - sum the - ratio * advantage over all T timesteps
    ## 5. Get the gradients by autodiff the loss and apply an opt step
    ## 6. Overwrite updated params

    key = jax.random.key(42)
    key, init_key = jax.random.split(key)

    
    n_inner = ppo_config['n_opt_steps']
    opt_state = self.optimizer.init(params)


    ### --- Defining multi-chip sharding
    B = 16
    xB = jax.random.uniform(shape=(B, im_size, im_size, channels))

    # collect the devices available
    devices = np.array(jax.devices())
    # create a mesh over these devices to call them, and call that axis "data" as we'll split along the
    # batch axis of xB
    mesh = Mesh(devices, axis_names=("data",))

    # define how to shard the data on the devices. Data is sharded across devices through the data axis while
    # parameters are replicated across all devices
    data_sharding = NamedSharding(mesh, P("data")) # P("data") partition axis 0 over the data mesh axis. Each device receives B/n samples (B/n, H, W, C)
    params_sharding = NamedSharding(mesh, P()) # P() means no axis partitioned

    # put on device all the things which require to be sharded, params and opt state are the same across devices.
    params = jax.device_put(params, params_sharding)
    opt_state = jax.device_put(opt_state, params_sharding)
    xB = jax.device_put(xB, data_sharding) # --> (B, H, W, C) split across chips.


    # let xB  = x_LR

    for _ in range(ppo_epochs):
        key, noise_key, step_key = jax.random.split(key)
        batch_step_keys = jax.device_put(jax.random.split(step_key, B), data_sharding)
        loss_values, params, opt_state, comps = ppo.train_step(x_LR, params, opt_state, n_inner, noise_key, batch_step_keys)

    ### -- TO DO:
    # wrap the train step in a full training loop over batches.
    # look into sharding the script across multiple chips: the train step itself is not fully sharded, since we 
    # need to shard the step up until the gradient calculations before accumulating the gradients and then applying the full update.
    # 


if __name__ == "__main__":
    main()
