"""Learned-guidance sequence inference over all conditional models trained/updated today.

Runs each model on the SAME 8 in-distribution sequences (val 32-35 + test 36-39) of the real
Re=1000 set, STRICTLY with learned physics guidance (condRes from the adapter) and NO linear
guidance (guidance_lambda = 0). The condRes signal is auto-selected per model from its
conditioning.train.cond_signal (gradient -> make_dx_func, field -> make_field_func), so field
adapters are fed the raw residual field they were trained on.

Each model is its own subprocess-free call; a failure in one is caught so the rest still run.
"""
import sys, os, copy, traceback
sys.path.insert(0, "/home/rhautier/ddpm-jax")
os.chdir("/home/rhautier/ddpm-jax")
import yaml
from src.sequence_inference import run_sequence_inference

BASE_INF = yaml.safe_load(open("configs/inference_config.yaml"))
SEQ_IDXS = [32, 33, 34, 35, 36, 37, 38, 39]   # val (32-35) + test (36-39) of the 40-seq Re=1000 set

# (name, model-config path [= cfgs[0]], checkpoint, out_tag)
MODELS = [
    ("gradient_frozen_60ep", "configs/config.yaml",
     "gs://ddpm-thesis-rh/checkpoints/ddpm/conditioned_frozen_base_60ep/ckpt_epoch_0059.pkl",
     "indist_re1000_grad_frozen60_"),
    ("gradient_full_60ep", "configs/config_full_finetune.yaml",
     "gs://ddpm-thesis-rh/checkpoints/ddpm/conditioned_full_finetune/ckpt_epoch_0059.pkl",
     "indist_re1000_grad_full60_"),
    ("field_frozen_60ep", "configs/config_field_cond.yaml",
     "gs://ddpm-thesis-rh/checkpoints/ddpm/conditioned_field_cond_60ep/ckpt_epoch_0059.pkl",
     "indist_re1000_field_frozen60_"),
    ("field_full_60ep", "configs/config_field_full_finetune.yaml",
     "gs://ddpm-thesis-rh/checkpoints/ddpm/conditioned_field_full_finetune/ckpt_epoch_0059.pkl",
     "indist_re1000_field_full60_"),
]

results = {}
for name, model_cfg_path, ckpt, out_tag in MODELS:
    print(f"\n########## INFERENCE: {name}  ({ckpt}) ##########", flush=True)
    model_cfg = yaml.safe_load(open(model_cfg_path))
    # sanity: learned path must be on for this to condition
    assert model_cfg["conditioning"]["train"]["enabled"], f"{name}: conditioning.train.enabled must be true"
    assert model_cfg["conditioning"]["inference"]["enabled"], f"{name}: conditioning.inference.enabled must be true"

    inf_cfg = copy.deepcopy(BASE_INF)
    sd = inf_cfg["sequence_diffusion"]
    sd["checkpoint"] = ckpt
    sd["out_tag"] = out_tag
    sd["seq_idxs"] = SEQ_IDXS
    sd["guidance_lambda"] = 0.0     # strictly learned, no linear
    sd["guidance_re"] = None
    try:
        run_sequence_inference([model_cfg, inf_cfg])
        results[name] = "OK"
    except Exception as e:
        traceback.print_exc()
        results[name] = f"FAILED: {e}"
        print(f"########## {name} FAILED: {e} ##########", flush=True)

print("\n########## INFERENCE QUEUE SUMMARY ##########", flush=True)
for name, status in results.items():
    print(f"  {name}: {status}", flush=True)
print("########## ALL INFERENCE DONE ##########", flush=True)
