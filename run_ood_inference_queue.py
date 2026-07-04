"""OOD + baseline sequence inference.

Runs the 4 conditional models (gradient/field x frozen/full, 60ep) AND the plain unconditional
base model on Re=500 and Re=2000 generated flows (8 seqs each), STRICTLY learned physics guidance
(guidance_lambda=0, no linear). For the conditional models, conditioning.inference.re is set to the
TARGET Re so the residual signal (condRes) encodes the target-Re physics — that is how the model is
"told" the flow is Re=500 / Re=2000. The base model runs unconditionally (learned off).

Also runs the plain base at Re=1000 in-dist (the missing plain baseline for the in-dist §3c table).

Model normalization stays at the Re=1000 training stats (mean 0, std 4.7988); the residual is
evaluated at the data's Re. Big pkls mirror to GCS (monitoring/sparse_reconstructions/); we clear
leaked /tmp files after every job as a belt-and-suspenders on top of the save_results_to_gcs fix.
"""
import sys, os, copy, glob, traceback
sys.path.insert(0, "/home/rhautier/ddpm-jax")
os.chdir("/home/rhautier/ddpm-jax")
import yaml
from src.sequence_inference import run_sequence_inference

BASE_INF = yaml.safe_load(open("configs/inference_config.yaml"))
OOD_SEQS  = [0, 1, 2, 3, 4, 5, 6, 7]
INDIST_SEQS = [32, 33, 34, 35, 36, 37, 38, 39]
BASE_CKPT = "gs://ddpm-thesis-rh/checkpoints/ddpm/ckpt_epoch_0299.pkl"

RE_DATA = {
    500:  "gs://ddpm-thesis-rh/flow-data/generated_kf/kf_re500_256_20seed.npy",
    1000: "gs://ddpm-thesis-rh/flow-data/kf_2d_re1000_256_40seed.npy",
    2000: "gs://ddpm-thesis-rh/flow-data/generated_kf/kf_re2000_256_20seed.npy",
}
# conditional model -> (its training config = cfgs[0], checkpoint)
COND = {
    "grad_frozen60":  ("configs/config.yaml",                    "gs://ddpm-thesis-rh/checkpoints/ddpm/conditioned_frozen_base_60ep/ckpt_epoch_0059.pkl"),
    "grad_full60":    ("configs/config_full_finetune.yaml",      "gs://ddpm-thesis-rh/checkpoints/ddpm/conditioned_full_finetune/ckpt_epoch_0059.pkl"),
    "field_frozen60": ("configs/config_field_cond.yaml",         "gs://ddpm-thesis-rh/checkpoints/ddpm/conditioned_field_cond_60ep/ckpt_epoch_0059.pkl"),
    "field_full60":   ("configs/config_field_full_finetune.yaml","gs://ddpm-thesis-rh/checkpoints/ddpm/conditioned_field_full_finetune/ckpt_epoch_0059.pkl"),
}

def run_one(name, model_cfg_path, ckpt, re, seq_idxs, out_tag, learned):
    cfg = yaml.safe_load(open(model_cfg_path))
    cfg["data"]["data_path"] = RE_DATA[re]
    cfg["conditioning"]["train"]["enabled"] = bool(learned)          # ConditionalUnet iff learned
    cfg["conditioning"]["inference"]["enabled"] = bool(learned)
    if learned:
        cfg["conditioning"]["inference"]["re"] = re                  # tell the model the target Re
    inf = copy.deepcopy(BASE_INF)
    sd = inf["sequence_diffusion"]
    sd.update({
        "checkpoint": ckpt, "gt_data_path": RE_DATA[re], "seq_idxs": seq_idxs,
        "out_tag": out_tag, "guidance_lambda": 0.0, "guidance_re": None,
        "degrade_input": "sparse_nnfill",
    })
    try:
        run_sequence_inference([cfg, inf])
        return "OK"
    except Exception as e:
        traceback.print_exc()
        return f"FAILED: {e}"
    finally:
        for f in glob.glob("/tmp/tmp*.pkl"):
            try: os.remove(f)
            except OSError: pass

# job list: (name, cfg, ckpt, re, seqs, out_tag, learned)
JOBS = [("base_indist_re1000", "configs/config.yaml", BASE_CKPT, 1000, INDIST_SEQS, "indist_re1000_base_", False)]
for re in (500, 2000):
    JOBS.append((f"base_ood_re{re}", "configs/config.yaml", BASE_CKPT, re, OOD_SEQS, f"ood_re{re}_base_", False))
    for name, (mcfg, ckpt) in COND.items():
        JOBS.append((f"{name}_ood_re{re}", mcfg, ckpt, re, OOD_SEQS, f"ood_re{re}_{name}_", True))

def already_done(out_tag, seqs, gcs_index):
    """True iff all 8 seq pkls for this out_tag are already on GCS (resume support)."""
    return all(f"sequence_reconstruction_{out_tag}seq{s}.pkl" in gcs_index for s in seqs)

# one listing of the GCS mirror so a relaunch skips completed jobs (idempotent resume)
import subprocess
_ls = subprocess.run(["gcloud", "storage", "ls", "gs://ddpm-thesis-rh/monitoring/sparse_reconstructions/"],
                     capture_output=True, text=True)
GCS_INDEX = {line.rsplit("/", 1)[-1] for line in _ls.stdout.splitlines()}

print(f"OOD/baseline queue: {len(JOBS)} jobs x 8 seqs\n", flush=True)
results = {}
for (name, mcfg, ckpt, re, seqs, tag, learned) in JOBS:
    if already_done(tag, seqs, GCS_INDEX):
        print(f"########## {name}: SKIP (already on GCS) ##########", flush=True)
        results[name] = "SKIP"
        continue
    print(f"\n########## {name}  re={re} learned={learned}  out_tag={tag} ##########", flush=True)
    results[name] = run_one(name, mcfg, ckpt, re, seqs, tag, learned)
    print(f"########## {name}: {results[name]} ##########", flush=True)

print("\n########## OOD QUEUE SUMMARY ##########", flush=True)
for k, v in results.items():
    print(f"  {k}: {v}", flush=True)
print("########## ALL OOD DONE ##########", flush=True)
