# TPU Development Environment — Setup & Operations Guide

A complete reference for the Cloud TPU workflow set up for this project: what
exists, why, how to connect, how to run, and how to manage cost. Written
2026-06-10.

---

## 1. Overview (the mental model)

You have free TPU access via Google's **TPU Research Cloud (TRC)** grant. The
setup gives you a single **TPU VM** you SSH into and develop on directly from
VSCode, with training data and outputs living in a **Google Cloud Storage (GCS)
bucket**.

```
  Your laptop (Windows + WSL)                    Google Cloud (project: csml-thesis)
  ┌─────────────────────────┐                    ┌──────────────────────────────────┐
  │ VSCode (Windows)         │   IAP tunnel       │  TPU VM  "ddpm-v4-8"               │
  │  Remote-SSH ──────────────────(encrypted)────►│  us-central2-b, 4× v4 chips        │
  │   └ ProxyCommand=wsl gcloud                    │   • your code (~/ddpm-jax)         │
  │                          │                    │   • venv with TPU JAX              │
  │ WSL: gcloud (grant acct) │                    │   • service account ───┐           │
  └─────────────────────────┘                    └────────────────────────│───────────┘
                                                                           │ read/write
                                                          ┌────────────────▼───────────┐
                                                          │ GCS bucket "ddpm-thesis-rh" │
                                                          │ us-central1                 │
                                                          │  • flow-data/*.npy (dataset)│
                                                          │  • monitoring/, checkpoints │
                                                          └─────────────────────────────┘
```

Key principle: **the laptop holds nothing heavy.** Code is in git, data and
checkpoints are in GCS, compute is the TPU VM. Any machine can reconnect.

---

## 2. The accounts (important — two Google accounts in play)

| Account | Role | Used for |
|---|---|---|
| `grg.hautier@gmail.com` | **TRC grant account** | Everything TPU. This is the active gcloud account and is **pinned** in the SSH config. |
| `romain.hautier@gmail.com` | Separate **paid** account | NOT used here at all. |

**Billing safety:** TPU usage bills to the *project* (`csml-thesis`), which is
enrolled in TRC, so the TPU is free. Billing has nothing to do with which
account SSHes in. The paid account is never referenced. To be doubly safe, the
IAP tunnel command hard-codes `--account=grg.hautier@gmail.com`, so even if you
switch your active gcloud account, the connection still uses the grant account.

---

## 3. What was provisioned (Google Cloud side)

### 3.1 The TRC grant
The grant (from your TRC acceptance email) includes, among spot allocations:
- **32 on-demand v4 chips in `us-central2-b`** ← the only *on-demand* line; this
  is what supports a continuously-held VM.
- Various **spot** (preemptible) allocations of v6e / v5e / v4 in other zones —
  larger, but can be reclaimed at any time.

We used a tiny slice of the on-demand allocation.

### 3.2 The TPU VM
| Property | Value |
|---|---|
| Name | `ddpm-v4-8` |
| Type | `v4-8` = **4 chips / 8 TensorCores**, single host (1 VM) |
| Zone | `us-central2-b` |
| Mode | **on-demand** (held until you delete it — no preemption) |
| Runtime image | `tpu-ubuntu2204-base` (Ubuntu 22.04, Python 3.10) |
| Created via | **Queued Resources** (`queued-resources create`) |

> **Why `v4-8`?** It is the *smallest* v4 slice (there is no v4-1/2/4) and a
> single host — which is the only configuration that gives a clean "SSH in +
> VSCode + edit and run" experience. Multi-host pods (`v4-16`+) require running
> commands across all worker VMs at once and have no single machine to live in.
> The v4-8 uses 8 of your 64 on-demand cores, leaving plenty of headroom.

> **Why queued resources?** Instead of `create` failing instantly when a zone is
> full, a queued request waits and provisions when capacity frees up, then holds
> it. It is the modern, recommended path.

### 3.3 Networking & security
- The project's `default` VPC had no subnet in `us-central2` (auto-mode skips
  that special region), so we created subnet **`tpu-us-central2`** (`10.10.0.0/20`).
- **IAP-only SSH.** We chose **Identity-Aware Proxy (IAP)** over a public IP:
  - The VM has **no internet-facing SSH port**. There is nothing to scan/brute-force.
  - Firewall rule **`allow-ingress-from-iap`** allows TCP:22 only from Google's
    IAP range `35.235.240.0/20`.
  - SSH is brokered through Google's authenticated proxy. This is the more secure
    choice for a long-lived VM.

### 3.4 The GCS bucket
| Property | Value |
|---|---|
| Name | `gs://ddpm-thesis-rh` |
| Location | `us-central1` (single region) |
| Access | Uniform bucket-level access |
| Permissions | VM service account `162596308004-compute@developer.gserviceaccount.com` granted `roles/storage.objectAdmin` |

> **Why `us-central1` and not `us-central2`?** `us-central2` is a TPU-only region
> and is **not a valid GCS bucket location**. `us-central1` is the adjacent,
> recommended region — low latency to the TPU over Google's internal network.

> **Auth on the VM:** code uses `gcsfs`, which authenticates via Application
> Default Credentials = the VM's service account automatically. No key file
> (`creds.txt`) is needed on the VM.

---

## 4. What changed in the repo

| File | Change | Why |
|---|---|---|
| `src/utils.py` | `GCS_BUCKET = "ddpm-thesis-rh"`; added `GCS_PROJECT`; `get_fs()` now passes the **project** (was incorrectly passing the bucket name) | Point at the real bucket; fix a latent gcsfs bug |
| `configs/config.yaml` | `gcs_bucket` → `ddpm-thesis-rh`; `data_path` → `gs://ddpm-thesis-rh/flow-data/kf_2d_re1000_256_40seed.npy` | Load training data from the bucket instead of a local path |
| `requirements-tpu.txt` | **new** — lean TPU dependency list | The root `requirements.txt` is a full Colab/GPU dump that would clobber the TPU `jaxlib`. **Never `pip install -r requirements.txt` on the TPU.** |
| `tpu-setup/tpu_setup.sh` | **new** — one-shot VM setup | Clone repo, build venv, install TPU JAX + deps, verify chips |
| `tpu-setup/TPU_SETUP.md` | **new** — this document | — |

### Local machine (not in repo)
- `~/.ssh/config` (WSL) — `Host ddpm-tpu` entry for terminal SSH from WSL.
- `C:\Users\hauti\.ssh\config` + key copy — for Windows-native VSCode Remote-SSH.

---

## 5. The data

| File | Size | Status |
|---|---|---|
| `kf_2d_re1000_256_40seed (1).npy` | 3.2 GB | Uploaded to `gs://ddpm-thesis-rh/flow-data/kf_2d_re1000_256_40seed.npy` (space/`(1)` dropped to match config) |
| `kmflow_sampled_data_irregnew (1).npz` | 6.3 GB | **Not uploaded** — not referenced anywhere in code/configs |

To upload more data later (from WSL):
```bash
gcloud storage cp LOCAL_FILE gs://ddpm-thesis-rh/flow-data/
```

---

## 6. Connecting from VSCode (Windows + WSL users)

> **Constraint:** VSCode cannot open a Remote-SSH connection from *inside* a
> Remote-WSL window (no nested remotes). Use a **plain Windows VSCode window**.

1. Install the **Remote - SSH** extension.
2. `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → **`ddpm-tpu`**.
3. Wait ~30 s while VS Code Server installs on the VM.
4. **Open Folder** → `/home/rhautier/ddpm-jax`.

The Windows SSH config runs the IAP tunnel through WSL (`ProxyCommand wsl.exe …
gcloud … start-iap-tunnel …`) using the pinned grant account, so no gcloud
install is needed on Windows.

### Terminal-only access (from WSL)
```bash
ssh ddpm-tpu
# or the canonical form:
gcloud alpha compute tpus tpu-vm ssh ddpm-v4-8 \
  --zone=us-central2-b --project=csml-thesis --tunnel-through-iap
```

### Known gotchas
- **Windows key permissions.** If Windows OpenSSH rejects the key as
  "unprotected," run once in PowerShell:
  ```powershell
  icacls C:\Users\hauti\.ssh\google_compute_engine /inheritance:r /grant:r "$($env:USERNAME):(R)"
  ```
- **Private repo clone.** `tpu_setup.sh` clones
  `github.com/RomainHautier/ddpm-in-jax`. If private, you'll need a GitHub token
  or deploy key on the VM (or push-from-local).

---

## 7. First-run setup on the VM

Once connected (VSCode terminal or `ssh ddpm-tpu`):
```bash
bash ~/ddpm-jax/tpu-setup/tpu_setup.sh
```
This is idempotent and:
1. Clones/updates the repo to `~/ddpm-jax`
2. Creates a venv at `~/venv-ddpm`
3. Installs `jax[tpu]==0.7.2` from the libtpu index
4. Installs `requirements-tpu.txt`
5. Verifies JAX sees the 4 TPU chips

Activate the env in any new shell:
```bash
source ~/venv-ddpm/bin/activate
```

Smoke-test data access from the bucket:
```bash
python3 -c "from src.utils import load_npy_from_gcs as L; \
  L('gs://ddpm-thesis-rh/flow-data/kf_2d_re1000_256_40seed.npy')"
```

---

## 8. Cost & lifecycle management (READ THIS)

The TPU is free under TRC, but **good hygiene still matters**:

- **The VM bills the project for as long as it exists** (TRC covers the TPU
  cost; storage/egress are separate and small). Since it is on-demand, it stays
  up until *you* delete it.
- **TPU VMs have no "stop."** Unlike GPU VMs, you cannot pause-and-resume. You
  either keep it running or **delete and recreate**.
- **Boot disk is ephemeral on delete.** Keep code in git and data/checkpoints in
  GCS so a deleted VM loses nothing important.

### Check status
```bash
gcloud compute tpus queued-resources list --zone=us-central2-b --project=csml-thesis
gcloud compute tpus tpu-vm list --zone=us-central2-b --project=csml-thesis
```

### Delete when done (frees the allocation)
```bash
gcloud compute tpus queued-resources delete ddpm-v4-8 \
  --zone=us-central2-b --project=csml-thesis --force --quiet
```
(`--force` also tears down the underlying VM.)

### Recreate later
```bash
gcloud compute tpus queued-resources create ddpm-v4-8 \
  --node-id=ddpm-v4-8 --project=csml-thesis --zone=us-central2-b \
  --accelerator-type=v4-8 --runtime-version=tpu-ubuntu2204-base \
  --network=default --subnetwork=tpu-us-central2
```

> **⚠️ After recreate, the VM's internal worker name and host key change**, so
> the SSH configs (which reference `t1v-n-7ea63afb-w-0` and a `HostKeyAlias`)
> will be **stale**. Re-run the dry-run to get the new values and update both
> configs:
> ```bash
> gcloud alpha compute tpus tpu-vm ssh ddpm-v4-8 --zone=us-central2-b \
>   --project=csml-thesis --tunnel-through-iap --dry-run
> ```
> Then run `tpu_setup.sh` again on the fresh VM.

---

## 9. Quick reference

| Thing | Value |
|---|---|
| Project | `csml-thesis` |
| Grant account | `grg.hautier@gmail.com` |
| TPU VM | `ddpm-v4-8` (v4-8, single host) |
| Zone | `us-central2-b` |
| Bucket | `gs://ddpm-thesis-rh` (us-central1) |
| VM service account | `162596308004-compute@developer.gserviceaccount.com` |
| Repo | `github.com/RomainHautier/ddpm-in-jax` → `~/ddpm-jax` on VM |
| Venv | `~/venv-ddpm` |
| SSH alias | `ssh ddpm-tpu` |
| Connection | IAP tunnel (no public IP) |
