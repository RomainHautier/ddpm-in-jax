#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot environment setup to run ON the TPU VM (ddpm-v4-8, us-central2-b).
# Idempotent: safe to re-run. Ensures Python >= 3.11 (jax 0.7.2 requires it;
# the tpu-ubuntu2204-base image ships only 3.10), creates a venv, installs
# TPU JAX + project deps, and verifies the 4 TPU chips are visible.
#
# Usage (on the VM):  bash ~/ddpm-jax/tpu-setup/tpu_setup.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_URL="https://github.com/RomainHautier/ddpm-in-jax"
REPO_DIR="$HOME/ddpm-jax"
VENV="$HOME/venv-ddpm"
JAX_VERSION="0.7.2"

echo "==> 1/6 Clone or update repo"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --ff-only || echo "   (skipping pull; uncommitted changes?)"
fi
cd "$REPO_DIR"

echo "==> 2/6 Ensure a Python >= 3.11 interpreter (jax $JAX_VERSION needs it)"
PYTHON_BIN=""
for cand in python3.12 python3.11; do
  if command -v "$cand" >/dev/null 2>&1; then PYTHON_BIN="$cand"; break; fi
done
if [ -z "$PYTHON_BIN" ]; then
  echo "   No python3.11/3.12 found (image has 3.10). Installing python3.11 via deadsnakes..."
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11 python3.11-venv
  PYTHON_BIN="python3.11"
fi
echo "   Using $($PYTHON_BIN -V) at $(command -v "$PYTHON_BIN")"

echo "==> 3/6 Create virtualenv at $VENV"
# Heal a half-made venv: recreate unless a working activate script exists.
if [ ! -f "$VENV/bin/activate" ]; then
  rm -rf "$VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -U pip

echo "==> 4/6 Install JAX for TPU ($JAX_VERSION)"
pip install -q "jax[tpu]==${JAX_VERSION}" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "==> 5/6 Install project dependencies"
pip install -q -r requirements-tpu.txt

echo "==> 6/6 Verify TPU is visible to JAX"
python - <<'PY'
import jax
devs = jax.devices()
print("JAX backend :", jax.default_backend())
print("JAX devices :", devs)
assert any(d.platform == "tpu" for d in devs), "No TPU devices found!"
print(f"OK: {len(devs)} TPU device(s) visible.")
PY

echo ""
echo "Done. Activate the env in new shells with:  source $VENV/bin/activate"
echo "Smoke-test data access from the bucket with:"
echo "  python3 -c \"from src.utils import load_npy_from_gcs as L; L('gs://ddpm-thesis-rh/flow-data/kf_2d_re1000_256_40seed.npy')\""
