#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot environment setup to run ON the TPU VM (ddpm-v4-8, us-central2-b).
# Idempotent: safe to re-run. Creates a venv, installs TPU JAX + project deps,
# and verifies the 4 TPU chips are visible.
#
# Usage (on the VM):  bash ~/ddpm-jax/scripts/tpu_setup.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_URL="https://github.com/RomainHautier/ddpm-in-jax"
REPO_DIR="$HOME/ddpm-jax"
VENV="$HOME/venv-ddpm"
JAX_VERSION="0.7.2"

echo "==> 1/5 Clone or update repo"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --ff-only || echo "   (skipping pull; uncommitted changes?)"
fi
cd "$REPO_DIR"

echo "==> 2/5 Create virtualenv at $VENV"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -U pip

echo "==> 3/5 Install JAX for TPU ($JAX_VERSION)"
pip install -q "jax[tpu]==${JAX_VERSION}" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "==> 4/5 Install project dependencies"
pip install -q -r requirements-tpu.txt

echo "==> 5/5 Verify TPU is visible to JAX"
python3 - <<'PY'
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
