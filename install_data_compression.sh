#!/bin/bash
set -euo pipefail

pip install --upgrade pip

# Install your local project
pip install -e .
pip install -r requirements.txt

# Install EBCC with Zarr support (always fresh clone)
EBCC_DIR="EBCC"
EBCC_REMOTE="https://github.com/spcl/EBCC.git"

if [ -d "$EBCC_DIR" ]; then
  echo "[EBCC] Removing existing folder..."
  rm -rf "$EBCC_DIR"
fi

echo "[EBCC] Cloning fresh repo..."
git clone --recursive "$EBCC_REMOTE" "$EBCC_DIR"

pushd "$EBCC_DIR"
pip install -e ".[zarr]"
popd
