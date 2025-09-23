#!/bin/bash
set -euo pipefail

pip install --upgrade pip

# Install dc_toolkit
pip install -e .
CC=$(which mpicc) pip install --no-binary=mpi4py mpi4py

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
