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

# Thread-pinning is no longer baked into the venv.  Export the codec-internal
# thread caps manually (e.g. in your sbatch script):
#
#   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#          BLOSC_NTHREADS=1 NUMBA_NUM_THREADS=1 \
#          VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1
#
# dc_toolkit's oversubscription-check (on by default) catches any lapse.
# Use --codec-threads N (where supported) to deliberately allow internal
# codec threading; --threads * --codec-threads must stay <= physical cores.
echo "[install] Done.  Remember to export thread-pinning env vars manually:"
echo "[install]   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\"
echo "[install]          BLOSC_NTHREADS=1 NUMBA_NUM_THREADS=1 \\"
echo "[install]          VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1"
