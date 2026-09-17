#!/bin/bash
set -euo pipefail

pip install --upgrade pip

# Install dc_toolkit
pip install -e .
CC=$(which mpicc) pip install --no-binary=mpi4py mpi4py

# Optional: EBCC (Error Bounded Climate Compressor), enabled in dc_toolkit with
# --with-ebcc.  Builds OpenJPEG + an HDF5 filter: needs cmake and HDF5 headers.
#   WITH_EBCC=1 bash install_dc_toolkit.sh
if [[ "${WITH_EBCC:-0}" == "1" ]]; then
  echo "[install] Installing EBCC (optional serializer)..."
  pip install "ebcc[zarr] @ git+https://github.com/spcl/EBCC.git"
fi

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
