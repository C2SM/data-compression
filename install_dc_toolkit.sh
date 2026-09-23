#!/bin/bash
set -euo pipefail

pip install --upgrade pip

pip install -e .
# Not a declared dependency: mpi4py is built with the mpicc on PATH so that it links
# the MPI that srun or mpirun starts.
CC=$(which mpicc) pip install --no-binary=mpi4py mpi4py

# WITH_EBCC=1 adds EBCC (Error Bounded Climate Compressor, evaluate_combos --with-ebcc).
# Its build (OpenJPEG + an HDF5 filter) needs cmake, a C/C++ toolchain and HDF5 headers.
if [[ "${WITH_EBCC:-0}" == "1" ]]; then
  echo "[install] Installing EBCC (optional serializer)..."
  pip install "ebcc[zarr] @ git+https://github.com/spcl/EBCC.git"
fi

# The codec and BLAS thread pools are sized from the variables below, which the venv does
# not set (santis.run exports them).  evaluate_combos and compress refuse to start unless
# all are 1 (--no-oversubscription-check only warns); the other commands do not check.
echo "[install] Done.  Remember to export thread-pinning env vars manually:"
echo "[install]   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\"
echo "[install]          BLOSC_NTHREADS=1 NUMBA_NUM_THREADS=1 \\"
echo "[install]          VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1"
