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

# === Bake thread-pinning env vars into the venv's activate script ============
#
# dc_toolkit's check_thread_oversubscription (utils.py) aborts at startup if
# any of OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS /
# BLOSC_NTHREADS / NUMBA_NUM_THREADS isn't pinned to 1.  Doing the export from
# inside the venv's bin/activate means users don't have to remember each
# session — `source venv/bin/activate` is now sufficient.
#
# We rely on $VIRTUAL_ENV to find the activate script.  That variable is set
# automatically by `source venv/bin/activate`, which the documented install
# flow runs before this script:
#     python -m venv venv && source venv/bin/activate && bash install_dc_toolkit.sh
# If $VIRTUAL_ENV is unset, we don't know which venv to modify, so we warn
# and skip the bake (rather than guessing './venv' or hard-failing); the
# user can then either re-run inside an activated venv, or export the vars
# manually each session.
# -----------------------------------------------------------------------------

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[thread-pin] WARNING: \$VIRTUAL_ENV is not set."
    echo "[thread-pin]   Skipping the activate-script modification."
    echo "[thread-pin]   To get the thread-pinning env vars baked in, install"
    echo "[thread-pin]   from inside an activated venv:"
    echo "[thread-pin]     python -m venv venv && source venv/bin/activate"
    echo "[thread-pin]     bash install_dc_toolkit.sh"
    echo "[thread-pin]   Otherwise, export them manually before running dc_toolkit:"
    echo "[thread-pin]     export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\"
    echo "[thread-pin]            OPENBLAS_NUM_THREADS=1 BLOSC_NTHREADS=1 \\"
    echo "[thread-pin]            NUMBA_NUM_THREADS=1"
else
    ACTIVATE="$VIRTUAL_ENV/bin/activate"
    SENTINEL="# === dc_toolkit thread pinning ==="

    # Idempotency guard: re-running the install script (e.g. to refresh EBCC)
    # must not append a second copy of the export block.  Sentinel-based check
    # is robust against the user editing surrounding lines, unlike line-count
    # tricks.
    if grep -qF "$SENTINEL" "$ACTIVATE"; then
        echo "[thread-pin] Already baked into $ACTIVATE; skipping."
    else
        echo "[thread-pin] Appending thread-pinning exports to $ACTIVATE"
        # Heredoc is unquoted so $SENTINEL expands at write time; none of the
        # literal `export VAR=1` lines below contain shell metacharacters, so
        # nothing else gets unintentionally interpreted.
        cat >> "$ACTIVATE" <<EOF

$SENTINEL
# Required by dc_toolkit's oversubscription check.  Pinning these to 1
# prevents N_threads x M_internal oversubscription when running the
# thread-per-combo sweep (see utils.check_thread_oversubscription).
# Added automatically by install_dc_toolkit.sh.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLOSC_NTHREADS=1
export NUMBA_NUM_THREADS=1
EOF
        echo "[thread-pin] Done.  Re-source the venv to pick up the new vars"
        echo "[thread-pin] in this shell:"
        echo "[thread-pin]   source $ACTIVATE"
    fi
fi
