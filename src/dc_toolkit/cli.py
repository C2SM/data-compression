# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import hashlib
import json
import math
import os
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import itertools
import subprocess
import csv

import click
import zarr
import numcodecs
import numcodecs.zarr3
import xarray as xr
from dc_toolkit import utils
from zarr_any_numcodecs import AnyNumcodecsArrayBytesCodec
from ebcc.zarr_filter import EBCCZarrFilter
import pandas as pd
import numpy as np
from mpi4py import MPI
import dask
import dask.array
import humanize
import psutil

# Heavyweight optional imports (matplotlib / sklearn / plotly / tqdm) are
# deferred: they are only used by the clustering and plotting commands below
# and are imported lazily inside each of those commands.  Keeping them out of
# the module-level import list means `dc_toolkit evaluate_combos` and
# `compress_with_optimal` don't pay the import cost, and environments without
# (e.g.) a matplotlib install can still run the main sweep.

import warnings
warnings.filterwarnings(
    "ignore",
    message="Numcodecs codecs are not in the Zarr version 3 specification.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message="Engine 'cfgrib' loading failed", category=RuntimeWarning,
)
warnings.filterwarnings("ignore", message="overflow encountered in square")


@click.group()
def cli():
    pass


# =============================================================================
# Helpers specific to this CLI
# =============================================================================

def _size_option_callback(ctx, param, value):
    if value is None:
        return None
    try:
        return utils.parse_size(value)
    except Exception as e:
        raise click.BadParameter(f"Invalid size '{value}': {e}")


def _is_ebcc_serializer(serializer) -> bool:
    return (
        isinstance(serializer, AnyNumcodecsArrayBytesCodec)
        and isinstance(serializer.codec, EBCCZarrFilter)
    )


def _is_zfpy_serializer(serializer) -> bool:
    return isinstance(serializer, numcodecs.zarr3.ZFPY)


def _merged_store_path(where_to_write: str, dataset_file: str) -> str:
    """One .zarr store per dataset; fields live as arrays inside.

    Uses Path.stem so `foo.nc` -> `foo.zarr` (not `foo.nc.zarr`).  The store
    name is derived only from the input filename, not its directory.
    """
    dataset_stem = Path(dataset_file).stem
    return str(Path(where_to_write) / f"{dataset_stem}.zarr")


def _version_banner(component_name: str) -> str:
    """
    Return a short string with the zarr version and a few other keys, for
    debugging provenance.  Prints from rank 0 only in the CLI commands.
    """
    zarr_ver = getattr(zarr, "__version__", "unknown")
    np_ver = getattr(np, "__version__", "unknown")
    dask_ver = getattr(dask, "__version__", "unknown")
    return (
        f"[env] {component_name} | zarr={zarr_ver} | numpy={np_ver} | "
        f"dask={dask_ver}"
    )


def _check_memory_headroom(required_bytes: int, label: str, threshold: float = 0.80) -> None:
    """
    Refuse to allocate `required_bytes` if it would exceed `threshold` of
    currently-available RAM.  Aborts the whole MPI world on violation.

    This is defense-in-depth against user misconfiguration, e.g. passing
    `--eval-data-size-limit 50GB` on a 64 GB node or `--threads 256` with
    large shards on a RAM-starved box.  Catches the error before numpy /
    dask raise a MemoryError halfway through a long run.

    Default threshold is 0.80: leaves ~25% headroom over our `required_bytes`
    estimate to absorb (a) rechunk transients, which the topology banner
    notes can push real peak 1.5-2x above the documented threads * shard_mib
    figure; (b) Python / dask / MPI / numpy overhead not in the estimate;
    (c) other processes on the same node.  Going above 0.80 is risky
    because the documented rechunk transient alone can exceed the remaining
    buffer; the function emits a one-time warning when called above that.

    Caveat: psutil.virtual_memory().available reports HOST memory, not the
    cgroup limit when running inside a container or a slurm allocation with
    --mem set.  In that case the kernel/slurm OOM-killer is the real guard,
    not this function.

    psutil is a hard dependency (see pyproject.toml), so the query itself
    is always available; we only wrap the call in try/except to tolerate
    the rare environment where /proc is unreadable (unusual containers)
    and treat it as "unable to check" rather than crashing.
    """
    if threshold > 0.80 and not getattr(_check_memory_headroom, "_warned_high", False):
        click.echo(
            f"[memcheck] WARNING: threshold {threshold:.2f} exceeds the "
            f"recommended 0.80 ceiling.  The 1.5-2x rechunk transient "
            f"documented for the write peak can fit inside the remaining "
            f"buffer up to ~0.80 but not above; OOM risk increases sharply."
        )
        _check_memory_headroom._warned_high = True
    try:
        avail = psutil.virtual_memory().available
    except Exception as e:
        click.echo(
            f"[memcheck] WARNING: could not query available memory ({e}); "
            f"skipping guard for {label}."
        )
        return
    if required_bytes > threshold * avail:
        click.echo(
            f"[memcheck] REFUSING to proceed: {label} needs "
            f"{humanize.naturalsize(required_bytes, binary=True)}, which "
            f"exceeds {int(threshold*100)}% of currently-available RAM "
            f"({humanize.naturalsize(avail, binary=True)}).\n"
            f"  Reduce the relevant flag (e.g. --eval-data-size-limit, "
            f"--threads, --shard-mib), raise --memory-threshold (max 0.80 "
            f"recommended), or run on a larger node."
        )
        MPI.COMM_WORLD.Abort(1)


def _sample_signature(
    dataset_file: str,
    var: str,
    eval_data_size_limit: int,
    sample_np: np.ndarray,
) -> dict:
    """
    Build a compact, deterministic signature of the representative sample
    used to parameterise the codec space.

    Purpose: the codec-space indices written by `evaluate_combos` are only
    valid in `compress_with_optimal` when both commands see an *identical*
    sample - because statistics (Asinh.linear_width, FixedOffsetScale.offset/
    scale, EBCC chunk geometry) are derived from that sample.  We hash enough
    of the sample's identity to detect a mismatch at the start of
    `compress_with_optimal` and refuse to continue silently.

    Design:
    - Hashes the full buffer bytes (sha256).  Fast enough for 5 GB (~5s on
      modern CPUs); the alternative of hashing summary stats can alias on
      pathological data.  We pay this once per variable, once per command.
    - Includes `(shape, dtype, dataset_stem, var, eval_data_size_limit)`
      alongside the content hash so a debug message can point at the
      mismatch cause.

    Returns a plain dict (json-serialisable).

    Unsupported: object-dtype arrays.  numpy stores object arrays as a
    buffer of pointer addresses, not their referents, so the hash would
    include process-local memory addresses and be unreproducible across
    runs.  We refuse early rather than silently emit a garbage signature.
    Climate data is never object-dtype in practice, so this is defensive.
    """
    if sample_np.dtype == object:
        raise ValueError(
            "_sample_signature does not support object-dtype arrays: the "
            "buffer holds pointer addresses, not values, so the hash would "
            "be process-local and not reproducible across runs."
        )
    h = hashlib.sha256()
    # memoryview over the numpy buffer avoids an extra copy.  We force
    # contiguity at the broadcast site, so the buffer is already C-ordered.
    mv = memoryview(np.ascontiguousarray(sample_np)).cast("B")
    # Stream in 64 MiB chunks to keep the worst-case transient allocation low.
    step = 64 * 1024 * 1024
    n = len(mv)
    for i in range(0, n, step):
        h.update(mv[i:i + step])
    return {
        "dataset_stem": Path(dataset_file).stem,
        "var": var,
        "eval_data_size_limit": int(eval_data_size_limit),
        "shape": list(sample_np.shape),
        "dtype": str(sample_np.dtype),
        "nbytes": int(sample_np.nbytes),
        "sha256": h.hexdigest(),
    }


def _signature_path(where_to_write: str, var: str) -> Path:
    return Path(where_to_write) / f"sample_signature_{var}.json"


# =============================================================================
# L1 error threshold lookup
# =============================================================================
# Source of truth is a Google Sheet (column layout: "Short Name", "Unit",
# "Existing L1 error", ...).  We mirror the latest successful fetch to a
# CSV alongside this module so offline / CI / network-degraded runs still
# have something to consult.  The on-disk copy is intended to be checked
# into the repo so a fresh clone has a working fallback even before the
# first online run.

_L1_THRESHOLDS_SHEET_ID = "1lHcX-HE2WpVCOeKyDvM4iFqjlWvkd14lJlA-CUoCxMM"
_L1_THRESHOLDS_SHEET_URL = (
    f"https://docs.google.com/spreadsheets/d/{_L1_THRESHOLDS_SHEET_ID}/export?format=csv"
)
_LOCAL_L1_THRESHOLDS_PATH = (
    Path(__file__).parent / "data" / "l1_error_thresholds.csv"
)


def _load_l1_error_thresholds(rank: int) -> "pd.DataFrame | None":
    """
    Load the L1 error threshold table.

    Resolution order:
      1. Remote Google Sheet (authoritative).  On success the result is
         mirrored to `_LOCAL_L1_THRESHOLDS_PATH` so the next offline run
         has an up-to-date snapshot.
      2. Local CSV at `_LOCAL_L1_THRESHOLDS_PATH` (committed to the repo).
         Used only when the remote fetch raises - typically: no network,
         the sheet was renamed/permissioned, or the CSV export endpoint
         is briefly down.

    Returns the DataFrame on success, or `None` if neither source yields
    one.  The caller is responsible for aborting with a useful message in
    that case.

    Only the calling rank reads/writes; the resulting DataFrame is
    broadcast by the caller (we do not enter MPI here so the helper stays
    usable from non-MPI contexts as well).

    The local cache write is best-effort: if the package directory is
    read-only (e.g. installed into a system site-packages, sandboxed CI),
    we log a note and still return the in-memory DataFrame.  The remote
    payload is the same data, so a failed cache is not a fatal condition.
    """
    # ---- 1) Try remote --------------------------------------------------
    try:
        thresholds = pd.read_csv(_L1_THRESHOLDS_SHEET_URL)
    except Exception as remote_err:
        click.echo(
            f"[Rank {rank}] Remote L1 threshold fetch failed: {remote_err}.  "
            f"Trying local fallback at {_LOCAL_L1_THRESHOLDS_PATH}."
        )
    else:
        # Refresh the on-disk fallback.  Best-effort - see docstring.
        try:
            _LOCAL_L1_THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
            thresholds.to_csv(_LOCAL_L1_THRESHOLDS_PATH, index=False)
        except Exception as cache_err:
            click.echo(
                f"[Rank {rank}] Note: could not refresh local L1 threshold "
                f"cache at {_LOCAL_L1_THRESHOLDS_PATH} ({cache_err}); "
                f"continuing with the remote copy."
            )
        return thresholds

    # ---- 2) Try local fallback -----------------------------------------
    if _LOCAL_L1_THRESHOLDS_PATH.is_file():
        try:
            thresholds = pd.read_csv(_LOCAL_L1_THRESHOLDS_PATH)
            click.echo(
                f"[Rank {rank}] Loaded L1 thresholds from local fallback "
                f"({_LOCAL_L1_THRESHOLDS_PATH})."
            )
            return thresholds
        except Exception as local_err:
            click.echo(
                f"[Rank {rank}] Local L1 threshold fallback at "
                f"{_LOCAL_L1_THRESHOLDS_PATH} is unreadable: {local_err}."
            )
    else:
        click.echo(
            f"[Rank {rank}] No local L1 threshold fallback found at "
            f"{_LOCAL_L1_THRESHOLDS_PATH}."
        )
    return None


@cli.command("evaluate_combos")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--where-to-write", "where_to_write", required=True,
              type=click.Path(dir_okay=True, file_okay=False, exists=False),
              help="Directory where sweep outputs are written: per-var config "
                   "space CSV, per-rank streaming partials, consolidated "
                   "`results_{var}.parquet`, and the legacy scored-results "
                   "`.npy`.  The directory is created if it doesn't exist.")
@click.option("--field-to-compress", default=None,
              help="Field to compress [if not given, all fields will be evaluated].")
@click.option("--eval-data-size-limit", default="5GB", callback=_size_option_callback,
              show_default=True,
              help="Representative-sample size budget (e.g. '5GB', '512MiB'). "
                   "If the field fits under this, the full field is used; otherwise "
                   "a strided subsample along the leading dim is used.  "
                   "The same sample is used to compute data-derived codec parameters "
                   "(Asinh.linear_width, FixedOffsetScale.offset/scale, etc.), so "
                   "`compress_with_optimal` must be invoked with the same value for "
                   "codec-space indices to resolve to identical codec objects.")
@click.option("--threads-per-rank", type=int, default=None,
              help="Threads per MPI rank.  Default: auto-detected from cores/rank.")
@click.option("--inner-chunk-mib", type=int, default=16, show_default=True,
              help="Target size of a zarr chunk (in MiB) during in-memory evaluation. "
                   "For the measured compression ratio to reflect production conditions, "
                   "pass the same value to compress_with_optimal.")
@click.option("--max-inner-chunk-mib", type=int, default=256, show_default=True,
              help="Hard ceiling on inner chunk size (in MiB).  Only enforced "
                   "with --no-spatial-split: when one timestep already exceeds "
                   "--inner-chunk-mib and spatial splitting is disabled, the "
                   "resulting chunk may be very large; if it also exceeds this "
                   "ceiling we emit a warning.  Codec internals (zstd block "
                   "limit, blosc memory) start to misbehave around the GB mark, "
                   "so leaving the default at 256 is a safety net.")
@click.option("--spatial-split/--no-spatial-split", default=True, show_default=True,
              help="When one timestep already exceeds --inner-chunk-mib, split "
                   "spatial dims (horizontal/cell first, vertical last -- "
                   "hiopy approach) until the chunk fits the target.  Disable "
                   "with --no-spatial-split to keep one timestep per chunk "
                   "with full spatial extent (matches the hiopy on-disk "
                   "layout, but produces oversized chunks).")
@click.option("--oversubscription-check/--no-oversubscription-check", default=True,
              show_default=True,
              help="At startup, warn/abort if OMP/BLOSC/MKL thread vars aren't pinned to 1.")
@click.option("--memory-threshold", type=click.FloatRange(0.05, 0.95), default=0.80,
              show_default=True,
              help="Fraction of currently-available RAM that any single tracked "
                   "allocation is allowed to occupy before the run is aborted.  "
                   "Defaults to 0.80; values above 0.80 emit a one-time warning "
                   "because the documented 1.5-2x rechunk transient can exceed "
                   "the remaining buffer.  Hard upper bound 0.95.")
@click.option("--override-existing-l1-error", type=float, default=None,
              help="Override the existing L1 error threshold from the lookup table.")
@click.option("--compressor-class", default="all",
              help="Compressor class (case-insensitive) or 'none' to skip.")
@click.option("--filter-class", default="all",
              help="Filter class (case-insensitive) or 'none' to skip.")
@click.option("--serializer-class", default="all",
              help="Serializer class (case-insensitive) or 'none' to skip.")
@click.option("--with-lossy/--without-lossy", default=True, show_default=True)
@click.option("--with-numcodecs-wasm/--without-numcodecs-wasm", default=True, show_default=True)
@click.option("--with-ebcc/--without-ebcc", default=True, show_default=True)
@click.option("--resume/--no-resume", default=False, show_default=True,
              help="If set and a `config_space_{var}_rank{rank}.csv` already "
                   "exists in --where-to-write, skip combos already present in "
                   "it (matched by (comp_idx, filt_idx, ser_idx)).  Useful for "
                   "picking up a sweep that died partway.  The streaming CSV "
                   "is appended rather than overwritten in resume mode.")
def evaluate_combos(dataset_file,
                    where_to_write,
                    field_to_compress, eval_data_size_limit,
                    threads_per_rank, inner_chunk_mib,
                    max_inner_chunk_mib, spatial_split,
                    oversubscription_check,
                    memory_threshold,
                    override_existing_l1_error,
                    compressor_class, filter_class, serializer_class,
                    with_lossy, with_numcodecs_wasm, with_ebcc,
                    resume):
    """
    Sweep compressor x filter x serializer combinations on a representative
    sample of the field to find the best configuration.

    Parallelism
    -----------
    - MPI ranks partition the config space (config_space[rank::size]).
    - Within each rank, a ThreadPoolExecutor runs N configs concurrently.
    - Recommended launch:
        mpirun -n <NODES> --ntasks-per-node=1 dc_toolkit evaluate_combos ...
      (or srun --nodes=<N> --ntasks-per-node=1 ... on Slurm).
    - 1 MPI rank per node is REQUIRED.  Multi-rank-per-node launches are
      rejected at startup; within-node parallelism is provided by threads,
      not MPI.
    - Cores-per-rank is auto-detected via sched_getaffinity; override with
      --threads-per-rank.

    The evaluation runs entirely in memory (MemoryStore) - no disk I/O per
    combo.  Use `compress_with_optimal` afterwards to materialise the winner
    against the full field.
    """
    # -------------------------------------------------------------------------
    # Topology + dask config
    # -------------------------------------------------------------------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    node_comm, ranks_on_node, _local_rank = utils.detect_node_topology(comm)

    # ---- Enforce 1 MPI rank per node --------------------------------------
    # The refactor is designed around shared-memory threading within the node.
    # Multi-rank-per-node launches are silently wasteful (sample duplication,
    # redundant MPI broadcasts) so we reject them outright.
    if ranks_on_node > 1:
        if rank == 0:
            click.echo(
                f"[topology] ERROR: detected {ranks_on_node} MPI rank(s) per node.\n"
                f"  This toolkit requires exactly 1 rank per node; within-node\n"
                f"  parallelism is provided by threads, not MPI.\n"
                f"  Relaunch with:\n"
                f"    mpirun -n <NODES> --ntasks-per-node=1 dc_toolkit evaluate_combos ...\n"
                f"  (or on Slurm:\n"
                f"    srun --nodes=<N> --ntasks-per-node=1 dc_toolkit evaluate_combos ...)"
            )
        comm.Abort(1)

    try:
        node_comm.Free()
    except Exception:
        pass

    cores_avail = utils.detect_cores_available()
    if threads_per_rank is None:
        threads_per_rank = utils.compute_default_threads_per_rank(ranks_on_node, cores_avail)

    utils.check_thread_oversubscription(
        abort_if_unsafe=oversubscription_check, rank=rank,
    )

    # Create the output directory once on rank 0, then barrier so all ranks
    # see it before anyone tries to write into it.  `exist_ok=True` so repeat
    # runs don't fail; the barrier avoids a rank-1..N race against rank 0 on
    # filesystems that don't handle concurrent creation gracefully (certain
    # Lustre / NFS configurations).
    if rank == 0:
        os.makedirs(where_to_write, exist_ok=True)
    comm.Barrier()

    # `array.chunk-size` must be set BEFORE any xarray open() that uses
    # `chunks="auto"` for the setting to be honored by the dask graph builder.
    # The threaded dask scheduler is used for the sample read and codec-space
    # construction below (both go through dask.compute).  We switch to the
    # synchronous scheduler right before the ThreadPoolExecutor (nested
    # `with` inside the sweep) so the per-combo threads do not nest thread
    # pools.
    #
    # Wrapping in `with` scopes these settings to evaluate_combos; they
    # revert on function exit instead of leaking into the parent process.
    # Matters for notebook/compose use; no-op for single-command CLI runs.
    with dask.config.set({
        "array.chunk-size": "512MiB",
        "scheduler": "threads",
        "num_workers": threads_per_rank,
    }):

        # Topology banner is printed later, after the config_space is built, so we
        # can report the EFFECTIVE parallelism (min(size*threads, num_loops))
        # rather than an overstated theoretical peak.

        # Version + environment banner (rank 0 only) — surfaces the exact
        # zarr/numpy/dask combination that ran so a regression in compression
        # ratio can be pinned to a library upgrade.
        if rank == 0:
            click.echo(_version_banner("evaluate_combos"))

        # -------------------------------------------------------------------------
        # Fetch threshold table (rank 0) and broadcast
        #
        # Skipped entirely when the user has supplied --override-existing-l1-error,
        # because we'd never read from the table in that case.  Avoids the slow,
        # network-dependent Google Sheets call for offline/CI runs.
        #
        # `_load_l1_error_thresholds` first tries the remote Google Sheet and,
        # on failure, falls back to the CSV cached at
        # `_LOCAL_L1_THRESHOLDS_PATH` (committed to the repo).  Returns None
        # if neither source yields a usable table; we then abort with a
        # message pointing the user at --override-existing-l1-error.
        # -------------------------------------------------------------------------
        thresholds = None
        if override_existing_l1_error is None:
            if rank == 0:
                thresholds = _load_l1_error_thresholds(rank=rank)
                if thresholds is None:
                    click.echo(
                        "[Rank 0] ERROR: could not load the L1 error "
                        "threshold table from either the remote Google "
                        "Sheet or the local fallback at "
                        f"{_LOCAL_L1_THRESHOLDS_PATH}.\n"
                        "  Re-run with --override-existing-l1-error <value> "
                        "to specify a threshold explicitly, or fix "
                        "connectivity / restore the local CSV before "
                        "retrying."
                    )
                    comm.Abort(1)

                buffer = io.BytesIO()
                thresholds.to_parquet(buffer, index=False)
                data_bytes = buffer.getvalue()
            else:
                data_bytes = None

            data_bytes = comm.bcast(data_bytes, root=0)
            if rank != 0:
                thresholds = pd.read_parquet(io.BytesIO(data_bytes))

        # -------------------------------------------------------------------------
        # Open dataset (lazy; shared by all ranks; each rank gets its own handle)
        # -------------------------------------------------------------------------
        ds = utils.open_dataset(dataset_file, field_to_compress, rank=rank)

        # -------------------------------------------------------------------------
        # Per-variable loop
        # -------------------------------------------------------------------------
        for var in ds.data_vars:
            if field_to_compress is not None and field_to_compress != var:
                continue
            da = ds[var]

            if override_existing_l1_error is None:
                threshold_row = thresholds[thresholds["Short Name"] == var]
                matching_units = (
                    threshold_row.iloc[0]["Unit"] == da.attrs.get("units", None)
                    if not threshold_row.empty else None
                )
                raw_threshold = (
                    threshold_row.iloc[0]["Existing L1 error"]
                    if not threshold_row.empty and matching_units else None
                )
                if raw_threshold is None or (isinstance(raw_threshold, float) and math.isnan(raw_threshold)):
                    existing_l1_error = None
                elif isinstance(raw_threshold, str):
                    existing_l1_error = float(raw_threshold.replace(",", "."))
                else:
                    existing_l1_error = float(raw_threshold)
            else:
                existing_l1_error = override_existing_l1_error

            # If we still couldn't pin down a threshold - either because
            # the variable isn't in the lookup table, the units don't
            # match, or the relevant cell is empty/NaN, AND no manual
            # override was supplied - we have no basis for the `keep`
            # filter that drives the whole sweep.  Letting the run
            # continue would silently disable filtering for this variable
            # (every combo would be marked keep=True), defeating the
            # point of the lookup.  Abort with an actionable message.
            if existing_l1_error is None:
                if rank == 0:
                    var_units = da.attrs.get("units", "N/A")
                    click.echo(
                        f"[var] {var} | ERROR: cannot determine an L1 "
                        f"error threshold for this variable (no row "
                        f"matching Short Name='{var}' with Unit="
                        f"'{var_units}' in the threshold table, or the "
                        f"row's 'Existing L1 error' cell is empty/NaN).\n"
                        f"  Re-run with --override-existing-l1-error "
                        f"<value> to specify a threshold explicitly, or "
                        f"add an entry for '{var}' to the lookup table "
                        f"(remote Google Sheet, or the local fallback at "
                        f"{_LOCAL_L1_THRESHOLDS_PATH})."
                    )
                comm.Abort(1)

            if rank == 0:
                click.echo(
                    f"[var] {var} | units={da.attrs.get('units', 'N/A')} | "
                    f"Existing L1 error={existing_l1_error}"
                )

            # -------------------------------------------------------------------------
            # Build representative sample ONCE on rank 0, broadcast to others.
            # -------------------------------------------------------------------------
            # Memory guardrail before the sample broadcast.
            field_bytes = int(da.dtype.itemsize) * int(np.prod(da.shape))
            actual_sample_bytes = min(field_bytes, int(eval_data_size_limit))
            multiplier = 2 if (rank == 0 and size > 1) else 1
            _check_memory_headroom(
                multiplier * actual_sample_bytes,
                label=f"sample for '{var}' on rank {rank} "
                      f"({humanize.naturalsize(actual_sample_bytes, binary=True)})",
                threshold=memory_threshold,
            )

            if rank == 0:
                sample_da_local = utils.build_representative_sample(
                    da, eval_data_size_limit, rank=rank,
                )
                # .compute() forces the dask read; we want the buffer, not a lazy handle.
                sample_da_local = sample_da_local.compute()
                sample_np_local = np.ascontiguousarray(sample_da_local.values)
                sample_meta = {
                    "dims": tuple(sample_da_local.dims),
                    "attrs": dict(sample_da_local.attrs),
                    "name":  sample_da_local.name,
                }
            else:
                sample_np_local = None
                sample_meta = None

            # Bcast the numpy buffer via MPI's buffer protocol.  bcast() the small
            # metadata dict via pickle (dims + attrs are tiny).
            sample_np  = utils.broadcast_numpy(sample_np_local, comm=comm, root=0)
            sample_meta = comm.bcast(sample_meta, root=0)

            # Free the rank-0 duplicate ASAP so we fall from 2x transient to 1x
            # steady state.  The broadcast has already committed the bytes to
            # every rank's buffer; the local copy is no longer needed.
            if rank == 0:
                del sample_np_local

            # ----------------------------------------------------------------
            # Sample reproducibility hash (M5).
            #
            # Rank 0 hashes the broadcast buffer and writes a
            # sample_signature_{var}.json next to the sweep results.
            # `compress_with_optimal` will re-hash and refuse to proceed on
            # mismatch, closing the "silent wrong combo" footgun from a
            # --eval-data-size-limit drift between the two commands.
            # ----------------------------------------------------------------
            if rank == 0:
                try:
                    sig = _sample_signature(
                        dataset_file=dataset_file,
                        var=str(var),
                        eval_data_size_limit=int(eval_data_size_limit),
                        sample_np=sample_np,
                    )
                    _signature_path(where_to_write, str(var)).write_text(
                        json.dumps(sig, indent=2)
                    )
                    click.echo(
                        f"[sample-hash] {var}: sha256={sig['sha256'][:16]}… "
                        f"shape={tuple(sig['shape'])} dtype={sig['dtype']} -> "
                        f"{_signature_path(where_to_write, str(var)).name}"
                    )
                except Exception as sig_err:
                    # Non-fatal: continue the sweep even if signature write
                    # fails.  compress_with_optimal will log a softer warning
                    # instead of blocking when the signature is absent.
                    click.echo(
                        f"[sample-hash] WARNING: could not write signature "
                        f"for {var}: {sig_err}"
                    )

            # Reconstruct a DataArray view around the broadcast buffer.  The codec-
            # space builders only use .dims / .shape / .values and basic arithmetic,
            # so a thin wrapper without xarray coords is sufficient and avoids a
            # second compute() on non-root ranks.
            sample_da = xr.DataArray(
                sample_np,
                dims=sample_meta["dims"],
                attrs=sample_meta["attrs"],
                name=sample_meta["name"],
            )

            # -------------------------------------------------------------------------
            # Build codec spaces from the SAMPLE (deterministic; compress_with_optimal
            # must use the same --eval-data-size-limit to reproduce these objects).
            # -------------------------------------------------------------------------
            compressors = utils.compressor_space(sample_da, with_lossy, with_numcodecs_wasm,
                                                 with_ebcc, compressor_class)
            filters     = utils.filter_space(sample_da, with_lossy, with_numcodecs_wasm,
                                             with_ebcc, filter_class)
            serializers = utils.serializer_space(sample_da, with_lossy, with_numcodecs_wasm,
                                                 with_ebcc, serializer_class)

            num_loops = len(compressors) * len(filters) * len(serializers)
            config_space = list(itertools.product(compressors, filters, serializers))
            configs_for_rank = config_space[rank::size]

            if rank == 0:
                # Honest topology banner: if the sweep is smaller than the theoretical
                # peak parallelism, say so rather than overstating.
                theoretical_peak = size * threads_per_rank
                effective = min(theoretical_peak, num_loops)
                trailer = ""
                if effective < theoretical_peak:
                    trailer = (
                        f" (only {effective} will run concurrently; "
                        f"{num_loops} combos total)"
                    )
                click.echo(
                    f"[topology] {size} node(s) | {cores_avail} core(s)/node | "
                    f"{threads_per_rank} thread(s)/node -> peak {theoretical_peak} "
                    f"parallel evaluations{trailer}."
                )
                # Memory budget banner — mirror of the one in compress_with_optimal.
                # Rank-0 transient peak is 2x sample_size (local + broadcast buffer
                # briefly alive together, then dropped).  Steady state per rank is
                # sample + optional EBCC float32 copy + per-thread working set.
                steady_mib = int(sample_np.nbytes / 2**20)
                ebcc_overhead_mib = int(sample_np.nbytes / 2**20) if (
                    any(_is_ebcc_serializer(ser) for (_, ser) in serializers)
                    and sample_np.dtype != np.float32
                ) else 0
                thread_pool_mib = threads_per_rank * max(1, inner_chunk_mib) * 2
                click.echo(
                    f"[memory] rank-0 transient peak ~= "
                    f"{int(2 * sample_np.nbytes / 2**20)} MiB (during Bcast); "
                    f"per-rank steady ~= {steady_mib + ebcc_overhead_mib} MiB "
                    f"(sample + EBCC copy) + ~{thread_pool_mib} MiB (threads x "
                    f"2 x inner_chunk_mib)."
                )
                # If the caller omitted --field-to-compress, flag the per-var
                # Bcast cost so they're not surprised by 30 variables x 5 GB
                # on a slow fabric.
                n_vars_total = sum(
                    1 for v in ds.data_vars
                    if field_to_compress is None or v == field_to_compress
                )
                if n_vars_total > 1:
                    click.echo(
                        f"[topology] sweep will iterate {n_vars_total} variables; "
                        f"one sample Bcast per variable (~"
                        f"{humanize.naturalsize(sample_np.nbytes, binary=True)} each "
                        f"over the interconnect)."
                    )
                click.echo(
                    f"[sweep] {num_loops} combos "
                    f"({len(compressors)} x {len(filters)} x {len(serializers)}) "
                    f"split across {size} rank(s); ~{len(configs_for_rank)} per rank, "
                    f"running {threads_per_rank}-wide."
                )
                pd.DataFrame(config_space).to_csv(
                    os.path.join(where_to_write, f"config_space_{var}.csv"),
                    index=False,
                )

            # -------------------------------------------------------------------------
            # Pre-materialise a float32 view of the sample IF any EBCC combo is present
            # and the dtype isn't already float32 (avoids per-thread float32 allocations).
            # -------------------------------------------------------------------------
            sample_np_ebcc = None
            any_ebcc_local = any(
                _is_ebcc_serializer(ser) for (_, ser) in serializers
            )
            if any_ebcc_local and sample_np.dtype != np.float32:
                sample_np_ebcc = np.ascontiguousarray(
                    np.squeeze(sample_np).astype(np.float32, copy=True)
                )
            elif any_ebcc_local:
                sample_np_ebcc = np.ascontiguousarray(np.squeeze(sample_np))

            # -------------------------------------------------------------------------
            # Per-combo evaluator (runs inside a thread)
            # -------------------------------------------------------------------------
            def _evaluate_one(cfg):
                (comp_idx, compressor), (filt_idx, filt), (ser_idx, serializer) = cfg

                # Prep data + dims for this serializer's expectations
                if _is_zfpy_serializer(serializer):
                    data_np = sample_np.reshape(-1)  # flat view; no copy
                    dims = ("flat_dim",)
                elif _is_ebcc_serializer(serializer):
                    data_np = sample_np_ebcc        # shared across threads
                    dims = tuple(d for d, s in zip(sample_da.dims, sample_da.shape) if s > 1)
                    if not dims:
                        dims = sample_da.dims
                else:
                    data_np = sample_np
                    dims = sample_da.dims

                # Pipeline assembly rules (match original semantics)
                filters_ = [filt]
                compressors_ = [compressor]
                serializer_ = serializer
                local_filt_idx = filt_idx
                local_comp_idx = comp_idx
                local_ser_idx = ser_idx

                if isinstance(serializer_, AnyNumcodecsArrayBytesCodec) or filt is None:
                    filters_ = None
                    filt = None
                    local_filt_idx = -1
                if compressor is None:
                    compressors_ = None
                    local_comp_idx = -1
                if serializer is None:
                    serializer_ = "auto"
                    local_ser_idx = -1

                # Chunks for the eval memory store.  Same algorithm as the
                # persist path (compute_chunk_and_shard_shape) so the measured
                # compression ratio reflects production conditions.  When
                # --no-spatial-split is set, chunks may exceed --inner-chunk-mib
                # (one timestep, full spatial); a warning is emitted once if
                # they also exceed --max-inner-chunk-mib.
                eval_chunks = utils.compute_chunk_shape_for_eval(
                    data_np.shape, data_np.dtype,
                    target_mib=inner_chunk_mib,
                    dims=dims,
                    max_target_mib=max_inner_chunk_mib,
                    allow_spatial_split=spatial_split,
                )
                _eval_chunk_bytes = int(np.dtype(data_np.dtype).itemsize) \
                                    * int(np.prod(eval_chunks))
                if (not spatial_split
                        and _eval_chunk_bytes > max_inner_chunk_mib * 2**20
                        and not getattr(_evaluate_one, "_warned_oversize", False)):
                    click.echo(
                        f"[chunks] WARNING: --no-spatial-split produced an eval "
                        f"chunk of "
                        f"{humanize.naturalsize(_eval_chunk_bytes, binary=True)} "
                        f"(shape {eval_chunks}), which exceeds "
                        f"--max-inner-chunk-mib ({max_inner_chunk_mib} MiB).  "
                        f"Codec internals may misbehave at this size.  "
                        f"Re-enable spatial splitting or lower the field size."
                    )
                    _evaluate_one._warned_oversize = True

                ratio, errors, eucd = utils.evaluate_codec_pipeline(
                    data_np, dims,
                    filters=filters_, compressors=compressors_, serializer=serializer_,
                    chunks=eval_chunks,
                )

                return {
                    "comp_idx":   local_comp_idx,
                    "filt_idx":   local_filt_idx,
                    "ser_idx":    local_ser_idx,
                    "compressor": str(compressor),
                    "filter":     str(filt),
                    "serializer": str(serializer),
                    "ratio":      float(ratio),
                    "errors":     errors,
                    "eucd":       float(eucd),
                }

            # -------------------------------------------------------------------------
            # Thread pool: submit all combos for this rank, collect as they complete
            # -------------------------------------------------------------------------
            results = []
            raw_values_explicit_with_names = []
            failures = []

            total_local = len(configs_for_rank)
            var_sweep_t0 = time.perf_counter()

            # ----- Resume support --------------------------------------------
            # Load already-completed (comp_idx, filt_idx, ser_idx) triples from
            # the per-rank CSV if --resume is set and the CSV exists.  We skip
            # these configs on submission and APPEND (not overwrite) to the CSV
            # so the resumed run ends with one complete audit trail.
            partial_csv_path = os.path.join(
                where_to_write, f"config_space_{var}_rank{rank}.csv"
            )
            failures_csv_path = os.path.join(
                where_to_write, f"failures_{var}_rank{rank}.csv"
            )
            already_done = set()
            open_mode = "w"
            if resume and Path(partial_csv_path).is_file():
                try:
                    prev = pd.read_csv(partial_csv_path)
                    already_done = set(
                        (int(a), int(b), int(c))
                        for a, b, c in zip(
                            prev["comp_idx"], prev["filt_idx"], prev["ser_idx"]
                        )
                    )
                    open_mode = "a"
                    if rank == 0:
                        click.echo(
                            f"[resume] rank 0 found {len(already_done)} previously-"
                            f"recorded combo(s) for '{var}'; skipping."
                        )
                except Exception as resume_err:
                    if rank == 0:
                        click.echo(
                            f"[resume] WARNING: failed to parse previous "
                            f"partial CSV {partial_csv_path}: {resume_err}. "
                            f"Starting from scratch."
                        )
                    already_done = set()
                    open_mode = "w"

            def _cfg_key(cfg):
                (comp_idx, _), (filt_idx, _), (ser_idx, _) = cfg
                return (int(comp_idx), int(filt_idx), int(ser_idx))

            configs_pending = [
                cfg for cfg in configs_for_rank
                if _cfg_key(cfg) not in already_done
            ]

            # ----- Streaming CSVs --------------------------------------------
            # partial_csv: one row per successful eval (kept or filtered-out)
            # failures_csv: one row per failure (exception).  Both are flushed
            # every FLUSH_EVERY rows to keep MDS pressure down while still
            # giving crash-safety.  Header written iff the file doesn't yet
            # exist (or is empty), so resume-mode appends cleanly to a prior
            # run AND still produces a valid CSV when only one of the two
            # files was created before the crash.
            FLUSH_EVERY = 100
            # Batched flushing reduces Lustre MDS round-trips from one-per-row
            # to roughly one-per-FLUSH_EVERY-rows (10k combos -> 100 flushes
            # instead of 10k).  Still flushes on normal exit via the `with`
            # block, so a clean finish leaves the CSV fully on disk.
            rows_since_flush = 0
            failed_rows_since_flush = 0

            partial_exists = (
                open_mode == "a"
                and Path(partial_csv_path).is_file()
                and Path(partial_csv_path).stat().st_size > 0
            )
            failures_exists = (
                open_mode == "a"
                and Path(failures_csv_path).is_file()
                and Path(failures_csv_path).stat().st_size > 0
            )

            with open(partial_csv_path, open_mode, newline="") as partial_csv_file, \
                 open(failures_csv_path, open_mode, newline="") as failures_csv_file:
                partial_csv_writer = csv.writer(partial_csv_file)
                failures_csv_writer = csv.writer(failures_csv_file)
                if not partial_exists:
                    partial_csv_writer.writerow([
                        "compressor", "filter", "serializer",
                        "comp_idx", "filt_idx", "ser_idx",
                        "ratio", "l1_rel", "l2_rel", "linf_rel", "eucd",
                        "keep",
                    ])
                if not failures_exists:
                    failures_csv_writer.writerow([
                        "compressor", "filter", "serializer", "error",
                    ])
                # From here on, per-combo threads provide parallelism.  Dask runs
                # serially inside each thread to avoid nested thread pools.  The
                # synchronous-scheduler setting is scoped with `with dask.config.set`
                # so it reverts automatically when we leave the sweep block - it
                # wouldn't leak in the CLI flow (one process per command), but this
                # keeps evaluate_combos safe to import into notebooks or compose in
                # longer-lived processes.
                with dask.config.set(scheduler="synchronous"):
                    with ThreadPoolExecutor(max_workers=threads_per_rank) as pool:
                        future_to_cfg = {
                            pool.submit(_evaluate_one, cfg): cfg for cfg in configs_pending
                        }
                        for fut in as_completed(future_to_cfg):
                            cfg = future_to_cfg[fut]

                            try:
                                r = fut.result()
                            except Exception as e:
                                # Never crash the sweep on a single combo failure.
                                # Failures are logged to failures_{var}_rank{rank}.csv
                                # and aggregated across ranks at the end of the sweep.
                                (_, compressor), (_, filt), (_, serializer) = cfg
                                failures.append((str(compressor), str(filt), str(serializer), repr(e)))
                                failures_csv_writer.writerow([
                                    str(compressor), str(filt), str(serializer), repr(e),
                                ])
                                failed_rows_since_flush += 1
                                if failed_rows_since_flush >= FLUSH_EVERY:
                                    failures_csv_file.flush()
                                    failed_rows_since_flush = 0
                                utils.progress_bar(total_local, print_every=100, key=str(var))
                                continue

                            l1_rel   = r["errors"]["Relative_Error_L1"]
                            l2_rel   = r["errors"]["Relative_Error_L2"]
                            linf_rel = r["errors"]["Relative_Error_Linf"]

                            keep = True
                            if existing_l1_error is not None:
                                keep = (l1_rel <= existing_l1_error)

                            # Per-rank streaming audit row.  Written for every
                            # successful evaluation, including filtered-out ones (the
                            # `keep` column distinguishes).  Batched flushes keep MDS
                            # pressure down while still surviving a mid-sweep crash
                            # up to FLUSH_EVERY rows.
                            partial_csv_writer.writerow([
                                r["compressor"], r["filter"], r["serializer"],
                                r["comp_idx"], r["filt_idx"], r["ser_idx"],
                                r["ratio"], l1_rel, l2_rel, linf_rel, r["eucd"],
                                keep,
                            ])
                            rows_since_flush += 1
                            if rows_since_flush >= FLUSH_EVERY:
                                partial_csv_file.flush()
                                rows_since_flush = 0

                            if keep:
                                results.append((
                                    (r["compressor"], r["filter"], r["serializer"],
                                     r["comp_idx"], r["filt_idx"], r["ser_idx"]),
                                    r["ratio"], l1_rel, r["eucd"],
                                ))
                                raw_values_explicit_with_names.append((
                                    r["ratio"], l1_rel, l2_rel, linf_rel, r["eucd"],
                                    r["compressor"], r["filter"], r["serializer"],
                                ))

                            utils.progress_bar(total_local, print_every=100, key=str(var))

                # Flush any remaining buffered rows before closing the files.
                partial_csv_file.flush()
                failures_csv_file.flush()

            # ----- Aggregate failure details across ranks (M2) ---------------
            # Gather the first few failures from every rank so the user can see
            # node-local issues (bad EBCC geometry on one host, codec-library
            # mismatch on another) even when rank 0 is clean.  Limit to 5 per
            # rank to keep the pickle small.
            sample_failures = failures[:5]
            all_failures = comm.gather(sample_failures, root=0)
            total_failures = comm.reduce(len(failures), op=MPI.SUM, root=0)

            if rank == 0 and total_failures and total_failures > 0:
                click.echo(
                    f"[warning] {total_failures} combo(s) failed total across "
                    f"{size} rank(s)."
                )
                shown = 0
                for r_idx, batch in enumerate(all_failures):
                    for compressor, filt, serializer, err in batch:
                        click.echo(
                            f"  [rank {r_idx}] {compressor} | {filt} | {serializer}: {err}"
                        )
                        shown += 1
                        if shown >= 30:
                            break
                    if shown >= 30:
                        break
                if total_failures > shown:
                    click.echo(
                        f"  ... and {total_failures - shown} more "
                        f"(full details in failures_{var}_rank*.csv)."
                    )

            # Per-variable sweep timing for the run manifest.
            var_sweep_seconds = time.perf_counter() - var_sweep_t0

            # -------------------------------------------------------------------------
            # Gather + best-combo selection (rank 0)
            # -------------------------------------------------------------------------
            results_gather = comm.gather(results, root=0)
            raw_gather     = comm.gather(raw_values_explicit_with_names, root=0)

            if rank == 0:
                click.echo("[sweep] complete. Writing results...")
                results_gather = list(itertools.chain.from_iterable(results_gather))
                raw_gather     = list(itertools.chain.from_iterable(raw_gather))

                lossy_option          = "with-lossy" if with_lossy else "without-lossy"
                numcodecs_wasm_option = "with-numcodecs-wasm" if with_numcodecs_wasm else "without-numcodecs-wasm"
                ebcc_option           = "with-ebcc" if with_ebcc else "without-ebcc"
                # `var` is used unconditionally here (was `field_to_compress or "all"`)
                # so that when the caller omits --field-to-compress and we iterate
                # over every data_var, each iteration produces a distinct filename.
                score_tag = [
                    var,
                    compressor_class, filter_class, serializer_class,
                    lossy_option, numcodecs_wasm_option, ebcc_option,
                ]
                npy_path = os.path.join(
                    where_to_write,
                    os.path.basename(dataset_file) + "_" + "_".join(score_tag)
                    + "_scored_results_with_names.npy",
                )
                np.save(npy_path, np.asarray(pd.DataFrame(raw_gather)))

                # Consolidate per-rank streaming CSVs into one parquet
                partial_paths = sorted(
                    Path(where_to_write).glob(f"config_space_{var}_rank*.csv")
                )
                if partial_paths:
                    consolidated = pd.concat(
                        [pd.read_csv(p) for p in partial_paths],
                        ignore_index=True,
                    )
                    parquet_path = os.path.join(
                        where_to_write, f"results_{var}.parquet"
                    )
                    consolidated.to_parquet(parquet_path, index=False)
                    click.echo(
                        f"[sweep] consolidated {len(partial_paths)} per-rank "
                        f"CSV(s) -> {parquet_path} "
                        f"({len(consolidated)} row(s))."
                    )

                if results_gather:
                    best = max(results_gather, key=lambda x: x[1])
                    click.echo(
                        "optimal combo:\n"
                        f"compressor : {best[0][0]}\n"
                        f"filter     : {best[0][1]}\n"
                        f"serializer : {best[0][2]}\n"
                        "corresponding indices in lists of instantiated objects:\n"
                        f"compressor : {best[0][3]}\n"
                        f"filter     : {best[0][4]}\n"
                        f"serializer : {best[0][5]}\n"
                        f"Compression Ratio: {best[1]:.3f} | "
                        f"Relative L1 Error: {best[2]:.3e} | "
                        f"Euclidean Distance: {best[3]:.3e}"
                    )
                else:
                    click.echo("[sweep] no combos passed the threshold filter.")

                # -------------------------------------------------------------
                # Run manifest (machine-readable summary per variable).
                # Useful for CI / downstream tooling that wants the best combo
                # without parsing stdout.  Everything is primitive/JSON-safe.
                # -------------------------------------------------------------
                manifest = {
                    "command": "evaluate_combos",
                    "dataset_file": os.fspath(dataset_file),
                    "var": str(var),
                    "where_to_write": os.fspath(where_to_write),
                    "args": {
                        "eval_data_size_limit": int(eval_data_size_limit),
                        "threads_per_rank": int(threads_per_rank),
                        "inner_chunk_mib": int(inner_chunk_mib),
                        "max_inner_chunk_mib": int(max_inner_chunk_mib),
                        "spatial_split": bool(spatial_split),
                        "compressor_class": compressor_class,
                        "filter_class": filter_class,
                        "serializer_class": serializer_class,
                        "with_lossy": bool(with_lossy),
                        "with_numcodecs_wasm": bool(with_numcodecs_wasm),
                        "with_ebcc": bool(with_ebcc),
                        "override_existing_l1_error": override_existing_l1_error,
                        "resume": bool(resume),
                    },
                    "topology": {
                        "size": int(size),
                        "cores_avail": int(cores_avail),
                    },
                    "existing_l1_error": existing_l1_error,
                    "num_combos": int(num_loops),
                    "num_passed": int(len(results_gather)),
                    "num_failed_total": int(total_failures or 0),
                    "var_sweep_seconds": float(var_sweep_seconds),
                    "env": {
                        "zarr": getattr(zarr, "__version__", None),
                        "numpy": getattr(np, "__version__", None),
                        "dask": getattr(dask, "__version__", None),
                    },
                    "sample_signature_path": os.fspath(
                        _signature_path(where_to_write, str(var))
                    ),
                    "outputs": {
                        "npy": os.fspath(npy_path),
                        "parquet": (
                            os.fspath(parquet_path) if partial_paths else None
                        ),
                        "config_space_csv": os.fspath(
                            Path(where_to_write) / f"config_space_{var}.csv"
                        ),
                    },
                    "best": None,
                }
                if results_gather:
                    manifest["best"] = {
                        "compressor": best[0][0],
                        "filter":     best[0][1],
                        "serializer": best[0][2],
                        "comp_idx":   int(best[0][3]),
                        "filt_idx":   int(best[0][4]),
                        "ser_idx":    int(best[0][5]),
                        "ratio":      float(best[1]),
                        "l1_rel":     float(best[2]),
                        "eucd":       float(best[3]),
                    }

                manifest_path = os.path.join(
                    where_to_write, f"manifest_{var}.json"
                )
                try:
                    with open(manifest_path, "w") as mf:
                        json.dump(manifest, mf, indent=2, default=str)
                    click.echo(f"[sweep] wrote manifest -> {manifest_path}")
                except Exception as manifest_err:
                    click.echo(
                        f"[sweep] WARNING: could not write manifest "
                        f"{manifest_path}: {manifest_err}"
                    )


@cli.command("compress_with_optimal")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument("where_to_write", type=click.Path(dir_okay=True, file_okay=False, exists=False))
@click.argument("field_to_compress")
@click.argument("comp_idx", type=int)
@click.argument("filt_idx", type=int)
@click.argument("ser_idx", type=int)
@click.option("--eval-data-size-limit", default="5GB", callback=_size_option_callback,
              show_default=True,
              help="Size budget for the sample used to build the codec space "
                   "(i.e. to compute data-derived codec parameters such as "
                   "Asinh.linear_width, FixedOffsetScale.offset/scale, and EBCC "
                   "chunk geometry).  The FULL FIELD is always compressed - this "
                   "flag does NOT control what gets written.  "
                   "Must match the value used in evaluate_combos so the codec-space "
                   "indices (comp_idx, filt_idx, ser_idx) resolve to identical codec "
                   "objects.")
@click.option("--inner-chunk-mib", type=int, default=16, show_default=True,
              help="Target size of a zarr inner chunk, in MiB. "
                   "For the measured compression ratio in evaluate_combos to reflect "
                   "production conditions, pass the same value here.")
@click.option("--max-inner-chunk-mib", type=int, default=256, show_default=True,
              help="Hard ceiling on inner chunk size (in MiB).  Only enforced "
                   "with --no-spatial-split: when one timestep already exceeds "
                   "--inner-chunk-mib and spatial splitting is disabled, the "
                   "resulting chunk may be very large; if it also exceeds this "
                   "ceiling we emit a warning.")
@click.option("--spatial-split/--no-spatial-split", default=True, show_default=True,
              help="When one timestep already exceeds --inner-chunk-mib, split "
                   "spatial dims (horizontal/cell first, vertical last -- "
                   "hiopy approach) until the chunk fits the target.  Disable "
                   "with --no-spatial-split to keep one timestep per chunk "
                   "with full spatial extent (matches the hiopy on-disk "
                   "layout, but produces oversized chunks AND skips sharding "
                   "since one shard would only bundle one chunk).")
@click.option("--shard-mib", type=int, default=512, show_default=True,
              help="Target shard size in MiB. Each shard contains an integer "
                   "number of inner chunks.  When one inner chunk already "
                   "meets or exceeds this target, sharding is skipped "
                   "automatically (a shard bundling <= 1 chunk would add "
                   "only index overhead).")
@click.option("--threads", type=int, default=None,
              help="Number of dask workers used for the parallel write. "
                   "Default: auto-detected from visible cores. "
                   "Peak memory use during the write is roughly threads * shard_mib.")
@click.option("--oversubscription-check/--no-oversubscription-check", default=True,
              show_default=True,
              help="At startup, warn/abort if OMP/BLOSC/MKL thread vars aren't pinned to 1.")
@click.option("--memory-threshold", type=click.FloatRange(0.05, 0.95), default=0.80,
              show_default=True,
              help="Fraction of currently-available RAM that any single tracked "
                   "allocation is allowed to occupy before the run is aborted.  "
                   "Defaults to 0.80; values above 0.80 emit a one-time warning "
                   "because the documented 1.5-2x rechunk transient can exceed "
                   "the remaining buffer.  Hard upper bound 0.95.")
@click.option("--verify/--no-verify", default=True, show_default=True,
              help="After the write finishes, re-read the persisted store and "
                   "recompute the relative L1/L2/Linf error norms and Euclidean "
                   "distance against the in-memory original.  On by default as a "
                   "safety net (catches silent codec bugs and I/O corruption).  "
                   "Cost: a full second pass of the dataset through the reader, "
                   "which roughly doubles the wall time of compress_with_optimal. "
                   "Pass --no-verify for routine production runs where the "
                   "(compressor, filter, serializer) combo is already trusted - "
                   "e.g. re-compressing sibling fields with a combo vetted on a "
                   "prior run - and re-reading the shared Zarr store is the "
                   "bottleneck.")
@click.option("--compressor-class", default="all")
@click.option("--filter-class", default="all")
@click.option("--serializer-class", default="all")
@click.option("--with-lossy/--without-lossy", default=True, show_default=True)
@click.option("--with-numcodecs-wasm/--without-numcodecs-wasm", default=True, show_default=True)
@click.option("--with-ebcc/--without-ebcc", default=True, show_default=True)
@click.option("--force/--no-force", default=False, show_default=True,
              help="Suppress the warning emitted when the (comp_idx, filt_idx, "
                   "ser_idx) you pass does not match the best combo recorded in "
                   "manifest_{field}.json by evaluate_combos.  Default is to "
                   "warn (non-fatal) so typos and stale indices get flagged.  "
                   "Pass --force when you deliberately want to write a non-best "
                   "combo (e.g. exploring the Pareto front, testing a fallback).")
def compress_with_optimal(dataset_file, where_to_write, field_to_compress,
                          comp_idx, filt_idx, ser_idx,
                          eval_data_size_limit,
                          inner_chunk_mib, max_inner_chunk_mib,
                          spatial_split, shard_mib,
                          threads, oversubscription_check, memory_threshold,
                          verify,
                          compressor_class, filter_class, serializer_class,
                          with_lossy, with_numcodecs_wasm, with_ebcc,
                          force):
    """
    Compress a single field with the combo chosen by evaluate_combos, streaming
    directly into the shared {where_to_write}/{dataset}.zarr store under
    component=field_to_compress.

    Run this command once per field; all invocations write into the same store.
    After all fields are compressed, call `merge_compressed_fields` to
    consolidate the metadata.

    What gets compressed
    --------------------
    The FULL FIELD.  Sampling is not used to decide what to write - it is used
    only to parameterize the codec space (see below).

    Codec-space reproducibility
    ---------------------------
    Several codecs have parameters derived from data statistics: Asinh's
    linear_width comes from a quantile of |da|; FixedOffsetScale's offset and
    scale come from da.mean/std/min/max; EBCC's chunk geometry comes from the
    field shape.  These are computed inside compressor_space / filter_space /
    serializer_space to produce a list of pre-instantiated codec objects, and
    comp_idx / filt_idx / ser_idx index into those lists.

    For an index produced by evaluate_combos to resolve to the SAME codec
    object here, both commands must build the codec space the same way - which
    means computing those statistics on the same data.  We do this by sampling
    the field with build_representative_sample and feeding the sample into the
    space builders.  Because the sampling is deterministic (np.linspace), the
    two commands produce identical samples as long as they see the same
    --eval-data-size-limit value.

    Mismatch failure mode: no crash, no warning - just a codec object with
    slightly different parameters than the one that won the sweep.  The
    symptom is a worse compression ratio than evaluate_combos reported.

    Parallelism
    -----------
    This command runs as a single MPI process, but the write itself is
    parallelised by dask's threaded scheduler.  Use `--threads N` to cap the
    number of dask workers; the default auto-detects from visible cores.
    Peak in-flight memory during the write is roughly `threads * shard_mib`
    bytes - reduce `--threads` if memory-constrained.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size > 1:
        if rank == 0:
            click.echo("compress_with_optimal is not meant to run in parallel. "
                       "Launch it with a single process.")
        # Collective abort: if we got here, the user launched this with
        # `mpirun -n >1`; sys.exit on rank 0 alone would leave ranks 1..N
        # blocking at the next collective.
        comm.Abort(1)

    os.makedirs(where_to_write, exist_ok=True)

    # Version + environment banner (for parity with evaluate_combos).
    click.echo(_version_banner("compress_with_optimal"))

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Manifest cross-checks (best combo + library versions).
    #
    # evaluate_combos writes `manifest_{var}.json` containing the winning
    # (comp_idx, filt_idx, ser_idx) triple AND the zarr / numpy / dask
    # versions that produced the sweep.  We check both here:
    #
    #   1. If the user's triple differs from the manifest's best, warn
    #      (non-fatal) - catches typos and stale indices without blocking
    #      legitimate "I want to try a different combo" workflows.
    #   2. If the library versions differ from the sweep's, warn (non-
    #      fatal) - a decode-path change across versions could shift the
    #      sample bytes and make the sample-signature hash check trip for
    #      reasons unrelated to user error.  This is defense-in-depth:
    #      the hash check itself still catches the mismatch; the warning
    #      just helps the user understand WHY it happened.
    #
    # --force suppresses both warnings (it's the "I know what I'm doing"
    # escape hatch).
    #
    # Both warnings are emitted here (not from the sample-hash block
    # further down) so users see them BEFORE the expensive work
    # (dataset open, .compute() of the sample) even starts.  Cheap to
    # read a JSON file; lets a user abort fast if they realise they
    # fat-fingered an index or loaded the wrong environment module.
    # -------------------------------------------------------------------------
    manifest_path = Path(where_to_write) / f"manifest_{field_to_compress}.json"
    if manifest_path.is_file() and not force:
        try:
            manifest = json.loads(manifest_path.read_text())
            # ---- best-combo check ----
            best = manifest.get("best")
            if best is not None:
                best_triple = (int(best["comp_idx"]),
                               int(best["filt_idx"]),
                               int(best["ser_idx"]))
                user_triple = (int(comp_idx), int(filt_idx), int(ser_idx))
                if user_triple != best_triple:
                    click.echo(
                        f"[manifest] WARNING: {manifest_path.name} says best "
                        f"is {best_triple}; you passed {user_triple}."
                    )
                    click.echo(
                        f"  Best combo per manifest:  "
                        f"compressor={best['compressor']}  "
                        f"filter={best['filter']}  "
                        f"serializer={best['serializer']}  "
                        f"ratio={best['ratio']:.3f}"
                    )
                    click.echo(
                        "  Proceeding anyway.  Pass --force to suppress this "
                        "warning, or re-run with the manifest triple to use "
                        "the sweep's best combo."
                    )
            # ---- library-version check ----
            # Minor version differences (e.g. dask 2026.3.0 -> 2026.3.1) are
            # usually harmless but decode paths in xarray / netCDF4 can shift
            # bytes across version upgrades, which would trip the sample-
            # signature hash check below.  We report differences here so the
            # user can connect a hash mismatch to a library upgrade rather
            # than hunting for a flag they didn't change.
            sweep_env = manifest.get("env", {}) or {}
            current_env = {
                "zarr":  getattr(zarr, "__version__", None),
                "numpy": getattr(np,   "__version__", None),
                "dask":  getattr(dask, "__version__", None),
            }
            env_deltas = [
                (pkg, sweep_env.get(pkg), current_env.get(pkg))
                for pkg in ("zarr", "numpy", "dask")
                if sweep_env.get(pkg) is not None
                and sweep_env.get(pkg) != current_env.get(pkg)
            ]
            if env_deltas:
                click.echo(
                    f"[manifest] WARNING: library versions differ from the "
                    f"sweep that wrote {manifest_path.name}:"
                )
                for pkg, sweep_ver, now_ver in env_deltas:
                    click.echo(f"  {pkg}: sweep={sweep_ver}  now={now_ver}")
                click.echo(
                    "  If the sample-signature check below reports a hash "
                    "mismatch, a decode-path change across these versions is "
                    "a likely cause.  Either rerun evaluate_combos in the "
                    "current environment, or switch back to the sweep's "
                    "environment.  Pass --force to suppress this warning."
                )
        except Exception as manifest_err:
            # Don't block the run on an unparseable manifest - users may have
            # hand-edited it, or it may be from an older toolkit version.
            click.echo(
                f"[manifest] WARNING: could not parse {manifest_path.name}: "
                f"{manifest_err}.  Skipping best-combo check."
            )

    # -------------------------------------------------------------------------
    # Thread & dask configuration
    #
    # The write goes through dask.array.to_zarr, which will parallelise the
    # codec pipeline across shards.  We make the worker count explicit so the
    # user can control peak memory (roughly: threads * shard_mib).
    # -------------------------------------------------------------------------
    cores_avail = utils.detect_cores_available()
    if threads is None:
        threads = cores_avail
    utils.check_thread_oversubscription(
        abort_if_unsafe=oversubscription_check, rank=rank,
    )

    # Both memory guardrails (write peak + codec-space sample) are deferred
    # until after the dataset is opened, so we can check against the ACTUAL
    # data size rather than a configuration upper bound.  Checking the
    # raw `threads * shard_mib` here would spuriously abort tiny fields
    # whose total bytes are smaller than a single shard.

    # Scope the scheduler + worker-count settings to this function so they
    # don't leak if compress_with_optimal is imported and called from a
    # notebook or longer-lived process.  No-op difference for the single-
    # command CLI flow (process exits immediately after), but matches the
    # pattern used in evaluate_combos.
    with dask.config.set(scheduler="threads", num_workers=int(threads)):
        click.echo(
            f"[topology] {cores_avail} core(s) visible; "
            f"dask will use {threads} worker(s) for the write. "
            f"Peak working set ~= {threads} x shard_mib ({shard_mib} MiB) = "
            f"{threads * shard_mib} MiB (documented; rechunk transients may "
            f"push 1.5-2x this on fields with adverse source chunking)."
        )

        # Open dataset + field (lazy).  The FULL FIELD `da` is what gets
        # compressed; the sample below is used only to build the codec space.
        ds = utils.open_dataset(dataset_file, field_to_compress)
        da = ds[field_to_compress]
        field_bytes = int(da.dtype.itemsize) * int(np.prod(da.shape))

        # Memory guardrail for the write: documented peak is threads * shard_mib,
        # but capped by field_bytes - you cannot have more transient working
        # memory than there is data to process.  For a field smaller than one
        # shard, the real peak is ~field_bytes; for a multi-GB field, it
        # saturates at threads * shard_mib.  Rechunk transients can push
        # 1.5-2x above that; we check against the documented peak as a floor.
        write_peak_bytes = min(int(threads) * int(shard_mib) * 2**20, field_bytes)
        _check_memory_headroom(
            write_peak_bytes,
            label=f"compress_with_optimal write peak for '{field_to_compress}' "
                  f"(min(threads x shard_mib, field bytes) = "
                  f"{humanize.naturalsize(write_peak_bytes, binary=True)})",
            threshold=memory_threshold,
        )

        # Now that `da` is known, check the ACTUAL sample allocation size
        # against available RAM (not the budget-as-upper-bound).
        actual_sample_bytes = min(field_bytes, int(eval_data_size_limit))
        _check_memory_headroom(
            actual_sample_bytes,
            label=f"codec-space sample for '{field_to_compress}' "
                  f"({humanize.naturalsize(actual_sample_bytes, binary=True)})",
            threshold=memory_threshold,
        )

        # Build the codec space from a representative sample of the field.
        #
        # The sample here has ONE purpose: to make compressor_space / filter_space /
        # serializer_space produce the same pre-instantiated codec objects as
        # evaluate_combos produced (those builders compute data-dependent parameters
        # like Asinh.linear_width, FixedOffsetScale.offset/scale, EBCC chunk geometry).
        # `build_representative_sample` is deterministic, so as long as the user
        # passes the same --eval-data-size-limit to both commands, both commands
        # see the same sample and produce identical codec objects.
        #
        # The sample is NOT what gets compressed - we compress `da` (the full field)
        # below.  This flag does not control write behavior.
        sample_for_codec_space = utils.build_representative_sample(
            da, eval_data_size_limit,
        ).compute()

        # --------------------------------------------------------------------
        # Sample reproducibility hash (M5 verification side).
        #
        # If evaluate_combos wrote a sample_signature_{var}.json in this
        # directory, recompute the hash here and compare.  Mismatch means
        # the codec-space objects won't match the ones that won the sweep,
        # so comp_idx / filt_idx / ser_idx resolve to a DIFFERENT codec
        # than the user thinks.  Refuse to continue.
        #
        # The signature file being ABSENT is not an error — evaluate_combos
        # may have been run with an older version of this toolkit, or the
        # user may have hand-picked indices from another source.  In that
        # case we warn once and proceed on trust.
        # --------------------------------------------------------------------
        sig_path = _signature_path(where_to_write, str(field_to_compress))
        if sig_path.is_file():
            try:
                expected = json.loads(sig_path.read_text())
                sample_np_view = np.ascontiguousarray(
                    sample_for_codec_space.values
                )
                observed = _sample_signature(
                    dataset_file=dataset_file,
                    var=str(field_to_compress),
                    eval_data_size_limit=int(eval_data_size_limit),
                    sample_np=sample_np_view,
                )
                # Compare the meaningful fields.  `dataset_stem` and `var`
                # must match.  `shape`/`dtype`/`nbytes`/`sha256` must match.
                # `eval_data_size_limit` mismatch is the most common cause.
                fields_to_check = (
                    "dataset_stem", "var", "eval_data_size_limit",
                    "shape", "dtype", "nbytes", "sha256",
                )
                mismatches = [
                    f for f in fields_to_check
                    if expected.get(f) != observed.get(f)
                ]
                if mismatches:
                    click.echo(
                        f"[sample-hash] MISMATCH vs {sig_path.name}: "
                        f"differing fields = {mismatches}"
                    )
                    for f in mismatches:
                        click.echo(
                            f"  {f}: expected={expected.get(f)} "
                            f"observed={observed.get(f)}"
                        )
                    click.echo(
                        "  The codec-space indices written by evaluate_combos "
                        "will resolve to different codec objects than the "
                        "sweep measured.  Most common cause: different "
                        "--eval-data-size-limit between the two commands.  "
                        "Re-run compress_with_optimal with the matching flag."
                    )
                    comm.Abort(1)
                else:
                    click.echo(
                        f"[sample-hash] OK, matches {sig_path.name} "
                        f"(sha256={observed['sha256'][:16]}…)."
                    )
                # Free the transient copy; sample_for_codec_space (the xarray
                # wrapper) still holds the underlying buffer via its .values.
                del sample_np_view
            except Exception as sig_err:
                click.echo(
                    f"[sample-hash] WARNING: could not verify signature "
                    f"{sig_path.name}: {sig_err}.  Proceeding without check."
                )
        else:
            click.echo(
                f"[sample-hash] no {sig_path.name} found - proceeding on "
                f"trust.  (For the full safety net, run evaluate_combos "
                f"first with the same --where-to-write.)"
            )

        compressors = utils.compressor_space(sample_for_codec_space, with_lossy,
                                             with_numcodecs_wasm, with_ebcc, compressor_class)
        filters     = utils.filter_space(sample_for_codec_space, with_lossy,
                                         with_numcodecs_wasm, with_ebcc, filter_class)
        serializers = utils.serializer_space(sample_for_codec_space, with_lossy,
                                             with_numcodecs_wasm, with_ebcc, serializer_class)

        # Index validation
        for name, idx, arr in [("comp_idx", comp_idx, compressors),
                               ("filt_idx", filt_idx, filters),
                               ("ser_idx",  ser_idx,  serializers)]:
            if not (-1 <= idx < len(arr)):
                click.echo(f"Invalid {name}: {idx} (must be in [-1, {len(arr) - 1}])")
                comm.Abort(1)

        optimal_compressor = compressors[comp_idx][1] if comp_idx != -1 else None
        optimal_filter     = filters[filt_idx][1]     if filt_idx != -1 else None
        optimal_serializer = serializers[ser_idx][1]  if ser_idx  != -1 else None

        # Per-serializer data shaping (on the FULL field, not the sample).
        # zfpy and EBCC have incompatible shape expectations; at most one branch
        # fires (elif documents that intent).
        data_to_persist = da
        if _is_zfpy_serializer(optimal_serializer):
            data_to_persist = da.stack(flat_dim=da.dims)
        elif _is_ebcc_serializer(optimal_serializer):
            data_to_persist = da.squeeze().astype("float32")

        # Pipeline assembly rules (same semantics as the original)
        filters_ = [optimal_filter]
        compressors_ = [optimal_compressor]
        serializer_ = optimal_serializer
        if isinstance(serializer_, AnyNumcodecsArrayBytesCodec) or optimal_filter is None:
            filters_ = None
        if optimal_compressor is None:
            compressors_ = None
        if optimal_serializer is None:
            serializer_ = "auto"

        # Compute sharding geometry for the FULL field.  Passes dim names so
        # vertical-like dims are kept whole when spatial splitting is needed
        # (hiopy approach).  shards may come back as None -- that signals
        # "skip sharding" because one inner chunk already meets the shard
        # target (a shard would bundle <= 1 chunk and add only index overhead).
        inner_chunks, shards = utils.compute_chunk_and_shard_shape(
            data_to_persist.shape, data_to_persist.dtype,
            inner_mib=inner_chunk_mib, shard_mib=shard_mib,
            dims=tuple(data_to_persist.dims),
            max_inner_mib=max_inner_chunk_mib,
            allow_spatial_split=spatial_split,
        )

        _itemsize = int(data_to_persist.dtype.itemsize)
        _inner_bytes = _itemsize * int(np.prod(inner_chunks))
        _shard_bytes = (_itemsize * int(np.prod(shards))
                        if shards is not None else _inner_bytes)
        if (not spatial_split
                and _inner_bytes > max_inner_chunk_mib * 2**20
                and rank == 0):
            click.echo(
                f"[chunks] WARNING: --no-spatial-split produced an inner chunk of "
                f"{humanize.naturalsize(_inner_bytes, binary=True)} "
                f"(shape {inner_chunks}), exceeding --max-inner-chunk-mib "
                f"({max_inner_chunk_mib} MiB).  Codec internals (zstd block "
                f"limit, blosc memory) may misbehave at this size.  Consider "
                f"re-enabling spatial splitting."
            )

        # Open (or create) the shared merged store.  mode='a' means new fields are
        # added alongside any fields previously written.
        merged_path = _merged_store_path(where_to_write, dataset_file)
        os.makedirs(Path(merged_path).parent, exist_ok=True)
        store = zarr.storage.LocalStore(merged_path, read_only=False)
        # Ensure a root group exists.  If the store is corrupted or the path is
        # unwritable we want to surface that now, not deep inside persist_with_codec_pipeline.
        try:
            zarr.open_group(store, mode="a", zarr_format=3)
        except Exception as e:
            click.echo(
                f"[persist] ERROR: cannot open or create zarr group at {merged_path}: {e}"
            )
            raise

        if shards is None:
            click.echo(
                f"[persist] {field_to_compress} -> {merged_path} "
                f"(inner chunks={inner_chunks}, "
                f"{humanize.naturalsize(_inner_bytes, binary=True)}; "
                f"sharding skipped -- one chunk >= shard target)"
            )
        else:
            click.echo(
                f"[persist] {field_to_compress} -> {merged_path} "
                f"(inner chunks={inner_chunks}, "
                f"{humanize.naturalsize(_inner_bytes, binary=True)}; "
                f"shards={shards}, "
                f"{humanize.naturalsize(_shard_bytes, binary=True)})"
            )

        # Refined memory guardrail using ACTUAL write-unit bytes (one task =
        # one shard if sharded, one chunk otherwise).  The earlier check at
        # the top of the dask context used `threads * shard_mib` as an
        # upper-bound estimate, but that can under-count when chunks are
        # oversized (--no-spatial-split + huge timestep) or over-count when
        # the field is small.  Now that we know the real geometry, re-check.
        _write_unit_bytes = _shard_bytes  # == _inner_bytes when shards is None
        _real_write_peak = min(int(threads) * int(_write_unit_bytes), field_bytes)
        _check_memory_headroom(
            _real_write_peak,
            label=f"compress_with_optimal real write peak for "
                  f"'{field_to_compress}' (threads x write-unit-bytes = "
                  f"{humanize.naturalsize(_real_write_peak, binary=True)})",
            threshold=memory_threshold,
        )

        persist_t0 = time.perf_counter()
        ratio, errors, eucd = utils.persist_with_codec_pipeline(
            data_to_persist, store,
            component=field_to_compress,
            filters=filters_, compressors=compressors_, serializer=serializer_,
            inner_chunks=inner_chunks, shards=shards,
            verify=verify, verbose=False, rank=rank,
        )
        persist_seconds = time.perf_counter() - persist_t0

        # Compose the summary.  Error metrics are only defined when --verify is on
        # (persist_with_codec_pipeline returns errors=None, eucd=None otherwise),
        # so we gate that tail of the message rather than crashing on None indexing.
        summary = (
            "optimal combo:\n"
            f"compressor : {optimal_compressor}\n"
            f"filter     : {optimal_filter}\n"
            f"serializer : {optimal_serializer}\n"
            "corresponding indices in lists of instantiated objects:\n"
            f"compressor : {comp_idx}\n"
            f"filter     : {filt_idx}\n"
            f"serializer : {ser_idx}\n"
            f"Compression Ratio: {ratio:.3f}"
        )
        if verify:
            summary += (
                f" | Relative L1 Error: {errors['Relative_Error_L1']:.3e}"
                f" | Euclidean Distance: {eucd:.3e}"
            )
        else:
            summary += "  (error metrics skipped: --no-verify)"
        click.echo(summary)

        # ------------------------------------------------------------------
        # Per-field persist manifest (machine-readable).
        # Mirrors the evaluate_combos manifest so a downstream tool can pick
        # up the exact combo that was written, which shards were produced,
        # and how long it took.
        # ------------------------------------------------------------------
        persist_manifest = {
            "command": "compress_with_optimal",
            "dataset_file": os.fspath(dataset_file),
            "var": str(field_to_compress),
            "where_to_write": os.fspath(where_to_write),
            "merged_store": merged_path,
            "args": {
                "comp_idx": int(comp_idx),
                "filt_idx": int(filt_idx),
                "ser_idx":  int(ser_idx),
                "eval_data_size_limit": int(eval_data_size_limit),
                "inner_chunk_mib": int(inner_chunk_mib),
                "max_inner_chunk_mib": int(max_inner_chunk_mib),
                "spatial_split": bool(spatial_split),
                "shard_mib": int(shard_mib),
                "threads": int(threads),
                "verify": bool(verify),
                "force": bool(force),
                "compressor_class": compressor_class,
                "filter_class": filter_class,
                "serializer_class": serializer_class,
            },
            "inner_chunks": list(inner_chunks),
            "inner_chunk_bytes": int(_inner_bytes),
            "shards": (list(shards) if shards is not None else None),
            "shard_bytes": (int(_shard_bytes) if shards is not None else None),
            "sharding_skipped": bool(shards is None),
            "compressor": str(optimal_compressor),
            "filter":     str(optimal_filter),
            "serializer": str(optimal_serializer),
            "ratio": float(ratio),
            "errors": {k: float(v) for k, v in (errors or {}).items()},
            "eucd": (float(eucd) if eucd is not None else None),
            "persist_seconds": float(persist_seconds),
            "env": {
                "zarr": getattr(zarr, "__version__", None),
                "numpy": getattr(np, "__version__", None),
                "dask": getattr(dask, "__version__", None),
            },
        }
        persist_manifest_path = os.path.join(
            where_to_write, f"persist_manifest_{field_to_compress}.json"
        )
        try:
            with open(persist_manifest_path, "w") as pmf:
                json.dump(persist_manifest, pmf, indent=2, default=str)
            click.echo(f"[persist] wrote manifest -> {persist_manifest_path}")
        except Exception as persist_manifest_err:
            click.echo(
                f"[persist] WARNING: could not write manifest "
                f"{persist_manifest_path}: {persist_manifest_err}"
            )

        # Release the LocalStore handles.  At CLI-shape this is cosmetic (the
        # process exits next), but keeps the function well-behaved when
        # imported and called from a longer-lived process.
        close = getattr(store, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


@cli.command("compress_fields_from_results")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument("where_to_write", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option("--vars", "vars_filter", default=None,
              help="Comma-separated list of variable names to process. "
                   "Default: every variable for which a results_{var}.parquet "
                   "or manifest_{var}.json exists in --where-to-write.")
@click.option("--eval-data-size-limit", default="5GB", callback=_size_option_callback,
              show_default=True,
              help="Must match the value used in the prior evaluate_combos run.")
@click.option("--inner-chunk-mib", type=int, default=16, show_default=True)
@click.option("--max-inner-chunk-mib", type=int, default=256, show_default=True,
              help="Hard ceiling on inner chunk size (MiB).  Triggers a warning "
                   "if exceeded under --no-spatial-split.")
@click.option("--spatial-split/--no-spatial-split", default=True, show_default=True,
              help="When one timestep already exceeds --inner-chunk-mib, split "
                   "spatial dims (horizontal/cell first, vertical last) until "
                   "the chunk fits target.  Pass --no-spatial-split to keep "
                   "one timestep per chunk regardless of size (hiopy-style "
                   "layout; sharding is then skipped automatically).")
@click.option("--shard-mib", type=int, default=512, show_default=True)
@click.option("--threads", type=int, default=None,
              help="Dask workers for the write.  Default: auto-detected.")
@click.option("--oversubscription-check/--no-oversubscription-check", default=True,
              show_default=True)
@click.option("--memory-threshold", type=click.FloatRange(0.05, 0.95), default=0.80,
              show_default=True,
              help="Fraction of currently-available RAM that any single tracked "
                   "allocation is allowed to occupy before the run is aborted.  "
                   "Defaults to 0.80; values above 0.80 emit a one-time warning "
                   "because the documented 1.5-2x rechunk transient can exceed "
                   "the remaining buffer.  Hard upper bound 0.95.")
@click.option("--verify/--no-verify", default=True, show_default=True)
@click.option("--compressor-class", default="all")
@click.option("--filter-class", default="all")
@click.option("--serializer-class", default="all")
@click.option("--with-lossy/--without-lossy", default=True, show_default=True)
@click.option("--with-numcodecs-wasm/--without-numcodecs-wasm", default=True, show_default=True)
@click.option("--with-ebcc/--without-ebcc", default=True, show_default=True)
@click.option("--skip-existing/--no-skip-existing", default=True, show_default=True,
              help="If a field is already present in the merged store, skip it. "
                   "Disable with --no-skip-existing to force re-compression.")
@click.option("--continue-on-error/--no-continue-on-error", default=True, show_default=True,
              help="If compressing one field fails, log and continue with the "
                   "rest (default).  Disable to fail the whole run on first error.")
def compress_fields_from_results(dataset_file, where_to_write, vars_filter,
                                  eval_data_size_limit, inner_chunk_mib,
                                  max_inner_chunk_mib, spatial_split, shard_mib,
                                  threads, oversubscription_check, memory_threshold,
                                  verify,
                                  compressor_class, filter_class, serializer_class,
                                  with_lossy, with_numcodecs_wasm, with_ebcc,
                                  skip_existing, continue_on_error):
    """
    Batch wrapper around compress_with_optimal.

    Reads the best (comp_idx, filt_idx, ser_idx) per variable from
    `manifest_{var}.json` (preferred) or `results_{var}.parquet` (fallback),
    then compresses each variable into the shared `{dataset}.zarr` store.
    The dataset is opened once and re-used across variables.

    This is the command most production pipelines want after an
    `evaluate_combos` run - it closes the loop without forcing the user to
    glue together N per-field invocations by hand.

    Launch as a SINGLE process (no mpirun).  Parallelism inside the write is
    provided by dask's threaded scheduler, same as compress_with_optimal.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size > 1:
        if rank == 0:
            click.echo("compress_fields_from_results is not meant to run in parallel. "
                       "Launch it with a single process.")
        comm.Abort(1)

    click.echo(_version_banner("compress_fields_from_results"))

    # Thread + dask config (same pattern as compress_with_optimal)
    cores_avail = utils.detect_cores_available()
    if threads is None:
        threads = cores_avail
    utils.check_thread_oversubscription(
        abort_if_unsafe=oversubscription_check, rank=rank,
    )
    # Per-variable memory checks (write peak AND codec-space sample) happen
    # inside the loop below, once we know each variable's actual size.
    # A single up-front `threads * shard_mib` check would spuriously abort
    # on small fields (e.g. tigge files where the field is smaller than
    # one shard) on memory-constrained nodes; a sum-based pre-check would
    # miss that variables are processed sequentially, not concurrently.

    # --------------------------------------------------------------------
    # Resolve (var, comp_idx, filt_idx, ser_idx) list from the where_to_write
    # directory.  Prefer manifest_{var}.json because it encodes exact codec
    # spellings and was designed for this; fall back to best-ratio from
    # results_{var}.parquet if the manifest is missing.
    #
    # We also capture the env block from the first manifest we read so we
    # can warn once if the sweep ran with different library versions than
    # this batch run (decode-path drift -> signature mismatches later).
    # --------------------------------------------------------------------
    wtw = Path(where_to_write)
    candidates = []
    sweep_env = None
    for mpath in sorted(wtw.glob("manifest_*.json")):
        var_name = mpath.stem.removeprefix("manifest_")
        try:
            m = json.loads(mpath.read_text())
            best = m.get("best")
            if best is None:
                click.echo(f"[batch] {var_name}: manifest has no best combo; skipping.")
                continue
            candidates.append({
                "var": var_name,
                "comp_idx": int(best["comp_idx"]),
                "filt_idx": int(best["filt_idx"]),
                "ser_idx":  int(best["ser_idx"]),
                "source": f"manifest {mpath.name}",
            })
            if sweep_env is None and m.get("env"):
                sweep_env = m["env"]
        except Exception as e:
            click.echo(f"[batch] WARNING: failed to parse {mpath}: {e}")

    # One-shot library-version cross-check.  Same rationale as the one in
    # compress_with_optimal: if zarr/numpy/dask differ between the sweep
    # and now, the sample bytes may shift (decode path) and the per-var
    # signature checks in the loop below may trip for environmental rather
    # than user reasons.  Reporting here connects the two for the user.
    if sweep_env:
        current_env = {
            "zarr":  getattr(zarr, "__version__", None),
            "numpy": getattr(np,   "__version__", None),
            "dask":  getattr(dask, "__version__", None),
        }
        env_deltas = [
            (pkg, sweep_env.get(pkg), current_env.get(pkg))
            for pkg in ("zarr", "numpy", "dask")
            if sweep_env.get(pkg) is not None
            and sweep_env.get(pkg) != current_env.get(pkg)
        ]
        if env_deltas:
            click.echo(
                "[batch] WARNING: library versions differ from the sweep "
                "that wrote these manifests:"
            )
            for pkg, sweep_ver, now_ver in env_deltas:
                click.echo(f"  {pkg}: sweep={sweep_ver}  now={now_ver}")
            click.echo(
                "  If per-variable signature checks below report hash "
                "mismatches, a decode-path change across these versions "
                "is a likely cause.  Either rerun evaluate_combos in the "
                "current environment, or switch back to the sweep's "
                "environment."
            )

    # Any parquet files without a companion manifest? Take best-ratio from them.
    known_vars = {c["var"] for c in candidates}
    for ppath in sorted(wtw.glob("results_*.parquet")):
        var_name = ppath.stem.removeprefix("results_")
        if var_name in known_vars:
            continue
        try:
            dfp = pd.read_parquet(ppath)
            kept = dfp[dfp["keep"] == True] if "keep" in dfp.columns else dfp
            if len(kept) == 0:
                click.echo(f"[batch] {var_name}: no kept rows in {ppath.name}; skipping.")
                continue
            best_row = kept.sort_values("ratio", ascending=False).iloc[0]
            candidates.append({
                "var": var_name,
                "comp_idx": int(best_row["comp_idx"]),
                "filt_idx": int(best_row["filt_idx"]),
                "ser_idx":  int(best_row["ser_idx"]),
                "source": f"parquet {ppath.name}",
            })
        except Exception as e:
            click.echo(f"[batch] WARNING: failed to parse {ppath}: {e}")

    if vars_filter:
        wanted = set(v.strip() for v in vars_filter.split(",") if v.strip())
        candidates = [c for c in candidates if c["var"] in wanted]
        missing = wanted - {c["var"] for c in candidates}
        if missing:
            click.echo(
                f"[batch] WARNING: --vars specified {sorted(missing)} "
                f"but no manifest/parquet was found for those."
            )

    if not candidates:
        click.echo(
            "[batch] ERROR: no variables to compress. Did evaluate_combos run "
            "against the same --where-to-write?"
        )
        comm.Abort(1)

    click.echo(
        f"[batch] will compress {len(candidates)} field(s): "
        f"{', '.join(c['var'] for c in candidates)}"
    )

    # Open dataset ONCE; pass the same da to each iteration.
    ds = utils.open_dataset(dataset_file, field_to_compress=None, rank=rank)

    merged_path = _merged_store_path(where_to_write, dataset_file)
    os.makedirs(Path(merged_path).parent, exist_ok=True)

    # Inspect the merged store (if any) to honor --skip-existing.
    existing_arrays = set()
    if Path(merged_path).is_dir():
        try:
            store_ro = zarr.storage.LocalStore(merged_path, read_only=True)
            g_ro = zarr.open_group(store_ro, mode="r")
            existing_arrays = set(g_ro.array_keys())
            close_ro = getattr(store_ro, "close", None)
            if callable(close_ro):
                try:
                    close_ro()
                except Exception:
                    pass
        except Exception:
            pass

    # ---- per-field loop ----
    results_by_var = {}
    any_error = False

    with dask.config.set(scheduler="threads", num_workers=int(threads)):
        for idx, c in enumerate(candidates, start=1):
            var = c["var"]
            click.echo(
                f"\n[batch] ({idx}/{len(candidates)}) {var} from {c['source']}: "
                f"comp={c['comp_idx']} filt={c['filt_idx']} ser={c['ser_idx']}"
            )
            if skip_existing and var in existing_arrays:
                click.echo(f"[batch] {var} already in {merged_path}; skipping.")
                results_by_var[var] = {"status": "skipped-existing"}
                continue
            if var not in ds.data_vars:
                msg = f"variable '{var}' not in dataset"
                if continue_on_error:
                    click.echo(f"[batch] WARNING: {msg}; skipping.")
                    results_by_var[var] = {"status": "missing-from-dataset"}
                    continue
                else:
                    click.echo(f"[batch] ERROR: {msg}; aborting (use "
                               f"--continue-on-error to skip).")
                    comm.Abort(1)

            try:
                field_t0 = time.perf_counter()
                da = ds[var]

                # Per-variable memory guards: check the ACTUAL allocation
                # size against available RAM, not configuration upper bounds.
                # Both the write peak (threads * shard_mib) and the codec-
                # space sample (eval_data_size_limit) are upper bounds; for
                # fields smaller than those bounds, the real allocation is
                # capped by field_bytes.  Aborting on the upper bound would
                # spuriously trip on tiny fields (e.g. tigge dx=2) on
                # memory-constrained nodes.
                field_bytes = int(da.dtype.itemsize) * int(np.prod(da.shape))

                write_peak_bytes = min(
                    int(threads) * int(shard_mib) * 2**20,
                    field_bytes,
                )
                _check_memory_headroom(
                    write_peak_bytes,
                    label=f"write peak for '{var}' "
                          f"(min(threads x shard_mib, field bytes) = "
                          f"{humanize.naturalsize(write_peak_bytes, binary=True)})",
                    threshold=memory_threshold,
                )

                actual_sample_bytes = min(field_bytes, int(eval_data_size_limit))
                _check_memory_headroom(
                    actual_sample_bytes,
                    label=f"codec-space sample for '{var}' "
                          f"({humanize.naturalsize(actual_sample_bytes, binary=True)})",
                    threshold=memory_threshold,
                )

                # Build codec space from the sample (same contract as
                # compress_with_optimal).  Verify against signature when present.
                sample_for_codec_space = utils.build_representative_sample(
                    da, eval_data_size_limit,
                ).compute()

                sig_path = _signature_path(where_to_write, str(var))
                if sig_path.is_file():
                    try:
                        expected = json.loads(sig_path.read_text())
                        sample_np_view = np.ascontiguousarray(
                            sample_for_codec_space.values
                        )
                        observed = _sample_signature(
                            dataset_file=dataset_file,
                            var=str(var),
                            eval_data_size_limit=int(eval_data_size_limit),
                            sample_np=sample_np_view,
                        )
                        mismatches = [
                            f for f in (
                                "dataset_stem", "var", "eval_data_size_limit",
                                "shape", "dtype", "nbytes", "sha256",
                            )
                            if expected.get(f) != observed.get(f)
                        ]
                        del sample_np_view
                        if mismatches:
                            click.echo(
                                f"[sample-hash] MISMATCH for {var}: "
                                f"differing fields = {mismatches}"
                            )
                            if continue_on_error:
                                click.echo(
                                    f"[batch] skipping {var} (use matching "
                                    f"--eval-data-size-limit to fix)."
                                )
                                results_by_var[var] = {"status": "signature-mismatch"}
                                continue
                            comm.Abort(1)
                    except Exception as sig_err:
                        click.echo(
                            f"[sample-hash] WARNING {var}: {sig_err}; proceeding."
                        )

                compressors = utils.compressor_space(
                    sample_for_codec_space, with_lossy, with_numcodecs_wasm,
                    with_ebcc, compressor_class,
                )
                filters_space = utils.filter_space(
                    sample_for_codec_space, with_lossy, with_numcodecs_wasm,
                    with_ebcc, filter_class,
                )
                serializers = utils.serializer_space(
                    sample_for_codec_space, with_lossy, with_numcodecs_wasm,
                    with_ebcc, serializer_class,
                )

                comp_idx = c["comp_idx"]; filt_idx = c["filt_idx"]; ser_idx = c["ser_idx"]
                for name, idx2, arr in [("comp_idx", comp_idx, compressors),
                                        ("filt_idx", filt_idx, filters_space),
                                        ("ser_idx",  ser_idx,  serializers)]:
                    if not (-1 <= idx2 < len(arr)):
                        raise IndexError(
                            f"Invalid {name}: {idx2} (must be in [-1, {len(arr)-1}]) for {var}"
                        )

                optimal_compressor = compressors[comp_idx][1] if comp_idx != -1 else None
                optimal_filter     = filters_space[filt_idx][1] if filt_idx != -1 else None
                optimal_serializer = serializers[ser_idx][1]  if ser_idx  != -1 else None

                data_to_persist = da
                if _is_zfpy_serializer(optimal_serializer):
                    data_to_persist = da.stack(flat_dim=da.dims)
                elif _is_ebcc_serializer(optimal_serializer):
                    data_to_persist = da.squeeze().astype("float32")

                filters_ = [optimal_filter]
                compressors_ = [optimal_compressor]
                serializer_ = optimal_serializer
                if isinstance(serializer_, AnyNumcodecsArrayBytesCodec) or optimal_filter is None:
                    filters_ = None
                if optimal_compressor is None:
                    compressors_ = None
                if optimal_serializer is None:
                    serializer_ = "auto"

                inner_chunks, shards = utils.compute_chunk_and_shard_shape(
                    data_to_persist.shape, data_to_persist.dtype,
                    inner_mib=inner_chunk_mib, shard_mib=shard_mib,
                    dims=tuple(data_to_persist.dims),
                    max_inner_mib=max_inner_chunk_mib,
                    allow_spatial_split=spatial_split,
                )

                _itemsize = int(data_to_persist.dtype.itemsize)
                _inner_bytes = _itemsize * int(np.prod(inner_chunks))
                _shard_bytes = (_itemsize * int(np.prod(shards))
                                if shards is not None else _inner_bytes)
                if (not spatial_split
                        and _inner_bytes > max_inner_chunk_mib * 2**20
                        and rank == 0):
                    click.echo(
                        f"[chunks] WARNING: --no-spatial-split for '{var}' "
                        f"produced an inner chunk of "
                        f"{humanize.naturalsize(_inner_bytes, binary=True)} "
                        f"(shape {inner_chunks}), exceeding "
                        f"--max-inner-chunk-mib ({max_inner_chunk_mib} MiB).  "
                        f"Codec internals may misbehave at this size."
                    )

                # Refined memory guardrail using ACTUAL write-unit bytes.
                _write_unit_bytes = _shard_bytes
                _real_write_peak = min(int(threads) * int(_write_unit_bytes),
                                       field_bytes)
                _check_memory_headroom(
                    _real_write_peak,
                    label=f"real write peak for '{var}' "
                          f"(threads x write-unit-bytes = "
                          f"{humanize.naturalsize(_real_write_peak, binary=True)})",
                    threshold=memory_threshold,
                )

                store = zarr.storage.LocalStore(merged_path, read_only=False)
                try:
                    try:
                        zarr.open_group(store, mode="a", zarr_format=3)
                    except Exception as e:
                        click.echo(
                            f"[persist] ERROR opening group at {merged_path}: {e}"
                        )
                        raise
                    if shards is None:
                        click.echo(
                            f"[persist] {var} -> {merged_path} "
                            f"(inner chunks={inner_chunks}, "
                            f"{humanize.naturalsize(_inner_bytes, binary=True)}; "
                            f"sharding skipped)"
                        )
                    else:
                        click.echo(
                            f"[persist] {var} -> {merged_path} "
                            f"(inner chunks={inner_chunks}, "
                            f"{humanize.naturalsize(_inner_bytes, binary=True)}; "
                            f"shards={shards}, "
                            f"{humanize.naturalsize(_shard_bytes, binary=True)})"
                        )
                    ratio, errors, eucd = utils.persist_with_codec_pipeline(
                        data_to_persist, store,
                        component=var,
                        filters=filters_, compressors=compressors_, serializer=serializer_,
                        inner_chunks=inner_chunks, shards=shards,
                        verify=verify, verbose=False, rank=rank,
                    )
                finally:
                    close = getattr(store, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass

                field_seconds = time.perf_counter() - field_t0
                summary = f"{var}: ratio={ratio:.3f}"
                if verify:
                    summary += (
                        f" L1_rel={errors['Relative_Error_L1']:.3e} "
                        f"eucd={eucd:.3e}"
                    )
                summary += f"  ({field_seconds:.1f}s)"
                click.echo(f"[batch] {summary}")
                results_by_var[var] = {
                    "status": "ok",
                    "ratio": float(ratio),
                    "errors": {k: float(v) for k, v in (errors or {}).items()},
                    "eucd": (float(eucd) if eucd is not None else None),
                    "seconds": float(field_seconds),
                    "comp_idx": int(comp_idx),
                    "filt_idx": int(filt_idx),
                    "ser_idx":  int(ser_idx),
                }

            except Exception as field_err:
                any_error = True
                click.echo(f"[batch] ERROR on {var}: {field_err!r}")
                results_by_var[var] = {"status": "error", "error": repr(field_err)}
                if not continue_on_error:
                    raise

    # Summary manifest for the whole batch.
    batch_manifest_path = os.path.join(where_to_write, "batch_manifest.json")
    try:
        with open(batch_manifest_path, "w") as bmf:
            json.dump(
                {
                    "command": "compress_fields_from_results",
                    "dataset_file": os.fspath(dataset_file),
                    "where_to_write": os.fspath(where_to_write),
                    "merged_store": merged_path,
                    "results": results_by_var,
                    "any_error": any_error,
                },
                bmf, indent=2, default=str,
            )
        click.echo(f"\n[batch] wrote summary -> {batch_manifest_path}")
    except Exception as bmf_err:
        click.echo(f"[batch] WARNING: could not write batch manifest: {bmf_err}")

    if any_error and not continue_on_error:
        comm.Abort(1)


@cli.command("merge_compressed_fields")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument("compressed_files_location", type=click.Path(dir_okay=True, file_okay=False, exists=False))
def merge_compressed_fields(dataset_file: str, compressed_files_location: str):
    """
    Consolidate metadata on the shared {dataset}.zarr store.

    Under the new design, `compress_with_optimal` writes each field directly
    into a shared LocalStore, so the old unzip+copy+rezip merge is unnecessary.
    This command just runs `zarr.consolidate_metadata` so downstream readers
    can open the store quickly without scanning every array.

    Args:
        dataset_file (str): Path to the original dataset file (used to
            derive the name of the merged .zarr store).
        compressed_files_location (str): Directory containing
            {dataset_basename}.zarr.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size > 1:
        if rank == 0:
            click.echo("merge_compressed_fields is not meant to run in parallel.")
        # Collective abort: sys.exit on rank 0 alone would leave ranks 1..N
        # blocking at the next collective.
        comm.Abort(1)

    merged_path = _merged_store_path(compressed_files_location, dataset_file)
    if not Path(merged_path).is_dir():
        click.echo(f"Expected merged store not found: {merged_path}")
        click.echo("Did compress_with_optimal run at least once with the same "
                   "`where_to_write`?")
        comm.Abort(1)

    # Open in a try/finally so the LocalStore handles are released even if
    # consolidate_metadata or the subsequent array listing raises.  Zarr v3's
    # LocalStore holds open file descriptors; at CLI-shape the OS would reap
    # them on process exit, but merging via an imported function (notebook /
    # longer-lived process) would leak them without an explicit close.
    store = zarr.storage.LocalStore(merged_path, read_only=False)
    try:
        zarr.consolidate_metadata(store)
        click.echo(f"[merge] consolidated metadata on {merged_path}")

        # Report what's inside
        g = zarr.open_group(store, mode="r")
        arr_names = list(g.array_keys())
        click.echo(f"[merge] arrays in store ({len(arr_names)}): {', '.join(arr_names) or '<none>'}")
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


@cli.command("open_zarr_and_inspect")
@click.argument("zarr_path", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--head", type=int, default=4, show_default=True,
              help="Per-array head slice size (across each dim) for a tiny preview. "
                   "Set to 0 to skip reading any data.")
def open_zarr_and_inspect(zarr_path: str, head: int):
    """
    Inspect a zarr v3 LocalStore without materialising full arrays.

    Shows: group tree, per-array metadata (shape, dtype, codecs, sharding,
    compression ratio from info_complete), and a tiny head slice.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size > 1:
        if rank == 0:
            click.echo("open_zarr_and_inspect is not meant to run in parallel.")
        # Collective abort: sys.exit on rank 0 alone would leave ranks 1..N
        # blocking at the next collective.
        comm.Abort(1)

    group, store = utils.open_zarr_localstore(zarr_path, read_only=True)
    click.echo(group.tree())
    click.echo("-" * 80)

    for array_name in group.array_keys():
        z = group[array_name]
        click.echo(f"Array: {array_name}")
        click.echo(z.info_complete())
        if head > 0:
            slicer = tuple(slice(0, min(head, s)) for s in z.shape)
            click.echo(f"Head slice {slicer}:")
            click.echo(z[slicer])
        click.echo("-" * 80)


@cli.command("from_nc_to_zarr")
@click.argument("nc_path", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.option("--out", "out_zarr", type=click.Path(dir_okay=True, file_okay=False), default=None,
              help="Output .zarr directory. Defaults to INPUT with .zarr extension.")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True,
              help="If set, remove the output directory before writing.")
@click.option("--consolidated/--no-consolidated", default=True, show_default=True,
              help="Write consolidated metadata so xr.open_zarr can do a fast open.")
@click.option("--preserve-source-chunks/--no-preserve-source-chunks",
              default=True, show_default=True,
              help="If set (default), open the netCDF with chunks={} so each "
                   "dask chunk maps 1:1 to an HDF5 chunk in the source and to "
                   "a single chunk-file in the output zarr.  This is the most "
                   "faithful per-chunk mapping for filesystem dedup.  Pass "
                   "--no-preserve-source-chunks to use xarray's chunks='auto' "
                   "instead - only useful for netCDF-3 sources or contiguous "
                   "netCDF-4 variables, where there is no native chunk "
                   "geometry to preserve.")
@click.option("--mask-and-scale/--no-mask-and-scale",
              default=False, show_default=True,
              help="Whether to apply CF mask_and_scale decoding (scale_factor, "
                   "add_offset, _FillValue) at read time.  Default: OFF for "
                   "this command (xarray's normal default is ON), because "
                   "decoding promotes packed int8/int16 variables to float "
                   "and quadruples their byte count, which confounds both the "
                   "absolute-storage and the dedup-ratio numbers in the VAST "
                   "experiment.  The encoding attrs ride along in var.attrs "
                   "regardless, so a downstream reader that opens the output "
                   "zarr with mask_and_scale=True (xarray's default) still "
                   "gets the decoded floats - no information is lost, the "
                   "values are just stored on disk in their packed form.")
@click.option("--decode-times/--no-decode-times",
              default=False, show_default=True,
              help="Whether to apply CF time decoding (units like 'days since "
                   "1970-01-01', calendar) at read time.  Default: OFF for "
                   "this command (xarray's normal default is ON), for "
                   "symmetry with --mask-and-scale: every on-disk numeric "
                   "form is preserved regardless of what CF says it "
                   "represents.  Effect on dedup is tiny (the time coord is "
                   "usually a single 1-D array of a few KB), but flipping it "
                   "off also sidesteps cftime/datetime64 round-trip variance "
                   "across xarray versions for non-standard calendars.  "
                   "Encoding attrs ride along in var.attrs, so a downstream "
                   "reader passing decode_times=True (xarray default) still "
                   "gets datetime64/cftime objects with no information loss.")
def from_nc_to_zarr(nc_path: str, out_zarr: str | None,
                    overwrite: bool, consolidated: bool,
                    preserve_source_chunks: bool,
                    mask_and_scale: bool,
                    decode_times: bool):
    """
    Convert a NetCDF file (.nc) to a zarr v3 LocalStore (.zarr directory)
    with NO compression, NO filters, and NO sharding.  Intended for
    filesystem-level deduplication experiments (e.g. VAST FS).

    Every data variable AND every coordinate is written with
    `compressors=None, filters=None`; the only codec left in the pipeline
    is the default bytes serializer, which is just an identity
    dtype/endianness step (not a compressor).  Coordinate arrays are
    explicitly included because lat/lon/time are usually identical across
    the files in a series, and a default-compressed coord would mask the
    dedup signal we're trying to measure on the storage side.

    Sharding is intentionally NOT applied (zarr v3's default when no
    `shards` key is passed): each chunk lands in its own file, so VAST
    sees chunk-level granularity.  Sharding would bundle multiple chunks
    per file with chunk offsets that depend on neighboring chunks, which
    would degrade chunk-level dedup into FS-block-level dedup.

    CF mask_and_scale decoding is OFF by default for this command
    (xarray's normal default is ON).  Packed integer dtypes (int8/int16
    with scale_factor/add_offset) stay packed on disk, which avoids both
    the int->float byte-count quadrupling and the float-bit fragility
    where two chunks with identical packed values could produce slightly
    different decoded floats if scale_factor/add_offset attrs drift across
    the file series.  Encoding attrs ride along in var.attrs, so a
    downstream reader passing mask_and_scale=True (xarray default) still
    gets the decoded floats with no information loss.

    CF decode_times is OFF by default for the same family of reasons:
    every on-disk numeric form is preserved regardless of what CF says it
    represents.  Effect on dedup is tiny (time coords are typically a few
    KB), but flipping it off also sidesteps cftime/datetime64 round-trip
    variance across xarray versions for non-standard calendars.

    Caveats
    -------
    - Any compression that was applied INSIDE the netCDF source file is
      undone at read time by the netCDF reader.  We never see the on-disk
      compressed bytes; we see the decoded array.  So "without any
      compression" here means: nothing on the zarr write side, regardless
      of how the netCDF was authored.
    - With --preserve-source-chunks (default), the output zarr's chunk
      structure mirrors the source's HDF5 chunk structure exactly.  For
      netCDF-3 or contiguous netCDF-4 variables this still works (xarray
      picks a single chunk covering the whole variable) but the per-chunk
      dedup story becomes less interesting.
    - --preserve-source-chunks gives chunk-level dedup ONLY when every
      file in the series shares the same HDF5 chunk shape.  Differing
      chunk shapes across the series would need a forced canonical
      rechunk; not implemented here.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size > 1:
        if rank == 0:
            click.echo("from_nc_to_zarr is not meant to run in parallel.")
        # Collective abort: sys.exit on rank 0 alone would leave ranks 1..N
        # blocking at the next collective.
        comm.Abort(1)

    if Path(nc_path).suffix.lower() != ".nc":
        click.echo(
            f"Expected a .nc file, got {nc_path}.  This command only "
            f"handles netCDF input; use from_zarr_to_netcdf for the "
            f"reverse direction."
        )
        comm.Abort(1)

    if out_zarr is None:
        out_zarr = str(Path(nc_path).with_suffix(".zarr"))

    out_path = Path(out_zarr)
    if out_path.exists():
        if overwrite:
            import shutil
            click.echo(f"[nc->zarr] removing existing {out_zarr} (--overwrite).")
            shutil.rmtree(out_zarr)
        else:
            click.echo(
                f"Output already exists: {out_zarr}.  "
                f"Pass --overwrite to replace, or pick a different --out."
            )
            comm.Abort(1)

    click.echo(f"[nc->zarr] reading {nc_path} ...")
    # chunks={} -> dask chunks track HDF5 chunks 1:1 (the default for this
    # command).  chunks="auto" -> dask picks a chunking, used as a fallback
    # for non-chunked sources.  We never use chunks=None because that would
    # eagerly materialise the whole field in RAM, and there's no need: we
    # always want lazy reads paired with the streaming to_zarr write.
    chunks = {} if preserve_source_chunks else "auto"
    # mask_and_scale=False keeps packed int dtypes packed; decode_times=False
    # keeps time coords as raw numerics.  See the docstring and the option
    # help text for why these are the dedup-friendly defaults.
    ds = xr.open_dataset(
        nc_path,
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
    )
    logical_bytes = int(ds.nbytes)
    click.echo(
        f"[nc->zarr] logical size = {humanize.naturalsize(logical_bytes, binary=True)} "
        f"| chunks = {'source-native' if preserve_source_chunks else 'auto'} "
        f"| mask_and_scale = {mask_and_scale} "
        f"| decode_times = {decode_times}"
    )

    # Per-variable encoding override.  Two layers of defense:
    # 1. Clear .encoding on every variable so any netCDF-side encoding keys
    #    (zlib, shuffle, chunksizes, _FillValue, ...) inherited from
    #    xr.open_dataset don't leak into xarray's encoding-translation layer.
    # 2. Pass an explicit `compressors=None, filters=None` per variable to
    #    `to_zarr`, which wins over anything still residual.
    # We iterate over ds.variables (data_vars + coords) so coordinate arrays
    # are included; see the docstring for why.
    encoding = {}
    for name in ds.variables:
        ds[name].encoding = {}
        encoding[name] = {
            "compressors": None,
            "filters": None,
        }

    click.echo(
        f"[nc->zarr] writing {out_zarr} (compressors=None, filters=None, "
        f"{len(encoding)} variable(s)) ..."
    )
    # mode="w-" = create-only; we already short-circuited on the
    # exists-and-not-overwrite path above, so this just guards against a
    # race with another process between the check and the write.
    ds.to_zarr(
        out_zarr,
        mode="w-",
        encoding=encoding,
        zarr_format=3,
        consolidated=consolidated,
    )
    click.echo(f"[nc->zarr] wrote {out_zarr}")


@cli.command("from_zarr_to_netcdf")
@click.argument("zarr_path", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--out", "out_nc", type=click.Path(dir_okay=False), default=None,
              help="Output NetCDF file. Defaults to INPUT with .nc extension.")
@click.option("--max-size", default="50GB", callback=_size_option_callback,
              show_default=True,
              help="Refuse to write if logical output would exceed this size. "
                   "NetCDF4 is not a great container for very large data; "
                   "for >50GB consider keeping the .zarr as-is.")
@click.option("--compression", default="zlib", show_default=True,
              help="NetCDF variable compression (zlib/none).")
@click.option("--complevel", default=4, show_default=True, help="zlib compression level.")
def from_zarr_to_netcdf(zarr_path: str, out_nc: str | None,
                        max_size: int, compression: str, complevel: int):
    """
    Convert a zarr v3 LocalStore (.zarr directory) to a NetCDF4 file.
    Writes are streamed via dask so the full dataset is never held in memory.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size > 1:
        if rank == 0:
            click.echo("from_zarr_to_netcdf is not meant to run in parallel.")
        # Collective abort: sys.exit on rank 0 alone would leave ranks 1..N
        # blocking at the next collective.
        comm.Abort(1)

    if out_nc is None:
        out_nc = str(Path(zarr_path).with_suffix(".nc"))

    # Load via xarray; this preserves dims/coords if consolidated metadata exists.
    # The previous heuristic (Path(zarr_path)/"zarr.json").exists() was wrong:
    # every zarr v3 store has a zarr.json, consolidated or not.  Consolidation
    # in v3 is a `consolidated_metadata` field *inside* that zarr.json.  We try
    # consolidated first (fast path) and fall back to a metadata scan if the
    # store wasn't processed by `merge_compressed_fields`.
    try:
        ds = xr.open_zarr(zarr_path, chunks="auto", consolidated=True)
    except Exception:
        ds = xr.open_zarr(zarr_path, chunks="auto", consolidated=False)

    logical_bytes = int(ds.nbytes)
    click.echo(f"[zarr->nc] logical size = {humanize.naturalsize(logical_bytes, binary=True)}")
    if logical_bytes > max_size:
        click.echo(
            f"Refusing to write: logical size exceeds --max-size "
            f"({humanize.naturalsize(max_size, binary=True)}). "
            f"Raise --max-size to proceed, or keep the data in .zarr."
        )
        comm.Abort(1)

    # Per-variable encoding: preserve dask chunks as NetCDF chunks, add compression.
    encoding = {}
    for name, var in ds.data_vars.items():
        enc = {}
        if isinstance(var.data, dask.array.Array):
            # Use one dask chunk per netcdf chunk.  Caps each chunk at the
            # first block shape to avoid overly large chunks.
            enc["chunksizes"] = tuple(b[0] for b in var.data.chunks)
        if compression == "zlib":
            enc["zlib"] = True
            enc["complevel"] = int(complevel)
        encoding[name] = enc

    click.echo(f"[zarr->nc] writing {out_nc} ...")
    ds.to_netcdf(out_nc, engine="h5netcdf", encoding=encoding)
    click.echo(f"[zarr->nc] wrote {out_nc}")


@cli.command("perform_clustering")
@click.argument("npy_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("l_error", type=str)
def perform_clustering(npy_file: str, l_error: str):
    """
    Calculates Elbow and Silhouette scores for compression Ratio VS chosen L-error over clusters ranging from 3 to 10.
    It can be executed only after evaluate_combos.

    Returns 2 plots for chosen L-error:
      - Elbow method VS Number of clusters
      - Silhouette score VS Number of clusters

    \b
    Args:
        npy_file (str): npy file with L-errors and compression ratios results for each combination of compressor, filter, and serializer
        l_error (str): choose between "L1", "L2", "LInf" to generate the plot
    """
    # Lazy imports: kept out of the module top-level so `evaluate_combos` /
    # `compress_with_optimal` don't pay the matplotlib+sklearn+tqdm import
    # cost on every invocation.  See the comment block near the top of this
    # file for the rationale.
    from tqdm import tqdm
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt

    scored_results = np.load(npy_file, allow_pickle=True)

    scored_results_pd = pd.DataFrame(scored_results)

    numeric_cols = scored_results_pd.select_dtypes(include=[np.number]).columns
    mask = np.isfinite(scored_results_pd[numeric_cols]).all(axis=1)
    scored_results_pd = scored_results_pd[mask].dropna()
    l_options = ["L1", "L2", "LInf"]
    error_index = [i_l+1 for i_l, l in enumerate(l_options) if l_error == l]
    clean_arr_inf = np.hstack((np.asarray(scored_results_pd[[0]]), np.asarray(scored_results_pd[[error_index[0]]])))

    k_values = range(3, 10)
    inertias = []
    silhouette_scores = []

    for k in tqdm(k_values, desc="Looping over k values"):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(clean_arr_inf)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(clean_arr_inf, labels))

    # Plot Elbow Curve
    plt.figure(figsize=(12, 5))
    plt.suptitle(l_error, fontsize=20)
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    # Plot Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, 'go-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.tight_layout()
    plt.show()


@cli.command("analyze_clustering")
@click.argument("npy_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--where-to-write", "where_to_write", required=True,
              type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="Directory containing the `config_space_{var}.csv` written by "
                   "evaluate_combos.  Must be the same directory passed as "
                   "--where-to-write to evaluate_combos for this run.")
@click.option("--var", "var", required=True, type=str,
              help="Variable (field) name to analyse.  Must match the `var` used "
                   "in the evaluate_combos run that produced the .npy and the "
                   "config_space_{var}.csv file (so for a field named 't' it's "
                   "`--var t`, and the tool will read "
                   "`{where_to_write}/config_space_t.csv`).")
def analyze_clustering(npy_file: str, where_to_write: str, var: str):
    """
    Performs clustering on all 3 L-errors, can be executed only after evaluate_combos.
    It can be executed only after evaluate_combos.

    Returns 3 plots for chosen L-error:
      - L1 VS Compression Ratio
      - L2 VS Compression Ratio
      - LInf VS Compression Ratio

    \b
    Args:
        npy_file (str): npy file with L-errors and compression ratios results for each combination of compressor, filter, and serializer
        where_to_write (str): --where-to-write
        var (str): --var
    """
    # Lazy imports: see the comment near the top of this file.
    from sklearn.cluster import KMeans
    import plotly.io as pio
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # evaluate_combos now writes config_space_{var}.csv into {where_to_write}
    # (renamed from the old cwd-relative `config_space.csv`).  Resolve it
    # explicitly from the required flags so analyze_clustering can be run
    # from any working directory.  Both flags are required — no magic
    # fallback to cwd — because guessing would reintroduce the same
    # footgun: the old `pd.read_csv("config_space.csv")` silently picked
    # up whatever happened to be in cwd (possibly a stale file from a
    # different run).
    config_csv_path = Path(where_to_write) / f"config_space_{var}.csv"
    if not config_csv_path.is_file():
        raise click.FileError(
            str(config_csv_path),
            hint=(
                f"Expected `config_space_{var}.csv` in {where_to_write}. "
                f"Run `dc_toolkit evaluate_combos ... --where-to-write {where_to_write}` "
                f"first, and confirm --var matches the field-to-compress used there."
            ),
        )
    config_idxs = pd.read_csv(config_csv_path)
    scored_results = np.load(str(npy_file), allow_pickle=True)

    scored_results_pd = pd.DataFrame(scored_results)

    numeric_cols = scored_results_pd.select_dtypes(include=[np.number]).columns
    mask = np.isfinite(scored_results_pd[numeric_cols]).all(axis=1)
    scored_results_pd = scored_results_pd[mask].dropna()

    clean_arr_l1 = utils.slice_array(scored_results_pd, [0, 1, 5, 6, 7])
    clean_arr_l2 = utils.slice_array(scored_results_pd, [0, 2, 5, 6, 7])
    clean_arr_linf = utils.slice_array(scored_results_pd, [0, 3, 5, 6, 7])

    max_n_rows, max_nclusters = 42976, 6
    adjusted_n_clusters = math.ceil(max_nclusters * len(scored_results_pd) / max_n_rows)

    # Plot Error and Similarity Metrics VS Ratio
    kmeans = KMeans(n_clusters=adjusted_n_clusters, random_state=0, n_init="auto")

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=[
                            "L1 VS Ratio KMeans Clustering", "L2 VS Ratio KMeans Clustering",
                            "LInf VS Ratio KMeans Clustering"
                        ])

    # L1 clustering
    clean_arr_l1_filtered = np.column_stack((clean_arr_l1[:, 0].astype(float), clean_arr_l1[:, 1].astype(float)))

    df_l1 = pd.DataFrame(clean_arr_l1_filtered, columns=["Ratio", "L1"])
    df_l1["compressor"] = clean_arr_l1[:, 2]
    df_l1["filter"] = clean_arr_l1[:, 3]
    df_l1["serializer"] = clean_arr_l1[:, 4]
    df_l1["compressor_idx"] = utils.get_indexes(clean_arr_l1[:, 2], config_idxs['0'])
    df_l1["filter_idx"] = utils.get_indexes(clean_arr_l1[:, 3], config_idxs['1'])
    df_l1["serializer_idx"] = utils.get_indexes(clean_arr_l1[:, 4], config_idxs['2'])

    y_kmeans = kmeans.fit_predict(pd.DataFrame(df_l1, columns=["Ratio", "L1"]))
    color = np.ones(y_kmeans.shape) if len(np.unique(y_kmeans)) == 1 else y_kmeans

    fig_l1 = px.scatter(df_l1, x="Ratio", y="L1", color=color,
                        title="L1 VS Ratio KMeans Clustering",
                        hover_data=["compressor", "filter", "serializer", "compressor_idx", "filter_idx", "serializer_idx"])

    fig.add_trace(
        go.Scatter(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            mode="markers+text",
            marker=dict(color="black", size=12, symbol="x"),
            textposition="top center",
            name="Centroids",
            showlegend=True
        ),
        row=1,
        col=1
    )
    fig.update_xaxes(title_text="Ratio", row=1, col=1)
    fig.update_yaxes(title_text="L1", row=1, col=1)

    for trace in fig_l1.data:
        fig.add_trace(trace, row=1, col=1)

    # L2 clustering
    clean_arr_l2_filtered = np.column_stack((clean_arr_l2[:, 0].astype(float), clean_arr_l2[:, 1].astype(float)))

    df_l2 = pd.DataFrame(clean_arr_l2_filtered, columns=["Ratio", "L2"])
    df_l2["compressor"] = clean_arr_l2[:, 2]
    df_l2["filter"] = clean_arr_l2[:, 3]
    df_l2["serializer"] = clean_arr_l2[:, 4]
    df_l2["compressor_idx"] = utils.get_indexes(clean_arr_l2[:, 2], config_idxs['0'])
    df_l2["filter_idx"] = utils.get_indexes(clean_arr_l2[:, 3], config_idxs['1'])
    df_l2["serializer_idx"] = utils.get_indexes(clean_arr_l2[:, 4], config_idxs['2'])

    y_kmeans = kmeans.fit_predict(pd.DataFrame(df_l2, columns=["Ratio", "L2"]))
    color = np.ones(y_kmeans.shape) if len(np.unique(y_kmeans)) == 1 else y_kmeans

    fig_l2 = px.scatter(df_l2, x="Ratio", y="L2", color=color,
                        title="L2 VS Ratio KMeans Clustering",
                        hover_data=["compressor", "filter", "serializer", "compressor_idx", "filter_idx", "serializer_idx"])

    fig.add_trace(
        go.Scatter(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            mode="markers+text",
            marker=dict(color="black", size=12, symbol="x"),
            textposition="top center",
            name="Centroids",
            showlegend=False
        ),
        row=2,
        col=1
    )
    fig.update_xaxes(title_text="Ratio", row=2, col=1)
    fig.update_yaxes(title_text="L2", row=2, col=1)
    for trace in fig_l2.data:
        fig.add_trace(trace, row=2, col=1)

    # LInf clustering
    clean_arr_linf_filtered = np.column_stack(
        (clean_arr_linf[:, 0].astype(float), clean_arr_linf[:, 1].astype(float)))

    df_linf = pd.DataFrame(clean_arr_linf_filtered, columns=["Ratio", "LInf"])
    df_linf["compressor"] = clean_arr_linf[:, 2]
    df_linf["filter"] = clean_arr_linf[:, 3]
    df_linf["serializer"] = clean_arr_linf[:, 4]
    df_linf["compressor_idx"] = utils.get_indexes(clean_arr_linf[:, 2], config_idxs['0'])
    df_linf["filter_idx"] = utils.get_indexes(clean_arr_linf[:, 3], config_idxs['1'])
    df_linf["serializer_idx"] = utils.get_indexes(clean_arr_linf[:, 4], config_idxs['2'])

    y_kmeans = kmeans.fit_predict(pd.DataFrame(df_linf, columns=["Ratio", "LInf"]))
    color = np.ones(y_kmeans.shape) if len(np.unique(y_kmeans)) == 1 else y_kmeans

    fig_linf = px.scatter(df_linf, x="Ratio", y="LInf", color=color,
                          title="LInf VS Ratio KMeans Clustering",
                          hover_data=["compressor", "filter", "serializer", "compressor_idx", "filter_idx",
                                      "serializer_idx"])

    fig.add_trace(
        go.Scatter(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            mode="markers+text",
            marker=dict(color="black", size=12, symbol="x"),
            textposition="top center",
            name="Centroids",
            showlegend=False
        ),
        row=3,
        col=1
    )
    fig.update_xaxes(title_text="Ratio", row=3, col=1)
    fig.update_yaxes(title_text="LInf", row=3, col=1)
    for trace in fig_linf.data:
        fig.add_trace(trace, row=3, col=1)

    fig.update_layout(
        title="",
        showlegend=False,
        height=900,
        hovermode="closest",
        template="plotly_white"
    )
    pio.renderers.default = "browser"
    fig.show()


@cli.command("plot_compression_errors")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument("where_to_write", type=click.Path(dir_okay=True, file_okay=False, exists=False))
@click.argument("field_to_compress")
@click.argument("comp_idx", type=int)
@click.argument("filt_idx", type=int)
@click.argument("ser_idx", type=int)
@click.option("--compressor-class", default="all", help="Same as in evaluate_combos.")
@click.option("--filter-class", default="all", help="Same as in evaluate_combos.")
@click.option("--serializer-class", default="all", help="Same as in evaluate_combos.")
@click.option("--with-lossy/--without-lossy", default=True, show_default=True, help="Same as in evaluate_combos.")
@click.option("--with-numcodecs-wasm/--without-numcodecs-wasm", default=True, show_default=True, help="Same as in evaluate_combos.")
@click.option("--with-ebcc/--without-ebcc", default=True, show_default=True, help="Same as in evaluate_combos.")
def plot_compression_errors(dataset_file: str, where_to_write: str, field_to_compress: str,
                            comp_idx: int, filt_idx: int, ser_idx: int, 
                            compressor_class: str = "all", filter_class: str = "all", serializer_class: str = "all",
                            with_lossy: bool = True, with_numcodecs_wasm: bool = True, with_ebcc: bool = True):
    """
    Plot the absolute errors arising from compression+decompression of a field
    with the desired combination of compressor, filter, and serializer.
    Additionally, plot the difference between a normal compression+decompression
    process and one using a version of the data shifted by 180 degrees
    longitudinally. This should show if the selected combination takes the
    periodicity of the data into account.

    Make sure to provide the field to compress in (lat, lon) format. Additional
    dimensions are not supported or removed if they have a single level.

    Make sure to provide the same --[compressor/filter/serializer]-class and the
    same --with/without-[lossy/numcodecs-wasm/ebcc] flags as in evaluate_combos,
    such that the same lists of instantiated objects are generated.
    
    Note on passing -1 as index:
    dc_toolkit plot_compression_errors ... --compressor-class X ... --- -1 -1 -1

    \b
    Args:
        dataset_file (str): Path to the input dataset file.
        where_to_write (str): Directory where the file containing the plots will be writtsaved.
        field_to_compress (str): Name of the field to compress/analyze.
        comp_idx (int): Index of the compressor to use.
        filt_idx (int): Index of the filter to use.
        ser_idx (int): Index of the serializer to use.
        compressor_class: --compressor-class
        filter_class: --filter-class
        serializer_class: --serializer-class
        with_lossy: --with-lossy/--without-lossy
        with_numcodecs_wasm: --with-numcodecs-wasm/--without-numcodecs-wasm
        with_ebcc: --with-ebcc/--without-ebcc
    """
    # Lazy import: see the comment near the top of this file.
    import matplotlib.pyplot as plt

    #############
    # GET COMBO #
    #############

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size > 1:
        if rank == 0:
            click.echo("This command is not meant to be run in parallel. Please run it with a single process.")
        # Collective abort: sys.exit on rank 0 alone would leave ranks 1..N
        # blocking at the next collective.
        comm.Abort(1)

    os.makedirs(where_to_write, exist_ok=True)

    ds = utils.open_dataset(dataset_file, field_to_compress)
    da = ds[field_to_compress].squeeze()

    da_memsize = da.nbytes
    click.echo(f"Squeezed (lat, lon) field_to_compress.nbytes = {humanize.naturalsize(da_memsize, binary=True)}")

    if not utils.is_lat_lon(da):
        click.echo(f"Field {field_to_compress} should be in lat-lon form, i.e. dimensions (lat, lon)! It currently has dimensions: {da.dims}.")
        comm.Abort(1)

    mem_threshold = 2.5  # GiB
    if da_memsize / (1024 ** 3) > mem_threshold:
        click.echo(f"Field {field_to_compress} is too large ({humanize.naturalsize(da_memsize, binary=True)}). "
                   f"To avoid high memory usage we only support fields up to {mem_threshold} GiB.")
        comm.Abort(1)

    compressors = utils.compressor_space(da, with_lossy, with_numcodecs_wasm, with_ebcc, compressor_class)
    filters = utils.filter_space(da, with_lossy, with_numcodecs_wasm, with_ebcc, filter_class)
    serializers = utils.serializer_space(da, with_lossy, with_numcodecs_wasm, with_ebcc, serializer_class)

    if -1 <= comp_idx < len(compressors):
        pass
    else:
        click.echo(f"Invalid comp_idx: {comp_idx}")
        comm.Abort(1)
    if -1 <= filt_idx < len(filters):
        pass
    else:
        click.echo(f"Invalid filt_idx: {filt_idx}")
        comm.Abort(1)
    if -1 <= ser_idx < len(serializers):
        pass
    else:
        click.echo(f"Invalid ser_idx: {ser_idx}")
        comm.Abort(1)

    selected_compressor = compressors[comp_idx][1] if comp_idx != -1 else None
    selected_filter = filters[filt_idx][1] if filt_idx != -1 else None
    selected_serializer = serializers[ser_idx][1] if ser_idx != -1 else None

    chunks_size = 'auto'

    if isinstance(selected_serializer, AnyNumcodecsArrayBytesCodec) and isinstance(selected_serializer.codec, EBCCZarrFilter):
        da = da.astype("float32")
        chunks_height = int(selected_serializer.codec.arglist[0])
        chunks_width = int(selected_serializer.codec.arglist[1])
        chunks_size = (chunks_height, chunks_width)

    filters_ = [selected_filter,]
    compressors_ = [selected_compressor,]
    serializer_ = selected_serializer

    if isinstance(serializer_, AnyNumcodecsArrayBytesCodec) or selected_filter is None:
        filters_ = None
    if selected_compressor is None:
        compressors_ = None
    if selected_serializer is None:
        serializer_ = "auto"

    ################
    # PROCESS DATA #
    ################

    click.echo(f"Shape of {field_to_compress}: {da.shape}")

    units = da.attrs["units"]

    lon_dim = da.dims[1]
    half_idx = da.sizes[lon_dim] // 2

    shifted_da = da.roll({lon_dim: -half_idx}, roll_coords=False)
    shifted_da_backshifted = shifted_da.roll({lon_dim: half_idx}, roll_coords=False)

    # Flatten data for ZFPY serializer
    if isinstance(selected_serializer, numcodecs.zarr3.ZFPY):
        # Save the original dims BEFORE mutating `da`.  Using `da.dims` on the
        # second stack call after the first one runs would read the stacked
        # shape (`("flat_dim",)`), so xarray would try to stack `shifted_da`
        # on a dimension it doesn't have and raise ValueError.
        orig_dims = da.dims
        da = da.stack(flat_dim=orig_dims)
        shifted_da = shifted_da.stack(flat_dim=orig_dims)

    ############
    # COMPRESS #
    ############

    # Normal compression
    store = utils.open_zarr_memstore()
    da_compressed = zarr.create_array(
        store=store,
        name=field_to_compress,
        data=da,
        chunks=chunks_size,
        filters=filters_,
        compressors=compressors_,
        serializer=serializer_
    )

    # Shifted compression
    shifted_store = utils.open_zarr_memstore()
    shifted_da_compressed = zarr.create_array(
        store=shifted_store,
        name=field_to_compress,
        data=shifted_da,
        chunks=chunks_size,
        filters=filters_,
        compressors=compressors_,
        serializer=serializer_
    )

    ##############
    # DECOMPRESS #
    ##############

    # Normal
    da_decompressed = xr.DataArray(da_compressed[:], dims=da.dims, coords=da.coords)

    # Shifted
    shifted_da_decompressed = xr.DataArray(shifted_da_compressed[:], dims=da.dims, coords=da.coords)

    # Reshape the data to its original dimensions for ZFPY serializer
    if isinstance(selected_serializer, numcodecs.zarr3.ZFPY):
        da = da.unstack("flat_dim")
        shifted_da = shifted_da.unstack("flat_dim")
        da_decompressed = da_decompressed.unstack("flat_dim")
        shifted_da_decompressed = shifted_da_decompressed.unstack("flat_dim")

    shifted_da_decompressed_backshifted = shifted_da_decompressed.roll({lon_dim: half_idx}, roll_coords=False)

    store.close()
    shifted_store.close()

    #########
    # PLOTS #
    #########

    fig, axes = plt.subplots(3, 3, layout='constrained', figsize=(16, 9))
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axes.flatten()

    fig.suptitle(f"Compression errors for variable {field_to_compress} ({units})", fontsize=16)

    ax1.set_title("Original", fontsize=10)
    tmp = ax1.imshow(da, interpolation='none')
    fig.colorbar(tmp, ax=ax1, shrink=0.7)

    ax2.set_title("Original compressed&decompressed", fontsize=10)
    tmp = ax2.imshow(da_decompressed, interpolation='none')
    fig.colorbar(tmp, ax=ax2, shrink=0.7)

    ax3.set_title("Absolute error [original - original c&d]", fontsize=10)
    absolute_error = np.abs(da - da_decompressed)
    tmp = ax3.imshow(absolute_error, interpolation='none', cmap='binary')
    fig.colorbar(tmp, ax=ax3, shrink=0.7)

    ax4.set_title("Shifted (by +180 deg)", fontsize=10)
    tmp = ax4.imshow(shifted_da, interpolation='none')
    fig.colorbar(tmp, ax=ax4, shrink=0.7)

    ax5.set_title("Shifted compressed&decompressed", fontsize=10)
    tmp = ax5.imshow(shifted_da_decompressed, interpolation='none')
    fig.colorbar(tmp, ax=ax5, shrink=0.7)

    ax6.set_title("Absolute error [shifted - shifted c&d]", fontsize=10)
    absolute_error = np.abs(shifted_da - shifted_da_decompressed)
    tmp = ax6.imshow(absolute_error, interpolation='none', cmap='binary')
    fig.colorbar(tmp, ax=ax6, shrink=0.7)

    ax7.set_title("Absolute error [original - (shifted-180)]", fontsize=10)
    absolute_error = np.abs(da - shifted_da_backshifted) / (np.abs(da) + 1e-20)
    tmp = ax7.imshow(absolute_error, interpolation='none', cmap='binary')
    fig.colorbar(tmp, ax=ax7, shrink=0.7)

    ax8.set_title("Absolute error [original c&d - (shifted c&d-180)]", fontsize=10)
    absolute_error = np.abs(da_decompressed - shifted_da_decompressed_backshifted)
    tmp = ax8.imshow(absolute_error, interpolation='none', cmap='binary')
    fig.colorbar(tmp, ax=ax8, shrink=0.7)

    ax9.set_title("Absolute error [original - (shifted c&d-180)]", fontsize=10)
    absolute_error = np.abs(da - shifted_da_decompressed_backshifted)
    tmp = ax9.imshow(absolute_error, interpolation='none', cmap='binary')
    fig.colorbar(tmp, ax=ax9, shrink=0.7)

    fig.savefig(os.path.join(where_to_write, f'{field_to_compress}_compression_errors.pdf'), bbox_inches='tight')
    plt.close(fig)


@cli.command("run_web_ui_vcluster")
@click.option("--user_account", type=str, default="", help="vCluster user account name")
@click.option("--uenv_image", type=str, default="", help="vCluster uenv image name")
@click.option("--uploaded_file", type=str, default="", help="Uploaded file from vcluster")
@click.option("--time", type=str, default="", help="Allocated time")
@click.option("--nodes", type=str, default="", help="Number of nodes")
@click.option("--ntasks-per-node", type=str, default="", help="Number of tasks per node")
def run_web_ui_vcluster(user_account: str = None, uenv_image: str = "", uploaded_file: str = "", time: str = "", nodes: str = "", ntasks_per_node: str = ""):
    """
    Web UI for data clustering, analysis, and compression to be launched from vcluster.

    \b
    Args:
        user_account (str): vcluster user id
        uploaded_file (str): path to file to use for analysis
        time (str): UI time limit
        nodes (str): number of nodes
        ntasks_per_node (str): number of tasks per node
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cmd_web_ui_vcluster = [
        "streamlit", "run", str(current_dir) + "/compression_analysis_ui_vcluster.py", "--", "--user_account", user_account, "--uenv_image", uenv_image,
        "--uploaded_file", uploaded_file, "--time", time, "--nodes", nodes, "--ntasks-per-node", ntasks_per_node
    ]
    subprocess.run(cmd_web_ui_vcluster)


@cli.command("run_web_ui")
def run_web_ui():
    """
    Web UI for data clustering, analysis, and compression to be launched from local terminal.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cmd_web_ui = [
        "streamlit", "run", str(current_dir) + "/compression_analysis_ui_web.py"
    ]
    subprocess.run(cmd_web_ui)


@cli.command("run_local_ui")
def run_local_ui():
    """
    Local UI for data clustering, analysis, and compression to be launched from local terminal.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cmd_local_ui = [
        "python", str(current_dir) + "/compression_analysis_ui_local.py"
    ]
    subprocess.run(cmd_local_ui)


@cli.command("help")
@click.pass_context
def help(ctx):
    for command in cli.commands.values():
        if command.name == "help":
            continue
        click.echo("-" * 80)
        click.echo()
        with click.Context(command, parent=ctx.parent, info_name=command.name) as ctx:
            click.echo(command.get_help(ctx=ctx))
        click.echo()


if __name__ == "__main__":
    cli()
