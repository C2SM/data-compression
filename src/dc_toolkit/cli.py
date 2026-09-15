# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
dc_toolkit command-line interface.

Pipeline
  evaluate_combos               sweep compressor x filter x serializer on a sample
                                -> manifest_{var}.json with the best combo
  compress_with_optimal         persist ONE field with a chosen combo
  compress_fields_from_results  persist EVERY field from the sweep manifests
  merge_compressed_fields       consolidate metadata of the shared .zarr store

Utilities
  open_zarr_and_inspect, from_nc_to_zarr, from_zarr_to_netcdf,
  perform_clustering, analyze_clustering, plot_compression_errors,
  run_web_ui, run_local_ui, run_web_ui_vcluster, help

Sections
  1. Helpers & shared CLI options
  2. Threads & memory guards
  3. Sample signature (sweep <-> persist reproducibility)
  4. Gates & thresholds
  5. Codec space + persistence shared by the compress commands
  6. evaluate_combos
  7. compress_with_optimal / compress_fields_from_results
  8. Store utilities & format conversion
  9. Analysis & plotting
 10. UIs & help
"""
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import dask
import dask.array
import humanize
import numpy as np
import pandas as pd
import psutil
import xarray as xr
import zarr
from mpi4py import MPI

from dc_toolkit import utils

# matplotlib / sklearn / plotly / tqdm are imported inside the analysis
# commands so the sweep and compress commands do not pay for them.

warnings.filterwarnings("ignore", message="Numcodecs codecs are not in the Zarr version 3 specification.*",
                        category=UserWarning)
warnings.filterwarnings("ignore", message="Engine 'cfgrib' loading failed", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in square")
warnings.filterwarnings("ignore", message=r".*leaked semaphore objects.*", category=UserWarning,
                        module=r"multiprocessing\.resource_tracker")


@click.group()
def cli():
    pass


# =============================================================================
# 1. HELPERS & SHARED CLI OPTIONS
# =============================================================================

def _hsize(nbytes) -> str:
    return humanize.naturalsize(nbytes, binary=True)


def _size_option_callback(ctx, param, value):
    if value is None:
        return None
    try:
        return utils.parse_size(value)
    except Exception as e:
        raise click.BadParameter(f"Invalid size '{value}': {e}")


_abort = utils.abort


def _require_single_process(command: str) -> None:
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        if comm.Get_rank() == 0:
            click.echo(f"{command} is not meant to run in parallel.  Launch it with a single process.")
        comm.Abort(1)


def _env_versions() -> dict:
    return {"zarr": getattr(zarr, "__version__", None),
            "numpy": getattr(np, "__version__", None),
            "dask": getattr(dask, "__version__", None)}


def _version_banner(command: str) -> str:
    env = _env_versions()
    return f"[env] {command} | zarr={env['zarr']} | numpy={env['numpy']} | dask={env['dask']}"


def _warn_env_drift(sweep_env: dict, source: str) -> None:
    """Warn when zarr/numpy/dask differ from the sweep (decode paths can shift
    bytes and trip the sample-signature check)."""
    now = _env_versions()
    deltas = [(p, sweep_env.get(p), now.get(p)) for p in ("zarr", "numpy", "dask")
              if sweep_env.get(p) is not None and sweep_env.get(p) != now.get(p)]
    if deltas:
        click.echo(f"[manifest] WARNING: library versions differ from the sweep that wrote {source}:")
        for pkg, then, cur in deltas:
            click.echo(f"  {pkg}: sweep={then}  now={cur}")
        click.echo("  A sample-hash mismatch below is then likely an environment change: rerun "
                   "evaluate_combos here, or switch back to the sweep's environment.")


def _merged_store_path(where_to_write: str, dataset_file: str) -> str:
    """One {dataset_stem}.zarr store per dataset; fields are arrays inside it."""
    return str(Path(where_to_write) / f"{Path(dataset_file).stem}.zarr")


def _close_store(store) -> None:
    close = getattr(store, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _read_json(path, label: str):
    """Parse a JSON file; None (with a warning) if missing or unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        click.echo(f"[{label}] WARNING: could not parse {path.name}: {e}")
        return None


def _write_json(path, payload: dict, label: str) -> None:
    try:
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        click.echo(f"[{label}] wrote manifest -> {path}")
    except Exception as e:
        click.echo(f"[{label}] WARNING: could not write {path}: {e}")


def _resolve_combo(spaces, comp_idx: int, filt_idx: int, ser_idx: int, context: str):
    """Index into (compressors, filters, serializers); -1 means 'none'.
    Raises click.ClickException on bad indices or a pairing combo_is_valid rejects."""
    picked = []
    for name, idx, space in zip(("comp_idx", "filt_idx", "ser_idx"), (comp_idx, filt_idx, ser_idx), spaces):
        if not (-1 <= idx < len(space)):
            raise click.ClickException(f"Invalid {name}: {idx} (must be in [-1, {len(space) - 1}]) for {context}")
        picked.append(space[idx][1] if idx != -1 else None)
    compressor, filt, serializer = picked
    if not utils.combo_is_valid(filt, serializer):
        raise click.ClickException(
            f"Invalid (filter, serializer) pairing for {context}: filter={filt} serializer={serializer}. "
            f"Rejected by combo_is_valid (e.g. FixedScaleOffset->ZFPY); choose a different combination.")
    return compressor, filt, serializer


def _add_options(options):
    def decorator(f):
        for opt in reversed(options):
            f = opt(f)
        return f
    return decorator


_CODEC_SPACE_OPTIONS = [
    click.option("--compressor-class", default="all", show_default=True,
                 help="Compressor class name (e.g. blosc, zstd), 'all', or 'none'."),
    click.option("--filter-class", default="all", show_default=True,
                 help="Filter class name (e.g. bitround, quantize), 'all', or 'none'."),
    click.option("--serializer-class", default="all", show_default=True,
                 help="Serializer class name (pcodec, zfpy), 'all', or 'none'."),
    click.option("--with-lossy/--without-lossy", default=True, show_default=True,
                 help="Include lossy filters and serializers in the codec space."),
]
_EVAL_LIMIT_OPTION = click.option(
    "--eval-data-size-limit", default="5GB", callback=_size_option_callback, show_default=True,
    help="Sample budget (e.g. 5GB, 512MiB) used to build the codec space.  Pass the same value "
         "to evaluate_combos and to the compress commands so codec indices resolve identically.")
_CHUNK_OPTIONS = [
    click.option("--inner-chunk-mib", type=int, default=16, show_default=True,
                 help="Target zarr chunk size in MiB.  Use the same value in the sweep and when persisting."),
    click.option("--max-inner-chunk-mib", type=int, default=256, show_default=True,
                 help="Warn when --no-spatial-split produces a chunk above this size (MiB)."),
    click.option("--spatial-split/--no-spatial-split", default=True, show_default=True,
                 help="Split spatial dims (horizontal first, vertical last) when one timestep "
                      "exceeds --inner-chunk-mib.  --no-spatial-split keeps one full timestep per "
                      "chunk and skips sharding."),
]
_CODEC_THREAD_OPTIONS = [
    click.option("--codec-threads", type=int, default=1, show_default=True,
                 help="Codec-internal threads per call (Blosc set live; OpenMP/MKL/OpenBLAS need "
                      "shell exports).  threads x codec-threads must not exceed the cores."),
    click.option("--oversubscription-check/--no-oversubscription-check", default=True, show_default=True,
                 help="Abort at startup unless OMP/BLOSC/MKL thread env vars are pinned to 1."),
]
_MEMORY_OPTION = click.option(
    "--memory-threshold", type=click.FloatRange(0.05, 0.95), default=0.80, show_default=True,
    help="Max fraction of available RAM a single tracked allocation may use before aborting.")
_PERSIST_OPTIONS = _CHUNK_OPTIONS + [
    click.option("--shard-mib", type=int, default=512, show_default=True,
                 help="Target shard size in MiB (an integer number of inner chunks).  Sharding is "
                      "skipped when one chunk already reaches this size."),
    click.option("--threads", type=int, default=None,
                 help="Dask workers for the write (default: visible cores).  Peak memory ~ threads x shard_mib."),
] + _CODEC_THREAD_OPTIONS + [_MEMORY_OPTION]
_VERIFY_OPTIONS = [
    click.option("--verify/--no-verify", default=True, show_default=True,
                 help="Re-read the store after writing and recompute the error norms "
                      "(roughly doubles wall time; skip for trusted re-runs)."),
    click.option("--verify-gate/--no-verify-gate", default=True, show_default=True,
                 help="With --verify, fail when production norms exceed the sweep thresholds "
                      "in manifest_{field}.json.  --no-verify-gate only warns."),
]


# =============================================================================
# 2. THREADS & MEMORY GUARDS
# =============================================================================

def _apply_codec_threads(codec_threads: int, rank: int = 0) -> None:
    """Blosc honours set_nthreads live; OpenMP/MKL read env vars at load time."""
    if codec_threads is None or int(codec_threads) <= 1:
        return
    n = int(codec_threads)
    mismatched = [(v, os.environ.get(v)) for v in utils.THREAD_ENV_VARS if os.environ.get(v) != str(n)]
    if mismatched and rank == 0:
        click.echo(f"[codec-threads] requested {n}; export these in the shell BEFORE running for full effect:")
        for v, cur in mismatched:
            click.echo(f"  {v}={'<unset>' if cur is None else cur} -> export {v}={n}")
    try:
        import numcodecs.blosc as _blosc
        _blosc.set_nthreads(n)
    except Exception as e:
        if rank == 0:
            click.echo(f"[codec-threads] WARNING: blosc.set_nthreads failed: {e}")


def _check_thread_product(threads: int, codec_threads: int, rank: int = 0) -> None:
    cores = utils.detect_cores_available()
    product = int(threads) * max(1, int(codec_threads or 1))
    if product > cores:
        if rank == 0:
            click.echo(f"[oversubscription] --threads * --codec-threads = {product} exceeds "
                       f"physical cores ({cores}).  Reduce one of the flags.")
        _abort(1)


def _configure_threads(threads: int, codec_threads: int, oversubscription_check: bool, rank: int = 0) -> None:
    _apply_codec_threads(codec_threads, rank=rank)
    _check_thread_product(threads, codec_threads, rank=rank)
    if int(codec_threads or 1) <= 1:
        utils.check_thread_oversubscription(abort_if_unsafe=oversubscription_check, rank=rank)


# Per-thread working set in units of the sample: decoded buffer (1x) + encoded
# MemoryStore (up to 1x) + codec scratch.  1.5x is an empirical upper bound.
PER_THREAD_WORKING_FACTOR = 1.5


def _per_rank_steady_estimate_bytes(sample_bytes: int, threads_per_rank: int, inner_chunk_mib: int) -> int:
    """sample + threads x 1.5 x sample + threads x 2 x chunk (float64 promotion)."""
    threads = max(1, int(threads_per_rank))
    return int(sample_bytes + threads * sample_bytes * PER_THREAD_WORKING_FACTOR
               + threads * 2 * max(1, int(inner_chunk_mib)) * 2**20)


def _max_sample_bytes_for_threads(budget_bytes: int, threads_per_rank: int, inner_chunk_mib: int) -> int:
    """Inverse of _per_rank_steady_estimate_bytes; 0 if nothing fits."""
    threads = max(1, int(threads_per_rank))
    available = budget_bytes - threads * 2 * max(1, int(inner_chunk_mib)) * 2**20
    return 0 if available <= 0 else int(available / (1.0 + threads * PER_THREAD_WORKING_FACTOR))


def _detect_node_memory_budget() -> tuple[int, str]:
    """(bytes, source): cgroup v2 limit, else cgroup v1, else host RAM.  The
    cgroup is what actually OOM-kills a SLURM task; psutil cannot see it."""
    try:
        with open("/sys/fs/cgroup/memory.max") as fh:
            val = fh.read().strip()
        if val and val != "max":
            return int(val), "cgroup v2 memory.max"
    except (OSError, ValueError):
        pass
    try:
        host_total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        host_total = 0
    for path in ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory.limit_in_bytes"):
        try:
            with open(path) as fh:
                val = int(fh.read().strip())
        except (OSError, ValueError):
            continue
        if host_total and val < host_total * 2:  # else: the ~2^63 "unlimited" sentinel
            return val, f"cgroup v1 ({path})"
    if host_total:
        return host_total, "host total RAM (sysconf)"
    return psutil.virtual_memory().total, "psutil host total"


def _check_node_memory_headroom(per_rank_steady_bytes: int, ranks_on_node: int, rank: int,
                                label: str, threshold: float = 0.80) -> None:
    """Abort if the sweep's steady-state footprint exceeds the node budget.  A
    cgroup limit is per task; host RAM is shared by all ranks on the node."""
    if rank != 0:
        return
    available, source = _detect_node_memory_budget()
    if source.startswith("cgroup"):
        required, scope = per_rank_steady_bytes, "per-rank (cgroup is per-task under SLURM)"
    else:
        required, scope = max(1, ranks_on_node) * per_rank_steady_bytes, f"per-node ({ranks_on_node} rank(s) x per-rank)"
    if required > threshold * available:
        click.echo(
            f"[memcheck] REFUSING to start sweep: memory requirement {_hsize(required)} ({scope}, "
            f"{_hsize(per_rank_steady_bytes)} steady-state per rank) exceeds {int(threshold*100)}% of the "
            f"detected budget {_hsize(available)} ({source}).\n"
            f"  Context: {label}\n"
            f"  Fixes: lower --ntasks-per-node, lower --eval-data-size-limit, request more RAM "
            f"(#SBATCH --mem=0), or raise --memory-threshold (max 0.95).")
        _abort(1)


_MEMCHECK_WARNED_HIGH = False


def _reset_memcheck_state() -> None:
    global _MEMCHECK_WARNED_HIGH
    _MEMCHECK_WARNED_HIGH = False


def _check_memory_headroom(required_bytes: int, label: str, threshold: float = 0.80) -> None:
    """Abort if `required_bytes` exceeds `threshold` of currently available
    host RAM (psutil; does not see cgroup limits)."""
    global _MEMCHECK_WARNED_HIGH
    if threshold > 0.80 and not _MEMCHECK_WARNED_HIGH:
        click.echo(f"[memcheck] WARNING: threshold {threshold:.2f} exceeds the recommended 0.80; "
                   f"the 1.5-2x rechunk transient may no longer fit.")
        _MEMCHECK_WARNED_HIGH = True
    try:
        avail = psutil.virtual_memory().available
    except Exception as e:
        click.echo(f"[memcheck] WARNING: could not query available memory ({e}); skipping guard for {label}.")
        return
    if required_bytes > threshold * avail:
        click.echo(f"[memcheck] REFUSING to proceed: {label} needs {_hsize(required_bytes)}, which exceeds "
                   f"{int(threshold*100)}% of currently-available RAM ({_hsize(avail)}).\n"
                   f"  Reduce the relevant flag (--eval-data-size-limit, --threads, --shard-mib), "
                   f"raise --memory-threshold (max 0.95), or run on a larger node.")
        _abort(1)


# =============================================================================
# 3. SAMPLE SIGNATURE
# =============================================================================
# Codec indices from the sweep only resolve to the same codec objects if the
# compress commands rebuild the codec space from an IDENTICAL sample.  The
# sweep records the sample's sha256 (plus the budget/policy that built it) in
# sample_signature_{var}.json and the compress commands check against it.

def _signature_path(where_to_write: str, var: str) -> Path:
    return Path(where_to_write) / f"sample_signature_{var}.json"


def _sample_signature(dataset_file: str, var: str, eval_data_size_limit: int, sample_np: np.ndarray) -> dict:
    if sample_np.dtype == object:
        raise ValueError("object-dtype arrays cannot be hashed reproducibly")
    h = hashlib.sha256()
    mv = memoryview(np.ascontiguousarray(sample_np)).cast("B")
    step = 64 * 1024 * 1024
    for i in range(0, len(mv), step):
        h.update(mv[i:i + step])
    return {"dataset_stem": Path(dataset_file).stem, "var": var,
            "eval_data_size_limit": int(eval_data_size_limit),
            "shape": list(sample_np.shape), "dtype": str(sample_np.dtype),
            "nbytes": int(sample_np.nbytes), "sha256": h.hexdigest()}


def _rebuild_sweep_sample(da, var: str, dataset_file: str, where_to_write: str, eval_data_size_limit: int):
    """Rebuild the sweep's sample for codec-space construction and compare its
    signature.  Returns (sample_da, status) with status one of
    "match", "mismatch", "no-signature", "unchecked"."""
    sig_path = _signature_path(where_to_write, var)
    expected = _read_json(sig_path, "sample-hash")
    limit, policy, vfloor = int(eval_data_size_limit), "cascade", None
    if expected is not None:  # the sweep may have auto-shrunk its budget
        limit = int(expected.get("effective_sample_limit", limit))
        policy = expected.get("sampling_policy") or "cascade"
        vfloor = expected.get("vertical_floor")
    sample = utils.build_representative_sample(da, limit, policy=policy, vertical_floor=vfloor).compute()

    if expected is None:
        if not sig_path.is_file():
            click.echo(f"[sample-hash] no {sig_path.name} found - proceeding on trust "
                       f"(run evaluate_combos first with the same --where-to-write for the full safety net).")
        return sample, "no-signature"
    try:
        observed = _sample_signature(dataset_file, var, int(eval_data_size_limit),
                                     np.ascontiguousarray(sample.values))
        fields = ["dataset_stem", "var", "shape", "dtype", "nbytes", "sha256"]
        if "effective_sample_limit" not in expected:  # old-style signature: the CLI limit built the sample
            fields.append("eval_data_size_limit")
        mismatches = [f for f in fields if expected.get(f) != observed.get(f)]
    except Exception as e:
        click.echo(f"[sample-hash] WARNING: could not verify {sig_path.name}: {e}.  Proceeding without check.")
        return sample, "unchecked"
    if mismatches:
        click.echo(f"[sample-hash] MISMATCH vs {sig_path.name}: differing fields = {mismatches}")
        for f in mismatches:
            click.echo(f"  {f}: expected={expected.get(f)} observed={observed.get(f)}")
        click.echo("  Causes: a different --eval-data-size-limit than at sweep time, or the field/dataset "
                   "changed since the sweep.")
        return sample, "mismatch"
    click.echo(f"[sample-hash] OK, matches {sig_path.name} (sha256={observed['sha256'][:16]}…).")
    return sample, "match"


# =============================================================================
# 4. GATES & THRESHOLDS
# =============================================================================
# All thresholds are RELATIVE errors.  --l1-threshold is the anchor; the other
# gates default to multiples of it.  A disabled gate resolves to +inf.

_L2_MULT_DEFAULT = 2.0     # RMS may run ~2x the mean-abs budget
_LINF_MULT_DEFAULT = 10.0  # single-cell trip-wire
_BIAS_MULT_DEFAULT = 0.5   # at most half the budget may be one-directional
_Q99_MULT_DEFAULT = 2.0    # extreme-tail allowance
_GATE_KEYS = ("l1", "l2", "linf", "bias", "q99")


def _derive_thresholds(l1, l2, linf, bias, q99, l2_gate, linf_gate, bias_gate, extremes_sensitive) -> dict:
    def pick(value, mult, enabled):
        if not enabled:
            return math.inf
        return float(value) if value is not None else mult * l1
    return {"l1": float(l1),
            "l2": pick(l2, _L2_MULT_DEFAULT, l2_gate),
            "linf": pick(linf, _LINF_MULT_DEFAULT, linf_gate),
            "bias": pick(bias, _BIAS_MULT_DEFAULT, bias_gate),
            "q99": pick(q99, _Q99_MULT_DEFAULT, extremes_sensitive)}


def _evaluate_gates(*, l1_rel, l2_rel, linf_rel, bias_rel, q99_rel, grad_rel,
                    decoded_min, decoded_max, n_corrupt, thr, grad_threshold, grad_gate,
                    phys_min, phys_max):
    """(keep, {gate: passed}).  A None metric or +inf limit passes.  Shared by
    the sweep and the production verify gate."""
    def le(val, lim):
        if val is None or lim is None or not math.isfinite(lim):
            return True
        return float(val) <= float(lim)

    reasons = {"pass_l1": le(l1_rel, thr.get("l1")), "pass_l2": le(l2_rel, thr.get("l2")),
               "pass_linf": le(linf_rel, thr.get("linf")), "pass_bias": le(bias_rel, thr.get("bias")),
               "pass_q99": le(q99_rel, thr.get("q99")), "pass_finite": int(n_corrupt or 0) == 0}
    bounds = True
    if phys_min is not None and decoded_min is not None and math.isfinite(decoded_min):
        bounds = bounds and decoded_min >= phys_min
    if phys_max is not None and decoded_max is not None and math.isfinite(decoded_max):
        bounds = bounds and decoded_max <= phys_max
    reasons["pass_bounds"] = bool(bounds)
    reasons["pass_grad"] = le(grad_rel, grad_threshold) if grad_gate else True
    return all(reasons.values()), reasons


def _evaluate_cr_drift(production_ratio, predicted_ratio, tol):
    """(ok, drift, direction) with direction in ok/under/over/skip.  Bounds
    wasted storage: an error gate cannot see a ratio far below the sweep's."""
    if (predicted_ratio is None or production_ratio is None or not math.isfinite(predicted_ratio)
            or predicted_ratio <= 0 or not math.isfinite(production_ratio)):
        return True, None, "skip"
    drift = (production_ratio - predicted_ratio) / predicted_ratio
    if abs(drift) <= tol:
        return True, drift, "ok"
    return False, drift, ("under" if drift < 0 else "over")


def _q99_cut(sample_np: np.ndarray):
    """99th percentile of |finite values|: the extreme-tail cut for the q99 gate."""
    finite = sample_np[np.isfinite(sample_np)]
    return float(np.quantile(np.abs(finite), 0.99)) if finite.size else None


def _fmt3(x) -> str:
    return f"{x:.3e}" if isinstance(x, float) else "n/a"


def _verify_against_manifest(var: str, errors: dict, manifest, overrides: dict | None = None):
    """Production verify gate: compare `errors` with the thresholds recorded by
    the sweep (CLI overrides win).  Returns (status, detail) where status is
    "pass", "fail" or "no-thresholds"."""
    man_thr = (manifest or {}).get("effective_thresholds", {}) or {}
    thr = {k: (float(man_thr[k]) if man_thr.get(k) is not None else math.inf) for k in _GATE_KEYS}
    for k, v in (overrides or {}).items():
        if v is not None:
            thr[k] = float(v)
    if not any(math.isfinite(v) for v in thr.values()):
        return "no-thresholds", ""
    keep, reasons = _evaluate_gates(
        l1_rel=errors.get("Relative_Error_L1"), l2_rel=errors.get("Relative_Error_L2"),
        linf_rel=errors.get("Relative_Error_Linf"), bias_rel=errors.get("Bias_Rel"),
        q99_rel=errors.get("Q99_Rel"), grad_rel=None, decoded_min=None, decoded_max=None,
        n_corrupt=errors.get("N_Corrupt", 0), thr=thr, grad_threshold=None, grad_gate=False,
        phys_min=(manifest or {}).get("phys_min"), phys_max=(manifest or {}).get("phys_max"))
    if keep:
        return "pass", ""
    failed = ", ".join(k for k, ok in reasons.items() if not ok)
    detail = (f"{var}: verify gate FAILED ({failed}) | "
              f"L1={_fmt3(errors.get('Relative_Error_L1'))} L2={_fmt3(errors.get('Relative_Error_L2'))} "
              f"Linf={_fmt3(errors.get('Relative_Error_Linf'))} bias={_fmt3(errors.get('Bias_Rel'))} "
              f"q99={_fmt3(errors.get('Q99_Rel'))} n_corrupt={errors.get('N_Corrupt', 0)} | thresholds: "
              + " ".join(f"{k}={thr[k]:.3e}" for k in _GATE_KEYS))
    return "fail", detail


# =============================================================================
# 5. CODEC SPACE + PERSISTENCE SHARED BY THE COMPRESS COMMANDS
# =============================================================================

def _codec_spaces(sample_da, full_da, with_lossy, compressor_class, filter_class, serializer_class):
    """(compressors, filters, serializers) built from the sample; the
    FixedScaleOffset range comes from the FULL field so it cannot overflow."""
    fso_range = utils.full_field_data_range(full_da)
    try:
        return (utils.compressor_space(sample_da, with_lossy, compressor_class),
                utils.filter_space(sample_da, with_lossy, filter_class, data_range=fso_range),
                utils.serializer_space(sample_da, with_lossy, serializer_class))
    except ValueError as e:
        raise click.ClickException(str(e))


def _persist_field(da, var: str, merged_path: str, codec_kwargs: dict, *, inner_chunk_mib: int,
                   max_inner_chunk_mib: int, spatial_split: bool, shard_mib: int, threads: int,
                   memory_threshold: float, verify: bool, q99_abs) -> dict:
    """Write one field into the shared store.  Returns a dict with ratio,
    errors, eucd, geometry and timing."""
    inner_chunks, shards = utils.compute_chunk_and_shard_shape(
        da.shape, da.dtype, inner_mib=inner_chunk_mib, shard_mib=shard_mib,
        dims=tuple(da.dims), allow_spatial_split=spatial_split)
    itemsize = int(da.dtype.itemsize)
    inner_bytes = itemsize * int(np.prod(inner_chunks))
    shard_bytes = itemsize * int(np.prod(shards)) if shards is not None else inner_bytes
    if not spatial_split and inner_bytes > max_inner_chunk_mib * 2**20:
        click.echo(f"[chunks] WARNING: --no-spatial-split for '{var}' produced an inner chunk of "
                   f"{_hsize(inner_bytes)} (shape {inner_chunks}), above --max-inner-chunk-mib "
                   f"({max_inner_chunk_mib} MiB).  Codec internals may misbehave at this size.")
    geometry = (f"inner chunks={inner_chunks}, {_hsize(inner_bytes)}; "
                + ("sharding skipped -- one chunk >= shard target" if shards is None
                   else f"shards={shards}, {_hsize(shard_bytes)}"))
    click.echo(f"[persist] {var} -> {merged_path} ({geometry})")

    field_bytes = itemsize * int(np.prod(da.shape))
    write_peak = min(int(threads) * shard_bytes, field_bytes)
    _check_memory_headroom(write_peak, label=f"write peak for '{var}' (threads x write-unit = {_hsize(write_peak)})",
                           threshold=memory_threshold)

    os.makedirs(Path(merged_path).parent, exist_ok=True)
    store = zarr.storage.LocalStore(merged_path, read_only=False)
    t0 = time.perf_counter()
    try:
        try:
            zarr.open_group(store, mode="a", zarr_format=3)  # surface a corrupt/unwritable store early
        except Exception as e:
            click.echo(f"[persist] ERROR: cannot open or create zarr group at {merged_path}: {e}")
            raise
        ratio, errors, eucd = utils.persist_with_codec_pipeline(
            da, store, component=var, codec_kwargs=codec_kwargs, inner_chunks=inner_chunks,
            shards=shards, verify=verify, verbose=False, q99_abs=q99_abs)
    finally:
        _close_store(store)
    return {"ratio": float(ratio), "errors": errors, "eucd": eucd,
            "inner_chunks": list(inner_chunks), "inner_chunk_bytes": int(inner_bytes),
            "shards": (list(shards) if shards is not None else None),
            "shard_bytes": (int(shard_bytes) if shards is not None else None),
            "sharding_skipped": shards is None, "seconds": time.perf_counter() - t0}


def _combo_summary(compressor, filt, serializer, comp_idx, filt_idx, ser_idx, ratio, errors, eucd) -> str:
    text = ("optimal combo:\n"
            f"compressor : {compressor}\nfilter     : {filt}\nserializer : {serializer}\n"
            "corresponding indices in lists of instantiated objects:\n"
            f"compressor : {comp_idx}\nfilter     : {filt_idx}\nserializer : {ser_idx}\n"
            f"Compression Ratio: {ratio:.3f}")
    if errors is not None:
        text += f" | Relative L1 Error: {errors['Relative_Error_L1']:.3e} | Euclidean Distance: {eucd:.3e}"
    else:
        text += "  (error metrics skipped: --no-verify)"
    return text


# =============================================================================
# 6. evaluate_combos
# =============================================================================

def _sweep_sample_limit(var, field_bytes, eval_data_size_limit, threads_per_rank, inner_chunk_mib,
                        ranks_on_node, memory_threshold, rank, size) -> int:
    """Shrink the sample budget so the per-rank steady state fits the node's
    memory budget, then run the memory guards.  Returns the effective limit."""
    node_budget, source = _detect_node_memory_budget()
    effective_budget = node_budget if source.startswith("cgroup") else node_budget // max(1, ranks_on_node)
    max_safe = _max_sample_bytes_for_threads(int(effective_budget * memory_threshold), threads_per_rank, inner_chunk_mib)
    limit = min(int(eval_data_size_limit), max_safe)
    if limit <= 0:
        if rank == 0:
            click.echo(f"[memcheck] FATAL: cannot fit any sample with threads_per_rank={threads_per_rank}, "
                       f"inner_chunk_mib={inner_chunk_mib}, node budget {_hsize(node_budget)} ({source}) at "
                       f"threshold {memory_threshold:.2f}.  Reduce --threads-per-rank or request more RAM.")
        _abort(1)
    if rank == 0 and limit < int(eval_data_size_limit):
        click.echo(f"[memcheck] auto-shrunk sample budget from {_hsize(eval_data_size_limit)} "
                   f"(--eval-data-size-limit) to {_hsize(limit)} to stay under {memory_threshold:.2f} x "
                   f"{_hsize(effective_budget)} ({source}) at {threads_per_rank} threads.")

    sample_bytes = min(field_bytes, limit)
    multiplier = 2 if (rank == 0 and size > 1) else 1  # rank 0 holds two copies during the Bcast
    _check_memory_headroom(multiplier * sample_bytes, label=f"sample for '{var}' on rank {rank} ({_hsize(sample_bytes)})",
                           threshold=memory_threshold)
    _check_node_memory_headroom(
        _per_rank_steady_estimate_bytes(sample_bytes, threads_per_rank, inner_chunk_mib),
        ranks_on_node=ranks_on_node, rank=rank, label=f"variable '{var}', sample {_hsize(sample_bytes)}",
        threshold=memory_threshold)
    return limit


def _sweep_build_sample(da, var, limit, sampling_policy, vertical_floor, comm, rank):
    """Rank 0 builds the sample and the full-field FSO range; both are
    broadcast.  Returns (sample_np, sample_da, fso_range)."""
    if rank == 0:
        try:
            local = utils.build_representative_sample(da, limit, rank=rank, policy=sampling_policy,
                                                      vertical_floor=vertical_floor).compute()
        except utils.SampleTooLargeError:
            comm.Abort(1)  # the other ranks are waiting in the Bcast
        sample_np_local = np.ascontiguousarray(local.values)
        meta = {"dims": tuple(local.dims), "attrs": dict(local.attrs), "name": local.name}
        fso_range = utils.full_field_data_range(da)
    else:
        sample_np_local = meta = fso_range = None
    sample_np = utils.broadcast_numpy(sample_np_local, comm=comm, root=0)
    meta = comm.bcast(meta, root=0)
    fso_range = comm.bcast(fso_range, root=0)
    del sample_np_local
    sample_da = xr.DataArray(sample_np, dims=meta["dims"], attrs=meta["attrs"], name=meta["name"])
    return sample_np, sample_da, fso_range


def _sweep_write_signature(dataset_file, var, where_to_write, eval_data_size_limit, sample_np,
                           effective_limit, sampling_policy, vertical_floor) -> None:
    try:
        sig = _sample_signature(dataset_file, var, int(eval_data_size_limit), sample_np)
        sig.update(effective_sample_limit=int(effective_limit), sampling_policy=str(sampling_policy),
                   vertical_floor=(int(vertical_floor) if vertical_floor is not None else None))
        path = _signature_path(where_to_write, var)
        path.write_text(json.dumps(sig, indent=2))
        click.echo(f"[sample-hash] {var}: sha256={sig['sha256'][:16]}… shape={tuple(sig['shape'])} "
                   f"dtype={sig['dtype']} -> {path.name}")
    except Exception as e:  # non-fatal: compress_with_optimal will warn instead of blocking
        click.echo(f"[sample-hash] WARNING: could not write signature for {var}: {e}")


def _sweep_config_space(compressors, filters, serializers, max_evals, rank) -> list:
    """Valid (c, f, s) triples, capped by --max-evals, then shuffled with a
    seed that depends only on the count (stable across --resume restarts)."""
    total = len(compressors) * len(filters) * len(serializers)
    space = [(c, f, s) for c, f, s in itertools.product(compressors, filters, serializers)
             if utils.combo_is_valid(f[1], s[1])]
    if rank == 0 and len(space) < total:
        click.echo(f"[combo-filter] skipped {total - len(space)} unsupported filter/serializer "
                   f"pairing(s) (e.g. FixedScaleOffset->ZFPY).")
    if max_evals is not None and max_evals < len(space):
        if rank == 0:
            click.echo(f"[max-evals] capping config space at {max_evals} (of {len(space)} possible).")
        space = space[:max_evals]
    perm = np.random.default_rng(seed=len(space) & 0xFFFFFFFF).permutation(len(space))
    return [space[i] for i in perm]


_PARTIAL_CSV_COLUMNS = [
    "compressor", "filter", "serializer", "comp_idx", "filt_idx", "ser_idx",
    "ratio", "l1_rel", "l2_rel", "linf_rel", "bias_rel", "q99_rel", "grad_rel",
    "decoded_min", "decoded_max", "n_corrupt", "eucd",
    "pass_l1", "pass_l2", "pass_linf", "pass_bias", "pass_q99", "pass_bounds", "pass_grad", "pass_finite",
    "keep",
]


def _sweep_run_rank(configs, var, where_to_write, rank, resume, threads_per_rank, evaluate_one, gate):
    """Evaluate this rank's configs in a thread pool, streaming every result to
    config_space_{var}_rank{rank}.csv (and failures to failures_...csv).
    `gate(errors)` -> (keep, reasons).  Returns (results, raw_rows, failures)."""
    partial_path = Path(where_to_write) / f"config_space_{var}_rank{rank}.csv"
    failures_path = Path(where_to_write) / f"failures_{var}_rank{rank}.csv"
    done, mode = set(), "w"
    if resume and partial_path.is_file():
        try:
            prev = pd.read_csv(partial_path)
            done = {(int(a), int(b), int(c)) for a, b, c in zip(prev["comp_idx"], prev["filt_idx"], prev["ser_idx"])}
            mode = "a"
            if rank == 0:
                click.echo(f"[resume] rank 0 found {len(done)} previously-recorded combo(s) for '{var}'; skipping.")
        except Exception as e:
            if rank == 0:
                click.echo(f"[resume] WARNING: failed to parse {partial_path}: {e}.  Starting from scratch.")
    pending = [cfg for cfg in configs if (int(cfg[0][0]), int(cfg[1][0]), int(cfg[2][0])) not in done]
    total = max(1, len(pending))

    results, raw_rows, failures = [], [], []
    FLUSH_EVERY = 100
    with open(partial_path, mode, newline="") as pf, open(failures_path, mode, newline="") as ff:
        pw, fw = csv.writer(pf), csv.writer(ff)
        if mode == "w" or pf.tell() == 0:
            pw.writerow(_PARTIAL_CSV_COLUMNS)
        if mode == "w" or ff.tell() == 0:
            fw.writerow(["compressor", "filter", "serializer", "error"])
        n_rows = n_fail = 0
        # dask runs serially inside each combo thread; the pool is the parallelism.
        with dask.config.set(scheduler="synchronous"), ThreadPoolExecutor(max_workers=threads_per_rank) as pool:
            futures = {pool.submit(evaluate_one, cfg): cfg for cfg in pending}
            for fut in as_completed(futures):
                (_, compressor), (_, filt), (_, serializer) = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:  # one broken combo never stops the sweep
                    failures.append((str(compressor), str(filt), str(serializer), repr(e)))
                    fw.writerow(failures[-1])
                    n_fail += 1
                    if n_fail % FLUSH_EVERY == 0:
                        ff.flush()
                    utils.progress_bar(total, print_every=100, key=str(var))
                    continue
                err = r["errors"]
                keep, reasons = gate(err)
                pw.writerow([
                    r["compressor"], r["filter"], r["serializer"], r["comp_idx"], r["filt_idx"], r["ser_idx"],
                    r["ratio"], err["Relative_Error_L1"], err["Relative_Error_L2"], err["Relative_Error_Linf"],
                    err.get("Bias_Rel"), err.get("Q99_Rel"), err.get("Grad_Rel"),
                    err.get("Decoded_Min"), err.get("Decoded_Max"), err.get("N_Corrupt", 0), r["eucd"],
                    reasons["pass_l1"], reasons["pass_l2"], reasons["pass_linf"], reasons["pass_bias"],
                    reasons["pass_q99"], reasons["pass_bounds"], reasons["pass_grad"], reasons["pass_finite"],
                    keep,
                ])
                n_rows += 1
                if n_rows % FLUSH_EVERY == 0:
                    pf.flush()
                if keep:
                    results.append(((r["compressor"], r["filter"], r["serializer"],
                                     r["comp_idx"], r["filt_idx"], r["ser_idx"]),
                                    r["ratio"], err["Relative_Error_L1"], r["eucd"]))
                    raw_rows.append((r["ratio"], err["Relative_Error_L1"], err["Relative_Error_L2"],
                                     err["Relative_Error_Linf"], r["eucd"],
                                     r["compressor"], r["filter"], r["serializer"]))
                utils.progress_bar(total, print_every=100, key=str(var))
    return results, raw_rows, failures


def _sweep_report_failures(failures, comm, rank, size, var) -> int:
    """Gather a few failures per rank onto rank 0 and print them.  Returns the
    total failure count (rank 0) or None."""
    gathered = comm.gather(failures[:5], root=0)
    total = comm.reduce(len(failures), op=MPI.SUM, root=0)
    if rank == 0 and total:
        click.echo(f"[warning] {total} combo(s) failed total across {size} rank(s).")
        shown = 0
        for r_idx, batch in enumerate(gathered):
            for compressor, filt, serializer, err in batch:
                if shown >= 30:
                    break
                click.echo(f"  [rank {r_idx}] {compressor} | {filt} | {serializer}: {err}")
                shown += 1
        if total > shown:
            click.echo(f"  ... and {total - shown} more (full details in failures_{var}_rank*.csv).")
    return total


def _sweep_select_best(where_to_write, var, results_gather):
    """Consolidate the per-rank CSVs into results_{var}.parquet and pick the
    best kept combo FROM DISK (so --resume of a finished field still finds it).
    Returns (best, n_passed, parquet_path)."""
    partial_paths = sorted(Path(where_to_write).glob(f"config_space_{var}_rank*.csv"))
    parquet_path = None
    if partial_paths:
        # keep_default_na: a "None" codec must stay the string "None", not NaN.
        consolidated = pd.concat([pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in partial_paths],
                                 ignore_index=True)
        parquet_path = os.path.join(where_to_write, f"results_{var}.parquet")
        consolidated.to_parquet(parquet_path, index=False)
        click.echo(f"[sweep] consolidated {len(partial_paths)} per-rank CSV(s) -> {parquet_path} "
                   f"({len(consolidated)} row(s)).")
        kept = consolidated[consolidated["keep"].astype(str).str.strip().str.lower().isin(("true", "1"))]
        if len(kept) == 0:
            return None, 0, parquet_path
        top = kept.loc[kept["ratio"].astype(float).idxmax()]
        best = ((top["compressor"], top["filter"], top["serializer"],
                 int(top["comp_idx"]), int(top["filt_idx"]), int(top["ser_idx"])),
                float(top["ratio"]), float(top["l1_rel"]), float(top["eucd"]))
        return best, int(len(kept)), parquet_path
    if results_gather:
        return max(results_gather, key=lambda x: x[1]), len(results_gather), None
    return None, 0, None


@cli.command("evaluate_combos")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--where-to-write", "where_to_write", required=True,
              type=click.Path(dir_okay=True, file_okay=False, exists=False),
              help="Output directory (config_space_{var}.csv, per-rank CSVs, results_{var}.parquet, "
                   "manifest_{var}.json, .npy).  Created if missing.")
@click.option("--field-to-compress", default=None, help="Field to sweep (default: every data variable).")
@_EVAL_LIMIT_OPTION
@click.option("--threads-per-rank", type=int, default=None,
              help="Combos evaluated concurrently per MPI rank (default: cores / ranks-on-node).")
@_add_options(_CODEC_THREAD_OPTIONS)
@_add_options(_CHUNK_OPTIONS)
@_MEMORY_OPTION
@click.option("--l1-threshold", type=float, required=True,
              help="Relative L1 error budget (e.g. 0.005 = 0.5%).  The anchor for the other gates.")
@click.option("--l2-threshold", type=float, default=None, help="Relative L2 budget (default: 2 x L1).")
@click.option("--linf-threshold", type=float, default=None,
              help="Relative Linf (worst cell) budget (default: 10 x L1).")
@click.option("--bias-threshold", type=float, default=None,
              help="Relative bias budget |mean signed error| / mean|orig| (default: 0.5 x L1).")
@click.option("--q99-threshold", type=float, default=None,
              help="Relative budget over cells with |value| >= the 99th percentile (default: 2 x L1).  "
                   "Only with --extremes-sensitive.")
@click.option("--l2-gate/--no-l2-gate", default=True, show_default=True, help="Enable the L2 gate.")
@click.option("--linf-gate/--no-linf-gate", default=True, show_default=True, help="Enable the Linf gate.")
@click.option("--bias-gate/--no-bias-gate", default=True, show_default=True, help="Enable the bias gate.")
@click.option("--extremes-sensitive/--no-extremes-sensitive", default=False, show_default=True,
              help="Enable the q99 extreme-tail gate (precip, gusts, CAPE, radiation peaks).")
@click.option("--phys-min", type=float, default=None, help="Reject combos whose decoded sample dips below this.")
@click.option("--phys-max", type=float, default=None, help="Reject combos whose decoded sample exceeds this.")
@click.option("--gradient-gate/--no-gradient-gate", default=False, show_default=True,
              help="Enable the spatial-gradient gate (extra decode per combo; for winds, pressure).")
@click.option("--gradient-threshold", type=float, default=0.1, show_default=True,
              help="Max relative L1 error of the finite-difference field (absolute fraction, not x L1).")
@click.option("--gradient-shortcircuit/--no-gradient-shortcircuit", default=True, show_default=True,
              help="Only compute the gradient for combos that already pass the cheap gates.")
@_add_options(_CODEC_SPACE_OPTIONS)
@click.option("--sampling-policy", type=click.Choice(["cascade", "balanced"]), default="cascade",
              show_default=True,
              help="How an over-budget field is thinned: 'cascade' spends the budget on time steps "
                   "first (keeping a minimum of vertical levels); 'balanced' treats every axis equally.")
@click.option("--vertical-floor", type=int, default=None,
              help="Minimum vertical levels kept by the cascade policy "
                   "(default: max(4, ceil(log2(n_levels)))).")
@click.option("--resume/--no-resume", default=True, show_default=True,
              help="Skip combos already present in config_space_{var}_rank{rank}.csv.")
@click.option("--max-evals", type=int, default=None, help="Cap total evaluations (quick test runs).")
@click.option("--allow-multi-rank-per-node/--no-allow-multi-rank-per-node", default=False, show_default=True,
              help="Allow several MPI ranks per node.  Each rank holds its own copy of the sample.")
@click.option("--bypass-zarr-sync/--no-bypass-zarr-sync", default=True, show_default=True,
              help="Give every thread its own zarr event loop (avoids zarr's global sync loop, which "
                   "serialises threads).  Required for the 1 rank x N threads topology.")
def evaluate_combos(dataset_file, where_to_write, field_to_compress, eval_data_size_limit,
                    threads_per_rank, codec_threads, oversubscription_check,
                    inner_chunk_mib, max_inner_chunk_mib, spatial_split, memory_threshold,
                    l1_threshold, l2_threshold, linf_threshold, bias_threshold, q99_threshold,
                    l2_gate, linf_gate, bias_gate, extremes_sensitive, phys_min, phys_max,
                    gradient_gate, gradient_threshold, gradient_shortcircuit,
                    compressor_class, filter_class, serializer_class, with_lossy,
                    sampling_policy, vertical_floor, resume, max_evals,
                    allow_multi_rank_per_node, bypass_zarr_sync):
    """
    Sweep compressor x filter x serializer combinations on a representative
    sample of each field, gate them on error thresholds, and record the best
    combo per field in manifest_{var}.json.

    \b
    Parallelism: MPI ranks split the config space; each rank runs
    --threads-per-rank combos concurrently.  Launch with one rank per node:
      srun --nodes=N --ntasks-per-node=1 --cpus-per-task=32 dc_toolkit evaluate_combos ...
    Everything runs in memory; use compress_with_optimal to write the winner.
    """
    _reset_memcheck_state()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    node_comm, ranks_on_node, _ = utils.detect_node_topology(comm)
    if ranks_on_node > 1 and not allow_multi_rank_per_node:
        if rank == 0:
            click.echo(f"[topology] ERROR: detected {ranks_on_node} MPI rank(s) per node.  This toolkit "
                       f"expects 1 rank per node (threads provide within-node parallelism).\n"
                       f"  Relaunch with --ntasks-per-node=1, or pass --allow-multi-rank-per-node.")
        comm.Abort(1)
    if ranks_on_node > 1 and rank == 0:
        click.echo(f"[topology] NOTE: {ranks_on_node} MPI rank(s) per node; each holds its own sample copy.")
    try:
        node_comm.Free()
    except Exception:
        pass

    cores_avail = utils.detect_cores_available()
    if threads_per_rank is None:
        threads_per_rank = utils.compute_default_threads_per_rank(ranks_on_node, cores_avail)
    if bypass_zarr_sync:
        utils.AsyncBypass.enable(threads_per_rank=threads_per_rank)
        if rank == 0:
            click.echo("[bypass-zarr-sync] enabled.")
    _configure_threads(threads_per_rank, codec_threads, oversubscription_check, rank=rank)

    if rank == 0:
        os.makedirs(where_to_write, exist_ok=True)
    comm.Barrier()

    eff_thr = _derive_thresholds(l1_threshold, l2_threshold, linf_threshold, bias_threshold, q99_threshold,
                                 l2_gate, linf_gate, bias_gate, extremes_sensitive)
    if rank == 0:
        click.echo(_version_banner("evaluate_combos"))
        fmt = lambda x: "off" if not math.isfinite(x) else f"{x:.3e}"  # noqa: E731
        grad = (f"on@{gradient_threshold:.3e} (shortcircuit={'on' if gradient_shortcircuit else 'OFF'})"
                if gradient_gate else "off")
        click.echo(f"[gates] thresholds (relative): L1={eff_thr['l1']:.3e} L2={fmt(eff_thr['l2'])} "
                   f"Linf={fmt(eff_thr['linf'])} bias={fmt(eff_thr['bias'])} q99={fmt(eff_thr['q99'])} | "
                   f"bounds=[{phys_min}, {phys_max}] | gradient={grad}")

    # array.chunk-size must be set before any open() with chunks="auto".
    with dask.config.set({"array.chunk-size": "512MiB", "scheduler": "threads", "num_workers": threads_per_rank}):
        ds = utils.open_dataset(dataset_file, field_to_compress, rank=rank)
        variables = [v for v in ds.data_vars if field_to_compress is None or v == field_to_compress]

        for var in variables:
            da = ds[var]
            var_t0 = time.perf_counter()
            if rank == 0:
                click.echo(f"[var] {var} | units={da.attrs.get('units', 'N/A')} | "
                           f"relative L1 threshold={eff_thr['l1']:.3e}")

            # ---- sample (rank 0 builds, everyone receives) ----
            field_bytes = int(da.dtype.itemsize) * int(np.prod(da.shape))
            limit = _sweep_sample_limit(var, field_bytes, eval_data_size_limit, threads_per_rank, inner_chunk_mib,
                                        ranks_on_node, memory_threshold, rank, size)
            sample_np, sample_da, fso_range = _sweep_build_sample(da, var, limit, sampling_policy, vertical_floor, comm, rank)
            if rank == 0:
                _sweep_write_signature(dataset_file, var, where_to_write, eval_data_size_limit, sample_np,
                                       limit, sampling_policy, vertical_floor)

            # ---- codec space, partitioned across ranks ----
            try:
                compressors = utils.compressor_space(sample_da, with_lossy, compressor_class)
                filters = utils.filter_space(sample_da, with_lossy, filter_class, data_range=fso_range)
                serializers = utils.serializer_space(sample_da, with_lossy, serializer_class)
            except ValueError as e:
                raise click.ClickException(str(e))
            config_space = _sweep_config_space(compressors, filters, serializers, max_evals, rank)
            num_loops = len(config_space)
            configs_for_rank = config_space[rank::size]

            if rank == 0:
                n_nodes = size // ranks_on_node if ranks_on_node else 1
                peak = size * threads_per_rank
                trailer = (f" (only {min(peak, num_loops)} will run concurrently; {num_loops} combos total)"
                           if num_loops < peak else "")
                click.echo(f"[topology] {n_nodes} node(s) x {ranks_on_node} rank(s)/node x {threads_per_rank} "
                           f"thread(s)/rank = {peak} parallel evaluations ({cores_avail} core(s)/rank){trailer}.")
                steady = _per_rank_steady_estimate_bytes(sample_np.nbytes, threads_per_rank, inner_chunk_mib)
                click.echo(f"[memory] rank-0 transient peak ~= {int(2 * sample_np.nbytes / 2**20)} MiB (during Bcast); "
                           f"per-rank steady ~= {int(sample_np.nbytes / 2**20)} MiB (sample) + "
                           f"~{int(threads_per_rank * PER_THREAD_WORKING_FACTOR * sample_np.nbytes / 2**20)} MiB "
                           f"({threads_per_rank} threads x {PER_THREAD_WORKING_FACTOR:.1f}x decode/encode cache) + "
                           f"~{threads_per_rank * max(1, inner_chunk_mib) * 2} MiB (threads x 2 x inner_chunk_mib) "
                           f"= {_hsize(steady)} total.")
                if len(variables) > 1:
                    click.echo(f"[topology] sweep will iterate {len(variables)} variables; one sample Bcast per "
                               f"variable (~{_hsize(sample_np.nbytes)} each over the interconnect).")
                click.echo(f"[sweep] {num_loops} combos ({len(compressors)} x {len(filters)} x {len(serializers)}) "
                           f"split across {size} rank(s); ~{len(configs_for_rank)} per rank, running {threads_per_rank}-wide.")
                pd.DataFrame(config_space).to_csv(os.path.join(where_to_write, f"config_space_{var}.csv"), index=False)

            q99_abs = _q99_cut(sample_np) if extremes_sensitive else None
            if extremes_sensitive and rank == 0:
                click.echo(f"[gates] {var}: q99(|value|)={q99_abs} (extreme-tail cut for the q99 gate)")
            grad_axes = tuple(range(1, sample_np.ndim)) if sample_np.ndim > 1 else (0,)
            eval_chunks = utils.compute_chunk_shape_for_eval(sample_np.shape, sample_np.dtype, target_mib=inner_chunk_mib,
                                                             dims=sample_da.dims, allow_spatial_split=spatial_split)
            eval_chunk_bytes = int(sample_np.dtype.itemsize) * int(np.prod(eval_chunks))
            if not spatial_split and eval_chunk_bytes > max_inner_chunk_mib * 2**20 and rank == 0:
                click.echo(f"[chunks] WARNING: --no-spatial-split produced an eval chunk of {_hsize(eval_chunk_bytes)} "
                           f"(shape {eval_chunks}), above --max-inner-chunk-mib ({max_inner_chunk_mib} MiB).")

            def evaluate_one(cfg):
                (comp_idx, compressor), (filt_idx, filt), (ser_idx, serializer) = cfg
                ratio, errors, eucd = utils.evaluate_codec_pipeline(
                    sample_np, sample_da.dims, utils.codec_pipeline_kwargs(compressor, filt, serializer),
                    chunks=eval_chunks, q99_abs=q99_abs, compute_gradient=gradient_gate,
                    gradient_axes=grad_axes if gradient_gate else None,
                    precheck_thresholds=eff_thr if (gradient_gate and gradient_shortcircuit) else None)
                return {"comp_idx": comp_idx if compressor is not None else -1,
                        "filt_idx": filt_idx if filt is not None else -1,
                        "ser_idx": ser_idx if serializer is not None else -1,
                        "compressor": str(compressor), "filter": str(filt), "serializer": str(serializer),
                        "ratio": float(ratio), "errors": errors, "eucd": float(eucd)}

            def gate(err):
                return _evaluate_gates(
                    l1_rel=err["Relative_Error_L1"], l2_rel=err["Relative_Error_L2"],
                    linf_rel=err["Relative_Error_Linf"], bias_rel=err.get("Bias_Rel"),
                    q99_rel=err.get("Q99_Rel"), grad_rel=err.get("Grad_Rel"),
                    decoded_min=err.get("Decoded_Min"), decoded_max=err.get("Decoded_Max"),
                    n_corrupt=err.get("N_Corrupt", 0), thr=eff_thr, grad_threshold=gradient_threshold,
                    grad_gate=gradient_gate, phys_min=phys_min, phys_max=phys_max)

            # ---- run this rank's share ----
            results, raw_rows, failures = _sweep_run_rank(configs_for_rank, var, where_to_write, rank, resume,
                                                          threads_per_rank, evaluate_one, gate)
            total_failures = _sweep_report_failures(failures, comm, rank, size, var)
            results_gather = comm.gather(results, root=0)
            raw_gather = comm.gather(raw_rows, root=0)
            if rank != 0:
                continue

            # ---- rank 0: results, winner, manifest ----
            click.echo("[sweep] complete. Writing results...")
            results_gather = list(itertools.chain.from_iterable(results_gather))
            raw_gather = list(itertools.chain.from_iterable(raw_gather))
            tag = "_".join([var, compressor_class, filter_class, serializer_class,
                            "with-lossy" if with_lossy else "without-lossy"])
            npy_path = os.path.join(where_to_write, f"{os.path.basename(dataset_file)}_{tag}_scored_results_with_names.npy")
            np.save(npy_path, np.asarray(pd.DataFrame(raw_gather)))
            best, n_passed, parquet_path = _sweep_select_best(where_to_write, var, results_gather)
            if best is not None:
                click.echo(_combo_summary(*best[0][:3], *best[0][3:], best[1], {"Relative_Error_L1": best[2]}, best[3]))
            else:
                click.echo("[sweep] no combos passed the threshold filter.")

            manifest = {
                "command": "evaluate_combos",
                "dataset_file": os.fspath(dataset_file), "var": str(var), "where_to_write": os.fspath(where_to_write),
                "args": {
                    "eval_data_size_limit": int(eval_data_size_limit), "threads_per_rank": int(threads_per_rank),
                    "codec_threads": int(codec_threads or 1), "inner_chunk_mib": int(inner_chunk_mib),
                    "max_inner_chunk_mib": int(max_inner_chunk_mib), "spatial_split": bool(spatial_split),
                    "compressor_class": compressor_class, "filter_class": filter_class,
                    "serializer_class": serializer_class, "with_lossy": bool(with_lossy),
                    "sampling_policy": sampling_policy, "vertical_floor": vertical_floor,
                    "l1_threshold": float(l1_threshold), "l2_threshold": l2_threshold,
                    "linf_threshold": linf_threshold, "bias_threshold": bias_threshold, "q99_threshold": q99_threshold,
                    "l2_gate": bool(l2_gate), "linf_gate": bool(linf_gate), "bias_gate": bool(bias_gate),
                    "extremes_sensitive": bool(extremes_sensitive), "phys_min": phys_min, "phys_max": phys_max,
                    "gradient_gate": bool(gradient_gate), "gradient_threshold": float(gradient_threshold),
                    "resume": bool(resume),
                },
                "topology": {"size": int(size), "cores_avail": int(cores_avail)},
                "effective_thresholds": {k: (None if not math.isfinite(v) else float(v)) for k, v in eff_thr.items()},
                "gradient_threshold": float(gradient_threshold) if gradient_gate else None,
                "phys_min": phys_min, "phys_max": phys_max,
                "num_combos": int(num_loops), "num_passed": int(n_passed),
                "num_failed_total": int(total_failures or 0),
                "num_filtered": int(num_loops - n_passed - (total_failures or 0)),
                "var_sweep_seconds": float(time.perf_counter() - var_t0),
                "env": _env_versions(),
                "sample_signature_path": os.fspath(_signature_path(where_to_write, var)),
                "outputs": {"npy": os.fspath(npy_path), "parquet": parquet_path,
                            "config_space_csv": os.fspath(Path(where_to_write) / f"config_space_{var}.csv")},
                "best": None if best is None else {
                    "compressor": best[0][0], "filter": best[0][1], "serializer": best[0][2],
                    "comp_idx": int(best[0][3]), "filt_idx": int(best[0][4]), "ser_idx": int(best[0][5]),
                    "ratio": float(best[1]), "l1_rel": float(best[2]), "eucd": float(best[3])},
            }
            _write_json(os.path.join(where_to_write, f"manifest_{var}.json"), manifest, "sweep")


# =============================================================================
# 7. compress_with_optimal / compress_fields_from_results
# =============================================================================

@cli.command("compress_with_optimal")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument("where_to_write", type=click.Path(dir_okay=True, file_okay=False, exists=False))
@click.argument("field_to_compress")
@click.argument("comp_idx", type=int)
@click.argument("filt_idx", type=int)
@click.argument("ser_idx", type=int)
@_EVAL_LIMIT_OPTION
@_add_options(_PERSIST_OPTIONS)
@_add_options(_VERIFY_OPTIONS)
@click.option("--l1-threshold", type=float, default=None, help="Override the verify-gate L1 threshold.")
@click.option("--l2-threshold", type=float, default=None, help="Override the verify-gate L2 threshold.")
@click.option("--linf-threshold", type=float, default=None, help="Override the verify-gate Linf threshold.")
@click.option("--bias-threshold", type=float, default=None, help="Override the verify-gate bias threshold.")
@_add_options(_CODEC_SPACE_OPTIONS)
@click.option("--force/--no-force", default=False, show_default=True,
              help="Silence the warnings when the indices differ from the manifest's best combo or the "
                   "library versions differ from the sweep.")
def compress_with_optimal(dataset_file, where_to_write, field_to_compress, comp_idx, filt_idx, ser_idx,
                          eval_data_size_limit, inner_chunk_mib, max_inner_chunk_mib, spatial_split, shard_mib,
                          threads, codec_threads, oversubscription_check, memory_threshold, verify, verify_gate,
                          l1_threshold, l2_threshold, linf_threshold, bias_threshold,
                          compressor_class, filter_class, serializer_class, with_lossy, force):
    """
    Compress ONE field with the (comp_idx, filt_idx, ser_idx) combo found by
    evaluate_combos into {where_to_write}/{dataset}.zarr.  Pass -1 for a
    missing component (after `--`, e.g. `-- ... t -1 0 3`).

    The full field is written; the sample only rebuilds the codec space, so
    --eval-data-size-limit must match the sweep.  Single process; dask threads
    parallelise the write.  Run merge_compressed_fields when all fields are in.
    """
    _require_single_process("compress_with_optimal")
    os.makedirs(where_to_write, exist_ok=True)
    click.echo(_version_banner("compress_with_optimal"))

    manifest_path = Path(where_to_write) / f"manifest_{field_to_compress}.json"
    manifest = _read_json(manifest_path, "manifest")
    if manifest is not None and not force:
        best = manifest.get("best")
        if best is not None:
            best_triple = (int(best["comp_idx"]), int(best["filt_idx"]), int(best["ser_idx"]))
            if (comp_idx, filt_idx, ser_idx) != best_triple:
                click.echo(f"[manifest] WARNING: {manifest_path.name} says best is {best_triple}; you passed "
                           f"{(comp_idx, filt_idx, ser_idx)}.\n  Best combo per manifest: compressor={best['compressor']}  "
                           f"filter={best['filter']}  serializer={best['serializer']}  ratio={best['ratio']:.3f}\n"
                           f"  Proceeding anyway (pass --force to silence).")
        _warn_env_drift(manifest.get("env") or {}, manifest_path.name)

    _reset_memcheck_state()
    cores_avail = utils.detect_cores_available()
    threads = cores_avail if threads is None else threads
    _configure_threads(threads, codec_threads, oversubscription_check)

    with dask.config.set(scheduler="threads", num_workers=int(threads)):
        click.echo(f"[topology] {cores_avail} core(s) visible; dask will use {threads} worker(s) for the write. "
                   f"Peak working set ~= {threads} x shard_mib ({shard_mib} MiB) = {threads * shard_mib} MiB.")
        ds = utils.open_dataset(dataset_file, field_to_compress)
        da = ds[field_to_compress]
        field_bytes = int(da.dtype.itemsize) * int(np.prod(da.shape))
        _check_memory_headroom(min(int(threads) * int(shard_mib) * 2**20, field_bytes),
                               label=f"write peak for '{field_to_compress}'", threshold=memory_threshold)
        _check_memory_headroom(min(field_bytes, int(eval_data_size_limit)),
                               label=f"codec-space sample for '{field_to_compress}'", threshold=memory_threshold)

        sample, status = _rebuild_sweep_sample(da, field_to_compress, dataset_file, where_to_write, eval_data_size_limit)
        if status == "mismatch":
            sys.exit(1)
        spaces = _codec_spaces(sample, da, with_lossy, compressor_class, filter_class, serializer_class)
        compressor, filt, serializer = _resolve_combo(spaces, comp_idx, filt_idx, ser_idx, field_to_compress)
        q99_abs = _q99_cut(np.ascontiguousarray(sample.values)) if verify else None

        merged_path = _merged_store_path(where_to_write, dataset_file)
        out = _persist_field(da, field_to_compress, merged_path, utils.codec_pipeline_kwargs(compressor, filt, serializer),
                             inner_chunk_mib=inner_chunk_mib, max_inner_chunk_mib=max_inner_chunk_mib,
                             spatial_split=spatial_split, shard_mib=shard_mib, threads=threads,
                             memory_threshold=memory_threshold, verify=verify, q99_abs=q99_abs)
        click.echo(_combo_summary(compressor, filt, serializer, comp_idx, filt_idx, ser_idx,
                                  out["ratio"], out["errors"], out["eucd"]))

        gate_failed = False
        if verify and out["errors"] is not None:
            overrides = {"l1": l1_threshold, "l2": l2_threshold, "linf": linf_threshold, "bias": bias_threshold}
            status, detail = _verify_against_manifest(field_to_compress, out["errors"], manifest, overrides)
            if status == "no-thresholds":
                click.echo(f"[verify-gate] WARNING: no thresholds available (no {manifest_path.name} and no "
                           f"--lX-threshold given); verification was advisory only.")
            elif status == "pass":
                click.echo("[verify-gate] PASS: production error norms are within the sweep thresholds.")
            else:
                click.echo(f"[verify-gate] FAIL: {detail}")
                if verify_gate:
                    click.echo("[verify-gate] aborting (pass --no-verify-gate to downgrade this to a warning).")
                    gate_failed = True
                else:
                    click.echo("[verify-gate] --no-verify-gate set: continuing (advisory only).")

        _write_json(os.path.join(where_to_write, f"persist_manifest_{field_to_compress}.json"), {
            "command": "compress_with_optimal",
            "dataset_file": os.fspath(dataset_file), "var": str(field_to_compress),
            "where_to_write": os.fspath(where_to_write), "merged_store": merged_path,
            "args": {"comp_idx": int(comp_idx), "filt_idx": int(filt_idx), "ser_idx": int(ser_idx),
                     "eval_data_size_limit": int(eval_data_size_limit), "inner_chunk_mib": int(inner_chunk_mib),
                     "max_inner_chunk_mib": int(max_inner_chunk_mib), "spatial_split": bool(spatial_split),
                     "shard_mib": int(shard_mib), "threads": int(threads), "verify": bool(verify),
                     "force": bool(force), "compressor_class": compressor_class,
                     "filter_class": filter_class, "serializer_class": serializer_class},
            "inner_chunks": out["inner_chunks"], "inner_chunk_bytes": out["inner_chunk_bytes"],
            "shards": out["shards"], "shard_bytes": out["shard_bytes"], "sharding_skipped": out["sharding_skipped"],
            "compressor": str(compressor), "filter": str(filt), "serializer": str(serializer),
            "ratio": out["ratio"],
            "errors": {k: (float(v) if v is not None else None) for k, v in (out["errors"] or {}).items()},
            "eucd": (float(out["eucd"]) if out["eucd"] is not None else None),
            "persist_seconds": float(out["seconds"]), "env": _env_versions(),
        }, "persist")
        if gate_failed:
            _abort(2)


@cli.command("compress_fields_from_results")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument("where_to_write", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option("--vars", "vars_filter", default=None,
              help="Comma-separated variables to process (default: every manifest_{var}.json / results_{var}.parquet).")
@_EVAL_LIMIT_OPTION
@_add_options(_PERSIST_OPTIONS)
@_add_options(_VERIFY_OPTIONS)
@click.option("--cr-drift-tol", type=click.FloatRange(0.0, 10.0), default=0.25, show_default=True,
              help="Allowed fractional drift between achieved and predicted compression ratio.")
@click.option("--cr-drift-gate/--no-cr-drift-gate", default=False, show_default=True,
              help="Fail a field whose ratio falls short of the prediction by more than --cr-drift-tol "
                   "(default: warn only).")
@_add_options(_CODEC_SPACE_OPTIONS)
@click.option("--skip-existing/--no-skip-existing", default=True, show_default=True,
              help="Skip fields already present in the merged store.")
@click.option("--continue-on-error/--no-continue-on-error", default=True, show_default=True,
              help="Log and continue when one field fails (default) instead of stopping the run.")
def compress_fields_from_results(dataset_file, where_to_write, vars_filter, eval_data_size_limit,
                                 inner_chunk_mib, max_inner_chunk_mib, spatial_split, shard_mib,
                                 threads, codec_threads, oversubscription_check, memory_threshold,
                                 verify, verify_gate, cr_drift_tol, cr_drift_gate,
                                 compressor_class, filter_class, serializer_class, with_lossy,
                                 skip_existing, continue_on_error):
    """
    Batch version of compress_with_optimal: read the best combo of every
    variable from manifest_{var}.json (or results_{var}.parquet) and persist
    each field into the shared {dataset}.zarr store, one after the other.
    Single process; dask threads parallelise each write.
    """
    _require_single_process("compress_fields_from_results")
    click.echo(_version_banner("compress_fields_from_results"))
    _reset_memcheck_state()
    threads = utils.detect_cores_available() if threads is None else threads
    _configure_threads(threads, codec_threads, oversubscription_check)

    # ---- which (var, combo) to write: manifests first, parquet as fallback ----
    wtw = Path(where_to_write)
    candidates, manifests, sweep_env = [], {}, None
    for mpath in sorted(wtw.glob("manifest_*.json")):
        var = mpath.stem.removeprefix("manifest_")
        m = _read_json(mpath, "batch")
        if m is None:
            continue
        if m.get("best") is None:
            click.echo(f"[batch] {var}: manifest has no best combo; skipping.")
            continue
        try:
            best = m["best"]
            candidates.append({"var": var, "comp_idx": int(best["comp_idx"]), "filt_idx": int(best["filt_idx"]),
                               "ser_idx": int(best["ser_idx"]), "source": f"manifest {mpath.name}"})
            manifests[var] = m
            sweep_env = sweep_env or m.get("env")
        except Exception as e:
            click.echo(f"[batch] WARNING: failed to parse {mpath}: {e}")
    if sweep_env:
        _warn_env_drift(sweep_env, "these manifests")
    for ppath in sorted(wtw.glob("results_*.parquet")):
        var = ppath.stem.removeprefix("results_")
        if var in manifests:
            continue
        try:
            df = pd.read_parquet(ppath)
            kept = df[df["keep"].astype(str).str.strip().str.lower().isin(("true", "1"))] if "keep" in df.columns else df
            if len(kept) == 0:
                click.echo(f"[batch] {var}: no kept rows in {ppath.name}; skipping.")
                continue
            row = kept.sort_values("ratio", ascending=False).iloc[0]
            candidates.append({"var": var, "comp_idx": int(row["comp_idx"]), "filt_idx": int(row["filt_idx"]),
                               "ser_idx": int(row["ser_idx"]), "source": f"parquet {ppath.name}"})
        except Exception as e:
            click.echo(f"[batch] WARNING: failed to parse {ppath}: {e}")
    if vars_filter:
        wanted = {v.strip() for v in vars_filter.split(",") if v.strip()}
        candidates = [c for c in candidates if c["var"] in wanted]
        missing = wanted - {c["var"] for c in candidates}
        if missing:
            click.echo(f"[batch] WARNING: --vars specified {sorted(missing)} but no manifest/parquet was found for those.")
    if not candidates:
        click.echo("[batch] ERROR: no variables to compress.  Did evaluate_combos run against the same --where-to-write?")
        sys.exit(1)
    click.echo(f"[batch] will compress {len(candidates)} field(s): {', '.join(c['var'] for c in candidates)}")

    ds = utils.open_dataset(dataset_file, field_to_compress=None)
    merged_path = _merged_store_path(where_to_write, dataset_file)
    existing = set()
    if Path(merged_path).is_dir():
        try:
            group, store = utils.open_zarr_localstore(merged_path, read_only=True)
            existing = set(group.array_keys())
            _close_store(store)
        except Exception:
            pass

    results_by_var, any_error = {}, False
    with dask.config.set(scheduler="threads", num_workers=int(threads)):
        for i, c in enumerate(candidates, start=1):
            var = c["var"]
            click.echo(f"\n[batch] ({i}/{len(candidates)}) {var} from {c['source']}: "
                       f"comp={c['comp_idx']} filt={c['filt_idx']} ser={c['ser_idx']}")
            if skip_existing and var in existing:
                click.echo(f"[batch] {var} already in {merged_path}; skipping.")
                results_by_var[var] = {"status": "skipped-existing"}
                continue
            if var not in ds.data_vars:
                if not continue_on_error:
                    click.echo(f"[batch] ERROR: variable '{var}' not in dataset; aborting.")
                    sys.exit(1)
                click.echo(f"[batch] WARNING: variable '{var}' not in dataset; skipping.")
                results_by_var[var] = {"status": "missing-from-dataset"}
                continue

            try:
                da = ds[var]
                field_bytes = int(da.dtype.itemsize) * int(np.prod(da.shape))
                _check_memory_headroom(min(int(threads) * int(shard_mib) * 2**20, field_bytes),
                                       label=f"write peak for '{var}'", threshold=memory_threshold)
                _check_memory_headroom(min(field_bytes, int(eval_data_size_limit)),
                                       label=f"codec-space sample for '{var}'", threshold=memory_threshold)

                sample, status = _rebuild_sweep_sample(da, var, dataset_file, where_to_write, eval_data_size_limit)
                if status == "mismatch":
                    if not continue_on_error:
                        sys.exit(1)
                    click.echo(f"[batch] skipping {var} (sample signature mismatch).")
                    results_by_var[var] = {"status": "signature-mismatch"}
                    continue
                spaces = _codec_spaces(sample, da, with_lossy, compressor_class, filter_class, serializer_class)
                compressor, filt, serializer = _resolve_combo(spaces, c["comp_idx"], c["filt_idx"], c["ser_idx"], var)
                q99_abs = _q99_cut(np.ascontiguousarray(sample.values)) if verify else None

                out = _persist_field(da, var, merged_path, utils.codec_pipeline_kwargs(compressor, filt, serializer),
                                     inner_chunk_mib=inner_chunk_mib, max_inner_chunk_mib=max_inner_chunk_mib,
                                     spatial_split=spatial_split, shard_mib=shard_mib, threads=threads,
                                     memory_threshold=memory_threshold, verify=verify, q99_abs=q99_abs)
                summary = f"{var}: ratio={out['ratio']:.3f}"
                if verify:
                    summary += f" L1_rel={out['errors']['Relative_Error_L1']:.3e} eucd={out['eucd']:.3e}"
                click.echo(f"[batch] {summary}  ({out['seconds']:.1f}s)")

                manifest = manifests.get(var) or _read_json(wtw / f"manifest_{var}.json", "verify-gate")
                if verify and out["errors"] is not None:
                    status, detail = _verify_against_manifest(var, out["errors"], manifest)
                    if status == "no-thresholds":
                        click.echo(f"[verify-gate] {var}: no thresholds in manifest; verification advisory only.")
                    elif status == "pass":
                        click.echo(f"[verify-gate] {var}: PASS")
                    elif verify_gate:
                        raise RuntimeError(detail)
                    else:
                        click.echo(f"[verify-gate] {detail} (advisory: --no-verify-gate set)")

                predicted = drift = None
                if verify:
                    predicted = ((manifest or {}).get("best") or {}).get("ratio")
                    ok, drift, direction = _evaluate_cr_drift(out["ratio"], predicted, cr_drift_tol)
                    if direction == "skip":
                        click.echo(f"[cr-drift] {var}: no predicted ratio in manifest; skipped.")
                    elif ok:
                        click.echo(f"[cr-drift] {var}: PASS (achieved {out['ratio']:.2f}x vs predicted "
                                   f"{predicted:.2f}x, drift {drift:+.1%})")
                    else:
                        msg = (f"{var}: CR drift {drift:+.1%} exceeds +/-{cr_drift_tol:.0%} "
                               f"(achieved {out['ratio']:.2f}x vs predicted {predicted:.2f}x)")
                        if direction == "under" and cr_drift_gate:
                            raise RuntimeError(f"cr-drift gate FAILED: {msg}")
                        click.echo(f"[cr-drift] WARNING {msg}"
                                   + ("" if direction == "under" else "  (better than predicted; informational)"))

                results_by_var[var] = {
                    "status": "ok", "ratio": out["ratio"],
                    "predicted_ratio": (float(predicted) if predicted is not None else None),
                    "cr_drift": (float(drift) if drift is not None else None),
                    "errors": {k: (float(v) if v is not None else None) for k, v in (out["errors"] or {}).items()},
                    "eucd": (float(out["eucd"]) if out["eucd"] is not None else None),
                    "seconds": float(out["seconds"]),
                    "comp_idx": int(c["comp_idx"]), "filt_idx": int(c["filt_idx"]), "ser_idx": int(c["ser_idx"]),
                }
            except Exception as e:
                any_error = True
                click.echo(f"[batch] ERROR on {var}: {e!r}")
                results_by_var[var] = {"status": "error", "error": repr(e)}
                if not continue_on_error:
                    raise

    _write_json(os.path.join(where_to_write, "batch_manifest.json"), {
        "command": "compress_fields_from_results", "dataset_file": os.fspath(dataset_file),
        "where_to_write": os.fspath(where_to_write), "merged_store": merged_path,
        "results": results_by_var, "any_error": any_error,
    }, "batch")
    if any_error and not continue_on_error:
        sys.exit(1)


# =============================================================================
# 8. STORE UTILITIES & FORMAT CONVERSION
# =============================================================================

@cli.command("merge_compressed_fields")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument("compressed_files_location", type=click.Path(dir_okay=True, file_okay=False, exists=False))
def merge_compressed_fields(dataset_file: str, compressed_files_location: str):
    """Consolidate metadata on {compressed_files_location}/{dataset}.zarr so
    readers open it without scanning every array."""
    _require_single_process("merge_compressed_fields")
    merged_path = _merged_store_path(compressed_files_location, dataset_file)
    if not Path(merged_path).is_dir():
        click.echo(f"Expected merged store not found: {merged_path}\n"
                   f"Did compress_with_optimal run at least once with the same `where_to_write`?")
        sys.exit(1)
    store = zarr.storage.LocalStore(merged_path, read_only=False)
    try:
        zarr.consolidate_metadata(store)
        click.echo(f"[merge] consolidated metadata on {merged_path}")
        names = list(zarr.open_group(store, mode="r").array_keys())
        click.echo(f"[merge] arrays in store ({len(names)}): {', '.join(names) or '<none>'}")
    finally:
        _close_store(store)


@cli.command("open_zarr_and_inspect")
@click.argument("zarr_path", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--head", type=int, default=4, show_default=True,
              help="Elements per dim to preview from each array (0 = metadata only).")
def open_zarr_and_inspect(zarr_path: str, head: int):
    """Print the group tree, per-array metadata (codecs, sharding, ratio) and a
    tiny head slice of a zarr v3 store."""
    _require_single_process("open_zarr_and_inspect")
    group, _store = utils.open_zarr_localstore(zarr_path, read_only=True)
    click.echo(group.tree())
    click.echo("-" * 80)
    for name in group.array_keys():
        z = group[name]
        click.echo(f"Array: {name}")
        click.echo(z.info_complete())
        if head > 0:
            slicer = tuple(slice(0, min(head, s)) for s in z.shape)
            click.echo(f"Head slice {slicer}:")
            click.echo(z[slicer])
        click.echo("-" * 80)


@cli.command("from_nc_to_zarr")
@click.argument("nc_path", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.option("--out", "out_zarr", type=click.Path(dir_okay=True, file_okay=False), default=None,
              help="Output .zarr directory (default: input path with .zarr).")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True,
              help="Remove an existing output directory first.")
@click.option("--consolidated/--no-consolidated", default=True, show_default=True,
              help="Write consolidated metadata.")
@click.option("--preserve-source-chunks/--no-preserve-source-chunks", default=True, show_default=True,
              help="Map each HDF5 chunk 1:1 to a zarr chunk (chunks={}).  Use --no-preserve-source-chunks "
                   "for netCDF-3 or contiguous variables (chunks='auto').")
@click.option("--mask-and-scale/--no-mask-and-scale", default=False, show_default=True,
              help="Apply CF scale_factor/add_offset/_FillValue at read time.  Off keeps packed ints packed "
                   "on disk (the attrs ride along, so readers still decode).")
@click.option("--decode-times/--no-decode-times", default=False, show_default=True,
              help="Apply CF time decoding at read time.  Off keeps the on-disk numeric form.")
@click.option("--threads", type=int, default=None, help="Dask workers (default: visible cores).")
def from_nc_to_zarr(nc_path, out_zarr, overwrite, consolidated, preserve_source_chunks,
                    mask_and_scale, decode_times, threads):
    """
    Convert a NetCDF file to an UNCOMPRESSED zarr v3 store (no filters, no
    compressors, no sharding; coordinates included), for filesystem-level
    deduplication experiments.  Whatever compression the netCDF had is undone
    by the reader; the output stores plain bytes.
    """
    _require_single_process("from_nc_to_zarr")
    if Path(nc_path).suffix.lower() != ".nc":
        click.echo(f"Expected a .nc file, got {nc_path}.  Use from_zarr_to_netcdf for the reverse direction.")
        sys.exit(1)
    out_zarr = out_zarr or str(Path(nc_path).with_suffix(".zarr"))
    if Path(out_zarr).exists():
        if not overwrite:
            click.echo(f"Output already exists: {out_zarr}.  Pass --overwrite to replace, or pick a different --out.")
            sys.exit(1)
        import shutil
        click.echo(f"[nc->zarr] removing existing {out_zarr} (--overwrite).")
        shutil.rmtree(out_zarr)
    threads = utils.detect_cores_available() if threads is None else threads

    click.echo(f"[nc->zarr] reading {nc_path} ...")
    with dask.config.set(scheduler="threads", num_workers=int(threads)):
        ds = xr.open_dataset(nc_path, chunks={} if preserve_source_chunks else "auto",
                             mask_and_scale=mask_and_scale, decode_times=decode_times)
        click.echo(f"[nc->zarr] logical size = {_hsize(int(ds.nbytes))} "
                   f"| chunks = {'source-native' if preserve_source_chunks else 'auto'} "
                   f"| mask_and_scale = {mask_and_scale} | decode_times = {decode_times} | dask workers = {threads}")
        # Clear netCDF-side encoding and force no codecs on every variable and coordinate.
        encoding = {}
        for name in ds.variables:
            ds[name].encoding = {}
            encoding[name] = {"compressors": None, "filters": None}
        click.echo(f"[nc->zarr] writing {out_zarr} (compressors=None, filters=None, {len(encoding)} variable(s)) ...")
        ds.to_zarr(out_zarr, mode="w-", encoding=encoding, zarr_format=3, consolidated=consolidated)
    click.echo(f"[nc->zarr] wrote {out_zarr}")


@cli.command("from_zarr_to_netcdf")
@click.argument("zarr_path", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--out", "out_nc", type=click.Path(dir_okay=False), default=None,
              help="Output NetCDF file (default: input path with .nc).")
@click.option("--max-size", default="50GB", callback=_size_option_callback, show_default=True,
              help="Refuse to write when the logical output exceeds this size.")
@click.option("--compression", default="zlib", show_default=True, help="NetCDF variable compression (zlib/none).")
@click.option("--complevel", default=4, show_default=True, help="zlib compression level.")
@click.option("--threads", type=int, default=None, help="Dask workers (default: visible cores).")
@click.option("--codec-threads", type=int, default=1, show_default=True,
              help="Codec-internal threads per decode call; threads x codec-threads must fit the cores.")
def from_zarr_to_netcdf(zarr_path, out_nc, max_size, compression, complevel, threads, codec_threads):
    """Convert a zarr v3 store to a NetCDF4 file, streamed through dask."""
    _require_single_process("from_zarr_to_netcdf")
    out_nc = out_nc or str(Path(zarr_path).with_suffix(".nc"))
    threads = utils.detect_cores_available() if threads is None else threads
    _apply_codec_threads(codec_threads)
    _check_thread_product(threads, codec_threads)

    with dask.config.set(scheduler="threads", num_workers=int(threads)):
        try:
            ds = xr.open_zarr(zarr_path, chunks="auto", consolidated=True)
        except Exception:
            ds = xr.open_zarr(zarr_path, chunks="auto", consolidated=False)
        logical_bytes = int(ds.nbytes)
        click.echo(f"[zarr->nc] logical size = {_hsize(logical_bytes)} | dask workers = {threads} "
                   f"| codec-threads = {codec_threads}")
        if logical_bytes > max_size:
            click.echo(f"Refusing to write: logical size exceeds --max-size ({_hsize(max_size)}).  "
                       f"Raise --max-size to proceed, or keep the data in .zarr.")
            sys.exit(1)
        encoding = {}
        for name, var in ds.data_vars.items():
            enc = {}
            if isinstance(var.data, dask.array.Array):
                enc["chunksizes"] = tuple(max(b) for b in var.data.chunks)
            if compression == "zlib":
                enc.update(zlib=True, complevel=int(complevel))
            encoding[name] = enc
        click.echo(f"[zarr->nc] writing {out_nc} ...")
        ds.to_netcdf(out_nc, engine="h5netcdf", encoding=encoding)
    click.echo(f"[zarr->nc] wrote {out_nc}")


# =============================================================================
# 9. ANALYSIS & PLOTTING
# =============================================================================
# The scored-results .npy holds one row per KEPT combo:
#   [ratio, L1, L2, LInf, eucd, compressor, filter, serializer]

def _load_scored_results(npy_file: str) -> pd.DataFrame:
    df = pd.DataFrame(np.load(npy_file, allow_pickle=True))
    numeric = df.select_dtypes(include=[np.number]).columns
    return df[np.isfinite(df[numeric]).all(axis=1)].dropna()


@cli.command("perform_clustering")
@click.argument("npy_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("l_error", type=str)
def perform_clustering(npy_file: str, l_error: str):
    """
    Elbow and silhouette scores of KMeans (k = 3..9) on compression ratio vs
    the chosen error, from a scored-results .npy written by evaluate_combos.

    \b
    Args:
        npy_file: *_scored_results_with_names.npy
        l_error:  "L1", "L2" or "LInf"
    """
    from tqdm import tqdm
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt

    options = ["L1", "L2", "LInf"]
    if l_error not in options:
        raise click.ClickException(f"--l-error must be one of {options}; got {l_error!r}.")
    df = _load_scored_results(npy_file)
    k_max = min(9, len(df) - 1)
    if k_max < 3:
        click.echo(f"[perform_clustering] only {len(df)} finite passing combo(s) in {Path(npy_file).name}; "
                   f"need >= 4 to cluster.  Nothing to plot.")
        return
    data = np.hstack((np.asarray(df[[0]]), np.asarray(df[[options.index(l_error) + 1]])))
    k_values = range(3, k_max + 1)
    inertias, silhouettes = [], []
    for k in tqdm(k_values, desc="Looping over k values"):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(data, labels))

    plt.figure(figsize=(12, 5))
    plt.suptitle(l_error, fontsize=20)
    plt.subplot(1, 2, 1); plt.plot(k_values, inertias, "bo-")
    plt.xlabel("Number of Clusters (k)"); plt.ylabel("Inertia"); plt.title("Elbow Method for Optimal k")
    plt.subplot(1, 2, 2); plt.plot(k_values, silhouettes, "go-")
    plt.xlabel("Number of Clusters (k)"); plt.ylabel("Silhouette Score"); plt.title("Silhouette Score for Optimal k")
    plt.tight_layout()
    plt.show()


@cli.command("analyze_clustering")
@click.argument("npy_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--where-to-write", "where_to_write", required=True,
              type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="Directory holding config_space_{var}.csv from evaluate_combos.")
@click.option("--var", "var", required=True, type=str, help="Field name used in evaluate_combos.")
def analyze_clustering(npy_file: str, where_to_write: str, var: str):
    """Interactive KMeans scatter plots of L1 / L2 / LInf vs compression ratio
    (opens in the browser), from a scored-results .npy."""
    from sklearn.cluster import KMeans
    import plotly.io as pio
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    config_csv = Path(where_to_write) / f"config_space_{var}.csv"
    if not config_csv.is_file():
        raise click.FileError(str(config_csv), hint=f"Run evaluate_combos with --where-to-write {where_to_write} "
                                                    f"first and check that --var matches its field.")
    config_idxs = pd.read_csv(config_csv)
    df = _load_scored_results(npy_file)
    if len(df) == 0:
        click.echo(f"[analyze_clustering] no finite passing combos in {Path(npy_file).name}; nothing to plot.")
        return

    max_n_rows, max_nclusters = 42976, 6
    n_clusters = max(1, min(len(df), math.ceil(max_nclusters * len(df) / max_n_rows)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    metrics = [("L1", 1), ("L2", 2), ("LInf", 3)]
    fig = make_subplots(rows=3, cols=1, subplot_titles=[f"{m} VS Ratio KMeans Clustering" for m, _ in metrics])
    hover = ["compressor", "filter", "serializer", "compressor_idx", "filter_idx", "serializer_idx"]

    for row, (metric, col) in enumerate(metrics, start=1):
        arr = utils.slice_array(df, [0, col, 5, 6, 7])
        points = pd.DataFrame(np.column_stack((arr[:, 0].astype(float), arr[:, 1].astype(float))),
                              columns=["Ratio", metric])
        points["compressor"], points["filter"], points["serializer"] = arr[:, 2], arr[:, 3], arr[:, 4]
        points["compressor_idx"] = utils.get_indexes(arr[:, 2], config_idxs["0"])
        points["filter_idx"] = utils.get_indexes(arr[:, 3], config_idxs["1"])
        points["serializer_idx"] = utils.get_indexes(arr[:, 4], config_idxs["2"])
        labels = kmeans.fit_predict(points[["Ratio", metric]])
        color = np.ones(labels.shape) if len(np.unique(labels)) == 1 else labels
        scatter = px.scatter(points, x="Ratio", y=metric, color=color,
                             title=f"{metric} VS Ratio KMeans Clustering", hover_data=hover)
        fig.add_trace(go.Scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
                                 mode="markers+text", marker=dict(color="black", size=12, symbol="x"),
                                 textposition="top center", name="Centroids", showlegend=(row == 1)),
                      row=row, col=1)
        fig.update_xaxes(title_text="Ratio", row=row, col=1)
        fig.update_yaxes(title_text=metric, row=row, col=1)
        for trace in scatter.data:
            fig.add_trace(trace, row=row, col=1)

    fig.update_layout(title="", showlegend=False, height=900, hovermode="closest", template="plotly_white")
    pio.renderers.default = "browser"
    fig.show()


@cli.command("plot_compression_errors")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument("where_to_write", type=click.Path(dir_okay=True, file_okay=False, exists=False))
@click.argument("field_to_compress")
@click.argument("comp_idx", type=int)
@click.argument("filt_idx", type=int)
@click.argument("ser_idx", type=int)
@_add_options(_CODEC_SPACE_OPTIONS)
def plot_compression_errors(dataset_file, where_to_write, field_to_compress, comp_idx, filt_idx, ser_idx,
                            compressor_class, filter_class, serializer_class, with_lossy):
    """
    Save a 3x3 PDF of absolute compression errors for one (lat, lon) field
    with one codec combo, including a copy shifted by 180 degrees in longitude
    to reveal whether the combo respects periodicity.

    Pass the same --*-class / --with-lossy flags as in evaluate_combos so the
    indices resolve to the same codecs; use `--` before negative indices.
    """
    import matplotlib.pyplot as plt

    _require_single_process("plot_compression_errors")
    os.makedirs(where_to_write, exist_ok=True)
    da = utils.open_dataset(dataset_file, field_to_compress)[field_to_compress].squeeze()
    click.echo(f"Squeezed (lat, lon) field_to_compress.nbytes = {_hsize(da.nbytes)}")
    if not utils.is_lat_lon(da):
        raise click.ClickException(f"Field {field_to_compress} must have dimensions (lat, lon); it has {da.dims}.")
    if da.nbytes / 2**30 > 2.5:
        raise click.ClickException(f"Field {field_to_compress} is too large ({_hsize(da.nbytes)}); max 2.5 GiB.")

    spaces = _codec_spaces(da, da, with_lossy, compressor_class, filter_class, serializer_class)
    compressor, filt, serializer = _resolve_combo(spaces, comp_idx, filt_idx, ser_idx, field_to_compress)
    codec_kwargs = utils.codec_pipeline_kwargs(compressor, filt, serializer)
    click.echo(f"Shape of {field_to_compress}: {da.shape}")

    lon_dim = da.dims[1]
    half = da.sizes[lon_dim] // 2
    shifted = da.roll({lon_dim: -half}, roll_coords=False)

    def roundtrip(arr):
        store = utils.open_zarr_memstore()
        z = zarr.create_array(store=store, name=field_to_compress, data=np.ascontiguousarray(arr.values),
                              chunks="auto", **codec_kwargs)
        out = xr.DataArray(z[:], dims=da.dims, coords=da.coords)
        store.close()
        return out

    decoded = roundtrip(da)
    shifted_decoded = roundtrip(shifted)
    shifted_back = shifted.roll({lon_dim: half}, roll_coords=False)
    shifted_decoded_back = shifted_decoded.roll({lon_dim: half}, roll_coords=False)

    panels = [
        ("Original", da, None),
        ("Original compressed&decompressed", decoded, None),
        ("Absolute error [original - original c&d]", np.abs(da - decoded), "binary"),
        ("Shifted (by +180 deg)", shifted, None),
        ("Shifted compressed&decompressed", shifted_decoded, None),
        ("Absolute error [shifted - shifted c&d]", np.abs(shifted - shifted_decoded), "binary"),
        ("Absolute error [original - (shifted-180)]", np.abs(da - shifted_back) / (np.abs(da) + 1e-20), "binary"),
        ("Absolute error [original c&d - (shifted c&d-180)]", np.abs(decoded - shifted_decoded_back), "binary"),
        ("Absolute error [original - (shifted c&d-180)]", np.abs(da - shifted_decoded_back), "binary"),
    ]
    fig, axes = plt.subplots(3, 3, layout="constrained", figsize=(16, 9))
    fig.suptitle(f"Compression errors for variable {field_to_compress} ({da.attrs['units']})", fontsize=16)
    for ax, (title, data, cmap) in zip(axes.flatten(), panels):
        ax.set_title(title, fontsize=10)
        fig.colorbar(ax.imshow(data, interpolation="none", cmap=cmap), ax=ax, shrink=0.7)
    fig.savefig(os.path.join(where_to_write, f"{field_to_compress}_compression_errors.pdf"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 10. UIs & HELP
# =============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))


@cli.command("run_web_ui_vcluster")
@click.option("--user_account", type=str, default="", help="vCluster user account name")
@click.option("--uenv_image", type=str, default="", help="vCluster uenv image name")
@click.option("--uploaded_file", type=str, default="", help="Uploaded file from vcluster")
@click.option("--time", type=str, default="", help="Allocated time")
@click.option("--nodes", type=str, default="", help="Number of nodes")
@click.option("--ntasks-per-node", type=str, default="", help="Number of tasks per node")
def run_web_ui_vcluster(user_account, uenv_image, uploaded_file, time, nodes, ntasks_per_node):
    """Streamlit web UI launched from a vcluster."""
    subprocess.run(["streamlit", "run", os.path.join(_HERE, "compression_analysis_ui_vcluster.py"), "--",
                    "--user_account", user_account, "--uenv_image", uenv_image, "--uploaded_file", uploaded_file,
                    "--time", time, "--nodes", nodes, "--ntasks-per-node", ntasks_per_node])


@cli.command("run_web_ui")
def run_web_ui():
    """Streamlit web UI launched from a local terminal."""
    subprocess.run(["streamlit", "run", os.path.join(_HERE, "compression_analysis_ui_web.py")])


@cli.command("run_local_ui")
def run_local_ui():
    """Desktop (Qt) UI launched from a local terminal."""
    subprocess.run(["python", os.path.join(_HERE, "compression_analysis_ui_local.py")])


@cli.command("help")
@click.pass_context
def help(ctx):
    """Print the help of every command."""
    for command in cli.commands.values():
        if command.name == "help":
            continue
        click.echo("-" * 80)
        click.echo()
        with click.Context(command, parent=ctx.parent, info_name=command.name) as sub:
            click.echo(command.get_help(ctx=sub))
        click.echo()


if __name__ == "__main__":
    cli()
