"""Helpers behind the dc_toolkit commands; cli.py passes them its parsed click parameters as one `opts`
namespace, which sweep_setup, _sweep_variable_body and single_process_setup extend.  A codec combination
is identified by its pipeline dict (utils.pipeline_to_dict): result rows, manifests and `compress` use it.

Sections
  1. Process, files & CLI plumbing
  2. Write threads & memory guards
  3. Gates & thresholds
  4. Pipelines & persistence
  5. Sweep                      (evaluate_combos)
  6. Compress                   (compress)
  7. Format conversion          (from_nc_to_zarr, from_zarr_to_netcdf)
  8. Analysis & plotting
  9. UI support                 (the streamlit and Qt front-ends)
"""
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import click
import dask
import dask.array
import numpy as np
import pandas as pd
import psutil
import xarray as xr
import zarr
from mpi4py import MPI

from dc_toolkit import utils


# =============================================================================
# 1. PROCESS, FILES & CLI PLUMBING
# =============================================================================

abort = utils.abort
hsize = utils.hsize


def opts(ctx) -> SimpleNamespace:
    """The command's click parameters as a namespace; a copy, so helpers may add attributes."""
    return SimpleNamespace(**ctx.params)


def size_option_callback(ctx, param, value):
    if value is None:
        return None
    try:
        return utils.parse_size(value)
    except Exception as e:
        raise click.BadParameter(f"Invalid size '{value}': {e}")


def add_options(options):
    """Apply a shared option group (reversed, so --help lists it in order)."""
    def decorator(f):
        for opt in reversed(options):
            f = opt(f)
        return f
    return decorator


def require_single_process(command: str) -> None:
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        if comm.Get_rank() == 0:
            click.echo(f"{command} is not meant to run in parallel.  Launch it with a single process.")
        comm.Abort(1)


def env_versions() -> dict:
    """Versions of the libraries behind the codecs and the sample, plus EBCC's tuning env vars; recorded in
    the manifests and in sweep_state_{var}.json, where any change voids the recorded rows."""
    env = {"zarr": getattr(zarr, "__version__", None),
           "numpy": getattr(np, "__version__", None),
           "dask": getattr(dask, "__version__", None)}
    for pkg in ("numcodecs", "zfpy", "pcodec"):
        try:
            env[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            env[pkg] = None
    if utils.EBCC_AVAILABLE:
        try:
            env["ebcc"] = importlib.metadata.version("ebcc")
        except importlib.metadata.PackageNotFoundError:
            env["ebcc"] = "unknown"
        env["ebcc_env"] = {v: os.environ[v] for v in utils.EBCC_ENV_VARS if v in os.environ}
    return env


def version_banner(command: str) -> str:
    env = env_versions()
    return f"[env] {command} | zarr={env['zarr']} | numpy={env['numpy']} | dask={env['dask']}"


def warn_env_drift(sweep_env: dict, source: str) -> None:
    """Warn when the sweep's recorded environment differs from this one: codec output may differ."""
    now = env_versions()
    deltas = [(k, then, now.get(k)) for k, then in sweep_env.items() if then is not None and then != now.get(k)]
    if deltas:
        click.echo(f"[manifest] WARNING: the environment differs from the sweep that wrote {source}:")
        for key, then, cur in deltas:
            click.echo(f"  {key}: sweep={then}  now={cur}")


def merged_store_path(where_to_write: str, dataset_file: str) -> str:
    """One {dataset_stem}.zarr store per dataset; fields are arrays inside it."""
    return str(Path(where_to_write) / f"{Path(dataset_file).stem}.zarr")


def close_store(store) -> None:
    close = getattr(store, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def existing_arrays(merged_path: str) -> set:
    """Array names in the merged store; empty when it is missing or cannot be opened."""
    if not Path(merged_path).is_dir():
        return set()
    try:
        group, store = utils.open_zarr_localstore(merged_path, read_only=True)
        names = set(group.array_keys())
        close_store(store)
        return names
    except Exception:
        return set()


def array_is_stock(merged_path: str, var: str) -> bool:
    """True unless the array's zarr.json names a codec that needs dc_toolkit to be read (the inner codecs
    of a sharded array included); True also when there is no readable zarr.json."""
    try:
        text = (Path(merged_path) / var / "zarr.json").read_text()
    except OSError:
        return True
    return not any(f'"{name}"' in text for name in utils.ENTRY_POINT_CODECS)


def staging_path(merged_path: str) -> Path:
    """A sibling store a field is written into until its gates pass: no half-written field enters the merged store."""
    return Path(f"{merged_path}.__staging__")


def remove_staged(merged_path: str, var=None) -> None:
    """Drop the staging copy of `var`, or the whole staging store."""
    path = staging_path(merged_path) if var is None else staging_path(merged_path) / var
    shutil.rmtree(path, ignore_errors=True)


def promote_staged(merged_path: str, var: str) -> None:
    """Rename the verified field into the merged store (a zarr v3 array does not record its own name),
    replacing an array of the same name, then remove the staging store."""
    target, aside = Path(merged_path) / var, staging_path(merged_path) / f"{var}.__replaced__"
    if target.exists():
        drop_consolidated_metadata(merged_path)  # it would describe the replaced array until the next consolidation
        os.replace(target, aside)  # two renames, so the old array is whole until the new one is in place
    os.replace(staging_path(merged_path) / var, target)
    remove_staged(merged_path)


def consolidate_store(merged_path: str) -> list:
    """Drop a killed run's staging leftovers and rewrite the consolidated metadata; returns the array names."""
    leftovers = sorted(p.name for p in staging_path(merged_path).iterdir()) if staging_path(merged_path).is_dir() else []
    if leftovers:
        click.echo(f"[store] discarding the unfinished write(s) of an interrupted run: {', '.join(leftovers)}")
        remove_staged(merged_path)
    store = zarr.storage.LocalStore(merged_path, read_only=False)
    try:
        return sorted(zarr.consolidate_metadata(store).array_keys())
    finally:
        close_store(store)


def drop_consolidated_metadata(merged_path: str) -> bool:
    """Remove the store's consolidated metadata (True if there was any) so readers scan the arrays instead
    of trusting a listing this run made stale."""
    root = Path(merged_path) / "zarr.json"
    if not root.is_file():
        return False
    try:
        meta = json.loads(root.read_text())
        if "consolidated_metadata" not in meta:
            return False
        meta.pop("consolidated_metadata")
        tmp = root.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, root)  # atomic: readers may have the store open
        return True
    except Exception as e:
        click.echo(f"[store] WARNING: could not drop the consolidated metadata of {merged_path}: {e}")
        return False


def read_json(path, label: str):
    """Parse a JSON file; None if missing, None with a warning if unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        click.echo(f"[{label}] WARNING: could not parse {path.name}: {e}")
        return None


def write_json(path, payload: dict, label: str) -> None:
    try:
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        click.echo(f"[{label}] wrote manifest -> {path}")
    except Exception as e:
        click.echo(f"[{label}] WARNING: could not write {path}: {e}")


def json_errors(errors) -> dict:
    """Error metrics as plain Python numbers for a manifest."""
    return {k: (None if v is None else int(v) if isinstance(v, (int, np.integer)) else float(v))
            for k, v in (errors or {}).items()}


# =============================================================================
# 2. WRITE THREADS & MEMORY GUARDS
# =============================================================================

def check_thread_count(threads: int) -> None:
    cores = utils.detect_cores_available()
    if int(threads) > cores:
        click.echo(f"[oversubscription] --threads {threads} exceeds the visible cores ({cores}).")
        abort(1)


def single_process_setup(opts) -> None:
    """Set opts.threads (--threads, else the visible cores); abort when it exceeds the visible cores or,
    with --oversubscription-check, when a thread env var is not 1."""
    reset_memcheck_state()
    if opts.threads is None:
        opts.threads = utils.detect_cores_available()
    check_thread_count(opts.threads)
    utils.check_thread_oversubscription(abort_if_unsafe=opts.oversubscription_check)


# Per-rank working set in samples: decoded buffer (1x) + encoded MemoryStore (up to 1x), and a filter's copy
# while encoding: ~2x at a rank's peak.  A node's ranks do not peak together (node average ~1.1x), so 2.0
# leaves headroom.
PER_RANK_WORKING_FACTOR = 2.0


def node_steady_estimate_bytes(sample_bytes: int, ranks_on_node: int, inner_chunk_mib: int) -> int:
    """One shared sample + ranks x PER_RANK_WORKING_FACTOR x sample + ranks x 2 x chunk (float64 temporaries)."""
    ranks = max(1, int(ranks_on_node))
    return int(sample_bytes + ranks * sample_bytes * PER_RANK_WORKING_FACTOR
               + ranks * 2 * max(1, int(inner_chunk_mib)) * 2**20)


def max_sample_bytes_for_ranks(budget_bytes: int, ranks_on_node: int, inner_chunk_mib: int) -> int:
    """Inverse of node_steady_estimate_bytes; 0 if nothing fits."""
    ranks = max(1, int(ranks_on_node))
    available = budget_bytes - ranks * 2 * max(1, int(inner_chunk_mib)) * 2**20
    return 0 if available <= 0 else int(available / (1.0 + ranks * PER_RANK_WORKING_FACTOR))


def _cgroup_v2_memory_paths():
    """The namespaced root (a container's limit), then this process's cgroup and its ancestors: under Slurm
    the root is absent and the task's own cgroup reads "max", while the limit sits on an ancestor."""
    yield "/sys/fs/cgroup/memory.max"
    try:
        with open("/proc/self/cgroup") as fh:
            rel = next((line.split(":", 2)[2].strip() for line in fh if line.startswith("0::")), "")
    except OSError:
        return
    parts = [p for p in rel.split("/") if p]
    while parts:
        yield "/sys/fs/cgroup/" + "/".join(parts) + "/memory.max"
        parts.pop()


def detect_node_memory_budget() -> tuple[int, str]:
    """(bytes, source) of the memory the node's ranks share: cgroup v2 limit, else cgroup v1, else host RAM.
    The cgroup is what OOM-kills a Slurm task; psutil cannot see it."""
    for path in _cgroup_v2_memory_paths():
        try:
            with open(path) as fh:
                val = fh.read().strip()
        except OSError:
            continue
        try:
            if val and val != "max":
                return int(val), f"cgroup v2 ({path})"
        except ValueError:
            continue
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


def check_node_memory_headroom(node_steady_bytes: int, ranks_on_node: int, rank: int,
                               label: str, threshold: float = 0.80) -> None:
    """Rank 0 aborts the job if `node_steady_bytes` exceeds `threshold` of the node budget; others return."""
    if rank != 0:
        return
    available, source = detect_node_memory_budget()
    if node_steady_bytes > threshold * available:
        click.echo(
            f"[memcheck] REFUSING to start sweep: memory requirement {hsize(node_steady_bytes)} per node "
            f"(one shared sample + {ranks_on_node} rank(s) x working set) exceeds {int(threshold*100)}% of the "
            f"detected budget {hsize(available)} ({source}).\n"
            f"  Context: {label}\n"
            f"  Fixes: start fewer ranks per node, lower --eval-data-size-limit, request more RAM "
            f"(#SBATCH --mem=0), or raise --memory-threshold (max 0.95).")
        abort(1)


_MEMCHECK_WARNED_HIGH = False


def reset_memcheck_state() -> None:
    global _MEMCHECK_WARNED_HIGH
    _MEMCHECK_WARNED_HIGH = False


def check_memory_headroom(required_bytes: int, label: str, threshold: float = 0.80) -> None:
    """Abort if `required_bytes` exceeds `threshold` of the available host RAM (psutil: blind to cgroups)."""
    global _MEMCHECK_WARNED_HIGH
    if threshold > 0.80 and not _MEMCHECK_WARNED_HIGH:
        if MPI.COMM_WORLD.Get_rank() == 0:
            click.echo(f"[memcheck] WARNING: threshold {threshold:.2f} exceeds the recommended 0.80; "
                       f"less memory is left for what the estimates do not count.")
        _MEMCHECK_WARNED_HIGH = True
    try:
        avail = psutil.virtual_memory().available
    except Exception as e:
        click.echo(f"[memcheck] WARNING: could not query available memory ({e}); skipping guard for {label}.")
        return
    if required_bytes > threshold * avail:
        click.echo(f"[memcheck] REFUSING to proceed: {label} needs {hsize(required_bytes)}, which exceeds "
                   f"{int(threshold*100)}% of currently-available RAM ({hsize(avail)}).\n"
                   f"  Reduce the relevant flag (--eval-data-size-limit for a sweep; --threads or --shard-mib "
                   f"for a write), raise --memory-threshold (max 0.95), or run on a larger node.")
        abort(1)


# =============================================================================
# 3. GATES & THRESHOLDS
# =============================================================================
# Thresholds are RELATIVE errors; the L2, Linf, bias and q99 ones default to multiples of --l1-threshold;
# off means +inf.

_L2_MULT_DEFAULT = 2.0     # RMS may run ~2x the mean-abs budget
_LINF_MULT_DEFAULT = 10.0
_BIAS_MULT_DEFAULT = 0.5   # at most half the budget may be one-directional
_Q99_MULT_DEFAULT = 2.0
GATE_KEYS = ("l1", "l2", "linf", "bias", "q99")


def derive_thresholds(opts) -> dict:
    def pick(value, mult, enabled):
        if not enabled:
            return math.inf
        return float(value) if value is not None else mult * opts.l1_threshold
    return {"l1": float(opts.l1_threshold),
            "l2": pick(opts.l2_threshold, _L2_MULT_DEFAULT, opts.l2_gate),
            "linf": pick(opts.linf_threshold, _LINF_MULT_DEFAULT, opts.linf_gate),
            "bias": pick(opts.bias_threshold, _BIAS_MULT_DEFAULT, opts.bias_gate),
            "q99": pick(opts.q99_threshold, _Q99_MULT_DEFAULT, opts.extremes_sensitive)}


def evaluate_gates(errors: dict, thr: dict, *, phys_min=None, phys_max=None, phys_slack=0.0,
                   grad_threshold=None, grad_gate=False):
    """(keep, {"pass_<gate>": bool}) for one metrics dict, for the sweep and the verify gate.  A None metric
    or +inf limit passes; `phys_slack` is the absolute excursion the bounds allow."""
    reasons = {f"pass_{t}": utils.within_limit(errors.get(m), thr.get(t)) for m, t in utils.CHEAP_GATES}
    reasons["pass_q99"] = utils.within_limit(errors.get("Q99_Rel"), thr.get("q99"))
    reasons["pass_finite"] = int(errors.get("N_Corrupt") or 0) == 0
    dmin, dmax = errors.get("Decoded_Min"), errors.get("Decoded_Max")
    bounds = True
    slack = float(phys_slack or 0.0)
    if phys_min is not None and dmin is not None and math.isfinite(dmin):
        bounds = bounds and dmin >= phys_min - slack
    if phys_max is not None and dmax is not None and math.isfinite(dmax):
        bounds = bounds and dmax <= phys_max + slack
    reasons["pass_bounds"] = bool(bounds)
    reasons["pass_grad"] = utils.within_limit(errors.get("Grad_Rel"), grad_threshold) if grad_gate else True
    return all(reasons.values()), reasons


def evaluate_cr_drift(production_ratio, predicted_ratio, tol):
    """(ok, drift, ok/under/over/skip): an error gate cannot see a ratio far below the sweep's."""
    if (predicted_ratio is None or production_ratio is None or not math.isfinite(predicted_ratio)
            or predicted_ratio <= 0 or not math.isfinite(production_ratio)):
        return True, None, "skip"
    drift = (production_ratio - predicted_ratio) / predicted_ratio
    if abs(drift) <= tol:
        return True, drift, "ok"
    return False, drift, ("under" if drift < 0 else "over")


def q99_cut(sample_np: np.ndarray):
    """The q99 gate's tail cut: the 99th percentile of |finite values|, or of the non-zero ones when
    that is 0.  A field that is 0 almost everywhere (cloud ice, hail) would otherwise put every cell in
    the tail, and its extremes are its largest non-zero values.  None when no value is finite."""
    finite = np.abs(sample_np[np.isfinite(sample_np)])
    if not finite.size:
        return None
    cut = float(np.quantile(finite, 0.99))
    if cut == 0:
        nonzero = finite[finite > 0]
        cut = float(np.quantile(nonzero, 0.99)) if nonzero.size else 0.0
    return cut


def fmt3(x) -> str:
    return f"{x:.3e}" if isinstance(x, float) else "n/a"


def verify_against_manifest(var: str, errors: dict, manifest, overrides=None):
    """Production verify gate: `errors` against the sweep's recorded thresholds, each overridden by a
    non-None `overrides` entry.  Returns (status, detail), status "pass", "fail" or "no-thresholds"."""
    man_thr = dict((manifest or {}).get("effective_thresholds", {}) or {})
    man_thr.update({k: v for k, v in (overrides or {}).items() if v is not None})
    thr = {k: (float(man_thr[k]) if man_thr.get(k) is not None else math.inf) for k in GATE_KEYS}
    if not any(math.isfinite(v) for v in thr.values()):
        return "no-thresholds", ""
    keep, reasons = evaluate_gates(errors, thr, phys_min=(manifest or {}).get("phys_min"),
                                   phys_max=(manifest or {}).get("phys_max"),
                                   phys_slack=(manifest or {}).get("phys_slack") or 0.0)
    if keep:
        return "pass", ""
    failed = ", ".join(k for k, ok in reasons.items() if not ok)
    detail = (f"{var}: verify gate FAILED ({failed}) | "
              f"L1={fmt3(errors.get('Relative_Error_L1'))} L2={fmt3(errors.get('Relative_Error_L2'))} "
              f"Linf={fmt3(errors.get('Relative_Error_Linf'))} bias={fmt3(errors.get('Bias_Rel'))} "
              f"q99={fmt3(errors.get('Q99_Rel'))} n_corrupt={errors.get('N_Corrupt', 0)} | thresholds: "
              + " ".join(f"{k}={thr[k]:.3e}" for k in GATE_KEYS))
    return "fail", detail


def verify_gate_verdict(var: str, out: dict, manifest, opts):
    """Run the verify gate and print the verdict.  Returns (status, detail), status "skipped" (--no-verify),
    "no-thresholds", "pass", "fail-advisory" (--no-verify-gate) or "fail"."""
    if not opts.verify or out["errors"] is None:
        return "skipped", ""
    overrides = {k: getattr(opts, f"{k}_threshold", None) for k in ("l1", "l2", "linf", "bias")}
    status, detail = verify_against_manifest(var, out["errors"], manifest, overrides)
    if status == "no-thresholds":
        click.echo(f"[verify-gate] {var}: no thresholds (none in manifest_{var}.json and no --l1-threshold ...); "
                   f"verification advisory only.")
    elif status == "pass":
        click.echo(f"[verify-gate] {var}: PASS, production error norms are within the sweep thresholds.")
    elif opts.verify_gate:
        click.echo(f"[verify-gate] FAIL: {detail}")
    else:
        status = "fail-advisory"
        click.echo(f"[verify-gate] FAIL: {detail} (advisory: --no-verify-gate set)")
    return status, detail


# =============================================================================
# 4. PIPELINES & PERSISTENCE
# =============================================================================

SPACE_KEYS = ("compressor_class", "filter_class", "serializer_class", "with_lossy", "with_ebcc")


def space_args(opts) -> dict:
    """The `space_args` dict codec_spaces takes."""
    return {k: getattr(opts, k) for k in SPACE_KEYS}


def codec_spaces(sample_da, space_args: dict, fso_range, chunk_shapes=None):
    """(compressors, filters, serializers) for the sample.  `fso_range` is the FULL field's (min, max):
    FixedScaleOffset and EBCC must not be parameterised from a sample that may miss the extremes."""
    try:
        return (utils.compressor_space(sample_da, space_args["with_lossy"], space_args["compressor_class"]),
                utils.filter_space(sample_da, space_args["with_lossy"], space_args["filter_class"],
                                   data_range=fso_range),
                utils.serializer_space(sample_da, space_args["with_lossy"], space_args["serializer_class"],
                                       with_ebcc=space_args["with_ebcc"], data_range=fso_range,
                                       chunk_shapes=chunk_shapes))
    except (ValueError, TypeError) as e:
        raise click.ClickException(str(e))


def parse_pipeline_arg(text: str) -> dict:
    """--pipeline: a JSON object, a file holding one (`@path` or a bare path), or a manifest's best.pipeline."""
    if not text.strip():
        raise click.ClickException("--pipeline is empty: give a JSON object or the path of a file holding one")
    try:
        if text.startswith("@"):
            raw = Path(text[1:]).read_text()
        else:
            raw = text if text.lstrip()[:1] in "{[" else Path(text).read_text()
        d = json.loads(raw)
    except Exception as e:
        raise click.ClickException(f"--pipeline must be a JSON object or the path of a file holding one: {e}")
    if not isinstance(d, dict):
        raise click.ClickException("--pipeline must be a JSON object with compressor, filter and serializer")
    if not any(k in d for k in ("compressor", "filter", "serializer")):
        best = d.get("best")
        best = best.get("pipeline") if isinstance(best, dict) else None
        if not isinstance(best, dict):
            raise click.ClickException("--pipeline needs a JSON object with compressor, filter and "
                                       "serializer, or a manifest_{var}.json holding best.pipeline")
        return best
    return d


def pipeline_codecs(pipeline: dict, context: str):
    """utils.pipeline_from_dict, raising a ClickException that names `context`."""
    try:
        return utils.pipeline_from_dict(pipeline)
    except Exception as e:
        raise click.ClickException(f"cannot rebuild the pipeline for {context}: {e}")


def validate_pipeline(combo, da, var: str) -> None:
    """Raise a ClickException for codecs that cannot pair, an AsType filter that would decode to another
    dtype, or an EBCC input the C library would exit on (wrong dtype, tile not dividing the frame, NaN/Inf)."""
    compressor, filt, serializer = combo
    if not utils.combo_is_valid(filt, serializer, compressor, dtype=da.dtype):
        raise click.ClickException(
            f"{var}: invalid pipeline {utils.pipeline_name(*combo)} (e.g. FixedScaleOffset->ZFPY, "
            f"BitRound->ZFPY below the mantissa width, "
            f"or EBCC with a compressor or a filter other than AsType).")
    if isinstance(filt, utils.zarrcodecs_nc.AsType):  # numcodecs reinterprets the bytes on a mismatch
        decode = filt.codec_config.get("decode_dtype")
        if decode is not None and np.dtype(decode) != da.dtype:
            raise click.ClickException(f"{var}: the AsType filter's decode_dtype {decode} is not the field's "
                                       f"dtype {da.dtype}.")
    if isinstance(serializer, utils.EBCC):
        astype = isinstance(filt, utils.zarrcodecs_nc.AsType)
        input_dtype = np.dtype(filt.codec_config.get("encode_dtype", da.dtype)) if astype else da.dtype
        if input_dtype != np.float32:
            raise click.ClickException(f"{var}: EBCC needs float32 input, got {input_dtype}"
                                       + (" from the AsType filter." if astype else
                                          "; the field needs the AsType (float32) filter in the pipeline."))
        if da.ndim < 2 or da.shape[-2] % serializer.height or da.shape[-1] % serializer.width:
            raise click.ClickException(f"{var}: the EBCC tile {serializer.height}x{serializer.width} does not "
                                       f"divide the field's frame {tuple(da.shape[-2:])}.")
        if not bool(dask.array.asarray(da.data).map_blocks(np.isfinite).all().compute()):
            raise click.ClickException(f"{var}: the field contains NaN/Inf, which EBCC cannot encode.")


def chunk_geometry(opts, manifest):
    """(geometry, sources): inner_chunk_mib, max_inner_chunk_mib and spatial_split from the CLI flag, else
    the sweep manifest's args (so the store matches what the sweep measured), else the default."""
    args = (manifest or {}).get("args") or {}
    defaults = {"inner_chunk_mib": 16, "max_inner_chunk_mib": 256, "spatial_split": True}
    geometry, sources = {}, {}
    for key, default in defaults.items():
        cli_value = getattr(opts, key)
        if cli_value is not None:
            geometry[key], sources[key] = cli_value, "cli"
        elif key in args:
            geometry[key], sources[key] = args[key], "manifest"
        else:
            geometry[key], sources[key] = default, "default"
    return geometry, sources


def persist_field(da, var: str, merged_path: str, combo, opts, geometry: dict, q99_abs) -> dict:
    """Write one field with `combo` into the staging store, creating the merged store if needed; aborts when
    the write peak does not fit in memory.  Returns ratio, errors, eucd, geometry and timing."""
    serializer = combo[2]
    forced_chunks = utils.ebcc_chunks(serializer, da.shape) if isinstance(serializer, utils.EBCC) else None
    inner_chunks, shards = utils.compute_chunk_and_shard_shape(
        da.shape, da.dtype, inner_mib=geometry["inner_chunk_mib"], shard_mib=opts.shard_mib,
        dims=tuple(da.dims), allow_spatial_split=geometry["spatial_split"], inner_chunks=forced_chunks)
    itemsize = int(da.dtype.itemsize)
    inner_bytes = itemsize * int(np.prod(inner_chunks))
    shard_bytes = itemsize * int(np.prod(shards)) if shards is not None else inner_bytes
    if not geometry["spatial_split"] and inner_bytes > geometry["max_inner_chunk_mib"] * 2**20:
        click.echo(f"[chunks] WARNING: --no-spatial-split for '{var}' produced an inner chunk of "
                   f"{hsize(inner_bytes)} (shape {inner_chunks}), above --max-inner-chunk-mib "
                   f"({geometry['max_inner_chunk_mib']} MiB).  Codec internals may misbehave at this size.")
    layout = (f"inner chunks={inner_chunks}, {hsize(inner_bytes)}; "
              + ("sharding skipped (a shard would hold < 2 chunks)" if shards is None
                 else f"shards={shards}, {hsize(shard_bytes)}"))
    click.echo(f"[persist] {var} -> {merged_path} ({layout})")

    # Per write task: source block, write unit, zarr's encode copy and encoded bytes; a small field peaks near 3x.
    source_block = itemsize * int(np.prod(da.data.chunksize))
    write_peak = min(int(opts.threads) * (max(source_block, shard_bytes) + 3 * shard_bytes), 3 * int(da.nbytes))
    check_memory_headroom(write_peak, threshold=opts.memory_threshold,
                          label=f"write peak for '{var}' (min(threads x (max(source block, write unit) + "
                                f"3 x write unit), 3 x field) = {hsize(write_peak)})")

    os.makedirs(Path(merged_path).parent, exist_ok=True)
    try:
        close_store(zarr.storage.LocalStore(merged_path, read_only=False))
        zarr.open_group(merged_path, mode="a", zarr_format=3)  # surface a corrupt/unwritable store early
    except Exception as e:
        click.echo(f"[persist] ERROR: cannot open or create zarr group at {merged_path}: {e}")
        raise
    store = zarr.storage.LocalStore(str(staging_path(merged_path)), read_only=False)
    t0 = time.perf_counter()
    try:
        ratio, errors, eucd = utils.persist_with_codec_pipeline(
            da, store, component=var, codec_kwargs=utils.codec_pipeline_kwargs(*combo),
            inner_chunks=inner_chunks, shards=shards, verify=opts.verify, q99_abs=q99_abs)
    finally:
        close_store(store)
    return {"ratio": float(ratio), "errors": errors, "eucd": eucd,
            "inner_chunks": list(inner_chunks), "inner_chunk_bytes": int(inner_bytes),
            "shards": (list(shards) if shards is not None else None),
            "shard_bytes": (int(shard_bytes) if shards is not None else None),
            "sharding_skipped": shards is None, "seconds": time.perf_counter() - t0}


# =============================================================================
# 5. SWEEP (evaluate_combos)
# =============================================================================

@dataclass
class SweepContext:
    comm: object
    rank: int
    size: int
    ranks_on_node: int
    cores_avail: int
    thresholds: dict
    node_comm: object
    local_rank: int
    leaders: object      # one rank per node; COMM_NULL on the others
    node_id: int
    n_nodes: int


def sweep_setup(opts) -> SweepContext:
    """Collective.  Sets opts.with_ebcc, installs an excepthook that ends a multi-rank job through MPI Abort,
    pins zarr's pool to one worker (one core per rank) and creates the output directory."""
    reset_memcheck_state()
    opts.with_ebcc = opts.with_ebcc or opts.serializer_class.lower() == "ebcc"
    if opts.with_ebcc and not opts.with_lossy:
        raise click.ClickException("EBCC is lossy: --with-ebcc / --serializer-class ebcc need --with-lossy.")
    if opts.with_ebcc and not utils.EBCC_AVAILABLE:
        raise click.ClickException("--with-ebcc needs the ebcc package: pip install -e '.[ebcc]'")
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size > 1:  # an uncaught exception on one rank would leave the others waiting in a collective
        def abort_on_error(exc_type, exc, tb):
            traceback.print_exception(exc_type, exc, tb)
            sys.stderr.flush()
            comm.Abort(1)
        sys.excepthook = abort_on_error
    node_comm, ranks_on_node, local_rank = utils.detect_node_topology(comm)
    leaders = comm.Split(0 if local_rank == 0 else MPI.UNDEFINED, key=rank)
    node_id = node_comm.bcast(leaders.Get_rank() if local_rank == 0 else None, root=0)
    n_nodes = comm.allreduce(1 if local_rank == 0 else 0)
    cores_avail = utils.detect_cores_available()
    cores = utils.detect_physical_cores() if size == 1 else 1
    if cores > 1:
        click.echo(f"[topology] NOTE: one rank on {cores} cores.  A rank evaluates one pipeline at a time; "
                   f"start one rank per core to use them all (mpirun -n {cores} dc_toolkit ...).")
    utils.check_thread_oversubscription(abort_if_unsafe=opts.oversubscription_check, rank=rank, comm=comm)
    zarr.config.set({"threading.max_workers": 1})
    if rank == 0:
        os.makedirs(opts.where_to_write, exist_ok=True)
    comm.Barrier()

    thr = derive_thresholds(opts)
    if rank == 0:
        click.echo(version_banner("evaluate_combos"))
        fmt = lambda x: "off" if not math.isfinite(x) else f"{x:.3e}"  # noqa: E731
        grad = (f"on@{opts.gradient_threshold:.3e} (shortcircuit={'on' if opts.gradient_shortcircuit else 'OFF'})"
                if opts.gradient_gate else "off")
        click.echo(f"[gates] thresholds (relative): L1={thr['l1']:.3e} L2={fmt(thr['l2'])} "
                   f"Linf={fmt(thr['linf'])} bias={fmt(thr['bias'])} q99={fmt(thr['q99'])} | "
                   f"bounds=[{opts.phys_min}, {opts.phys_max}]"
                   f"{f' +-{opts.phys_tolerance:g} of range' if getattr(opts, 'phys_tolerance', 0) else ''} | gradient={grad}")
    return SweepContext(comm, rank, size, ranks_on_node, cores_avail, thr, node_comm, local_rank, leaders,
                        node_id, n_nodes)


def sweep_variables(ds, field, rank: int) -> list:
    """The named field, else every non-empty integer/float32/float64 array with a dim, except CF bounds
    (named by a `bounds` or `climatology` attribute): grid geometry, which a lossy codec would move."""
    bounds = {str(src[a]) for v in ds.variables.values() for src in (v.attrs, v.encoding)
              for a in ("bounds", "climatology") if a in src}
    names = [v for v in ds.data_vars if field in (None, v)]
    geometry = [v for v in names if v in bounds and field is None]
    if rank == 0 and (geometry or field in bounds):
        click.echo(f"[var] skipping grid geometry (CF bounds): {', '.join(geometry)}" if geometry else
                   f"[var] {field} is CF bounds (grid geometry); sweeping it because it was named.")
    names = [v for v in names if v not in geometry]
    usable = [v for v in names if ds[v].ndim > 0 and ds[v].size > 0
              and (ds[v].dtype.kind in "iu" or ds[v].dtype in (np.float32, np.float64))]
    skipped = [f"{v} ({ds[v].dtype}, {ds[v].ndim}-d)" for v in names if v not in usable]
    if skipped and field is not None:
        raise click.ClickException(f"cannot sweep {skipped[0]}: only non-empty integer/float32/float64 arrays.")
    if skipped and rank == 0:
        click.echo(f"[var] skipping variable(s) the codecs cannot take: {', '.join(skipped)}")
    return usable


def sweep_sample_limit(var: str, da, opts, sweep: SweepContext) -> int:
    """Collective.  The sample budget: --eval-data-size-limit, raised to the minimum sample and shrunk so
    one shared sample beside the node's rank working sets fits the node memory budget; then the memory
    guards.  Returns the limit, the same on every rank."""
    ranks, chunk_mib = sweep.ranks_on_node, opts.inner_chunk_mib
    field_bytes, floor_bytes = int(da.nbytes), utils.minimum_sample_bytes(da)
    node_budget, source = detect_node_memory_budget()
    max_safe = max_sample_bytes_for_ranks(int(node_budget * opts.memory_threshold), ranks, chunk_mib)
    wanted = max(int(opts.eval_data_size_limit), floor_bytes)
    # Rank 0 sizes the sample for everyone, so every rank adopts the smallest node's limit.
    limit = sweep.comm.allreduce(min(wanted, max_safe), op=MPI.MIN)
    if limit < floor_bytes:
        if sweep.rank == 0:
            click.echo(f"[memcheck] FATAL: the smallest sample of '{var}' ({hsize(floor_bytes)}: "
                       f"{utils.SAMPLE_MIN_KEEP} time steps x {utils.SAMPLE_MIN_KEEP} levels, or all where there are "
                       f"fewer) does not fit with {ranks} rank(s) per node, inner_chunk_mib={chunk_mib} and a node "
                       f"memory budget of {hsize(node_budget)} (from {source}) at threshold "
                       f"{opts.memory_threshold:.2f}.  Start fewer ranks per node or request more RAM.")
        abort(1)
    if sweep.rank == 0 and floor_bytes > int(opts.eval_data_size_limit) and field_bytes > int(opts.eval_data_size_limit):
        click.echo(f"[sample] raised the sample budget of '{var}' from {hsize(opts.eval_data_size_limit)} "
                   f"(--eval-data-size-limit) to {hsize(floor_bytes)} to keep {utils.SAMPLE_MIN_KEEP} time steps "
                   f"and {utils.SAMPLE_MIN_KEEP} levels.")
    elif sweep.rank == 0 and limit < int(opts.eval_data_size_limit):
        click.echo(f"[memcheck] auto-shrunk sample budget from {hsize(opts.eval_data_size_limit)} "
                   f"(--eval-data-size-limit) to {hsize(limit)} to stay under {opts.memory_threshold:.2f} x "
                   f"{hsize(node_budget)} per node (from {source}) with {ranks} rank(s) per node.")

    sample_bytes = min(field_bytes, limit)
    multiplier = 2 if sweep.rank == 0 else 1  # rank 0 holds its build beside the window it fills
    check_memory_headroom(multiplier * sample_bytes, threshold=opts.memory_threshold,
                          label=f"sample for '{var}' on rank {sweep.rank} ({hsize(sample_bytes)})")
    check_node_memory_headroom(node_steady_estimate_bytes(sample_bytes, ranks, chunk_mib), ranks_on_node=ranks,
                               rank=sweep.rank, label=f"variable '{var}', sample {hsize(sample_bytes)}",
                               threshold=opts.memory_threshold)
    return limit


def sweep_build_sample(da, limit: int, opts, sweep: SweepContext):
    """Collective.  Rank 0 builds the sample, each node maps one shared copy (shared_sample_window) and all
    ranks split the full-field range pass; the dim coords travel along so every rank classifies the dims
    alike.  Returns (sample_np, sample_da, fso_range, win); the caller frees win, collectively."""
    comm, rank = sweep.comm, sweep.rank
    if rank == 0:
        try:
            local = utils.build_representative_sample(da, limit, rank=rank, policy=opts.sampling_policy,
                                                      vertical_floor=opts.vertical_floor).compute()
        except utils.SampleTooLargeError:
            comm.Abort(1)  # the other ranks are waiting in a collective
        sample_np_local = np.ascontiguousarray(local.values)
        meta = {"dims": tuple(local.dims), "attrs": dict(local.attrs), "name": local.name,
                "shape": tuple(sample_np_local.shape), "dtype": str(sample_np_local.dtype),
                "coords": {d: (np.asarray(local.coords[d].values), dict(local.coords[d].attrs),
                               dict(local.coords[d].encoding)) for d in local.dims if d in local.coords}}
    else:
        sample_np_local = meta = None
    fso_range = utils.full_field_data_range(da, comm=comm)
    meta = comm.bcast(meta, root=0)
    sample_np, win = shared_sample_window(sample_np_local, meta["shape"], meta["dtype"], sweep)
    del sample_np_local
    coords = {d: xr.Variable((d,), v, attrs=a, encoding=e) for d, (v, a, e) in meta["coords"].items()}
    sample_da = xr.DataArray(sample_np, dims=meta["dims"], coords=coords, attrs=meta["attrs"], name=meta["name"])
    return sample_np, sample_da, fso_range, win


def shared_sample_window(local_np, shape, dtype, sweep: SweepContext):
    """Collective.  One copy of the sample per node in an MPI-3 shared-memory window, filled from rank 0's
    `local_np` through the node leaders and mapped read-only; returns (sample_np, win).  A checksum across
    all ranks catches a rank reading the window before its fill, which would otherwise give wrong metrics."""
    dtype, count = np.dtype(dtype), int(np.prod(shape))
    try:
        win = MPI.Win.Allocate_shared(count * dtype.itemsize if sweep.local_rank == 0 else 0, dtype.itemsize,
                                      comm=sweep.node_comm)
    except MPI.Exception as e:
        click.echo(f"[shared-sample] FATAL on rank {sweep.rank}: MPI.Win.Allocate_shared failed ({e}).  The shared "
                   f"memory may be smaller than the sample ({hsize(count * dtype.itemsize)}; in a container raise "
                   f"docker run --shm-size), or the MPI has no shared-memory communicator.")
        sweep.comm.Abort(1)
    buf, _ = win.Shared_query(0)
    sample_np = np.frombuffer(buf, dtype=dtype, count=count).reshape(shape)
    if sweep.local_rank == 0:
        if sweep.rank == 0:
            sample_np[...] = local_np
        sweep.leaders.Bcast(sample_np, root=0)
    sweep.node_comm.Barrier()
    sample_np.flags.writeable = False
    probe = np.ascontiguousarray(sample_np.reshape(-1)[::4099])
    h = zlib.crc32(probe.view(np.uint8))  # over the bytes: exact whatever NaN/Inf the sample holds
    if sweep.comm.allreduce(h, MPI.MIN) != sweep.comm.allreduce(h, MPI.MAX):
        if sweep.rank == 0:
            click.echo("[shared-sample] FATAL: the sample differs between ranks after the fill.")
        sweep.comm.Abort(1)
    return sample_np, win


_ROW_KEY_COLUMNS = ["pipeline", "ratio", "l1_rel", "l2_rel", "linf_rel", "eucd"]  # a whole row has all of them


def rank_files(where_to_write, prefix: str, var: str) -> list:
    """The per-rank files of one variable, matched exactly: a glob on "{prefix}_{var}_rank*" would also
    match variable "{var}_rank0" and break on a name holding glob characters."""
    pattern = re.compile(rf"^{re.escape(prefix)}_{re.escape(str(var))}_rank\d+\.csv$")
    return sorted(p for p in Path(where_to_write).iterdir() if pattern.match(p.name))


def read_rank_csvs(where_to_write, var: str, quarantine: bool = False) -> pd.DataFrame:
    """Every readable config_space_{var}_rank*.csv as one frame, minus rows lacking a key metric.  An
    unparseable file is skipped, and with `quarantine` renamed to *.unreadable so its rank starts afresh."""
    frames = []
    for path in rank_files(where_to_write, "config_space", var):
        try:
            frames.append(pd.read_csv(path, on_bad_lines="skip").dropna(subset=_ROW_KEY_COLUMNS))
        except Exception as e:
            click.echo(f"[sweep] WARNING: cannot read {path.name} ({e}); "
                       + ("set aside as *.unreadable." if quarantine else "skipped."))
            if quarantine:
                path.rename(path.with_name(path.name + ".unreadable"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PARTIAL_CSV_COLUMNS)


def sweep_recorded_rows(var, sample_np, q99_abs, fso_range, opts, sweep: SweepContext) -> set:
    """Collective: the pipelines whose recorded rows this run reuses, decided on rank 0 before any rank opens
    its CSV.  Rows are reused only with --resume, when sweep_state_{var}.json matches this run's dataset,
    sample, full-field range, sampling and chunk settings and library versions (else, and with --no-resume,
    the per-rank CSVs are removed), and when they carry every metric the current gates need.  Failed combos
    are always retried."""
    done = set()
    if sweep.rank == 0:
        where = Path(opts.where_to_write)
        state = json.loads(json.dumps({
            "dataset_file": os.path.abspath(opts.dataset_file), "sample_shape": list(sample_np.shape),
            "dtype": str(sample_np.dtype), "full_field_range": fso_range,
            "sampling_policy": opts.sampling_policy, "vertical_floor": opts.vertical_floor,
            "inner_chunk_mib": opts.inner_chunk_mib, "spatial_split": opts.spatial_split,
            "sample_digest": hashlib.blake2b(memoryview(sample_np).cast("B"), digest_size=16).hexdigest(),
            "metrics": utils.METRIC_DEFINITIONS, "env": env_versions()}, default=str))
        state_path = where / f"sweep_state_{var}.json"
        previous = read_json(state_path, "resume")
        rank_csvs = rank_files(where, "config_space", var)
        changed = (["sweep_state file (missing)"] if previous is None and rank_csvs
                   else sorted(k for k in state if previous is not None and previous.get(k) != state[k]))
        if opts.resume and changed and rank_csvs:
            click.echo(f"[resume] {var}: the recorded rows were measured with another {', '.join(changed)}; "
                       f"starting this field from scratch.")
        for path in (rank_csvs if changed or not opts.resume else []) + rank_files(where, "failures", var):
            path.unlink()
        state_path.write_text(json.dumps(state, indent=2))
        if opts.resume and not changed:
            for path in rank_csvs:
                drop_partial_last_line(path)
            prev = read_rank_csvs(where, var, quarantine=True)
            done = set(prev.loc[reusable_rows(prev, q99_abs, opts, sweep.thresholds), "pipeline"])
    return sweep.comm.bcast(done, root=0)


def reusable_rows(prev: pd.DataFrame, q99_abs, opts, thresholds: dict) -> pd.Series:
    """Rows carrying every metric the current gates need: q99 and gradient are recorded only with their gate
    on (the gradient, with the shortcircuit, only for rows passing the cheap gates).  A field with no finite
    value has no q99 cut, so no row records it; requiring it would re-evaluate the field for ever."""
    ok = pd.Series(True, index=prev.index)
    if opts.extremes_sensitive and q99_abs is not None and math.isfinite(q99_abs):
        ok &= prev["q99_rel"].notna()
    if opts.gradient_gate:
        needed = pd.Series(True, index=prev.index)
        if opts.gradient_shortcircuit:
            for column, key in (("l1_rel", "l1"), ("l2_rel", "l2"), ("linf_rel", "linf"), ("bias_rel", "bias")):
                if math.isfinite(thresholds[key]):
                    needed &= ~(prev[column] > thresholds[key])
        ok &= prev["grad_rel"].notna() | ~needed
    return ok


def sweep_config_space(compressors, filters, serializers, max_evals, rank, dtype, all_finite: bool) -> list:
    """Triples to evaluate: the EBCC ones (none unless `all_finite`, never cut), then the valid non-EBCC
    product cut to --max-evals (a quick-test knob, not a sample) and shuffled so each node's every-n_nodes-th
    share mixes cheap and expensive codecs; the seed depends only on the count, so --resume keeps the order."""
    regular = [s for s in serializers if not isinstance(s, utils.EBCC)]
    total = len(compressors) * len(filters) * len(regular)
    config_space = [(c, f, s) for c, f, s in itertools.product(compressors, filters, regular)
                    if utils.combo_is_valid(f, s, c, dtype=dtype)]
    if rank == 0 and len(config_space) < total:
        click.echo(f"[combo-filter] skipped {total - len(config_space)} unsupported filter/serializer "
                   f"pairing(s) (FixedScaleOffset->ZFPY, BitRound->ZFPY below the mantissa width).")
    if max_evals is not None and max_evals < len(config_space):
        if rank == 0:
            click.echo(f"[max-evals] capping config space at {max_evals} (of {len(config_space)} possible).")
        config_space = config_space[:max_evals]
    entries, reason = utils.ebcc_sweep_entries(filters, serializers, dtype, all_finite)
    if reason and rank == 0:
        click.echo(f"[ebcc] skipping EBCC combos: {reason}.")
    # EBCC first: its combos are the slowest, and a node that claims one last runs long after the others
    perm = np.random.default_rng(seed=(len(config_space) + len(entries)) & 0xFFFFFFFF).permutation(len(config_space))
    return entries + [config_space[i] for i in perm]


def config_space_table(config_space) -> pd.DataFrame:
    """config_space_{var}.csv: labels and pipeline JSON of every planned combo."""
    return pd.DataFrame([{"name": utils.pipeline_name(*cfg), "compressor": utils.codec_label(cfg[0]),
                          "filter": utils.codec_label(cfg[1]), "serializer": utils.codec_label(cfg[2]),
                          "pipeline": utils.pipeline_json(*cfg)} for cfg in config_space])


def sweep_banner(var, spaces, config_space, n_pending, sample_np, opts, sweep: SweepContext, n_vars: int):
    compressors, filters, serializers = spaces
    ranks, num_loops = sweep.ranks_on_node, len(config_space)
    trailer = (f" (only {n_pending} will run concurrently; {n_pending} of {num_loops} combos to evaluate)"
               if 0 < n_pending < sweep.size else "")
    click.echo(f"[topology] {sweep.n_nodes} node(s) x {ranks} rank(s)/node = {sweep.size} parallel evaluations, "
               f"one shared sample per node{trailer}.")
    steady = node_steady_estimate_bytes(sample_np.nbytes, ranks, opts.inner_chunk_mib)
    click.echo(f"[memory] rank-0 transient peak ~= {int(2 * sample_np.nbytes / 2**20)} MiB (building the sample); "
               f"per-node steady ~= {int(sample_np.nbytes / 2**20)} MiB (shared sample) + "
               f"~{int(ranks * PER_RANK_WORKING_FACTOR * sample_np.nbytes / 2**20)} MiB "
               f"({ranks} ranks x {PER_RANK_WORKING_FACTOR:.1f}x decode/encode cache) + "
               f"~{ranks * max(1, opts.inner_chunk_mib) * 2} MiB (ranks x 2 x inner_chunk_mib) "
               f"= {hsize(steady)} total.")
    if n_vars > 1:
        click.echo(f"[topology] sweep will iterate {n_vars} variables; one sample Bcast per "
                   f"variable (~{hsize(sample_np.nbytes)} to each node over the interconnect).")
    n_ebcc = sum(isinstance(cfg[2], utils.EBCC) for cfg in config_space)
    n_regular = sum(not isinstance(s, utils.EBCC) for s in serializers)
    cap = f", --max-evals {opts.max_evals}" if opts.max_evals is not None else ""
    click.echo(f"[sweep] {num_loops} combos: {num_loops - n_ebcc} from the {len(compressors)} x {len(filters)} x "
               f"{n_regular} grid (valid pairings{cap}) + {n_ebcc} EBCC; {n_pending} to evaluate, "
               f"~{-(-n_pending // sweep.n_nodes)} per node, claimed by its ranks as they free up.")


def sweep_evaluators(var, sample_np, sample_da, q99_abs, opts, sweep: SweepContext):
    """(evaluate_one(cfg) -> result dict, gate(errors) -> (keep, reasons)) for one variable."""
    grad_axes = utils.horizontal_axes(sample_da) or (tuple(range(1, sample_np.ndim)) if sample_np.ndim > 1 else (0,))
    eval_chunks = utils.compute_chunk_shape_for_eval(sample_np.shape, sample_np.dtype, target_mib=opts.inner_chunk_mib,
                                                     dims=sample_da.dims, allow_spatial_split=opts.spatial_split)
    eval_chunk_bytes = int(sample_np.dtype.itemsize) * int(np.prod(eval_chunks))
    if not opts.spatial_split and eval_chunk_bytes > opts.max_inner_chunk_mib * 2**20 and sweep.rank == 0:
        click.echo(f"[chunks] WARNING: --no-spatial-split produced an eval chunk of {hsize(eval_chunk_bytes)} "
                   f"(shape {eval_chunks}), above --max-inner-chunk-mib ({opts.max_inner_chunk_mib} MiB).")

    def evaluate_one(cfg):
        compressor, filt, serializer = cfg
        chunks = utils.ebcc_chunks(serializer, sample_np.shape) if isinstance(serializer, utils.EBCC) else eval_chunks
        ratio, errors, eucd = utils.evaluate_codec_pipeline(
            sample_np, sample_da.dims, utils.codec_pipeline_kwargs(*cfg),
            chunks=chunks, q99_abs=q99_abs, compute_gradient=opts.gradient_gate,
            gradient_axes=grad_axes if opts.gradient_gate else None,
            precheck_thresholds=sweep.thresholds if (opts.gradient_gate and opts.gradient_shortcircuit) else None)
        return {"name": utils.pipeline_name(*cfg), "compressor": utils.codec_label(compressor),
                "filter": utils.codec_label(filt), "serializer": utils.codec_label(serializer),
                "pipeline": utils.pipeline_json(*cfg),
                "ratio": float(ratio), "errors": errors, "eucd": float(eucd)}

    def gate(errors):
        return evaluate_gates(errors, sweep.thresholds, phys_min=opts.phys_min, phys_max=opts.phys_max,
                              phys_slack=getattr(opts, "phys_slack", 0.0),
                              grad_threshold=opts.gradient_threshold, grad_gate=opts.gradient_gate)

    return evaluate_one, gate


def drop_partial_last_line(path) -> None:
    """Cut a half-written last row (no newline) of a killed run, so appended rows start on a fresh line."""
    with open(path, "rb+") as fh:
        data = fh.read()
        if data and not data.endswith(b"\n"):
            fh.truncate(data.rfind(b"\n") + 1)


PARTIAL_CSV_COLUMNS = [
    "name", "compressor", "filter", "serializer", "pipeline",
    "ratio", "l1_rel", "l2_rel", "linf_rel", "bias_rel", "q99_rel", "grad_rel",
    "decoded_min", "decoded_max", "n_corrupt", "eucd",
    "pass_l1", "pass_l2", "pass_linf", "pass_bias", "pass_q99", "pass_bounds", "pass_grad", "pass_finite",
    "keep",
]


def node_counter(sweep: SweepContext):
    """Collective over the node.  Returns (claim, win): claim() hands out 0, 1, 2, ... across the node's
    ranks, each once, by an atomic fetch-and-add in shared memory that waits on no other process.  win is
    inside a Lock_all epoch: the caller calls win.Unlock_all(), then win.Free() collectively."""
    # Without this hint MPICH routes the atomics through the counter's owner, which answers only on its next
    # MPI call: a claim would wait until the owner finished its combo.
    info = MPI.Info.Create()
    info.Set("disable_shm_accumulate", "false")
    win = MPI.Win.Allocate_shared(8 if sweep.local_rank == 0 else 0, 8, info=info, comm=sweep.node_comm)
    info.Free()
    if sweep.local_rank == 0:
        win.Lock(0, MPI.LOCK_EXCLUSIVE)
        win.Put(np.zeros(1, np.int64), 0)
        win.Unlock(0)
    sweep.node_comm.Barrier()
    one, got = np.ones(1, np.int64), np.empty(1, np.int64)
    win.Lock_all()

    def claim() -> int:
        win.Fetch_and_op(one, got, 0, op=MPI.SUM)
        win.Flush(0)
        return int(got[0])
    return claim, win


def sweep_run_rank(config_space, pending, var, opts, sweep: SweepContext, evaluate_one, gate) -> list:
    """Collective over the node.  Evaluate this node's share of `pending` (every n_nodes-th, from node_id),
    one combo per claim, streaming rows and failures to config_space_/failures_{var}_rank{rank}.csv with
    this run's verdicts (sweep_select_best re-gates).  Returns this rank's (name, pipeline JSON, error) failures."""
    rank = sweep.rank
    partial_path = Path(opts.where_to_write) / f"config_space_{var}_rank{rank}.csv"
    failures_path = Path(opts.where_to_write) / f"failures_{var}_rank{rank}.csv"
    mode = "a" if partial_path.is_file() else "w"
    share = pending[sweep.node_id::sweep.n_nodes]
    report_every, next_report = max(1, len(share) // 10), 0
    claim, win = node_counter(sweep)

    failures = []
    # line-buffered: a walltime kill loses at most the combo in flight
    with open(partial_path, mode, newline="", buffering=1) as pf, \
            open(failures_path, "w", newline="", buffering=1) as ff:
        pw, fw = csv.writer(pf), csv.writer(ff)
        if pf.tell() == 0:
            pw.writerow(PARTIAL_CSV_COLUMNS)
        fw.writerow(["name", "pipeline", "error"])
        while True:
            i = claim()
            if i >= len(share):
                break
            if rank == 0 and i >= next_report:  # rank 0 sees the claims of its whole node
                utils.progress_bar(i, len(share), "node 0")
                next_report = i + report_every
            cfg = config_space[share[i]]
            try:
                r = evaluate_one(cfg)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:  # one broken combo never stops the sweep; a Rust panic is a BaseException
                failures.append((utils.pipeline_name(*cfg), utils.pipeline_json(*cfg), repr(e)))
                fw.writerow(failures[-1])
                continue
            err = r["errors"]
            keep, reasons = gate(err)
            pw.writerow([
                r["name"], r["compressor"], r["filter"], r["serializer"], r["pipeline"],
                r["ratio"], err["Relative_Error_L1"], err["Relative_Error_L2"], err["Relative_Error_Linf"],
                err.get("Bias_Rel"), err.get("Q99_Rel"), err.get("Grad_Rel"),
                err.get("Decoded_Min"), err.get("Decoded_Max"), err.get("N_Corrupt", 0), r["eucd"],
                reasons["pass_l1"], reasons["pass_l2"], reasons["pass_linf"], reasons["pass_bias"],
                reasons["pass_q99"], reasons["pass_bounds"], reasons["pass_grad"], reasons["pass_finite"],
                keep,
            ])
    win.Unlock_all()
    win.Free()
    return failures


def sweep_report_failures(failures, var, sweep: SweepContext) -> int:
    """Collective.  Rank 0 prints a few failures per rank and returns the total; None elsewhere."""
    gathered = sweep.comm.gather(failures[:5], root=0)
    total = sweep.comm.reduce(len(failures), op=MPI.SUM, root=0)
    if sweep.rank == 0 and total:
        click.echo(f"[warning] {total} combo(s) failed total across {sweep.size} rank(s).")
        shown = 0
        for r_idx, batch in enumerate(gathered):
            for name, _pipeline, err in batch:
                if shown >= 30:
                    break
                click.echo(f"  [rank {r_idx}] {name}: {err}")
                shown += 1
        if total > shown:
            click.echo(f"  ... and {total - shown} more (full details in failures_{var}_rank*.csv).")
    return total


_METRIC_COLUMNS = {
    "l1_rel": "Relative_Error_L1", "l2_rel": "Relative_Error_L2", "linf_rel": "Relative_Error_Linf",
    "bias_rel": "Bias_Rel", "q99_rel": "Q99_Rel", "grad_rel": "Grad_Rel",
    "decoded_min": "Decoded_Min", "decoded_max": "Decoded_Max", "n_corrupt": "N_Corrupt",
}


def regate(df: pd.DataFrame, gate) -> pd.DataFrame:
    """Recompute pass_* and keep with the current gates (resumed rows may carry other thresholds' verdicts)."""
    verdict_columns = [c for c in PARTIAL_CSV_COLUMNS if c.startswith("pass_") or c == "keep"]
    verdicts = []
    for row in df[list(_METRIC_COLUMNS)].itertuples(index=False):
        errors = {key: (None if pd.isna(v) else v) for key, v in zip(_METRIC_COLUMNS.values(), row)}
        keep, reasons = gate(errors)
        verdicts.append({**reasons, "keep": keep})
    df = pd.concat([df.drop(columns=verdict_columns, errors="ignore").reset_index(drop=True),
                    pd.DataFrame(verdicts, columns=verdict_columns)], axis=1)
    return df[PARTIAL_CSV_COLUMNS]


def sweep_select_best(where_to_write, var, gate, planned: set):
    """Consolidate the per-rank CSVs into results_{var}.parquet, keeping the `planned` pipelines, re-gated,
    and pick the kept combo with the best ratio (ties: lower L1, then pipeline) FROM DISK, so --resume of a
    finished field still finds it.  Returns (best {name, pipeline, ratio, l1_rel, eucd} or None, n_passed, path)."""
    consolidated = read_rank_csvs(where_to_write, var)
    # A combo re-evaluated for a metric its row lacked has several rows; last() merges their non-null values.
    consolidated = consolidated.groupby("pipeline", as_index=False, sort=False).last()
    stale = ~consolidated["pipeline"].isin(planned)
    if stale.any():
        click.echo(f"[sweep] ignoring {int(stale.sum())} recorded row(s) outside this sweep's codec space.")
        consolidated = consolidated[~stale]
    consolidated = regate(consolidated, gate)
    parquet_path = os.path.join(where_to_write, f"results_{var}.parquet")
    consolidated.to_parquet(parquet_path, index=False)
    click.echo(f"[sweep] consolidated the per-rank CSVs -> {parquet_path} ({len(consolidated)} row(s)).")
    kept = kept_rows(consolidated)
    if len(kept) == 0:
        return None, 0, parquet_path
    top = kept.sort_values(["ratio", "l1_rel", "pipeline"], ascending=[False, True, True]).iloc[0]
    best = {"name": str(top["name"]), "pipeline": json.loads(top["pipeline"]),
            "ratio": float(top["ratio"]), "l1_rel": float(top["l1_rel"]), "eucd": float(top["eucd"])}
    return best, int(len(kept)), parquet_path


SWEEP_ARG_KEYS = (
    "eval_data_size_limit", "inner_chunk_mib", "max_inner_chunk_mib",
    "spatial_split", "compressor_class", "filter_class", "serializer_class", "with_lossy", "with_ebcc",
    "sampling_policy", "vertical_floor", "l1_threshold", "l2_threshold", "linf_threshold", "bias_threshold",
    "q99_threshold", "l2_gate", "linf_gate", "bias_gate", "extremes_sensitive", "phys_min", "phys_max",
    "phys_tolerance",
    "gradient_gate", "gradient_threshold", "resume", "max_evals",
)


def sweep_manifest(var, opts, sweep: SweepContext, *, num_combos, n_passed, total_failures, seconds,
                   parquet_path, best, q99_abs) -> dict:
    return {
        "command": "evaluate_combos",
        "dataset_file": os.fspath(opts.dataset_file), "var": str(var), "where_to_write": os.fspath(opts.where_to_write),
        "args": {k: getattr(opts, k) for k in SWEEP_ARG_KEYS},
        "topology": {"size": int(sweep.size), "cores_avail": int(sweep.cores_avail),
                     "ranks_on_node": int(sweep.ranks_on_node)},
        "effective_thresholds": {k: (None if not math.isfinite(v) else float(v)) for k, v in sweep.thresholds.items()},
        "gradient_threshold": float(opts.gradient_threshold) if opts.gradient_gate else None,
        "phys_min": opts.phys_min, "phys_max": opts.phys_max,
        "phys_slack": float(getattr(opts, "phys_slack", 0.0)),
        "q99_abs": q99_abs,
        "num_combos": int(num_combos), "num_passed": int(n_passed),
        "num_failed_total": int(total_failures or 0),
        "num_filtered": int(num_combos - n_passed - (total_failures or 0)),
        "var_sweep_seconds": float(seconds),
        "env": env_versions(),
        "outputs": {"parquet": parquet_path,
                    "config_space_csv": os.fspath(Path(opts.where_to_write) / f"config_space_{var}.csv")},
        "best": best,
    }


def sweep_variable(da, var: str, opts, sweep: SweepContext, n_vars: int) -> None:
    """Collective: sweep one field; rank 0 writes its results, winner and manifest."""
    rank = sweep.rank
    t0 = time.perf_counter()
    if rank == 0:
        click.echo(f"[var] {var} | units={da.attrs.get('units', 'N/A')} | "
                   f"relative L1 threshold={sweep.thresholds['l1']:.3e}")
    limit = sweep_sample_limit(var, da, opts, sweep)
    sample_np, sample_da, fso_range, win = sweep_build_sample(da, limit, opts, sweep)
    _sweep_variable_body(da, var, opts, sweep, n_vars, t0, sample_np, sample_da, fso_range)
    # Free is collective: after an error the exception must reach the abort hook rather than wait here.
    del sample_np, sample_da
    win.Free()


def _sweep_variable_body(da, var, opts, sweep: SweepContext, n_vars, t0, sample_np, sample_da, fso_range):
    # Collective, like sweep_variable: every rank runs it, and an early return must be identical on every rank;
    # sets opts.phys_slack.
    rank = sweep.rank
    span = float(fso_range[1] - fso_range[0]) if fso_range else 0.0
    opts.phys_slack = float(getattr(opts, "phys_tolerance", 0.0) or 0.0) * span   # absolute; the manifest carries it
    if opts.phys_slack and rank == 0:
        click.echo(f"[gates] {var}: bounds slack {opts.phys_slack:g} ({opts.phys_tolerance:g} of the range {span:g})")
    # q99_cut holds ~3 sample-sized temporaries: rank 0 computes it, the others receive the float
    q99_abs = sweep.comm.bcast(q99_cut(sample_np) if rank == 0 else None, root=0) if opts.extremes_sensitive else None
    if opts.extremes_sensitive and rank == 0:
        click.echo(f"[gates] {var}: q99(|value|)={q99_abs} (extreme-tail cut for the q99 gate)")
    # A sample without variation says nothing about the codecs: every pipeline reproduces it exactly and
    # the ratio ranks overheads.  Only a field that is constant throughout is stored without a search.
    sample_range = sweep.comm.bcast(utils.finite_range(sample_np) if rank == 0 else None, root=0)
    constant = sample_range is not None and sample_range[0] == sample_range[1]
    try:
        if sample_range is None or (constant and fso_range is not None):
            what = "has no finite value" if sample_range is None else f"holds the single value {sample_range[0]:g}"
            raise click.ClickException(
                f"the sample {what}" + (f" while the field spans [{fso_range[0]:g}, {fso_range[1]:g}]; raise "
                                        f"--eval-data-size-limit so that the sample reaches that variation"
                                        if fso_range is not None else ", and so has the field: nothing to compress"))
        if constant:
            if rank == 0:
                click.echo(f"[sample] {var}: the field is constant ({sample_range[0]:g}); the search is skipped "
                           f"and Zstd stores it losslessly.")
            spaces = ([utils.zarrcodecs_nc.Zstd(level=6)], [None], [None])
        else:
            chunks = [utils.compute_chunk_shape_for_eval(shape, sample_np.dtype, target_mib=opts.inner_chunk_mib,
                                                          dims=sample_da.dims, allow_spatial_split=opts.spatial_split)
                      for shape in (sample_np.shape, da.shape)]
            spaces = codec_spaces(sample_da, space_args(opts), fso_range, chunk_shapes=chunks)
    except click.ClickException as e:  # decided from the shared sample and range: the same on all ranks
        if opts.field_to_compress is not None:
            raise
        if rank == 0:
            click.echo(f"[var] skipping {var}: {e.message}")
        return
    done = sweep_recorded_rows(var, sample_np, q99_abs, fso_range, opts, sweep)
    if opts.with_ebcc and rank == 0:
        tile, reason = utils.ebcc_tile(sample_da)
        click.echo(f"[ebcc] {var}: " + (f"tile {tile[0]}x{tile[1]}" if tile else f"not applicable ({reason})"))
    all_finite = True
    if any(isinstance(s, utils.EBCC) for s in spaces[2]):
        all_finite = sweep.comm.bcast(bool(np.isfinite(sample_np).all()) if rank == 0 else None, root=0)
    config_space = sweep_config_space(*spaces, opts.max_evals, rank, sample_np.dtype, all_finite)
    keys = [utils.pipeline_json(*cfg) for cfg in config_space]
    pending = [i for i, key in enumerate(keys) if key not in done]
    if rank == 0:
        if not config_space:
            click.echo(f"[sweep] {var}: the codec space is empty for these classes and this dtype.")
        if done:
            click.echo(f"[resume] {sum(k in done for k in keys)} of {len(keys)} combo(s) of '{var}' are already "
                       f"recorded; skipping those.")
        sweep_banner(var, spaces, config_space, len(pending), sample_np, opts, sweep, n_vars)
        config_space_table(config_space).to_csv(os.path.join(opts.where_to_write, f"config_space_{var}.csv"),
                                                index=False)

    evaluate_one, gate = sweep_evaluators(var, sample_np, sample_da, q99_abs, opts, sweep)
    failures = sweep_run_rank(config_space, pending, var, opts, sweep, evaluate_one, gate)
    total_failures = sweep_report_failures(failures, var, sweep)
    sweep.comm.Barrier()  # every rank's CSV is complete before rank 0 consolidates
    if rank != 0:
        return

    click.echo("[sweep] complete. Writing results...")
    best, n_passed, parquet_path = sweep_select_best(opts.where_to_write, var, gate, set(keys))
    if best is not None:
        click.echo(f"best pipeline: {best['name']}\nCompression Ratio: {best['ratio']:.3f} | "
                   f"Relative L1 Error: {best['l1_rel']:.3e} | Euclidean Distance: {best['eucd']:.3e}")
    else:
        click.echo("[sweep] no combos passed the threshold filter.")
    manifest = sweep_manifest(var, opts, sweep, num_combos=len(config_space), n_passed=n_passed,
                              total_failures=total_failures, seconds=time.perf_counter() - t0,
                              parquet_path=parquet_path, best=best, q99_abs=q99_abs)
    write_json(os.path.join(opts.where_to_write, f"manifest_{var}.json"), manifest, "sweep")


# =============================================================================
# 6. COMPRESS (compress)
# =============================================================================

def compress_candidates(opts):
    """(candidates, manifests, dropped).  One candidate {var, name, pipeline, ratio, source} per field: with
    --pipeline, that pipeline for every --vars field; else the manifest best, falling back to the best kept
    row of results_{var}.parquet (with --stock-codecs-only, the best a bare zarr client can decode).
    `manifests` holds every readable manifest; `dropped` maps each field without a usable pipeline to why."""
    wtw = Path(opts.where_to_write)
    wanted = {v.strip() for v in opts.vars_filter.split(",") if v.strip()} if opts.vars_filter else None
    manifests, sweep_env = {}, None
    for mpath in sorted(wtw.glob("manifest_*.json")):
        m = read_json(mpath, "compress")
        if m is not None:
            manifests[mpath.stem.removeprefix("manifest_")] = m
            sweep_env = sweep_env or m.get("env")
    if sweep_env:
        warn_env_drift(sweep_env, "these manifests")

    candidates, dropped = [], {}
    if opts.pipeline is not None:
        if not wanted:
            raise click.ClickException("--pipeline needs --vars to name the field(s) it applies to.")
        pipeline = parse_pipeline_arg(opts.pipeline)
        if getattr(opts, "stock_codecs_only", False) and not utils.pipeline_is_stock(pipeline):
            raise click.ClickException("--stock-codecs-only refuses this --pipeline: one of its codecs needs "
                                       "dc_toolkit's zarr.codecs entry point to be read.")
        name = utils.pipeline_name(*pipeline_codecs(pipeline, "--pipeline"))
        candidates = [{"var": v, "name": name, "pipeline": pipeline, "ratio": None, "source": "--pipeline"}
                      for v in sorted(wanted)]
    else:
        stock, deferred = bool(getattr(opts, "stock_codecs_only", False)), {}
        for var, m in manifests.items():
            if wanted and var not in wanted:
                continue
            best = m.get("best")
            if best is None:
                dropped[var] = "the sweep kept no combo (manifest has no best)"
            elif not isinstance(best, dict) or not isinstance(best.get("pipeline"), dict):
                dropped[var] = "the manifest best has no pipeline; re-run evaluate_combos"
            elif not stock or utils.pipeline_is_stock(best["pipeline"]):
                candidates.append({"var": var, "name": best.get("name", "?"), "pipeline": best["pipeline"],
                                   "ratio": best.get("ratio"), "source": f"manifest_{var}.json"})
            else:
                deferred[var] = best.get("name", "?")
                click.echo(f"[compress] {var}: the manifest best {best.get('name', '?')} needs dc_toolkit's codec entry "
                           f"point to be read; --stock-codecs-only takes the best stock row of the parquet.")
        for ppath in sorted(wtw.glob("results_*.parquet")):
            var = ppath.stem.removeprefix("results_")
            if (wanted and var not in wanted) or var in dropped or any(c["var"] == var for c in candidates):
                continue
            try:
                row = best_kept_row(pd.read_parquet(ppath), stock_only=stock)
                if row is None:
                    dropped[var] = f"no kept rows in {ppath.name}" + (" with stock codecs only" if stock else "")
                    continue
                candidates.append({"var": var, "name": str(row["name"]), "pipeline": json.loads(row["pipeline"]),
                                   "ratio": float(row["ratio"]),
                                   "source": ppath.name + (" (stock codecs only)" if stock else "")})
            except Exception as e:
                dropped[var] = f"cannot use {ppath.name}: {e}"
        for var, name in deferred.items():
            if var not in dropped and not any(c["var"] == var for c in candidates):
                dropped[var] = f"the manifest best {name} is not stock and results_{var}.parquet is missing"
        if wanted:
            candidates = [c for c in candidates if c["var"] in wanted]
            dropped = {v: dropped.get(v, "no manifest_{var}.json or results_{var}.parquet in WHERE_TO_WRITE")
                       for v in sorted(wanted - {c["var"] for c in candidates})}
    for var, reason in dropped.items():
        click.echo(f"[compress] ERROR: {var} has no usable pipeline: {reason}.")
    if candidates:
        click.echo(f"[compress] will compress {len(candidates)} field(s): {', '.join(c['var'] for c in candidates)}")
    return candidates, manifests, dropped


def compress_one(da, var: str, cand: dict, manifest, merged_path: str, opts) -> dict:
    """Persist one field and promote it into the merged store only if the verify and CR-drift gates pass
    (else RuntimeError; an existing array stays).  Returns the batch_manifest.json entry."""
    combo = pipeline_codecs(cand["pipeline"], var)
    validate_pipeline(combo, da, var)
    geometry, sources = chunk_geometry(opts, manifest)
    click.echo(f"[chunks] {var}: " + ", ".join(f"{k}={v} ({sources[k]})" for k, v in geometry.items()))
    q99_abs = (manifest or {}).get("q99_abs") if opts.verify else None
    try:
        out = persist_field(da, var, merged_path, combo, opts, geometry, q99_abs)
        summary = f"{var}: {cand['name']} -> ratio={out['ratio']:.3f}"
        if opts.verify:
            summary += f" L1_rel={out['errors']['Relative_Error_L1']:.3e} eucd={out['eucd']:.3e}"
        click.echo(f"[compress] {summary}  ({out['seconds']:.1f}s)")

        verify_status, detail = verify_gate_verdict(var, out, manifest, opts)
        if verify_status == "fail":
            raise RuntimeError(detail)

        predicted = cand.get("ratio")
        ok, drift, direction = evaluate_cr_drift(out["ratio"], predicted, opts.cr_drift_tol)
        if direction == "skip":
            click.echo(f"[cr-drift] {var}: no predicted ratio for this pipeline; skipped.")
        elif ok:
            click.echo(f"[cr-drift] {var}: PASS (achieved {out['ratio']:.2f}x vs predicted "
                       f"{predicted:.2f}x, drift {drift:+.1%})")
        else:
            msg = (f"{var}: CR drift {drift:+.1%} exceeds +/-{opts.cr_drift_tol:.0%} "
                   f"(achieved {out['ratio']:.2f}x vs predicted {predicted:.2f}x)")
            if direction == "under" and opts.cr_drift_gate:
                raise RuntimeError(f"cr-drift gate FAILED: {msg}")
            click.echo(f"[cr-drift] WARNING {msg}"
                       + ("" if direction == "under" else "  (better than predicted; informational)"))
        promote_staged(merged_path, var)
    except BaseException:  # also Ctrl-C and a memory guard's SystemExit
        remove_staged(merged_path, var)
        raise

    return {
        "status": "ok", "name": cand["name"], "pipeline": cand["pipeline"], "source": cand["source"],
        "verify_gate": verify_status, "cr_drift_status": direction,
        "ratio": out["ratio"],
        "predicted_ratio": (float(predicted) if predicted is not None else None),
        "cr_drift": (float(drift) if drift is not None else None),
        "errors": json_errors(out["errors"]),
        "eucd": (float(out["eucd"]) if out["eucd"] is not None else None),
        "inner_chunks": out["inner_chunks"], "shards": out["shards"], "sharding_skipped": out["sharding_skipped"],
        "seconds": float(out["seconds"]),
    }


# =============================================================================
# 7. FORMAT CONVERSION
# =============================================================================

def nc_to_zarr(opts) -> None:
    """NetCDF -> UNCOMPRESSED zarr v3 store (no filters, compressors, sharding), for filesystem-dedup experiments."""
    if Path(opts.nc_path).suffix.lower() != ".nc":
        raise click.ClickException(f"Expected a .nc file, got {opts.nc_path}.  "
                                   f"Use from_zarr_to_netcdf for the reverse direction.")
    out_zarr = opts.out_zarr or str(Path(opts.nc_path).with_suffix(".zarr"))
    if Path(out_zarr).exists():
        if not opts.overwrite:
            raise click.ClickException(f"Output already exists: {out_zarr}.  Pass --overwrite to replace, "
                                       f"or pick a different --out.")
        click.echo(f"[nc->zarr] removing existing {out_zarr} (--overwrite).")
        shutil.rmtree(out_zarr)
    threads = utils.detect_cores_available() if opts.threads is None else int(opts.threads)

    click.echo(f"[nc->zarr] reading {opts.nc_path} ...")
    with dask.config.set(scheduler="threads", num_workers=threads):
        ds = xr.open_dataset(opts.nc_path, chunks={} if opts.preserve_source_chunks else "auto",
                             mask_and_scale=opts.mask_and_scale, decode_times=opts.decode_times)
        click.echo(f"[nc->zarr] logical size = {hsize(int(ds.nbytes))} "
                   f"| chunks = {'source-native' if opts.preserve_source_chunks else 'auto'} "
                   f"| mask_and_scale = {opts.mask_and_scale} | decode_times = {opts.decode_times} "
                   f"| dask workers = {threads}")
        # explicit None: zarr would otherwise default to a Zstd compressor
        encoding = {}
        for name in ds.variables:
            ds[name].encoding = {}
            encoding[name] = {"compressors": None, "filters": None}
        click.echo(f"[nc->zarr] writing {out_zarr} (compressors=None, filters=None, {len(encoding)} variable(s)) ...")
        ds.to_zarr(out_zarr, mode="w-", encoding=encoding, zarr_format=3, consolidated=opts.consolidated)
    click.echo(f"[nc->zarr] wrote {out_zarr}")


def zarr_to_netcdf(opts) -> None:
    """zarr v3 store -> NetCDF4 file, streamed through dask."""
    out_nc = opts.out_nc or str(Path(opts.zarr_path).with_suffix(".nc"))
    threads = utils.detect_cores_available() if opts.threads is None else int(opts.threads)
    check_thread_count(threads)

    with dask.config.set(scheduler="threads", num_workers=threads):
        # consolidated=False: a listing a later --no-consolidate write made stale would drop or misdescribe fields
        ds = xr.open_zarr(opts.zarr_path, chunks="auto", consolidated=False)
        logical_bytes = int(ds.nbytes)
        click.echo(f"[zarr->nc] logical size = {hsize(logical_bytes)} | dask workers = {threads}")
        if logical_bytes > opts.max_size:
            raise click.ClickException(f"Refusing to write: logical size exceeds --max-size ({hsize(opts.max_size)}).  "
                                       f"Raise --max-size to proceed, or keep the data in .zarr.")
        encoding = {}
        for name, var in ds.data_vars.items():
            enc = {}
            if isinstance(var.data, dask.array.Array):
                enc["chunksizes"] = tuple(max(b) for b in var.data.chunks)
            if opts.compression == "zlib":
                enc.update(zlib=True, complevel=int(opts.complevel))
            encoding[name] = enc
        click.echo(f"[zarr->nc] writing {out_nc} ...")
        ds.to_netcdf(out_nc, engine="h5netcdf", encoding=encoding)
    click.echo(f"[zarr->nc] wrote {out_nc}")


# =============================================================================
# 8. ANALYSIS & PLOTTING
# =============================================================================
# matplotlib, plotly, sklearn and tqdm are imported inside these functions: the sweep and compress do not load them.

L_ERRORS = ("L1", "L2", "LInf")
_L_ERROR_COLUMNS = {"L1": "l1_rel", "L2": "l2_rel", "LInf": "linf_rel"}


def kept_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that passed every gate (`keep` round-trips through CSV and parquet as bool or string)."""
    if "keep" not in df.columns:
        return df
    return df[df["keep"].astype(str).str.strip().str.lower().isin(("true", "1"))]


def best_kept_row(df: pd.DataFrame, stock_only: bool = False):
    """Best-ratio kept row (ties: lower L1, then pipeline), or None; `stock_only`: stock codecs only."""
    kept = kept_rows(df)
    if stock_only:
        kept = kept[kept["pipeline"].map(lambda p: utils.pipeline_is_stock(json.loads(p)))]
    if len(kept) == 0:
        return None
    return kept.sort_values(["ratio", "l1_rel", "pipeline"], ascending=[False, True, True]).iloc[0]


def load_results(parquet_file: str) -> pd.DataFrame:
    """Kept rows of a results parquet with finite metrics, best ratio first."""
    df = kept_rows(pd.read_parquet(parquet_file))
    metrics = ["ratio"] + list(_L_ERROR_COLUMNS.values()) + ["eucd"]
    df = df[np.isfinite(df[metrics].astype(float)).all(axis=1)]
    return df.sort_values("ratio", ascending=False).reset_index(drop=True)


def elbow_silhouette_plot(df: pd.DataFrame, l_error: str) -> None:
    """KMeans (k = 3..9) on (ratio, l_error): elbow and silhouette curves."""
    from tqdm import tqdm
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt

    data = df[["ratio", _L_ERROR_COLUMNS[l_error]]].astype(float).to_numpy()
    k_values = range(3, min(9, len(df) - 1) + 1)
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


def clustering_figure(df: pd.DataFrame):
    """Interactive KMeans scatter plots of L1 / L2 / LInf vs ratio; hover shows the pipeline."""
    from sklearn.cluster import KMeans
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    max_n_rows, max_nclusters = 42976, 6
    n_clusters = max(1, min(len(df), math.ceil(max_nclusters * len(df) / max_n_rows)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    fig = make_subplots(rows=3, cols=1, subplot_titles=[f"{m} VS Ratio KMeans Clustering" for m in L_ERRORS])
    for row, metric in enumerate(L_ERRORS, start=1):
        points = pd.DataFrame({"Ratio": df["ratio"].astype(float).to_numpy(),
                               metric: df[_L_ERROR_COLUMNS[metric]].astype(float).to_numpy(),
                               "pipeline": df["name"].astype(str).to_numpy(),
                               "compressor": df["compressor"].astype(str).to_numpy(),
                               "filter": df["filter"].astype(str).to_numpy(),
                               "serializer": df["serializer"].astype(str).to_numpy()})
        labels = kmeans.fit_predict(points[["Ratio", metric]])
        color = np.ones(labels.shape) if len(np.unique(labels)) == 1 else labels
        scatter = px.scatter(points, x="Ratio", y=metric, color=color,
                             hover_data=["pipeline", "compressor", "filter", "serializer"])
        fig.add_trace(go.Scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
                                 mode="markers+text", marker=dict(color="black", size=12, symbol="x"),
                                 textposition="top center", name="Centroids", showlegend=(row == 1)),
                      row=row, col=1)
        fig.update_xaxes(title_text="Ratio", row=row, col=1)
        fig.update_yaxes(title_text=metric, row=row, col=1)
        for trace in scatter.data:
            fig.add_trace(trace, row=row, col=1)
    fig.update_layout(title="", showlegend=False, height=900, hovermode="closest", template="plotly_white")
    return fig


def plot_pipeline(field: str, pipeline_arg, manifest_dir: str):
    """(compressor, filt, serializer) to plot: --pipeline, else the best of manifest_{field}.json."""
    if pipeline_arg is not None:
        return pipeline_codecs(parse_pipeline_arg(pipeline_arg), "--pipeline")
    manifest = read_json(Path(manifest_dir) / f"manifest_{field}.json", "plot")
    best = (manifest or {}).get("best") or {}
    if "pipeline" not in best:
        raise click.ClickException(f"no pipeline for {field}: pass --pipeline or point --manifest-dir at a "
                                   f"directory holding manifest_{field}.json from evaluate_combos.")
    return pipeline_codecs(best["pipeline"], field)


def error_plot_panels(da, field: str, combo):
    """Round-trip a (lat, lon) field and its 180-degree-rolled copy through `combo`.  Returns (da, panels):
    `da` cast to float32 when the serializer is EBCC, and the nine (title, data, cmap) panels."""
    compressor, filt, serializer = combo
    chunks = "auto"
    if isinstance(serializer, utils.EBCC):  # validate_pipeline already checked dtype/tile/finite
        if da.dtype != np.float32:  # the cast replaces the AsType filter
            da, filt = da.astype("float32"), None
        chunks = utils.ebcc_chunks(serializer, da.shape)
    codec_kwargs = utils.codec_pipeline_kwargs(compressor, filt, serializer)
    lon_dim = da.dims[1]
    half = da.sizes[lon_dim] // 2
    shifted = da.roll({lon_dim: -half}, roll_coords=False)

    def roundtrip(arr):
        store = zarr.storage.MemoryStore()
        z = zarr.create_array(store=store, name=field, data=np.ascontiguousarray(arr.values),
                              chunks=chunks, **codec_kwargs)
        out = xr.DataArray(z[:], dims=da.dims, coords=da.coords)
        store.close()
        return out

    decoded = roundtrip(da)
    shifted_decoded = roundtrip(shifted)
    shifted_back = shifted.roll({lon_dim: half}, roll_coords=False)
    shifted_decoded_back = shifted_decoded.roll({lon_dim: half}, roll_coords=False)
    return da, [
        ("Original", da, None),
        ("Original compressed&decompressed", decoded, None),
        ("Absolute error [original - original c&d]", np.abs(da - decoded), "binary"),
        ("Shifted (by +180 deg)", shifted, None),
        ("Shifted compressed&decompressed", shifted_decoded, None),
        ("Absolute error [shifted - shifted c&d]", np.abs(shifted - shifted_decoded), "binary"),
        ("Relative error [original - (shifted-180)] / |original|", np.abs(da - shifted_back) / (np.abs(da) + 1e-20), "binary"),
        ("Absolute error [original c&d - (shifted c&d-180)]", np.abs(decoded - shifted_decoded_back), "binary"),
        ("Absolute error [original - (shifted c&d-180)]", np.abs(da - shifted_decoded_back), "binary"),
    ]


def save_error_plot(field: str, da, panels, path: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, layout="constrained", figsize=(16, 9))
    fig.suptitle(f"Compression errors for variable {field} ({da.attrs.get('units', 'N/A')})", fontsize=16)
    for ax, (title, data, cmap) in zip(axes.flatten(), panels):
        ax.set_title(title, fontsize=10)
        fig.colorbar(ax.imshow(data, interpolation="none", cmap=cmap), ax=ax, shrink=0.7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 9. UI SUPPORT
# =============================================================================
# Class names are those utils._select_classes accepts.

UI_CLASS_OPTIONS = {  # "astype" is left out: it exists for float64 fields only (EBCC brings its own cast)
    "compressor": ["all", "blosc", "lz4", "zstd", "zlib", "bz2", "lzma", "none"],
    "filter": ["all", "delta", "bitround", "quantize", "fixedscaleoffset", "none"],
    "serializer": ["all", "pcodec", "zfpy", "ebcc", "none"],
}
UI_DEFAULT_L1 = 0.005


def ui_launcher(user_account=None, time=None, nodes=None, ntasks_per_node=None, uenv_image=None,
                partition=None) -> list:
    """srun prefix when a vcluster account is given, else [].  Without uenv_image srun keeps the session's uenv."""
    if not user_account:
        return []
    cmd = ["srun", "-A", user_account, "--time", time or "00:15:00", "--nodes", nodes or "1",
           "--ntasks-per-node", ntasks_per_node or "1", "--partition=" + (partition or "debug")]
    return cmd + (["--uenv=" + uenv_image, "--view=default"] if uenv_image else [])


def ui_env() -> dict:
    """os.environ with the codec thread pools at 1 (the oversubscription check) and Open MPI allowed as root."""
    return {**os.environ, **{v: "1" for v in utils.THREAD_ENV_VARS},
            "OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"}


def ui_mpirun() -> list:
    """mpirun with one rank per physical core if a launcher is on PATH, else [] (one rank): a sweep's
    parallelism comes from its ranks, and Open MPI refuses more ranks than cores."""
    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    return [launcher, "-n", str(utils.detect_physical_cores())] if launcher else []


def ui_sweep_command(launcher, dataset, out_dir, field, classes: dict, with_lossy: bool, with_ebcc: bool,
                     l1_threshold: float) -> list:
    return (launcher or ui_mpirun()) + ["dc_toolkit", "evaluate_combos", dataset, "--where-to-write", out_dir,
                       "--field-to-compress", field, "--l1-threshold", str(l1_threshold),
                       "--compressor-class", classes["compressor"], "--filter-class", classes["filter"],
                       "--serializer-class", classes["serializer"],
                       "--with-lossy" if with_lossy else "--without-lossy",
                       "--with-ebcc" if with_ebcc else "--without-ebcc"]


def ui_compress_command(launcher, dataset, out_dir, field, pipeline: dict) -> list:
    """Rewrite the field (the user may pick several pipelines in a row) and
    fail loudly, so the UI never offers a stale store."""
    return launcher + ["dc_toolkit", "compress", dataset, out_dir, "--vars", field,
                       "--pipeline", json.dumps(pipeline), "--no-skip-existing", "--no-continue-on-error"]


def ui_results(out_dir: str, field: str) -> pd.DataFrame:
    """Kept rows of the field's sweep, best ratio first (empty if no sweep ran)."""
    path = Path(out_dir) / f"results_{field}.parquet"
    return load_results(str(path)) if path.is_file() else pd.DataFrame()


def zip_directory(path: str) -> str:
    """Zip a store directory next to itself; returns the archive path."""
    return shutil.make_archive(path, "zip", Path(path).parent, Path(path).name)
