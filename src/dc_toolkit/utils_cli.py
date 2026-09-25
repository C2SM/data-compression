"""Helpers behind the dc_toolkit commands; cli.py passes them its parsed click parameters as one `opts`
namespace, which sweep_setup, _sweep_variable_body and single_process_setup extend.  A codec combination
is identified by its pipeline dict (utils.pipeline_to_dict): result rows, manifests and `compress` use it.

Sections
  1. Process, files & CLI plumbing
  2. Write threads & memory guards
  3. Gates & thresholds
  4. Pipelines & persistence
  5. Sweep                      (evaluate_combos)
  6. Compress                   (compress, merge_compressed_fields)
  7. Store inspection & format conversion
                               (open_zarr_and_inspect, from_nc_to_zarr, from_zarr_to_netcdf)
  8. Analysis & plotting        (perform_clustering, analyze_clustering, plot_compression_errors)
  9. UI support                 (the streamlit and Qt front-ends)
"""
import ast
import csv
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
import traceback
import zlib
from dataclasses import dataclass
from functools import lru_cache
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

from dc_toolkit import codecs, utils


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


def finite_option_callback(ctx, param, value):
    """Refuse NaN and +-inf: a non-finite bound or budget silently rejects or passes every combo."""
    if value is not None and (value != value or abs(value) == float("inf")):
        raise click.BadParameter("must be finite")
    return value


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
            dist = importlib.metadata.distribution("ebcc")
            env["ebcc"] = dist.version
            env["ebcc_commit"] = json.loads(dist.read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id")
        except (importlib.metadata.PackageNotFoundError, ValueError):
            env["ebcc"] = "unknown"
        env["ebcc_env"] = {v: os.environ[v] for v in utils.EBCC_ENV_VARS if v in os.environ}
    return env


@lru_cache(maxsize=None)
def provenance() -> dict:
    """The dc_toolkit that wrote an output: its version and, from a git checkout, `git describe`."""
    try:
        version = importlib.metadata.version("dc_toolkit")
    except importlib.metadata.PackageNotFoundError:
        version = None
    try:
        git = subprocess.run(["git", "-C", str(Path(__file__).parent), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        git = None
    return {"dc_toolkit": version, "git": git}


@lru_cache(maxsize=None)
def measurement_digest() -> str:
    """Digest of the code that turns a pipeline into a result row (docstrings and comments aside): a change
    to it voids the recorded rows through sweep_state_{var}.json."""
    h = hashlib.blake2b(digest_size=8)
    h.update(repr((utils._ACC_INIT, utils._ACC_MAX, utils._ACC_MIN, utils.CHEAP_GATES)).encode())
    for obj in (utils._zarr_roundtrip, utils._info_bytes, utils._iter_chunk_slices, utils._error_sums,
                utils._merge_sums, utils._errors_from_sums, utils._rel, utils.within_limit,
                utils.evaluate_codec_pipeline, utils._gradient_rel_l1, utils._gradient_pieces,
                utils._classify_sample_dims, utils._is_time_like_coord, utils._is_vertical_like_coord,
                utils._is_vertical_like_dim, utils.horizontal_axes, utils._compute_inner_chunk_shape,
                utils._shrink_order, utils.compute_chunk_shape_for_eval, utils.codec_pipeline_kwargs, utils.ebcc_chunks,
                codecs.ZFPYRank, codecs.ZFPYFlat, codecs._ZFPYFlatCodec, codecs.EBCC, q99_cut, sweep_evaluators):
        tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body
                    and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)):
                node.body = body[1:] or [ast.Pass()]
        h.update(ast.dump(tree).encode())
    return h.hexdigest()


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


_REPLACED = ".__replaced__"


def remove_staged(merged_path: str, var=None) -> None:
    """Drop the staging copy of `var`, or the whole staging store, first moving back an array a killed
    promote_staged had set aside."""
    staging = staging_path(merged_path)
    if var is None and staging.is_dir():
        for aside in staging.glob(f"*{_REPLACED}"):
            target = Path(merged_path) / aside.name.removesuffix(_REPLACED)
            if not target.exists():
                os.replace(aside, target)
                click.echo(f"[store] restored {target.name}, set aside by an interrupted replacement.")
    shutil.rmtree(staging if var is None else staging / var, ignore_errors=True)


def promote_staged(merged_path: str, var: str) -> None:
    """Rename the verified field into the merged store (a zarr v3 array does not record its own name),
    replacing an array of the same name, then remove the staging store.  Drops the consolidated metadata,
    which would not describe the new array."""
    target, aside = Path(merged_path) / var, staging_path(merged_path) / f"{var}{_REPLACED}"
    drop_consolidated_metadata(merged_path)  # raises before any rename when it cannot
    if target.exists():
        os.replace(target, aside)  # two renames, so the old array is whole until the new one is in place
    try:
        os.replace(staging_path(merged_path) / var, target)
    except BaseException:
        if aside.exists() and not target.exists():
            os.replace(aside, target)
        raise
    remove_staged(merged_path)


def consolidate_store(merged_path: str) -> list:
    """Drop a killed run's staging leftovers and rewrite the consolidated metadata; returns the array names."""
    staging = staging_path(merged_path)
    leftovers = sorted(p.name for p in staging.iterdir() if not p.name.endswith(_REPLACED)) if staging.is_dir() else []
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
    of trusting a listing this run made stale; raises when it cannot."""
    root = Path(merged_path) / "zarr.json"
    if not root.is_file():
        return False
    meta = json.loads(root.read_text())
    if "consolidated_metadata" not in meta:
        return False
    meta.pop("consolidated_metadata")
    atomic_write(root, json.dumps(meta, indent=2))  # readers may have the store open
    return True


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


def atomic_write(path, text: str) -> None:
    """Write through a temporary file and a rename, so a kill never leaves a truncated file."""
    tmp = Path(f"{path}.{socket.gethostname()}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path, payload: dict, label: str) -> None:
    atomic_write(path, json.dumps(payload, indent=2, default=str))
    click.echo(f"[{label}] wrote {path}")


def json_safe(x):
    """`x` with tuples as lists and non-finite floats as None: strict JSON, as other zarr readers expect."""
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def state_digest(state: dict) -> str:
    """Digest of a sweep_state_{var}.json payload; the manifest records it, so compress can tell that a
    later sweep of the field replaced the state its manifest came from."""
    return hashlib.blake2b(json.dumps(state, sort_keys=True).encode(), digest_size=16).hexdigest()


def _lock_owner_alive(owner: dict) -> bool:
    """Whether the process that took a lock may still run: its Slurm step is running (its job, when it
    ran outside a step), else, on this host, its pid exists; unknown counts as alive.  A lock naming this
    very step (or this job, outside a step) is alive only as a live pid here: else an earlier incarnation
    of a requeued job left it.  Another cluster's jobs are unknown here."""
    job, step = str(owner.get("slurm_job_id") or ""), str(owner.get("slurm_step_id") or "")
    cluster, here = owner.get("slurm_cluster"), os.environ.get("SLURM_CLUSTER_NAME")
    if job and cluster and here and cluster != here:
        return True
    if job:
        if job == os.environ.get("SLURM_JOB_ID") and (not step.isdigit() or step == os.environ.get("SLURM_STEP_ID")):
            return owner.get("host") == socket.gethostname() and _pid_alive(owner.get("pid"))
        try:
            out = subprocess.run(["squeue", "-h", "-j", job] + (["-s", "-o", "%i"] if step.isdigit() else ["-o", "%T"]),
                                 capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return True
        if out.returncode == 0:
            listed = out.stdout.split()  # steps print as <job>.<step>, or <array>_<task>.<step>
            return any(s.rsplit(".", 1)[-1] == step for s in listed) if step.isdigit() else bool(listed)
        return "Invalid job id" not in out.stderr
    if owner.get("host") == socket.gethostname():
        return _pid_alive(owner.get("pid"))
    return True


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, TypeError):
        return True
    return True


def _lock_owner_text(owner: dict) -> str:
    return (f"host {owner.get('host')}, pid {owner.get('pid')}"
            + (f", Slurm step {owner['slurm_job_id']}.{owner.get('slurm_step_id')} on {owner.get('slurm_cluster')}"
               if owner.get("slurm_job_id") else "")
            + f", since {owner.get('time')}")


def acquire_lock(path: Path):
    """Create `path` exclusively, recording this process; a lock whose owner is gone is taken over.
    Returns None, or the description of a live owner."""
    me = {"host": socket.gethostname(), "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
          "slurm_step_id": os.environ.get("SLURM_STEP_ID"), "slurm_cluster": os.environ.get("SLURM_CLUSTER_NAME"),
          "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                text, age = path.read_text(), time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            try:
                owner = json.loads(text)
            except ValueError:
                owner = None
            if owner is None and age < 60:  # its owner is still writing it
                return f"a lock at {path} taken {age:.0f} s ago"
            if isinstance(owner, dict) and _lock_owner_alive(owner):
                return _lock_owner_text(owner)
            try:  # remove the stale lock only if no other run took it over while its owner was checked
                if path.read_text() == text:
                    path.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as fh:
            json.dump(me, fh)
        time.sleep(1)  # a run that judged the same stale lock may have replaced this one meanwhile
        holder = read_json(path, "lock")
        if holder == json.loads(json.dumps(me)):
            return None
        return _lock_owner_text(holder) if isinstance(holder, dict) else f"a lock at {path} taken meanwhile"
    return f"a lock at {path} that could not be taken over"


def release_lock(path: Path) -> None:
    """Remove `path` if it still names this process."""
    owner = read_json(path, "lock")
    if isinstance(owner, dict) and owner.get("pid") == os.getpid() and owner.get("host") == socket.gethostname():
        path.unlink(missing_ok=True)


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


# Per-rank peak in samples: the decoded copy (1x), the encoded bytes (1/ratio, up to 1x) and zarr's chunks in
# flight (~0.15x); measured 1.2-1.34x for compressing pipelines, 2.03x at ratio 1.
PER_RANK_WORKING_FACTOR = 2.0
# Per-rank temporaries in inner chunks: the error sums' float64 copies of a float32 chunk and their masks
# (~8x); the gradient's pieces (~132 MiB) fit in it at the default 16 MiB.
PER_RANK_CHUNK_FACTOR = 8


def node_steady_estimate_bytes(sample_bytes: int, ranks_on_node: int, inner_chunk_mib: int) -> int:
    """One shared sample + ranks x (PER_RANK_WORKING_FACTOR x sample + PER_RANK_CHUNK_FACTOR x chunk)."""
    ranks = max(1, int(ranks_on_node))
    return int(sample_bytes + ranks * sample_bytes * PER_RANK_WORKING_FACTOR
               + ranks * PER_RANK_CHUNK_FACTOR * max(1, int(inner_chunk_mib)) * 2**20)


def max_sample_bytes_for_ranks(budget_bytes: int, ranks_on_node: int, inner_chunk_mib: int) -> int:
    """Inverse of node_steady_estimate_bytes; 0 if nothing fits."""
    ranks = max(1, int(ranks_on_node))
    available = budget_bytes - ranks * PER_RANK_CHUNK_FACTOR * max(1, int(inner_chunk_mib)) * 2**20
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


def _cgroup_headroom():
    """Bytes the job's cgroup v2 still grants (limit - usage + clean file cache, which the kernel reclaims), or
    None."""
    for path in _cgroup_v2_memory_paths():
        base = Path(path).parent
        try:
            limit = (base / "memory.max").read_text().strip()
            if limit == "max":
                continue
            current = int((base / "memory.current").read_text())
            stat = dict(line.split() for line in (base / "memory.stat").read_text().splitlines())
            clean = sum(int(stat.get(k, 0)) for k in ("active_file", "inactive_file")) - sum(
                int(stat.get(k, 0)) for k in ("file_dirty", "file_writeback"))
            return int(limit) - current + max(0, clean)
        except (OSError, ValueError):
            continue
    return None


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


def available_memory(threshold: float):
    """Memory available now: the host's (psutil), or less when the job's cgroup grants less; None when it
    cannot be queried.  Warns once when `threshold` is above the recommended 0.80."""
    global _MEMCHECK_WARNED_HIGH
    if threshold > 0.80 and not _MEMCHECK_WARNED_HIGH:
        if MPI.COMM_WORLD.Get_rank() == 0:
            click.echo(f"[memcheck] WARNING: threshold {threshold:.2f} exceeds the recommended 0.80; "
                       f"less memory is left for what the estimates do not count.")
        _MEMCHECK_WARNED_HIGH = True
    try:
        avail = psutil.virtual_memory().available
    except Exception as e:
        click.echo(f"[memcheck] WARNING: could not query available memory ({e}).")
        return None
    cgroup = _cgroup_headroom()
    return avail if cgroup is None else min(avail, cgroup)


def check_memory_headroom(required_bytes: int, label: str, threshold: float = 0.80) -> None:
    """Abort if `required_bytes` exceeds `threshold` of available_memory()."""
    avail = available_memory(threshold)
    if avail is None:
        click.echo(f"[memcheck] WARNING: skipping the guard for {label}.")
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


def gate_bounds(phys_min, phys_max, phys_slack=0.0):
    """(low, high) the bounds gate counts excursions from, the slack included; None without bounds."""
    if phys_min is None and phys_max is None:
        return None
    slack = float(phys_slack or 0.0)
    return (-math.inf if phys_min is None else phys_min - slack, math.inf if phys_max is None else phys_max + slack)


def evaluate_gates(errors: dict, thr: dict, *, grad_threshold=None, grad_gate=False):
    """(keep, {"pass_<gate>": bool}) for one metrics dict, for the sweep and the verify gate.  A None metric
    or +inf limit passes.  The bounds are applied when the metrics are measured: N_Bounds counts the cells a
    round trip moves across them."""
    reasons = {f"pass_{t}": utils.within_limit(errors.get(m), thr.get(t)) for m, t in utils.CHEAP_GATES}
    reasons["pass_q99"] = utils.within_limit(errors.get("Q99_Rel"), thr.get("q99"))
    reasons["pass_finite"] = int(errors.get("N_Corrupt") or 0) == 0
    reasons["pass_bounds"] = int(errors.get("N_Bounds") or 0) == 0
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
    """(cut, over_nonzero): the q99 gate's tail cut, the 99th percentile of |finite values|, or of the
    non-zero ones (over_nonzero) when that is 0.  A field that is 0 almost everywhere (cloud ice, hail)
    would otherwise put every cell in the tail, and its extremes are its largest non-zero values.  The cut
    is None when no value is finite."""
    finite = sample_np[np.isfinite(sample_np)]
    finite = np.abs(finite.astype(np.float64) if finite.dtype.kind in "iu" else finite)  # |int8(-128)| wraps
    if not finite.size:
        return None, False
    cut = float(np.quantile(finite, 0.99))
    if cut != 0:
        return cut, False
    nonzero = finite[finite > 0]
    return (float(np.quantile(nonzero, 0.99)) if nonzero.size else 0.0), bool(nonzero.size)


def fmt3(x) -> str:
    return f"{x:.3e}" if isinstance(x, float) else "n/a"


def verify_thresholds(manifest, overrides=None) -> dict:
    """The verify gate's relative thresholds: the manifest's, each overridden by a non-None `overrides`
    entry; without a manifest, L2, Linf and bias default to multiples of an --l1-threshold, as in the sweep."""
    man_thr = dict((manifest or {}).get("effective_thresholds", {}) or {})
    given = {k: v for k, v in (overrides or {}).items() if v is not None}
    if not man_thr and given.get("l1") is not None:
        man_thr = {"l2": _L2_MULT_DEFAULT * given["l1"], "linf": _LINF_MULT_DEFAULT * given["l1"],
                   "bias": _BIAS_MULT_DEFAULT * given["l1"]}
    man_thr.update(given)
    return {k: (float(man_thr[k]) if man_thr.get(k) is not None else math.inf) for k in GATE_KEYS}


def verify_against_manifest(var: str, errors: dict, manifest, overrides=None):
    """Production verify gate: `errors` against verify_thresholds.  The finite and bounds gates need no
    threshold and always apply.  Returns (status, detail), status "pass", "fail" or "no-thresholds"."""
    thr = verify_thresholds(manifest, overrides)
    keep, reasons = evaluate_gates(errors, thr)
    if keep and not any(math.isfinite(v) for v in thr.values()):
        return "no-thresholds", ""
    if keep:
        return "pass", ""
    failed = ", ".join(k for k, ok in reasons.items() if not ok)
    detail = (f"{var}: verify gate FAILED ({failed}) | "
              f"L1={fmt3(errors.get('Relative_Error_L1'))} L2={fmt3(errors.get('Relative_Error_L2'))} "
              f"Linf={fmt3(errors.get('Relative_Error_Linf'))} bias={fmt3(errors.get('Bias_Rel'))} "
              f"q99={fmt3(errors.get('Q99_Rel'))} n_corrupt={errors.get('N_Corrupt', 0)} "
              f"n_bounds={errors.get('N_Bounds', 0)} | thresholds: "
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
        click.echo(f"[verify-gate] {var}: no thresholds (none in manifest_{var}.json and no --l1-threshold ...): "
                   f"the error norms are advisory; the finite and bounds gates passed.")
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


# Pipelines that cannot give a missing value back as missing: FixedScaleOffset and zfp write numbers into
# it, Delta spreads it over the rest of the chunk, EBCC's C library exits on it.
def keeps_nonfinite(codec, dtype) -> bool:
    if isinstance(codec, (utils.zarrcodecs_nc.FixedScaleOffset, utils.zarrcodecs_nc.ZFPY, utils.EBCC)):
        return False
    return not (isinstance(codec, utils.zarrcodecs_nc.Delta) and np.dtype(dtype).kind == "f")


def codec_spaces(sample_da, space_args: dict, fso_range, chunk_shapes=None, nonfinite: int = 0):
    """(compressors, filters, serializers) for the sample.  `fso_range` is the FULL field's (min, max):
    FixedScaleOffset and EBCC must not be parameterised from a sample that may miss the extremes.  When
    the field holds `nonfinite` NaN/Inf cells, the codecs that cannot keep them are left out."""
    try:
        spaces = (utils.compressor_space(sample_da, space_args["with_lossy"], space_args["compressor_class"]),
                  utils.filter_space(sample_da, space_args["with_lossy"], space_args["filter_class"],
                                     data_range=fso_range),
                  utils.serializer_space(sample_da, space_args["with_lossy"], space_args["serializer_class"],
                                         with_ebcc=space_args["with_ebcc"], data_range=fso_range,
                                         chunk_shapes=chunk_shapes))
    except (ValueError, TypeError) as e:
        raise click.ClickException(str(e))
    kept = tuple([c for c in space if c is None or not nonfinite or keeps_nonfinite(c, sample_da.dtype)]
                 for space in spaces)
    for space, left, what in zip(spaces, kept, ("compressor", "filter", "serializer")):
        if space and not left:
            raise click.ClickException(f"the {what} class has nothing for this field: it holds NaN/Inf, which "
                                       f"they cannot keep")
    return kept


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
    """Raise a ClickException for codecs that cannot pair, a filter built for another dtype (numcodecs would
    reinterpret the bytes), or an EBCC input the C library would exit on (wrong dtype, tile not dividing the
    frame, NaN/Inf, float64 beyond the float32 range)."""
    compressor, filt, serializer = combo
    if not utils.combo_is_valid(filt, serializer, compressor, dtype=da.dtype):
        raise click.ClickException(
            f"{var}: invalid pipeline {utils.pipeline_name(*combo)} (e.g. FixedScaleOffset->ZFPY, "
            f"BitRound->ZFPY below the mantissa width, "
            f"or EBCC with a compressor or a filter other than AsType).")
    config = getattr(filt, "codec_config", None) or {}
    for key in ("dtype", "decode_dtype"):
        if config.get(key) is not None and np.dtype(config[key]) != da.dtype:
            raise click.ClickException(f"{var}: the pipeline's {utils.codec_label(filt)} was built for "
                                       f"{config[key]}, not the field's {da.dtype}.")
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
        data = dask.array.asarray(da.data)
        finite, peak = dask.compute(dask.array.isfinite(data).all(), abs(data).max())
        if not bool(finite):
            raise click.ClickException(f"{var}: the field contains NaN/Inf, which EBCC cannot encode.")
        if float(peak) > float(np.finfo(np.float32).max):
            raise click.ClickException(f"{var}: the field exceeds the float32 range EBCC casts to.")


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


def store_layout(da, combo, geometry: dict, opts):
    """(inner_chunks, shards) of the stored field: EBCC's tile, else the chunk rule at the geometry's size."""
    forced = utils.ebcc_chunks(combo[2], da.shape) if isinstance(combo[2], utils.EBCC) else None
    return utils.compute_chunk_and_shard_shape(
        da.shape, da.dtype, inner_mib=geometry["inner_chunk_mib"], shard_mib=opts.shard_mib, dims=tuple(da.dims),
        allow_spatial_split=geometry["spatial_split"], inner_chunks=forced)


def read_blocks(shape, dtype, write_unit, target_bytes: int = 512 * 2**20) -> tuple:
    """Block shape compress reads the source in: whole write units, widened along the last axis they do not
    span to about `target_bytes`, so a task reads, writes and re-reads its own shards."""
    block = list(write_unit)
    unit_bytes = int(np.dtype(dtype).itemsize) * int(np.prod(write_unit))
    partial = [a for a in range(len(shape)) if write_unit[a] < shape[a]]
    if partial:
        a = partial[-1]
        block[a] = min(shape[a], write_unit[a] * max(1, target_bytes // max(1, unit_bytes)))
    return tuple(block)


def write_concurrency(block_bytes: int, inner_bytes: int, nblocks: int, opts, var: str) -> int:
    """Blocks compress holds at once: as many as fit in --memory-threshold of the available memory, at most
    --threads; aborts when one does not fit.  Per block: the block, its encoded chunks and shards and what the
    allocator keeps of them (the write peaks near 3.5x the block; the verify's read-back follows it), and the
    error sums' chunk temporaries."""
    per_block = 4 * block_bytes + PER_RANK_CHUNK_FACTOR * inner_bytes
    avail = available_memory(opts.memory_threshold)
    fit = int(opts.threads) if avail is None else int(opts.memory_threshold * avail // per_block)
    if fit < 1:
        click.echo(f"[memcheck] REFUSING to write '{var}': one block needs {hsize(per_block)}, above "
                   f"{int(opts.memory_threshold * 100)}% of the available {hsize(avail)}.  Lower --shard-mib or "
                   f"raise --memory-threshold (max 0.95).")
        abort(1)
    return max(1, min(int(opts.threads), nblocks, fit))


def persist_field(da, var: str, merged_path: str, combo, opts, layout, q99_abs, bounds, tasks: int) -> dict:
    """Write one field, chunked in read_blocks, with `combo` into the staging store, `tasks` blocks at a
    time, creating the merged store if needed.  Returns ratio, errors, eucd, geometry and timing."""
    inner_chunks, shards = layout
    itemsize = int(da.dtype.itemsize)
    inner_bytes = itemsize * int(np.prod(inner_chunks))
    shard_bytes = itemsize * int(np.prod(shards)) if shards is not None else inner_bytes
    layout_text = (f"inner chunks={inner_chunks}, {hsize(inner_bytes)}; "
                   + ("sharding skipped (a shard would hold < 2 chunks)" if shards is None
                      else f"shards={shards}, {hsize(shard_bytes)}"))
    block_bytes = itemsize * int(np.prod(da.data.chunksize))
    click.echo(f"[persist] {var} -> {merged_path} ({layout_text}; read in blocks of {da.data.chunksize}, "
               f"{hsize(block_bytes)}, {tasks} at a time)")

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
        with dask.config.set(num_workers=tasks):
            ratio, errors, eucd = utils.persist_with_codec_pipeline(
                da, store, component=var, codec_kwargs=utils.codec_pipeline_kwargs(*combo),
                inner_chunks=inner_chunks, shards=shards, verify=opts.verify, q99_abs=q99_abs, bounds=bounds)
    finally:
        close_store(store)
    return {"ratio": float(ratio), "errors": errors, "eucd": eucd,
            "inner_chunks": list(inner_chunks), "inner_chunk_bytes": int(inner_bytes),
            "shards": (list(shards) if shards is not None else None),
            "shard_bytes": (int(shard_bytes) if shards is not None else None),
            "sharding_skipped": shards is None, "seconds": time.perf_counter() - t0}


def fso_range_problem(combo, errors):
    """Why the pipeline's FixedScaleOffset cannot hold the written field (the verify pass's Source_Min and
    Source_Max), or None: it does not clip, and a value beyond its range wraps.  The field's extremes are
    scaled in its own dtype, as the codec does; the encoding is monotone, so they decide."""
    filt = combo[1]
    if not isinstance(filt, utils.zarrcodecs_nc.FixedScaleOffset) or not errors:
        return None
    cfg = filt.codec_config
    offset, scale, astype = float(cfg["offset"]), float(cfg["scale"]), np.dtype(cfg["astype"])
    smin, smax = errors.get("Source_Min"), errors.get("Source_Max")
    if astype.kind not in "iu" or smin is None or not math.isfinite(smin):
        return None  # a float target overflows to inf, which the finite gate counts
    with np.errstate(over="ignore", invalid="ignore"):
        coded = np.around((np.array([smin, smax], dtype=cfg["dtype"]) - offset) * scale)
    info = np.iinfo(astype)
    if np.all(np.isfinite(coded)) and info.min <= coded.min() and coded.max() <= info.max:
        return None
    return (f"the field spans [{smin:g}, {smax:g}], which its FixedScaleOffset to {astype} codes as "
            f"[{coded[0]:g}, {coded[1]:g}], beyond [{info.min}, {info.max}]")


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
    """Collective.  Sets opts.with_ebcc, pins zarr's pool to one worker (one core per rank) and creates the
    output directory."""
    reset_memcheck_state()
    opts.with_ebcc = opts.with_ebcc or opts.serializer_class.lower() == "ebcc"
    if opts.with_ebcc and not opts.with_lossy:
        raise click.ClickException("EBCC is lossy: --with-ebcc / --serializer-class ebcc need --with-lossy.")
    if opts.with_ebcc and not utils.EBCC_AVAILABLE:
        raise click.ClickException("--with-ebcc needs the ebcc package: pip install -e '.[ebcc]'")
    if opts.phys_min is not None and opts.phys_max is not None and opts.phys_min > opts.phys_max:
        raise click.ClickException(f"--phys-min {opts.phys_min:g} is above --phys-max {opts.phys_max:g}.")
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    node_comm, ranks_on_node, local_rank = utils.detect_node_topology(comm)
    leaders = comm.Split(0 if local_rank == 0 else MPI.UNDEFINED, key=rank)
    node_id = node_comm.bcast(leaders.Get_rank() if local_rank == 0 else None, root=0)
    n_nodes = comm.allreduce(1 if local_rank == 0 else 0)
    cores_avail = utils.detect_cores_available()
    cores = utils.detect_physical_cores() if size == 1 else 1
    if cores > 1:
        click.echo(f"[topology] NOTE: one rank on {cores} cores.  A rank evaluates one pipeline at a time; "
                   f"start one rank per core to use them all (mpirun -n {cores} dc_toolkit ...).")
    utils.check_thread_oversubscription(abort_if_unsafe=opts.oversubscription_check, rank=rank)
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
    field_bytes, (floor_bytes, floor_kept) = int(da.nbytes), utils.minimum_sample(da)
    node_budget, source = detect_node_memory_budget()
    max_safe = max_sample_bytes_for_ranks(int(node_budget * opts.memory_threshold), ranks, chunk_mib)
    wanted = max(int(opts.eval_data_size_limit), floor_bytes)
    # Rank 0 sizes the sample for everyone, so every rank adopts the smallest node's limit.
    limit = sweep.comm.allreduce(min(wanted, max_safe), op=MPI.MIN)
    if limit < floor_bytes:
        if sweep.rank == 0:
            click.echo(f"[memcheck] FATAL: the smallest sample of '{var}' ({hsize(floor_bytes)}: {floor_kept}) "
                       f"does not fit with {ranks} rank(s) per node, inner_chunk_mib={chunk_mib} and a node "
                       f"memory budget of {hsize(node_budget)} (from {source}) at threshold "
                       f"{opts.memory_threshold:.2f}.  Start fewer ranks per node or request more RAM.")
        abort(1)
    if sweep.rank == 0 and floor_bytes > int(opts.eval_data_size_limit) and field_bytes > int(opts.eval_data_size_limit):
        click.echo(f"[sample] raised the sample budget of '{var}' from {hsize(opts.eval_data_size_limit)} "
                   f"(--eval-data-size-limit) to {hsize(floor_bytes)}, its smallest sample: {floor_kept}.")
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


def sweep_build_sample(da, limit: int, scan, opts, sweep: SweepContext):
    """Collective.  Rank 0 builds the sample from the time steps and levels on which the field varies
    (scan.varying), and each node maps one shared copy (shared_sample_window); the dim coords travel along
    so every rank classifies the dims alike.  Returns (sample_np, sample_da, win); the caller frees win,
    collectively."""
    comm, rank = sweep.comm, sweep.rank
    if rank == 0:
        candidates = {d: np.flatnonzero(v) for d, v in scan.varying.items() if v.any() and not v.all()}
        try:
            local = utils.build_representative_sample(da, limit, rank=rank, policy=opts.sampling_policy,
                                                      vertical_floor=opts.vertical_floor,
                                                      candidates=candidates).compute()
        except utils.SampleTooLargeError as e:
            raise click.ClickException(str(e))  # sweep_dataset aborts the job
        sample_np_local = np.ascontiguousarray(local.values)
        meta = {"dims": tuple(local.dims), "attrs": dict(local.attrs), "name": local.name,
                "shape": tuple(sample_np_local.shape), "dtype": str(sample_np_local.dtype),
                "coords": {d: (np.asarray(local.coords[d].values), dict(local.coords[d].attrs),
                               dict(local.coords[d].encoding)) for d in local.dims if d in local.coords}}
    else:
        sample_np_local = meta = None
    meta = comm.bcast(meta, root=0)
    sample_np, win = shared_sample_window(sample_np_local, meta["shape"], meta["dtype"], sweep)
    del sample_np_local
    coords = {d: xr.Variable((d,), v, attrs=a, encoding=e) for d, (v, a, e) in meta["coords"].items()}
    sample_da = xr.DataArray(sample_np, dims=meta["dims"], coords=coords, attrs=meta["attrs"], name=meta["name"])
    return sample_np, sample_da, win


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
            df = pd.read_csv(path, on_bad_lines="skip")
            numeric = [c for c in ("ratio", "eucd", *_METRIC_COLUMNS) if c in df]
            df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")  # a header-only file reads as text
            df = df.dropna(subset=_ROW_KEY_COLUMNS)
            if len(df):
                frames.append(df)
        except Exception as e:
            click.echo(f"[sweep] WARNING: cannot read {path.name} ({e}); "
                       + ("set aside as *.unreadable." if quarantine else "skipped."))
            if quarantine:
                path.rename(path.with_name(path.name + ".unreadable"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PARTIAL_CSV_COLUMNS)


def sweep_recorded_rows(var, sample_np, q99_abs, fso_range, bounds, opts, sweep: SweepContext):
    """Collective: (reused pipelines, crashes, state digest), decided on rank 0 before any rank opens its
    CSV.  Rows are reused only with --resume, when sweep_state_{var}.json matches this run's dataset,
    sample, sampling and chunk settings, bounds, measuring code, metric definitions and library versions,
    and when they carry every metric the current gates need; otherwise the per-rank CSVs are removed and
    the previous results, manifest and plan are kept as *.previous.  Failed combos are always retried.
    `crashes` maps each pipeline in flight when an earlier run died to how often, and whether it died
    evaluated alone (then it is left out)."""
    done, crashes, digest = set(), {}, None
    if sweep.rank == 0:
        where = Path(opts.where_to_write)
        state = json.loads(json.dumps({
            "dataset_file": os.path.abspath(opts.dataset_file), "sample_shape": list(sample_np.shape),
            "dtype": str(sample_np.dtype), "full_field_range": fso_range,
            "sampling_policy": opts.sampling_policy, "vertical_floor": opts.vertical_floor,
            "inner_chunk_mib": opts.inner_chunk_mib, "spatial_split": opts.spatial_split,
            "sample_digest": hashlib.blake2b(memoryview(sample_np).cast("B"), digest_size=16).hexdigest(),
            "bounds": bounds, "metric_definitions": utils.METRIC_DEFINITIONS, "code": measurement_digest(),
            "row_columns": PARTIAL_CSV_COLUMNS, "env": env_versions()}, default=str))
        digest = state_digest(state)
        state_path = where / f"sweep_state_{var}.json"
        previous = read_json(state_path, "resume")
        rank_csvs = rank_files(where, "config_space", var)
        # The range is informative: the pipelines that depend on it carry it in their JSON.
        changed = (["sweep_state file (missing)"] if previous is None and rank_csvs
                   else sorted(k for k in state if k != "full_field_range" and previous is not None
                               and previous.get(k) != state[k]))
        restart = bool(changed) or not opts.resume
        if opts.resume and changed and rank_csvs:
            click.echo(f"[resume] {var}: the recorded rows were measured with another {', '.join(changed)}; "
                       f"starting this field from scratch.")
        crashes_path = where / f"crashes_{var}.json"
        crashes = {} if restart else (read_json(crashes_path, "resume") or {})
        journals = rank_files(where, "inflight", var)
        for path in journals:
            for line in path.read_text(errors="replace").splitlines():  # a node crash can leave garbage
                mode, name, key = (line.split("\t", 2) + ["", ""])[:3]
                if key and not restart:
                    entry = crashes.setdefault(key, {"name": name, "crashes": 0, "alone": False})
                    entry["crashes"] += 1
                    entry["alone"] = entry["alone"] or mode == "alone"
        if crashes:  # recorded before the journals go, so a kill in between loses no crash
            atomic_write(crashes_path, json.dumps(crashes, indent=2))
        else:
            crashes_path.unlink(missing_ok=True)
        for path in journals:
            path.unlink()
        if restart:
            moved = [n for n in (f"results_{var}.parquet", f"manifest_{var}.json", f"config_space_{var}.csv")
                     if (where / n).is_file()]
            for n in moved:
                os.replace(where / n, where / f"{n}.previous")
            if moved:
                click.echo(f"[resume] {var}: kept the previous {', '.join(moved)} as *.previous.")
        for path in (rank_csvs if restart else []) + rank_files(where, "failures", var):
            path.unlink()
        atomic_write(state_path, json.dumps(state, indent=2))
        if not restart:
            for path in rank_csvs:
                drop_partial_last_line(path)
            prev = read_rank_csvs(where, var, quarantine=True)
            done = set(prev.loc[reusable_rows(prev, q99_abs, opts, sweep.thresholds), "pipeline"])
    done, crashes = sweep.comm.bcast((done, crashes), root=0)
    return done, crashes, digest


def missing_metrics(prev: pd.DataFrame, q99_abs, opts, thresholds: dict) -> dict:
    """{pass column: rows lacking the metric its enabled gate needs}.  q99 and gradient are recorded only
    with their gate on (the gradient, with the shortcircuit, only for rows passing the cheap gates).  A
    field with no finite value has no q99 cut, so no row records it; requiring it would re-evaluate the
    field for ever."""
    out = {}
    if opts.extremes_sensitive and q99_abs is not None and math.isfinite(q99_abs):
        out["pass_q99"] = prev["q99_rel"].isna()
    if opts.gradient_gate:
        needed = pd.Series(True, index=prev.index)
        if opts.gradient_shortcircuit:
            for column, key in (("l1_rel", "l1"), ("l2_rel", "l2"), ("linf_rel", "linf"), ("bias_rel", "bias")):
                if math.isfinite(thresholds[key]):
                    needed &= ~(prev[column] > thresholds[key])
        out["pass_grad"] = prev["grad_rel"].isna() & needed
    return out


def reusable_rows(prev: pd.DataFrame, q99_abs, opts, thresholds: dict) -> pd.Series:
    """Rows carrying every metric the current gates need (missing_metrics)."""
    ok = pd.Series(True, index=prev.index)
    for missing in missing_metrics(prev, q99_abs, opts, thresholds).values():
        ok &= ~missing
    return ok


def sweep_config_space(compressors, filters, serializers, max_evals, rank, dtype, all_finite: bool) -> list:
    """Triples to evaluate: the EBCC ones (none unless `all_finite`, never cut), then the valid non-EBCC
    product, a seeded uniform subset of --max-evals of them when capped (a quick-test knob), shuffled so
    each node's every-n_nodes-th share mixes cheap and expensive codecs; the seeds depend only on the counts,
    so --resume keeps the order."""
    regular = [s for s in serializers if not isinstance(s, utils.EBCC)]
    total = len(compressors) * len(filters) * len(regular)
    config_space = [(c, f, s) for c, f, s in itertools.product(compressors, filters, regular)
                    if utils.combo_is_valid(f, s, c, dtype=dtype)]
    if rank == 0 and len(config_space) < total:
        click.echo(f"[combo-filter] skipped {total - len(config_space)} unsupported filter/serializer "
                   f"pairing(s) (FixedScaleOffset->ZFPY, BitRound->ZFPY below the mantissa width).")
    if max_evals is not None and max_evals < len(config_space):
        if rank == 0:
            click.echo(f"[max-evals] evaluating a uniform subset of {max_evals} of the {len(config_space)} combos.")
        keep = np.sort(np.random.default_rng(seed=len(config_space)).choice(len(config_space), max_evals,
                                                                            replace=False))
        config_space = [config_space[i] for i in keep]
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
               f"~{ranks * max(1, opts.inner_chunk_mib) * PER_RANK_CHUNK_FACTOR} MiB (ranks x "
               f"{PER_RANK_CHUNK_FACTOR} x inner_chunk_mib) "
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


def sweep_evaluators(var, sample_np, sample_da, q99_abs, bounds, opts, sweep: SweepContext):
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
            sample_np, sample_da.dims, utils.codec_pipeline_kwargs(*cfg), chunks=chunks, q99_abs=q99_abs,
            bounds=bounds, compute_gradient=opts.gradient_gate, gradient_axes=grad_axes if opts.gradient_gate else None,
            precheck_thresholds=sweep.thresholds if (opts.gradient_gate and opts.gradient_shortcircuit) else None)
        return {"name": utils.pipeline_name(*cfg), "compressor": utils.codec_label(compressor),
                "filter": utils.codec_label(filt), "serializer": utils.codec_label(serializer),
                "pipeline": utils.pipeline_json(*cfg),
                "ratio": float(ratio), "errors": errors, "eucd": float(eucd)}

    def gate(errors):
        return evaluate_gates(errors, sweep.thresholds, grad_threshold=opts.gradient_threshold,
                              grad_gate=opts.gradient_gate)

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
    "decoded_min", "decoded_max", "n_corrupt", "n_bounds", "eucd",
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


def _evaluate_and_record(cfg, mode: str, evaluate_one, gate, pw, fw, jf, failures: list) -> None:
    """Evaluate one combo, journaled in `jf` as "mode, name, pipeline" while it runs, and write its row or
    failure."""
    name, key = utils.pipeline_name(*cfg), utils.pipeline_json(*cfg)
    jf.seek(0)
    jf.truncate()
    jf.write(f"{mode}\t{name}\t{key}\n")
    jf.flush()
    try:
        r = evaluate_one(cfg)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # one broken combo never stops the sweep; a Rust panic is a BaseException
        failures.append((name, key, repr(e)))
        fw.writerow(failures[-1])
    else:
        err = r["errors"]
        keep, reasons = gate(err)
        pw.writerow([
            r["name"], r["compressor"], r["filter"], r["serializer"], r["pipeline"],
            r["ratio"], err["Relative_Error_L1"], err["Relative_Error_L2"], err["Relative_Error_Linf"],
            err.get("Bias_Rel"), err.get("Q99_Rel"), err.get("Grad_Rel"),
            err.get("Decoded_Min"), err.get("Decoded_Max"), err.get("N_Corrupt", 0), err.get("N_Bounds", 0), r["eucd"],
            reasons["pass_l1"], reasons["pass_l2"], reasons["pass_linf"], reasons["pass_bias"],
            reasons["pass_q99"], reasons["pass_bounds"], reasons["pass_grad"], reasons["pass_finite"],
            keep,
        ])
    jf.seek(0)
    jf.truncate()
    jf.flush()


class _Journal:
    """inflight_{var}_rank{rank}.csv, holding the combo this rank evaluates: a rank killed inside one (a crash,
    the OOM killer) leaves it behind.  A signal from outside (a cancel, the walltime, --signal) and any Python
    exit clear it: the combo is not at fault."""
    SIGNALS = tuple(getattr(signal, n) for n in ("SIGTERM", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2", "SIGALRM",
                                                 "SIGXCPU") if hasattr(signal, n))

    def __init__(self, where, var, rank):
        self.path = Path(where) / f"inflight_{var}_rank{rank}.csv"

    def __enter__(self):
        self.fh = open(self.path, "w", buffering=1)
        self.previous = {s: signal.signal(s, self._on_term) for s in self.SIGNALS  # a handler of others stays
                         if signal.getsignal(s) == signal.SIG_DFL}
        return self.fh

    def _on_term(self, signum, frame):
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.flush()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def __exit__(self, *exc):
        for s, handler in self.previous.items():
            signal.signal(s, handler)
        self.fh.close()
        self.path.unlink(missing_ok=True)


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
            open(failures_path, "w", newline="", buffering=1) as ff, _Journal(opts.where_to_write, var, rank) as jf:
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
            _evaluate_and_record(config_space[share[i]], "crowd", evaluate_one, gate, pw, fw, jf, failures)
    win.Unlock_all()
    win.Free()
    return failures


def sweep_run_alone(config_space, isolated, var, opts, sweep: SweepContext, evaluate_one, gate) -> list:
    """Collective.  Rank 0 evaluates, one at a time, the combos in flight when an earlier run died,
    journaled as evaluated alone: one that kills the rank again is then known to be the culprit and left out
    by the next run.  The other ranks wait.  Returns rank 0's failures."""
    failures = []
    if isolated and sweep.rank == 0:
        click.echo(f"[resume] {var}: evaluating one at a time the {len(isolated)} combo(s) in flight when an "
                   f"earlier run died.")
        where = Path(opts.where_to_write)
        with open(where / f"config_space_{var}_rank0.csv", "a", newline="", buffering=1) as pf, \
                open(where / f"failures_{var}_rank0.csv", "a", newline="", buffering=1) as ff, \
                _Journal(where, var, 0) as jf:
            pw, fw = csv.writer(pf), csv.writer(ff)
            for i in isolated:
                _evaluate_and_record(config_space[i], "alone", evaluate_one, gate, pw, fw, jf, failures)
    sweep.comm.Barrier()
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
    "decoded_min": "Decoded_Min", "decoded_max": "Decoded_Max", "n_corrupt": "N_Corrupt", "n_bounds": "N_Bounds",
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


def sweep_select_best(where_to_write, var, gate, planned: set, missing):
    """Consolidate the per-rank CSVs into results_{var}.parquet, keeping the `planned` pipelines, re-gated,
    a row lacking a metric an enabled gate needs (`missing`, its re-evaluation failed) failing that gate, and
    pick the kept combo with the best ratio (ties: lower L1, then pipeline) FROM DISK, so --resume of a
    finished field still finds it.  Returns (best {name, pipeline, ratio, l1_rel, eucd} or None, n_passed,
    n_rows, path)."""
    consolidated = read_rank_csvs(where_to_write, var)
    # A combo re-evaluated for a metric its row lacked has several rows; last() merges their non-null values.
    consolidated = consolidated.groupby("pipeline", as_index=False, sort=False).last()
    stale = ~consolidated["pipeline"].isin(planned)
    if stale.any():
        click.echo(f"[sweep] ignoring {int(stale.sum())} recorded row(s) outside this sweep's codec space.")
        consolidated = consolidated[~stale]
    consolidated = regate(consolidated, gate)
    for column, rows in missing(consolidated).items():
        if rows.any():
            click.echo(f"[sweep] {int(rows.sum())} row(s) lack the metric of {column} (their re-evaluation "
                       f"failed): not kept.")
            consolidated.loc[rows, [column, "keep"]] = False
    parquet_path = os.path.join(where_to_write, f"results_{var}.parquet")
    consolidated.to_parquet(parquet_path + ".tmp", index=False)
    os.replace(parquet_path + ".tmp", parquet_path)
    click.echo(f"[sweep] consolidated the per-rank CSVs -> {parquet_path} ({len(consolidated)} row(s)).")
    kept = kept_rows(consolidated)
    if len(kept) == 0:
        return None, 0, int(len(consolidated)), parquet_path
    top = kept.sort_values(["ratio", "l1_rel", "pipeline"], ascending=[False, True, True]).iloc[0]
    best = {"name": str(top["name"]), "pipeline": json.loads(top["pipeline"]),
            "ratio": float(top["ratio"]), "l1_rel": float(top["l1_rel"]), "eucd": float(top["eucd"])}
    return best, int(len(kept)), int(len(consolidated)), parquet_path


SWEEP_ARG_KEYS = (
    "eval_data_size_limit", "inner_chunk_mib", "max_inner_chunk_mib",
    "spatial_split", "compressor_class", "filter_class", "serializer_class", "with_lossy", "with_ebcc",
    "sampling_policy", "vertical_floor", "l1_threshold", "l2_threshold", "linf_threshold", "bias_threshold",
    "q99_threshold", "l2_gate", "linf_gate", "bias_gate", "extremes_sensitive", "phys_min", "phys_max",
    "phys_tolerance",
    "gradient_gate", "gradient_threshold", "resume", "max_evals",
)


def sweep_manifest(var, opts, sweep: SweepContext, *, num_combos, n_rows, n_passed, total_failures, crashed,
                   seconds, parquet_path, best, q99_abs, digest) -> dict:
    return {
        "command": "evaluate_combos",
        "dataset_file": os.path.abspath(opts.dataset_file), "var": str(var),
        "where_to_write": os.fspath(opts.where_to_write),
        "args": {k: getattr(opts, k) for k in SWEEP_ARG_KEYS},
        "topology": {"size": int(sweep.size), "cores_avail": int(sweep.cores_avail),
                     "ranks_on_node": int(sweep.ranks_on_node)},
        "effective_thresholds": {k: (None if not math.isfinite(v) else float(v)) for k, v in sweep.thresholds.items()},
        "gradient_threshold": float(opts.gradient_threshold) if opts.gradient_gate else None,
        "phys_min": opts.phys_min, "phys_max": opts.phys_max,
        "phys_slack": float(getattr(opts, "phys_slack", 0.0)),
        "q99_abs": q99_abs,
        "num_combos": int(num_combos), "num_rows": int(n_rows), "num_passed": int(n_passed),
        "num_filtered": int(n_rows - n_passed), "num_failed_total": int(total_failures or 0),
        "crashed": crashed,
        "var_sweep_seconds": float(seconds),
        "env": env_versions(),
        "provenance": provenance(),
        "sweep_state_digest": digest,
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
    scan = utils.scan_field(da, comm=sweep.comm)
    if not scan.readable:  # compress could not read it either; the same on every rank
        message = f"{var} could not be read in full (see [range] above); sweep it again once the file reads cleanly"
        if opts.field_to_compress is not None:
            raise click.ClickException(message)
        if rank == 0:
            click.echo(f"[var] skipping {message}")
        return
    sample_np, sample_da, win = sweep_build_sample(da, limit, scan, opts, sweep)
    _sweep_variable_body(da, var, opts, sweep, n_vars, t0, sample_np, sample_da, scan)
    # Free is collective: after an error the exception must reach the abort hook rather than wait here.
    del sample_np, sample_da
    win.Free()


def _sweep_variable_body(da, var, opts, sweep: SweepContext, n_vars, t0, sample_np, sample_da, scan):
    # Collective, like sweep_variable: every rank runs it, and an early return must be identical on every rank;
    # sets opts.phys_slack.
    rank, fso_range = sweep.rank, scan.range
    span = float(fso_range[1] - fso_range[0]) if fso_range else 0.0
    opts.phys_slack = float(getattr(opts, "phys_tolerance", 0.0) or 0.0) * span   # absolute; the manifest carries it
    bounds = gate_bounds(opts.phys_min, opts.phys_max, opts.phys_slack)
    if rank == 0:
        if opts.phys_slack:
            click.echo(f"[gates] {var}: bounds slack {opts.phys_slack:g} ({opts.phys_tolerance:g} of the range {span:g})")
        if bounds and fso_range and (fso_range[0] < bounds[0] or fso_range[1] > bounds[1]):
            click.echo(f"[gates] {var}: the field itself spans [{fso_range[0]:g}, {fso_range[1]:g}], across the "
                       f"bounds; the bounds gate counts only the cells a pipeline moves across them.")
        if fso_range and max(abs(fso_range[0]), abs(fso_range[1])) >= 1e30:
            click.echo(f"[range] WARNING: {var} holds values up to {max(map(abs, fso_range)):g}: a fill value the "
                       f"file does not declare?  The relative norms are measured against them.")
        if scan.nonfinite:
            click.echo(f"[range] {var}: {scan.nonfinite} NaN/Inf cell(s); FixedScaleOffset, Delta (floats), zfp and "
                       f"EBCC cannot keep them and are left out.")
    # q99_cut holds ~3 sample-sized temporaries: rank 0 computes it, the others receive the float
    q99_abs, over_nonzero = (sweep.comm.bcast(q99_cut(sample_np) if rank == 0 else None, root=0)
                             if opts.extremes_sensitive else (None, False))
    if opts.extremes_sensitive and rank == 0:
        basis = "of the non-zero |values|: the plain one is 0" if over_nonzero else "of |value|"
        click.echo(f"[gates] {var}: q99 cut={q99_abs} (99th percentile {basis}; the q99 gate's extreme tail)")
    # The errors and ratios of a sample without variation do not carry over to a field that varies.  A field
    # whose finite values are all one number, or that has none, needs no search.
    sample_range = sweep.comm.bcast(utils.finite_range(sample_np) if rank == 0 else None, root=0)
    lock = Path(opts.where_to_write) / f"sweep_{var}.lock"
    owner = None
    try:
        if fso_range is None or fso_range[0] == fso_range[1]:
            what = (f"every finite value of the field is {fso_range[0]:g}" if fso_range is not None
                    else "the field has no finite value")
            if rank == 0:
                click.echo(f"[sample] {var}: {what}; the search is skipped and Zstd stores it losslessly.")
            spaces = ([utils.zarrcodecs_nc.Zstd(level=6)], [None], [None])
        elif sample_range is None or sample_range[0] == sample_range[1]:
            what = "has no finite value" if sample_range is None else f"holds the single value {sample_range[0]:g}"
            raise click.ClickException(f"the sample {what} while the field spans [{fso_range[0]:g}, "
                                       f"{fso_range[1]:g}]; raise --eval-data-size-limit so that the sample "
                                       f"reaches that variation")
        else:
            chunks = [utils.compute_chunk_shape_for_eval(shape, sample_np.dtype, target_mib=opts.inner_chunk_mib,
                                                          dims=sample_da.dims, allow_spatial_split=opts.spatial_split)
                      for shape in (sample_np.shape, da.shape)]
            spaces = codec_spaces(sample_da, space_args(opts), fso_range, chunk_shapes=chunks, nonfinite=scan.nonfinite)
        owner = sweep.comm.bcast(acquire_lock(lock) if rank == 0 else None, root=0)
        if owner:
            raise click.ClickException(f"another sweep of {var} is writing into {opts.where_to_write} ({owner}); "
                                       f"remove {lock} if it is not running")
    except click.ClickException as e:  # decided from the shared sample, scan and lock: the same on all ranks
        if opts.field_to_compress is not None:
            raise
        if rank == 0:
            click.echo(f"[var] skipping {var}: {e.message}")
        return
    try:
        _sweep_field(da, var, opts, sweep, n_vars, t0, sample_np, sample_da, scan, spaces, q99_abs, bounds)
    finally:
        if rank == 0:
            release_lock(lock)


def _sweep_field(da, var, opts, sweep: SweepContext, n_vars, t0, sample_np, sample_da, scan, spaces, q99_abs, bounds):
    # Collective: the search itself, under the field's lock.
    rank = sweep.rank
    done, crashes, digest = sweep_recorded_rows(var, sample_np, q99_abs, scan.range, bounds, opts, sweep)
    if opts.with_ebcc and rank == 0:
        tile, reason = utils.ebcc_tile(sample_da)
        click.echo(f"[ebcc] {var}: " + (f"tile {tile[0]}x{tile[1]}" if tile else f"not applicable ({reason})"))
    config_space = sweep_config_space(*spaces, opts.max_evals, rank, sample_np.dtype, scan.nonfinite == 0)
    keys = [utils.pipeline_json(*cfg) for cfg in config_space]
    culprits = {k for k, c in crashes.items() if c.get("alone")}
    pending = [i for i, key in enumerate(keys) if key not in done and key not in culprits]
    isolated = [i for i in pending if keys[i] in crashes]
    normal = [i for i in pending if keys[i] not in crashes]
    crashed = sorted(crashes[k]["name"] for k in culprits if k in set(keys))
    if rank == 0:
        if not config_space:
            click.echo(f"[sweep] {var}: the codec space is empty for these classes and this dtype.")
        if done:
            click.echo(f"[resume] {sum(k in done for k in keys)} of {len(keys)} combo(s) of '{var}' are already "
                       f"recorded; skipping those.")
        if crashed:
            click.echo(f"[resume] {var}: leaving out {len(crashed)} combo(s) that killed their rank when evaluated "
                       f"alone: {'; '.join(crashed)}")
        sweep_banner(var, spaces, config_space, len(pending), sample_np, opts, sweep, n_vars)
        config_space_table(config_space).to_csv(os.path.join(opts.where_to_write, f"config_space_{var}.csv"),
                                                index=False)

    evaluate_one, gate = sweep_evaluators(var, sample_np, sample_da, q99_abs, bounds, opts, sweep)
    failures = sweep_run_rank(config_space, normal, var, opts, sweep, evaluate_one, gate)
    failures += sweep_run_alone(config_space, isolated, var, opts, sweep, evaluate_one, gate)
    total_failures = sweep_report_failures(failures, var, sweep)
    sweep.comm.Barrier()  # every rank's CSV is complete before rank 0 consolidates
    if rank != 0:
        return

    click.echo("[sweep] complete. Writing results...")
    best, n_passed, n_rows, parquet_path = sweep_select_best(
        opts.where_to_write, var, gate, set(keys), lambda df: missing_metrics(df, q99_abs, opts, sweep.thresholds))
    if best is not None:
        click.echo(f"best pipeline: {best['name']}\nCompression Ratio: {best['ratio']:.3f} | "
                   f"Relative L1 Error: {best['l1_rel']:.3e} | Euclidean Distance: {best['eucd']:.3e}")
    else:
        click.echo("[sweep] no combos passed the threshold filter.")
    manifest = sweep_manifest(var, opts, sweep, num_combos=len(config_space), n_rows=n_rows, n_passed=n_passed,
                              total_failures=(total_failures or 0) + len(crashed), crashed=crashed,
                              seconds=time.perf_counter() - t0, parquet_path=parquet_path, best=best,
                              q99_abs=q99_abs, digest=digest)
    write_json(os.path.join(opts.where_to_write, f"manifest_{var}.json"), manifest, "sweep")


def end_job(exc: BaseException, comm) -> None:
    """End a multi-rank job through MPI Abort, since a rank leaving on its own would hang the others in a
    collective.  Rank 0 reports first; another rank reports only if the job still runs two seconds later."""
    if isinstance(exc, KeyboardInterrupt):
        code = 130
    elif isinstance(exc, click.ClickException):
        code = exc.exit_code
    elif isinstance(exc, SystemExit):
        code = exc.code if isinstance(exc.code, int) else 1
    else:
        code = 1
    if comm.Get_rank() != 0:
        time.sleep(2)
    if isinstance(exc, click.ClickException):
        click.echo(f"Error: {exc.format_message()}", err=True)
    elif not isinstance(exc, (SystemExit, KeyboardInterrupt)):
        traceback.print_exception(exc)
    sys.stdout.flush()
    sys.stderr.flush()
    comm.Abort(code)


def sweep_dataset(opts) -> None:
    """evaluate_combos: sweep the selected variables of the dataset one after the other; collective."""
    comm = MPI.COMM_WORLD
    try:
        sweep = sweep_setup(opts)
        # array.chunk-size must be set before any open() with chunks="auto"; synchronous: one core per rank.
        with dask.config.set({"array.chunk-size": "512MiB", "scheduler": "synchronous"}):
            ds = utils.open_dataset(opts.dataset_file, opts.field_to_compress, rank=sweep.rank)
            variables = sweep_variables(ds, opts.field_to_compress, sweep.rank)
            for var in variables:
                sweep_variable(ds[var], var, opts, sweep, n_vars=len(variables))
    except BaseException as e:
        if comm.Get_size() == 1:
            raise
        end_job(e, comm)


# =============================================================================
# 6. COMPRESS (compress, merge_compressed_fields)
# =============================================================================

def manifest_superseded(where_to_write, var: str, manifest: dict):
    """Why manifest_{var}.json does not come from the last sweep of `var`, or None: its sweep_state_digest
    must match sweep_state_{var}.json, which every sweep of the field rewrites when it starts."""
    state = read_json(Path(where_to_write) / f"sweep_state_{var}.json", "compress")
    if state is None:
        return f"sweep_state_{var}.json is missing; re-run evaluate_combos --resume"
    if manifest.get("sweep_state_digest") != state_digest(state):
        return (f"manifest_{var}.json does not match sweep_state_{var}.json (a later sweep of {var} has not "
                f"finished, or an older dc_toolkit wrote it); re-run evaluate_combos --resume")
    return None


def same_file(a, b) -> bool:
    try:
        return os.path.samefile(a, b)
    except (OSError, TypeError):
        return os.path.abspath(str(a)) == os.path.abspath(str(b))


def compress_candidates(opts):
    """(candidates, manifests, dropped).  One candidate {var, name, pipeline, ratio, source} per field: with
    --pipeline, that pipeline for every --vars field; else the best of manifest_{var}.json, or, with
    --stock-codecs-only when that best needs dc_toolkit to be read, the best stock row of the same sweep's
    results_{var}.parquet.  A manifest must parse and match sweep_state_{var}.json.  `manifests` holds the
    usable manifests; `dropped` maps each field without a usable pipeline to why."""
    wtw = Path(opts.where_to_write)
    wanted = {v.strip() for v in opts.vars_filter.split(",") if v.strip()} if opts.vars_filter is not None else None
    if wanted is not None and not wanted:
        raise click.ClickException("--vars names no field.")
    manifests, dropped, sweep_env = {}, {}, None
    for mpath in sorted(wtw.glob("manifest_*.json")):
        var = mpath.stem.removeprefix("manifest_")
        if wanted and var not in wanted:
            continue
        try:
            m = json.loads(mpath.read_text())
        except Exception as e:
            dropped[var] = f"cannot parse {mpath.name} ({e})"
            continue
        stale = manifest_superseded(wtw, var, m)
        if stale:
            dropped[var] = stale
            continue
        if not same_file(m.get("dataset_file"), opts.dataset_file):
            click.echo(f"[compress] WARNING: {mpath.name} comes from a sweep of {m.get('dataset_file')}, not of "
                       f"{opts.dataset_file}.")
        manifests[var] = m
        sweep_env = sweep_env or m.get("env")
    if sweep_env:
        warn_env_drift(sweep_env, "these manifests")

    candidates = []
    if opts.pipeline is not None:
        if not wanted:
            raise click.ClickException("--pipeline needs --vars to name the field(s) it applies to.")
        pipeline = parse_pipeline_arg(opts.pipeline)
        if opts.stock_codecs_only and not utils.pipeline_is_stock(pipeline):
            raise click.ClickException("--stock-codecs-only refuses this --pipeline: one of its codecs needs "
                                       "dc_toolkit's zarr.codecs entry point to be read.")
        name = utils.pipeline_name(*pipeline_codecs(pipeline, "--pipeline"))
        candidates = [{"var": v, "name": name, "pipeline": pipeline, "ratio": None, "source": "--pipeline"}
                      for v in sorted(wanted - set(dropped))]
    else:
        for var, m in manifests.items():
            best = m.get("best")
            if best is None:
                dropped[var] = "the sweep kept no combo (manifest has no best)"
            elif not isinstance(best, dict) or not isinstance(best.get("pipeline"), dict):
                dropped[var] = "the manifest best has no pipeline; re-run evaluate_combos"
            elif not opts.stock_codecs_only or utils.pipeline_is_stock(best["pipeline"]):
                candidates.append({"var": var, "name": best.get("name", "?"), "pipeline": best["pipeline"],
                                   "ratio": best.get("ratio"), "source": f"manifest_{var}.json"})
            else:
                ppath = wtw / f"results_{var}.parquet"
                try:
                    row = best_kept_row(pd.read_parquet(ppath), stock_only=True)
                except Exception as e:
                    dropped[var] = f"the manifest best needs dc_toolkit to be read and {ppath.name} is unreadable ({e})"
                    continue
                if row is None:
                    dropped[var] = f"the manifest best needs dc_toolkit to be read and {ppath.name} keeps no stock row"
                    continue
                click.echo(f"[compress] {var}: the manifest best {best.get('name', '?')} needs dc_toolkit's codec "
                           f"entry point to be read; --stock-codecs-only takes the best stock row of {ppath.name}.")
                candidates.append({"var": var, "name": str(row["name"]), "pipeline": json.loads(row["pipeline"]),
                                   "ratio": float(row["ratio"]), "source": f"{ppath.name} (stock codecs only)"})
        swept = {p.stem.removeprefix("results_") for p in wtw.glob("results_*.parquet")}
        for var in sorted((wanted or swept) - set(manifests) - set(dropped)):
            dropped[var] = (f"results_{var}.parquet has no manifest_{var}.json (the sweep did not finish); re-run "
                            f"evaluate_combos --resume" if var in swept else f"no manifest_{var}.json in WHERE_TO_WRITE")
    for var, reason in dropped.items():
        click.echo(f"[compress] ERROR: {var} has no usable pipeline: {reason}.")
    if candidates:
        click.echo(f"[compress] will compress {len(candidates)} field(s): {', '.join(c['var'] for c in candidates)}")
    return candidates, manifests, dropped


def source_identity(path) -> dict:
    st = os.stat(path)
    return {"path": os.path.realpath(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def compress_plan(da, var: str, cand: dict, manifest, opts):
    """(combo, layout, request).  `request` identifies the array this run would write: source file, field,
    pipeline, layout and the verify gate's thresholds and bounds; compress records it on the array and
    --skip-existing compares it."""
    if da.ndim == 0:
        raise click.ClickException(f"{var} is a scalar; compress writes arrays with at least one dim")
    combo = pipeline_codecs(cand["pipeline"], var)
    geometry, sources = chunk_geometry(opts, manifest)
    click.echo(f"[chunks] {var}: " + ", ".join(f"{k}={v} ({sources[k]})" for k, v in geometry.items()))
    layout = store_layout(da, combo, geometry, opts)
    m = manifest or {}
    overrides = {k: getattr(opts, f"{k}_threshold", None) for k in ("l1", "l2", "linf", "bias")}
    request = json_safe({
        "source": source_identity(opts.dataset_file), "var": var, "pipeline": cand["pipeline"],
        "inner_chunks": layout[0], "shards": layout[1],
        "thresholds": verify_thresholds(manifest, overrides) if opts.verify else None,
        "bounds": gate_bounds(m.get("phys_min"), m.get("phys_max"), m.get("phys_slack")) if opts.verify else None,
        "q99_abs": m.get("q99_abs") if opts.verify else None})
    return combo, layout, request


def stored_mismatch(merged_path: str, var: str, request: dict, predicted, opts):
    """Why the array `var` in the store is not what this run would write (compress_plan's request, and a
    ratio within --cr-drift-tol of `predicted` under --cr-drift-gate), or None."""
    try:
        attrs = json.loads((Path(merged_path) / var / "zarr.json").read_text()).get("attributes") or {}
    except (OSError, ValueError):
        return "its zarr.json is unreadable"
    try:
        record = json.loads(attrs.get("dc_toolkit") or "{}")
    except (TypeError, ValueError):
        record = {}
    stored = record.get("request") if isinstance(record, dict) else None
    if not isinstance(stored, dict):
        return "it records no dc_toolkit request"
    if opts.stock_codecs_only and not array_is_stock(merged_path, var):
        return "it needs dc_toolkit to be read"
    if opts.verify and opts.verify_gate and record.get("verify_gate") == "fail-advisory":
        return "it failed the verify gate (written with --no-verify-gate)"
    _, drift, direction = evaluate_cr_drift(record.get("ratio"), predicted, opts.cr_drift_tol)
    if opts.cr_drift_gate and direction == "under":
        return f"its ratio falls {-drift:.0%} short of the sweep's"
    changed = [k for k in request if k not in ("thresholds", "bounds", "q99_abs") and stored.get(k) != request[k]]
    if opts.verify and any(stored.get(k) != request[k] for k in ("thresholds", "bounds", "q99_abs")):
        changed.append("verify gate")
    return f"its {', '.join(changed)} changed" if changed else None


def compress_one(da, var: str, cand: dict, manifest, merged_path: str, opts, combo, layout, request) -> dict:
    """Persist one field and promote it into the merged store only if the verify, FixedScaleOffset range and
    CR-drift gates pass (else RuntimeError; an existing array stays).  Returns the batch_manifest.json entry."""
    unit = layout[1] if layout[1] is not None else layout[0]
    da = da.chunk(dict(zip(da.dims, read_blocks(da.shape, da.dtype, unit))))
    itemsize = int(da.dtype.itemsize)
    tasks = write_concurrency(itemsize * int(np.prod(da.data.chunksize)), itemsize * int(np.prod(layout[0])),
                              int(np.prod(da.data.numblocks)), opts, var)
    with dask.config.set(num_workers=tasks):
        validate_pipeline(combo, da, var)
    m = manifest or {}
    q99_abs = m.get("q99_abs") if opts.verify else None
    bounds = gate_bounds(m.get("phys_min"), m.get("phys_max"), m.get("phys_slack")) if opts.verify else None
    try:
        out = persist_field(da, var, merged_path, combo, opts, layout, q99_abs, bounds, tasks)
        summary = f"{var}: {cand['name']} -> ratio={out['ratio']:.3f}"
        if opts.verify:
            summary += f" L1_rel={out['errors']['Relative_Error_L1']:.3e} eucd={out['eucd']:.3e}"
        click.echo(f"[compress] {summary}  ({out['seconds']:.1f}s)")

        verify_status, detail = verify_gate_verdict(var, out, manifest, opts)
        if verify_status == "fail":
            raise RuntimeError(detail)
        problem = fso_range_problem(combo, out["errors"])
        if problem:
            raise RuntimeError(f"{var}: {problem}")

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
        entry = {
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
        staging = zarr.storage.LocalStore(str(staging_path(merged_path)), read_only=False)
        try:
            # A JSON string: netCDF, which from_zarr_to_netcdf writes, has no nested attributes.
            zarr.open_array(store=staging, path=var, mode="r+").update_attributes({"dc_toolkit": json.dumps(json_safe({
                "request": request, "verify_gate": verify_status, "ratio": out["ratio"], "errors": entry["errors"],
                "written_by": provenance(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}), sort_keys=True)})
        finally:
            close_store(staging)
        promote_staged(merged_path, var)
    except BaseException:  # also Ctrl-C and a memory guard's SystemExit
        remove_staged(merged_path, var)
        raise
    return entry


def compress_fields(opts) -> None:
    """compress: write the candidate fields into the merged store, each staged and gated first, then
    consolidate it and record batch_manifest.json; exits with status 1 when any field failed."""
    click.echo(version_banner("compress"))
    single_process_setup(opts)
    os.makedirs(opts.where_to_write, exist_ok=True)
    candidates, manifests, dropped = compress_candidates(opts)
    if not candidates and not dropped:
        raise click.ClickException("no variables to compress.  Did evaluate_combos run against the same directory?")
    merged_path = merged_store_path(opts.where_to_write, opts.dataset_file)
    if Path(merged_path).resolve() == Path(opts.dataset_file).resolve():
        raise click.ClickException(f"the output store {merged_path} is the input dataset; pick another "
                                   f"WHERE_TO_WRITE.")
    lock = Path(f"{merged_path}.lock")
    owner = acquire_lock(lock)
    if owner:
        raise click.ClickException(f"another compress is writing {merged_path} ({owner}); remove {lock} if it "
                                   f"is not running")
    try:
        any_error = _compress_locked(opts, candidates, manifests, dropped, merged_path)
    finally:
        release_lock(lock)
    if any_error:
        sys.exit(1)


def _compress_locked(opts, candidates, manifests, dropped, merged_path) -> bool:
    if Path(opts.dataset_file).suffix.lower() == ".nc":
        import netCDF4
        # A block reads a slice of many large HDF5 chunks; with a cache, each slice would read its whole chunk.
        netCDF4.set_chunk_cache(0, 1, 0.0)
    ds = utils.open_dataset(opts.dataset_file, chunks=None)  # compress_one reads it in whole write units
    remove_staged(merged_path)
    existing = existing_arrays(merged_path)

    results = {var: {"status": "no-pipeline", "reason": reason} for var, reason in dropped.items()}
    any_error, stopped_at = bool(dropped), None
    with dask.config.set(scheduler="threads", num_workers=opts.threads), \
            zarr.config.set({"threading.max_workers": opts.threads}):
        for i, cand in enumerate(candidates, start=1):
            var = cand["var"]
            click.echo(f"\n[compress] ({i}/{len(candidates)}) {var} from {cand['source']}: {cand['name']}")
            if var not in ds.data_vars:
                any_error = True
                click.echo(f"[compress] ERROR: variable '{var}' not in dataset; skipping.")
                results[var] = {"status": "missing-from-dataset"}
                if not opts.continue_on_error:
                    click.echo("[compress] stopping at the first failure (--no-continue-on-error).")
                    stopped_at = i
                    break
                continue
            try:
                combo, layout, request = compress_plan(ds[var], var, cand, manifests.get(var), opts)
                if opts.skip_existing and var in existing:
                    reason = stored_mismatch(merged_path, var, request, cand.get("ratio"), opts)
                    if reason is None:
                        click.echo(f"[compress] {var} already in {merged_path} as requested; skipping.")
                        results[var] = {"status": "skipped-existing"}
                        continue
                    click.echo(f"[compress] {var} in {merged_path}: {reason}; rewriting it.")
                results[var] = compress_one(ds[var], var, cand, manifests.get(var), merged_path, opts,
                                            combo, layout, request)
            except (Exception, SystemExit) as e:  # SystemExit: a memory guard refused the write
                any_error = True
                message = (e.message if isinstance(e, click.ClickException)
                           else "refused by a guard (see above)" if isinstance(e, SystemExit) else repr(e))
                click.echo(f"[compress] ERROR on {var}: {message}")
                results[var] = {"status": "error", "error": message}
                if not opts.continue_on_error:
                    click.echo("[compress] stopping at the first failure (--no-continue-on-error).")
                    stopped_at = i
                    break

    for cand in candidates[stopped_at:] if stopped_at else []:
        results.setdefault(cand["var"], {"status": "not-attempted", "reason": "the run stopped at an earlier failure"})
    if opts.stock_codecs_only:
        left = sorted(v for v in existing_arrays(merged_path) if not array_is_stock(merged_path, v))
        if left:
            any_error = True
            click.echo(f"[compress] ERROR: {merged_path} still holds array(s) that need dc_toolkit to be read: "
                       f"{', '.join(left)}.")
    if Path(merged_path).is_dir():
        if opts.consolidate:
            names = consolidate_store(merged_path)
            click.echo(f"[compress] consolidated metadata on {merged_path} ({len(names)} array(s): {', '.join(names)})")
        else:  # a listing from an earlier run would describe this run's fields wrongly
            try:
                drop_consolidated_metadata(merged_path)
                click.echo("[compress] --no-consolidate: the store has no consolidated metadata; readers scan the "
                           "arrays until the next consolidation (dc_toolkit merge_compressed_fields DATASET "
                           "WHERE_TO_WRITE).")
            except Exception as e:
                any_error = True
                click.echo(f"[compress] ERROR: cannot drop the consolidated metadata of {merged_path}, which no "
                           f"longer describes the store: {e!r}")
    write_json(os.path.join(opts.where_to_write, "batch_manifest.json"), {
        "command": "compress", "dataset_file": os.path.abspath(opts.dataset_file),
        "where_to_write": os.fspath(opts.where_to_write), "merged_store": merged_path,
        "results": results, "any_error": any_error, "env": env_versions(), "provenance": provenance(),
    }, "compress")
    return any_error


def consolidate_merged_store(dataset_file, compressed_files_location) -> None:
    """merge_compressed_fields: consolidate the metadata of the store compress wrote for
    `dataset_file` under `compressed_files_location`."""
    merged_path = merged_store_path(compressed_files_location, dataset_file)
    if not Path(merged_path).is_dir():
        raise click.ClickException(f"store not found: {merged_path}.  Did compress run with the same directory?")
    lock = Path(f"{merged_path}.lock")
    owner = acquire_lock(lock)
    if owner:
        raise click.ClickException(f"a compress is writing {merged_path} ({owner}); remove {lock} if it is not "
                                   f"running")
    try:
        names = consolidate_store(merged_path)
    finally:
        release_lock(lock)
    click.echo(f"[merge] consolidated metadata on {merged_path} ({len(names)} array(s): {', '.join(names)})")


# =============================================================================
# 7. STORE INSPECTION & FORMAT CONVERSION
# =============================================================================

def inspect_store(zarr_path, head: int) -> None:
    """open_zarr_and_inspect: print the group tree, each array's metadata and its first `head` elements
    per dim."""
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


def perform_clustering(parquet_file, l_error: str) -> None:
    """perform_clustering: the elbow and silhouette plot of the kept rows of a results parquet."""
    df = load_results(parquet_file)
    if len(df) < 4:
        click.echo(f"[perform_clustering] only {len(df)} finite passing combo(s) in {Path(parquet_file).name}; "
                   f"need >= 4 to cluster.  Nothing to plot.")
        return
    elbow_silhouette_plot(df, l_error)


def analyze_clustering(parquet_file) -> None:
    """analyze_clustering: the interactive clustering figure of the kept rows, opened in the browser."""
    import plotly.io as pio

    df = load_results(parquet_file)
    if len(df) == 0:
        click.echo(f"[analyze_clustering] no finite passing combos in {Path(parquet_file).name}; nothing to plot.")
        return
    pio.renderers.default = "browser"
    clustering_figure(df).show()


def plot_compression_errors(opts) -> None:
    """plot_compression_errors: the error grid of one (lat, lon) field under one pipeline, saved as
    {field}_compression_errors.pdf in opts.where_to_write."""
    field = opts.field_to_compress
    os.makedirs(opts.where_to_write, exist_ok=True)
    da = utils.open_dataset(opts.dataset_file, field)[field].squeeze()
    click.echo(f"Squeezed (lat, lon) field_to_compress.nbytes = {utils.hsize(da.nbytes)}")
    if not utils.is_lat_lon(da):
        raise click.ClickException(f"Field {field} must have dimensions (lat, lon); it has {da.dims}.")
    if da.nbytes / 2**30 > 2.5:
        raise click.ClickException(f"Field {field} is too large ({utils.hsize(da.nbytes)}); max 2.5 GiB.")

    combo = plot_pipeline(field, opts.pipeline, opts.manifest_dir or opts.where_to_write)
    validate_pipeline(combo, da, field)
    click.echo(f"pipeline: {utils.pipeline_name(*combo)}")
    da, panels = error_plot_panels(da, field, combo)
    save_error_plot(field, da, panels, os.path.join(opts.where_to_write, f"{field}_compression_errors.pdf"))


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
