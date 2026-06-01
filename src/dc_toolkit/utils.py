# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import os
import math
import warnings
import click
import humanize
import threading
import asyncio
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import dask
import dask.array
import pandas as pd
import xarray as xr
import zarr
import numcodecs
from zarr.codecs import numcodecs as zarrcodecs_nc  # zarr-native codecs (replaces deprecated numcodecs.zarr3)
import zfpy
from mpi4py import MPI
import time
from collections import defaultdict
from itertools import product
import atexit
import re


class CombinationProducedNonFiniteError(Exception):
    """
    Raised when a codec combination's decoded sample contains NaN or
    +/-inf values, signalling that the (compressor, filter, serializer)
    triple is unsuitable for this field's value range.

    Caught by the per-combo try/except in `cli.evaluate_combos`, which
    routes it to `failures_<var>_rank<n>.csv` with a clear reason
    string.  Without this exception, the float64 cast in the metrics
    loop would raise a `RuntimeWarning: invalid value encountered in
    cast` per non-finite chunk - the combo would still be filtered out
    by the L1 threshold downstream, but the warning floods the SLURM
    log (191 occurrences in production job 843234) and the failure
    reason wouldn't be recorded explicitly.
    """
    pass


class SampleTooLargeError(Exception):
    """
    Raised by `build_representative_sample` when a field's irreducible
    spatial footprint exceeds the requested size limit.

    "Irreducible" means: after striding every available time-like and
    vertical-like dim down to a single index, a single horizontal slab
    of the field still exceeds the byte limit.  Subsampling horizontal
    dims is not permitted because it changes the spatial structure
    codecs exploit during compression scoring.

    Callers should surface this with a clear remedy (raise the limit,
    drop --threads-per-rank, or move to a larger node).  The exception
    carries the irreducible byte count and the limit it failed for so
    the caller can format a precise message.
    """
    def __init__(self, message, irreducible_bytes=None, size_limit_bytes=None,
                 dims=None, spatial_dims=None):
        super().__init__(message)
        self.irreducible_bytes = irreducible_bytes
        self.size_limit_bytes  = size_limit_bytes
        self.dims              = dims
        self.spatial_dims      = spatial_dims


# =============================================================================
# SIZE PARSING
# =============================================================================

_SIZE_UNITS = {
    "B":   1,
    "KB":  10**3,  "MB":  10**6,  "GB":  10**9,  "TB":  10**12,
    "KIB": 2**10,  "MIB": 2**20,  "GIB": 2**30,  "TIB": 2**40,
}


def parse_size(size_str: str) -> int:
    """Parse human-readable sizes like '5GB', '500MiB', '10GiB' to bytes."""
    if isinstance(size_str, (int, float)):
        return int(size_str)
    s = str(size_str).strip().upper().replace(" ", "")
    for unit in sorted(_SIZE_UNITS.keys(), key=lambda u: -len(u)):
        if s.endswith(unit):
            return int(float(s[: -len(unit)]) * _SIZE_UNITS[unit])
    # No unit -> treat as bytes
    return int(float(s))


# =============================================================================
# DATASET OPENING
# =============================================================================

def open_zarr_memstore():
    return zarr.storage.MemoryStore()


def open_zarr_localstore(path: str, read_only: bool = True):
    """Open a zarr v3 LocalStore.  Returns (group, store).

    IMPORTANT: do NOT close the store before you finish reading through the
    returned group.  Keep both alive for the lifetime of any dask graph
    that derives from it.
    """
    store = zarr.storage.LocalStore(path, read_only=read_only)
    return zarr.open_group(store, mode="r" if read_only else "a"), store


def open_dataset(
    dataset_file: str,
    field_to_compress: Optional[str] = None,
    rank: int = 0,
):
    """
    Open a dataset lazily with a dask backend.

    Supports: .nc, .grib, .zarr (pure LocalStore).
    .zarr.zip is NO LONGER SUPPORTED.

    Note: the store handle is kept alive inside the xarray Dataset.  Do not
    close it externally.  Zarr v3 LocalStore doesn't require explicit close
    for reads, but we let xarray own the lifecycle regardless.
    """
    p = Path(dataset_file)
    suffix = p.suffix.lower()

    if suffix == ".nc":
        ds = xr.open_dataset(dataset_file, chunks="auto")
    elif suffix == ".grib":
        ds = xr.open_dataset(
            dataset_file, chunks="auto", engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )
    elif suffix == ".zarr":
        # LocalStore: xarray keeps a reference through the dask graph.
        # consolidated=None -> auto-detect; uses consolidated metadata if the
        # store was processed by merge_compressed_fields, otherwise falls back
        # to a full metadata scan.
        ds = xr.open_zarr(dataset_file, chunks="auto", consolidated=None)
    else:
        if rank == 0:
            click.echo(
                f"Unsupported file format: {suffix}. "
                f"Only .nc / .grib / .zarr are supported "
                f"(.zarr.zip was removed in the refactor)."
            )
            click.echo("Aborting...")
        # Collective abort: sys.exit on rank 0 alone would hang siblings at
        # the next collective.  Abort(1) tears down the whole MPI world.
        MPI.COMM_WORLD.Abort(1)

    if field_to_compress is not None and field_to_compress not in ds.data_vars:
        if rank == 0:
            click.echo(f"Field {field_to_compress} not found in dataset.")
            click.echo(f"Available fields: {list(ds.data_vars.keys())}.")
            click.echo("Aborting...")
        MPI.COMM_WORLD.Abort(1)

    if rank == 0:
        click.echo(f"dataset.nbytes = {humanize.naturalsize(ds.nbytes, binary=True)}")
        if field_to_compress is not None:
            click.echo(
                f"{field_to_compress}.nbytes = "
                f"{humanize.naturalsize(ds[field_to_compress].nbytes, binary=True)}"
            )

    return ds


def is_lat_lon(da):
    dims = da.dims
    return (
        len(dims) == 2
        and re.search(r"lat", dims[0]) is not None
        and re.search(r"lon", dims[1]) is not None
    )


# =============================================================================
# REPRESENTATIVE SAMPLING  (replaces the old corner-slice strategy)
# =============================================================================

# Time-like dim name patterns; FALLBACK when CF metadata is unavailable.
# Primary detection uses CF-conventions attributes on the coord variable.
_TIME_LIKE_DIM_RE = re.compile(
    r'^(?:time|.*_time|t|step|forecast_reference_time|forecast_period|'
    r'ensemble|realization|member|reftime|valid_time|epoch)$',
    re.IGNORECASE,
)

# CF time-units pattern: "<unit> since <reference time>".
_CF_TIME_UNITS_RE = re.compile(r'^\s*\w+\s+since\s+', re.IGNORECASE)

# CF standard_name values that mark a time-related axis.
_CF_TIME_STANDARD_NAMES = frozenset({
    'time',
    'forecast_reference_time',
    'forecast_period',
})


def _is_time_like_coord(da: xr.DataArray, dim_name: str) -> bool:
    """Decide whether `dim_name` of `da` is a time axis.

    Order of evidence (most authoritative first):
      1. Corresponding coord variable's `units` attr starts with a CF
         time-units pattern like 'days since', 'hours since' — this is
         the canonical CF time signature; no other coordinate type uses
         this format.
      2. `standard_name` is one of the CF time-axis standard names.
      3. `axis` attr is 'T' (CF specifies four axis-type codes:
         X, Y, Z, T).
      4. `calendar` attr is present.  Only time vars have calendars,
         so this is a strong positive signal even when other CF attrs
         are missing.
      5. Fallback: dim NAME matches `_TIME_LIKE_DIM_RE`.

    Both `coord.attrs` and `coord.encoding` are checked because
    xarray's `decode_times=True` (the default) moves CF time attributes
    from `attrs` to `encoding` after parsing — we have to look in both
    to be robust to the caller's open_dataset choices.

    Dims without a corresponding coord variable (which happens in
    netCDF when a dim isn't backed by a same-name 1-D variable) skip
    straight to the name-regex fallback.

    Returns True on the first positive signal.
    """
    coord = da.coords.get(dim_name)
    if coord is not None:
        # Merge attrs and encoding; encoding wins on conflicts because
        # decoded time vars store the original units/calendar in
        # encoding rather than attrs.
        merged = {**dict(coord.attrs), **dict(coord.encoding)}

        units = merged.get('units')
        if isinstance(units, str) and _CF_TIME_UNITS_RE.match(units):
            return True

        std_name = merged.get('standard_name')
        if isinstance(std_name, str) and std_name in _CF_TIME_STANDARD_NAMES:
            return True

        if merged.get('axis') == 'T':
            return True

        if 'calendar' in merged:
            return True

    return bool(_TIME_LIKE_DIM_RE.match(dim_name))


def _find_time_like_dim(da: xr.DataArray) -> Tuple[Optional[int], Optional[str]]:
    """First time-like dim's (index, name) in `da.dims`, or (None, None)
    if no dim qualifies under either CF metadata or the name fallback."""
    for i, name in enumerate(da.dims):
        if _is_time_like_coord(da, name):
            return i, name
    return None, None


def _is_vertical_like_coord(da: xr.DataArray, dim_name: str) -> bool:
    """
    Classify a dim as vertical (level/height/depth-like).

    Order of evidence (most authoritative first), mirroring
    `_is_time_like_coord`:
      1. Coord's `axis` attr is 'Z' (CF standard for vertical).
      2. Coord's `standard_name` matches a vertical-axis CF name
         (height, altitude, depth, atmosphere_*_coordinate, ...).
      3. Coord's `positive` attr is set ('up' or 'down') — CF vertical
         marker that's allowed even without axis=Z.
      4. Name fallback via `_is_vertical_like_dim` (the existing name
         heuristic used by `_shrink_order`).
    """
    coord = da.coords.get(dim_name)
    if coord is not None:
        merged = {**dict(coord.attrs), **dict(coord.encoding)}
        if merged.get('axis') == 'Z':
            return True
        sn = merged.get('standard_name')
        if isinstance(sn, str) and sn.lower() in {
            'height', 'altitude', 'depth', 'air_pressure', 'pressure',
            'model_level_number', 'atmosphere_hybrid_sigma_pressure_coordinate',
            'atmosphere_hybrid_height_coordinate',
            'atmosphere_sigma_coordinate',
            'atmosphere_ln_pressure_coordinate',
            'atmosphere_sleve_coordinate',
        }:
            return True
        if 'positive' in merged and str(merged['positive']).lower() in ('up', 'down'):
            return True

    return _is_vertical_like_dim(dim_name)


def _classify_sample_dims(
    da: xr.DataArray,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Classify every dim of `da` into one of:
      - time-like (CF axis=T or time-units or name regex)
      - vertical-like (CF axis=Z or vertical CF standard_name or name regex)
      - spatial (everything else; preserved during sampling)

    Returns (time_dims, vertical_dims, spatial_dims), each as a list of
    (position-in-da.dims, name).  Time and vertical are STRIDE dims;
    spatial dims are preserved whole so codec scoring sees the real
    spatial structure of the field.

    Note: the time check runs first.  If a coord is both T- and Z-like
    (impossible under CF, but defensive), it's classified as time.
    """
    time_dims: List[Tuple[int, str]]     = []
    vertical_dims: List[Tuple[int, str]] = []
    spatial_dims: List[Tuple[int, str]]  = []
    for i, name in enumerate(da.dims):
        if _is_time_like_coord(da, name):
            time_dims.append((i, name))
        elif _is_vertical_like_coord(da, name):
            vertical_dims.append((i, name))
        else:
            spatial_dims.append((i, name))
    return time_dims, vertical_dims, spatial_dims


def build_representative_sample(
    da: xr.DataArray,
    size_limit_bytes: int,
    rank: int = 0,
    policy: str = "cascade",
    vertical_floor: int | None = None,
) -> xr.DataArray:
    """
    Return a subset of `da` that fits STRICTLY within `size_limit_bytes`,
    built to be representative of the full field.

    Strategy
    --------
    - If the whole field fits under the limit: return it unchanged.
    - Otherwise: classify dims into time / vertical / spatial.  Spatial
      dims (horizontal grid: lat, lon, cell, ncells, x, y, ...) are
      preserved whole so codecs still see the real spatial structure
      they exploit during scoring.  Time and vertical dims are
      stride-sampled.
    - The slice budget is divided between the time group and the vertical
      group according to `policy`:
        * "cascade" (default): drain the time axis first (temporal
          diversity is what codec scoring cares about for time-varying
          fields), keeping a budget-aware minimum of vertical levels
          (`vertical_floor`).  Adjacent model levels are highly
          correlated, so spending budget on temporal spread captures more
          of the variety that affects codec ranking.
        * "balanced": the legacy log-space split, giving each stride axis
          ~budget^(1/n_dims) indices (equal treatment).
      For single-axis fields (no vertical, or no time) the two policies are
      identical.  For under-budget fields neither runs (full field returned).
    - Within each strided dim, evenly-spaced indices are picked via
      `np.linspace`, mirroring the single-dim behaviour:
      deterministic, edge-inclusive, reproducible across
      `evaluate_combos` and `compress_with_optimal`.
    - If a field's irreducible spatial footprint (1 element along every
      stride dim) exceeds the limit, this raises `SampleTooLargeError`
      rather than silently violating the budget.  Pre-patch the
      function used `max(1, …)` and accepted the budget violation; that
      under-budgeted memory model is what caused the production OOMs
      on R02B10 out_15 (300 GiB fields, single time slice = 37.5 GiB
      against a 5 GB budget).

    Memory contract
    ---------------
    The caller is entitled to assume `sampled.nbytes <= size_limit_bytes`
    on successful return.  This is the foundation of the per-rank steady
    estimate in `cli._per_rank_steady_estimate_bytes` and the cgroup
    headroom check.

    Errors
    ------
    SampleTooLargeError: irreducible spatial footprint > size_limit_bytes.
      Carries `irreducible_bytes`, `size_limit_bytes`, `dims`,
      `spatial_dims` for caller-side message formatting.
    """
    nbytes = int(da.dtype.itemsize) * int(np.prod(da.shape))
    if nbytes <= size_limit_bytes:
        if rank == 0:
            click.echo(
                f"[sample] field fits under limit "
                f"({humanize.naturalsize(nbytes, binary=True)} "
                f"<= {humanize.naturalsize(size_limit_bytes, binary=True)}); "
                f"evaluating on full field."
            )
        return da

    time_dims, vertical_dims, spatial_dims = _classify_sample_dims(da)
    # Stride priority: time first (captures temporal variation, the most
    # informative axis for codec scoring across forecast steps), then vertical.
    stride_dims: List[Tuple[int, str]] = list(time_dims) + list(vertical_dims)

    if not stride_dims:
        # Nothing safe to thin — return whole field with a warning.  In
        # practice the variables that hit this branch are small bookkeeping
        # arrays (CF coord-bounds, SCRIP remap weights) where being a few
        # GiB over isn't a problem.  A LARGE variable without any time or
        # vertical axis would deserve human review; surface that via the
        # warning so it doesn't pass silently.
        if rank == 0:
            click.echo(
                f"[sample] WARNING: variable '{da.name}' has dims {da.dims} "
                f"with no time-like or vertical axis; cannot stride-sample.  "
                f"Returning the full field "
                f"({humanize.naturalsize(nbytes, binary=True)}), which "
                f"exceeds the "
                f"{humanize.naturalsize(size_limit_bytes, binary=True)} cap.  "
                f"Normal for small bookkeeping arrays; investigate if the "
                f"variable is large and downstream OOMs."
            )
        return da

    # Irreducible footprint = bytes at 1 index along every stride dim.
    spatial_sizes = [da.sizes[d] for _, d in spatial_dims]
    irreducible_bytes = int(da.dtype.itemsize) * int(
        np.prod(spatial_sizes) if spatial_sizes else 1
    )

    if irreducible_bytes > size_limit_bytes:
        spatial_names = [d for _, d in spatial_dims]
        msg = (
            f"variable '{da.name}' has irreducible spatial footprint "
            f"{humanize.naturalsize(irreducible_bytes, binary=True)} "
            f"(spatial dims {spatial_names}) which exceeds the "
            f"{humanize.naturalsize(size_limit_bytes, binary=True)} budget.  "
            f"Spatial dims must be preserved for codec representativeness.  "
            f"Remedies (any one):\n"
            f"  - raise --eval-data-size-limit (memory permitting)\n"
            f"  - reduce --threads-per-rank to free memory for a larger sample\n"
            f"  - request more RAM (#SBATCH --mem=0) or a larger node"
        )
        if rank == 0:
            click.echo(f"[sample] FATAL: {msg}")
        raise SampleTooLargeError(
            msg,
            irreducible_bytes=irreducible_bytes,
            size_limit_bytes=int(size_limit_bytes),
            dims=tuple(da.dims),
            spatial_dims=tuple(spatial_names),
        )

    # Slice budget: how many (time x level) index-combinations fit once the
    # spatial dims are preserved whole.
    max_product = float(size_limit_bytes) / float(irreducible_bytes)

    plan = _allocate_stride_plan(
        da, time_dims, vertical_dims, max_product,
        policy=policy, vertical_floor=vertical_floor,
    )

    # Build isel dict; only set indices for dims we actually thinned.  Index
    # selection is evenly-spaced (linspace), edge-inclusive, for EVERY thinned
    # stride dim (time and vertical alike) — never consecutive.
    stride_dims = list(time_dims) + list(vertical_dims)
    indices_isel: dict = {}
    for _, name in stride_dims:
        size = int(da.sizes[name])
        n_keep = plan[name]
        if n_keep < size:
            idx = np.linspace(0, size - 1, num=n_keep, dtype=int)
            indices_isel[name] = np.unique(idx).tolist()

    sampled = da.isel(indices_isel) if indices_isel else da

    if rank == 0:
        # Compact plan summary, in original dim order:
        plan_parts = []
        for _, name in stride_dims:
            plan_parts.append(f"{name}={plan[name]}/{da.sizes[name]}")
        spatial_part = (
            f" | preserved spatial: {', '.join(d for _, d in spatial_dims)}"
            if spatial_dims else ""
        )
        click.echo(
            f"[sample] field is "
            f"{humanize.naturalsize(nbytes, binary=True)} > limit "
            f"{humanize.naturalsize(size_limit_bytes, binary=True)}; "
            f"policy={policy}; "
            f"strided {', '.join(plan_parts)}"
            f"{spatial_part} -> "
            f"{humanize.naturalsize(int(sampled.nbytes), binary=True)}."
        )

    return sampled


def _balanced_group_plan(dims_info, budget) -> dict:
    """
    Legacy log-space split of `budget` slices across the dims in `dims_info`
    (list of (pos, name, size)).  Sort ascending by size so small dims clamp
    early and pass freed budget upward.  Returns {name: n_keep}.
    """
    plan: dict = {}
    remaining = float(budget)
    remaining_dims = len(dims_info)
    for _, name, size in sorted(dims_info, key=lambda t: t[2]):
        target = remaining ** (1.0 / remaining_dims) if remaining_dims > 0 else 1.0
        n_keep = max(1, min(size, int(target)))
        plan[name] = n_keep
        remaining = remaining / max(1, n_keep)
        remaining_dims -= 1
    return plan


def _allocate_stride_plan(
    da, time_dims, vertical_dims, max_product, policy="cascade",
    vertical_floor=None,
) -> dict:
    """
    Decide how many indices to keep along each stride dim.

    Returns {dim_name: n_keep} covering every time and vertical dim.

    cascade (default)
    -----------------
    Drain the time group first; protect a budget-aware minimum of vertical
    levels.  With combined time size T and vertical size V and slice budget
    P = max_product:

        vfloor     = vertical_floor (default max(4, ceil(log2 V)), capped at V)
        base_level = min(vfloor, floor(sqrt(P)))   # never exceeds balanced's
                                                    # level count -> cascade
                                                    # keeps >= balanced timesteps
        time_keep  = min(T, floor(P / base_level))
        level_keep = min(V, floor(P / time_keep))   # spill time slack to levels

    The min(vfloor, floor(sqrt P)) cap is what prevents a tight-budget
    inversion: at small P it collapses to the balanced split; only when the
    budget is generous (sqrt(P) > vfloor) does cascade trade levels for
    timesteps.  Within the time group (or vertical group) the group's
    allocation is distributed across its member dims log-space.

    balanced
    --------
    The legacy behaviour: one log-space split across all stride dims jointly.
    """
    time_info = [(i, name, int(da.sizes[name])) for i, name in time_dims]
    vert_info = [(i, name, int(da.sizes[name])) for i, name in vertical_dims]

    if policy == "balanced":
        return _balanced_group_plan(time_info + vert_info, max_product)

    # --- cascade ---
    T = int(np.prod([s for _, _, s in time_info])) if time_info else 1
    V = int(np.prod([s for _, _, s in vert_info])) if vert_info else 1
    P = max(1.0, float(max_product))

    if vertical_floor is not None:
        vfloor = max(1, int(vertical_floor))
    else:
        # max(4, ceil(log2 V)): 10 lev -> 4, 60 -> 6, 137 -> 8.
        vfloor = max(4, int(math.ceil(math.log2(V)))) if V > 1 else 1
    vfloor = min(vfloor, V)

    sqrtP = int(math.floor(math.sqrt(P)))
    base_level = max(1, min(vfloor, sqrtP))
    time_keep = max(1, min(T, int(math.floor(P / base_level))))
    level_keep = max(1, min(V, int(math.floor(P / time_keep))))

    # Distribute each group's allocation across its member dims (log-space),
    # so multi-time or multi-vertical fields are handled gracefully.  Single
    # dim per group is the common case and reduces to {name: keep}.
    plan: dict = {}
    plan.update(_distribute_group(time_info, time_keep))
    plan.update(_distribute_group(vert_info, level_keep))
    return plan


def _distribute_group(dims_info, group_keep) -> dict:
    """Distribute `group_keep` total indices across the dims in a group.

    One dim -> {name: min(size, group_keep)}.  Multiple dims -> log-space
    split (reusing the balanced splitter on the group's own budget).
    """
    if not dims_info:
        return {}
    if len(dims_info) == 1:
        _, name, size = dims_info[0]
        return {name: max(1, min(size, int(group_keep)))}
    return _balanced_group_plan(dims_info, group_keep)


# =============================================================================
# CHUNK & SHARD SIZING
# =============================================================================

# -----------------------------------------------------------------------------
# Vertical-dim recognition for hiopy-style shrink order.
# -----------------------------------------------------------------------------
_VERTICAL_DIM_NAMES = {
    "lev", "level", "levels", "plev", "plevs", "pressure", "pressure_level",
    "height", "altitude", "alt", "depth", "z",
    "model_level", "model_level_number", "ml",
    "vertical", "vert",
    "bottom_top", "bottom_top_stag",
    "mlev", "ilev", "lev_p", "lev_l", "soil_layers_stag",
    "isobaric", "isobaric1", "isobaric2",
    "sigma", "sigma_level",
    "hybrid", "hybrid_level",
}


def _is_vertical_like_dim(name) -> bool:
    """True if `name` looks like a vertical (level/height/depth) dim."""
    if name is None:
        return False
    n = str(name).lower().strip()
    if n in _VERTICAL_DIM_NAMES:
        return True
    if n.startswith(("lev", "plev", "ilev", "mlev")):
        return True
    if n.endswith(("_lev", "_level", "_levels")):
        return True
    return False


def _shrink_order(shape, dims) -> list:
    """
    Return axis indices in the order they should be shrunk when the chunk
    is over target.  Skips axis 0 (the leading dim, handled separately).

    hiopy approach: shrink non-vertical (horizontal/cell) dims first, last
    spatial dim first (C-order); shrink vertical dims last.  When `dims`
    is None or empty, fall back to plain last-dim-first.
    """
    ndim = len(shape)
    if ndim <= 1:
        return []
    if dims is None or len(dims) != ndim:
        return list(range(ndim - 1, 0, -1))

    horizontal, vertical = [], []
    for i, name in enumerate(dims):
        if i == 0:
            continue
        (vertical if _is_vertical_like_dim(name) else horizontal).append(i)
    # Within each group, shrink the LAST (fastest-varying) dim first.
    return sorted(horizontal, reverse=True) + sorted(vertical, reverse=True)


def _compute_inner_chunk_shape(
    shape,
    dtype,
    dims,
    target_bytes: int,
    allow_spatial_split: bool = True,
) -> Tuple[Tuple[int, ...], str]:
    """
    Core inner-chunk sizing algorithm.  Returns (inner_chunk_shape, mode).

    `mode` is a short string describing which branch was taken, suitable
    for logging:
      - "leading-fits"   : one leading slice fits in target; chunked along leading
      - "spatial-split"  : leading slice exceeds target; spatial dims also shrunk
      - "temporal-only"  : leading slice exceeds target but split was disabled;
                           one timestep per chunk, full spatial (may exceed target)

    Algorithm:
      1. If one slice along the leading dim fits in target, pack as many
         leading slices as fit.
      2. Else set leading=1.  If allow_spatial_split is False, return now
         (caller is responsible for any oversize warning).
      3. Else walk spatial dims in `_shrink_order` and reduce each until
         the chunk fits in target_bytes.
    """
    itemsize = int(np.dtype(dtype).itemsize)
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    if ndim == 0:
        return (), "leading-fits"

    inner = list(shape)

    # Bytes for one leading slice (the full trailing tile).
    trailing = int(np.prod(shape[1:])) if ndim > 1 else 1
    bytes_per_leading = itemsize * trailing
    if bytes_per_leading == 0:
        return tuple(inner), "leading-fits"

    if bytes_per_leading <= target_bytes:
        leading = max(1, target_bytes // bytes_per_leading)
        inner[0] = int(min(shape[0], leading))
        return tuple(int(x) for x in inner), "leading-fits"

    # One leading slice already exceeds target.
    inner[0] = 1

    if not allow_spatial_split:
        return tuple(int(x) for x in inner), "temporal-only"

    # Walk spatial dims in hiopy shrink order, reducing each until we fit.
    for axis in _shrink_order(shape, dims):
        chunk_bytes = itemsize * int(np.prod(inner))
        if chunk_bytes <= target_bytes:
            break
        per_row = chunk_bytes // inner[axis] if inner[axis] > 0 else chunk_bytes
        if per_row <= 0:
            inner[axis] = 1
            continue
        new_size = max(1, target_bytes // per_row)
        inner[axis] = int(min(inner[axis], new_size))

    return tuple(int(x) for x in inner), "spatial-split"


def compute_chunk_shape_for_eval(
    shape,
    dtype,
    target_mib: int = 16,
    dims=None,
    max_target_mib: int = 256,
    allow_spatial_split: bool = True,
):
    """
    Pick a chunk shape for in-memory evaluation.  Same algorithm as the
    persist path so the measured compression ratio reflects production.

    Parameters
    ----------
    shape, dtype : array geometry.
    target_mib : soft target chunk size in MiB.
    dims : tuple of dim names (used to keep vertical-like dims whole when
           spatial splitting is needed).  Pass None to fall back to plain
           last-dim-first ordering.
    max_target_mib : hard ceiling in MiB.  Only relevant when
           allow_spatial_split=False (in that case we may emit a chunk
           larger than target_mib; we warn once if it also exceeds this
           ceiling).  Caller is responsible for the warning.
    allow_spatial_split : if False, never split spatial dims; keep
           (1, ...full spatial...) and accept oversized chunks.
    """
    target_bytes = int(target_mib) * 2**20
    inner, _mode = _compute_inner_chunk_shape(
        shape, dtype, dims, target_bytes,
        allow_spatial_split=allow_spatial_split,
    )
    return inner


def compute_chunk_and_shard_shape(
    shape,
    dtype,
    inner_mib: int = 16,
    shard_mib: int = 512,
    dims=None,
    max_inner_mib: int = 256,
    allow_spatial_split: bool = True,
) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
    """
    Auto-compute (inner_chunk_shape, shard_shape) for zarr v3 sharding.

    Returns
    -------
    (inner, shards) where:
      - `inner` is the inner chunk shape.
      - `shards` is the shard shape, OR `None` to signal "skip sharding"
        (the inner chunk already meets/exceeds the shard target, so a
        shard would bundle <= 1 chunk and add only index overhead).

    Parameters
    ----------
    inner_mib : soft target for inner chunk (MiB).
    shard_mib : soft target for shard (MiB).  Each shard must contain an
                integer number of inner chunks on every axis.
    dims : optional tuple of dim names.  When provided, vertical-like
           dims are shrunk last (hiopy approach).
    max_inner_mib : hard ceiling on inner chunk size in MiB.  Used by the
           caller to decide whether to warn; the algorithm itself respects
           inner_mib when allow_spatial_split=True.
    allow_spatial_split : if True (default), spatial dims are split when a
           single timestep already exceeds inner_mib.  If False, chunks
           remain (1, ...full spatial...) and may exceed the target.
    """
    itemsize = int(np.dtype(dtype).itemsize)
    inner_target_bytes = int(inner_mib) * 2**20
    shard_target_bytes = int(shard_mib) * 2**20

    inner, _mode = _compute_inner_chunk_shape(
        shape, dtype, dims, inner_target_bytes,
        allow_spatial_split=allow_spatial_split,
    )

    # ---- shard shape ----
    # Skip sharding when one chunk already fills (or exceeds) a shard:
    # bundling a single chunk in a shard buys nothing and costs index bytes.
    inner_bytes = itemsize * int(np.prod(inner))
    if inner_bytes == 0 or inner_bytes >= shard_target_bytes:
        return inner, None

    multiplier = shard_target_bytes // inner_bytes
    if multiplier <= 1:
        return inner, None

    shard = list(inner)
    shard[0] = min(int(shape[0]), inner[0] * int(multiplier))
    # Must be an integer multiple of inner[0].
    shard[0] = (shard[0] // inner[0]) * inner[0]
    shard[0] = max(shard[0], inner[0])
    shard_t = tuple(int(x) for x in shard)
    if shard_t == inner:
        return inner, None
    return inner, shard_t


# =============================================================================
# CODEC SPACES
# =============================================================================

# ---- Compressor parameter grids (bytes -> bytes, data-independent) ----------
_BLOSC_CNAMES      = ("lz4", "lz4hc", "zstd")               # dropped blosclz
_BLOSC_CLEVELS     = (1, 5, 9)
_BLOSC_SHUFFLES    = (0, 1)                                 # dropped shuffle=2 bit
_LZ4_ACCELERATIONS = (1, 10, 100)
_ZSTD_LEVELS       = (6, 12, 22)                            # dropped 1
_ZLIB_LEVELS       = (3, 6, 9)                              # dropped 1
_BZ2_LEVELS        = (3, 6, 9)                              # dropped 1
_LZMA_PRESETS      = (3, 6, 9)                              # dropped 1

# ---- Filter parameter grids (array -> array) --------------------------------
# The top value in each tuple is effectively lossless
# and acts as the upper-bound reference point.
_BITROUND_KEEPBITS_F32 = (3, 7, 11, 13, 17, 23)             # dropped 5, 9, 15; 23 -> lossless
_BITROUND_KEEPBITS_F64 = (3, 7, 11, 17, 23, 30, 37, 44, 52) # dropped 5, 9, 13 (mirroring f32 logic); 52 -> lossless
_QUANTIZE_DIGITS_F32   = (1, 3, 4, 5, 6, 7)                 # dropped 2 (adjacent-redundant); 7 -> ~lossless
_QUANTIZE_DIGITS_F64   = (1, 3, 4, 5, 6, 7, 9, 11, 13, 15)  # dropped 2 (adjacent-redundant); 15 -> ~lossless

# ---- FixedScaleOffset target integer widths (array -> array, data-dependent) -
# The field's [min, max] is mapped onto the full range of each integer width.
# More bits -> finer absolute precision (less lossy) but larger encoded ints.
# u8 is very lossy (256 levels), u32 is effectively lossless for f32 inputs.
_FSO_TARGET_UINTS      = ("uint8", "uint16", "uint32")

# ---- Serializer parameter grids (array -> bytes) ----------------------------
_PCODEC_LEVELS         = (6, 8, 10, 12)      # dropped 4; dropped 0 ("no compression")
_PCODEC_DELTA_ORDERS   = (0, 7)              # dropped 3 (middle); endpoints cover delta-mode space
_ZFPY_K_GRID           = (0, 1, 2, 3)        # k -> compute_fixed_*_param(k);
                                             # fixed-rate / fixed-precision: 8/16/32/64 bits
                                             # fixed-accuracy:               0.5/0.25/0.0625/0.0039


def compressor_space(da, with_lossy=True, compressor_class="all"):
    """
    Bytes->bytes compressor space.  Data-independent: the `da` argument is
    accepted only for signature symmetry with filter_space / serializer_space,
    and the lossy flag is ignored (every codec here is lossless).
    Returns [(index, codec), ...].

    Byte-shuffle note: a STANDALONE numcodecs Shuffle codec cannot be used in
    this pipeline.  Shuffle is a bytes->bytes codec that requires its input
    length to be an exact multiple of `elementsize`; in Zarr v3 the bytes->bytes
    stage runs AFTER the array->bytes serializer, so with a compressing
    serializer (ZFPY / PCodec) Shuffle receives an already-compressed,
    variable-length blob and raises "buffer is not an integer multiple of
    elementsize".  The byte-shuffle transform is instead obtained via Blosc's
    own `shuffle` parameter (swept in _BLOSC_SHUFFLES), which is the supported
    way to byte-shuffle in Zarr v3.

    Note: standalone GZip has been removed from the space.  GZip and Zlib
    both run DEFLATE with different wrapping headers, so they produce
    identical CR for identical input; keeping both was strict redundancy.
    """
    _COMPRESSORS = [
        zarrcodecs_nc.Blosc, zarrcodecs_nc.LZ4, zarrcodecs_nc.Zstd,
        zarrcodecs_nc.Zlib, zarrcodecs_nc.BZ2, zarrcodecs_nc.LZMA,
    ]
    _COMPRESSOR_MAP = {cls.__name__.lower(): cls for cls in _COMPRESSORS}

    space = []
    if compressor_class.lower() == "all":
        pass
    elif compressor_class.lower() in _COMPRESSOR_MAP:
        _COMPRESSORS = [_COMPRESSOR_MAP[compressor_class.lower()]]
    elif compressor_class.lower() == "none":
        _COMPRESSORS = []
        space.append(None)

    for compressor in _COMPRESSORS:
        if compressor is zarrcodecs_nc.Blosc:
            for cname in _BLOSC_CNAMES:
                for clevel in _BLOSC_CLEVELS:
                    for shuffle in _BLOSC_SHUFFLES:
                        space.append(compressor(cname=cname, clevel=clevel, shuffle=shuffle))
        elif compressor is zarrcodecs_nc.LZ4:
            for acceleration in _LZ4_ACCELERATIONS:
                space.append(compressor(acceleration=acceleration))
        elif compressor is zarrcodecs_nc.Zstd:
            for level in _ZSTD_LEVELS:
                space.append(compressor(level=level))
        elif compressor is zarrcodecs_nc.Zlib:
            for level in _ZLIB_LEVELS:
                space.append(compressor(level=level))
        elif compressor is zarrcodecs_nc.BZ2:
            for level in _BZ2_LEVELS:
                space.append(compressor(level=level))
        elif compressor is zarrcodecs_nc.LZMA:
            for preset in _LZMA_PRESETS:
                space.append(compressor(preset=preset))

    return list(enumerate(space))


def filter_space(da, with_lossy=True, filter_class="all", data_range=None):
    """
    Array->array filter space.  For integer dtypes only Delta is meaningful
    (BitRound/Quantize are float-only).  If the user asks for a filter class
    that is incompatible with the dtype, we warn explicitly rather than
    silently honouring the dtype override.

    `data_range`: optional (global_min, global_max) of the FULL field, used to
    build FixedScaleOffset's affine integer packing.  Pass the true full-field
    extremes here whenever `da` is only a strided evaluation sample, so the
    uint packing cannot overflow on production values outside the sample range.
    If None, FixedScaleOffset falls back to the min/max of `da` (safe only when
    `da` is the whole field).

    Returns [(index, codec), ...].
    """
    is_int = (da.dtype.kind == "i")
    _FILTERS = [zarrcodecs_nc.Delta]
    if with_lossy:
        _FILTERS += [zarrcodecs_nc.BitRound, zarrcodecs_nc.Quantize,
                     zarrcodecs_nc.FixedScaleOffset]
        # AsType down-cast is only meaningful when the field is WIDER than
        # 32-bit float (e.g. f64 -> f32).  For already-32-bit fields there is
        # nothing to narrow, so it is not added.
        if np.issubdtype(da.dtype, np.floating) and da.dtype.itemsize > 4:
            _FILTERS.append(zarrcodecs_nc.AsType)
    if is_int:
        # Integer fields: only Delta is algorithmically meaningful.  Surface
        # the override to the user instead of silently dropping their
        # --filter-class request.
        if filter_class.lower() not in ("all", "delta", "none"):
            click.echo(
                f"[filter_space] integer dtype {da.dtype}: only Delta is "
                f"available; ignoring --filter-class={filter_class}.",
                err=True,
            )
        _FILTERS = [zarrcodecs_nc.Delta]

    _FILTER_MAP = {cls.__name__.lower(): cls for cls in _FILTERS}

    space = []
    if filter_class.lower() == "all":
        pass
    elif filter_class.lower() in _FILTER_MAP:
        _FILTERS = [_FILTER_MAP[filter_class.lower()]]
    elif filter_class.lower() == "none":
        _FILTERS = []
        space.append(None)

    for filt in _FILTERS:
        if filt is zarrcodecs_nc.Delta:
            if np.issubdtype(da.dtype, np.number):
                space.append(filt(dtype=str(da.dtype)))
        elif filt is zarrcodecs_nc.BitRound:
            for keepbits in valid_keepbits_for_bitround(da):
                space.append(filt(keepbits=keepbits))
        elif filt is zarrcodecs_nc.Quantize:
            for digits in valid_digits_for_quantize(da):
                space.append(filt(digits=digits, dtype=str(da.dtype)))
        elif filt is zarrcodecs_nc.FixedScaleOffset:
            for cfg in fixed_scale_offset_configs(da, data_range=data_range):
                space.append(filt(**cfg))
        elif filt is zarrcodecs_nc.AsType:
            # Down-cast wider floats to float32 as a cheap ~2x lossy pre-step.
            # encode_dtype is the narrowed type; decode_dtype is the original.
            space.append(filt(encode_dtype="float32", decode_dtype=str(da.dtype)))

    return list(enumerate(space))


def serializer_space(da, with_lossy=True, serializer_class="all"):
    """
    Array->bytes serializer space.  PCodec is always present.  ZFPY is added
    when with_lossy=True.

    For integer dtypes only ZFPY's fixed-rate mode is meaningful; the other
    two modes are skipped.

    Returns [(index, codec), ...].
    """
    is_int = (da.dtype.kind == "i")
    _SERIALIZERS = [zarrcodecs_nc.PCodec]
    if with_lossy:
        _SERIALIZERS.append(zarrcodecs_nc.ZFPY)

    _SERIALIZER_MAP = {cls.__name__.lower(): cls for cls in _SERIALIZERS}

    space = []
    if serializer_class.lower() == "all":
        pass
    elif serializer_class.lower() in _SERIALIZER_MAP:
        _SERIALIZERS = [_SERIALIZER_MAP[serializer_class.lower()]]
    elif serializer_class.lower() == "none":
        _SERIALIZERS = []
        space.append(None)

    for serializer in _SERIALIZERS:
        if serializer is zarrcodecs_nc.PCodec:
            for level in _PCODEC_LEVELS:
                for delta_encoding_order in _PCODEC_DELTA_ORDERS:
                    space.append(serializer(
                        level=level, mode_spec="auto",
                        delta_spec="auto", delta_encoding_order=delta_encoding_order,
                    ))
        elif serializer is zarrcodecs_nc.ZFPY:
            _ZFP_MODES = [
                ("fixed-accuracy",  zfpy.mode_fixed_accuracy,  "tolerance", compute_fixed_accuracy_param),
                ("fixed-precision", zfpy.mode_fixed_precision, "precision", compute_fixed_precision_param),
                ("fixed-rate",      zfpy.mode_fixed_rate,      "rate",      compute_fixed_rate_param),
            ]
            if is_int:
                _ZFP_MODES = [m for m in _ZFP_MODES if m[0] == "fixed-rate"]
            for mode_str, zfpy_mode, param_name, param_fn in _ZFP_MODES:
                for k in _ZFPY_K_GRID:
                    val = param_fn(k)
                    space.append(serializer(mode=zfpy_mode, **{param_name: val}))

    return list(enumerate(space))


def valid_keepbits_for_bitround(xr_dataarray):
    """Return the BitRound keepbits grid for the dtype of `xr_dataarray`."""
    dtype = xr_dataarray.dtype
    if np.issubdtype(dtype, np.float64):
        return _BITROUND_KEEPBITS_F64
    elif np.issubdtype(dtype, np.float32):
        return _BITROUND_KEEPBITS_F32
    else:
        raise TypeError(
            f"Unsupported dtype '{dtype}'. BitRound only supports float32 and float64."
        )


def valid_digits_for_quantize(xr_dataarray):
    """Return the Quantize digits grid for the dtype of `xr_dataarray`."""
    dtype = xr_dataarray.dtype
    if np.issubdtype(dtype, np.float64):
        return _QUANTIZE_DIGITS_F64
    elif np.issubdtype(dtype, np.float32):
        return _QUANTIZE_DIGITS_F32
    else:
        raise TypeError(
            f"Unsupported dtype '{dtype}'. Quantize only supports float32 and float64."
        )


def full_field_data_range(da):
    """
    Compute the finite (global_min, global_max) over the ENTIRE field, for use
    as FixedScaleOffset's affine anchor.  This is a single streaming reduction
    (no compression), far cheaper than the codec sweep, and parallelises over
    dask chunks.  Returns None if the field has no finite values or is
    constant (in which case FixedScaleOffset is not applicable anyway).

    Computed once (e.g. on rank 0) and broadcast to all ranks so the codec
    space is identical everywhere and reproducible in compress_with_optimal.
    """
    import dask.array as _dask_array
    data = da.data if hasattr(da, "data") else np.asarray(da)
    try:
        if isinstance(data, _dask_array.Array):
            finite = _dask_array.isfinite(data)
            # masked reductions; nan-aware min/max over finite entries only
            mn = _dask_array.where(finite, data, np.inf).min()
            mx = _dask_array.where(finite, data, -np.inf).max()
            dmin = float(mn.compute())
            dmax = float(mx.compute())
        else:
            arr = np.asarray(data)
            fin = arr[np.isfinite(arr)]
            if fin.size == 0:
                return None
            dmin = float(fin.min())
            dmax = float(fin.max())
    except Exception:
        return None
    if not (np.isfinite(dmin) and np.isfinite(dmax)) or dmax <= dmin:
        return None
    return (dmin, dmax)


def fixed_scale_offset_configs(xr_dataarray, data_range=None):
    """
    Build FixedScaleOffset (offset, scale, astype) tuples that map the field's
    [min, max] onto the full range of each target unsigned-integer width.

    Encode is round((x - offset) * scale); to fill [0, 2**bits - 1] we use
    offset = data_min and scale = (2**bits - 1) / (data_max - data_min).

    CRITICAL - range source.  FixedScaleOffset packs floats into unsigned
    integers anchored to [data_min, data_max].  numcodecs does NO overflow
    clipping (see its docstring warning), so a production value OUTSIDE the
    range used to build the codec encodes to a negative or too-large integer
    and silently CORRUPTS on the unsigned cast.  The range therefore MUST cover
    the WHOLE field, not just the strided evaluation sample (which may miss the
    global extremes on un-sampled timesteps/levels).

      - `data_range=(global_min, global_max)`: use these true full-field
        extremes (the correct production path; caller computes them once over
        the full dask array and threads them through filter_space).
      - `data_range=None`: fall back to the min/max of `xr_dataarray` itself.
        Only safe when that array IS the full field (e.g. small fields that
        fit under the eval-data-size-limit, or unit tests).  A WARNING-worthy
        path for strided samples; callers that have the full field should
        always pass `data_range`.

    Returns [] for constant / non-finite fields (no usable range) and skips
    integer widths that cannot improve on the source itemsize.  dtype is the
    decoded (original) dtype.
    """
    dtype = xr_dataarray.dtype
    if not np.issubdtype(dtype, np.floating):
        return []

    if data_range is not None:
        dmin, dmax = float(data_range[0]), float(data_range[1])
    else:
        # Fallback: derive from the given array (valid only if it is the full
        # field).  Finite values only.
        arr = np.asarray(xr_dataarray.values)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return []
        dmin = float(finite.min())
        dmax = float(finite.max())

    span = dmax - dmin
    if not (np.isfinite(dmin) and np.isfinite(dmax)) or span <= 0.0:
        # constant / non-finite field: nothing for an affine packer to do
        return []

    src_itemsize = dtype.itemsize
    configs = []
    for uw in _FSO_TARGET_UINTS:
        bits = np.dtype(uw).itemsize * 8
        # Only worth it if the packed integer is smaller than the source value.
        if np.dtype(uw).itemsize >= src_itemsize:
            continue
        scale = (float(2 ** bits) - 1.0) / span
        configs.append(dict(offset=dmin, scale=scale,
                            dtype=str(dtype), astype=uw))
    return configs


def combo_is_valid(filt, serializer):
    """
    Reject (filter, serializer) pairings that are known to be broken or
    meaningless, so the product builder can skip them instead of paying for a
    guaranteed per-combo failure.

    Currently: a FixedScaleOffset filter emits an UNSIGNED-INTEGER array, and
    feeding that to a ZFPY serializer in anything other than fixed-rate mode
    crashes inside numcodecs' zfpy backend (it lacks the integer-mode encode
    path and raises AttributeError: 'ZFPY' object has no attribute
    'compression_kwargs').  ZFPY's own integer support is fixed-rate only.
    Since serializer_space derives its ZFPY mode from the ORIGINAL float dtype
    (so all three modes are present), we must reject FSO->ZFPY combos here at
    the pairing level.  PCodec handles the integer output natively, so
    FixedScaleOffset->PCodec is allowed and is the intended integer pairing.

    AsType(encode='float32') keeps the data floating, so it is unaffected.
    """
    if isinstance(filt, zarrcodecs_nc.FixedScaleOffset) and \
       isinstance(serializer, zarrcodecs_nc.ZFPY):
        return False
    return True


def compute_fixed_precision_param(param: int) -> int:  return 1 << (param + 3)
def compute_fixed_rate_param(param: int) -> int:       return 1 << (param + 3)
def compute_fixed_accuracy_param(param: int) -> float: return math.ldexp(1.0, -(1 << param))


def inclusive_range(start, end, step=1):
    if step == 0:
        raise ValueError("step must not be zero")
    values = []
    i = start
    if step > 0:
        while i <= end:
            values.append(i); i += step
        if values[-1] != end:
            values.append(end)
    else:
        while i >= end:
            values.append(i); i += step
        if values[-1] != end:
            values.append(end)
    return values


# =============================================================================
# CODEC PIPELINE ASSEMBLY
# =============================================================================

def _codec_kwargs(filters, compressors, serializer):
    """
    Build the keyword arguments that describe a codec pipeline for
    zarr.create_array / Group.create_array (forwarded through
    dask.array.to_zarr as **zarr_array_kwargs).

    Zarr v3's user-facing API takes the three components as SEPARATE kwargs:
      - filters=    list of array->array codecs, or None
      - serializer= a single array->bytes codec, or "auto" to use the default
      - compressors=list of bytes->bytes codecs, or None

    We only include a key when the caller has a value for it, so zarr's own
    defaults apply when a component is missing.

    `serializer == "auto"` is a sentinel from the CLI meaning "let zarr pick";
    we pass it through literally because zarr accepts "auto" as a valid value.
    """
    kwargs = {}
    if filters is not None:
        kwargs["filters"] = filters
    if compressors is not None:
        kwargs["compressors"] = compressors
    if serializer is not None:
        kwargs["serializer"] = serializer
    return kwargs


# =============================================================================
# CODEC PIPELINE - EVALUATION (no persistence, thread-safe)
# =============================================================================

def _info_bytes(info):
    """
    Return (count_bytes, count_bytes_stored) from a zarr ArrayInfo.

    Prefers the public attributes exposed by recent zarr versions, but falls
    back to the private underscore-prefixed names for older ones.  Protects
    against a future zarr upgrade renaming / removing the private fields.
    """
    count = getattr(info, "count_bytes", None)
    if count is None:
        count = info._count_bytes
    stored = getattr(info, "count_bytes_stored", None)
    if stored is None:
        stored = info._count_bytes_stored
    return int(count), int(stored)


def _iter_chunk_slices(shape, chunk_shape):
    """Yield tuples of slice() covering `shape` in `chunk_shape` steps."""
    ranges = [range(0, s, c) for s, c in zip(shape, chunk_shape)]
    for start in product(*ranges):
        yield tuple(
            slice(st, min(st + c, s))
            for st, c, s in zip(start, chunk_shape, shape)
        )


# =============================================================================
# ZARR SYNC-API BYPASS  (opt-in via cli --bypass-zarr-sync)
# =============================================================================
# zarr 3's sync wrapper (zarr.core.sync.sync) runs every coroutine on a
# process-global event loop, serialising codec calls from concurrent worker
# threads down to ~1 effective core (measured 5x slowdown at 1x32 vs 32x1).
# We bypass it by calling zarr.api.asynchronous.create_array directly, with
# a persistent event loop per worker thread.
#
# A single shared bounded ThreadPoolExecutor is wired as the default
# executor on every per-thread loop.  Without that, asyncio.to_thread()
# inside zarr's native codecs lazily creates a 32-worker default executor
# per loop -> 32 user threads x 32 workers = ~1024 OS threads (validation
# job 844391: 30 GB RAM, AveCPU/wall = 2.1).  The shared executor caps
# total OS threads at user_threads + shared_workers.

try:
    from zarr.api.asynchronous import create_array as _zarr_async_create_array
    _ASYNC_BYPASS_AVAILABLE = True
except ImportError:
    _zarr_async_create_array = None
    _ASYNC_BYPASS_AVAILABLE = False


_thread_local_loops = threading.local()
_shared_executor = None
_shared_executor_lock = threading.Lock()


def _get_or_create_shared_executor(max_workers: int):
    """Lazy, thread-safe singleton ThreadPoolExecutor."""
    global _shared_executor
    if _shared_executor is None:
        with _shared_executor_lock:
            if _shared_executor is None:
                from concurrent.futures import ThreadPoolExecutor
                _shared_executor = ThreadPoolExecutor(
                    max_workers=max(1, int(max_workers)),
                    thread_name_prefix="bypass_codec",
                )
    return _shared_executor


def _shutdown_shared_executor() -> None:
    global _shared_executor
    with _shared_executor_lock:
        if _shared_executor is not None:
            _shared_executor.shutdown(wait=False, cancel_futures=False)
            _shared_executor = None


def _get_thread_event_loop() -> asyncio.AbstractEventLoop:
    """
    Return this thread's persistent asyncio loop, creating it on first call.
    The shared bounded executor is bound as default executor on creation;
    without this, asyncio.to_thread() inside zarr's native codecs spawns a
    32-worker default executor PER per-thread loop.
    """
    loop = getattr(_thread_local_loops, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if _shared_executor is not None:
            loop.set_default_executor(_shared_executor)
        _thread_local_loops.loop = loop
    return loop


class _AsyncBypass:
    """Toggle for the async-direct codec dispatch (cli --bypass-zarr-sync)."""
    enabled: bool = False
    threads_per_rank: int = 1

    @classmethod
    def enable(cls, threads_per_rank: int = 1) -> None:
        if not _ASYNC_BYPASS_AVAILABLE:
            raise RuntimeError(
                "Cannot enable --bypass-zarr-sync: "
                "zarr.api.asynchronous.create_array is not importable. "
                "Upgrade zarr or run without the flag."
            )
        cls.enabled = True
        cls.threads_per_rank = max(1, int(threads_per_rank))
        _get_or_create_shared_executor(cls.threads_per_rank)

    @classmethod
    def disable(cls) -> None:
        cls.enabled = False
        _shutdown_shared_executor()


AsyncBypass = _AsyncBypass


async def _zarr_pipeline_async(sample_np, dims, codec_kwargs, chunks):
    """async create + encode + info + decode."""
    store = zarr.storage.MemoryStore()

    with Timer("eval.create_array"):
        z = await _zarr_async_create_array(
            store=store,
            name="_tmp_eval",
            shape=sample_np.shape,
            dtype=sample_np.dtype,
            chunks=chunks,
            zarr_format=3,
            dimension_names=tuple(dims),
            **codec_kwargs,
        )

    with Timer("eval.encode"):
        # See _zarr_pipeline_sync: suppress only the benign NaN-fill cast warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in cast",
                category=RuntimeWarning,
            )
            await z.setitem(Ellipsis, sample_np)

    with Timer("eval.info_complete"):
        info = await z.info_complete()
        count_bytes, count_bytes_stored = _info_bytes(info)
        ratio = count_bytes / count_bytes_stored

    with Timer("eval.decode"):
        decomp_full = await z.getitem(Ellipsis)

    return decomp_full, ratio


def _zarr_pipeline_sync(sample_np, dims, codec_kwargs, chunks):
    """sync create + encode + info + decode."""
    store = zarr.storage.MemoryStore()

    with Timer("eval.create_array"):
        z = zarr.create_array(
            store=store,
            name="_tmp_eval",
            shape=sample_np.shape,
            dtype=sample_np.dtype,
            chunks=chunks,
            zarr_format=3,
            dimension_names=tuple(dims),
            **codec_kwargs,
        )

    with Timer("eval.encode"):
        # FixedScaleOffset casts NaN-fill cells to int, which numpy flags as
        # "invalid value encountered in cast".  This is benign: fill cells are
        # masked out of every error norm downstream, so the garbage ints never
        # affect scoring.  Suppress ONLY this specific message -- a genuine FSO
        # overflow does NOT warn (numpy wraps silently; that path is guarded by
        # full_field_data_range + the verify gate), so nothing real is hidden.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in cast",
                category=RuntimeWarning,
            )
            z[...] = sample_np

    with Timer("eval.info_complete"):
        info = z.info_complete()
        count_bytes, count_bytes_stored = _info_bytes(info)
        ratio = count_bytes / count_bytes_stored

    with Timer("eval.decode"):
        decomp_full = z[...]

    return decomp_full, ratio


# =============================================================================
# CODEC PIPELINE - EVALUATION (no persistence, thread-safe)
# =============================================================================


def evaluate_codec_pipeline(
    sample_np: np.ndarray,
    dims,
    filters,
    compressors,
    serializer,
    chunks,
    q99_abs: float | None = None,
    compute_gradient: bool = False,
    gradient_axes=None,
    precheck_thresholds: dict | None = None,
):
    """
    Measure (compression_ratio, errors, euclidean_distance) for a codec
    pipeline against `sample_np`.  In-memory zarr store, no I/O.  Safe to
    call concurrently from multiple threads.

    The full sample is decoded once via z[...]; the metrics loop slices
    that buffer chunk-wise to bound the float64 promotion peak.  Peak
    per-rank memory is ~2 * sample_nbytes (original + decoded) — caller
    must size NTASKS_PER_NODE accordingly.

    Dispatch: if AsyncBypass is enabled, the zarr operations route through
    a per-thread persistent event loop (sidesteps zarr 3's process-global
    sync() loop that otherwise serialises codec dispatch from concurrent
    threads).  The metrics phase is identical in both paths.
    """
    codec_kwargs = _codec_kwargs(filters, compressors, serializer)

    if _AsyncBypass.enabled:
        loop = _get_thread_event_loop()
        decomp_full, ratio = loop.run_until_complete(
            _zarr_pipeline_async(sample_np, dims, codec_kwargs, chunks)
        )
    else:
        decomp_full, ratio = _zarr_pipeline_sync(
            sample_np, dims, codec_kwargs, chunks
        )

    # Chunk-wise error accumulation: bounds the float64 promotion peak at
    # one chunk's worth.
    #
    # Non-finite handling (see PR introducing the production-grade gates):
    #   - Cells that are non-finite in the ORIGINAL are legitimate fill
    #     (ocean points in a land field, below-surface levels, masked
    #     regions).  They are EXCLUDED from every norm so fill never
    #     poisons the statistics.
    #   - Cells that are finite in the original but non-finite in the
    #     DECODED output are genuine corruption.  They are excluded from
    #     the norms (so the norms stay meaningful) but COUNTED in
    #     `n_corrupt`; the caller treats n_corrupt > 0 as a hard reject
    #     (pass_finite = False).  This replaces the old behaviour of
    #     raising CombinationProducedNonFiniteError for any non-finite.
    #
    # Alongside the L-norms we also accumulate, in the same single pass:
    #   - signed error sum  -> relative bias (mean signed error / ||o||_1)
    #   - decoded min/max over valid cells -> physical-bounds gate
    #   - (optional) error restricted to the extreme tail |o| >= q99_abs
    #     -> q99 relative error gate for extremes-sensitive fields.
    with Timer("eval.metrics"):
        l1_err = 0.0; l2_err_sq = 0.0; linf_err = 0.0
        l1_ori = 0.0; l2_ori_sq = 0.0; linf_ori = 0.0
        signed_err = 0.0
        n_valid = 0
        n_corrupt = 0
        decoded_min = math.inf
        decoded_max = -math.inf
        q99_err = 0.0; q99_ori = 0.0
        want_q99 = q99_abs is not None and math.isfinite(q99_abs)

        with np.errstate(invalid="ignore"):
            for sl in _iter_chunk_slices(sample_np.shape, chunks):
                orig = sample_np[sl]
                decomp = decomp_full[sl]

                finite_orig = np.isfinite(orig)
                finite_dec = np.isfinite(decomp)
                # Corruption: valid input, broken output.
                n_corrupt += int(np.count_nonzero(finite_orig & ~finite_dec))
                # Cells we actually score: finite on both sides.
                valid = finite_orig & finite_dec
                nv = int(np.count_nonzero(valid))
                if nv == 0:
                    continue
                n_valid += nv

                o = orig[valid].astype(np.float64, copy=False)
                d = decomp[valid].astype(np.float64, copy=False)
                e = d - o
                e_abs = np.abs(e)
                o_abs = np.abs(o)

                l1_err     += float(e_abs.sum())
                l2_err_sq  += float((e * e).sum())
                linf_err    = max(linf_err, float(e_abs.max(initial=0.0)))
                signed_err += float(e.sum())

                l1_ori     += float(o_abs.sum())
                l2_ori_sq  += float((o_abs * o_abs).sum())
                linf_ori    = max(linf_ori, float(o_abs.max(initial=0.0)))

                decoded_min = min(decoded_min, float(d.min()))
                decoded_max = max(decoded_max, float(d.max()))

                if want_q99:
                    ext = o_abs >= q99_abs
                    if ext.any():
                        q99_err += float(e_abs[ext].sum())
                        q99_ori += float(o_abs[ext].sum())

        del decomp_full

        # A finite accumulator is now guaranteed for ordinary fields because
        # corrupted cells were excluded above.  This scalar guard only trips
        # on pathological inputs (e.g. an original that overflowed float64).
        if not (math.isfinite(l1_err) and math.isfinite(l2_err_sq)
                and math.isfinite(linf_err)):
            raise CombinationProducedNonFiniteError(
                f"error accumulators are non-finite even after masking "
                f"(l1_err={l1_err}, l2_err_sq={l2_err_sq}, linf_err={linf_err})"
            )

    l2_err = math.sqrt(l2_err_sq)
    l2_ori = math.sqrt(l2_ori_sq)

    def _safe_div(a, b):
        return float(a) / float(b) if b != 0 else float("inf")

    errors = {
        "Relative_Error_L1":   _safe_div(l1_err,  l1_ori),
        "Relative_Error_L2":   _safe_div(l2_err,  l2_ori),
        "Relative_Error_Linf": _safe_div(linf_err, linf_ori),
        # Signed mean error / mean |original|.  Same denominator as L1, so
        # |Bias_Rel| <= Relative_Error_L1 always holds by construction.
        "Bias_Rel":            _safe_div(abs(signed_err), l1_ori),
        # Decoded value range over valid cells -> physical-bounds gate.
        "Decoded_Min":         (decoded_min if n_valid else float("nan")),
        "Decoded_Max":         (decoded_max if n_valid else float("nan")),
        # Count of valid-input/broken-output cells -> finite (Layer 6) gate.
        "N_Corrupt":           int(n_corrupt),
        "N_Valid":             int(n_valid),
        # Extreme-tail relative error; None when not requested.
        "Q99_Rel":             (_safe_div(q99_err, q99_ori) if want_q99 else None),
    }

    # ---- optional gradient (spatial-structure) metric ------------------
    # Relative L1 error of the per-axis finite-difference field, combined
    # across the requested axes.  Computed on the in-memory arrays (not the
    # streaming accumulator) because finite differencing is a neighbourhood
    # op; done one axis at a time with intermediates freed between axes to
    # bound the transient.  NaN-masked: only positions finite in both the
    # original and decoded difference fields contribute.
    #
    # SHORT-CIRCUIT: the gradient re-decodes the sample (a second full
    # pipeline) and is by far the most expensive part of an evaluation.  A
    # combo that already fails any cheap gate (L1/L2/Linf/bias) can never be
    # kept, so its gradient value is irrelevant.  When `precheck_thresholds`
    # is supplied, skip the gradient for such combos: Grad_Rel stays None
    # (pass_grad becomes a no-op in _evaluate_gates) and the combo is rejected
    # by the failing cheap gate anyway.  This is SEMANTICALLY IDENTICAL to
    # computing the gradient for every combo — the kept set and the winner are
    # unchanged — but makes --gradient-gate nearly free.  Pass None to force
    # the gradient on every combo (the validation/debug path).
    do_grad = compute_gradient
    if compute_gradient and precheck_thresholds is not None:
        def _passes(val, lim):
            # Mirrors _evaluate_gates._le: a None/+inf limit is a no-op pass.
            if val is None or lim is None or not math.isfinite(lim):
                return True
            return float(val) <= float(lim)
        do_grad = (
            _passes(errors["Relative_Error_L1"],   precheck_thresholds.get("l1"))
            and _passes(errors["Relative_Error_L2"],   precheck_thresholds.get("l2"))
            and _passes(errors["Relative_Error_Linf"], precheck_thresholds.get("linf"))
            and _passes(errors["Bias_Rel"],            precheck_thresholds.get("bias"))
        )

    if do_grad:
        if _AsyncBypass.enabled:
            # decomp_full was deleted above to free memory; recompute it.
            loop = _get_thread_event_loop()
            decomp_full2, _ = loop.run_until_complete(
                _zarr_pipeline_async(sample_np, dims, codec_kwargs, chunks)
            )
        else:
            decomp_full2, _ = _zarr_pipeline_sync(
                sample_np, dims, codec_kwargs, chunks
            )
        errors["Grad_Rel"] = _gradient_rel_l1(
            sample_np, decomp_full2, axes=gradient_axes
        )
        del decomp_full2
    else:
        errors["Grad_Rel"] = None

    return ratio, errors, l2_err


def _gradient_rel_l1(orig: np.ndarray, decoded: np.ndarray, axes=None) -> float:
    """
    Relative L1 error of the finite-difference (gradient) field, combined
    over `axes` (default: every axis except the leading one, treated as
    time).  Returns Sum|d(decoded) - d(orig)| / Sum|d(orig)| accumulated
    across axes.  Non-finite difference positions are masked out.

    Done axis-by-axis with np.diff so at most one axis' worth of float64
    differences is live at a time.
    """
    ndim = orig.ndim
    if axes is None:
        axes = tuple(range(1, ndim)) if ndim > 1 else (0,)
    err_sum = 0.0
    ori_sum = 0.0
    with np.errstate(invalid="ignore"):
        for ax in axes:
            do = np.diff(orig.astype(np.float64, copy=False), axis=ax)
            dd = np.diff(decoded.astype(np.float64, copy=False), axis=ax)
            m = np.isfinite(do) & np.isfinite(dd)
            if not m.any():
                del do, dd, m
                continue
            de = np.abs(dd[m] - do[m])
            err_sum += float(de.sum())
            ori_sum += float(np.abs(do[m]).sum())
            del do, dd, m, de
    if ori_sum == 0:
        return 0.0 if err_sum == 0 else float("inf")
    return err_sum / ori_sum


# =============================================================================
# CODEC PIPELINE - PERSISTENCE (writes to a shared LocalStore, with sharding)
# =============================================================================

def persist_with_codec_pipeline(
    da,                           # xarray DataArray (dask-backed)
    store,                        # zarr.storage.LocalStore
    component: str,               # array name inside the store (== field name)
    filters,
    compressors,
    serializer,
    inner_chunks=None,
    shards=None,
    verify: bool = True,
    verbose: bool = True,
    rank: int = 0,
):
    """
    Write `da` into `store` at `component` using the codec pipeline.
    Returns (compression_ratio, errors, euclidean_distance).

    - If `shards` is given: Dask chunks are rechunked to the shard shape so
      each Dask task writes exactly one shard (no write-amplification).
    - If `shards` is None: sharding is skipped entirely (the `shards=` kwarg
      is NOT passed to zarr.create_array).  This happens automatically when
      one inner chunk already meets or exceeds the shard target -- a shard
      would bundle <= 1 chunk and add only index overhead.  Dask is then
      rechunked to the inner chunk shape (each task writes one chunk).
    - Uses `overwrite=True` as the dask-level kwarg (replaces the deprecated
      v2-era `mode='w'` shape).  `chunks`, `shards`, codec kwargs,
      `dimension_names`, `zarr_format=3` are passed through **zarr_array_kwargs
      and forwarded by dask to zarr.create_array.  `mode=` is NOT accepted by
      zarr v3's create_array -- it's a storage-level concept, not an array one.
    """
    assert isinstance(da.data, dask.array.Array), \
        "persist_with_codec_pipeline expects a dask-backed xr.DataArray"

    # Auto-size chunks/shards if not provided.
    if inner_chunks is None or shards is None:
        auto_inner, auto_shard = compute_chunk_and_shard_shape(
            da.shape, da.dtype, dims=tuple(da.dims),
        )
        if inner_chunks is None:
            inner_chunks = auto_inner
        if shards is None:
            shards = auto_shard  # may itself be None -> skip sharding

    codec_kwargs = _codec_kwargs(filters, compressors, serializer)

    # Pick the dask write unit:
    #   - With sharding: one task per shard (avoids partial-shard rewrites).
    #   - Without sharding: one task per inner chunk.
    write_unit = shards if shards is not None else inner_chunks
    dask_arr = da.data.rechunk(write_unit)

    zarr_kwargs = dict(
        zarr_format=3,
        dimension_names=tuple(da.dims),
        chunks=inner_chunks,
        **codec_kwargs,
    )
    if shards is not None:
        zarr_kwargs["shards"] = shards

    with Timer("dask.array.to_zarr"):
        # See _zarr_pipeline_sync: FixedScaleOffset casts NaN-fill to int and
        # numpy warns benignly; fill is masked out of the verify norms. Suppress
        # only that specific message (overflow is silent + guarded elsewhere).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in cast",
                category=RuntimeWarning,
            )
            dask.array.to_zarr(
                dask_arr,
                store,
                component=component,
                overwrite=True,     # dask-level kwarg; NOT forwarded to zarr.create_array
                compute=True,
                **zarr_kwargs,
            )

    # Reopen the written array and compute stats.  Read-only by design:
    # only info_complete() and the verification read follow, neither writes.
    group = zarr.open_group(store, mode="r")
    z = group[component]
    info = z.info_complete()
    count_bytes, count_bytes_stored = _info_bytes(info)
    ratio = count_bytes / count_bytes_stored
    if verbose and rank == 0:
        click.echo("-" * 80)
        click.echo(info)

    errors = None
    euclidean_distance = None
    if verify:
        with Timer("compute_errors_distances"):
            # Load back with shard-aligned (or inner-aligned) chunks for
            # efficient reads.
            z_dask = dask.array.from_zarr(z, chunks=write_unit)
            _pprint, errors, euclidean_distance, _nrm = \
                compute_errors_distances(z_dask, da.data)
        if verbose and rank == 0:
            click.echo("-" * 80)
            click.echo(_pprint)
            click.echo("-" * 80)
            click.echo(f"Euclidean Distance: {euclidean_distance}")
            click.echo("-" * 80)

    return ratio, errors, euclidean_distance


# =============================================================================
# ERROR METRICS  (used by the persist path; dask-lazy)
# =============================================================================

def compute_errors_distances(da_compressed, da):
    # Mask non-finite cells in the ORIGINAL (legitimate fill); score only
    # finite-in-original positions, mirroring evaluate_codec_pipeline.  A
    # cell finite in the original but non-finite in the decoded output is
    # corruption: excluded from the norms, counted in n_corrupt, surfaced
    # so the verify gate can reject it.
    finite_orig = np.isfinite(da)
    finite_dec  = np.isfinite(da_compressed)
    valid       = finite_orig & finite_dec
    n_corrupt   = (finite_orig & ~finite_dec).sum()

    # Zero out masked positions so the reductions stay finite; counts of
    # contributing cells are tracked via `valid`.  NOTE: the caller passes
    # raw dask arrays (da.data / from_zarr), which have no xarray-style
    # `.where()` method — use dask.array.where(cond, x, y) instead.
    o  = dask.array.where(valid, da, 0)
    dc = dask.array.where(valid, da_compressed, 0)
    da_error = dc - o

    norm_L1_error    = np.abs(da_error).sum()
    norm_L2_error    = np.sqrt((da_error ** 2).sum())
    norm_Linf_error  = np.abs(da_error).max()
    signed_error_sum = da_error.sum()

    norm_L1_original   = np.abs(o).sum()
    norm_L2_original   = np.sqrt((o ** 2).sum())
    norm_Linf_original = np.abs(o).max()

    computed = dask.compute(
        norm_L1_error, norm_L1_original,
        norm_L2_error, norm_L2_original,
        norm_Linf_error, norm_Linf_original,
        signed_error_sum, n_corrupt,
    )
    (l1e, l1o, l2e, l2o, linfe, linfo, signed, ncorrupt) = computed

    def _safe_rel(err, ori):
        """Relative error with sane behavior for zero-norm originals.
        Returns 0.0 when both error and original are zero (trivially correct),
        inf when error is non-zero but original is zero."""
        if ori == 0:
            return 0.0 if err == 0 else float("inf")
        return float(err) / float(ori)

    relative_error_L1   = _safe_rel(l1e,    l1o)
    relative_error_L2   = _safe_rel(l2e,    l2o)
    relative_error_Linf = _safe_rel(linfe,  linfo)
    bias_rel            = _safe_rel(abs(float(signed)), l1o)

    euclidean_distance = l2e
    normalized_euclidean_distance = relative_error_L2

    errors = {
        "Relative_Error_L1":   relative_error_L1,
        "Relative_Error_L2":   relative_error_L2,
        "Relative_Error_Linf": relative_error_Linf,
        "Bias_Rel":            bias_rel,
        "N_Corrupt":           int(ncorrupt),
    }
    errors_ = {k: (f"{v:.3e}" if isinstance(v, float) else str(v))
               for k, v in errors.items()}
    return (
        "\n".join(f"{k:20s}: {v}" for k, v in errors_.items()),
        errors,
        euclidean_distance,
        normalized_euclidean_distance,
    )


# =============================================================================
# PARALLELISM & TOPOLOGY
# =============================================================================

def detect_node_topology(comm=None):
    """
    Return (node_comm, ranks_on_node, local_rank).

    Uses the MPI-3 standard COMM_TYPE_SHARED split, with a hostname fallback
    for old MPI implementations that don't support it.  Works generically on
    any system without assumptions about cores, RAM, or cluster shape.
    """
    comm = comm or MPI.COMM_WORLD
    try:
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.Get_rank())
    except Exception:
        import socket
        node_name = socket.gethostname()
        all_names = comm.allgather(node_name)
        color_map = {n: i for i, n in enumerate(sorted(set(all_names)))}
        node_comm = comm.Split(color_map[node_name], key=comm.Get_rank())
    return node_comm, node_comm.Get_size(), node_comm.Get_rank()


def detect_cores_available() -> int:
    """Respect cgroups / Slurm cpusets where possible."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def compute_default_threads_per_rank(ranks_on_node: int, cores_available: int | None = None) -> int:
    if cores_available is None:
        cores_available = detect_cores_available()
    return max(1, cores_available // max(1, ranks_on_node))


def broadcast_numpy(arr, comm=None, root: int = 0) -> np.ndarray:
    """
    Broadcast a numpy array from `root` to all ranks using the MPI buffer
    protocol (Bcast, uppercase).  This is O(log N_ranks) in wall-time for
    large payloads, versus O(N_ranks) for the pickle-based `comm.bcast`.

    Usage
    -----
    Rank `root` passes the real array; non-root ranks pass None and receive
    the broadcast array as the return value:

        if rank == 0:
            payload = expensive_read()
        else:
            payload = None
        payload = broadcast_numpy(payload, comm=comm, root=0)

    Implementation
    --------------
    Done in two phases:
      1. Broadcast (shape, dtype) via pickle — a few bytes, cheap.
      2. Allocate a matching buffer on non-root ranks and Bcast the raw bytes.

    This keeps the fast path (step 2) off of pickle, which matters once the
    payload is >O(100 MB) - the whole point of using Bcast instead of bcast.

    Notes
    -----
    - Non-contiguous inputs are made contiguous on root before broadcast;
      the returned array is always C-contiguous.
    - The dtype is round-tripped through str(dtype) / np.dtype(str), which
      covers all the standard numpy dtypes used in this toolkit.  Custom
      structured dtypes are not supported.
    - Payload ceiling: MPI_Bcast's `count` argument is a C `int`
      (max 2^31 - 1).  We let mpi4py pick the MPI type from the numpy
      dtype (MPI.DOUBLE for float64, MPI.FLOAT for float32, etc.) instead
      of wrapping as MPI.BYTE.  This means `count` is `numel`, not
      `numel * itemsize`, so the ceiling scales with the element size:
      ~16 GB for float64, ~8 GB for float32.  An earlier implementation
      used `[buf, MPI.BYTE]` which silently capped at ~2 GB and would
      overflow or corrupt payloads at the toolkit's default 5 GB sample
      budget.  The wire protocol is identical; only the MPI type handle
      changes.
    """
    comm = comm or MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Phase 1: metadata (tiny; pickle is fine).
    if rank == root:
        if arr is None:
            raise ValueError(
                "broadcast_numpy: root rank must provide a numpy array, got None."
            )
        meta = (tuple(arr.shape), str(arr.dtype))
    else:
        meta = None
    shape, dtype_str = comm.bcast(meta, root=root)

    # Phase 2: contiguous buffer + Bcast on the buffer protocol.
    # Passing `buf` bare lets mpi4py infer the MPI type from the numpy
    # dtype.  Do NOT wrap as `[buf, MPI.BYTE]` - that reduces the effective
    # payload ceiling by a factor of itemsize (see docstring).
    if rank == root:
        buf = np.ascontiguousarray(arr)
    else:
        buf = np.empty(shape, dtype=np.dtype(dtype_str))
    comm.Bcast(buf, root=root)
    return buf


def check_thread_oversubscription(abort_if_unsafe: bool = True, rank: int = 0, comm=None) -> None:
    """
    Warn if codec-internal thread counts aren't pinned to 1.  Pin zarr v3's
    internal threadpool regardless (safe no-op on older zarr versions).

    If abort_if_unsafe is True (the default) and any env var is misconfigured,
    all ranks are killed collectively via comm.Abort(1).  This is intentional:
    sys.exit on rank 0 alone would hang the siblings at the next collective.
    """
    comm = comm or MPI.COMM_WORLD
    env_vars = [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "BLOSC_NTHREADS", "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "OMP_THREAD_LIMIT",
    ]
    problems = []
    for v in env_vars:
        val = os.environ.get(v)
        if val is None:
            problems.append(f"{v}=<unset>")
        else:
            try:
                if int(val) != 1:
                    problems.append(f"{v}={val}")
            except ValueError:
                problems.append(f"{v}={val}")

    if problems:
        if rank == 0:
            click.echo(
                "[oversubscription-check] WARNING: codec-internal thread "
                "variables not pinned to 1:"
            )
            for p in problems:
                click.echo(f"  - {p}")
            click.echo(
                "  With thread-per-combo parallelism this causes "
                "N_threads x M_internal oversubscription."
            )
            click.echo(
                "  Suggested: export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "OPENBLAS_NUM_THREADS=1 BLOSC_NTHREADS=1 NUMBA_NUM_THREADS=1 "
                "VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1"
            )
            if abort_if_unsafe:
                click.echo("  Aborting (use --no-oversubscription-check to override).")
        if abort_if_unsafe:
            # Collective abort: all ranks die, not just rank 0.  sys.exit on
            # rank 0 alone would leave siblings hanging at the next collective.
            comm.Abort(1)

    # Pin zarr v3's internal thread pool only when oversubscription is a real
    # risk: multi-rank-per-node (each rank's process otherwise spawns its own
    # default executor of ~32 workers, giving N_ranks * 32 threads on a
    # N-core node).  With 1 rank-per-node, no pin is needed: a single rank
    # uses one ~32-worker pool, which matches the 32 cores it's been given.
    # The bypass case has its own bounded shared executor — also no pin.
    if not _AsyncBypass.enabled:
        try:
            _, ranks_on_node, _ = detect_node_topology(MPI.COMM_WORLD)
            if ranks_on_node > 1:
                zarr.config.set({"threading.max_workers": 1})
        except Exception:
            pass


# =============================================================================
# MISC UTILITIES  (kept from original)
# =============================================================================

def get_indexes(arr, indices):
    codec_to_id = []
    for ind in indices:
        codec_to_id.append(ind[1:-1].split(", ", 1))
    id_ls = []
    codec_id_dict = {key: val for val, key in codec_to_id}
    for item in arr:
        if item == "None":
            id_ls.append(-1)
        elif item in list(codec_id_dict.keys()):
            id_ls.append(codec_id_dict[item])
        else:
            id_ls.append(-1)  # unknown item — append -1 instead of returning an exception object
    return np.asarray(id_ls)


def slice_array(arr: pd.array, indices_ls: list) -> np.ndarray:
    arr_ls = [arr[[ind]] for ind in indices_ls]
    return np.hstack(tuple(arr_ls))


def validate_percentage(ctx, param, value):
    if value is None:
        return None
    try:
        value = float(value)
    except ValueError:
        raise click.BadParameter("Percentage must be a number.")
    if not (1 <= value <= 99):
        raise click.BadParameter("Percentage must be between 1 and 99.")
    return value


# =============================================================================
# PROGRESS BAR  (thread-safe)
# =============================================================================

_PROGRESS_LOCK = threading.Lock()
_PROGRESS_COUNTERS = defaultdict(int)


def progress_bar(total_configs, print_every=100, bar_width=40, key: str = "default"):
    """
    Thread-safe progress bar.  Rank 0 only.  `key` distinguishes concurrent
    progress streams if ever needed.  Call once per completed unit of work;
    the counter is tracked internally via `_PROGRESS_COUNTERS[key]`.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        return
    with _PROGRESS_LOCK:
        _PROGRESS_COUNTERS[key] += 1
        done = _PROGRESS_COUNTERS[key]
        percent = done / total_configs
        filled = int(bar_width * percent)
        bar = "*" * filled + "-" * (bar_width - filled)
        if done % print_every == 0 or done == total_configs:
            click.echo(
                f"[Rank {rank}] Progress: |{bar}| {percent*100:6.2f}% "
                f"({done}/{total_configs})"
            )


# =============================================================================
# TIMING  (thread-safe)
# =============================================================================

_TIMINGS_LOCK = threading.Lock()
_TIMINGS = defaultdict(list)


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        with _TIMINGS_LOCK:
            _TIMINGS[self.label].append(time.perf_counter() - self.start)


@atexit.register
def print_profile_summary():
    if not _TIMINGS:
        return
    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    print("\n=== Timing Summary (rank 0; ranks balanced via deterministic shuffle) ===")
    print("Sum of Total = thread-seconds inside the eval pipeline (excludes bcast,")
    print("file I/O, dask graph setup, and result-write overhead).")
    print()
    label_width = max(len(label) for label in _TIMINGS.keys())
    # Per-label totals; grand total is the sum across all labels and is the
    # denominator for the % column ("how much of the eval pipeline did this
    # phase consume?").  Edge case: if no time was recorded, show 0% to
    # avoid a ZeroDivisionError.
    totals = {label: sum(durations) for label, durations in _TIMINGS.items()}
    grand_total = sum(totals.values()) or 1.0
    header = (f"{'Label':<{label_width}} | {'Calls':>5} | {'Avg (s)':>10} | "
              f"{'Total (s)':>12} | {'% total':>7}")
    print(header)
    print("-" * len(header))
    for label, durations in sorted(_TIMINGS.items()):
        total = totals[label]; count = len(durations); avg = total / count
        pct = 100.0 * total / grand_total
        print(f"{label:<{label_width}} | {count:>5} | {avg:>10.6f} | "
              f"{total:>12.6f} | {pct:>6.2f}%")
    print("=" * len(header))
