"""Library behind the ``dc_toolkit`` CLI.  Sections:
  1. Sizes & dataset I/O
  2. Representative sampling      (which slices of a big field to score)
  3. Chunk & shard sizing         (zarr geometry, shared by eval and persist)
  4. Codec spaces & pipelines     (compressor x filter x serializer grids, EBCC optional)
  5. In-memory evaluation         (encode -> decode -> error metrics)
  6. Persistence                  (dask -> zarr LocalStore, optional verify)
  7. MPI & topology
  8. Progress & timing"""
import atexit
import json
import math
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional, Tuple

import click
import dask
import dask.array
import humanize
import numpy as np
import psutil
import xarray as xr
import zarr
import zfpy
from mpi4py import MPI
from zarr.core.sync import sync as _zarr_sync
from zarr.codecs import numcodecs as zarrcodecs_nc
from zarr.registry import get_codec_class

from dc_toolkit.codecs import EBCC, EBCC_AVAILABLE, _EBCC_TILE_MAX, _EBCC_TILE_MIN, ZFPYFlat, ZFPYRank


class CombinationProducedNonFiniteError(Exception):
    """A codec combination produced non-finite error accumulators."""


class SampleTooLargeError(Exception):
    """One horizontal slab of the field already exceeds the sample budget."""

    def __init__(self, message, irreducible_bytes=None, size_limit_bytes=None,
                 dims=None, spatial_dims=None):
        super().__init__(message)
        self.irreducible_bytes = irreducible_bytes
        self.size_limit_bytes = size_limit_bytes
        self.dims = dims
        self.spatial_dims = spatial_dims


# Codec-internal thread counts; check_thread_oversubscription requires 1, since a sweep rank or a
# compress dask thread owns a core.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLOSC_NTHREADS", "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "OMP_THREAD_LIMIT",
)
# Read by the EBCC C library at encode time; they change its output, so the manifest records them.
EBCC_ENV_VARS = (
    "EBCC_INIT_BASE_ERROR_QUANTILE", "EBCC_ERROR_BOUND_SLACK", "EBCC_DISABLE_MEAN_ADJUSTMENT",
    "EBCC_DISABLE_PURE_BASE_COMPRESSION_FALLBACK",
    "EBCC_DISABLE_PURE_BASE_COMPRESSION_FALLBACK_CONSISTENCY", "EBCC_ERROR_BOUND_STRICT_MODE",
)


def abort(code: int = 1) -> None:
    """Exit the job: MPI Abort when multi-rank (a lone sys.exit would hang the others), else sys.exit."""
    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.Abort(code)
    sys.exit(code)


# =============================================================================
# 1. SIZES & DATASET I/O
# =============================================================================

_SIZE_UNITS = {
    "B": 1,
    "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
    "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40,
}


def parse_size(size_str) -> int:
    """'5GB' / '512MiB' / '1000' -> bytes."""
    if isinstance(size_str, (int, float)):
        return int(size_str)
    s = str(size_str).strip().upper().replace(" ", "")
    for unit in sorted(_SIZE_UNITS, key=len, reverse=True):
        if s.endswith(unit):
            return int(float(s[:-len(unit)]) * _SIZE_UNITS[unit])
    return int(float(s))


def hsize(nbytes) -> str:
    """Bytes -> '1.2 GiB'."""
    return humanize.naturalsize(nbytes, binary=True)


def open_zarr_localstore(path: str, read_only: bool = True):
    """(group, store) of a zarr v3 LocalStore; keep both alive.  Skips (possibly stale) consolidated metadata."""
    store = zarr.storage.LocalStore(path, read_only=read_only)
    return zarr.open_group(store, mode="r" if read_only else "a", use_consolidated=False), store


def open_dataset(dataset_file: str, field_to_compress: Optional[str] = None, rank: int = 0, chunks="auto"):
    """Lazy .nc / .grib / .zarr dataset, dask-backed unless `chunks` is None; aborts on another suffix or a
    missing field.  A netCDF float variable declaring no _FillValue or missing_value gets netCDF's default
    fill as its _FillValue: cells never written hold it, and would enter every norm as 9.97e36."""
    suffix = Path(dataset_file).suffix.lower()
    if suffix == ".nc":
        ds = xr.open_dataset(dataset_file, chunks=chunks)
        undeclared = [v for v, x in ds.data_vars.items()
                      if np.dtype(x.encoding.get("dtype", x.dtype)).kind == "f"
                      and not {"_FillValue", "missing_value"} & (set(x.encoding) | set(x.attrs))]
        if undeclared:
            import netCDF4
            raw = xr.open_dataset(dataset_file, chunks=chunks, decode_cf=False)
            for v in undeclared:
                raw[v].attrs["_FillValue"] = netCDF4.default_fillvals[np.dtype(raw[v].dtype).str[1:]]
            ds = xr.decode_cf(raw)
    elif suffix == ".grib":
        ds = xr.open_dataset(dataset_file, chunks=chunks, engine="cfgrib",
                             backend_kwargs={"indexpath": ""})
    elif suffix == ".zarr":
        ds = xr.open_zarr(dataset_file, chunks=chunks, consolidated=None)
    else:
        if rank == 0:
            click.echo(f"Unsupported file format: {suffix}. Only .nc / .grib / .zarr are supported.")
        abort(1)

    if field_to_compress is not None and field_to_compress not in ds.data_vars:
        if rank == 0:
            click.echo(f"Field {field_to_compress} not found in dataset. "
                       f"Available fields: {list(ds.data_vars)}.")
        abort(1)

    if rank == 0:
        click.echo(f"dataset.nbytes = {hsize(ds.nbytes)}")
        if field_to_compress is not None:
            click.echo(f"{field_to_compress}.nbytes = {hsize(ds[field_to_compress].nbytes)}")
    return ds


def is_lat_lon(da) -> bool:
    dims = da.dims
    return len(dims) == 2 and re.search(r"lat", dims[0]) is not None \
        and re.search(r"lon", dims[1]) is not None


# =============================================================================
# 2. REPRESENTATIVE SAMPLING
# =============================================================================
# Horizontal dims are never thinned, so codecs see the real spatial structure.

SAMPLE_MIN_KEEP = 3  # indices a sample keeps at least across the time-like dims, and across the vertical ones

_TIME_LIKE_DIM_RE = re.compile(
    r"^(?:time|.*_time|t|step|forecast_reference_time|forecast_period|"
    r"ensemble|realization|member|member_id|reftime|valid_time|epoch)$", re.IGNORECASE)
_CF_TIME_UNITS_RE = re.compile(r"^\s*\w+\s+since\s+", re.IGNORECASE)
_CF_TIME_STANDARD_NAMES = frozenset({"time", "forecast_reference_time", "forecast_period", "realization"})
_CF_VERTICAL_STANDARD_NAMES = frozenset({
    "height", "altitude", "depth", "air_pressure", "pressure", "model_level_number",
    "atmosphere_hybrid_sigma_pressure_coordinate", "atmosphere_hybrid_height_coordinate",
    "atmosphere_sigma_coordinate", "atmosphere_ln_pressure_coordinate",
    "atmosphere_sleve_coordinate",
})
_VERTICAL_DIM_NAMES = {
    "lev", "level", "levels", "plev", "plevs", "pressure", "pressure_level",
    "height", "altitude", "alt", "depth", "z", "model_level", "model_level_number", "ml",
    "vertical", "vert", "bottom_top", "bottom_top_stag", "mlev", "ilev", "lev_p", "lev_l",
    "soil_layers_stag", "isobaric", "isobaric1", "isobaric2", "sigma", "sigma_level",
    "hybrid", "hybrid_level",
}


def _coord_meta(da: xr.DataArray, dim: str) -> dict:
    """attrs + encoding of `dim`'s coord ({} if none): xarray moves decoded CF attrs to .encoding."""
    coord = da.coords.get(dim)
    return {} if coord is None else {**dict(coord.attrs), **dict(coord.encoding)}


def _is_time_like_coord(da: xr.DataArray, dim: str) -> bool:
    m = _coord_meta(da, dim)
    units, std = m.get("units"), m.get("standard_name")
    if isinstance(units, str) and _CF_TIME_UNITS_RE.match(units):
        return True
    if (isinstance(std, str) and std in _CF_TIME_STANDARD_NAMES) or m.get("axis") == "T" or "calendar" in m:
        return True
    return bool(_TIME_LIKE_DIM_RE.match(dim))


def _is_vertical_like_dim(name) -> bool:
    """Name-only heuristic (also used by the chunk shrink order)."""
    if name is None:
        return False
    n = str(name).lower().strip()
    return (n in _VERTICAL_DIM_NAMES
            or n.startswith(("lev", "plev", "ilev", "mlev"))
            or n.endswith(("_lev", "_level", "_levels")))


def _is_vertical_like_coord(da: xr.DataArray, dim: str) -> bool:
    m = _coord_meta(da, dim)
    std = m.get("standard_name")
    if m.get("axis") == "Z" or (isinstance(std, str) and std.lower() in _CF_VERTICAL_STANDARD_NAMES):
        return True
    if str(m.get("positive", "")).lower() in ("up", "down"):
        return True
    return _is_vertical_like_dim(dim)


def _classify_sample_dims(da: xr.DataArray):
    """(time_dims, vertical_dims, spatial_dims) as lists of (pos, name), by CF metadata, then by name."""
    time_dims, vertical_dims, spatial_dims = [], [], []
    for i, name in enumerate(da.dims):
        if _is_time_like_coord(da, name):
            time_dims.append((i, name))
        elif _is_vertical_like_coord(da, name):
            vertical_dims.append((i, name))
        else:
            spatial_dims.append((i, name))
    return time_dims, vertical_dims, spatial_dims


def horizontal_axes(da: xr.DataArray) -> tuple:
    """Positions of the dims that are neither time nor vertical and have more than one index."""
    return tuple(pos for pos, name in _classify_sample_dims(da)[2] if da.sizes[name] > 1)


def minimum_sample(da: xr.DataArray) -> tuple:
    """(bytes, description) of the smallest sample build_representative_sample takes from `da`: one
    horizontal slab times the fewest time steps and levels it keeps, SAMPLE_MIN_KEEP of each (all,
    where there are fewer); the whole field when it has no time or vertical dim."""
    time_dims, vertical_dims, spatial_dims = _classify_sample_dims(da)
    if not time_dims + vertical_dims:
        return int(da.nbytes), "the whole field, which has no time or vertical dim to thin"
    keep, parts = 1, []
    for group, label in ((time_dims, "time step"), (vertical_dims, "level")):
        if group:
            plan = _distribute_group([(i, n, int(da.sizes[n])) for i, n in group], SAMPLE_MIN_KEEP)
            keep *= int(np.prod(list(plan.values())))
            k = plan[group[0][1]]
            parts.append(f"{k} {label}{'s' if k != 1 else ''}" if len(group) == 1
                         else " x ".join(f"{n}={plan[n]}" for _, n in group))
    slab = int(da.dtype.itemsize) * int(np.prod([da.sizes[d] for _, d in spatial_dims]))
    return min(int(da.nbytes), slab * keep), " x ".join(parts)


def build_representative_sample(da: xr.DataArray, size_limit_bytes: int, rank: int = 0,
                                policy: str = "cascade", vertical_floor: int | None = None,
                                candidates: Optional[dict] = None) -> xr.DataArray:
    """Deterministic subset of `da` within `size_limit_bytes`, but never below minimum_sample:
    along each time and vertical dim, the middle index of equal blocks of its `candidates` (dim ->
    indices; default all), e.g. the levels on which the field varies (without such dims, the whole
    field and a warning).  "cascade" keeps its floor of levels, then spends the budget on time steps;
    "balanced" splits it evenly in log space.  Raises SampleTooLargeError when one horizontal slab
    does not fit."""
    nbytes = int(da.nbytes)
    if nbytes <= size_limit_bytes:
        if rank == 0:
            click.echo(f"[sample] field fits under limit ({hsize(nbytes)} <= "
                       f"{hsize(size_limit_bytes)}); evaluating on full field.")
        return da

    time_dims, vertical_dims, spatial_dims = _classify_sample_dims(da)
    stride_dims = time_dims + vertical_dims
    if not stride_dims:
        if rank == 0:
            click.echo(f"[sample] WARNING: '{da.name}' dims {da.dims} have no time/vertical "
                       f"axis; cannot stride-sample.  Using the full field ({hsize(nbytes)}) "
                       f"above the {hsize(size_limit_bytes)} cap.")
        return da

    spatial_sizes = [da.sizes[d] for _, d in spatial_dims]
    irreducible_bytes = int(da.dtype.itemsize) * int(np.prod(spatial_sizes) if spatial_sizes else 1)
    if irreducible_bytes > size_limit_bytes:
        spatial_names = [d for _, d in spatial_dims]
        msg = (f"variable '{da.name}': one horizontal slab is {hsize(irreducible_bytes)} "
               f"(spatial dims {spatial_names}), above the {hsize(size_limit_bytes)} budget.  "
               f"Remedies: raise --eval-data-size-limit, start fewer ranks per node, "
               f"or request more RAM (#SBATCH --mem=0).")
        if rank == 0:
            click.echo(f"[sample] FATAL: {msg}")
        raise SampleTooLargeError(msg, irreducible_bytes=irreducible_bytes,
                                  size_limit_bytes=int(size_limit_bytes),
                                  dims=tuple(da.dims), spatial_dims=tuple(spatial_names))

    max_product = float(size_limit_bytes) / float(irreducible_bytes)
    cand = {n: np.asarray((candidates or {}).get(n, np.arange(da.sizes[n])), dtype=int) for _, n in stride_dims}
    cand = {n: (c if c.size else np.arange(da.sizes[n])) for n, c in cand.items()}
    plan = _allocate_stride_plan(da, time_dims, vertical_dims, max_product, policy=policy,
                                 vertical_floor=vertical_floor, sizes={n: c.size for n, c in cand.items()})
    isel = {}
    for _, name in stride_dims:
        c, n_keep = cand[name], plan[name]
        if n_keep < da.sizes[name]:  # block midpoints: the ends of a dim (model top, first step) are the least typical
            isel[name] = c[((np.arange(n_keep) + 0.5) * c.size / n_keep).astype(int)].tolist()
    sampled = da.isel(isel) if isel else da

    if rank == 0:
        strided = ", ".join(f"{n}={plan[n]}/{da.sizes[n]}"
                            + (f" {isel[n]}" if n in isel and len(isel[n]) <= 8 else "")
                            + (f" (of the {cand[n].size} that vary)" if cand[n].size < da.sizes[n] else "")
                            for _, n in stride_dims)
        spatial = (" | preserved spatial: " + ", ".join(d for _, d in spatial_dims)) if spatial_dims else ""
        click.echo(f"[sample] field is {hsize(nbytes)} > limit {hsize(size_limit_bytes)}; "
                   f"policy={policy}; strided {strided}{spatial} -> {hsize(int(sampled.nbytes))}.")
    return sampled


def _balanced_group_plan(dims_info, budget) -> dict:
    """Log-space split of `budget` indices across dims [(pos, name, size)]."""
    plan, remaining, n_left = {}, float(budget), len(dims_info)
    for _, name, size in sorted(dims_info, key=lambda t: t[2]):
        target = remaining ** (1.0 / n_left) if n_left > 0 else 1.0
        plan[name] = max(1, min(size, int(target)))
        remaining /= max(1, plan[name])
        n_left -= 1
    return plan


def _distribute_group(dims_info, group_keep) -> dict:
    """{dim: n_keep} for one group of dims, about `group_keep` indices in all and never fewer than
    SAMPLE_MIN_KEEP (or the whole group, when it has fewer)."""
    if not dims_info:
        return {}
    floor = min(SAMPLE_MIN_KEEP, int(np.prod([size for _, _, size in dims_info])))
    if len(dims_info) == 1:
        _, name, size = dims_info[0]
        return {name: min(size, max(floor, int(group_keep)))}
    plan = _balanced_group_plan(dims_info, group_keep)
    for _, name, size in sorted(dims_info, key=lambda t: -t[2]):  # small dims split the floor unevenly
        while int(np.prod(list(plan.values()))) < floor and plan[name] < size:
            plan[name] += 1
    return plan


def _allocate_stride_plan(da, time_dims, vertical_dims, max_product,
                          policy="cascade", vertical_floor=None, sizes=None) -> dict:
    """{dim_name: n_keep} for every time and vertical dim, at least SAMPLE_MIN_KEEP time steps and
    levels (or all, where there are fewer) even when `max_product` slabs cannot hold them; `sizes`
    (dim -> indices to choose from) overrides the dims' lengths."""
    size = lambda n: int((sizes or {}).get(n, da.sizes[n]))  # noqa: E731
    time_info = [(i, n, size(n)) for i, n in time_dims]
    vert_info = [(i, n, size(n)) for i, n in vertical_dims]
    T = int(np.prod([s for _, _, s in time_info])) if time_info else 1
    V = int(np.prod([s for _, _, s in vert_info])) if vert_info else 1
    t_min, v_min = min(T, SAMPLE_MIN_KEEP), min(V, SAMPLE_MIN_KEEP)
    kept = lambda plan, info: int(np.prod([plan[n] for _, n, _ in info])) if info else 1  # noqa: E731
    # what each group keeps at the least: several small dims can overshoot the minimum (2 x 2 for 3)
    t_floor = kept(_distribute_group(time_info, t_min), time_info)
    v_floor = kept(_distribute_group(vert_info, v_min), vert_info)
    P = max(float(max_product), float(t_floor * v_floor))
    if policy == "balanced":
        plan = _balanced_group_plan(time_info + vert_info, P)
        if kept(plan, time_info) < t_min:
            plan.update(_distribute_group(time_info, t_min))
            plan.update(_distribute_group(vert_info, max(v_min, P // kept(plan, time_info))))
        elif kept(plan, vert_info) < v_min:
            plan.update(_distribute_group(vert_info, v_min))
            plan.update(_distribute_group(time_info, max(t_min, P // kept(plan, vert_info))))
        return plan

    if vertical_floor is not None:
        vfloor = int(vertical_floor)
    else:
        vfloor = max(4, int(math.ceil(math.log2(V)))) if V > 1 else 1
    # The floor of levels shrinks to what leaves room for the minimum of time steps, never below v_floor.
    level_target = max(v_floor, min(vfloor, V, int(P // t_floor)))
    time_plan = _distribute_group(time_info, max(t_min, min(T, int(P // level_target))))
    return {**time_plan, **_distribute_group(vert_info, max(v_min, min(V, int(P // kept(time_plan, time_info)))))}


# =============================================================================
# 3. CHUNK & SHARD SIZING
# =============================================================================

def _shrink_order(shape, dims) -> list:
    """Non-leading axes in shrink order: horizontal, then vertical, fastest first (hiopy convention)."""
    ndim = len(shape)
    if ndim <= 1:
        return []
    if dims is None or len(dims) != ndim:
        return list(range(ndim - 1, 0, -1))
    horizontal, vertical = [], []
    for i, name in enumerate(dims):
        if i:
            (vertical if _is_vertical_like_dim(name) else horizontal).append(i)
    return sorted(horizontal, reverse=True) + sorted(vertical, reverse=True)


def _compute_inner_chunk_shape(shape, dtype, dims, target_bytes: int,
                               allow_spatial_split: bool = True) -> Tuple[int, ...]:
    """Pack leading slices up to target_bytes; an oversized slice keeps 1 index and may shrink spatially."""
    itemsize = int(np.dtype(dtype).itemsize)
    shape = tuple(int(s) for s in shape)
    if not shape:
        return ()
    inner = list(shape)
    bytes_per_leading = itemsize * (int(np.prod(shape[1:])) if len(shape) > 1 else 1)
    if bytes_per_leading == 0:
        return tuple(inner)
    if bytes_per_leading <= target_bytes:
        inner[0] = int(min(shape[0], max(1, target_bytes // bytes_per_leading)))
        return tuple(inner)

    inner[0] = 1
    if not allow_spatial_split:
        return tuple(inner)
    for axis in _shrink_order(shape, dims):
        chunk_bytes = itemsize * int(np.prod(inner))
        if chunk_bytes <= target_bytes:
            break
        per_row = chunk_bytes // inner[axis] if inner[axis] > 0 else chunk_bytes
        inner[axis] = 1 if per_row <= 0 else int(min(inner[axis], max(1, target_bytes // per_row)))
    return tuple(int(x) for x in inner)


def compute_chunk_shape_for_eval(shape, dtype, target_mib: int = 16, dims=None,
                                 allow_spatial_split: bool = True) -> Tuple[int, ...]:
    """Chunk shape for in-memory evaluation (same algorithm as persist)."""
    return _compute_inner_chunk_shape(shape, dtype, dims, int(target_mib) * 2**20,
                                      allow_spatial_split=allow_spatial_split)


def compute_chunk_and_shard_shape(shape, dtype, inner_mib: int = 16, shard_mib: int = 512,
                                  dims=None, allow_spatial_split: bool = True, inner_chunks=None):
    """(inner_chunk_shape, shard_shape), shard_shape None when a shard would hold fewer than two
    inner chunks.  `inner_chunks` forces the inner chunk shape (codecs with a fixed frame)."""
    itemsize = int(np.dtype(dtype).itemsize)
    if inner_chunks is not None:
        inner = tuple(int(x) for x in inner_chunks)
    else:
        inner = _compute_inner_chunk_shape(shape, dtype, dims, int(inner_mib) * 2**20,
                                           allow_spatial_split=allow_spatial_split)
    inner_bytes = itemsize * int(np.prod(inner))
    shard_target = int(shard_mib) * 2**20
    if inner_bytes == 0 or inner_bytes >= shard_target:
        return inner, None
    multiplier = shard_target // inner_bytes
    if multiplier <= 1:
        return inner, None
    shard = list(inner)
    shard[0] = max(inner[0], (min(int(shape[0]), inner[0] * multiplier) // inner[0]) * inner[0])
    shard = tuple(int(x) for x in shard)
    return inner, (None if shard == inner else shard)


# =============================================================================
# 4. CODEC SPACES & PIPELINES
# =============================================================================
# Result rows, resume state and manifests identify a combination by its pipeline dict
# (pipeline_to_dict), never by its position in the codec lists built from the grids below.

_BLOSC_CNAMES = ("lz4", "lz4hc", "zstd")
_BLOSC_CLEVELS = (1, 5, 9)
_BLOSC_SHUFFLES = (0, 1)
_LZ4_ACCELERATIONS = (1, 10, 100)
_ZSTD_LEVELS = (6, 12, 22)
_ZLIB_LEVELS = (3, 6, 9)
_BZ2_LEVELS = (3, 6, 9)
_LZMA_PRESETS = (3, 6, 9)
# The top value of each filter grid is effectively lossless.
_BITROUND_KEEPBITS_F32 = (3, 7, 11, 13, 17, 23)
_BITROUND_KEEPBITS_F64 = (3, 7, 11, 17, 23, 30, 37, 44, 52)
_QUANTIZE_DIGITS_F32 = (1, 3, 4, 5, 6, 7)
_QUANTIZE_DIGITS_F64 = (1, 3, 4, 5, 6, 7, 9, 11, 13, 15)
# No uint8: PCodec refuses 8-bit input and ZFPY every FixedScaleOffset output (combo_is_valid).
_FSO_TARGET_UINTS = ("uint16", "uint32")
_PCODEC_LEVELS = (6, 8, 10, 12)
_PCODEC_DELTA_ORDERS = (0, 7)
_ZFPY_K_GRID = (0, 1, 2, 3)  # rate/precision: 8/16/32/64 bits; accuracy: 0.5/0.25/0.0625/0.0039


def compute_fixed_precision_param(k: int) -> int: return 1 << (k + 3)
def compute_fixed_rate_param(k: int) -> int: return 1 << (k + 3)
def compute_fixed_accuracy_param(k: int) -> float: return math.ldexp(1.0, -(1 << k))


def _select_classes(classes, requested: str, kind: str):
    """Resolve 'all' / '<name>' / 'none' into (classes, include_none)."""
    req = requested.lower()
    by_name = {cls.__name__.lower(): cls for cls in classes}
    if req == "all":
        return list(classes), False
    if req == "none":
        return [], True
    if req in by_name:
        return [by_name[req]], False
    raise ValueError(f"Unknown {kind} class '{requested}'. "
                     f"Choose one of: all, none, {', '.join(by_name)}.")


def compressor_space(da, with_lossy=True, compressor_class="all"):
    """Lossless bytes->bytes compressors (`with_lossy` is unused).  Byte shuffling comes only from
    Blosc's `shuffle` (a standalone Shuffle breaks after a compressing serializer), with `typesize`
    set to the item size: zarr passes these codecs raw bytes, and a 1-byte shuffle is a no-op."""
    classes, include_none = _select_classes(
        [zarrcodecs_nc.Blosc, zarrcodecs_nc.LZ4, zarrcodecs_nc.Zstd,
         zarrcodecs_nc.Zlib, zarrcodecs_nc.BZ2, zarrcodecs_nc.LZMA],
        compressor_class, "compressor")
    space = [None] if include_none else []
    for cls in classes:
        if cls is zarrcodecs_nc.Blosc:
            space += [cls(cname=c, clevel=l, shuffle=s, typesize=int(da.dtype.itemsize))
                      for c in _BLOSC_CNAMES for l in _BLOSC_CLEVELS for s in _BLOSC_SHUFFLES]
        elif cls is zarrcodecs_nc.LZ4:
            space += [cls(acceleration=a) for a in _LZ4_ACCELERATIONS]
        elif cls is zarrcodecs_nc.Zstd:
            space += [cls(level=l) for l in _ZSTD_LEVELS]
        elif cls is zarrcodecs_nc.Zlib:
            space += [cls(level=l) for l in _ZLIB_LEVELS]
        elif cls is zarrcodecs_nc.BZ2:
            space += [cls(level=l) for l in _BZ2_LEVELS]
        elif cls is zarrcodecs_nc.LZMA:
            space += [cls(preset=p) for p in _LZMA_PRESETS]
    return space


def filter_space(da, with_lossy=True, filter_class="all", data_range=None):
    """Array->array filters; integer dtypes get Delta only, and a float field without `with_lossy` gets no
    filter (Delta is not exactly invertible on floats).  FixedScaleOffset needs `data_range`, the FULL
    field's (min, max).  Raises ValueError when the requested class has nothing for this field."""
    if da.dtype.kind == "f" and not with_lossy:
        if filter_class.lower() in ("all", "none"):
            return [None]
        raise ValueError(f"--filter-class {filter_class} has nothing lossless for a float field; use none")
    classes = [zarrcodecs_nc.Delta]
    if with_lossy:
        classes += [zarrcodecs_nc.BitRound, zarrcodecs_nc.Quantize, zarrcodecs_nc.FixedScaleOffset]
        if np.issubdtype(da.dtype, np.floating) and da.dtype.itemsize > 4:
            classes.append(zarrcodecs_nc.AsType)
    if da.dtype.kind in "iu":
        if filter_class.lower() not in ("all", "delta", "none"):
            if MPI.COMM_WORLD.Get_rank() == 0:
                click.echo(f"[filter_space] integer dtype {da.dtype}: only Delta is available; "
                           f"ignoring --filter-class={filter_class}.", err=True)
            filter_class = "all"
        classes = [zarrcodecs_nc.Delta]
    classes, include_none = _select_classes(classes, filter_class, "filter")

    space = [None] if include_none else []
    for cls in classes:
        if cls is zarrcodecs_nc.Delta:
            if np.issubdtype(da.dtype, np.number):
                space.append(cls(dtype=str(da.dtype)))
        elif cls is zarrcodecs_nc.BitRound:
            space += [cls(keepbits=k) for k in valid_keepbits_for_bitround(da)]
        elif cls is zarrcodecs_nc.Quantize:
            space += [cls(digits=d, dtype=str(da.dtype)) for d in valid_digits_for_quantize(da)]
        elif cls is zarrcodecs_nc.FixedScaleOffset:
            space += [cls(**cfg) for cfg in fixed_scale_offset_configs(da, data_range=data_range)]
        elif cls is zarrcodecs_nc.AsType:
            space.append(cls(encode_dtype="float32", decode_dtype=str(da.dtype)))
    if not space:
        raise ValueError(f"--filter-class {filter_class} has nothing for this field"
                         + (": FixedScaleOffset needs the full field's value range"
                            if zarrcodecs_nc.FixedScaleOffset in classes else ""))
    return space


# ---- EBCC (optional): the sweep's error targets and tiles -----------------------
# Maximum absolute error targets as fractions of the FULL field's value range (the paper's
# 0.1 %..10 % band plus one decade below); EBCC's own floor is range/65535 (uint16 base layer).
_EBCC_ERROR_FRACTIONS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)


def _ebcc_tile_size(n: int):
    """Largest divisor of n within [32, 2047], or None."""
    if n < _EBCC_TILE_MIN:
        return None
    if n <= _EBCC_TILE_MAX:
        return n
    return next((d for d in range(_EBCC_TILE_MAX, _EBCC_TILE_MIN - 1, -1) if n % d == 0), None)


def ebcc_tile(da):
    """((height, width), "") of the EBCC tile over a float field's last two (lat, lon) dims, or (None, why)."""
    if not EBCC_AVAILABLE:
        return None, "the ebcc package is not installed"
    if da.ndim < 2 or da.dtype.kind != "f":
        return None, "needs a float field with at least 2 dims"
    spatial = {name for _, name in _classify_sample_dims(da)[2]}
    frame = tuple(da.dims[-2:])
    if not all(d in spatial for d in frame):
        return None, f"last two dims {frame} are not a (lat, lon) frame"
    sizes = tuple(int(da.sizes[d]) for d in frame)
    tile = (_ebcc_tile_size(sizes[0]), _ebcc_tile_size(sizes[1]))
    if None in tile:
        return None, f"no tile in [{_EBCC_TILE_MIN}, {_EBCC_TILE_MAX}] divides the frame {sizes}"
    return tile, ""


def ebcc_chunks(codec: EBCC, shape) -> Tuple[int, ...]:
    """One EBCC tile per chunk: (1, ..., 1, height, width)."""
    return (1,) * (len(shape) - 2) + (codec.height, codec.width)


def ebcc_sweep_entries(filters, serializers, dtype, all_finite: bool):
    """(triples, reason): one (None, filter, EBCC) triple per EBCC serializer, the filter None for
    float32 and else the float32 AsType, which EBCC needs even when --filter-class left it out of
    `filters`.  Empty, with a reason, when `all_finite` is False (EBCC cannot encode NaN/Inf)."""
    ebccs = [s for s in serializers if isinstance(s, EBCC)]
    if not ebccs:
        return [], ""
    if not all_finite:
        return [], "the field holds NaN/Inf, which EBCC cannot encode"
    dtype, filt = np.dtype(dtype), None
    if dtype != np.float32:
        astype = [f for f in filters if isinstance(f, zarrcodecs_nc.AsType)]
        filt = astype[0] if astype else zarrcodecs_nc.AsType(encode_dtype="float32", decode_dtype=str(dtype))
    return [(None, filt, s) for s in ebccs], ""


def serializer_space(da, with_lossy=True, serializer_class="all", with_ebcc=False, data_range=None,
                     chunk_shapes=None):
    """Array->bytes serializers; plain bytes with 'all' and 'none' (lossy filter + lossless compressor
    is the classic recipe, and all an 8-bit field gets: pco refuses those).  ZFPY needs `with_lossy`
    and floats (fixed-rate zfp is never exact on integers, and Delta, their only filter, turns that
    into a random walk on decode).  EBCC needs `with_ebcc` (or class 'ebcc'), `with_lossy` and float
    (lat, lon) frames; `data_range` (full-field min, max) scales its error targets, else they are
    relative to each tile.  ZFPYFlat is planned when `chunk_shapes` (the sample's and store's chunks)
    is not given or ZFPYRank encodes one of them at 2-D or more.  Raises ValueError when the space is
    empty."""
    if serializer_class.lower() == "ebcc":
        with_ebcc = True
    zfp_ok = da.dtype.kind == "f"
    if serializer_class.lower() == "zfpy" and not (zfp_ok and with_lossy):
        raise ValueError("ZFPY is lossy: --serializer-class zfpy needs --with-lossy" if zfp_ok else
                         f"ZFPY takes floats only, not {da.dtype}")
    classes = [zarrcodecs_nc.PCodec] + ([zarrcodecs_nc.ZFPY] if with_lossy and zfp_ok else [])
    if with_ebcc and with_lossy:
        if not EBCC_AVAILABLE:
            raise ValueError("--with-ebcc needs the ebcc package: pip install -e '.[ebcc]'")
        classes.append(EBCC)
    classes, include_none = _select_classes(classes, serializer_class, "serializer")
    if with_ebcc and with_lossy and EBCC not in classes:
        classes.append(EBCC)                     # --with-ebcc holds under a named class too
    space, why = ([None] if include_none or serializer_class.lower() == "all" else []), None
    for cls in classes:
        if cls is zarrcodecs_nc.PCodec:
            space += [cls(level=l, mode_spec="auto", delta_spec="auto", delta_encoding_order=d)
                      for l in _PCODEC_LEVELS for d in _PCODEC_DELTA_ORDERS]
        elif cls is zarrcodecs_nc.ZFPY:
            modes = [
                (zfpy.mode_fixed_accuracy, "tolerance", compute_fixed_accuracy_param),
                (zfpy.mode_fixed_precision, "precision", compute_fixed_precision_param),
                (zfpy.mode_fixed_rate, "rate", compute_fixed_rate_param),
            ]
            variants = [ZFPYRank]
            if not chunk_shapes or any(ZFPYFlat.encode_shape(c) != ZFPYRank.encode_shape(c) for c in chunk_shapes):
                variants.append(ZFPYFlat)
            space += [v(mode=mode, **{param: fn(k)})
                      for v in variants for mode, param, fn in modes for k in _ZFPY_K_GRID]
        elif cls is EBCC:
            tile, why = ebcc_tile(da)
            if tile is not None:
                span = float(data_range[1] - data_range[0]) if data_range else None
                space += [EBCC.from_params(*tile, r * span) if span else
                          EBCC.from_params(*tile, r, mode="relative_error_target")
                          for r in _EBCC_ERROR_FRACTIONS]
    if da.dtype.itemsize == 1:
        space = [s for s in space if not isinstance(s, zarrcodecs_nc.PCodec)]
        why = why or "pcodec does not take 8-bit fields"
    if not space:
        raise ValueError(f"--serializer-class {serializer_class} has nothing for this field" + (f": {why}" if why else ""))
    return space


def valid_keepbits_for_bitround(da):
    if np.issubdtype(da.dtype, np.float64):
        return _BITROUND_KEEPBITS_F64
    if np.issubdtype(da.dtype, np.float32):
        return _BITROUND_KEEPBITS_F32
    raise TypeError(f"Unsupported dtype '{da.dtype}'. BitRound only supports float32 and float64.")


def valid_digits_for_quantize(da):
    if np.issubdtype(da.dtype, np.float64):
        return _QUANTIZE_DIGITS_F64
    if np.issubdtype(da.dtype, np.float32):
        return _QUANTIZE_DIGITS_F32
    raise TypeError(f"Unsupported dtype '{da.dtype}'. Quantize only supports float32 and float64.")


def finite_range(a: np.ndarray):
    """(min, max) over the finite values of `a`, or None when it has none; copies `a` only when it
    holds +-inf."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all NaN
        lo, hi = np.nanmin(a), np.nanmax(a)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        a = a[np.isfinite(a)]
        if not a.size:
            return None
        lo, hi = a.min(), a.max()
    return float(lo), float(hi)


@dataclass
class FieldScan:
    """What one pass over the whole field found (scan_field)."""
    range: Optional[tuple]  # finite (min, max), min == max for a constant field; None: no finite value or unread
    readable: bool          # False when a block could not be read, twice
    nonfinite: int          # NaN and Inf cells
    varying: dict           # dim -> bool per index: the single time dim's and single vertical dim's indices
                            # holding more than one finite value


def scan_field(da, comm=None) -> FieldScan:
    """Collective over `comm` when given: one pass over the whole field, its blocks split across the
    ranks (one block in memory at a time, a failed read retried once), for a FieldScan.  All ranks get
    the same answer."""
    data = da.data if isinstance(da.data, dask.array.Array) else dask.array.from_array(np.asarray(da.data))
    rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    time_dims, vertical_dims, _ = _classify_sample_dims(da)
    per_index = {name: (pos, np.full(da.sizes[name], np.inf), np.full(da.sizes[name], -np.inf))
                 for group in (time_dims, vertical_dims) if len(group) == 1 for pos, name in group}
    offsets = [np.concatenate(([0], np.cumsum(c)[:-1])) for c in data.chunks]
    dmin, dmax, failed, nonfinite = np.inf, -np.inf, 0, 0
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        for idx in list(np.ndindex(*data.numblocks))[rank::size]:
            for attempt in (1, 2):
                try:
                    b = np.asarray(data.blocks[idx].compute())
                    break
                except Exception as e:
                    b = None
                    click.echo(f"[range] WARNING rank {rank}: reading block {idx} failed ({e!r})"
                               + ("; retrying" if attempt == 1 else ""), err=True)
            if b is None:
                failed = 1
                break
            r = finite_range(b)
            if r is not None:
                dmin, dmax = min(dmin, r[0]), max(dmax, r[1])
            if b.dtype.kind == "f":
                nonfinite += b.size - int(np.count_nonzero(np.isfinite(b)))
                if per_index and np.isinf(b).any():
                    b = np.where(np.isinf(b), np.nan, b)  # the per-index extremes are over finite values
            for pos, lo, hi in per_index.values():
                other = tuple(a for a in range(b.ndim) if a != pos)
                blo, bhi = np.fmin.reduce(b, axis=other), np.fmax.reduce(b, axis=other)  # NaN ignored
                sl = slice(offsets[pos][idx[pos]], offsets[pos][idx[pos]] + b.shape[pos])
                lo[sl] = np.fmin(lo[sl], blo)
                hi[sl] = np.fmax(hi[sl], bhi)
    if comm is not None:
        failed = comm.allreduce(failed, op=MPI.MAX)
        dmin, dmax = comm.allreduce(dmin, op=MPI.MIN), comm.allreduce(dmax, op=MPI.MAX)
        nonfinite = comm.allreduce(nonfinite, op=MPI.SUM)
        for _, lo, hi in per_index.values():
            comm.Allreduce(MPI.IN_PLACE, lo, op=MPI.MIN)
            comm.Allreduce(MPI.IN_PLACE, hi, op=MPI.MAX)
    finite = not failed and np.isfinite(dmin) and np.isfinite(dmax)
    return FieldScan(range=(float(dmin), float(dmax)) if finite else None, readable=not failed,
                     nonfinite=int(nonfinite), varying={n: hi > lo for n, (_, lo, hi) in per_index.items()})


def fixed_scale_offset_configs(da, data_range=None):
    """FixedScaleOffset kwargs mapping `data_range`, the full field's [min, max], onto each uint in
    _FSO_TARGET_UINTS narrower than the float.  Empty without a range: FixedScaleOffset does not
    clip, so parameters from a sample would silently corrupt values outside the sample's range."""
    dtype = da.dtype
    if not np.issubdtype(dtype, np.floating) or data_range is None:
        return []
    dmin, dmax = float(data_range[0]), float(data_range[1])
    span = dmax - dmin
    if not (np.isfinite(dmin) and np.isfinite(dmax)) or span <= 0.0:
        return []
    configs = []
    for uw in _FSO_TARGET_UINTS:
        if np.dtype(uw).itemsize >= dtype.itemsize:
            continue
        bits = np.dtype(uw).itemsize * 8
        configs.append(dict(offset=dmin, scale=(2.0 ** bits - 1.0) / span, dtype=str(dtype), astype=uw))
    return configs


def combo_is_valid(filt, serializer, compressor=None, dtype=None) -> bool:
    """Reject pairings that crash, corrupt or gain nothing: FixedScaleOffset's unsigned ints into ZFPY
    (every mode), or 8-bit ones into PCodec; BitRound below the mantissa width into ZFPY, which codes
    its integer bit view lossily so decoding corrupts exponents (checked when `dtype`, the field's, is
    given); EBCC with a compressor (gains nothing) or a filter other than AsType (breaks its error
    bound; validate_pipeline checks that the cast is to float32)."""
    if isinstance(serializer, EBCC):
        return compressor is None and (filt is None or isinstance(filt, zarrcodecs_nc.AsType))
    if (isinstance(filt, zarrcodecs_nc.BitRound) and isinstance(serializer, zarrcodecs_nc.ZFPY)
            and dtype is not None and np.issubdtype(dtype, np.floating)):
        keepbits = int((getattr(filt, "codec_config", None) or {}).get("keepbits", 0))
        if keepbits < np.finfo(dtype).nmant:
            return False
    if isinstance(filt, zarrcodecs_nc.FixedScaleOffset):
        if isinstance(serializer, zarrcodecs_nc.ZFPY):
            return False
        if isinstance(serializer, zarrcodecs_nc.PCodec):
            cfg = getattr(filt, "codec_config", None)
            astype = cfg.get("astype") if isinstance(cfg, dict) else None
            if astype is not None and np.dtype(astype).itemsize == 1:
                return False
    return True


def codec_pipeline_kwargs(compressor, filt, serializer) -> dict:
    """zarr.create_array kwargs for a (compressor, filter, serializer) triple; None means none: no
    filter, no compressor (not zarr's default Zstd), the plain bytes serializer."""
    return {"filters": [filt] if filt is not None else None,
            "compressors": [compressor] if compressor is not None else None,
            "serializer": "auto" if serializer is None else serializer}


# Checked before zarr's registry, which returns stock ZFPY for ZFPYRank's "numcodecs.zfpy".
_CODEC_CLASSES = {"numcodecs.zfpy": ZFPYRank, "numcodecs.zfpy_flat": ZFPYFlat,
                  "numcodecs.ebcc_filter": EBCC}
# Resolvable only through dc_toolkit's zarr.codecs entry point (README, "Reading a store without dc_toolkit").
ENTRY_POINT_CODECS = frozenset(("numcodecs.zfpy_flat", "numcodecs.ebcc_filter"))


def pipeline_is_stock(pipeline: dict) -> bool:
    """True when every codec of a pipeline dict decodes in a bare zarr client."""
    codecs = (pipeline.get(k) for k in ("compressor", "filter", "serializer"))
    return all(not isinstance(c, dict) or c.get("name") not in ENTRY_POINT_CODECS for c in codecs)


def codec_from_dict(d):
    if d is None:
        return None
    cls = _CODEC_CLASSES.get(d["name"]) or get_codec_class(d["name"])
    return cls.from_dict(d)


def pipeline_to_dict(compressor, filt, serializer) -> dict:
    """The identity of a combination: the zarr JSON form of its codecs."""
    return {"compressor": None if compressor is None else compressor.to_dict(),
            "filter": None if filt is None else filt.to_dict(),
            "serializer": None if serializer is None else serializer.to_dict()}


def pipeline_from_dict(d: dict):
    """(compressor, filt, serializer) codec objects from pipeline_to_dict's output."""
    missing = [k for k in ("compressor", "filter", "serializer") if k not in d]
    if missing:
        raise ValueError(f"pipeline dict lacks {missing}; expected keys compressor, filter, serializer")
    return codec_from_dict(d["compressor"]), codec_from_dict(d["filter"]), codec_from_dict(d["serializer"])


def pipeline_json(compressor, filt, serializer) -> str:
    """Canonical one-line JSON of a pipeline (the key used by --resume)."""
    return json.dumps(pipeline_to_dict(compressor, filt, serializer), sort_keys=True, separators=(",", ":"))


def codec_label(codec) -> str:
    """Label like 'zstd(level=6)', keys sorted so it does not depend on the codec's origin; '-' for None."""
    if codec is None:
        return "-"
    if isinstance(codec, EBCC):
        return repr(codec)
    d = codec.to_dict()
    name = d["name"].removeprefix("numcodecs.")
    return f"{name}({', '.join(f'{k}={v}' for k, v in sorted(d['configuration'].items()))})"


def pipeline_name(compressor, filt, serializer) -> str:
    return " | ".join(codec_label(c) for c in (compressor, filt, serializer))


# =============================================================================
# 5. IN-MEMORY EVALUATION
# =============================================================================

# What the metrics measure; the resume state records it beside a digest of the code that measures them.
METRIC_DEFINITIONS = ("N_Corrupt: cells whose finiteness the round trip changes",
                      "N_Bounds: cells the round trip moves from inside the physical bounds (with the slack) to outside",
                      "Q99_Rel: cells with |x| >= the q99 cut, taken over the non-zero values when the plain one is 0",
                      "Grad_Rel: finite differences along the horizontal dims, else the non-leading ones")


def _info_bytes(info) -> Tuple[int, int]:
    """(count_bytes, count_bytes_stored) from a zarr ArrayInfo."""
    count = getattr(info, "count_bytes", None)
    stored = getattr(info, "count_bytes_stored", None)
    return int(info._count_bytes if count is None else count), \
        int(info._count_bytes_stored if stored is None else stored)


def _iter_chunk_slices(shape, chunk_shape):
    ranges = [range(0, s, c) for s, c in zip(shape, chunk_shape)]
    for start in product(*ranges):
        yield tuple(slice(st, min(st + c, s)) for st, c, s in zip(start, chunk_shape, shape))


def within_limit(value, limit) -> bool:
    """Gate predicate: a None value or a None/+inf limit passes."""
    if value is None or limit is None or not math.isfinite(limit):
        return True
    return float(value) <= float(limit)


# (metric, threshold) keys of the cheap gates, shared by the gradient precheck and utils_cli.evaluate_gates.
CHEAP_GATES = (("Relative_Error_L1", "l1"), ("Relative_Error_L2", "l2"),
               ("Relative_Error_Linf", "linf"), ("Bias_Rel", "bias"))


def _rel(err, ori) -> float:
    """Relative error; 0/0 is 0 (a zero field reproduced exactly), x/0 is inf."""
    if ori == 0:
        return 0.0 if err == 0 else float("inf")
    return float(err) / float(ori)


# FixedScaleOffset casts NaN fill to int and numpy warns on every chunk; silenced, since N_Corrupt
# counts those cells.  A real FSO overflow is silent: the full-field range and compress's range check
# (utils_cli.fso_range_problem) guard against it.
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)


def _zarr_roundtrip(sample_np, dims, codec_kwargs, chunks):
    """One MemoryStore round trip; returns (decoded, compression ratio)."""
    with Timer("eval.create_array"):
        z = zarr.create_array(store=zarr.storage.MemoryStore(), name="_tmp_eval",
                              shape=sample_np.shape, dtype=sample_np.dtype, chunks=chunks,
                              zarr_format=3, dimension_names=tuple(dims), **codec_kwargs)
    with Timer("eval.encode"):
        z[...] = sample_np
    with Timer("eval.info_complete"):
        count_bytes, count_bytes_stored = _info_bytes(z.info_complete())
    with Timer("eval.decode"):
        decoded = z[...]
    _zarr_sync(z.store.clear())  # free the encoded bytes here: the array object can outlive this call in a GC cycle
    return decoded, count_bytes / count_bytes_stored


_ACC_INIT = {"l1_err": 0.0, "l2_err_sq": 0.0, "linf_err": 0.0, "signed_err": 0.0, "l1_ori": 0.0,
             "l2_ori_sq": 0.0, "linf_ori": 0.0, "q99_err": 0.0, "q99_ori": 0.0, "n_valid": 0, "n_corrupt": 0,
             "n_bounds": 0, "decoded_min": math.inf, "decoded_max": -math.inf,
             "source_min": math.inf, "source_max": -math.inf}
_ACC_MAX = ("linf_err", "linf_ori", "decoded_max", "source_max")
_ACC_MIN = ("decoded_min", "source_min")


def _error_sums(orig_all, decoded_all, chunks, q99_abs=None, bounds=None) -> dict:
    """Accumulators of the error metrics, chunk by chunk, over the cells finite in both arrays; n_corrupt
    counts the cells finite in only one, n_bounds those inside `bounds` (low, high) in the original and
    outside in the decoded array, source_min/max span the finite original.  `q99_abs` None skips the tail."""
    acc = dict(_ACC_INIT)
    with np.errstate(invalid="ignore"):
        for sl in _iter_chunk_slices(orig_all.shape, chunks):
            orig, dec = orig_all[sl], decoded_all[sl]
            finite_orig, finite_dec = np.isfinite(orig), np.isfinite(dec)
            acc["n_corrupt"] += int(np.count_nonzero(finite_orig != finite_dec))  # data lost, or fill made data
            if orig.dtype.kind != "f":
                acc["source_min"], acc["source_max"] = min(acc["source_min"], orig.min()), max(acc["source_max"], orig.max())
            elif finite_orig.any():
                acc["source_min"] = min(acc["source_min"], float(np.min(orig, where=finite_orig, initial=np.inf)))
                acc["source_max"] = max(acc["source_max"], float(np.max(orig, where=finite_orig, initial=-np.inf)))
            valid = finite_orig & finite_dec
            nv = int(np.count_nonzero(valid))
            if nv == 0:
                continue
            acc["n_valid"] += nv
            # Two float64 temporaries per chunk (the boolean index copied already), then all in place.
            o_abs = orig[valid].astype(np.float64, copy=False)
            e_abs = dec[valid].astype(np.float64, copy=False)
            acc["decoded_min"] = min(acc["decoded_min"], float(e_abs.min()))
            acc["decoded_max"] = max(acc["decoded_max"], float(e_abs.max()))
            if bounds is not None:
                lo, hi = bounds
                acc["n_bounds"] += int(np.count_nonzero((o_abs >= lo) & (o_abs <= hi) & ((e_abs < lo) | (e_abs > hi))))
            e_abs -= o_abs
            acc["signed_err"] += float(e_abs.sum()); acc["l2_err_sq"] += float(np.dot(e_abs, e_abs))
            np.abs(e_abs, out=e_abs); np.abs(o_abs, out=o_abs)
            acc["l1_err"] += float(e_abs.sum()); acc["linf_err"] = max(acc["linf_err"], float(e_abs.max(initial=0.0)))
            acc["l1_ori"] += float(o_abs.sum()); acc["l2_ori_sq"] += float(np.dot(o_abs, o_abs))
            acc["linf_ori"] = max(acc["linf_ori"], float(o_abs.max(initial=0.0)))
            if q99_abs is not None:
                ext = o_abs >= q99_abs
                if ext.any():
                    acc["q99_err"] += float(e_abs[ext].sum()); acc["q99_ori"] += float(o_abs[ext].sum())
    return acc


def _merge_sums(parts) -> dict:
    """One accumulator dict from several _error_sums results."""
    acc = dict(_ACC_INIT)
    for p in parts:
        for k, v in p.items():
            acc[k] = max(acc[k], v) if k in _ACC_MAX else min(acc[k], v) if k in _ACC_MIN else acc[k] + v
    return acc


def _errors_from_sums(acc: dict, want_q99: bool):
    """(errors dict, L2 error) from _error_sums accumulators; raises CombinationProducedNonFiniteError when a
    norm overflowed."""
    if not all(map(math.isfinite, (acc["l1_err"], acc["l2_err_sq"], acc["linf_err"]))):
        raise CombinationProducedNonFiniteError(
            f"non-finite error accumulators after masking (l1_err={acc['l1_err']}, "
            f"l2_err_sq={acc['l2_err_sq']}, linf_err={acc['linf_err']})")
    l2_err, l2_ori, n_valid = math.sqrt(acc["l2_err_sq"]), math.sqrt(acc["l2_ori_sq"]), acc["n_valid"]
    finite_src = math.isfinite(acc["source_min"])
    errors = {
        "Relative_Error_L1": _rel(acc["l1_err"], acc["l1_ori"]),
        "Relative_Error_L2": _rel(l2_err, l2_ori),
        "Relative_Error_Linf": _rel(acc["linf_err"], acc["linf_ori"]),
        "Bias_Rel": _rel(abs(acc["signed_err"]), acc["l1_ori"]),
        "Decoded_Min": float(acc["decoded_min"]) if n_valid else float("nan"),
        "Decoded_Max": float(acc["decoded_max"]) if n_valid else float("nan"),
        "Source_Min": float(acc["source_min"]) if finite_src else float("nan"),
        "Source_Max": float(acc["source_max"]) if finite_src else float("nan"),
        "N_Corrupt": int(acc["n_corrupt"]),
        "N_Bounds": int(acc["n_bounds"]),
        "N_Valid": int(n_valid),
        "Q99_Rel": _rel(acc["q99_err"], acc["q99_ori"]) if want_q99 else None,
    }
    return errors, l2_err


def evaluate_codec_pipeline(sample_np: np.ndarray, dims, codec_kwargs: dict, chunks,
                            q99_abs: float | None = None, bounds=None, compute_gradient: bool = False,
                            gradient_axes=None, precheck_thresholds: dict | None = None):
    """Round-trip `sample_np` through a pipeline in memory; returns (compression_ratio, errors_dict,
    euclidean_distance).  Fill (non-finite in the original) is left out of every norm; a cell whose
    finiteness the round trip changes (data turned NaN/Inf, or fill turned into data) is corruption,
    counted in N_Corrupt; with `bounds` (low, high), N_Bounds counts the cells moved outside them.  With
    `precheck_thresholds`, combos failing a cheap gate skip the gradient metric, one more pass over the sample."""
    decoded, ratio = _zarr_roundtrip(sample_np, dims, codec_kwargs, chunks)
    want_q99 = q99_abs is not None and math.isfinite(q99_abs)
    with Timer("eval.metrics"):
        errors, l2_err = _errors_from_sums(
            _error_sums(sample_np, decoded, chunks, q99_abs if want_q99 else None, bounds), want_q99)
    do_grad = compute_gradient
    if compute_gradient and precheck_thresholds is not None:
        do_grad = all(within_limit(errors[m], precheck_thresholds.get(t)) for m, t in CHEAP_GATES)
    errors["Grad_Rel"] = _gradient_rel_l1(sample_np, decoded, axes=gradient_axes) if do_grad else None
    return ratio, errors, l2_err


def _gradient_pieces(shape, axis: int, max_elems: int):
    """Slices covering an array of `shape` in pieces of at most about `max_elems` elements, consecutive
    pieces along `axis` sharing one index: every neighbouring pair along `axis` lies in exactly one piece."""
    n, rest = shape[axis], [a for a in range(len(shape)) if a != axis]
    rest_elems = int(np.prod([shape[a] for a in rest])) if rest else 1
    span = max(1, max_elems // rest_elems - 1)  # differences per piece along `axis`
    block = {a: shape[a] for a in rest}
    if 2 * rest_elems > max_elems:  # even two indices along `axis` are too many: split the other axes too
        inner, budget = 1, max(1, max_elems // 2)
        for a in reversed(rest):
            if inner * shape[a] > budget:
                block[a] = max(1, budget // inner)
                block.update({b: 1 for b in rest[:rest.index(a)]})
                break
            inner *= shape[a]
    ranges = [range(0, shape[a], block[a]) for a in rest]
    for a0 in range(0, n - 1, span):
        for starts in product(*ranges):
            sl = [slice(None)] * len(shape)
            sl[axis] = slice(a0, min(n, a0 + span + 1))
            for a, st in zip(rest, starts):
                sl[a] = slice(st, min(shape[a], st + block[a]))
            yield tuple(sl)


def _gradient_rel_l1(orig: np.ndarray, decoded: np.ndarray, axes=None, max_elems: int = 4 << 20) -> float:
    """Sum|d(decoded) - d(orig)| / Sum|d(orig)| over finite differences along `axes` (default: all but the
    leading one; axis 0 for 1-D), in pieces of about 32 MiB of float64 (_gradient_pieces)."""
    if axes is None:
        axes = tuple(range(1, orig.ndim)) if orig.ndim > 1 else (0,)
    err_sum = ori_sum = 0.0
    with np.errstate(invalid="ignore"):
        for ax in sorted({a % orig.ndim for a in axes}):
            for sl in _gradient_pieces(orig.shape, ax, max_elems):
                do = np.diff(orig[sl].astype(np.float64), axis=ax)
                dd = np.diff(decoded[sl].astype(np.float64), axis=ax)
                m = np.isfinite(do) & np.isfinite(dd)
                dd -= do
                np.abs(dd, out=dd)
                np.abs(do, out=do)
                err_sum += float(np.sum(dd, where=m))
                ori_sum += float(np.sum(do, where=m))
    if ori_sum == 0:
        return 0.0 if err_sum == 0 else float("inf")
    return err_sum / ori_sum


# =============================================================================
# 6. PERSISTENCE
# =============================================================================

def persist_with_codec_pipeline(da, store, component: str, codec_kwargs: dict, inner_chunks, shards,
                                verify: bool = True, q99_abs=None, bounds=None):
    """Write dask-backed `da`, whose blocks hold whole write units (shards, else inner chunks), to `store`
    at `component`, one task per block: with `verify` the task re-reads what it wrote and adds up the error
    sums of its block, so the field is read once.  Returns (compression_ratio, errors, euclidean_distance),
    the last two None without `verify`."""
    data = da.data
    assert isinstance(data, dask.array.Array), "expects a dask-backed xr.DataArray"
    unit = shards if shards is not None else inner_chunks
    assert all(x % u == 0 for c, u in zip(data.chunks, unit) for x in c[:-1]), "blocks must hold whole write units"
    z = zarr.create_array(store=store, name=component, shape=data.shape, dtype=data.dtype, chunks=inner_chunks,
                          shards=shards, zarr_format=3, dimension_names=tuple(da.dims), overwrite=True,
                          **codec_kwargs)
    want_q99 = verify and q99_abs is not None and math.isfinite(q99_abs)

    def write(block, region):
        z[region] = block
        return _error_sums(np.asarray(block), z[region], inner_chunks, q99_abs if want_q99 else None,
                           bounds) if verify else None

    starts = [np.concatenate(([0], np.cumsum(c)[:-1])) for c in data.chunks]
    tasks = [dask.delayed(write)(block, tuple(slice(int(s[i]), int(s[i] + c[i]))
                                              for s, c, i in zip(starts, data.chunks, idx)))
             for idx, block in zip(np.ndindex(*data.numblocks), data.to_delayed().ravel())]
    with Timer("persist.write"):
        parts = dask.compute(*tasks)
    count_bytes, count_bytes_stored = _info_bytes(z.info_complete())
    if not verify:
        return count_bytes / count_bytes_stored, None, None
    errors, l2_err = _errors_from_sums(_merge_sums(parts), want_q99)
    return count_bytes / count_bytes_stored, errors, l2_err


# =============================================================================
# 7. MPI & TOPOLOGY
# =============================================================================

def detect_node_topology(comm=None):
    """Collective over `comm` (default COMM_WORLD): (node_comm, ranks_on_node, local_rank) of this rank's
    node."""
    comm = comm or MPI.COMM_WORLD
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.Get_rank())
    return node_comm, node_comm.Get_size(), node_comm.Get_rank()


def detect_cores_available() -> int:
    """Logical CPUs in this process's affinity mask (Slurm and cgroup cpusets, not a CPU quota such as
    docker --cpus), else os.cpu_count()."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def detect_physical_cores() -> int:
    """Rank count for the launch hint and the UI: the affinity mask capped by the physical cores
    (Open MPI's default slot count on a whole machine; SMT siblings in a cpuset are not halved)."""
    avail = detect_cores_available()
    return max(1, min(avail, psutil.cpu_count(logical=False) or avail))


def check_thread_oversubscription(abort_if_unsafe: bool = True, rank: int = 0) -> None:
    """Warn on rank 0 when a THREAD_ENV_VARS entry is not 1, then end the job (abort) unless
    `abort_if_unsafe` is off (not a collective: the first rank to call it ends the job)."""
    problems = []
    for v in THREAD_ENV_VARS:
        val = os.environ.get(v)
        try:
            if val is None or int(val) != 1:
                problems.append(f"{v}={'<unset>' if val is None else val}")
        except ValueError:
            problems.append(f"{v}={val}")
    if problems:
        if rank == 0:
            click.echo("[oversubscription-check] WARNING: codec-internal thread variables not pinned to 1:")
            for p in problems:
                click.echo(f"  - {p}")
            click.echo("  Suggested: export " + " ".join(f"{v}=1" for v in THREAD_ENV_VARS))
            if abort_if_unsafe:
                click.echo("  Aborting (use --no-oversubscription-check to override).")
        if abort_if_unsafe:
            abort(1)


# =============================================================================
# 8. PROGRESS & TIMING
# =============================================================================

def progress_bar(done: int, total: int, label: str, bar_width: int = 40) -> None:
    pct = done / max(1, total)
    bar = "*" * int(bar_width * pct) + "-" * (bar_width - int(bar_width * pct))
    click.echo(f"[{label}] Progress: |{bar}| {pct*100:6.2f}% ({done}/{total})")


_TIMINGS = defaultdict(list)


class Timer:
    """`with Timer("label"):` accumulates wall time per label."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        _TIMINGS[self.label].append(time.perf_counter() - self.start)


@atexit.register
def print_profile_summary():
    if not _TIMINGS or MPI.COMM_WORLD.Get_rank() != 0:
        return
    print("\n=== Timing Summary (rank 0) ===")
    print("Sum of Total = seconds inside the timed sections (excludes the sample build,")
    print("dask graph setup and result-write overhead).\n")
    width = max(len(label) for label in _TIMINGS)
    totals = {label: sum(d) for label, d in _TIMINGS.items()}
    grand = sum(totals.values()) or 1.0
    header = f"{'Label':<{width}} | {'Calls':>5} | {'Avg (s)':>10} | {'Total (s)':>12} | {'% total':>7}"
    print(header); print("-" * len(header))
    for label, durations in sorted(_TIMINGS.items()):
        total, count = totals[label], len(durations)
        print(f"{label:<{width}} | {count:>5} | {total / count:>10.6f} | "
              f"{total:>12.6f} | {100.0 * total / grand:>6.2f}%")
    print("=" * len(header))
