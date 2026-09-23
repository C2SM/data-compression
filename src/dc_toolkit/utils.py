"""
Library behind the ``dc_toolkit`` CLI.

Sections
  1. Sizes & dataset I/O
  2. Representative sampling      (which slices of a big field to score)
  3. Chunk & shard sizing         (zarr geometry, shared by eval and persist)
  4. Codec spaces & pipelines     (compressor x filter x serializer grids; EBCC optional;
                                   a pipeline's JSON form is its identity in results and manifests)
  5. In-memory evaluation         (encode -> decode -> error metrics)
  6. Persistence                  (dask -> zarr LocalStore, optional verify)
  7. MPI & topology
  8. Progress & timing
"""
import asyncio
import atexit
import importlib
import json
import math
import os
import re
import struct
import sys
import time
import warnings
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Optional, Tuple

import click
import dask
import dask.array
import humanize
import numpy as np
import xarray as xr
import zarr
import numcodecs
import numcodecs.zfpy
import zfpy
from mpi4py import MPI
from zarr.core.sync import sync as _zarr_sync
from zarr.codecs import numcodecs as zarrcodecs_nc
from zarr.codecs.numcodecs._codecs import _NumcodecsArrayBytesCodec
from zarr.registry import get_codec_class, register_codec

# EBCC (Error Bounded Climate Compressor) is optional: pip install -e ".[ebcc]".
os.environ.setdefault("EBCC_LOG_LEVEL", "4")  # the C library logs to stderr; 4 = errors only
try:
    importlib.import_module("ebcc.zarr_filter")  # registers "ebcc_filter" with numcodecs
    from ebcc.filter_wrapper import EBCC_Filter
    EBCC_AVAILABLE = True
except ImportError:
    EBCC_AVAILABLE = False


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


# Codec-internal thread pools that must be pinned to 1: every rank owns one
# core (see check_thread_oversubscription).
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLOSC_NTHREADS", "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "OMP_THREAD_LIMIT",
)
# Environment variables the EBCC C library reads at encode time; they change
# the bytes it produces, so the sweep manifest records them.
EBCC_ENV_VARS = (
    "EBCC_INIT_BASE_ERROR_QUANTILE", "EBCC_ERROR_BOUND_SLACK", "EBCC_DISABLE_MEAN_ADJUSTMENT",
    "EBCC_DISABLE_PURE_BASE_COMPRESSION_FALLBACK",
    "EBCC_DISABLE_PURE_BASE_COMPRESSION_FALLBACK_CONSISTENCY", "EBCC_ERROR_BOUND_STRICT_MODE",
)


def abort(code: int = 1) -> None:
    """Exit the whole job: MPI Abort under multi-rank (sys.exit on one rank
    would leave the others hanging at the next collective), sys.exit otherwise."""
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
    """Open a zarr v3 LocalStore; returns (group, store).  Keep both alive.
    Reads the arrays' own metadata, so a store whose consolidated metadata is
    stale (arrays added since) is listed correctly."""
    store = zarr.storage.LocalStore(path, read_only=read_only)
    return zarr.open_group(store, mode="r" if read_only else "a", use_consolidated=False), store


def open_dataset(dataset_file: str, field_to_compress: Optional[str] = None, rank: int = 0):
    """Lazily open a .nc / .grib / .zarr dataset with a dask backend."""
    suffix = Path(dataset_file).suffix.lower()
    if suffix == ".nc":
        ds = xr.open_dataset(dataset_file, chunks="auto")
    elif suffix == ".grib":
        ds = xr.open_dataset(dataset_file, chunks="auto", engine="cfgrib",
                             backend_kwargs={"indexpath": ""})
    elif suffix == ".zarr":
        ds = xr.open_zarr(dataset_file, chunks="auto", consolidated=None)
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
# A field larger than the sample budget is thinned along its time-like and
# vertical-like dims only; horizontal dims are kept whole so codecs see the
# real spatial structure.  Dim classification uses CF metadata first and a
# name heuristic as fallback.

_TIME_LIKE_DIM_RE = re.compile(
    r"^(?:time|.*_time|t|step|forecast_reference_time|forecast_period|"
    r"ensemble|realization|member|reftime|valid_time|epoch)$", re.IGNORECASE)
_CF_TIME_UNITS_RE = re.compile(r"^\s*\w+\s+since\s+", re.IGNORECASE)
_CF_TIME_STANDARD_NAMES = frozenset({"time", "forecast_reference_time", "forecast_period"})
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
    """attrs + encoding of the coord backing `dim` (xarray moves decoded CF
    attrs into .encoding, so both must be checked); {} if no coord."""
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
    """Return (time_dims, vertical_dims, spatial_dims) as lists of (pos, name)."""
    time_dims, vertical_dims, spatial_dims = [], [], []
    for i, name in enumerate(da.dims):
        if _is_time_like_coord(da, name):
            time_dims.append((i, name))
        elif _is_vertical_like_coord(da, name):
            vertical_dims.append((i, name))
        else:
            spatial_dims.append((i, name))
    return time_dims, vertical_dims, spatial_dims


def build_representative_sample(da: xr.DataArray, size_limit_bytes: int, rank: int = 0,
                                policy: str = "cascade",
                                vertical_floor: int | None = None) -> xr.DataArray:
    """
    Subset of `da` with nbytes <= size_limit_bytes, deterministic and
    reproducible (evenly spaced indices along time/vertical dims).  A field
    without time/vertical dims is returned whole, with a warning, even above
    the budget.

    policy="cascade": spend the budget on time steps first, keeping a
    budget-aware minimum of vertical levels.  policy="balanced": equal
    log-space split across all stride dims.  Raises SampleTooLargeError when
    a single horizontal slab does not fit.
    """
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
    plan = _allocate_stride_plan(da, time_dims, vertical_dims, max_product,
                                 policy=policy, vertical_floor=vertical_floor)
    isel = {}
    for _, name in stride_dims:
        size, n_keep = int(da.sizes[name]), plan[name]
        if n_keep < size:
            isel[name] = np.unique(np.linspace(0, size - 1, num=n_keep, dtype=int)).tolist()
    sampled = da.isel(isel) if isel else da

    if rank == 0:
        strided = ", ".join(f"{n}={plan[n]}/{da.sizes[n]}" for _, n in stride_dims)
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
    if not dims_info:
        return {}
    if len(dims_info) == 1:
        _, name, size = dims_info[0]
        return {name: max(1, min(size, int(group_keep)))}
    return _balanced_group_plan(dims_info, group_keep)


def _allocate_stride_plan(da, time_dims, vertical_dims, max_product,
                          policy="cascade", vertical_floor=None) -> dict:
    """{dim_name: n_keep} for every time and vertical dim."""
    time_info = [(i, n, int(da.sizes[n])) for i, n in time_dims]
    vert_info = [(i, n, int(da.sizes[n])) for i, n in vertical_dims]
    if policy == "balanced":
        return _balanced_group_plan(time_info + vert_info, max_product)

    T = int(np.prod([s for _, _, s in time_info])) if time_info else 1
    V = int(np.prod([s for _, _, s in vert_info])) if vert_info else 1
    P = max(1.0, float(max_product))
    if vertical_floor is not None:
        vfloor = max(1, int(vertical_floor))
    else:
        vfloor = max(4, int(math.ceil(math.log2(V)))) if V > 1 else 1
    vfloor = min(vfloor, V)
    # Cap the level floor at sqrt(P) so a tight budget degrades to the balanced split.
    base_level = max(1, min(vfloor, int(math.floor(math.sqrt(P)))))
    time_keep = max(1, min(T, int(math.floor(P / base_level))))
    level_keep = max(1, min(V, int(math.floor(P / time_keep))))
    return {**_distribute_group(time_info, time_keep), **_distribute_group(vert_info, level_keep)}


# =============================================================================
# 3. CHUNK & SHARD SIZING
# =============================================================================

def _shrink_order(shape, dims) -> list:
    """Axes (excluding 0) in shrink order: horizontal dims first, vertical last,
    fastest-varying first within each group (hiopy convention)."""
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
    """Pack leading slices up to target_bytes; if one slice is already too
    big, keep 1 leading index and (optionally) shrink spatial dims."""
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
    """(inner_chunk_shape, shard_shape).  shard_shape is None when a shard
    would hold fewer than two inner chunks: the chunk is at least half the
    shard target, or the leading axis has fewer than two chunks.
    `inner_chunks` forces the inner chunk shape (codecs with a fixed frame)."""
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
# The sweep builds three lists of codec objects from the grids below (some
# parameters depend on the field's dtype and value range).  A combination is
# identified everywhere by its pipeline dict, the zarr JSON form of its three
# codecs (see pipeline_to_dict), never by its position in these lists.

# Compressors (bytes -> bytes)
_BLOSC_CNAMES = ("lz4", "lz4hc", "zstd")
_BLOSC_CLEVELS = (1, 5, 9)
_BLOSC_SHUFFLES = (0, 1)
_LZ4_ACCELERATIONS = (1, 10, 100)
_ZSTD_LEVELS = (6, 12, 22)
_ZLIB_LEVELS = (3, 6, 9)
_BZ2_LEVELS = (3, 6, 9)
_LZMA_PRESETS = (3, 6, 9)
# Filters (array -> array); the top value of each grid is effectively lossless
_BITROUND_KEEPBITS_F32 = (3, 7, 11, 13, 17, 23)
_BITROUND_KEEPBITS_F64 = (3, 7, 11, 17, 23, 30, 37, 44, 52)
_QUANTIZE_DIGITS_F32 = (1, 3, 4, 5, 6, 7)
_QUANTIZE_DIGITS_F64 = (1, 3, 4, 5, 6, 7, 9, 11, 13, 15)
# FixedScaleOffset targets.  uint8 excluded: PCodec refuses 8-bit input and
# ZFPY rejects every FSO output (see combo_is_valid).
_FSO_TARGET_UINTS = ("uint16", "uint32")
# Serializers (array -> bytes)
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
    """Lossless bytes->bytes compressors (`with_lossy` is accepted for
    signature symmetry only).  Byte shuffling is only available through Blosc's
    own `shuffle` parameter (a standalone Shuffle codec breaks after a
    compressing serializer); Blosc gets the field's item size as `typesize`
    because zarr hands these codecs raw bytes, and shuffling single bytes is a
    no-op."""
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
    """Array->array filters.  Integer dtypes get Delta only.  `data_range`
    must be the FULL field's (min, max) when `da` is a sample: FixedScaleOffset
    does not clip, so out-of-range production values would corrupt silently."""
    classes = [zarrcodecs_nc.Delta]
    if with_lossy:
        classes += [zarrcodecs_nc.BitRound, zarrcodecs_nc.Quantize, zarrcodecs_nc.FixedScaleOffset]
        if np.issubdtype(da.dtype, np.floating) and da.dtype.itemsize > 4:
            classes.append(zarrcodecs_nc.AsType)  # f64 -> f32 down-cast
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
    return space


# ---- ZFPY: two encoders, because neither rank wins --------------------------
# zfp codes 4^d blocks, so the rank a chunk is encoded at decides which
# correlations it can exploit.  The chunk's own rank keeps cross-axis gradients
# inside a block and wins when the slower axes are correlated; 1-D gives them
# up and wins when those axes carry little signal, since zfp then spends no
# bits on noise and leaves the compressor after it more redundancy.  Neither
# dominates, so the sweep carries both and each field picks its winner.  They
# need distinct codec names because a pipeline's identity is its codecs' zarr
# JSON (the resume key, the parquet pipeline column, the manifest's winner).


# (Comments, not docstrings: zarr's wrapper replaces a codec class's docstring.)
# ZFPYRank encodes each chunk at its own rank, folded only as far as zfp requires
# (at most 4-D; per-axis header budget 2**24 at 2-D, 2**16 at 3-D, 2**12 at 4-D,
# which DYAMOND's cell axis overflows): size-1 axes dropped, then the slowest pair
# folded until the rank and every axis fit.  Its plain "zfpy" name decodes anywhere.
class ZFPYRank(zarrcodecs_nc.ZFPY, codec_name="zfpy"):
    _ZFP_MAX_PER_AXIS = {1: 2**48, 2: 2**24, 3: 2**16, 4: 2**12}

    @classmethod
    def encode_shape(cls, shape) -> tuple:
        dims = tuple(d for d in shape if d > 1) or (1,)
        max_rank = max(cls._ZFP_MAX_PER_AXIS)          # zfp stops at 4-D
        while len(dims) > 1 and (len(dims) > max_rank
                                 or any(d > cls._ZFP_MAX_PER_AXIS[len(dims)] for d in dims)):
            dims = (dims[0] * dims[1],) + dims[2:]     # C-order keeps the fold contiguous
        return dims

    async def _encode_single(self, chunk_data, chunk_spec):
        arr = np.ascontiguousarray(chunk_data.as_ndarray_like())
        out = await asyncio.to_thread(self._codec.encode, arr.reshape(self.encode_shape(arr.shape)))
        return chunk_spec.prototype.buffer.from_bytes(out)


class _ZFPYFlatCodec(numcodecs.zfpy.ZFPY):
    """zfpy under a second numcodecs id: zarr's numcodecs wrapper resolves
    codec_name against the numcodecs registry."""

    codec_id = "zfpy_flat"


numcodecs.register_codec(_ZFPYFlatCodec)


# ZFPYFlat encodes every chunk as 1-D.  Decoding is stock zfp (the stream carries
# its own shape), but a client without dc_toolkit cannot resolve the name.
class ZFPYFlat(ZFPYRank, codec_name="zfpy_flat"):
    @classmethod
    def encode_shape(cls, shape) -> tuple:
        return (max(1, int(np.prod(shape))),)


register_codec("numcodecs.zfpy_flat", ZFPYFlat)   # in-process reads must not depend on the install's entry points


# ---- EBCC (optional): JPEG 2000 base layer + error-bounded residual ----------
# Compresses float32 (lat, lon) frames; each chunk must be exactly one tile of
# the frame.  NaN/Inf or a tile that does not divide the frame make the C
# library EXIT THE PROCESS, so callers validate before encoding (ebcc_tile,
# ebcc_sweep_entries for the sweep, utils_cli.validate_pipeline for persist
# and plots).
# Maximum absolute error targets as fractions of the FULL field's value range
# (the paper's 0.1 %..10 % band plus one decade below); EBCC's own floor is
# range/65535 (uint16 base layer).
_EBCC_ERROR_FRACTIONS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
# Start of both EBCC rate-control searches (OpenJPEG rate = base_cr/2).  The
# bisection bracket depends on it, so it shifts the achieved ratio by ~10 %; it
# is part of the arglist, hence of the pipeline's identity.
_EBCC_BASE_CR = 2.0
# EBCC_MIN/MAX_INTERNAL_IMAGE_DIM in ebcc_codec.h: the C filter exits outside them.
_EBCC_TILE_MIN, _EBCC_TILE_MAX = 32, 2047


def _f32(bits) -> float:
    """The float32 EBCC packs into a uint32 arglist entry."""
    return struct.unpack("f", struct.pack("I", int(bits)))[0]


class EBCC(_NumcodecsArrayBytesCodec, codec_name="ebcc_filter"):
    """zarr v3 wrapper of ebcc.zarr_filter.EBCCZarrFilter; stored in zarr.json
    as "numcodecs.ebcc_filter" with EBCC's integer arglist
    [height, width, f32bits(base_cr), mode, f32bits(target)] where mode is
    0 none / 1 max_error_target / 2 relative_error_target."""

    def __init__(self, **codec_config):
        if not EBCC_AVAILABLE:
            raise ImportError("the EBCC serializer needs the ebcc package: pip install -e '.[ebcc]'")
        arglist = list(codec_config.get("arglist") or [])
        try:  # anything else makes the C library exit the process
            mode = int(arglist[3])
            ok = (mode in (0, 1, 2) and len(arglist) == (4 if mode == 0 else 5)
                  and all(_EBCC_TILE_MIN <= int(v) <= _EBCC_TILE_MAX for v in arglist[:2])
                  and all(math.isfinite(_f32(v)) and _f32(v) > 0 for v in arglist[2:3] + arglist[4:5]))
        except (IndexError, TypeError, ValueError, struct.error):
            ok = False
        if not ok:
            raise ValueError(f"EBCC arglist must be [height, width, base_cr bits, mode 0/1/2, target bits (modes 1, 2)] "
                             f"with the tile sides in [{_EBCC_TILE_MIN}, {_EBCC_TILE_MAX}] and positive base_cr and "
                             f"target; got {arglist}")
        super().__init__(**codec_config)

    @classmethod
    def from_params(cls, height: int, width: int, target: float, mode: str = "max_error_target",
                    base_cr: float = _EBCC_BASE_CR):
        opts = EBCC_Filter(base_cr=base_cr, height=height, width=width,
                           residual_opt=(mode, target)).hdf_filter_opts
        return cls(arglist=[int(v) for v in opts])

    @property
    def arglist(self) -> list:
        return [int(v) for v in self.codec_config["arglist"]]

    @property
    def height(self) -> int:
        return self.arglist[0]

    @property
    def width(self) -> int:
        return self.arglist[1]

    def __repr__(self) -> str:
        a = self.arglist
        mode = {0: "none", 1: "max_error_target", 2: "relative_error_target"}.get(a[3], a[3])
        target = f", {mode}={_f32(a[4]):g}" if len(a) > 4 else ""
        return f"EBCC(height={a[0]}, width={a[1]}, base_cr={_f32(a[2]):g}{target})"


register_codec("numcodecs.ebcc_filter", EBCC)


def _ebcc_tile_size(n: int):
    """Largest divisor of n within [32, 2047], or None."""
    if n < _EBCC_TILE_MIN:
        return None
    if n <= _EBCC_TILE_MAX:
        return n
    return next((d for d in range(_EBCC_TILE_MAX, _EBCC_TILE_MIN - 1, -1) if n % d == 0), None)


def ebcc_tile(da):
    """((height, width), "") of the EBCC tile for `da`, or (None, reason).
    The last two dims must be a horizontal (lat, lon) frame of a float field."""
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


def ebcc_sweep_entries(filters, serializers, sample_np):
    """Standalone (None, filter, EBCC) sweep triples for every EBCC serializer;
    the filter is None for float32 and the AsType cast to float32 otherwise
    (EBCC's own requirement, so it is added even when --filter-class left
    AsType out of `filters`).  Returns (triples, reason) with triples empty
    when EBCC cannot run on this sample."""
    ebccs = [s for s in serializers if isinstance(s, EBCC)]
    if not ebccs:
        return [], ""
    if not np.isfinite(sample_np).all():
        return [], "the sample contains NaN/Inf, which EBCC cannot encode"
    filt = None
    if sample_np.dtype != np.float32:
        astype = [f for f in filters if isinstance(f, zarrcodecs_nc.AsType)]
        filt = astype[0] if astype else zarrcodecs_nc.AsType(encode_dtype="float32",
                                                             decode_dtype=str(sample_np.dtype))
    return [(None, filt, s) for s in ebccs], ""


def serializer_space(da, with_lossy=True, serializer_class="all", with_ebcc=False, data_range=None,
                     chunk_shapes=None):
    """Array->bytes serializers: plain bytes (with 'all' and 'none': a lossy
    filter in front of a lossless byte compressor is the classic recipe, and
    8-bit fields have nothing else, since pco refuses them), PCodec, ZFPY when
    lossy is allowed
    (floats only: zfp's fixed-rate mode is never exact on integers, and Delta,
    their only filter, turns its error into a random walk on decode), EBCC when
    requested AND lossy is allowed, for float (lat, lon) frame stacks.  `data_range` (full-field
    min, max) scales the EBCC error targets; without it EBCC falls back to
    targets relative to each tile's own range.  `chunk_shapes`: the chunks ZFPY
    will meet (the sample's and the store's); ZFPYFlat is planned unless
    ZFPYRank already encodes every one of them as 1-D."""
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
        space = [None] + [s for s in space if s is not None and not isinstance(s, zarrcodecs_nc.PCodec)]
    if not space:  # a named class with nothing for this field: the caller skips or refuses it
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


def full_field_data_range(da, comm=None):
    """Finite (min, max) over the whole (dask-backed) field, or None if the
    field is constant / has no finite values.  One pass over the blocks; with
    `comm` the blocks are split across its ranks (a collective call)."""
    data = da.data if hasattr(da, "data") else np.asarray(da)
    try:
        if isinstance(data, dask.array.Array):
            rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
            mine = list(data.blocks.ravel())[rank::size]
            dmin, dmax = np.inf, -np.inf
            if mine:
                finite = [dask.array.isfinite(b) for b in mine]
                vals = dask.compute(*[dask.array.where(f, b, np.inf).min() for f, b in zip(finite, mine)],
                                    *[dask.array.where(f, b, -np.inf).max() for f, b in zip(finite, mine)])
                dmin, dmax = float(min(vals[:len(mine)])), float(max(vals[len(mine):]))
            if comm is not None:
                dmin, dmax = comm.allreduce(dmin, MPI.MIN), comm.allreduce(dmax, MPI.MAX)
        else:
            arr = np.asarray(data)
            fin = arr[np.isfinite(arr)]
            if fin.size == 0:
                return None
            dmin, dmax = float(fin.min()), float(fin.max())
    except Exception:
        return None
    if not (np.isfinite(dmin) and np.isfinite(dmax)) or dmax <= dmin:
        return None
    return (dmin, dmax)


def fixed_scale_offset_configs(da, data_range=None):
    """FixedScaleOffset kwargs mapping [min, max] onto each uint width in
    _FSO_TARGET_UINTS narrower than the float.  Falls back to `da`'s own range
    when data_range is None (only valid if `da` is the full field)."""
    dtype = da.dtype
    if not np.issubdtype(dtype, np.floating):
        return []
    if data_range is not None:
        dmin, dmax = float(data_range[0]), float(data_range[1])
    else:
        arr = np.asarray(da.values)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return []
        dmin, dmax = float(finite.min()), float(finite.max())
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
    """Reject pairings that crash or corrupt inside the codecs.  FixedScaleOffset
    emits unsigned ints, which ZFPY (every mode) and 8-bit PCodec refuse.
    BitRound below the mantissa width emits the integer bit view of the floats;
    ZFPY accepts it, compresses the bit pattern lossily, and the decode-side
    reinterpretation corrupts exponents (`dtype`, the field's, enables that
    check).  EBCC runs alone: a filter before it breaks its error bound
    (AsType's float32 down-cast excepted) and a compressor after it gains
    nothing."""
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
    """zarr.create_array kwargs for one (compressor, filter, serializer) triple.
    None means none: no filter, no compressor (zarr would otherwise default to
    Zstd), and the plain bytes serializer."""
    return {"filters": [filt] if filt is not None else None,
            "compressors": [compressor] if compressor is not None else None,
            "serializer": "auto" if serializer is None else serializer}


# Our classes, used when a pipeline is rebuilt from its dict: ZFPYRank must
# replace zarr's stock ZFPY (same codec name in zarr.json); ZFPYFlat and EBCC
# have no stock class.
_CODEC_CLASSES = {"numcodecs.zfpy": ZFPYRank, "numcodecs.zfpy_flat": ZFPYFlat,
                  "numcodecs.ebcc_filter": EBCC}
# Names a bare zarr client cannot resolve: they exist only through dc_toolkit's
# zarr.codecs entry point (README, "Reading a store without dc_toolkit").
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
    """Short human-readable form: 'blosc(clevel=1, cname=lz4, shuffle=0)' (keys
    sorted, so the label does not depend on where the codec came from); '-' for None."""
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


# (metric key, threshold key) of the cheap gates that the gradient precheck
# and utils_cli.evaluate_gates share.
CHEAP_GATES = (("Relative_Error_L1", "l1"), ("Relative_Error_L2", "l2"),
               ("Relative_Error_Linf", "linf"), ("Bias_Rel", "bias"))


def _rel(err, ori) -> float:
    """Relative error; 0/0 is 0 (a zero field reproduced exactly), x/0 is inf."""
    if ori == 0:
        return 0.0 if err == 0 else float("inf")
    return float(err) / float(ori)


# FixedScaleOffset casts NaN fill cells to int and numpy warns; fill cells are
# masked out of every norm, so the warning is noise.  (A real FSO overflow is
# silent and guarded by full_field_data_range + the verify gate instead.)
# Installed once, at import.
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)


def _zarr_roundtrip(sample_np, dims, codec_kwargs, chunks):
    """create + encode + info + decode on a MemoryStore.  Returns (decoded, ratio)."""
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
    _zarr_sync(z.store.clear())  # free the encoded bytes now; the array object can outlive this call in a GC cycle
    return decoded, count_bytes / count_bytes_stored


def _error_sums(sample_np, decoded, chunks, q99_abs):
    """Chunk-wise accumulators behind evaluate_codec_pipeline's norms, over the
    cells finite in both arrays: (l1_err, l2_err_sq, linf_err, signed_err,
    l1_ori, l2_ori_sq, linf_ori, q99_err, q99_ori, n_valid, n_corrupt,
    decoded_min, decoded_max).  `q99_abs` None skips the tail sums."""
    l1_err = l2_err_sq = linf_err = signed_err = 0.0
    l1_ori = l2_ori_sq = linf_ori = 0.0
    q99_err = q99_ori = 0.0
    n_valid = n_corrupt = 0
    decoded_min, decoded_max = math.inf, -math.inf
    with np.errstate(invalid="ignore"):
        for sl in _iter_chunk_slices(sample_np.shape, chunks):
            orig, dec = sample_np[sl], decoded[sl]
            finite_orig, finite_dec = np.isfinite(orig), np.isfinite(dec)
            n_corrupt += int(np.count_nonzero(finite_orig & ~finite_dec))
            valid = finite_orig & finite_dec
            nv = int(np.count_nonzero(valid))
            if nv == 0:
                continue
            n_valid += nv
            # Two float64 temporaries per chunk (the boolean index already copied):
            # e = decoded - orig, then |e| and |orig| in place.
            o_abs = orig[valid].astype(np.float64, copy=False)
            e_abs = dec[valid].astype(np.float64, copy=False)
            decoded_min, decoded_max = min(decoded_min, float(e_abs.min())), max(decoded_max, float(e_abs.max()))
            e_abs -= o_abs
            signed_err += float(e_abs.sum()); l2_err_sq += float(np.dot(e_abs, e_abs))
            np.abs(e_abs, out=e_abs); np.abs(o_abs, out=o_abs)
            l1_err += float(e_abs.sum()); linf_err = max(linf_err, float(e_abs.max(initial=0.0)))
            l1_ori += float(o_abs.sum()); l2_ori_sq += float(np.dot(o_abs, o_abs))
            linf_ori = max(linf_ori, float(o_abs.max(initial=0.0)))
            if q99_abs is not None:
                ext = o_abs >= q99_abs
                if ext.any():
                    q99_err += float(e_abs[ext].sum()); q99_ori += float(o_abs[ext].sum())
    return (l1_err, l2_err_sq, linf_err, signed_err, l1_ori, l2_ori_sq, linf_ori, q99_err, q99_ori,
            n_valid, n_corrupt, decoded_min, decoded_max)


def evaluate_codec_pipeline(sample_np: np.ndarray, dims, codec_kwargs: dict, chunks,
                            q99_abs: float | None = None, compute_gradient: bool = False,
                            gradient_axes=None, precheck_thresholds: dict | None = None):
    """
    Round-trip `sample_np` through a codec pipeline in memory and score it.
    Returns (compression_ratio, errors_dict, euclidean_distance).

    Cells that are non-finite in the original are fill and excluded from every
    norm.  Cells finite in the original but non-finite after decode are
    corruption: excluded from the norms and counted in N_Corrupt.

    The gradient metric is one more pass over the sample, so with
    `precheck_thresholds` it is skipped for combos that already fail a cheap
    gate (L1/L2/Linf/bias).
    """
    decoded, ratio = _zarr_roundtrip(sample_np, dims, codec_kwargs, chunks)
    want_q99 = q99_abs is not None and math.isfinite(q99_abs)
    with Timer("eval.metrics"):
        (l1_err, l2_err_sq, linf_err, signed_err, l1_ori, l2_ori_sq, linf_ori, q99_err, q99_ori,
         n_valid, n_corrupt, decoded_min, decoded_max) = _error_sums(sample_np, decoded, chunks,
                                                                     q99_abs if want_q99 else None)
    if not all(map(math.isfinite, (l1_err, l2_err_sq, linf_err))):
        raise CombinationProducedNonFiniteError(
            f"non-finite error accumulators after masking "
            f"(l1_err={l1_err}, l2_err_sq={l2_err_sq}, linf_err={linf_err})")

    l2_err, l2_ori = math.sqrt(l2_err_sq), math.sqrt(l2_ori_sq)
    errors = {
        "Relative_Error_L1": _rel(l1_err, l1_ori),
        "Relative_Error_L2": _rel(l2_err, l2_ori),
        "Relative_Error_Linf": _rel(linf_err, linf_ori),
        "Bias_Rel": _rel(abs(signed_err), l1_ori),
        "Decoded_Min": decoded_min if n_valid else float("nan"),
        "Decoded_Max": decoded_max if n_valid else float("nan"),
        "N_Corrupt": int(n_corrupt),
        "N_Valid": int(n_valid),
        "Q99_Rel": _rel(q99_err, q99_ori) if want_q99 else None,
    }

    do_grad = compute_gradient
    if compute_gradient and precheck_thresholds is not None:
        do_grad = all(within_limit(errors[m], precheck_thresholds.get(t)) for m, t in CHEAP_GATES)
    errors["Grad_Rel"] = _gradient_rel_l1(sample_np, decoded, axes=gradient_axes) if do_grad else None
    return ratio, errors, l2_err


def _gradient_rel_l1(orig: np.ndarray, decoded: np.ndarray, axes=None) -> float:
    """Sum|d(decoded) - d(orig)| / Sum|d(orig)| over finite differences along
    `axes` (default: every axis but the leading one).  Differences that do not
    run along the leading axis are taken over blocks of leading indices (about
    32 MiB of float64 each), so the temporaries do not grow with the sample."""
    if axes is None:
        axes = tuple(range(1, orig.ndim)) if orig.ndim > 1 else (0,)
    axes = tuple(ax % orig.ndim for ax in axes)
    n = orig.shape[0]
    step = max(1, n) if 0 in axes else max(1, (32 << 20) // (8 * max(1, int(np.prod(orig.shape[1:])))))
    err_sum = ori_sum = 0.0
    with np.errstate(invalid="ignore"):
        for start in range(0, n, step):
            o = orig[start:start + step].astype(np.float64, copy=False)
            d = decoded[start:start + step].astype(np.float64, copy=False)
            for ax in axes:
                do, dd = np.diff(o, axis=ax), np.diff(d, axis=ax)
                m = np.isfinite(do) & np.isfinite(dd)
                if m.any():
                    err_sum += float(np.abs(dd[m] - do[m]).sum())
                    ori_sum += float(np.abs(do[m]).sum())
                del do, dd, m
    if ori_sum == 0:
        return 0.0 if err_sum == 0 else float("inf")
    return err_sum / ori_sum


# =============================================================================
# 6. PERSISTENCE
# =============================================================================

def persist_with_codec_pipeline(da, store, component: str, codec_kwargs: dict,
                                inner_chunks=None, shards=None, verify: bool = True, q99_abs=None):
    """
    Write dask-backed `da` into `store` at `component` via dask.array.to_zarr.
    Returns (compression_ratio, errors, euclidean_distance); the last two are
    None unless `verify` re-reads the store.  One dask task writes one shard
    (or one chunk when sharding is skipped).
    """
    assert isinstance(da.data, dask.array.Array), "expects a dask-backed xr.DataArray"
    if inner_chunks is None or shards is None:
        auto_inner, auto_shard = compute_chunk_and_shard_shape(da.shape, da.dtype, dims=tuple(da.dims))
        inner_chunks = auto_inner if inner_chunks is None else inner_chunks
        shards = auto_shard if shards is None else shards

    write_unit = shards if shards is not None else inner_chunks
    zarr_kwargs = dict(zarr_format=3, dimension_names=tuple(da.dims), chunks=inner_chunks, **codec_kwargs)
    if shards is not None:
        zarr_kwargs["shards"] = shards

    with Timer("dask.array.to_zarr"):
        dask.array.to_zarr(da.data.rechunk(write_unit), store, component=component,
                           overwrite=True, compute=True, **zarr_kwargs)

    # use_consolidated=False: consolidated metadata written before this array does not list it.
    z = zarr.open_group(store, mode="r", use_consolidated=False)[component]
    count_bytes, count_bytes_stored = _info_bytes(z.info_complete())
    ratio = count_bytes / count_bytes_stored

    errors = euclidean_distance = None
    if verify:
        with Timer("compute_errors_distances"):
            errors, euclidean_distance = compute_errors_distances(
                dask.array.from_zarr(z, chunks=write_unit), da.data, q99_abs=q99_abs)
    return ratio, errors, euclidean_distance


def compute_errors_distances(da_compressed, da, q99_abs=None):
    """Error norms between two dask arrays (one dask.compute), masked like
    evaluate_codec_pipeline and with the same keys except N_Valid and Grad_Rel.
    Returns (errors, l2_error)."""
    da, da_compressed = da.astype(np.float64), da_compressed.astype(np.float64)  # integer squares would wrap
    finite_orig, finite_dec = np.isfinite(da), np.isfinite(da_compressed)
    valid = finite_orig & finite_dec
    o = dask.array.where(valid, da, 0)
    dec = dask.array.where(valid, da_compressed, np.nan)  # so nanmin/nanmax skip the masked cells
    err = dask.array.where(valid, da_compressed, 0) - o
    reductions = [
        np.abs(err).sum(), np.abs(o).sum(),
        np.sqrt((err ** 2).sum()), np.sqrt((o ** 2).sum()),
        np.abs(err).max(), np.abs(o).max(),
        err.sum(), (finite_orig & ~finite_dec).sum(),
        dask.array.nanmin(dec), dask.array.nanmax(dec),
    ]
    if q99_abs is not None:
        tail = np.abs(o) >= float(q99_abs)
        reductions += [dask.array.where(tail, np.abs(err), 0.0).sum(),
                       dask.array.where(tail, np.abs(o), 0.0).sum()]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        computed = dask.compute(*reductions)
    l1e, l1o, l2e, l2o, linfe, linfo, signed, ncorrupt, dmin, dmax = computed[:10]
    errors = {
        "Relative_Error_L1": _rel(l1e, l1o),
        "Relative_Error_L2": _rel(l2e, l2o),
        "Relative_Error_Linf": _rel(linfe, linfo),
        "Bias_Rel": _rel(abs(float(signed)), l1o),
        "Decoded_Min": float(dmin), "Decoded_Max": float(dmax),
        "N_Corrupt": int(ncorrupt),
        "Q99_Rel": _rel(computed[10], computed[11]) if q99_abs is not None else None,
    }
    return errors, l2e


# =============================================================================
# 7. MPI & TOPOLOGY
# =============================================================================

def detect_node_topology(comm=None):
    """(node_comm, ranks_on_node, local_rank) via MPI-3 shared-memory split,
    with a hostname fallback for old MPI implementations."""
    comm = comm or MPI.COMM_WORLD
    try:
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.Get_rank())
    except Exception:
        import socket
        names = comm.allgather(socket.gethostname())
        color = {n: i for i, n in enumerate(sorted(set(names)))}
        node_comm = comm.Split(color[socket.gethostname()], key=comm.Get_rank())
    return node_comm, node_comm.Get_size(), node_comm.Get_rank()


def detect_cores_available() -> int:
    """Cores visible to this process (respects cgroups / Slurm cpusets)."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def check_thread_oversubscription(abort_if_unsafe: bool = True, rank: int = 0, comm=None) -> None:
    """Warn on rank 0, and abort collectively unless `abort_if_unsafe` is off,
    when any THREAD_ENV_VARS entry is not 1."""
    comm = comm or MPI.COMM_WORLD
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
            comm.Abort(1)


# =============================================================================
# 8. PROGRESS & TIMING
# =============================================================================

_PROGRESS_COUNTERS = defaultdict(int)


def progress_bar(total, print_every=100, bar_width=40, key: str = "default"):
    """Progress line of rank 0's own share; call once per completed unit."""
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        return
    _PROGRESS_COUNTERS[key] += 1
    done = _PROGRESS_COUNTERS[key]
    if done % print_every == 0 or done == total:
        pct = done / total
        bar = "*" * int(bar_width * pct) + "-" * (bar_width - int(bar_width * pct))
        click.echo(f"[Rank {rank}] Progress: |{bar}| {pct*100:6.2f}% ({done}/{total})")


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
    print("\n=== Timing Summary (rank 0; ranks balanced via deterministic shuffle) ===")
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
