# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Library behind the ``dc_toolkit`` CLI.

Sections
  1. Sizes & dataset I/O
  2. Representative sampling      (which slices of a big field to score)
  3. Chunk & shard sizing         (zarr geometry, shared by eval and persist)
  4. Codec spaces                 (compressor x filter x serializer grids)
  5. Zarr sync bypass             (per-thread event loops for the sweep)
  6. In-memory evaluation         (encode -> decode -> error metrics)
  7. Persistence                  (dask -> zarr LocalStore, optional verify)
  8. MPI, threads & topology
  9. Result helpers, progress, timing
"""
import asyncio
import atexit
import math
import os
import re
import sys
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from typing import Optional, Tuple

import click
import dask
import dask.array
import humanize
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import zfpy
from mpi4py import MPI
from zarr.api.asynchronous import create_array as _zarr_async_create_array
from zarr.codecs import numcodecs as zarrcodecs_nc


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


# Codec-internal thread pools that must be pinned to 1 under thread-per-combo
# parallelism (see check_thread_oversubscription).
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLOSC_NTHREADS", "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "OMP_THREAD_LIMIT",
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


def open_zarr_memstore():
    return zarr.storage.MemoryStore()


def open_zarr_localstore(path: str, read_only: bool = True):
    """Open a zarr v3 LocalStore; returns (group, store).  Keep both alive."""
    store = zarr.storage.LocalStore(path, read_only=read_only)
    return zarr.open_group(store, mode="r" if read_only else "a"), store


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
        click.echo(f"dataset.nbytes = {humanize.naturalsize(ds.nbytes, binary=True)}")
        if field_to_compress is not None:
            click.echo(f"{field_to_compress}.nbytes = "
                       f"{humanize.naturalsize(ds[field_to_compress].nbytes, binary=True)}")
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
    if std in _CF_TIME_STANDARD_NAMES or m.get("axis") == "T" or "calendar" in m:
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
    reproducible (evenly spaced indices along time/vertical dims).

    policy="cascade": spend the budget on time steps first, keeping a
    budget-aware minimum of vertical levels.  policy="balanced": equal
    log-space split across all stride dims.  Raises SampleTooLargeError when
    a single horizontal slab does not fit.
    """
    nbytes = int(da.dtype.itemsize) * int(np.prod(da.shape))
    hsize = lambda b: humanize.naturalsize(b, binary=True)  # noqa: E731
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
               f"Remedies: raise --eval-data-size-limit, reduce --threads-per-rank, "
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
                                  dims=None, allow_spatial_split: bool = True):
    """(inner_chunk_shape, shard_shape).  shard_shape is None when one inner
    chunk already reaches the shard target (a shard would hold <= 1 chunk)."""
    itemsize = int(np.dtype(dtype).itemsize)
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
# 4. CODEC SPACES
# =============================================================================
# Each space is a list of (index, codec-or-None).  Indices are what the sweep
# records and what compress_with_optimal resolves, so the grids below and the
# dtype-dependent parts must be rebuilt identically in both commands.

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
    """Lossless bytes->bytes compressors.  `da`/`with_lossy` are accepted for
    signature symmetry only.  Byte shuffling is only available through Blosc's
    own `shuffle` parameter (a standalone Shuffle codec breaks after a
    compressing serializer)."""
    classes, include_none = _select_classes(
        [zarrcodecs_nc.Blosc, zarrcodecs_nc.LZ4, zarrcodecs_nc.Zstd,
         zarrcodecs_nc.Zlib, zarrcodecs_nc.BZ2, zarrcodecs_nc.LZMA],
        compressor_class, "compressor")
    space = [None] if include_none else []
    for cls in classes:
        if cls is zarrcodecs_nc.Blosc:
            space += [cls(cname=c, clevel=l, shuffle=s)
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
    return list(enumerate(space))


def filter_space(da, with_lossy=True, filter_class="all", data_range=None):
    """Array->array filters.  Integer dtypes get Delta only.  `data_range`
    must be the FULL field's (min, max) when `da` is a sample: FixedScaleOffset
    does not clip, so out-of-range production values would corrupt silently."""
    classes = [zarrcodecs_nc.Delta]
    if with_lossy:
        classes += [zarrcodecs_nc.BitRound, zarrcodecs_nc.Quantize, zarrcodecs_nc.FixedScaleOffset]
        if np.issubdtype(da.dtype, np.floating) and da.dtype.itemsize > 4:
            classes.append(zarrcodecs_nc.AsType)  # f64 -> f32 down-cast
    if da.dtype.kind == "i":
        if filter_class.lower() not in ("all", "delta", "none"):
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
    return list(enumerate(space))


class ZFPYFlat(zarrcodecs_nc.ZFPY, codec_name="zfpy"):
    """ZFPY that flattens each chunk to 1-D before zfp.  zfp's per-axis header
    budget shrinks with rank (2**24 at 2-D, 2**16 at 3-D) and the DYAMOND cell
    axis overflows it; 1-D never does.  The store keeps the natural shape and
    the plain "zfpy" codec name, so stock readers decode it."""

    async def _encode_single(self, chunk_data, chunk_spec):
        arr = np.ascontiguousarray(chunk_data.as_ndarray_like()).reshape(-1)
        out = await asyncio.to_thread(self._codec.encode, arr)
        return chunk_spec.prototype.buffer.from_bytes(out)


def serializer_space(da, with_lossy=True, serializer_class="all"):
    """Array->bytes serializers: PCodec always, ZFPY when lossy is allowed
    (fixed-rate mode only for integer dtypes)."""
    classes = [zarrcodecs_nc.PCodec] + ([zarrcodecs_nc.ZFPY] if with_lossy else [])
    classes, include_none = _select_classes(classes, serializer_class, "serializer")
    space = [None] if include_none else []
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
            if da.dtype.kind == "i":
                modes = modes[2:]
            space += [ZFPYFlat(mode=mode, **{param: fn(k)})
                      for mode, param, fn in modes for k in _ZFPY_K_GRID]
    return list(enumerate(space))


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


def full_field_data_range(da):
    """Finite (min, max) over the whole (dask-backed) field, or None if the
    field is constant / has no finite values."""
    data = da.data if hasattr(da, "data") else np.asarray(da)
    try:
        if isinstance(data, dask.array.Array):
            finite = dask.array.isfinite(data)
            dmin = float(dask.array.where(finite, data, np.inf).min().compute())
            dmax = float(dask.array.where(finite, data, -np.inf).max().compute())
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
    _FSO_TARGET_UINTS.  Falls back to `da`'s own range when data_range is None
    (only valid if `da` is the full field)."""
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


def combo_is_valid(filt, serializer) -> bool:
    """Reject pairings that crash inside the codecs: FixedScaleOffset emits
    unsigned ints, which ZFPY (non-fixed-rate modes) and 8-bit PCodec refuse."""
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
    A None component is omitted so zarr's own default applies (note: for the
    compressor this means zarr's default Zstd, not "no compression")."""
    kwargs = {}
    if filt is not None:
        kwargs["filters"] = [filt]
    if compressor is not None:
        kwargs["compressors"] = [compressor]
    kwargs["serializer"] = "auto" if serializer is None else serializer
    return kwargs


# =============================================================================
# 5. ZARR SYNC BYPASS
# =============================================================================
# zarr 3's sync API funnels every call through one process-global event loop,
# which serialises codec work from concurrent threads (5x slowdown at 32
# threads).  With the bypass on, each user thread owns a persistent event loop
# and all loops share one bounded ThreadPoolExecutor, so total OS threads stay
# at user_threads + shared_workers instead of user_threads * 32.

_thread_local_loops = threading.local()
_shared_executor = None
_shared_executor_lock = threading.Lock()


def _get_or_create_shared_executor(max_workers: int) -> ThreadPoolExecutor:
    global _shared_executor
    with _shared_executor_lock:
        if _shared_executor is None:
            _shared_executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)),
                                                  thread_name_prefix="bypass_codec")
    return _shared_executor


def _get_thread_event_loop() -> asyncio.AbstractEventLoop:
    loop = getattr(_thread_local_loops, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if _shared_executor is not None:
            loop.set_default_executor(_shared_executor)
        _thread_local_loops.loop = loop
    return loop


class AsyncBypass:
    """Process-wide toggle for the bypass (cli --bypass-zarr-sync)."""
    enabled: bool = False
    threads_per_rank: int = 1

    @classmethod
    def enable(cls, threads_per_rank: int = 1) -> None:
        cls.enabled = True
        cls.threads_per_rank = max(1, int(threads_per_rank))
        _get_or_create_shared_executor(cls.threads_per_rank)

    @classmethod
    def run(cls, coro):
        """Run a zarr coroutine: on this thread's loop if enabled, otherwise on
        zarr's global sync loop (identical to the sync API)."""
        if cls.enabled:
            return _get_thread_event_loop().run_until_complete(coro)
        from zarr.core.sync import sync
        return sync(coro)


# =============================================================================
# 6. IN-MEMORY EVALUATION
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


# FixedScaleOffset casts NaN fill cells to int and numpy warns; fill cells are
# masked out of every norm, so the warning is noise.  (A real FSO overflow is
# silent and guarded by full_field_data_range + the verify gate instead.)
_IGNORE_CAST_WARNING = dict(action="ignore", message="invalid value encountered in cast",
                            category=RuntimeWarning)


async def _zarr_roundtrip(sample_np, dims, codec_kwargs, chunks):
    """create + encode + info + decode on a MemoryStore.  Returns (decoded, ratio)."""
    with Timer("eval.create_array"):
        z = await _zarr_async_create_array(
            store=zarr.storage.MemoryStore(), name="_tmp_eval",
            shape=sample_np.shape, dtype=sample_np.dtype, chunks=chunks,
            zarr_format=3, dimension_names=tuple(dims), **codec_kwargs)
    with Timer("eval.encode"), warnings.catch_warnings():
        warnings.filterwarnings(**_IGNORE_CAST_WARNING)
        await z.setitem(Ellipsis, sample_np)
    with Timer("eval.info_complete"):
        count_bytes, count_bytes_stored = _info_bytes(await z.info_complete())
    with Timer("eval.decode"):
        decoded = await z.getitem(Ellipsis)
    return decoded, count_bytes / count_bytes_stored


def evaluate_codec_pipeline(sample_np: np.ndarray, dims, codec_kwargs: dict, chunks,
                            q99_abs: float | None = None, compute_gradient: bool = False,
                            gradient_axes=None, precheck_thresholds: dict | None = None):
    """
    Round-trip `sample_np` through a codec pipeline in memory and score it.
    Returns (compression_ratio, errors_dict, euclidean_distance).  Thread-safe.

    Cells that are non-finite in the original are fill and excluded from every
    norm.  Cells finite in the original but non-finite after decode are
    corruption: excluded from the norms and counted in N_Corrupt.

    The gradient metric needs a second decode, so with `precheck_thresholds`
    it is skipped for combos that already fail a cheap gate (L1/L2/Linf/bias).
    """
    decoded, ratio = AsyncBypass.run(_zarr_roundtrip(sample_np, dims, codec_kwargs, chunks))

    with Timer("eval.metrics"):
        l1_err = l2_err_sq = linf_err = signed_err = 0.0
        l1_ori = l2_ori_sq = linf_ori = 0.0
        q99_err = q99_ori = 0.0
        n_valid = n_corrupt = 0
        decoded_min, decoded_max = math.inf, -math.inf
        want_q99 = q99_abs is not None and math.isfinite(q99_abs)

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
                o = orig[valid].astype(np.float64, copy=False)
                d = dec[valid].astype(np.float64, copy=False)
                e = d - o
                e_abs, o_abs = np.abs(e), np.abs(o)
                l1_err += float(e_abs.sum()); l2_err_sq += float((e * e).sum())
                linf_err = max(linf_err, float(e_abs.max(initial=0.0)))
                signed_err += float(e.sum())
                l1_ori += float(o_abs.sum()); l2_ori_sq += float((o_abs * o_abs).sum())
                linf_ori = max(linf_ori, float(o_abs.max(initial=0.0)))
                decoded_min, decoded_max = min(decoded_min, float(d.min())), max(decoded_max, float(d.max()))
                if want_q99:
                    ext = o_abs >= q99_abs
                    if ext.any():
                        q99_err += float(e_abs[ext].sum()); q99_ori += float(o_abs[ext].sum())
        del decoded
        if not all(map(math.isfinite, (l1_err, l2_err_sq, linf_err))):
            raise CombinationProducedNonFiniteError(
                f"non-finite error accumulators after masking "
                f"(l1_err={l1_err}, l2_err_sq={l2_err_sq}, linf_err={linf_err})")

    l2_err, l2_ori = math.sqrt(l2_err_sq), math.sqrt(l2_ori_sq)
    rel = lambda a, b: float(a) / float(b) if b != 0 else float("inf")  # noqa: E731
    errors = {
        "Relative_Error_L1": rel(l1_err, l1_ori),
        "Relative_Error_L2": rel(l2_err, l2_ori),
        "Relative_Error_Linf": rel(linf_err, linf_ori),
        "Bias_Rel": rel(abs(signed_err), l1_ori),
        "Decoded_Min": decoded_min if n_valid else float("nan"),
        "Decoded_Max": decoded_max if n_valid else float("nan"),
        "N_Corrupt": int(n_corrupt),
        "N_Valid": int(n_valid),
        "Q99_Rel": rel(q99_err, q99_ori) if want_q99 else None,
    }

    do_grad = compute_gradient
    if compute_gradient and precheck_thresholds is not None:
        def passes(val, lim):
            return True if (val is None or lim is None or not math.isfinite(lim)) else float(val) <= float(lim)
        do_grad = (passes(errors["Relative_Error_L1"], precheck_thresholds.get("l1"))
                   and passes(errors["Relative_Error_L2"], precheck_thresholds.get("l2"))
                   and passes(errors["Relative_Error_Linf"], precheck_thresholds.get("linf"))
                   and passes(errors["Bias_Rel"], precheck_thresholds.get("bias")))
    if do_grad:
        decoded2, _ = AsyncBypass.run(_zarr_roundtrip(sample_np, dims, codec_kwargs, chunks))
        errors["Grad_Rel"] = _gradient_rel_l1(sample_np, decoded2, axes=gradient_axes)
        del decoded2
    else:
        errors["Grad_Rel"] = None
    return ratio, errors, l2_err


def _gradient_rel_l1(orig: np.ndarray, decoded: np.ndarray, axes=None) -> float:
    """Sum|d(decoded) - d(orig)| / Sum|d(orig)| over finite differences along
    `axes` (default: every axis but the leading one)."""
    if axes is None:
        axes = tuple(range(1, orig.ndim)) if orig.ndim > 1 else (0,)
    err_sum = ori_sum = 0.0
    with np.errstate(invalid="ignore"):
        for ax in axes:
            do = np.diff(orig.astype(np.float64, copy=False), axis=ax)
            dd = np.diff(decoded.astype(np.float64, copy=False), axis=ax)
            m = np.isfinite(do) & np.isfinite(dd)
            if m.any():
                err_sum += float(np.abs(dd[m] - do[m]).sum())
                ori_sum += float(np.abs(do[m]).sum())
            del do, dd, m
    if ori_sum == 0:
        return 0.0 if err_sum == 0 else float("inf")
    return err_sum / ori_sum


# =============================================================================
# 7. PERSISTENCE
# =============================================================================

def persist_with_codec_pipeline(da, store, component: str, codec_kwargs: dict,
                                inner_chunks=None, shards=None, verify: bool = True,
                                verbose: bool = True, rank: int = 0, q99_abs=None):
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

    with Timer("dask.array.to_zarr"), warnings.catch_warnings():
        warnings.filterwarnings(**_IGNORE_CAST_WARNING)
        dask.array.to_zarr(da.data.rechunk(write_unit), store, component=component,
                           overwrite=True, compute=True, **zarr_kwargs)

    z = zarr.open_group(store, mode="r")[component]
    info = z.info_complete()
    count_bytes, count_bytes_stored = _info_bytes(info)
    ratio = count_bytes / count_bytes_stored
    if verbose and rank == 0:
        click.echo("-" * 80); click.echo(info)

    errors = euclidean_distance = None
    if verify:
        with Timer("compute_errors_distances"):
            report, errors, euclidean_distance, _ = compute_errors_distances(
                dask.array.from_zarr(z, chunks=write_unit), da.data, q99_abs=q99_abs)
        if verbose and rank == 0:
            click.echo("-" * 80); click.echo(report)
            click.echo("-" * 80); click.echo(f"Euclidean Distance: {euclidean_distance}")
            click.echo("-" * 80)
    return ratio, errors, euclidean_distance


def compute_errors_distances(da_compressed, da, q99_abs=None):
    """Dask-lazy error norms between two arrays, masked like
    evaluate_codec_pipeline.  Returns (report_str, errors, l2_error, rel_l2)."""
    finite_orig, finite_dec = np.isfinite(da), np.isfinite(da_compressed)
    valid = finite_orig & finite_dec
    o = dask.array.where(valid, da, 0)
    err = dask.array.where(valid, da_compressed, 0) - o
    reductions = [
        np.abs(err).sum(), np.abs(o).sum(),
        np.sqrt((err ** 2).sum()), np.sqrt((o ** 2).sum()),
        np.abs(err).max(), np.abs(o).max(),
        err.sum(), (finite_orig & ~finite_dec).sum(),
    ]
    if q99_abs is not None:
        tail = np.abs(o) >= float(q99_abs)
        reductions += [dask.array.where(tail, np.abs(err), 0.0).sum(),
                       dask.array.where(tail, np.abs(o), 0.0).sum()]
    computed = dask.compute(*reductions)
    l1e, l1o, l2e, l2o, linfe, linfo, signed, ncorrupt = computed[:8]

    def rel(e, o):
        if o == 0:
            return 0.0 if e == 0 else float("inf")
        return float(e) / float(o)

    errors = {
        "Relative_Error_L1": rel(l1e, l1o),
        "Relative_Error_L2": rel(l2e, l2o),
        "Relative_Error_Linf": rel(linfe, linfo),
        "Bias_Rel": rel(abs(float(signed)), l1o),
        "N_Corrupt": int(ncorrupt),
        "Q99_Rel": rel(computed[8], computed[9]) if q99_abs is not None else None,
    }
    report = "\n".join(f"{k:20s}: {v:.3e}" if isinstance(v, float) else f"{k:20s}: {v}"
                       for k, v in errors.items())
    return report, errors, l2e, errors["Relative_Error_L2"]


# =============================================================================
# 8. MPI, THREADS & TOPOLOGY
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


def compute_default_threads_per_rank(ranks_on_node: int, cores_available: int | None = None) -> int:
    cores = detect_cores_available() if cores_available is None else cores_available
    return max(1, cores // max(1, ranks_on_node))


def broadcast_numpy(arr, comm=None, root: int = 0) -> np.ndarray:
    """Bcast a numpy array from `root` (non-root ranks pass None).  Uses the
    buffer protocol with the dtype's MPI type, so the 2^31 count limit is in
    elements, not bytes (~16 GB for float64)."""
    comm = comm or MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == root:
        if arr is None:
            raise ValueError("broadcast_numpy: root rank must provide a numpy array.")
        meta = (tuple(arr.shape), str(arr.dtype))
    else:
        meta = None
    shape, dtype_str = comm.bcast(meta, root=root)
    buf = np.ascontiguousarray(arr) if rank == root else np.empty(shape, dtype=np.dtype(dtype_str))
    comm.Bcast(buf, root=root)
    return buf


def check_thread_oversubscription(abort_if_unsafe: bool = True, rank: int = 0, comm=None) -> None:
    """Abort (collectively) unless every THREAD_ENV_VARS entry equals 1.  Also
    pins zarr's internal pool when several ranks share a node without the
    bypass, since each rank would otherwise spawn its own ~32-thread pool."""
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
    if not AsyncBypass.enabled:
        try:
            if detect_node_topology(comm)[1] > 1:
                zarr.config.set({"threading.max_workers": 1})
        except Exception:
            pass


# =============================================================================
# 9. RESULT HELPERS, PROGRESS, TIMING
# =============================================================================

def get_indexes(arr, indices) -> np.ndarray:
    """Map codec repr strings back to their integer index using a
    config_space_{var}.csv column ("(idx, repr)" strings); -1 for "None" or
    unknown reprs (warned).  Used for plot hover labels."""
    codec_id = {}
    for ind in indices:
        idx_str, codec_repr = str(ind)[1:-1].split(", ", 1)
        codec_id[codec_repr] = int(idx_str)
    ids, unknown = [], 0
    for item in arr:
        if item == "None":
            ids.append(-1)
        elif item in codec_id:
            ids.append(codec_id[item])
        else:
            ids.append(-1); unknown += 1
    if unknown:
        click.echo(f"[get_indexes] WARNING: {unknown} codec repr(s) not found in the config-space "
                   f"CSV (labelled -1).  Are the .npy and CSV from the same evaluate_combos run?",
                   err=True)
    return np.asarray(ids, dtype=int)


def slice_array(arr: pd.array, indices_ls: list) -> np.ndarray:
    return np.hstack(tuple(arr[[ind]] for ind in indices_ls))


_PROGRESS_LOCK = threading.Lock()
_PROGRESS_COUNTERS = defaultdict(int)


def progress_bar(total, print_every=100, bar_width=40, key: str = "default"):
    """Thread-safe progress line on rank 0; call once per completed unit."""
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        return
    with _PROGRESS_LOCK:
        _PROGRESS_COUNTERS[key] += 1
        done = _PROGRESS_COUNTERS[key]
        if done % print_every == 0 or done == total:
            pct = done / total
            bar = "*" * int(bar_width * pct) + "-" * (bar_width - int(bar_width * pct))
            click.echo(f"[Rank {rank}] Progress: |{bar}| {pct*100:6.2f}% ({done}/{total})")


_TIMINGS_LOCK = threading.Lock()
_TIMINGS = defaultdict(list)


class Timer:
    """`with Timer("label"):` accumulates wall time per label (thread-safe)."""

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
    if not _TIMINGS or MPI.COMM_WORLD.Get_rank() != 0:
        return
    print("\n=== Timing Summary (rank 0; ranks balanced via deterministic shuffle) ===")
    print("Sum of Total = thread-seconds inside the eval pipeline (excludes bcast,")
    print("file I/O, dask graph setup, and result-write overhead).\n")
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
