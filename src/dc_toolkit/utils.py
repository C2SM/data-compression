# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import sys
import os
import math
import click
import humanize
import threading
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import dask
import dask.array
import pandas as pd
import xarray as xr
import zarr
from zarr_any_numcodecs import AnyNumcodecsArrayArrayCodec, AnyNumcodecsArrayBytesCodec
import numcodecs
import numcodecs.zarr3
import zfpy
from ebcc.filter_wrapper import EBCC_Filter
from ebcc.zarr_filter import EBCCZarrFilter
from mpi4py import MPI
import time
from collections import defaultdict
from itertools import product
import atexit
import re

# numcodecs-wasm filters
from numcodecs_wasm_asinh import Asinh
from numcodecs_wasm_fixed_offset_scale import FixedOffsetScale
# numcodecs-wasm serializers
from numcodecs_wasm_zfp import Zfp

os.environ["EBCC_LOG_LEVEL"] = "4"  # ERROR (suppress WARN and below)


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

def build_representative_sample(
    da: xr.DataArray,
    size_limit_bytes: int,
    rank: int = 0,
) -> xr.DataArray:
    """
    Return a subset of `da` that fits within `size_limit_bytes`, built to be
    representative of the full field.

    Strategy
    --------
    - If the whole field fits under the limit: return it unchanged.
    - Otherwise: keep trailing spatial dims full (that's where codecs exploit
      smoothness), stride-sample along the LEADING dim (usually time or
      ensemble).  This preserves spatial structure and samples across the
      leading axis instead of taking a corner.
    - A deterministic `np.linspace`-style stride is used so results are
      reproducible between `evaluate_combos` and `compress_with_optimal`.
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

    leading_dim = da.dims[0]
    leading_size = da.sizes[leading_dim]

    # Per-slice (fixing leading index) byte cost.
    trailing_bytes = int(da.dtype.itemsize) * int(np.prod(da.shape[1:]))
    if trailing_bytes == 0:
        return da  # degenerate; nothing to sample

    max_slices = max(1, size_limit_bytes // trailing_bytes)
    max_slices = int(min(max_slices, leading_size))

    # Evenly-spaced indices spanning [0, leading_size-1].
    indices = np.linspace(0, leading_size - 1, num=max_slices, dtype=int)
    indices = np.unique(indices).tolist()

    sampled = da.isel({leading_dim: indices})

    if rank == 0:
        click.echo(
            f"[sample] field is "
            f"{humanize.naturalsize(nbytes, binary=True)} > limit "
            f"{humanize.naturalsize(size_limit_bytes, binary=True)}; "
            f"sampled {len(indices)}/{leading_size} along '{leading_dim}' "
            f"-> {humanize.naturalsize(sampled.nbytes, binary=True)}."
        )

    return sampled


# =============================================================================
# CHUNK & SHARD SIZING
# =============================================================================

def compute_chunk_shape_for_eval(shape, dtype, target_mib: int = 16):
    """
    Pick a simple chunk shape for in-memory evaluation.  Target ~target_mib
    per chunk.  Chunks the leading dim; keeps trailing dims whole.
    """
    itemsize = np.dtype(dtype).itemsize
    trailing = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    bytes_per_leading = itemsize * trailing
    if bytes_per_leading == 0:
        return tuple(shape)
    target_bytes = target_mib * 2**20
    leading_chunk = max(1, target_bytes // bytes_per_leading)
    leading_chunk = int(min(leading_chunk, shape[0]))
    return (leading_chunk,) + tuple(shape[1:])


def compute_chunk_and_shard_shape(
    shape,
    dtype,
    inner_mib: int = 16,
    shard_mib: int = 512,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Auto-compute (inner_chunk_shape, shard_shape) for zarr v3 sharding.

    Rules of thumb:
      - Inner chunks: ~inner_mib (good for partial reads).
      - Shards:       ~shard_mib (amortizes per-file overhead; big write unit).
      - Shard is an integer multiple of inner on every axis.
    """
    itemsize = np.dtype(dtype).itemsize
    ndim = len(shape)

    # ---- inner chunk shape: target inner_mib bytes ----
    inner_target_bytes = inner_mib * 2**20
    inner = list(shape)
    # Reduce the leading dim first; keep trailing dims whole where possible.
    trailing = int(np.prod(shape[1:])) if ndim > 1 else 1
    bytes_per_leading = itemsize * trailing
    if bytes_per_leading > 0:
        inner[0] = max(1, min(shape[0], inner_target_bytes // bytes_per_leading))
    inner = tuple(int(x) for x in inner)

    # ---- shard shape: integer multiple of inner, target shard_mib ----
    inner_bytes = itemsize * int(np.prod(inner))
    if inner_bytes == 0:
        return inner, inner
    shard_target_bytes = shard_mib * 2**20
    multiplier = max(1, shard_target_bytes // inner_bytes)

    # Grow the leading dim by `multiplier`, capped by the array extent.
    shard = list(inner)
    shard[0] = min(shape[0], inner[0] * multiplier)
    # Round down to an exact multiple of inner[0].
    shard[0] = (shard[0] // inner[0]) * inner[0]
    shard[0] = max(shard[0], inner[0])
    shard = tuple(int(x) for x in shard)

    return inner, shard


# -----------------------------------------------------------------------------
# Kept from original: EBCC-specific chunk search
# -----------------------------------------------------------------------------
def compute_chunks(data, min_height=0, max_height=None, min_width=0, max_width=None):
    lat_dim = data.shape[0]
    lon_dim = data.shape[1]
    height = lat_dim
    width = lon_dim

    if max_height is None:
        max_height = lat_dim
    if max_width is None:
        max_width = lat_dim

    keep_searching_height = True
    keep_searching_width = True

    for n in [2, 3, 5]:
        for m in range(10):
            d = n * (m + 1)
            for p in range(7, -1, -1):
                if keep_searching_height or keep_searching_width:
                    if keep_searching_height:
                        n_chunks_height = d ** p
                        if height % n_chunks_height == 0:
                            new_height = height / n_chunks_height
                            if (new_height >= min_height) and (new_height <= max_height):
                                height = new_height
                                keep_searching_height = False
                    if keep_searching_width and p > 0:
                        n_chunks_width = d ** (p - 1)
                        if width % n_chunks_width == 0:
                            new_width = width / n_chunks_width
                            if (new_width >= min_width) and (new_width <= max_width):
                                width = new_width
                                keep_searching_width = False
                else:
                    return (height, width, n_chunks_height, n_chunks_width)

    # All loops exhausted without both dims having found a valid factoring.
    # Before this raise, control fell off the end and the function implicitly
    # returned None; the caller (serializer_space's EBCC branch) then tried to
    # unpack None into 4 names and crashed with a cryptic TypeError.  Now we
    # report the actual problem: no (height_divisor, width_divisor) of the form
    # (n*(m+1))^p with n in {2,3,5}, m in [0..9], p in [0..7] maps both
    # dimensions into their allowed [min_*, max_*] bands.  Typical trigger:
    # a field with one dimension that doesn't factor cleanly into small primes
    # (e.g. a prime width, or a dimension smaller than min_*).
    raise ValueError(
        f"compute_chunks: no valid EBCC chunking found for shape "
        f"({lat_dim}, {lon_dim}) under constraints "
        f"height in [{min_height}, {max_height}], "
        f"width in [{min_width}, {max_width}]. "
        f"EBCC needs each dim divisible by (n*(m+1))^p for small n/m/p so the "
        f"resulting block fits in the allowed range.  Resolved so far: "
        f"height_done={not keep_searching_height}, "
        f"width_done={not keep_searching_width}.  "
        f"Workarounds: widen the min/max bounds, pad the field to a "
        f"factor-friendly shape, or pick a non-EBCC serializer."
    )


# =============================================================================
# CODEC SPACES
# =============================================================================

def compressor_space(da, with_lossy=True, with_numcodecs_wasm=True, with_ebcc=True, compressor_class="all"):
    compressor_space = []
    _COMPRESSORS = [
        numcodecs.zarr3.Blosc, numcodecs.zarr3.LZ4, numcodecs.zarr3.Zstd,
        numcodecs.zarr3.Zlib, numcodecs.zarr3.GZip, numcodecs.zarr3.BZ2,
        numcodecs.zarr3.LZMA,
    ]
    _COMPRESSOR_MAP = {cls.__name__.lower(): cls for cls in _COMPRESSORS}

    if compressor_class.lower() == "all":
        pass
    elif compressor_class.lower() in _COMPRESSOR_MAP:
        _COMPRESSORS = [_COMPRESSOR_MAP[compressor_class.lower()]]
    elif compressor_class.lower() == "none":
        _COMPRESSORS = []
        compressor_space.append(None)

    for compressor in _COMPRESSORS:
        if compressor == numcodecs.zarr3.Blosc:
            for cname in numcodecs.blosc.list_compressors():
                for clevel in [1, 6, 9]:
                    for shuffle in [-1, 0, 1, 2]:
                        compressor_space.append(compressor(cname=cname, clevel=clevel, shuffle=shuffle))
        elif compressor == numcodecs.zarr3.LZ4:
            for acceleration in [1, 10, 100]:
                compressor_space.append(compressor(acceleration=acceleration))
        elif compressor == numcodecs.zarr3.Zstd:
            for level in [0, 1, 9, 22]:
                compressor_space.append(compressor(level=level))
        elif compressor == numcodecs.zarr3.Zlib:
            for level in [1, 6, 9]:
                compressor_space.append(compressor(level=level))
        elif compressor == numcodecs.zarr3.GZip:
            for level in [1, 6, 9]:
                compressor_space.append(compressor(level=level))
        elif compressor == numcodecs.zarr3.BZ2:
            for level in [1, 6, 9]:
                compressor_space.append(compressor(level=level))
        elif compressor == numcodecs.zarr3.LZMA:
            for preset in [1, 6, 9]:
                compressor_space.append(compressor(preset=preset))

    return list(zip(range(len(compressor_space)), compressor_space))


def filter_space(da, with_lossy=True, with_numcodecs_wasm=True, with_ebcc=True, filter_class="all"):
    filter_space = []

    _FILTERS = [numcodecs.zarr3.Delta]
    if with_lossy:
        _FILTERS += [numcodecs.zarr3.BitRound, numcodecs.zarr3.Quantize]
    if with_numcodecs_wasm:
        if with_lossy:
            _FILTERS.append(Asinh)
        _FILTERS.append(FixedOffsetScale)
    if da.dtype.kind == "i":
        _FILTERS = [numcodecs.zarr3.Delta]

    _FILTER_MAP = {cls.__name__.lower(): cls for cls in _FILTERS}

    if filter_class.lower() == "all":
        pass
    elif filter_class.lower() in _FILTER_MAP:
        _FILTERS = [_FILTER_MAP[filter_class.lower()]]
    elif filter_class.lower() == "none":
        _FILTERS = []
        filter_space.append(None)

    for filt in _FILTERS:
        if filt == numcodecs.zarr3.Delta:
            if np.issubdtype(da.dtype, np.number):
                filter_space.append(filt(dtype=str(da.dtype)))
        elif filt == numcodecs.zarr3.BitRound:
            for keepbits in valid_keepbits_for_bitround(da, step=9):
                filter_space.append(filt(keepbits=keepbits))
        elif filt == numcodecs.zarr3.Quantize:
            for digits in valid_digits_for_quantize(da, step=4):
                filter_space.append(filt(digits=digits, dtype=str(da.dtype)))
        elif filt == Asinh:
            filter_space.append(
                AnyNumcodecsArrayArrayCodec(
                    filt(linear_width=compute_linear_width(da, quantile=0.01, compute=True))
                )
            )
        elif filt == FixedOffsetScale:
            mean_val, std_val, min_val, max_val = dask.compute(
                da.mean(skipna=True), da.std(skipna=True),
                da.min(skipna=True), da.max(skipna=True),
            )

            def _safe_scale(x, min_eps=1e-12):
                if not np.isfinite(x):
                    return None
                if abs(x) < min_eps:
                    return None
                return float(x)

            std_safe = _safe_scale(std_val)
            if np.isfinite(mean_val) and std_safe is not None:
                filter_space.append(AnyNumcodecsArrayArrayCodec(
                    filt(offset=float(mean_val), scale=std_safe)
                ))

            rng = max_val - min_val
            rng_safe = _safe_scale(rng)
            if np.isfinite(min_val) and rng_safe is not None:
                filter_space.append(AnyNumcodecsArrayArrayCodec(
                    filt(offset=float(min_val), scale=rng_safe)
                ))

    return list(zip(range(len(filter_space)), filter_space))


def serializer_space(da, with_lossy=True, with_numcodecs_wasm=True, with_ebcc=True, serializer_class="all"):
    is_int = (da.dtype.kind == "i")
    serializer_space = []

    _SERIALIZERS = [numcodecs.zarr3.PCodec]
    if with_lossy:
        _SERIALIZERS.append(numcodecs.zarr3.ZFPY)
    if with_ebcc and with_lossy:
        _SERIALIZERS.append(EBCCZarrFilter)
    if with_numcodecs_wasm and with_lossy:
        _SERIALIZERS.append(Zfp)

    _SERIALIZER_MAP = {cls.__name__.lower(): cls for cls in _SERIALIZERS}

    if serializer_class.lower() == "all":
        pass
    elif serializer_class.lower() in _SERIALIZER_MAP:
        _SERIALIZERS = [_SERIALIZER_MAP[serializer_class.lower()]]
    elif serializer_class.lower() == "none":
        _SERIALIZERS = []
        serializer_space.append(None)

    for serializer in _SERIALIZERS:
        if serializer == numcodecs.zarr3.PCodec:
            for level in [0, 4, 8, 12]:
                for delta_encoding_order in [0, 3, 7]:
                    serializer_space.append(serializer(
                        level=level, mode_spec="auto",
                        delta_spec="auto", delta_encoding_order=delta_encoding_order,
                    ))
        elif serializer in (numcodecs.zarr3.ZFPY, Zfp):
            _ZFP_MODES = [
                ("fixed-accuracy",  zfpy.mode_fixed_accuracy,  "tolerance", compute_fixed_accuracy_param),
                ("fixed-precision", zfpy.mode_fixed_precision, "precision", compute_fixed_precision_param),
                ("fixed-rate",      zfpy.mode_fixed_rate,      "rate",      compute_fixed_rate_param),
            ]
            if is_int:
                _ZFP_MODES = [m for m in _ZFP_MODES if m[0] == "fixed-rate"]
            for mode_str, zfpy_mode, param_name, param_fn in _ZFP_MODES:
                for k in range(3):
                    val = param_fn(k)
                    if serializer is numcodecs.zarr3.ZFPY:
                        serializer_space.append(serializer(mode=zfpy_mode, **{param_name: val}))
                    else:
                        codec = serializer(mode=mode_str, **{param_name: val})
                        serializer_space.append(AnyNumcodecsArrayBytesCodec(codec))
        elif serializer == EBCCZarrFilter:
            data = da.squeeze()  # TODO: add more checks on the shape of the data
            height, width, n_chunks_height, n_chunks_width = compute_chunks(
                data, min_height=32, max_height=2047, min_width=32, max_width=2047
            )
            for atol in [1e-2, 1e-3, 1e-6, 1e-9]:
                ebcc_filter = EBCC_Filter(
                    base_cr=2, height=height, width=width,
                    data_dim=len(data.shape),
                    residual_opt=("max_error_target", atol),
                )
                zarr_filter = serializer(ebcc_filter.hdf_filter_opts)
                serializer_space.append(AnyNumcodecsArrayBytesCodec(zarr_filter))

    return list(zip(range(len(serializer_space)), serializer_space))


def valid_keepbits_for_bitround(xr_dataarray, step=1):
    dtype = xr_dataarray.dtype
    if np.issubdtype(dtype, np.float64):
        return inclusive_range(1, 52, step)
    elif np.issubdtype(dtype, np.float32):
        return inclusive_range(1, 23, step)
    else:
        raise TypeError(f"Unsupported dtype '{dtype}'. BitRound only supports float32 and float64.")


def valid_digits_for_quantize(xr_dataarray, step=1):
    dtype = xr_dataarray.dtype
    if np.issubdtype(dtype, np.float64):
        return inclusive_range(1, 15, step)
    elif np.issubdtype(dtype, np.float32):
        return inclusive_range(1, 7, step)
    else:
        raise TypeError(f"Unsupported dtype '{dtype}'. Quantize only supports float32 and float64.")


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


def compute_linear_width(
    da: xr.DataArray, *,
    quantile: float = 0.01, skipna: bool = True,
    floor: float | None = None, cap: float | None = None,
    compute: bool = False,
) -> float | xr.DataArray:
    finite = xr.apply_ufunc(np.isfinite, da, dask="parallelized")
    abs_da = xr.apply_ufunc(np.abs, da.where(finite), dask="parallelized")
    lw = abs_da.quantile(quantile, skipna=skipna)
    if "quantile" in lw.dims:
        lw = lw.squeeze("quantile", drop=True)
    if floor is not None or cap is not None:
        lw = lw.clip(
            min=floor if floor is not None else None,
            max=cap   if cap   is not None else None,
        )
    return float(lw.compute()) if compute else lw


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


def evaluate_codec_pipeline(
    sample_np: np.ndarray,
    dims,
    filters,
    compressors,
    serializer,
    chunks,
):
    """
    Measure (compression_ratio, errors, euclidean_distance) for a codec
    pipeline against `sample_np` (a numpy array).

    Uses an in-memory zarr store - no disk I/O, no zip wrapping.
    Safe to call from multiple threads concurrently: each call creates its
    own MemoryStore and does not touch shared state.

    Error norms are accumulated chunk-wise to avoid ever holding a full
    decompressed copy of the sample in memory.
    """
    store = zarr.storage.MemoryStore()

    codec_kwargs = _codec_kwargs(filters, compressors, serializer)

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
    z[...] = sample_np  # triggers full codec pipeline

    # --- compression ratio ---
    info = z.info_complete()
    count_bytes, count_bytes_stored = _info_bytes(info)
    ratio = count_bytes / count_bytes_stored

    # --- chunk-wise error accumulation (never holds a full decompressed copy) ---
    l1_err = 0.0; l2_err_sq = 0.0; linf_err = 0.0
    l1_ori = 0.0; l2_ori_sq = 0.0; linf_ori = 0.0

    for sl in _iter_chunk_slices(sample_np.shape, chunks):
        orig = sample_np[sl]
        decomp = z[sl]
        # promote to float64 for the accumulation only
        err = decomp.astype(np.float64, copy=False) - orig.astype(np.float64, copy=False)
        ori_abs = np.abs(orig, dtype=np.float64) if orig.dtype.kind in "fc" \
                  else np.abs(orig.astype(np.float64))
        err_abs = np.abs(err)

        l1_err     += float(err_abs.sum())
        l2_err_sq  += float((err * err).sum())
        linf_err    = max(linf_err, float(err_abs.max(initial=0.0)))

        l1_ori     += float(ori_abs.sum())
        l2_ori_sq  += float((ori_abs * ori_abs).sum())
        linf_ori    = max(linf_ori, float(ori_abs.max(initial=0.0)))

    l2_err = math.sqrt(l2_err_sq)
    l2_ori = math.sqrt(l2_ori_sq)

    def _safe_div(a, b):
        return float(a) / float(b) if b != 0 else float("inf")

    errors = {
        "Relative_Error_L1":   _safe_div(l1_err,  l1_ori),
        "Relative_Error_L2":   _safe_div(l2_err,  l2_ori),
        "Relative_Error_Linf": _safe_div(linf_err, linf_ori),
    }

    return ratio, errors, l2_err


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
    - Uses `overwrite=True` as the dask-level kwarg (replaces the deprecated
      v2-era `mode='w'` shape).  `chunks`, `shards`, codec kwargs,
      `dimension_names`, `zarr_format=3` are passed through **zarr_array_kwargs
      and forwarded by dask to zarr.create_array.  `mode=` is NOT accepted by
      zarr v3's create_array — it's a storage-level concept, not an array one.
    """
    assert isinstance(da.data, dask.array.Array), \
        "persist_with_codec_pipeline expects a dask-backed xr.DataArray"

    # Auto-size chunks/shards if not provided.
    if inner_chunks is None or shards is None:
        auto_inner, auto_shard = compute_chunk_and_shard_shape(da.shape, da.dtype)
        inner_chunks = inner_chunks or auto_inner
        shards       = shards       or auto_shard

    codec_kwargs = _codec_kwargs(filters, compressors, serializer)

    # Align Dask chunks with shard shape so each write = one shard.
    dask_arr = da.data.rechunk(shards)

    zarr_kwargs = dict(
        zarr_format=3,
        dimension_names=tuple(da.dims),
        chunks=inner_chunks,
        shards=shards,
        **codec_kwargs,
    )

    with Timer("dask.array.to_zarr"):
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
            # Load back with shard-aligned chunks for efficient reads.
            z_dask = dask.array.from_zarr(z, chunks=shards)
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
    da_error = da_compressed - da

    norm_L1_error    = np.abs(da_error).sum()
    norm_L2_error    = np.sqrt((da_error ** 2).sum())
    norm_Linf_error  = np.abs(da_error).max()

    norm_L1_original   = np.abs(da).sum()
    norm_L2_original   = np.sqrt((da ** 2).sum())
    norm_Linf_original = np.abs(da).max()

    computed = dask.compute(
        norm_L1_error, norm_L1_original,
        norm_L2_error, norm_L2_original,
        norm_Linf_error, norm_Linf_original,
    )
    (l1e, l1o, l2e, l2o, linfe, linfo) = computed

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

    euclidean_distance = l2e
    normalized_euclidean_distance = relative_error_L2

    errors = {
        "Relative_Error_L1":   relative_error_L1,
        "Relative_Error_L2":   relative_error_L2,
        "Relative_Error_Linf": relative_error_Linf,
    }
    errors_ = {k: f"{v:.3e}" for k, v in errors.items()}
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
                "OPENBLAS_NUM_THREADS=1 BLOSC_NTHREADS=1 NUMBA_NUM_THREADS=1"
            )
            if abort_if_unsafe:
                click.echo("  Aborting (use --no-oversubscription-check to override).")
        if abort_if_unsafe:
            # Collective abort: all ranks die, not just rank 0.  sys.exit on
            # rank 0 alone would leave siblings hanging at the next collective.
            comm.Abort(1)

    # Pin zarr v3's internal thread pool (best-effort; may not exist on older versions).
    try:
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
            if "EBCC" in item:
                fetch_new_idx = [value for key, value in codec_id_dict.items() if "EBCC" in key][0]
                id_ls.append(fetch_new_idx)
            else:
                return IndexError(f"{item} not in list {list(codec_id_dict.keys())}")
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
        duration = time.perf_counter() - self.start
        with _TIMINGS_LOCK:
            _TIMINGS[self.label].append(duration)


@atexit.register
def print_profile_summary():
    if not _TIMINGS:
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 0:
        return

    print("\n=== Timing Summary ===")
    label_width = max(len(label) for label in _TIMINGS.keys())
    header = f"{'Label':<{label_width}} | {'Calls':>5} | {'Avg (s)':>10} | {'Total (s)':>10}"
    print(header)
    print("-" * len(header))
    for label, durations in sorted(_TIMINGS.items()):
        total = sum(durations); count = len(durations); avg = total / count
        print(f"{label:<{label_width}} | {count:>5} | {avg:>10.6f} | {total:>10.6f}")
    print("=" * len(header))
