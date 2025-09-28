# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys
import os
import shutil
import math
import zipfile
import click
import humanize
from pathlib import Path
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
import atexit
import re

# numcodecs-wasm filters
from numcodecs_wasm_asinh import Asinh
from numcodecs_wasm_fixed_offset_scale import FixedOffsetScale
# numcodecs-wasm serializers
from numcodecs_wasm_zfp import Zfp

os.environ["EBCC_LOG_LEVEL"] = "4"  # ERROR (suppress WARN and below)

def get_indexes(arr, indices):
    codec_to_id = []
    for ind in indices:
        codec_to_id.append(ind[1:-1].split(', ', 1))
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
                return IndexError(f'{item} not in list {list(codec_id_dict.keys())}')
    return np.asarray(id_ls)

def open_dataset(dataset_file: str, field_to_compress: str | None = None, field_percentage_to_compress: float | None = None, rank: int = 0):
    dataset_filepath = Path(dataset_file)    
    if dataset_filepath.suffix == ".nc":
        ds = xr.open_dataset(dataset_file, chunks="auto")  # auto for Dask backend
    elif dataset_filepath.suffix == ".grib":
        ds = xr.open_dataset(dataset_file, chunks="auto", engine="cfgrib", backend_kwargs={"indexpath": ""})
    else:
        if rank == 0:
            click.echo(f"Unsupported file format: {dataset_filepath.suffix}. Only .nc and .grib are supported.")
            click.echo("Aborting...")
        sys.exit(1)

    if field_to_compress is not None and field_to_compress not in ds.data_vars:
        if rank == 0:
            click.echo(f"Field {field_to_compress} not found in NetCDF file.")
            click.echo(f"Available fields in the dataset: {list(ds.data_vars.keys())}.")
            click.echo("Aborting...")
        sys.exit(1)

    if rank == 0:
        click.echo(f"dataset_file.nbytes = {humanize.naturalsize(ds.nbytes, binary=True)}")
        if field_to_compress is not None:
            nbytes = ds[field_to_compress].nbytes * (field_percentage_to_compress / 100) if field_percentage_to_compress else ds[field_to_compress].nbytes
            click.echo(f"field_to_compress.nbytes = {humanize.naturalsize(nbytes, binary=True)}")

    return ds


def is_lat_lon(da):

    lat_pattern = r'lat'
    lon_pattern = r'lon'

    dims = da.dims

    if len(dims) == 2 and re.search(lat_pattern, dims[0]) and re.search(lon_pattern, dims[1]):
        return True

    return False


def open_zarr_zipstore(zarr_zipstore_file: str):
    store = zarr.storage.ZipStore(zarr_zipstore_file, read_only=True)
    return zarr.open_group(store, mode='r'), store


def open_zarr_memstore():
    store = zarr.storage.MemoryStore()
    return store


def compress_with_zarr(data, dataset_file, field_to_compress, where_to_write, filters, compressors, serializer, verbose=True, rank=0):
    assert isinstance(data.data, dask.array.Array)

    basename = os.path.basename(dataset_file)
    zarr_file = os.path.join(where_to_write, basename)
    zarr_file = f"{zarr_file}.=.field_{field_to_compress}.=.rank_{rank}.zarr.zip"

    with Timer("zarr.create_array"):
        store = zarr.storage.ZipStore(zarr_file, mode='w')
        zarr.create_array(
            store=store,
            name=field_to_compress,
            data=data,
            chunks="auto",
            filters=filters,
            compressors=compressors,
            serializer=serializer,
            )
        store.close()

    group, store = open_zarr_zipstore(zarr_file)
    z = group[field_to_compress]
    z_dask = dask.array.from_zarr(z)

    info_array = z.info_complete()
    compression_ratio = info_array._count_bytes / info_array._count_bytes_stored
    if verbose and rank == 0:
        click.echo(80* "-")
        click.echo(info_array)

    with Timer("compute_errors_distances"):
        pprint_, errors, euclidean_distance, normalized_euclidean_distance = compute_errors_distances(z_dask, data.data)
    if verbose and rank == 0:
        click.echo(80* "-")
        click.echo(pprint_)
        click.echo(80* "-")
        click.echo(f"Euclidean Distance: {euclidean_distance}")
        click.echo(80* "-")

    store.close()

    return compression_ratio, errors, euclidean_distance


def compute_errors_distances(da_compressed, da):
    da_error = da_compressed - da

    # These are still lazy Dask computations
    norm_L1_error = np.abs(da_error).sum()
    norm_L2_error = np.sqrt((da_error**2).sum())
    norm_Linf_error = np.abs(da_error).max()

    norm_L1_original = np.abs(da).sum()
    norm_L2_original = np.sqrt((da**2).sum())
    norm_Linf_original = np.abs(da).max()

    # Group all into one call to compute for efficiency
    computed = dask.compute(
        norm_L1_error,
        norm_L1_original,
        norm_L2_error,
        norm_L2_original,
        norm_Linf_error,
        norm_Linf_original,
    )

    (
        norm_L1_error_val,
        norm_L1_original_val,
        norm_L2_error_val,
        norm_L2_original_val,
        norm_Linf_error_val,
        norm_Linf_original_val,
    ) = computed

    relative_error_L1 = norm_L1_error_val / norm_L1_original_val
    relative_error_L2 = norm_L2_error_val / norm_L2_original_val
    relative_error_Linf = norm_Linf_error_val / norm_Linf_original_val

    euclidean_distance = norm_L2_error_val
    normalized_euclidean_distance = relative_error_L2

    errors = {
        "Relative_Error_L1": relative_error_L1,
        "Relative_Error_L2": relative_error_L2,
        "Relative_Error_Linf": relative_error_Linf,
    }

    errors_ = {k: f"{v:.3e}" for k, v in errors.items()}
    return "\n".join(f"{k:20s}: {v}" for k, v in errors_.items()), errors, euclidean_distance, normalized_euclidean_distance


def compute_chunks(data, min_height=0, max_height=None, min_width=0, max_width=None):
    lat_dim = data.shape[0]
    lon_dim = data.shape[1]
    height = lat_dim
    width = lon_dim

    if max_height is None: max_height = lat_dim
    if max_width is None: max_width = lat_dim

    keep_searching_height = True
    keep_searching_width = True

    for n in [2, 3, 5]:
        for m in range(10):
            d = n * (m+1)
            for p in range(7, -1, -1):

                if keep_searching_height or keep_searching_width:

                    if keep_searching_height:
                        n_chunks_height = d**p

                        if height % n_chunks_height == 0:
                            new_height = height / n_chunks_height

                            if (new_height >= min_height) and (new_height <= max_height):
                                height = new_height
                                keep_searching_height = False

                    if keep_searching_width and p > 0:
                        n_chunks_width = d**(p-1)

                        if width % n_chunks_width == 0:
                            new_width = width / n_chunks_width

                            if (new_width >= min_width) and (new_width <= max_width):
                                width = new_width
                                keep_searching_width = False

                else:
                    return (height, width, n_chunks_height, n_chunks_width)


def compressor_space(da, with_lossy=True, with_numcodecs_wasm=True, with_ebcc=True, compressor_class="all"):
    # https://numcodecs.readthedocs.io/en/stable/zarr3.html#compressors-bytes-to-bytes-codecs

    compressor_space = []

    _COMPRESSORS = [numcodecs.zarr3.Blosc, numcodecs.zarr3.LZ4, numcodecs.zarr3.Zstd, numcodecs.zarr3.Zlib, numcodecs.zarr3.GZip, numcodecs.zarr3.BZ2, numcodecs.zarr3.LZMA]
    _COMPRESSOR_MAP = {cls.__name__.lower(): cls for cls in _COMPRESSORS}

    if compressor_class.lower() == "all":
        pass  # use all compressors
    elif compressor_class.lower() in _COMPRESSOR_MAP:
        _COMPRESSORS = [_COMPRESSOR_MAP[compressor_class.lower()]]
    elif compressor_class.lower() == "none":
        _COMPRESSORS = []
        compressor_space.append(None)
    else:
        pass  # use all compressors

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
    # https://numcodecs.readthedocs.io/en/stable/zarr3.html#filters-array-to-array-codecs
    # https://numcodecs-wasm.readthedocs.io/en/latest/

    filter_space = []

    _FILTERS = [numcodecs.zarr3.Delta]
    if with_lossy:
        _FILTERS += [numcodecs.zarr3.BitRound, numcodecs.zarr3.Quantize]
    if with_numcodecs_wasm:
        if with_lossy:
            _FILTERS.append(Asinh)
        _FILTERS.append(FixedOffsetScale)
    if da.dtype.kind == 'i':
        _FILTERS = [numcodecs.zarr3.Delta]

    _FILTER_MAP = {cls.__name__.lower(): cls for cls in _FILTERS}

    if filter_class.lower() == "all":
        pass  # use all filters
    elif filter_class.lower() in _FILTER_MAP:
        _FILTERS = [_FILTER_MAP[filter_class.lower()]]
    elif filter_class.lower() == "none":
        _FILTERS = []
        filter_space.append(None)
    else:
        pass  # use all filters

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
            filter_space.append(AnyNumcodecsArrayArrayCodec(filt(linear_width=compute_linear_width(da, quantile=0.01, compute=True))))
        elif filt == FixedOffsetScale:
            # Compute required stats ONCE (global reductions; memory-light but do trigger a compute)
            mean_val, std_val, min_val, max_val = dask.compute(
                da.mean(skipna=True),
                da.std(skipna=True),
                da.min(skipna=True),
                da.max(skipna=True),
            )

            # Helper to validate finite, nonzero scale (avoid divide-by-zero / NaNs)
            def _safe_scale(x, min_eps=1e-12):
                if not np.isfinite(x):
                    return None
                if abs(x) < min_eps:
                    return None
                return float(x)

            # Normalize: (x - mean) / std
            std_safe = _safe_scale(std_val)
            if np.isfinite(mean_val) and std_safe is not None:
                filter_space.append(AnyNumcodecsArrayArrayCodec(filt(offset=float(mean_val), scale=std_safe)))

            # Standardize to [0,1]-like: (x - min) / (max - min)
            rng = max_val - min_val
            rng_safe = _safe_scale(rng)
            if np.isfinite(min_val) and rng_safe is not None:
                filter_space.append(AnyNumcodecsArrayArrayCodec(filt(offset=float(min_val), scale=rng_safe)))

    return list(zip(range(len(filter_space)), filter_space))


def serializer_space(da, with_lossy=True, with_numcodecs_wasm=True, with_ebcc=True, serializer_class="all"):
    # https://numcodecs.readthedocs.io/en/stable/zarr3.html#serializers-array-to-bytes-codecs
    # https://numcodecs-wasm.readthedocs.io/en/latest/
    rank = MPI.COMM_WORLD.Get_rank()
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
        pass  # use all serializers
    elif serializer_class.lower() in _SERIALIZER_MAP:
        _SERIALIZERS = [_SERIALIZER_MAP[serializer_class.lower()]]
    elif serializer_class.lower() == "none":
        _SERIALIZERS = []
        serializer_space.append(None)
    else:
        pass  # use all serializers

    for serializer in _SERIALIZERS:
        if serializer == numcodecs.zarr3.PCodec:
            for level in [0, 4, 8, 12]:
                for delta_encoding_order in [0, 3, 7]:
                    serializer_space.append(serializer(
                            level=level,
                            mode_spec="auto",
                            delta_spec="auto",
                            delta_encoding_order=delta_encoding_order
                        )
                    )
        elif serializer in (numcodecs.zarr3.ZFPY, Zfp):
            # https://github.com/LLNL/zfp/tree/develop/python
            # https://github.com/LLNL/zfp/blob/develop/tests/python/test_numpy.py
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
            # https://github.com/spcl/ebcc
            data = da.squeeze()  # TODO: add more checks on the shape of the data

            height, width, n_chunks_height, n_chunks_width = compute_chunks( data,
                                                                             min_height=32,
                                                                             max_height=2047,
                                                                             min_width=32,
                                                                             max_width=2047 )

            if rank == 0:
                click.echo(f"Using (lat_chunks * lon_chunks) = ({n_chunks_height} * {n_chunks_width}) = {n_chunks_height*n_chunks_width} chunks for EBCC serializers.")

            for atol in [1e-2, 1e-3, 1e-6, 1e-9]:
                ebcc_filter = EBCC_Filter(
                        base_cr=2,
                        height=height,
                        width=width,
                        data_dim=len(data.shape),
                        residual_opt=("max_error_target", atol)
                    )
                zarr_filter = serializer(ebcc_filter.hdf_filter_opts)
                serializer_space.append(AnyNumcodecsArrayBytesCodec(zarr_filter))

    return list(zip(range(len(serializer_space)), serializer_space))


def valid_keepbits_for_bitround(xr_dataarray, step=1):
    dtype = xr_dataarray.dtype
    if np.issubdtype(dtype, np.float64):
        return inclusive_range(1, 52, step)  # float64 mantissa is 52 bits
    elif np.issubdtype(dtype, np.float32):
        return inclusive_range(1, 23, step)  # float32 mantissa is 23 bits
    else:
        raise TypeError(f"Unsupported dtype '{dtype}'. BitRound only supports float32 and float64.")


def valid_digits_for_quantize(xr_dataarray, step=1):
    dtype = xr_dataarray.dtype
    if np.issubdtype(dtype, np.float64):
        return inclusive_range(1, 15, step)  # ~15–16 significant digits for float64
    elif np.issubdtype(dtype, np.float32):
        return inclusive_range(1, 7, step)   # ~7 significant digits for float32
    else:
        raise TypeError(f"Unsupported dtype '{dtype}'. Quantize only supports float32 and float64.")


def compute_fixed_precision_param(param: int) -> int:
    # https://github.com/LLNL/zfp/tree/develop/tests/python
    return 1 << (param + 3)

def compute_fixed_rate_param(param: int) -> int:
    # https://github.com/LLNL/zfp/tree/develop/tests/python
    return 1 << (param + 3)

def compute_fixed_accuracy_param(param: int) -> float:
    # https://github.com/LLNL/zfp/tree/develop/tests/python
    return math.ldexp(1.0, -(1 << param))


def inclusive_range(start, end, step=1):
    if step == 0:
        raise ValueError("step must not be zero")

    values = []
    i = start
    if step > 0:
        while i <= end:
            values.append(i)
            i += step
        if values[-1] != end:
            values.append(end)
    else:
        while i >= end:
            values.append(i)
            i += step
        if values[-1] != end:
            values.append(end)

    return values


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


def slice_array(arr: pd.array, indices_ls: list) -> np.ndarray:
    arr_ls = []
    for ind in indices_ls:
        arr_ls.append(arr[[ind]])

    sliced_arr = np.hstack(
        tuple(arr_ls)
    )
    return sliced_arr


def unzip_file(zip_path: str, extract_to: str = None):
    if extract_to is None:
        extract_to = os.path.splitext(zip_path)[0]  # default: same name as zip

    # Remove the extract_to path if it exists
    if os.path.exists(extract_to):
        if os.path.isfile(extract_to):
            os.remove(extract_to)
        else:
            shutil.rmtree(extract_to)

    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    return extract_to


def copy_folder_contents(src_folder: str, dst_folder: str):
    os.makedirs(dst_folder, exist_ok=True)

    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dst_path = os.path.join(dst_folder, item)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


def progress_bar(i, total_configs, print_every=100, bar_width=40):
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        return

    percent = (i + 1) / total_configs
    filled = int(bar_width * percent)
    bar = "*" * filled + "-" * (bar_width - filled)
    if int(i + 1) % print_every == 0 or (i + 1) == total_configs:
        click.echo(f"[Rank {rank}] Progress: |{bar}| {percent*100:6.2f}% ({i+1}/{total_configs}) [{total_configs} loops per MPI task]")


# Global registry of all timings
_TIMINGS = defaultdict(list)

class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        duration = time.perf_counter() - self.start
        _TIMINGS[self.label].append(duration)

@atexit.register
def print_profile_summary():
    if not _TIMINGS:
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank != 0:
        return # Only the root process prints the summary

    print("\n=== compress_with_zarr Timing Summary ===")

    # Determine max label width for formatting
    label_width = max(len(label) for label in _TIMINGS.keys())

    header = (
        f"{'Label':<{label_width}} | {'Calls':>5} | {'Avg (s)':>10} | {'Total (s)':>10}"
    )
    print(header)
    print("-" * len(header))

    for label, durations in sorted(_TIMINGS.items()):
        total = sum(durations)
        count = len(durations)
        avg = total / count
        print(f"{label:<{label_width}} | {count:>5} | {avg:>10.6f} | {total:>10.6f}")

    print("=" * len(header))


def compute_linear_width(
    da: xr.DataArray,
    *,
    quantile: float = 0.01,
    skipna: bool = True,
    floor: float | None = None,
    cap: float | None = None,
    compute: bool = False,
) -> float | xr.DataArray:
    """Lazy, Dask-enabled estimate of Asinh.linear_width via small-quantile(|da|)."""
    # mask to finite values (lazy)
    finite = xr.apply_ufunc(np.isfinite, da, dask="parallelized")
    abs_da = xr.apply_ufunc(np.abs, da.where(finite), dask="parallelized")

    # small quantile of |da| (lazy)
    lw = abs_da.quantile(quantile, skipna=skipna)

    # drop the 'quantile' coord that xarray adds
    if "quantile" in lw.dims:
        lw = lw.squeeze("quantile", drop=True)

    # Dask-safe bounds without apply_ufunc
    if floor is not None or cap is not None:
        lw = lw.clip(min=floor if floor is not None else None,
                     max=cap if cap is not None else None)

    return float(lw.compute()) if compute else lw
