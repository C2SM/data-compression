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
import traceback
import asyncio
from collections import OrderedDict
from collections.abc import Sequence
import functools
import inspect
import zipfile
import click
import humanize
import numpy as np
import dask
import pandas as pd
import xarray as xr
import yaml
import pywt
import zarr
from zarr_any_numcodecs import AnyNumcodecsArrayArrayCodec, AnyNumcodecsArrayBytesCodec, AnyNumcodecsBytesBytesCodec
import numcodecs
import numcodecs.zarr3
import zfpy
from ebcc.filter_wrapper import EBCC_Filter
from ebcc.zarr_filter import EBCCZarrFilter
from mpi4py import MPI
import time
from collections import defaultdict
import atexit

# numcodecs-wasm filters
from numcodecs_wasm_asinh import Asinh
from numcodecs_wasm_bit_round import BitRound
from numcodecs_wasm_fixed_offset_scale import FixedOffsetScale
from numcodecs_wasm_identity import Identity
from numcodecs_wasm_linear_quantize import LinearQuantize
from numcodecs_wasm_log import Log
from numcodecs_wasm_round import Round
from numcodecs_wasm_swizzle_reshape import SwizzleReshape
from numcodecs_wasm_uniform_noise import UniformNoise
# numcodecs-wasm compressors
from numcodecs_wasm_zlib import Zlib
from numcodecs_wasm_zstd import Zstd
# numcodecs-wasm serializers
from numcodecs_wasm_pco import Pco
from numcodecs_wasm_sperr import Sperr
from numcodecs_wasm_sz3 import Sz3
from numcodecs_wasm_zfp import Zfp


_WITH_NUMCODECS_WASM = os.getenv("WITH_NUMCODECS_WASM", "true").lower() in ("1", "true", "yes")
_WITH_EBCC = os.getenv("WITH_EBCC", "true").lower() in ("1", "true", "yes")


def open_netcdf(netcdf_file: str, field_to_compress: str | None = None, field_percentage_to_compress: float | None = None, rank: int = 0):
    ds = xr.open_dataset(netcdf_file, chunks="auto")

    if field_to_compress is not None and field_to_compress not in ds.data_vars:
        if rank == 0:
            click.echo(f"Field {field_to_compress} not found in NetCDF file.")
            click.echo(f"Available fields in the dataset: {list(ds.data_vars.keys())}.")
            click.echo("Aborting...")
        sys.exit(1)

    if rank == 0:
        click.echo(f"netcdf_file.nbytes = {humanize.naturalsize(ds.nbytes, binary=True)}")
        if field_to_compress is not None:
            nbytes = ds[field_to_compress].nbytes * (field_percentage_to_compress / 100) if field_percentage_to_compress else ds[field_to_compress].nbytes
            click.echo(f"field_to_compress.nbytes = {humanize.naturalsize(nbytes, binary=True)}")

    return ds


def open_zarr_zipstore(zarr_zipstore_file: str):
    store = zarr.storage.ZipStore(zarr_zipstore_file, read_only=True)
    return zarr.open_group(store, mode='r'), store


def compress_with_zarr(data, netcdf_file, field_to_compress, where_to_write, filters, compressors, serializer='auto', verbose=True, rank=0):
    assert isinstance(data.data, dask.array.Array)
    
    basename = os.path.basename(netcdf_file)
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

    with Timer("calc_dwt_dist"):
        # TODO: make it Dask compatible, otherwise use the euclidean which we already compute
        dwt_dist = euclidean_distance #calc_dwt_dist(z[:], data)
    if verbose and rank == 0:
        click.echo(f"DWT Distance: {dwt_dist}")
        click.echo(80* "-")

    store.close()

    return compression_ratio, errors, dwt_dist


def ordered_yaml_loader():
    class OrderedLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return OrderedDict(loader.construct_pairs(node))

    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)

    return OrderedLoader


def get_filter_parameters(parameters_file: str, filter_name: str):
    with open(parameters_file, "r") as f:
        params = yaml.load(f, Loader=ordered_yaml_loader())

    try:
        filter_config = params[filter_name]["params"]
        return tuple(filter_config.values())

    except Exception:
        click.echo("An unexpected error occurred:", err=True)
        traceback.print_exc(file=sys.stderr)


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


def calc_dwt_dist(input_1, input_2, n_levels=4, wavelet="haar"):
    dwt_data_1 = pywt.wavedec(input_1, wavelet=wavelet, level=n_levels)
    dwt_data_2 = pywt.wavedec(input_2, wavelet=wavelet, level=n_levels)
    distances = [np.linalg.norm(c1 - c2) for c1, c2 in zip(dwt_data_1, dwt_data_2)]
    dwt_distance = np.sqrt(sum(d**2 for d in distances))
    return dwt_distance


def compressor_space(da):
    # https://numcodecs.readthedocs.io/en/stable/zarr3.html#compressors-bytes-to-bytes-codecs

    # TODO: take care of integer data types
    compressor_space = []

    _COMPRESSORS = [numcodecs.zarr3.Blosc, numcodecs.zarr3.LZ4, numcodecs.zarr3.Zstd, numcodecs.zarr3.Zlib, numcodecs.zarr3.GZip, numcodecs.zarr3.BZ2, numcodecs.zarr3.LZMA]
    for compressor in _COMPRESSORS:
        if compressor == numcodecs.zarr3.Blosc:
            for cname in numcodecs.blosc.list_compressors():
                for clevel in inclusive_range(1,9,4):
                    for shuffle in inclusive_range(0,2):
                        compressor_space.append(numcodecs.zarr3.Blosc(cname=cname, clevel=clevel, shuffle=shuffle))
        elif compressor == numcodecs.zarr3.LZ4:
            # The larger the acceleration value, the faster the algorithm, but also the lesser the compression
            for acceleration in inclusive_range(0, 16, 6):
                compressor_space.append(numcodecs.zarr3.LZ4(acceleration=acceleration))
        elif compressor == numcodecs.zarr3.Zstd:
            for level in inclusive_range(-7, 22, 10):
                compressor_space.append(numcodecs.zarr3.Zstd(level=level))
        elif compressor == numcodecs.zarr3.Zlib:
            for level in inclusive_range(1,9,4):
                compressor_space.append(numcodecs.zarr3.Zlib(level=level))
        elif compressor == numcodecs.zarr3.GZip:
            for level in inclusive_range(1,9,4):
                compressor_space.append(numcodecs.zarr3.GZip(level=level))
        elif compressor == numcodecs.zarr3.BZ2:
            for level in inclusive_range(1,9,4):
                compressor_space.append(numcodecs.zarr3.BZ2(level=level))
        elif compressor == numcodecs.zarr3.LZMA:
            # https://docs.python.org/3/library/lzma.html
            for preset in inclusive_range(1,9,4):
                compressor_space.append(numcodecs.zarr3.LZMA(preset=preset))

    return list(zip(range(len(compressor_space)), compressor_space))


def filter_space(da):
    # https://numcodecs.readthedocs.io/en/stable/zarr3.html#filters-array-to-array-codecs
    # https://numcodecs-wasm.readthedocs.io/en/latest/
    
    # TODO: take care of integer data types
    filter_space = []
    
    _FILTERS = [numcodecs.zarr3.Delta, numcodecs.zarr3.BitRound, numcodecs.zarr3.Quantize]
    if _WITH_NUMCODECS_WASM:
        _FILTERS += [Asinh, FixedOffsetScale, Log, UniformNoise]
    if da.dtype.kind == 'i':
        _FILTERS = [numcodecs.zarr3.Delta]
    base_scale = 10 ** np.floor(np.log10(np.abs(da).max().compute().item()))
    for filter in _FILTERS:
        if filter == numcodecs.zarr3.Delta:
            filter_space.append(numcodecs.zarr3.Delta(dtype=str(da.dtype)))
        elif filter == numcodecs.zarr3.BitRound:
            # If keepbits is equal to the maximum allowed for the data type, this is equivalent to no transform.
            for keepbits in valid_keepbits_for_bitround(da, step=9):
                filter_space.append(numcodecs.zarr3.BitRound(keepbits=keepbits))
        elif filter == numcodecs.zarr3.Quantize:
            for digits in valid_keepbits_for_bitround(da, step=9):
                filter_space.append(numcodecs.zarr3.Quantize(digits=digits, dtype=str(da.dtype)))
        elif filter == Asinh:
            for linear_width in [base_scale/10, base_scale, base_scale*10]:
                filter_space.append(AnyNumcodecsArrayArrayCodec(Asinh(linear_width=linear_width)))
        elif filter == FixedOffsetScale:
            # Setting o=mean(x) and s=std(x) normalizes that data
            filter_space.append(AnyNumcodecsArrayArrayCodec(FixedOffsetScale(offset=da.mean(skipna=True).compute().item(), scale=da.std(skipna=True).compute().item())))
            # Setting o=min(x) and s=max(x)−min(x)standardizes the data
            filter_space.append(AnyNumcodecsArrayArrayCodec(FixedOffsetScale(offset=da.min(skipna=True).compute().item(), scale=da.max(skipna=True).compute().item()-da.min(skipna=True).compute().item())))
        elif filter == Log:
            if bool((da > 0).all()):
                filter_space.append(AnyNumcodecsArrayArrayCodec(Log()))
        elif filter == UniformNoise:
            for seed in [0]:
                for scale in [base_scale/10, base_scale, base_scale*10]:
                    filter_space.append(AnyNumcodecsArrayArrayCodec(UniformNoise(scale=scale, seed=seed)))

    return list(zip(range(len(filter_space)), filter_space))


def serializer_space(da):
    # https://numcodecs.readthedocs.io/en/stable/zarr3.html#serializers-array-to-bytes-codecs
    # https://numcodecs-wasm.readthedocs.io/en/latest/
    
    # TODO: take care of integer data types
    serializer_space = []
    
    _SERIALIZERS = [numcodecs.zarr3.PCodec] # numcodecs.zarr3.ZFPY
    if _WITH_EBCC:
        _SERIALIZERS += [EBCCZarrFilter]
    if _WITH_NUMCODECS_WASM:
        _SERIALIZERS += [Sperr, Sz3]
    if da.dtype.kind == 'i':
        _SERIALIZERS = [numcodecs.zarr3.PCodec, numcodecs.zarr3.ZFPY, EBCCZarrFilter, Sz3]
    for serializer in _SERIALIZERS:
        if serializer == numcodecs.zarr3.PCodec:
            # https://github.com/pcodec/pcodec
            # PCodec supports only the following numerical dtypes: uint16, uint32, uint64, int16, int32, int64, float16, float32, and float64.
            for level in inclusive_range(0, 12, 4):  # where 12 take the longest and compresses the most
                for mode_spec in ["auto", "classic"]:
                    for delta_spec in ["auto", "none", "try_consecutive", "try_lookback"]:
                        for delta_encoding_order in inclusive_range(0,7,4):
                            serializer_space.append(numcodecs.zarr3.PCodec(
                                    level=level,
                                    mode_spec=mode_spec,
                                    delta_spec=delta_spec,
                                    delta_encoding_order=delta_encoding_order if delta_spec in ["try_consecutive", "auto"] else None
                                )
                            )
        elif serializer == numcodecs.zarr3.ZFPY:
            # https://github.com/LLNL/zfp/tree/develop/python
            # https://github.com/LLNL/zfp/blob/develop/tests/python/test_numpy.py
            for mode in [zfpy.mode_fixed_accuracy,
                         zfpy.mode_fixed_precision,
                         zfpy.mode_fixed_rate,]:
                for compress_param_num in range(3):
                    if mode == zfpy.mode_fixed_accuracy:
                        serializer_space.append(numcodecs.zarr3.ZFPY(
                            mode=mode, tolerance=compute_fixed_accuracy_param(compress_param_num)
                        ))
                    elif mode == zfpy.mode_fixed_precision:
                        serializer_space.append(numcodecs.zarr3.ZFPY(
                            mode=mode, precision=compute_fixed_precision_param(compress_param_num)
                        ))
                    elif mode == zfpy.mode_fixed_rate:
                        serializer_space.append(numcodecs.zarr3.ZFPY(
                            mode=mode, rate=compute_fixed_rate_param(compress_param_num)
                        ))
        elif serializer == EBCCZarrFilter:
            # https://github.com/spcl/ebcc
            data = da.squeeze()  # TODO: add more checks on the shape of the data
            for atol in [1e-2, 1e-3, 1e-6, 1e-9]:
                ebcc_filter = EBCC_Filter(
                        base_cr=2, 
                        height=data.shape[0],
                        width=data.shape[1],
                        residual_opt=("max_error_target", atol)
                    )
                zarr_filter = EBCCZarrFilter(ebcc_filter.hdf_filter_opts)
                serializer_space.append(AnyNumcodecsArrayBytesCodec(zarr_filter))
        elif serializer == Sperr:
            # https://github.com/juntyr/numcodecs-rs/blob/main/codecs/sperr/tests/config.rs
            for mode in ["bpp", "psnr", "pwe"]:
                serializer_space.append(AnyNumcodecsArrayBytesCodec(Sperr(mode=mode, bpp=1.0, psnr=42.0, pwe=0.1)))
        elif serializer == Sz3:
            # https://github.com/juntyr/numcodecs-rs/blob/main/codecs/sz3/tests/config.rs
            abs_err = [1.0, 1e-1, 1e-2]
            rel_err = [1.0, 1e-1, 1e-2]
            l2      = [1.0, 1e-1]
            for eb_mode in ["abs-and-rel", "abs-or-rel", "abs", "rel", "psnr", "l2"]:
                for predictor in ["cubic-interpolation-lorenzo", "linear-interpolation", "lorenzo-regression"]:
                    if eb_mode in ["abs-and-rel", "abs-or-rel"]:
                        for eb_abs in abs_err:
                            for eb_rel in rel_err:
                                serializer_space.append(AnyNumcodecsArrayBytesCodec(Sz3(eb_mode=eb_mode, eb_abs=eb_abs, eb_rel=eb_rel, predictor=predictor)))
                    elif eb_mode == "abs":
                        for eb_abs in abs_err:
                            serializer_space.append(AnyNumcodecsArrayBytesCodec(Sz3(eb_mode=eb_mode, eb_abs=eb_abs, predictor=predictor)))
                    elif eb_mode == "rel":
                        for eb_rel in rel_err:
                            serializer_space.append(AnyNumcodecsArrayBytesCodec(Sz3(eb_mode=eb_mode, eb_rel=eb_rel, predictor=predictor)))
                    elif eb_mode == "psnr":
                        serializer_space.append(AnyNumcodecsArrayBytesCodec(Sz3(eb_mode=eb_mode, eb_psnr=1.0, predictor=predictor)))
                    elif eb_mode == "l2":
                        for eb_l2 in l2:
                            serializer_space.append(AnyNumcodecsArrayBytesCodec(Sz3(eb_mode=eb_mode, eb_l2=eb_l2, predictor=predictor)))

    return list(zip(range(len(serializer_space)), serializer_space))


def valid_keepbits_for_bitround(xr_dataarray, step=1):
    dtype = xr_dataarray.dtype
    if np.issubdtype(dtype, np.float64):
        return inclusive_range(1, 52, step)  # float64 mantissa is 52 bits
    elif np.issubdtype(dtype, np.float32):
        return inclusive_range(1, 23, step)  # float32 mantissa is 23 bits
    else:
        raise TypeError(f"Unsupported dtype '{dtype}'. BitRound only supports float32 and float64.")


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


def progress_bar(rank, i, total_configs, print_every=100, bar_width=40):
    if rank != 0:
        return
    percent = (i + 1) / total_configs
    filled = int(bar_width * percent)
    bar = "*" * filled + "-" * (bar_width - filled)
    if int(i + 1) % print_every == 0 or (i + 1) == total_configs:
        click.echo(f"[Rank {rank}] Progress: |{bar}| {percent*100:6.2f}% ({i+1}/{total_configs})")


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
