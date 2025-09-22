# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import io
import traceback
import click
from tqdm import tqdm
from pathlib import Path
import shutil
import itertools
import numcodecs
import numcodecs.zarr3
import xarray as xr
from dc_toolkit import utils
from zarr_any_numcodecs import AnyNumcodecsArrayBytesCodec
from ebcc.zarr_filter import EBCCZarrFilter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpi4py import MPI
import dask
import dask.array
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess

import warnings
warnings.filterwarnings(
    "ignore",
    message="Numcodecs codecs are not in the Zarr version 3 specification and may not be supported by other zarr implementations",
    category=UserWarning,
    module="numcodecs.zarr3"
)
warnings.filterwarnings("ignore", message="overflow encountered in square")


@click.group()
def cli():
    pass


@cli.command("evaluate_combos")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("where_to_write", type=click.Path(dir_okay=True, file_okay=False, exists=False))
@click.option("--field-to-compress", default=None, help="Field to compress [if not given, all fields will be compressed].")
@click.option("--field-percentage-to-compress", default=None, callback=utils.validate_percentage, help="Compress a percentage of the field [1-99%]. If not given, the whole field will be compressed.")
@click.option("--override-existing-l1-error", type=float, default=None, help="Override the existing L1 error threshold from the lookup table. If provided, this value will be used instead of the spreadsheet value.")
@click.option("--compressor-class", default="all", help="Compressor class to use (case insensitive), i.e. specified one instead of the full list `all` [`none` skips all compressors].")
@click.option("--filter-class", default="all", help="Filter class to use (case insensitive), i.e. specified one instead of the full list `all` [`none` skips all filters].")
@click.option("--serializer-class", default="all", help="Serializer class to use (case insensitive), i.e. specified one instead of the full list `all` [`none` skips all serializers].")
@click.option("--with-lossy/--without-lossy", default=True, show_default=True, help="Enable or disable lossy compressors/filters/serializers.")
@click.option("--with-numcodecs-wasm/--without-numcodecs-wasm", default=True, show_default=True, help="Enable or disable Numcodecs-wasm codecs.")
@click.option("--with-ebcc/--without-ebcc", default=True, show_default=True, help="Enable or disable EBCC serializer.")
def evaluate_combos(dataset_file: str, where_to_write: str, 
                    field_to_compress: str | None = None, field_percentage_to_compress: str | None = None, override_existing_l1_error: float | None = None,
                    compressor_class: str = "all", filter_class: str = "all", serializer_class: str = "all",
                    with_lossy: bool = True, with_numcodecs_wasm: bool = True, with_ebcc: bool = True):
    """
    Loop over combinations of compressors, filters, and serializers to find the optimal configuration for compressing a given field in a dataset file.

    List of compressors : Blosc, LZ4, Zstd, Zlib, GZip, BZ2, LZMA \n
    List of filters     : Delta, BitRound, Quantize, Asinh, FixedOffsetScale \n
    List of serializers : PCodec, ZFPY, EBCCZarrFilter, Zfp

    \b
    Args:
        dataset_file (str): Path to the input dataset file.
        where_to_write (str): Directory where the output files will be written.
        field_to_compress: --field-to-compress
        field_percentage_to_compress: --field-percentage-to-compress
        override_existing_l1_error: --override-existing-l1-error
        compressor_class: --compressor-class
        filter_class: --filter-class
        serializer_class: --serializer-class
        with_lossy: --with-lossy/--without-lossy
        with_numcodecs_wasm: --with-numcodecs-wasm/--without-numcodecs-wasm
        with_ebcc: --with-ebcc/--without-ebcc
    """
    dask.config.set(scheduler="single-threaded")
    dask.config.set(array__chunk_size="512MiB")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    os.makedirs(where_to_write, exist_ok=True) 

    if rank == 0:
        try:
            # Lookup table for valid thresholds
            # https://docs.google.com/spreadsheets/d/1lHcX-HE2WpVCOeKyDvM4iFqjlWvkd14lJlA-CUoCxMM
            sheet_id = "1lHcX-HE2WpVCOeKyDvM4iFqjlWvkd14lJlA-CUoCxMM"
            sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            thresholds = pd.read_csv(sheet_url)
        except Exception as e:
            print(f"[Rank 0] Failed to fetch thresholds: {e}")
            sys.exit(1)
        # Convert DataFrame to bytes for broadcasting
        buffer = io.BytesIO()
        thresholds.to_parquet(buffer, index=False)
        data_bytes = buffer.getvalue()
    else:
        data_bytes = None

    data_bytes = comm.bcast(data_bytes, root=0)

    if rank != 0:
        buffer = io.BytesIO(data_bytes)
        thresholds = pd.read_parquet(buffer)

    # This is opened by all MPI processes -lazy evaluation with Dask backend-
    ds = utils.open_dataset(dataset_file, field_to_compress, field_percentage_to_compress, rank=rank)

    for var in ds.data_vars:
        if field_to_compress is not None and field_to_compress != var:
            continue
        da = ds[var]

        if override_existing_l1_error is None:
            lookup = var
            threshold_row = thresholds[thresholds["Short Name"] == lookup]
            matching_units = threshold_row.iloc[0]["Unit"] == da.attrs.get("units", None) if not threshold_row.empty else None
            existing_l1_error = threshold_row.iloc[0]["Existing L1 error"] if not threshold_row.empty and matching_units else None
            existing_l1_error = float(existing_l1_error.replace(",", ".")) if existing_l1_error else None
        else:
            existing_l1_error = override_existing_l1_error

        if rank == 0:
            click.echo(f"Processing variable: {var} (Units: {da.attrs.get('units', 'N/A')}, Existing L1 Error: {existing_l1_error})")

        if field_percentage_to_compress is not None:
            field_percentage_to_compress = float(field_percentage_to_compress)
            slices = {dim: slice(0, max(1, int(size * (field_percentage_to_compress / 100)))) for dim, size in da.sizes.items()}
            da = da.isel(**slices)

        compressors = utils.compressor_space(da, with_lossy, with_numcodecs_wasm, with_ebcc, compressor_class)
        filters = utils.filter_space(da, with_lossy, with_numcodecs_wasm, with_ebcc, filter_class)
        serializers = utils.serializer_space(da, with_lossy, with_numcodecs_wasm, with_ebcc, serializer_class)

        num_compressors = len(compressors)
        num_filters = len(filters)
        num_serializers = len(serializers)

        num_loops = num_compressors * num_filters * num_serializers
        if rank == 0:
            click.echo(f"Number of loops: {num_loops} ({num_compressors} compressors, {num_filters} filters, {num_serializers} serializers) -divided across {size} MPI task(s)-")

        config_space = list(itertools.product(compressors, filters, serializers))
        configs_for_rank = config_space[rank::size]

        results = []
        raw_values_explicit_with_names = []
        total_configs = len(configs_for_rank)
        if rank == 0:
            pd.DataFrame(config_space).to_csv("config_space.csv", index=False)
        for i, ((comp_idx, compressor), (filt_idx, filt), (ser_idx, serializer)) in enumerate(configs_for_rank):
            data_to_compress = da
            if isinstance(serializer, numcodecs.zarr3.ZFPY):
                data_to_compress = da.stack(flat_dim=da.dims)
            if isinstance(serializer, AnyNumcodecsArrayBytesCodec) and isinstance(serializer.codec, EBCCZarrFilter):
                data_to_compress = da.squeeze().astype("float32")

            filters_ = [filt,]
            compressors_ = [compressor,]
            serializer_ = serializer
            
            if isinstance(serializer_, AnyNumcodecsArrayBytesCodec) or filt is None:
                filters_ = None  # TODO: fix (?) filter stacking with EBCC & numcodecs-wasm serializers
                filt = None
                filt_idx = -1
            if compressor is None:
                compressors_ = None
                comp_idx = -1
            if serializer is None:
                serializer_ = "auto"
                ser_idx = -1

            try:
                compression_ratio, errors, euclidean_distance = utils.compress_with_zarr(
                    data_to_compress,
                    dataset_file,
                    var,
                    where_to_write,
                    filters=filters_,
                    compressors=compressors_,
                    serializer=serializer_,
                    verbose=False,
                    rank=rank,
                )

                l1_error_rel = errors["Relative_Error_L1"]
                l2_error_rel = errors["Relative_Error_L2"]
                linf_error_rel = errors["Relative_Error_Linf"]

                # TODO: refine criteria based on the thresholds table
                if existing_l1_error:
                    if l1_error_rel <= existing_l1_error:
                        results.append(((str(compressor), str(filt), str(serializer), comp_idx, filt_idx, ser_idx), compression_ratio, l1_error_rel, euclidean_distance))
                        raw_values_explicit_with_names.append((compression_ratio, l1_error_rel, l2_error_rel, linf_error_rel, euclidean_distance, str(compressor), str(filt), str(serializer)))
                else:
                    results.append(((str(compressor), str(filt), str(serializer), comp_idx, filt_idx, ser_idx), compression_ratio, l1_error_rel, euclidean_distance))
                    raw_values_explicit_with_names.append((compression_ratio, l1_error_rel, l2_error_rel, linf_error_rel, euclidean_distance, str(compressor), str(filt), str(serializer)))

            except:
                click.echo(f"Failed to compress with {compressor}, {filt}, {serializer} [Indices: {comp_idx}, {filt_idx}, {ser_idx}]")
                traceback.print_exc(file=sys.stderr)
                sys.exit(1)

            utils.progress_bar(i, total_configs, print_every=100)

        results_gather = comm.gather(results, root=0)
        raw_values_explicit_with_names_gather = comm.gather(raw_values_explicit_with_names, root=0)

        if rank == 0:
            click.echo("Compressors analysis completed. Writing files...")
            # Flatten list of lists
            results_gather = list(itertools.chain.from_iterable(results_gather))
            raw_values_explicit_with_names_gather = list(itertools.chain.from_iterable(raw_values_explicit_with_names_gather))

            # Needed for clustering
            lossy_option = "with-lossy" if with_lossy else "without-lossy"
            numcodecs_wasm_option = "with-numcodecs-wasm" if with_numcodecs_wasm else "without-numcodecs-wasm"
            ebcc_option = "with-ebcc" if with_ebcc else "without-ebcc"
            score_results_file_name = [field_to_compress, compressor_class, filter_class, serializer_class, lossy_option, numcodecs_wasm_option, ebcc_option]
            np.save(os.path.basename(dataset_file) + '_' + '_'.join(score_results_file_name) + '_scored_results_with_names.npy', np.asarray(pd.DataFrame(raw_values_explicit_with_names_gather)))
            best_combo = max(results_gather, key=lambda x: x[1])
            msg = (
                "optimal combo: \n"
                f"compressor : {best_combo[0][0]}\nfilter     : {best_combo[0][1]}\nserializer : {best_combo[0][2]}\n"
                "corresponding indices in lists of instantiated objects:\n"
                f"compressor : {best_combo[0][3]}\nfilter     : {best_combo[0][4]}\nserializer : {best_combo[0][5]}\n"
                f"Compression Ratio: {best_combo[1]:.3f} | Relative L1 Error: {best_combo[2]:.3e} | Euclidean Distance: {best_combo[3]:.3e}"
            )
            click.echo(msg)


@cli.command("compress_with_optimal")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("where_to_write", type=click.Path(dir_okay=True, file_okay=False, exists=False))
@click.argument("field_to_compress")
@click.argument("comp_idx", type=int)
@click.argument("filt_idx", type=int)
@click.argument("ser_idx", type=int)
@click.option("--compressor-class", default="all", help="Same as in evaluate_combos.")
@click.option("--filter-class", default="all", help="Same as in evaluate_combos.")
@click.option("--serializer-class", default="all", help="Same as in evaluate_combos.")
@click.option("--with-lossy/--without-lossy", default=True, show_default=True, help="Same as in evaluate_combos.")
@click.option("--with-numcodecs-wasm/--without-numcodecs-wasm", default=True, show_default=True, help="Same as in evaluate_combos.")
@click.option("--with-ebcc/--without-ebcc", default=True, show_default=True, help="Same as in evaluate_combos.")
def compress_with_optimal(dataset_file, where_to_write, field_to_compress, 
                          comp_idx, filt_idx, ser_idx, 
                          compressor_class: str = "all", filter_class: str = "all", serializer_class: str = "all",
                          with_lossy: bool = True, with_numcodecs_wasm: bool = True, with_ebcc: bool = True):
    """
    Compress a field with the optimal combination of 
    compressor, filter, and serializer as generated by the evaluate_combos command.

    Make sure to provide the same --[compressor/filter/serializer]-class and the same --with/without-[lossy/numcodecs-wasm/ebcc] flags as in evaluate_combos,
    such that the same lists of instantiated objects are generated.
    
    Note on passing -1 as index:
    dc_toolkit compress_with_optimal ... --compressor-class X ... --- -1 -1 -1

    \b
    Args:
        dataset_file (str): Path to the input dataset file.
        where_to_write (str): Directory where the compressed output will be written.
        field_to_compress (str): Name of the field to compress.
        comp_idx (int): Index of the compressor to use.
        filt_idx (int): Index of the filter to use.
        ser_idx (int): Index of the serializer to use.
        compressor_class: --compressor-class
        filter_class: --filter-class
        serializer_class: --serializer-class
        with_lossy: --with-lossy/--without-lossy
        with_numcodecs_wasm: --with-numcodecs-wasm/--without-numcodecs-wasm
        with_ebcc: --with-ebcc/--without-ebcc
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size > 1:
        if rank == 0:
            click.echo("This command is not meant to be run in parallel. Please run it with a single process.")
        sys.exit(1)

    os.makedirs(where_to_write, exist_ok=True)

    ds = utils.open_dataset(dataset_file, field_to_compress)
    da = ds[field_to_compress]

    compressors = utils.compressor_space(da, with_lossy, with_numcodecs_wasm, with_ebcc, compressor_class)
    filters = utils.filter_space(da, with_lossy, with_numcodecs_wasm, with_ebcc, filter_class)
    serializers = utils.serializer_space(da, with_lossy, with_numcodecs_wasm, with_ebcc, serializer_class)

    if -1 <= comp_idx < len(compressors):
        pass
    else:
        click.echo(f"Invalid comp_idx: {comp_idx}")
        sys.exit(1)
    if -1 <= filt_idx < len(filters):
        pass
    else:
        click.echo(f"Invalid filt_idx: {filt_idx}")
        sys.exit(1)
    if -1 <= ser_idx < len(serializers):
        pass
    else:
        click.echo(f"Invalid ser_idx: {ser_idx}")
        sys.exit(1)

    optimal_compressor = compressors[comp_idx][1] if comp_idx != -1 else None
    optimal_filter = filters[filt_idx][1] if filt_idx != -1 else None
    optimal_serializer = serializers[ser_idx][1] if ser_idx != -1 else None

    if isinstance(optimal_serializer, numcodecs.zarr3.ZFPY):
        da = da.stack(flat_dim=da.dims)
    if isinstance(optimal_serializer, AnyNumcodecsArrayBytesCodec) and isinstance(optimal_serializer.codec, EBCCZarrFilter):
        da = da.squeeze().astype("float32")

    filters_ = [optimal_filter,]
    compressors_ = [optimal_compressor,]
    serializer_ = optimal_serializer

    if isinstance(serializer_, AnyNumcodecsArrayBytesCodec) or optimal_filter is None:
        filters_ = None
    if optimal_compressor is None:
        compressors_ = None
    if optimal_serializer is None:
        serializer_ = "auto"

    compression_ratio, errors, euclidean_distance = utils.compress_with_zarr(
        da,
        dataset_file,
        field_to_compress,
        where_to_write,
        filters=filters_,
        compressors=compressors_,
        serializer=serializer_,
        verbose=False,
    )

    click.echo(f" | {(optimal_compressor, optimal_filter, optimal_serializer)} | --> Ratio: {compression_ratio:.3f} | Error: {errors['Relative_Error_L1']:.3e} | Euclidean Distance: {euclidean_distance:.3e}")
    msg = (
        "optimal combo: \n"
        f"compressor : {optimal_compressor}\nfilter     : {optimal_filter}\nserializer : {optimal_serializer}\n"
        "corresponding indices in lists of instantiated objects:\n"
        f"compressor : {comp_idx}\nfilter     : {filt_idx}\nserializer : {ser_idx}\n"
        f"Compression Ratio: {compression_ratio:.3f} | Relative L1 Error: {errors['Relative_Error_L1']:.3e} | Euclidean Distance: {euclidean_distance:.3e}"
    )
    click.echo(msg)

@cli.command("merge_compressed_fields")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("compressed_files_location", type=click.Path(dir_okay=True, file_okay=False, exists=False))
def merge_compressed_fields(dataset_file: str, compressed_files_location: str):
    """
    Once all fields have been compressed, this command merges them into a single Zarr Zipped file.

    \b
    Args:
        dataset_file (str): Path to the input dataset file.
        compressed_files_location (str): Directory where the compressed files are located.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size > 1:
        if rank == 0:
            click.echo("This command is not meant to be run in parallel. Please run it with a single process.")
        sys.exit(1)

    # populate this folder with the compressed fields
    dataset_filename = Path(dataset_file).name
    merged_folder = Path(compressed_files_location) / f"{dataset_filename}.zarr"
    if Path(merged_folder).exists():
        shutil.rmtree(merged_folder)
    os.makedirs(merged_folder)

    for var in utils.open_dataset(dataset_file).data_vars:
        compressed_field = f"{Path(compressed_files_location) / dataset_filename}.=.field_{var}.=.rank_{rank}.zarr.zip"
        if not Path(compressed_field).exists():
            click.echo("All fields must be compressed first.")
            sys.exit(1)
        extract_to = utils.unzip_file(compressed_field)
        utils.copy_folder_contents(extract_to, merged_folder)
        shutil.rmtree(extract_to)

    zipped_merged_folder = str(merged_folder) + ".zip"
    if Path(zipped_merged_folder).exists():
        os.remove(zipped_merged_folder)
    shutil.make_archive(merged_folder, 'zip', merged_folder)

    if Path(merged_folder).exists():
        shutil.rmtree(merged_folder)


@cli.command("open_zarr_zip_file_and_inspect")
@click.argument("zarr_zip_file", type=click.Path(exists=True, dir_okay=False))
def open_zarr_zip_file_and_inspect(zarr_zip_file: str):
    """
    Open a Zarr Zipped file and inspect its contents.

    \b
    Args:
        zarr_zip_file (str): Path to the Zarr file.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size > 1:
        if rank == 0:
            click.echo("This command is not meant to be run in parallel. Please run it with a single process.")
        sys.exit(1)

    zarr_group, store = utils.open_zarr_zipstore(zarr_zip_file)

    click.echo(zarr_group.tree())

    click.echo(80* "-")
    for array_name in zarr_group.array_keys():
        click.echo(f"Array: {array_name}")
        click.echo(zarr_group[array_name].info_complete())
        click.echo(zarr_group[array_name][:])
        click.echo(80* "-")

    store.close()


@cli.command("from_zarr_zip_to_netcdf")
@click.argument("zarr_zip_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_nc", type=click.Path(dir_okay=False), default=None,
              help="Output NetCDF file. Defaults to INPUT with .nc extension.")
def from_zarr_zip_to_netcdf(zarr_zip_file: str, out_nc: str | None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size > 1:
        if rank == 0:
            click.echo("This command is not meant to be run in parallel. Please run it with a single process.")
        sys.exit(1)

    if out_nc is None:
        out_nc = os.path.splitext(zarr_zip_file)[0] + ".nc"

    zgroup, store = utils.open_zarr_zipstore(zarr_zip_file)
    try:
        names = list(zgroup.array_keys())
        if not names:
            raise click.ClickException("No arrays found in the Zarr store.")
        ds = xr.Dataset({
            n: xr.DataArray(dask.array.from_zarr(zgroup[n]),
                            dims=[f"{n}_d{i}" for i in range(zgroup[n].ndim)],
                            name=n)
            for n in names
        })
        ds.to_netcdf(out_nc, engine="h5netcdf")
        click.echo(f"Wrote NetCDF: {out_nc}")
    finally:
        store.close()


@cli.command("perform_clustering")
@click.argument("npy_file", type=click.Path(exists=True, dir_okay=False))
def perform_clustering(npy_file: str):
    scored_results = np.load(npy_file, allow_pickle=True)

    scored_results_pd = pd.DataFrame(scored_results)

    numeric_cols = scored_results_pd.select_dtypes(include=[np.number]).columns
    mask = np.isfinite(scored_results_pd[numeric_cols]).all(axis=1)
    scored_results_pd = scored_results_pd[mask].dropna()
    clean_arr_inf = np.hstack((np.asarray(scored_results_pd[[0]]), np.asarray(scored_results_pd[[2]])))

    k_values = range(3, 10)
    inertias = []
    silhouette_scores = []

    for k in tqdm(k_values, desc="Looping over k values"):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(clean_arr_inf)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(clean_arr_inf, labels))

    # Plot Elbow Curve
    plt.figure(figsize=(12, 5))
    plt.suptitle("LInf")
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    # Plot Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, 'go-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.tight_layout()
    plt.show()


@cli.command("analyze_clustering")
@click.argument("npy_file", type=click.Path(exists=True, dir_okay=False))
def analyze_clustering(npy_file: str):
    scored_results = np.load(str(npy_file), allow_pickle=True)

    scored_results_pd = pd.DataFrame(scored_results)

    numeric_cols = scored_results_pd.select_dtypes(include=[np.number]).columns
    mask = np.isfinite(scored_results_pd[numeric_cols]).all(axis=1)
    scored_results_pd = scored_results_pd[mask].dropna()

    clean_arr_l1 = utils.slice_array(scored_results_pd, [0, 1, 5, 6, 7])
    clean_arr_l2 = utils.slice_array(scored_results_pd, [0, 2, 5, 6, 7])
    clean_arr_linf = utils.slice_array(scored_results_pd, [0, 3, 5, 6, 7])
    clean_arr_dwt = utils.slice_array(scored_results_pd, [0, 4, 5, 6, 7])


    # Plot Error and Similarity Metrics VS Ratio
    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=[
                            "L1 VS Ratio KMeans Clustering", "L2 VS Ratio KMeans Clustering",
                            "LInf VS Ratio KMeans Clustering"
                        ])

    # L1 clustering
    clean_arr_l1_filtered = np.column_stack((clean_arr_l1[:, 0].astype(float), clean_arr_l1[:, 1].astype(float)))

    df_l1 = pd.DataFrame(clean_arr_l1_filtered, columns=["Ratio", "L1"])
    df_l1["compressor"] = clean_arr_l1[:, 2]
    df_l1["filter"] = clean_arr_l1[:, 3]
    df_l1["serializer"] = clean_arr_l1[:, 4]

    y_kmeans = kmeans.fit_predict(pd.DataFrame(df_l1, columns=["Ratio", "L1"]))
    color = np.ones(y_kmeans.shape) if len(np.unique(y_kmeans)) == 1 else y_kmeans

    fig_l1 = px.scatter(df_l1, x="Ratio", y="L1", color=color,
                        title="L1 VS Ratio KMeans Clustering",
                        hover_data=["compressor", "filter", "serializer"])

    fig.add_trace(
        go.Scatter(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            mode="markers+text",
            marker=dict(color="black", size=12, symbol="x"),
            textposition="top center",
            name="Centroids",
            showlegend=True
        ),
        row=1,
        col=1
    )
    fig.update_xaxes(title_text="Ratio", row=1, col=1)
    fig.update_yaxes(title_text="L1", row=1, col=1)

    for trace in fig_l1.data:
        fig.add_trace(trace, row=1, col=1)

        # L2 clustering
        clean_arr_l2_filtered = np.column_stack((clean_arr_l2[:, 0].astype(float), clean_arr_l2[:, 1].astype(float)))

        df_l2 = pd.DataFrame(clean_arr_l2_filtered, columns=["Ratio", "L2"])
        df_l2["compressor"] = clean_arr_l2[:, 2]
        df_l2["filter"] = clean_arr_l2[:, 3]
        df_l2["serializer"] = clean_arr_l2[:, 4]

        y_kmeans = kmeans.fit_predict(pd.DataFrame(df_l2, columns=["Ratio", "L2"]))
        color = np.ones(y_kmeans.shape) if len(np.unique(y_kmeans)) == 1 else y_kmeans

        fig_l2 = px.scatter(df_l2, x="Ratio", y="L2", color=color,
                            title="L2 VS Ratio KMeans Clustering",
                            hover_data=["compressor", "filter", "serializer"])

        fig.add_trace(
            go.Scatter(
                x=kmeans.cluster_centers_[:, 0],
                y=kmeans.cluster_centers_[:, 1],
                mode="markers+text",
                marker=dict(color="black", size=12, symbol="x"),
                textposition="top center",
                name="Centroids",
                showlegend=False
            ),
            row=2,
            col=1
        )
        fig.update_xaxes(title_text="Ratio", row=2, col=1)
        fig.update_yaxes(title_text="L2", row=2, col=1)
        for trace in fig_l2.data:
            fig.add_trace(trace, row=2, col=1)

        # LInf clustering
        clean_arr_linf_filtered = np.column_stack(
            (clean_arr_linf[:, 0].astype(float), clean_arr_linf[:, 1].astype(float)))

        df_linf = pd.DataFrame(clean_arr_linf_filtered, columns=["Ratio", "LInf"])
        df_linf["compressor"] = clean_arr_linf[:, 2]
        df_linf["filter"] = clean_arr_linf[:, 3]
        df_linf["serializer"] = clean_arr_linf[:, 4]
        df_linf["compressor_idx"] = utils.get_indexes(clean_arr_linf[:, 2], config_idxs['0'])
        df_linf["filter_idx"] = utils.get_indexes(clean_arr_linf[:, 3], config_idxs['1'])
        df_linf["serializer_idx"] = utils.get_indexes(clean_arr_linf[:, 4], config_idxs['2'])

        y_kmeans = kmeans.fit_predict(pd.DataFrame(df_linf, columns=["Ratio", "LInf"]))
        color = np.ones(y_kmeans.shape) if len(np.unique(y_kmeans)) == 1 else y_kmeans

        fig_linf = px.scatter(df_linf, x="Ratio", y="LInf", color=color,
                              title="LInf VS Ratio KMeans Clustering",
                              hover_data=["compressor", "filter", "serializer", "compressor_idx", "filter_idx",
                                          "serializer_idx"])

        fig.add_trace(
            go.Scatter(
                x=kmeans.cluster_centers_[:, 0],
                y=kmeans.cluster_centers_[:, 1],
                mode="markers+text",
                marker=dict(color="black", size=12, symbol="x"),
                textposition="top center",
                name="Centroids",
                showlegend=False
            ),
            row=3,
            col=1
        )
        fig.update_xaxes(title_text="Ratio", row=3, col=1)
        fig.update_yaxes(title_text="LInf", row=3, col=1)
        for trace in fig_linf.data:
            fig.add_trace(trace, row=3, col=1)

    fig.update_layout(
        title="",
        showlegend=False,
        height=900,
        hovermode="closest",
        template="plotly_white"
    )
    pio.renderers.default = "browser"
    fig.show()


@cli.command("run_web_ui_santis")
@click.option("--user_account", type=str, default="", help="Santis user account name")
@click.option("--uploaded_file", type=str, default="", help="Upload file from santis")
@click.option("--time", type=str, default="", help="Allocated time")
@click.option("--nodes", type=str, default="", help="Number of nodes")
@click.option("--ntasks-per-node", type=str, default="", help="Number of tasks per node")
def run_web_ui_santis(user_account: str = None, uploaded_file: str = "", time: str = "", nodes: str = "", ntasks_per_node: str = ""):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cmd_web_ui_santis = [
        "streamlit", "run", str(current_dir) + "/compression_analysis_ui_web.py", "--", "--user_account", user_account,
        "--uploaded_file", uploaded_file, "--time", time, "--nodes", nodes, "--ntasks-per-node", ntasks_per_node
    ]
    subprocess.run(cmd_web_ui_santis)

@cli.command("run_web_ui")
def run_web_ui():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cmd_web_ui = [
        "streamlit", "run", str(current_dir) + "/compression_analysis_ui_web.py"
    ]
    subprocess.run(cmd_web_ui)

@cli.command("run_local_ui")
def run_local_ui():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cmd_local_ui = [
        "python", str(current_dir) + "/compression_analysis_ui_local.py"
    ]
    subprocess.run(cmd_local_ui)


@cli.command("help")
@click.pass_context
def help(ctx):
    for command in cli.commands.values():
        if command.name == "help":
            continue
        click.echo("-" * 80)
        click.echo()
        with click.Context(command, parent=ctx.parent, info_name=command.name) as ctx:
            click.echo(command.get_help(ctx=ctx))
        click.echo()


if __name__ == "__main__":
    cli()
