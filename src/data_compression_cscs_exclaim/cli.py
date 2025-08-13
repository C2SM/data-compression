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
import inspect
import traceback
import click
from tqdm import tqdm
from pathlib import Path
import zarr
import shutil
import itertools
import numcodecs
import numcodecs.zarr3
from data_compression_cscs_exclaim import utils
from numcodecs_combinators.stack import CodecStack
from numcodecs_wasm_asinh import Asinh
from numcodecs_wasm_bit_round import BitRound
from numcodecs_wasm_linear_quantize import LinearQuantize
from numcodecs_wasm_zlib import Zlib
from zarr_any_numcodecs import AnyNumcodecsArrayArrayCodec, AnyNumcodecsArrayBytesCodec, AnyNumcodecsBytesBytesCodec
from ebcc.filter_wrapper import EBCC_Filter
from ebcc.zarr_filter import EBCCZarrFilter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numcodecs_wasm_pco import Pco
from numcodecs_wasm_sperr import Sperr
from numcodecs_wasm_sz3 import Sz3
from numcodecs_wasm_zfp import Zfp
from sklearn.preprocessing import StandardScaler
from mpi4py import MPI
import dask
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings(
    "ignore",
    message="Numcodecs codecs are not in the Zarr version 3 specification and may not be supported by other zarr implementations",
    category=UserWarning,
    module="numcodecs.zarr3"
)
warnings.filterwarnings("ignore", message="overflow encountered in square")

os.environ["EBCC_LOG_LEVEL"] = "4"  # ERROR (suppress WARN and below)


@click.group()
def cli():
    pass


@cli.command("linear_quantization_zlib_compressors")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def linear_quantization_zlib_compressors(
    netcdf_file: str, field_to_compress: str, parameters_file: str
):
    ds = utils.open_netcdf(netcdf_file, field_to_compress)

    linear_quantization_bits, zlib_level = utils.get_filter_parameters(
        parameters_file, inspect.currentframe().f_code.co_name
    )

    utils.compress_with_zarr(ds[field_to_compress], netcdf_file, field_to_compress,
        filters=[AnyNumcodecsArrayArrayCodec(LinearQuantize(bits=linear_quantization_bits, dtype=str(ds[field_to_compress].dtype)))],
        compressors=[AnyNumcodecsBytesBytesCodec(Zlib(level=zlib_level))],
    )


@cli.command("bitround_zlib_compressors")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def bitround_zlib_compressors(netcdf_file: str, field_to_compress: str, parameters_file: str):
    ds = utils.open_netcdf(netcdf_file, field_to_compress)

    bitround_bits, zlib_level = utils.get_filter_parameters(
        parameters_file, inspect.currentframe().f_code.co_name
    )

    utils.compress_with_zarr(ds[field_to_compress], netcdf_file, field_to_compress,
        filters=[AnyNumcodecsArrayArrayCodec(BitRound(keepbits=bitround_bits))],
        compressors=[AnyNumcodecsBytesBytesCodec(Zlib(level=zlib_level))],
    )


@cli.command("zfp_asinh_compressors")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def zfp_asinh_compressors(netcdf_file: str, field_to_compress: str, parameters_file: str):
    ds = utils.open_netcdf(netcdf_file, field_to_compress)

    asinh_linear_width, zfp_mode, zfp_tolerance = utils.get_filter_parameters(
        parameters_file, inspect.currentframe().f_code.co_name
    )

    utils.compress_with_zarr(ds[field_to_compress], netcdf_file, field_to_compress,
        filters=[AnyNumcodecsArrayArrayCodec(Asinh(linear_width=asinh_linear_width))],
        compressors=None,
        serializer=numcodecs.zarr3.ZFPY(tolerance=zfp_tolerance)
    )


@cli.command("sz3_eb_compressors")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def sz3_eb_compressors(netcdf_file: str, field_to_compress: str, parameters_file: str):
    ds = utils.open_netcdf(netcdf_file, field_to_compress)

    sz3_eb_mode, sz3_eb_rel = utils.get_filter_parameters(
        parameters_file, inspect.currentframe().f_code.co_name
    )

    sz3_ = Sz3(eb_mode=sz3_eb_mode, eb_rel=sz3_eb_rel)

    utils.compress_with_zarr(ds[field_to_compress], netcdf_file, field_to_compress,
        filters=None,
        compressors=None,
        serializer=AnyNumcodecsArrayBytesCodec(sz3_)
    )


@cli.command("ebcc")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
def ebcc(netcdf_file: str, field_to_compress: str):
    
    ds = utils.open_netcdf(netcdf_file, field_to_compress)
    da = ds[field_to_compress].squeeze().astype("float32")

    atol = 1e-2
    ebcc_filter = EBCC_Filter(
            base_cr=2, 
            height=da.shape[0],
            width=da.shape[1],
            residual_opt=("max_error_target", atol)
        )
    zarr_filter = EBCCZarrFilter(ebcc_filter.hdf_filter_opts)

    utils.compress_with_zarr(da, netcdf_file, field_to_compress,
        filters=None,
        compressors=None,
        serializer=AnyNumcodecsArrayBytesCodec(zarr_filter),
    )


@cli.command("compress_with_optimal")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("where_to_write", type=click.Path(dir_okay=True, file_okay=False, exists=False))
@click.argument("field_to_compress")
@click.argument("comp_idx", type=int)
@click.argument("filt_idx", type=int)
@click.argument("ser_idx", type=int)
def compress_with_optimal(netcdf_file, where_to_write, field_to_compress, comp_idx, filt_idx, ser_idx):
    os.makedirs(where_to_write, exist_ok=True)

    ds = utils.open_netcdf(netcdf_file, field_to_compress)
    da = ds[field_to_compress]

    compressors = utils.compressor_space(da)
    filters = utils.filter_space(da)
    serializers = utils.serializer_space(da)

    optimal_compressor = compressors[comp_idx][1]
    optimal_filter = filters[filt_idx][1]
    optimal_serializer = serializers[ser_idx][1]
    
    if isinstance(optimal_serializer, AnyNumcodecsArrayBytesCodec):
        if isinstance(optimal_serializer.codec, (Pco, Sperr, Sz3, Zfp)):
            da = da.stack(flat_dim=da.dims)
        elif isinstance(optimal_serializer.codec, EBCCZarrFilter):
            da = da.squeeze().astype("float32")

    compression_ratio, errors, dwt_dist = utils.compress_with_zarr(
        da,
        netcdf_file,
        field_to_compress,
        where_to_write,
        filters=None if isinstance(optimal_serializer, AnyNumcodecsArrayBytesCodec) else [optimal_filter,],  # TODO: fix (?) filter stacking with EBCC & numcodecs-wasm serializers
        compressors=[optimal_compressor,],
        serializer=optimal_serializer,
        verbose=False,
    )

    click.echo(f" | {(optimal_compressor, optimal_filter, optimal_serializer)} | --> Ratio: {compression_ratio:.3f} | Error: {errors['Relative_Error_L1']:.3e} | DWT: {dwt_dist:.3e}")


@cli.command("merge_compressed_fields")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
def merge_compressed_fields(netcdf_file: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size > 1:
        if rank == 0:
            click.echo("This command is not meant to be run in parallel. Please run it with a single process.")
        sys.exit(1)

    # populate this folder with the compressed fields
    merged_folder = f"{netcdf_file}.zarr"
    if Path(merged_folder).exists():
        shutil.rmtree(merged_folder)
    os.makedirs(merged_folder)

    for var in utils.open_netcdf(netcdf_file).data_vars:
        compressed_field = f"{netcdf_file}.=.field_{var}.=.rank_{rank}.zarr.zip"
        if not Path(compressed_field).exists():
            click.echo("All fields must be compressed first.")
            sys.exit(1)
        extract_to = utils.unzip_file(compressed_field)
        utils.copy_folder_contents(extract_to, merged_folder)
        shutil.rmtree(extract_to)

    zipped_merged_folder = merged_folder + ".zip"
    if Path(zipped_merged_folder).exists():
        os.remove(zipped_merged_folder)
    shutil.make_archive(merged_folder, 'zip', merged_folder)

    if Path(merged_folder).exists():
        shutil.rmtree(merged_folder)


@cli.command("open_zarr_file_and_inspect")
@click.argument("zarr_file", type=click.Path(exists=True, dir_okay=False))
def open_zarr_file_and_inspect(zarr_file: str):
    zarr_group, store = utils.open_zarr_zipstore(zarr_file)

    click.echo(zarr_group.tree())

    click.echo(80* "-")
    for array_name in zarr_group.array_keys():
        click.echo(f"Array: {array_name}")
        click.echo(zarr_group[array_name].info_complete())
        click.echo(zarr_group[array_name][:])
        click.echo(80* "-")

    store.close()


@cli.command("summarize_compression")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("where_to_write", type=click.Path(dir_okay=True, file_okay=False, exists=False))
@click.option("--field-to-compress", default=None, help="Field to compress [if not given, all fields will be compressed].")
@click.option("--field-percentage-to-compress", default=None, callback=utils.validate_percentage, help="Compress a percentage of the field [1-99%]. If not given, the whole field will be compressed.")
def summarize_compression(netcdf_file: str, where_to_write: str, field_to_compress: str | None = None, field_percentage_to_compress: str | None = None):
    ## https://numcodecs.readthedocs.io/en/stable/zarr3.html#zarr-3-codecs
    ## https://numcodecs-wasm.readthedocs.io/en/latest/
    os.makedirs(where_to_write, exist_ok=True)

    dask.config.set(scheduler="single-threaded")
    dask.config.set(array__chunk_size="512MiB")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        try:
            # Lookup table for valid thresholds
            # https://docs.google.com/spreadsheets/d/1lHcX-HE2WpVCOeKyDvM4iFqjlWvkd14lJlA-CUoCxMM/edit?gid=0#gid=0
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

    # This is opened by all MPI processes
    ds = utils.open_netcdf(netcdf_file, field_to_compress, field_percentage_to_compress, rank=rank)

    for var in ds.data_vars:
        if field_to_compress is not None and field_to_compress != var:
            continue
        da = ds[var]

        # TODO: fix this hack
        lookup = var
        if var == "temp":
            lookup = "t"

        threshold_row = thresholds[thresholds["Short Name"] == lookup]
        matching_units = threshold_row.iloc[0]["Unit"] == da.attrs.get("units", None) if not threshold_row.empty else None
        existing_l1_error = threshold_row.iloc[0]["Existing L1 error"] if not threshold_row.empty and matching_units else None
        existing_l1_error = float(existing_l1_error.replace(",", ".")) if existing_l1_error else None

        if rank == 0:
            click.echo(f"Processing variable: {var} (Units: {da.attrs.get('units', 'N/A')}, Existing L1 Error: {existing_l1_error})")

        if field_percentage_to_compress is not None:
            field_percentage_to_compress = float(field_percentage_to_compress)
            slices = {dim: slice(0, max(1, int(size * (field_percentage_to_compress / 100)))) for dim, size in da.sizes.items()}
            da = da.isel(**slices)

        compressors = utils.compressor_space(da)
        filters = utils.filter_space(da)
        serializers = utils.serializer_space(da)

        num_compressors = len(compressors)
        num_filters = len(filters)
        num_serializers = len(serializers)

        num_loops = num_compressors * num_filters * num_serializers
        if rank == 0:
            click.echo(f"Number of loops: {num_loops} ({num_compressors} compressors, {num_filters} filters, {num_serializers} serializers) -divided across multiple processes-")

        config_space = list(itertools.product(compressors, filters, serializers))
        configs_for_rank = config_space[rank::size]

        results = []
        raw_values_explicit = []
        raw_values_explicit_with_names = []
        total_configs = len(configs_for_rank)
        for i, ((comp_idx, compressor), (filt_idx, filter), (ser_idx, serializer)) in enumerate(configs_for_rank):
            data_to_compress = da
            if isinstance(serializer, AnyNumcodecsArrayBytesCodec):
                if isinstance(serializer.codec, (Pco, Sperr, Sz3, Zfp)):
                    data_to_compress = da.stack(flat_dim=da.dims)
                elif isinstance(serializer.codec, EBCCZarrFilter):
                    data_to_compress = da.squeeze().astype("float32")

            try:
                compression_ratio, errors, dwt_dist = utils.compress_with_zarr(
                    data_to_compress,
                    netcdf_file,
                    var,
                    where_to_write,
                    filters=None if isinstance(serializer, AnyNumcodecsArrayBytesCodec) else [filter,],  # TODO: fix (?) filter stacking with EBCC & numcodecs-wasm serializers
                    compressors=[compressor,],
                    serializer=serializer,
                    verbose=False,
                    rank=rank,
                )

                l1_error_rel = errors["Relative_Error_L1"]
                l2_error_rel = errors["Relative_Error_L2"]
                linf_error_rel = errors["Relative_Error_Linf"]
                raw_values_explicit.append((compression_ratio, l1_error_rel, l2_error_rel, linf_error_rel, dwt_dist))

                # TODO: refine criteria based on the thersholds table
                if existing_l1_error:
                    if l1_error_rel <= existing_l1_error:
                        results.append(((str(compressor), str(filter), str(serializer), comp_idx, filt_idx, ser_idx), compression_ratio, l1_error_rel, dwt_dist))
                        raw_values_explicit_with_names.append((compression_ratio, l1_error_rel, l2_error_rel, linf_error_rel, dwt_dist, str(compressor), str(filter), str(serializer)))
                else:
                    results.append(((str(compressor), str(filter), str(serializer), comp_idx, filt_idx, ser_idx), compression_ratio, l1_error_rel, dwt_dist))
                    raw_values_explicit_with_names.append((compression_ratio, l1_error_rel, l2_error_rel, linf_error_rel, dwt_dist, str(compressor), str(filter), str(serializer)))

            except:
                if rank == 0:
                    click.echo(f"Failed to compress with {compressor}, {filter}, {serializer} [Indices: {comp_idx}, {filt_idx}, {ser_idx}]")
                    traceback.print_exc(file=sys.stderr)
                sys.exit(1)

            utils.progress_bar(rank, i, total_configs, print_every=100)

        results_gather = comm.gather(results, root=0)
        raw_values_explicit_gather = comm.gather(raw_values_explicit, root=0)
        raw_values_explicit_with_names_gather = comm.gather(raw_values_explicit_with_names, root=0)

        if rank == 0:
            click.echo("Compressors analysis completed. Writing files...")
            # Flatten list of lists
            results_gather = list(itertools.chain.from_iterable(results_gather))
            raw_values_explicit_gather = list(itertools.chain.from_iterable(raw_values_explicit_gather))
            raw_values_explicit_with_names_gather = list(itertools.chain.from_iterable(raw_values_explicit_with_names_gather))

            # Needed for clustering
            np.save(os.path.basename(netcdf_file) + '_scored_results_raw.npy', np.asarray(pd.DataFrame(raw_values_explicit_gather)))
            np.save(os.path.basename(netcdf_file) + '_scored_results_with_names.npy', np.asarray(pd.DataFrame(raw_values_explicit_with_names_gather)))

            best_combo = max(results_gather, key=lambda x: x[1])
            click.echo("Best combo (valid threshold & max CR):")
            click.echo(f" | {best_combo[0]} | --> Ratio: {best_combo[1]:.3f} | Error: {best_combo[2]:.3e} | DWT: {best_combo[3]:.3e}")


@cli.command("perform_clustering")
@click.argument("npy_file", type=click.Path(exists=True, dir_okay=False))
def perform_clustering(npy_file: str):
    scored_results = np.load(npy_file, allow_pickle=True)

    scored_results_pd = pd.DataFrame(scored_results)

    numeric_cols = scored_results_pd.select_dtypes(include=[np.number]).columns
    mask = np.isfinite(scored_results_pd[numeric_cols]).all(axis=1)
    scored_results_pd = scored_results_pd[mask].dropna()
    clean_arr_dwt = np.hstack((np.asarray(scored_results_pd[[0]]), np.asarray(scored_results_pd[[4]])))

    k_values = range(3, 10)
    inertias = []
    silhouette_scores = []

    for k in tqdm(k_values, desc="Looping over k values"):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(clean_arr_dwt)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(clean_arr_dwt, labels))

    # Plot Elbow Curve
    plt.figure(figsize=(12, 5))
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

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[
                            "L1 VS Ratio KMeans Clustering", "L2 VS Ratio KMeans Clustering",
                            "LInf VS Ratio KMeans Clustering", "DWT VS Ratio KMeans Clustering"
                        ])

    # L1 clustering
    clean_arr_l1_filtered = np.column_stack((clean_arr_l1[:, 0].astype(float), clean_arr_l1[:, 1].astype(float)))
    y_kmeans = kmeans.fit_predict(clean_arr_l1_filtered)
    df_l1 = pd.DataFrame(clean_arr_l1_filtered, columns=["Ratio", "L1"])
    df_l1["compressor"] = clean_arr_l1[:, 2]
    df_l1["filter"] = clean_arr_l1[:, 3]
    df_l1["serializer"] = clean_arr_l1[:, 4]
    fig_l1 = px.scatter(df_l1, x="Ratio", y="L1", color=y_kmeans,
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
    for trace in fig_l1.data:
        fig.add_trace(trace, row=1, col=1)

    # L2 clustering
    clean_arr_l2_filtered = np.column_stack((clean_arr_l2[:, 0].astype(float), clean_arr_l2[:, 1].astype(float)))
    y_kmeans = kmeans.fit_predict(clean_arr_l2_filtered)
    df_l2 = pd.DataFrame(clean_arr_l2_filtered, columns=["Ratio", "L2"])
    df_l2["compressor"] = clean_arr_l2[:, 2]
    df_l2["filter"] = clean_arr_l2[:, 3]
    df_l2["serializer"] = clean_arr_l2[:, 4]
    fig_l2 = px.scatter(df_l2, x="Ratio", y="L2", color=y_kmeans,
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
        row=1,
        col=2
    )
    for trace in fig_l2.data:
        fig.add_trace(trace, row=1, col=2)


    # LInf clustering
    clean_arr_linf_filtered = np.column_stack((clean_arr_linf[:, 0].astype(float), clean_arr_linf[:, 1].astype(float)))
    y_kmeans = kmeans.fit_predict(clean_arr_linf_filtered)
    df_linf = pd.DataFrame(clean_arr_linf_filtered, columns=["Ratio", "LInf"])
    df_linf["compressor"] = clean_arr_linf[:, 2]
    df_linf["filter"] = clean_arr_linf[:, 3]
    df_linf["serializer"] = clean_arr_linf[:, 4]
    fig_linf = px.scatter(df_linf, x="Ratio", y="LInf", color=y_kmeans,
                     title="LInf VS Ratio KMeans Clustering",
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
    for trace in fig_linf.data:
        fig.add_trace(trace, row=2, col=1)


    # DWT clustering
    clean_arr_dwt_filtered = np.column_stack((clean_arr_dwt[:, 0].astype(float), clean_arr_dwt[:, 1].astype(float)))
    y_kmeans = kmeans.fit_predict(clean_arr_dwt_filtered)
    df_dwt = pd.DataFrame(clean_arr_dwt_filtered, columns=["Ratio", "DWT"])
    df_dwt["compressor"] = clean_arr_dwt[:, 2]
    df_dwt["filter"] = clean_arr_dwt[:, 3]
    df_dwt["serializer"] = clean_arr_dwt[:, 4]
    fig_dwt = px.scatter(df_dwt, x="Ratio", y="DWT", color=y_kmeans,
                     title="DWT VS Ratio KMeans Clustering",
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
        col=2
    )
    for trace in fig_dwt.data:
        fig.add_trace(trace, row=2, col=2)

    fig.update_layout(
        title="",
        showlegend=False,
        height=900,
        hovermode="closest",
        template="plotly_white"
    )
    pio.renderers.default = "browser"
    fig.show()


@cli.command("check_errors_at_dateline")
@click.argument("message")
def check_errors_at_dateline(message: str):
    print(message)


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
