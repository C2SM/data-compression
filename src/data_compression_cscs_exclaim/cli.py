# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import inspect
import click
from tqdm import tqdm
import itertools
import numcodecs
import numcodecs.zarr3
from data_compression_cscs_exclaim import utils
from numcodecs_combinators.stack import CodecStack
from numcodecs_wasm_asinh import Asinh
from numcodecs_wasm_bit_round import BitRound
from numcodecs_wasm_linear_quantize import LinearQuantize
from numcodecs_wasm_sz3 import Sz3
from numcodecs_wasm_zfp import Zfp
from numcodecs_wasm_zlib import Zlib
from zarr_any_numcodecs import AnyNumcodecsArrayArrayCodec, AnyNumcodecsArrayBytesCodec, AnyNumcodecsBytesBytesCodec
from ebcc.zarr_filter import EBCCZarrFilter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
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
    from ebcc.filter_wrapper import EBCC_Filter
    from ebcc.zarr_filter import EBCCZarrFilter
    
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


@cli.command("summarize_compression")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
def summarize_compression(netcdf_file: str, field_to_compress: str):
    ## https://numcodecs.readthedocs.io/en/stable/zarr3.html#zarr-3-codecs

    ds = utils.open_netcdf(netcdf_file, field_to_compress)
    da = ds[field_to_compress]

    #extract 20% of dataset
    # slices = {dim: slice(0, int(size * 0.8)) for dim, size in da.sizes.items()}
    # da = da.isel(**slices)

    compressors = utils.compressor_space(da)
    filters = utils.filter_space(da)
    serializers = utils.serializer_space(da)
    
    num_compressors = len(compressors)
    num_filters = len(filters)
    num_serializers = len(serializers)
    num_loops = num_compressors * num_filters * num_serializers
    click.echo(f"Number of loops: {num_loops} ({num_compressors} compressors, {num_filters} filters, {num_serializers} serializers)")

    # Lookup table for valid thresholds
    # https://docs.google.com/spreadsheets/d/1lHcX-HE2WpVCOeKyDvM4iFqjlWvkd14lJlA-CUoCxMM/edit?gid=0#gid=0
    sheet_id = "1lHcX-HE2WpVCOeKyDvM4iFqjlWvkd14lJlA-CUoCxMM"
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    thresholds = pd.read_csv(sheet_url)

    results = []
    raw_values_explicit = []
    raw_values_explicit_with_names = []
    for compressor, filter, serializer in tqdm(
        itertools.product(compressors, filters, serializers),
        total=num_loops,
        desc="Executing compression combinations",
    ):
        compression_ratio, errors, dwt_dist = utils.compress_with_zarr(
            da if not isinstance(serializer, EBCCZarrFilter) else da.squeeze().astype("float32"),
            netcdf_file,
            field_to_compress,
            filters=None if isinstance(serializer, EBCCZarrFilter) else [filter,],  # TODO: fix filter stacking with EBCC
            compressors=[compressor,],
            serializer=serializer if not isinstance(serializer, EBCCZarrFilter) else AnyNumcodecsArrayBytesCodec(serializer),
            verbose=False
        )

        l1_error_rel = errors["Relative_Error_L1"]
        l2_error_rel = errors["Relative_Error_L2"]
        linf_error_rel = errors["Relative_Error_Linf"]
        raw_values_explicit.append((compression_ratio, l1_error_rel, l2_error_rel, linf_error_rel, dwt_dist))
        # TODO: refine criteria based on the thersholds table
        if l1_error_rel <= 1e-2:
            results.append(((compressor, filter, serializer), compression_ratio, l1_error_rel, dwt_dist))
            raw_values_explicit_with_names.append((compression_ratio, l1_error_rel, l2_error_rel, linf_error_rel, dwt_dist, str(compressor.codec_name), str(filter.codec_name), str(serializer.codec_name)))
    
    # Needed for clustering
    np.save('scored_results_raw.npy', np.asarray(pd.DataFrame(raw_values_explicit)))
    np.save('scored_results_with_names.npy', np.asarray(pd.DataFrame(raw_values_explicit_with_names)))

    max_compression_ratio = 0
    for (cfg, ratio, error, dwt) in results:
        if ratio > max_compression_ratio:
            max_compression_ratio = ratio
            best_combo = (cfg, ratio, error, dwt)

    click.echo("Best combo (valid threshold & max CR):")
    click.echo(f"\ {best_combo[0]} \ --> Ratio: {best_combo[1]:.3f} | Error: {best_combo[2]:.3e} | DWT: {best_combo[3]:.3e}")

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

    clean_arr_l1 = np.hstack((np.asarray(scored_results_pd[[0]]),
                                      np.asarray(scored_results_pd[[1]]),
                                      np.asarray(scored_results_pd[[5]]),
                                      np.asarray(scored_results_pd[[6]]),
                                      np.asarray(scored_results_pd[[7]]),
                                      ))

    clean_arr_l2 = np.hstack((np.asarray(scored_results_pd[[0]]),
                                      np.asarray(scored_results_pd[[2]]),
                                      np.asarray(scored_results_pd[[5]]),
                                      np.asarray(scored_results_pd[[6]]),
                                      np.asarray(scored_results_pd[[7]]),
                                      ))

    clean_arr_linf = np.hstack((np.asarray(scored_results_pd[[0]]),
                                      np.asarray(scored_results_pd[[3]]),
                                      np.asarray(scored_results_pd[[5]]),
                                      np.asarray(scored_results_pd[[6]]),
                                      np.asarray(scored_results_pd[[7]]),
                                      ))

    clean_arr_dwt = np.hstack((np.asarray(scored_results_pd[[0]]),
                                      np.asarray(scored_results_pd[[4]]),
                                      np.asarray(scored_results_pd[[5]]),
                                      np.asarray(scored_results_pd[[6]]),
                                      np.asarray(scored_results_pd[[7]]),
                                      ))


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
    df_l1["cluster"] = y_kmeans.astype(str)
    df_l1["compressor"] = clean_arr_l1[:, 2]
    df_l1["filter"] = clean_arr_l1[:, 3]
    df_l1["serializer"] = clean_arr_l1[:, 4]
    fig_l1 = px.scatter(df_l1, x="Ratio", y="L1", color="cluster",
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
    df_l2["cluster"] = y_kmeans.astype(str)
    df_l2["compressor"] = clean_arr_l2[:, 2]
    df_l2["filter"] = clean_arr_l2[:, 3]
    df_l2["serializer"] = clean_arr_l2[:, 4]
    fig_l2 = px.scatter(df_l2, x="Ratio", y="L2", color="cluster",
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
    df_linf["cluster"] = y_kmeans.astype(str)
    df_linf["compressor"] = clean_arr_linf[:, 2]
    df_linf["filter"] = clean_arr_linf[:, 3]
    df_linf["serializer"] = clean_arr_linf[:, 4]
    fig_linf = px.scatter(df_linf, x="Ratio", y="LInf", color="cluster",
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
    df_dwt["cluster"] = y_kmeans.astype(str)
    df_dwt["compressor"] = clean_arr_dwt[:, 2]
    df_dwt["filter"] = clean_arr_dwt[:, 3]
    df_dwt["serializer"] = clean_arr_dwt[:, 4]
    fig_dwt = px.scatter(df_dwt, x="Ratio", y="DWT", color="cluster",
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

    fig.update_layout(title="", showlegend=False)
    pio.renderers.default = "browser"
    fig.show()


@cli.command("models_evaluation")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def models_evaluation(netcdf_file: str, field_to_compress: str, parameters_file: str):
    # Array slicing
    ds = utils.open_netcdf(netcdf_file, field_to_compress)
    ds_coords = dict(ds.coords).keys()
    lat_key = [key for key in ds_coords if key.startswith("lat")][0]
    lon_key = [key for key in ds_coords if key.startswith("lon")][0]
    lat_upper = round(ds[field_to_compress][lat_key].shape[0] * 0.2)
    lon_upper = round(ds[field_to_compress][lon_key].shape[0] * 0.2)
    ds = ds.isel(latitude=slice(0, lat_upper), longitude=slice(0, lon_upper))
    dwt_dists = {}

    # linear_quantization_zlib_compressors
    linear_quantization_bits, zlib_level = utils.get_filter_parameters(
        parameters_file, "linear_quantization_zlib_compressors"
    )
    ds_linquant = {}
    linquant_compressor = CodecStack(
        LinearQuantize(bits=linear_quantization_bits, dtype=str(ds[field_to_compress].dtype)),
        Zlib(level=zlib_level),
    )
    ds_linquant[field_to_compress] = linquant_compressor.encode_decode_data_array(
        ds[field_to_compress]
    ).compute()
    lquant_dwt_distances = utils.calc_dwt_dist(
        ds_linquant[field_to_compress], ds[field_to_compress], n_levels=4
    )
    dwt_dists["lquant_dwt_distances"] = lquant_dwt_distances
    print("Linear Quantization DWT disctance: " + f"{lquant_dwt_distances}")

    # bitround_zlib_compressors
    bitround_bits, zlib_level = utils.get_filter_parameters(
        parameters_file, "bitround_zlib_compressors"
    )
    ds_bitround = {}
    bitround_compressor = CodecStack(
        BitRound(keepbits=bitround_bits),
        Zlib(level=zlib_level),
    )
    ds_bitround[field_to_compress] = bitround_compressor.encode_decode_data_array(
        ds[field_to_compress]
    ).compute()
    bitround_dwt_distances = utils.calc_dwt_dist(
        ds_bitround[field_to_compress], ds[field_to_compress], n_levels=4
    )
    dwt_dists["bitround_dwt_distances"] = bitround_dwt_distances
    print("Bit Round DWT disctance: " + f"{bitround_dwt_distances}")

    # zfp_asinh_compressors
    asinh_linear_width, zfp_mode, zfp_tolerance = utils.get_filter_parameters(
        parameters_file, "zfp_asinh_compressors"
    )
    ds_zfp = {}
    zfp_compressor = CodecStack(
        Asinh(linear_width=asinh_linear_width),
        Zfp(mode=zfp_mode, tolerance=zfp_tolerance),
    )
    ds_zfp[field_to_compress] = zfp_compressor.encode_decode_data_array(
        ds[field_to_compress]
    ).compute()
    zfp_dwt_distances = utils.calc_dwt_dist(ds_zfp[field_to_compress], ds[field_to_compress], n_levels=4)
    dwt_dists["zfp_dwt_distances"] = zfp_dwt_distances
    print("Zfp DWT disctance: " + f"{zfp_dwt_distances}")

    # sz3_eb_compressors
    sz3_eb_mode, sz3_eb_rel = utils.get_filter_parameters(parameters_file, "sz3_eb_compressors")
    ds_sz3 = {}
    sz3_compressor = CodecStack(Sz3(eb_mode=sz3_eb_mode, eb_rel=sz3_eb_rel))
    ds_sz3[field_to_compress] = sz3_compressor.encode_decode_data_array(
        ds[field_to_compress]
    ).compute()
    sz3_dwt_distances = utils.calc_dwt_dist(ds_sz3[field_to_compress], ds[field_to_compress], n_levels=4)
    dwt_dists["sz3_eb_compressors"] = sz3_dwt_distances
    print("SZ3 DWT disctance: " + f"{sz3_dwt_distances}")

    min_key = min(dwt_dists, key=dwt_dists.get)
    print("Best compression method: " + f"{min_key}")


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
