# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import glob
import os
import re
import subprocess
import tempfile
import streamlit as st
from data_compression_cscs_exclaim import utils
from ebcc.zarr_filter import EBCCZarrFilter
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numcodecs_wasm_sperr import Sperr
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray
from io import BytesIO


# Page title
st.title("Upload a file and evaluate compressors")

uploaded_file = st.file_uploader("Choose a netcdf file")

@st.cache_resource
def create_cluster_plots(clean_arr_l1, clean_arr_l2, clean_arr_linf, clean_arr_dwt):
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
    clean_arr_linf_filtered = np.column_stack(
        (clean_arr_linf[:, 0].astype(float), clean_arr_linf[:, 1].astype(float)))
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

    return fig

if uploaded_file is not None and uploaded_file.name.endswith(".nc"):
    netcdf_file = uploaded_file
    netcdf_file_xr = xarray.open_dataset(netcdf_file)
    if uploaded_file.size > 1e7:
        percent_size = 1e7 / uploaded_file.size

        bytes_io = BytesIO(uploaded_file.read())
        bytes_io.seek(0)
        ds = xarray.open_dataset(bytes_io)

        for coord in list(ds.coords):
            if len(ds[coord]) > 1:
                ds = ds.isel({coord: slice(0, int(ds[coord].size * percent_size))})

        netcdf_file_xr = ds
        uploaded_file.name = uploaded_file.name + "_reduced.nc"
        ds.to_netcdf(uploaded_file.name)

    options = [var for var in netcdf_file_xr.data_vars]
    if "selected_column" not in st.session_state:
        st.session_state.selected_column = options[0]
    field_to_compress = st.selectbox(
        "Choose a field:",
        options=options,
        index=options.index(st.session_state.selected_column),
        key="selected_column",
    )
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        path_to_modified_file = tmp.name
        netcdf_file_xr.to_netcdf(path_to_modified_file)

    if st.button("Analyze compressors"):
        cmd_compress = [
            "mpirun",
            "-n",
            "8",
            "data_compression_cscs_exclaim",
            "summarize_compression",
            path_to_modified_file,
            field_to_compress
        ]

        st.info("Analyzing compressors...")
        progress_bar = st.progress(0)
        progress_text = st.empty()
        progress_text_1 = st.empty()
        log_box = st.empty()

        total_steps = 37635
        current_max = 0
        full_log = ""

        with subprocess.Popen(
            cmd_compress,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        ) as process:
            for line in process.stdout:
                match = re.search(r"Rank \d+:\s+.*?(\d+)/(\d+)", line)
                if match:
                    current_max = max(current_max, int(match.group(1)))

                    percent = int(100 * int(match.group(2)) / current_max)
                    progress_bar.progress(percent)
                    progress_text.text(f"{percent}%")
                    progress_text_1.text(f"{total_steps}/{current_max}")


        scored_results = np.load("scored_results_with_names.npy", allow_pickle=True)

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
        fig = create_cluster_plots(clean_arr_l1, clean_arr_l2, clean_arr_linf, clean_arr_dwt)
        fig.update_layout(title="", showlegend=False, width=500, height=1000)
        st.plotly_chart(fig, use_container_width=True)

    comp_idx = st.number_input('comp_idx', min_value=0, max_value=65, value=10)
    filt_idx = st.number_input('filt_idx', min_value=0, max_value=24, value=10)
    ser_idx = st.number_input('ser_idx', min_value=0, max_value=193, value=10)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    if st.button("Compress file"):
        cmd_compress = [
            "data_compression_cscs_exclaim",
            "compress_with_optimal",
            path_to_modified_file,
            field_to_compress,
            str(comp_idx), str(filt_idx), str(ser_idx)
        ]
        try:
            temp_dir = os.path.dirname(path_to_modified_file)
            before = set(os.listdir(temp_dir))
            result = subprocess.run(cmd_compress, capture_output=True, text=True, check=True)
            st.success("Compression completed successfully")
            after = set(os.listdir(temp_dir))
            generated_file_name = list(after - before)[0]
            output_file_path = os.path.join(temp_dir, generated_file_name)
            with open(output_file_path, "rb") as data_file:
                st.download_button(
                    label="Download Compressed File",
                    data=data_file,
                    file_name=os.path.basename(output_file_path),
                )
        except subprocess.CalledProcessError as e:
            st.error("Compression failed")
            st.code(e.stderr)
