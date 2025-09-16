# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import os
import subprocess
import tempfile
import argparse
import shutil
import time

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray
from io import BytesIO

from dc_toolkit import utils

where_am_i = subprocess.run(["uname", "-a"], capture_output=True, text=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='default.csv')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--user_account', type=str, default=None)
    parser.add_argument('--uploaded_file', type=str, default=None)
    parser.add_argument('--time', type=str, default=None)
    parser.add_argument('--nodes', type=str, default=None)
    parser.add_argument('--ntasks-per-node', type=str, default=None)
    return parser.parse_args()

st.title("Upload a file and evaluate compressors")


if parse_args().uploaded_file is None:
    uploaded_file = st.file_uploader("Choose a netcdf file")
else:
    uploaded_file = open(parse_args().uploaded_file, "rb")

def find_file(base_path, file_name):
    for root, dirs, files in os.walk(base_path):
        if file_name in files:
            return os.path.join(root, file_name)
    return None

@st.cache_data
def load_scored_results(uploaded_file_name: str):
    return np.load(uploaded_file_name + "_scored_results_with_names.npy", allow_pickle=True)


@st.cache_resource
def create_cluster_plots(clean_arr_l1, clean_arr_l2, clean_arr_linf):
    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=[
                            "L1 VS Ratio KMeans Clustering", "L2 VS Ratio KMeans Clustering",
                            "LInf VS Ratio KMeans Clustering"
                        ])

    # L1 clustering
    clean_arr_l1_filtered = np.column_stack((clean_arr_l1[:, 0].astype(float), clean_arr_l1[:, 1].astype(float)))
    y_kmeans = kmeans.fit_predict(clean_arr_l1_filtered)
    df_l1 = pd.DataFrame(clean_arr_l1_filtered, columns=["Ratio", "L1"])
    df_l1["compressor"] = clean_arr_l1[:, 2]
    df_l1["filter"] = clean_arr_l1[:, 3]
    df_l1["serializer"] = clean_arr_l1[:, 4]
    df_l1["compressor_idx"] = np.arange(len(clean_arr_l1[:, 2]))
    df_l1["filter_idx"] = np.arange(len(clean_arr_l1[:, 3]))
    df_l1["serializer_idx"] = np.arange(len(clean_arr_l1[:, 4]))

    fig_l1 = px.scatter(df_l1, x="Ratio", y="L1", color=y_kmeans,
                        title="L1 VS Ratio KMeans Clustering", hover_data=["compressor", "filter", "serializer", "compressor_idx", "filter_idx", "serializer_idx"])

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
    y_kmeans = kmeans.fit_predict(clean_arr_l2_filtered)
    df_l2 = pd.DataFrame(clean_arr_l2_filtered, columns=["Ratio", "L2"])
    df_l2["compressor"] = clean_arr_l2[:, 2]
    df_l2["filter"] = clean_arr_l2[:, 3]
    df_l2["serializer"] = clean_arr_l2[:, 4]
    df_l2["compressor_idx"] = np.arange(len(clean_arr_l2[:, 2]))
    df_l2["filter_idx"] = np.arange(len(clean_arr_l2[:, 3]))
    df_l2["serializer_idx"] = np.arange(len(clean_arr_l2[:, 4]))

    fig_l2 = px.scatter(df_l2, x="Ratio", y="L2", color=y_kmeans,
                        title="L2 VS Ratio KMeans Clustering", hover_data=["compressor", "filter", "serializer", "compressor_idx", "filter_idx", "serializer_idx"])

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
    y_kmeans = kmeans.fit_predict(clean_arr_linf_filtered)
    df_linf = pd.DataFrame(clean_arr_linf_filtered, columns=["Ratio", "LInf"])
    df_linf["compressor"] = clean_arr_linf[:, 2]
    df_linf["filter"] = clean_arr_linf[:, 3]
    df_linf["serializer"] = clean_arr_linf[:, 4]
    df_linf["compressor_idx"] = np.arange(len(clean_arr_linf[:, 2]))
    df_linf["filter_idx"] = np.arange(len(clean_arr_linf[:, 3]))
    df_linf["serializer_idx"] = np.arange(len(clean_arr_linf[:, 4]))

    fig_linf = px.scatter(df_linf, x="Ratio", y="LInf", color=y_kmeans,
                          title="LInf VS Ratio KMeans Clustering", hover_data=["compressor", "filter", "serializer", "compressor_idx", "filter_idx", "serializer_idx"])

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
    return fig


@st.cache_data
def load_and_resize_netcdf(file_content, original_name, max_size_bytes=1e7):
    bytes_io = BytesIO(file_content)
    bytes_io.seek(0)
    ds = xarray.open_dataset(bytes_io)
    new_name = original_name

    if len(file_content) > max_size_bytes:
        percent_size = max_size_bytes / len(file_content)
        for coord in list(ds.coords):
            if len(ds[coord]) > 1:
                ds = ds.isel({coord: slice(0, int(ds[coord].size * percent_size))})
        new_name = original_name + "_reduced.nc"

    return ds, new_name


if uploaded_file is not None and uploaded_file.name.endswith(".nc"):
    file_content = uploaded_file.read()
    netcdf_file_xr, display_file_name = load_and_resize_netcdf(file_content, uploaded_file.name)

    options_field = [var for var in netcdf_file_xr.data_vars]
    if "selected_column" not in st.session_state:
        st.session_state.selected_column = options_field[0]

    field_to_compress = st.selectbox(
        "Choose a field:",
        options=options_field,
        index=options_field.index(st.session_state.selected_column),
        key="selected_column",
    )

    compressor_class = ""
    filter_class = ""
    serializer_class = ""
    l1_error_class = ""

    st.session_state.expander_state = True

    with st.expander("Advanced Selection", expanded=st.session_state.expander_state):
        options_compressor = ["", "Blosc", "LZ4", "Zstd", "Zlib", "GZip", "BZ2", "LZMA", "None"]
        compressor_class = st.selectbox(
            "Choose a compressor:",
            options=options_compressor,
        )

        options_filter = ["", "Delta", "BitRound", "Quantize", "Asinh", "FixedOffsetScale", "None"]
        filter_class = st.selectbox(
            "Choose a filter:",
            options=options_filter,
        )

        options_serializer = ["", "PCodec", "ZFPY", "EBCCZarrFilter", "Zfp", "Sperr", "Sz3", "None"]
        serializer_class = st.selectbox(
            "Choose a serialzer:",
            options=options_serializer,
        )

        options_l1_error = ["Pre-defined"] + [str(round(x, 1)) for x in np.arange(0., 1., 0.1)]
        l1_error_class = st.selectbox(
            "Set max L1 error:",
            options=options_l1_error,
        )

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        path_to_modified_file = tmp.name
        netcdf_file_xr.to_netcdf(path_to_modified_file)
        netcdf_file_xr.close()

    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
        st.session_state.analysis_performed = False
        st.session_state.temp_plot_file = None

    if st.button("Analyze compressors"):
        if "santis" in where_am_i.stdout.strip():
            cmd_compress = [
                "srun",
                "-A", parse_args().user_account,
                "--time", parse_args().time,
                "--nodes", parse_args().nodes,
                "--ntasks-per-node", parse_args().ntasks_per_node,
                "--uenv=prgenv-gnu/25.06:rc5",
                "--view=default",
                "--partition=debug",
                "dc_toolkit",
                "evaluate_combos",
                parse_args().uploaded_file,
                os.getcwd(),
                "--field-to-compress=" + field_to_compress,
                "--compressor-class=" + compressor_class,
                "--filter-class=" + filter_class,
                "--serializer-class=" + serializer_class,
                "--override-existing-l1-error=" + l1_error_class
            ]
        else:
            cmd_compress = [
                "mpirun",
                "-n",
                "8",
                "dc_toolkit",
                "evaluate_combos",
                path_to_modified_file,
                os.getcwd(),
                "--field-to-compress="+field_to_compress,
                "--compressor-class="+compressor_class,
                "--filter-class="+filter_class,
                "--serializer-class="+serializer_class,
                "--override-existing-l1-error=" + l1_error_class
            ]

        st.info("Analyzing compressors...")
        progress_text = st.empty()

        with subprocess.Popen(
            cmd_compress,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        ) as proc:
            for line in proc.stdout:
                progress_text.text(f"{line}")

        
        path_to_modified_file = display_file_name if "santis" in where_am_i.stdout.strip() else path_to_modified_file
        scored_results = load_scored_results(os.path.basename(path_to_modified_file))

        scored_results_pd = pd.DataFrame(scored_results)

        numeric_cols = scored_results_pd.select_dtypes(include=[np.number]).columns
        mask = np.isfinite(scored_results_pd[numeric_cols]).all(axis=1)
        scored_results_pd = scored_results_pd[mask].dropna()

        clean_arr_l1 = utils.slice_array(scored_results_pd, [0, 1, 5, 6, 7])
        clean_arr_l2 = utils.slice_array(scored_results_pd, [0, 2, 5, 6, 7])
        clean_arr_linf = utils.slice_array(scored_results_pd, [0, 3, 5, 6, 7])

        st.session_state.analysis_data = {
            "l1": clean_arr_l1,
            "l2": clean_arr_l2,
            "linf": clean_arr_linf,
        }

        fig_to_save = create_cluster_plots(
            clean_arr_l1, clean_arr_l2, clean_arr_linf
        )

        temp_dir = tempfile.gettempdir()
        temp_html_path = os.path.join(temp_dir, f"cluster_plots_{os.getpid()}.html")
        os.makedirs(os.path.dirname(temp_html_path), exist_ok=True)
        fig_to_save.write_html(temp_html_path, full_html=True, include_plotlyjs='cdn')

        st.session_state.temp_plot_file = temp_html_path
        st.session_state.analysis_performed = True

        st.rerun()

    if st.session_state.analysis_performed and st.session_state.temp_plot_file:
        plot_file_path = st.session_state.temp_plot_file

        with open(plot_file_path, "rb") as f:
            st.download_button(
                label="Download Plots as HTML",
                data=f,
                file_name="cluster_plots.html",
                mime="text/html",
            )

    comp_idx = st.number_input('comp_idx', min_value=1, max_value=79, value=10)
    filt_idx = st.number_input('filt_idx', min_value=1, max_value=16, value=10)
    ser_idx = st.number_input('ser_idx', min_value=1, max_value=34, value=10)

    if st.button("Compress file"):
        da = xarray.open_dataset(path_to_modified_file)[field_to_compress]

        max_compressor_value = len(utils.compressor_space(da, compressor_class))
        check_compx_val = comp_idx <= max_compressor_value

        max_filter_value = len(utils.filter_space(da, filter_class))
        check_filter_val = filt_idx <= max_filter_value

        max_serializer_value = len(utils.serializer_space(da, serializer_class))
        check_serializer_val = ser_idx <= max_serializer_value

        if not check_compx_val:
            placeholder = st.empty()
            style_str = f"""
            <div style='text-align: center; color: {"black"}; font-size: 1.2rem; margin-top: 10px; padding: 10px; border: 1px solid {"black"}; border-radius: 5px;'>
                ⚠️ Compressor input value {comp_idx} is higher than the max allowed {max_compressor_value}
            </div>
            """

            placeholder.markdown(style_str, unsafe_allow_html=True)
            time.sleep(4)
            placeholder.empty()
            #st.toast(f"Compressor input value {comp_idx} is higher than the max allowed {max_compressor_value}")
        elif not check_filter_val:
            placeholder = st.empty()
            style_str = f"""
            <div style='text-align: center; color: {"black"}; font-size: 1.2rem; margin-top: 10px; padding: 10px; border: 1px solid {"black"}; border-radius: 5px;'>
                ⚠️ Filter input value {filt_idx} is higher than the max allowed {max_filter_value}
            </div>
            """

            placeholder.markdown(style_str, unsafe_allow_html=True)
            time.sleep(4)
            placeholder.empty()
            #st.toast(f"Filter input value {filt_idx} is higher than the max allowed {max_filter_value}")
        elif not check_serializer_val:
            placeholder = st.empty()
            style_str = f"""
            <div style='text-align: center; color: {"black"}; font-size: 1.2rem; margin-top: 10px; padding: 10px; border: 1px solid {"black"}; border-radius: 5px;'>
                ⚠️ Serializer input value {ser_idx} is higher than the max allowed {max_serializer_value}
            </div>
            """

            placeholder.markdown(style_str, unsafe_allow_html=True)
            time.sleep(4)
            placeholder.empty()
            #st.toast(f"Serializer input value {ser_idx} is higher than the max allowed {max_serializer_value}")
        else:
            if "santis" in where_am_i.stdout.strip():
                temp_dir = os.getcwd()
                cmd_compress = [
                    "srun",
                    "-A", parse_args().user_account,
                    "--time", parse_args().t,
                    "--nodes", parse_args().nodes,
                    "--ntasks-per-node", parse_args().ntasks_per_node,
                    "--uenv=prgenv-gnu/25.06:rc5",
                    "--view=default",
                    "--partition=debug",
                    "dc_toolkit",
                    "compress_with_optimal",
                    path_to_modified_file,
                    os.getcwd(),
                    field_to_compress,
                    str(comp_idx), str(filt_idx), str(ser_idx)
                ]
            else:
                temp_dir = os.path.dirname(path_to_modified_file)
                cmd_compress = [
                    "mpirun",
                    "-n",
                    "8",
                    "dc_toolkit",
                    "compress_with_optimal",
                    path_to_modified_file,
                    temp_dir,
                    field_to_compress,
                    str(comp_idx), str(filt_idx), str(ser_idx)
                ]

            before = set(os.listdir(temp_dir))
            status = st.empty()
            status.info("Compressing file...")
            subprocess.run(cmd_compress)
            status.empty()
            st.success(f"Compression completed successfully. File saved in {os.getcwd()}")
            if "santis" in where_am_i.stdout.strip():
                after = set(os.listdir(temp_dir))
                output_file_path = os.path.join(temp_dir, list(after - before)[0])
                dest_path = os.path.join(temp_dir, list(after - before)[0])
                shutil.copy(output_file_path, dest_path)
                split_tmp_name = os.path.basename(output_file_path).split(".=.", 1)
                compressed_file_name = f"{uploaded_file.name}.=.{split_tmp_name[1]}"
            else:
                split_tmp_name = os.path.basename(path_to_modified_file).split(".=.", 1)
                compressed_file_name = f"{uploaded_file.name}.=.{split_tmp_name[0]}"
                shutil.copy(path_to_modified_file, os.getcwd())
                output_file_path = os.path.basename(path_to_modified_file)

            with open(output_file_path, "rb") as data_file:
                st.download_button(
                    label="Download compressed file locally",
                    data=data_file,
                    file_name=compressed_file_name,
                )
            os.remove(output_file_path)
            if os.path.exists(path_to_modified_file):
                os.remove(path_to_modified_file)
