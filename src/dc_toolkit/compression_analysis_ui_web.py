# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import math
import os
import re
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

def find_latest_file(folder_path, filename_prefix):
    latest_file = None
    latest_mtime = -1

    for filename in os.listdir(folder_path):
        if filename.startswith(filename_prefix):
            full_path = os.path.join(folder_path, filename)
            if os.path.isfile(full_path):
                mtime = os.path.getmtime(full_path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_file = full_path

    return latest_file

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
def load_scored_results(file_name: str, params_str: list[str]):
    return np.load(file_name + params_str + "_scored_results_with_names.npy", allow_pickle=True)


@st.cache_resource
def create_cluster_plots(clean_arr_l1, clean_arr_l2, clean_arr_linf, n_clusters):
    config_idxs = pd.read_csv("config_space.csv")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")

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
    df_l1["compressor_idx"] = utils.get_indexes(clean_arr_l1[:, 2], config_idxs['0'])
    df_l1["filter_idx"] = utils.get_indexes(clean_arr_l1[:, 3], config_idxs['1'])
    df_l1["serializer_idx"] = utils.get_indexes(clean_arr_l1[:, 4], config_idxs['2'])
    # To account for overlapping cases
    df_l1["Ratio"] = df_l1["Ratio"] + np.random.normal(0, 0.00001, size=len(df_l1))

    y_kmeans = kmeans.fit_predict(pd.DataFrame(df_l1, columns=["Ratio", "L1"]))
    color = np.ones(y_kmeans.shape) if len(np.unique(y_kmeans)) == 1 else y_kmeans

    fig_l1 = px.scatter(df_l1, x="Ratio", y="L1", color=color,
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

    df_l2 = pd.DataFrame(clean_arr_l2_filtered, columns=["Ratio", "L2"])
    df_l2["compressor"] = clean_arr_l2[:, 2]
    df_l2["filter"] = clean_arr_l2[:, 3]
    df_l2["serializer"] = clean_arr_l2[:, 4]
    df_l2["compressor_idx"] = utils.get_indexes(clean_arr_l2[:, 2], config_idxs['0'])
    df_l2["filter_idx"] = utils.get_indexes(clean_arr_l2[:, 3], config_idxs['1'])
    df_l2["serializer_idx"] = utils.get_indexes(clean_arr_l2[:, 4], config_idxs['2'])
    # To account for overlapping cases
    df_l2["Ratio"] = df_l2["Ratio"] + np.random.normal(0, 0.00001, size=len(df_l2))

    y_kmeans = kmeans.fit_predict(pd.DataFrame(df_l2, columns=["Ratio", "L2"]))
    color = np.ones(y_kmeans.shape) if len(np.unique(y_kmeans)) == 1 else y_kmeans

    fig_l2 = px.scatter(df_l2, x="Ratio", y="L2", color=color,
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

    df_linf = pd.DataFrame(clean_arr_linf_filtered, columns=["Ratio", "LInf"])
    df_linf["compressor"] = clean_arr_linf[:, 2]
    df_linf["filter"] = clean_arr_linf[:, 3]
    df_linf["serializer"] = clean_arr_linf[:, 4]
    df_linf["compressor_idx"] = utils.get_indexes(clean_arr_linf[:, 2], config_idxs['0'])
    df_linf["filter_idx"] = utils.get_indexes(clean_arr_linf[:, 3], config_idxs['1'])
    df_linf["serializer_idx"] = utils.get_indexes(clean_arr_linf[:, 4], config_idxs['2'])
    # To account for overlapping cases
    df_linf["Ratio"] = df_linf["Ratio"] + np.random.normal(0, 0.00001, size=len(df_linf))

    y_kmeans = kmeans.fit_predict(pd.DataFrame(df_linf, columns=["Ratio", "LInf"]))
    color = np.ones(y_kmeans.shape) if len(np.unique(y_kmeans)) == 1 else y_kmeans

    fig_linf = px.scatter(df_linf, x="Ratio", y="LInf", color=color,
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
    if "selected_column_to_compress" not in st.session_state:
        st.session_state.selected_column_to_compress = options_field[0]

    field_to_compress = st.selectbox(
        "Choose a field:",
        options=options_field,
        index=options_field.index(st.session_state.selected_column_to_compress),
        key="selected_column_to_compress",
    )

    compressor_class = ""
    filter_class = ""
    serializer_class = ""
    l1_error_class = ""

    st.session_state.expander_state = True

    with st.expander("Advanced Selection", expanded=st.session_state.expander_state):
        options_compressor = ["all", "Blosc", "LZ4", "Zstd", "Zlib", "GZip", "BZ2", "LZMA", "None"]
        options_filter = ["all", "Delta", "BitRound", "Quantize", "Asinh", "FixedOffsetScale", "None"]
        options_serializer = ["all", "PCodec", "ZFPY", "EBCCZarrFilter", "Zfp", "Sperr", "Sz3", "None"]
        options_with = ["with", "without"]

        col1, col2, col3 = st.columns(3)

        with col1:
            compressor_class = st.selectbox(
                "Choose a compressor:",
                options=options_compressor,
            )
            lossy_class = st.selectbox(
                "Lossy:",
                options=options_with,
            )

        with col2:
            filter_class = st.selectbox(
                "Choose a filter:",
                options=options_filter,
            )
            numcodecs_wasm_class = st.selectbox(
                "Numcodecs-wasm:",
                options=options_with,
            )

        with col3:
            serializer_class = st.selectbox(
                "Choose a serializer:",
                options=options_serializer,
            )
            ebcc_class = st.selectbox(
                "EBCC:",
                options=options_with,
            )


        predefined_l1 = st.checkbox("Use pre-defined l1 error", value=True)
        l1_error_class = st.number_input('l1_error_class', min_value=0.0, max_value=1.0, value=0.0, step=0.0000000001, format="%.10f", disabled=predefined_l1)

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        path_to_modified_file = tmp.name
        netcdf_file_xr.to_netcdf(path_to_modified_file)
        netcdf_file_xr.close()

    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
        st.session_state.analysis_performed = False
        st.session_state.temp_plot_file = None

    with_lossy_option = "--with-lossy" if lossy_class == "with" else "--without-lossy"
    with_numcodecs_option = "--with-numcodecs-wasm" if numcodecs_wasm_class == "with" else "--without-numcodecs-wasm"
    with_ebcc_option = "--with-ebcc" if ebcc_class == "with" else "--without-ebcc"
    if st.button("Analyze compressors"):
        if "santis" in where_am_i.stdout.strip():
            if predefined_l1:
                cmd_compress = [
                    "srun",
                    "-A", parse_args().user_account,
                    "--time", parse_args().time,
                    "--nodes", parse_args().nodes,
                    "--ntasks-per-node", parse_args().ntasks_per_node,
                    "--uenv=prgenv-gnu/24.11:v2",
                    "--view=default",
                    "--partition=debug",
                    "dc_toolkit",
                    "evaluate_combos",
                    display_file_name,
                    os.getcwd(),
                    "--field-to-compress=" + field_to_compress,
                    "--compressor-class=" + compressor_class,
                    "--filter-class=" + filter_class,
                    "--serializer-class=" + serializer_class,
                    *[with_lossy_option, with_numcodecs_option, with_ebcc_option]
                ]
            else:
                cmd_compress = [
                    "srun",
                    "-A", parse_args().user_account,
                    "--time", parse_args().time,
                    "--nodes", parse_args().nodes,
                    "--ntasks-per-node", parse_args().ntasks_per_node,
                    "--uenv=prgenv-gnu/24.11:v2",
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
                    "--override-existing-l1-error=" + str(l1_error_class),
                    *[with_lossy_option, with_numcodecs_option, with_ebcc_option]
                ]
        else:
            if predefined_l1:
                cmd_compress = [
                    "mpirun",
                    "-n",
                    "8",
                    "dc_toolkit",
                    "evaluate_combos",
                    display_file_name,
                    os.getcwd(),
                    "--field-to-compress="+field_to_compress,
                    "--compressor-class="+compressor_class,
                    "--filter-class="+filter_class,
                    "--serializer-class="+serializer_class,
                    *[with_lossy_option, with_numcodecs_option, with_ebcc_option]
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
                    "--field-to-compress=" + field_to_compress,
                    "--compressor-class=" + compressor_class,
                    "--filter-class=" + filter_class,
                    "--serializer-class=" + serializer_class,
                    "--override-existing-l1-error=" + str(l1_error_class),
                    *[with_lossy_option, with_numcodecs_option, with_ebcc_option]
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

        pattern = r'--.*?-?'
        with_lossy = re.sub(pattern, '', with_lossy_option, count=1)
        with_numcodesc_wasm = re.sub(pattern, '', with_numcodecs_option, count=1)
        with_ebcc = re.sub(pattern, '', with_ebcc_option, count=1)

        score_results_file_name = [field_to_compress, compressor_class, filter_class, serializer_class, with_lossy, with_numcodesc_wasm, with_ebcc]
        params_str = '_' + '_'.join(score_results_file_name)
        scored_results = load_scored_results(os.path.basename(path_to_modified_file), params_str)
        scored_results_pd = pd.DataFrame(scored_results)
        max_n_rows, max_nclusters = 42976, 6
        adjusted_n_clusters = math.ceil(max_nclusters*len(scored_results_pd)/max_n_rows)

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
            clean_arr_l1, clean_arr_l2, clean_arr_linf, adjusted_n_clusters
        )

        temp_dir = tempfile.gettempdir()
        temp_html_path = os.path.join(temp_dir, f"cluster_plots_{os.getpid()}{params_str}.html")
        os.makedirs(os.path.dirname(temp_html_path), exist_ok=True)
        fig_to_save.write_html(temp_html_path, full_html=True, include_plotlyjs='cdn')

        st.session_state.temp_plot_file = temp_html_path
        st.session_state.analysis_performed = True

        st.rerun()

    if st.session_state.analysis_performed and st.session_state.temp_plot_file:
        plot_file_path = st.session_state.temp_plot_file
        with_lossy = "with_lossy" if lossy_class == "with" else "without_lossy"
        with_numcodecs_wasm = "with_numcodecs_wasm" if numcodecs_wasm_class == "with" else "without_numcodecs_wasm"
        with_ebcc = "with_ebcc" if ebcc_class == "with" else "without_ebcc"
        cluster_results_file_name = [c for c in (field_to_compress, compressor_class, filter_class, serializer_class, with_lossy, with_numcodecs_wasm, with_ebcc)]
        params_str = '_' + '_'.join(cluster_results_file_name)

        with open(plot_file_path, "rb") as f:
            st.download_button(
                label="Download Plots as HTML",
                data=f,
                file_name=f"cluster_plots{params_str}.html",
                mime="text/html",
            )

    comp_idx = st.number_input('comp_idx', min_value=0, max_value=79, value=10)
    filt_idx = st.number_input('filt_idx', min_value=0, max_value=16, value=10)
    ser_idx = st.number_input('ser_idx', min_value=0, max_value=34, value=10)

    if st.button("Compress file"):
        da = xarray.open_dataset(path_to_modified_file)[field_to_compress]
        with_lossy = True if lossy_class == "with" else False
        with_numcodecs_wasm = True if numcodecs_wasm_class == "with" else False
        with_ebcc = True if ebcc_class == "with" else False

        compressors_options = utils.compressor_space(da=da, with_lossy=with_lossy,
                                                     with_numcodecs_wasm=with_numcodecs_wasm, with_ebcc=with_ebcc,
                                                     compressor_class=compressor_class)
        filters_options = utils.filter_space(da=da, with_lossy=with_lossy, with_numcodecs_wasm=with_numcodecs_wasm,
                                             with_ebcc=with_ebcc, filter_class=filter_class)
        serializers_options = utils.serializer_space(da=da, with_lossy=with_lossy,
                                                     with_numcodecs_wasm=with_numcodecs_wasm, with_ebcc=with_ebcc,
                                                     serializer_class=serializer_class)

        max_compressor_value = len(compressors_options)
        check_compx_val = comp_idx <= max_compressor_value

        max_filter_value = len(filters_options)
        check_filter_val = filt_idx <= max_filter_value

        max_serializer_value = len(serializers_options)
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
        else:
            if "santis" in where_am_i.stdout.strip():
                temp_dir = os.getcwd()
                cmd_compress = [
                    "srun",
                    "-A", parse_args().user_account,
                    "--time", parse_args().time,
                    "--nodes", parse_args().nodes,
                    "--ntasks-per-node", parse_args().ntasks_per_node,
                    "--uenv=prgenv-gnu/24.11:v2",
                    "--view=default",
                    "--partition=debug",
                    "dc_toolkit",
                    "compress_with_optimal",
                    display_file_name,
                    os.getcwd(),
                    field_to_compress,
                    str(comp_idx), str(filt_idx), str(ser_idx),
                    *[with_lossy_option, with_numcodecs_option, with_ebcc_option]
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
                    str(comp_idx), str(filt_idx), str(ser_idx),
                    *[with_lossy_option, with_numcodecs_option, with_ebcc_option]
                ]

            before = set(os.listdir(temp_dir))
            status = st.empty()
            status.info("Compressing file...")
            subprocess.run(cmd_compress)
            status.empty()
            st.success(f"Compression completed successfully.")
            if "santis" in where_am_i.stdout.strip():
                output_file_path = find_latest_file(os.getcwd(), os.path.basename(display_file_name))
                split_tmp_name = os.path.basename(display_file_name).split(".=.", 1)
                compressed_file_name = f"{os.path.basename(uploaded_file.name)}.=.{split_tmp_name[0]}"
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
