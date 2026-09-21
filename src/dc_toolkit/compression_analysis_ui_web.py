"""
Streamlit UI: upload a netCDF file, sweep the codec space on one field, look
at the results, write the field with a chosen pipeline and download the store.

Launched by `dc_toolkit run_web_ui` (sweeps run as one local MPI rank) or by
`dc_toolkit run_web_ui_vcluster` (sweeps run under srun; the extra arguments
select the allocation and name the file on the cluster).
"""
import argparse
import json
import os
import subprocess
import tempfile
from io import BytesIO

import streamlit as st
import xarray as xr

from dc_toolkit import utils_cli

OUT_DIR = "out"


def parse_args():
    parser = argparse.ArgumentParser()
    for name in ("user_account", "uenv_image", "uploaded_file", "time", "nodes", "ntasks-per-node", "partition"):
        parser.add_argument(f"--{name}", type=str, default=None)
    return parser.parse_known_args()[0]  # streamlit may add its own argv


@st.cache_data
def load_and_resize_netcdf(file_content, original_name, max_size_bytes=1e7):
    """Open the upload; above max_size_bytes, keep a leading block of every
    dimension so the interactive sweep stays quick."""
    ds = xr.open_dataset(BytesIO(file_content))
    if len(file_content) > max_size_bytes:
        dims = [d for d in ds.dims if ds.sizes[d] > 1]
        scale = (max_size_bytes / len(file_content)) ** (1 / max(1, len(dims)))
        ds = ds.isel({d: slice(0, max(1, int(ds.sizes[d] * scale))) for d in dims})
        original_name += "_reduced.nc"
    return ds, original_name


def run_streaming(cmd, status):
    """Run a command and mirror its output lines into a streamlit placeholder."""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                          env=utils_cli.ui_env()) as proc:
        for line in proc.stdout:
            status.text(line.rstrip())
    return proc.returncode


args = parse_args()
launcher = utils_cli.ui_launcher(args.user_account, args.time, args.nodes, args.ntasks_per_node, args.uenv_image,
                                 args.partition)
compress_launcher = utils_cli.ui_launcher(args.user_account, args.time, "1", "1", args.uenv_image,
                                          args.partition)  # compress is a single-process command
st.title("Evaluate compressors and compress a netCDF field")

if args.uploaded_file:  # vcluster: the file already lives on the cluster
    dataset_path, ds, source = args.uploaded_file, xr.open_dataset(args.uploaded_file), args.uploaded_file
else:
    uploaded = st.file_uploader("Choose a netCDF file", type=["nc"])
    if uploaded is None:
        st.stop()
    ds, name = load_and_resize_netcdf(uploaded.read(), uploaded.name)
    dataset_path, source = os.path.join(tempfile.gettempdir(), name), uploaded.file_id
    ds.to_netcdf(dataset_path)

if not list(ds.data_vars):
    st.warning("This file has no data variables to compress.")
    st.stop()
field = st.selectbox("Field to compress", list(ds.data_vars))
with st.expander("Codec space", expanded=True):
    col1, col2, col3 = st.columns(3)
    classes = {"compressor": col1.selectbox("Compressor class", utils_cli.UI_CLASS_OPTIONS["compressor"]),
               "filter": col2.selectbox("Filter class", utils_cli.UI_CLASS_OPTIONS["filter"]),
               "serializer": col3.selectbox("Serializer class", utils_cli.UI_CLASS_OPTIONS["serializer"])}
    with_lossy = st.checkbox("Include lossy codecs", value=True)
    with_ebcc = st.checkbox("Add EBCC (optional package; float lat/lon frames only)", value=False)
    l1_threshold = st.number_input("Relative L1 error budget", min_value=1e-10, max_value=1.0,
                                   value=utils_cli.UI_DEFAULT_L1, format="%.6f")

os.makedirs(OUT_DIR, exist_ok=True)
current = (source, field)  # results and archives belong to one upload and one field
if st.button("Evaluate combos"):
    st.session_state.pop("archive", None)
    status = st.empty()
    status.info("Sweeping the codec space ...")
    rc = run_streaming(utils_cli.ui_sweep_command(launcher, dataset_path, OUT_DIR, field, classes,
                                                  with_lossy, with_ebcc, l1_threshold), status)
    status.empty()
    if rc != 0:
        st.error(f"evaluate_combos failed with exit code {rc}.")
        st.session_state.pop("swept", None)
    else:
        st.session_state["swept"] = current

results = utils_cli.ui_results(OUT_DIR, field) if st.session_state.get("swept") == current else None
if results is not None:
    if results.empty:
        st.warning("No combination passed the gates; loosen the error budget and sweep again.")
        st.stop()
    st.subheader(f"{len(results)} combinations passed the gates (best ratio first)")
    st.dataframe(results[["name", "ratio", "l1_rel", "l2_rel", "linf_rel"]].head(50), hide_index=True)
    fig = utils_cli.clustering_figure(results)
    st.plotly_chart(fig)
    st.download_button("Download the plots as HTML", fig.to_html(full_html=True, include_plotlyjs="cdn"),
                       file_name=f"cluster_plots_{field}.html", mime="text/html")

    choice = st.selectbox("Pipeline to write", list(results["name"]))
    if st.button("Compress field"):
        st.session_state.pop("archive", None)
        pipeline = json.loads(results.loc[results["name"] == choice, "pipeline"].iloc[0])
        status = st.empty()
        status.info("Compressing ...")
        rc = run_streaming(utils_cli.ui_compress_command(compress_launcher, dataset_path, OUT_DIR, field, pipeline),
                           status)
        status.empty()
        store = utils_cli.merged_store_path(OUT_DIR, dataset_path)
        if rc != 0 or not os.path.isdir(store):
            st.error(f"compress failed with exit code {rc}.")
        else:
            st.session_state["archive"] = (current, utils_cli.zip_directory(store), choice)
    archived = st.session_state.get("archive")
    if archived and archived[0] == current:  # outside the button branch: it survives the next rerun
        st.success(f"Wrote {field} with {archived[2]}.")
        with open(archived[1], "rb") as fh:
            st.download_button("Download the compressed store (zip)", fh.read(),
                               file_name=os.path.basename(archived[1]))
