
import importlib.util
import subprocess
import sys

# PyQt6 might encounter issues when installed from pyproject.toml
print("Installing PyQt6")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "PyQt6", "--only-binary", ":all:"
])

import sys
import os
import subprocess
import tempfile

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QComboBox, QProgressBar, QTextEdit, QFormLayout,
                             QSpinBox, QToolButton, QFrame, QFileDialog, QMessageBox)

import xarray as xr

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dc_toolkit import utils
import zipfile


def load_scored_results(file_name: str):
    return np.load(file_name + "_scored_results_with_names.npy", allow_pickle=True)

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
            showlegend=False,
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
    fig.show()

def load_and_resize_netcdf(ds, file_path, max_size_bytes=1e7):
    file_bytes_size= os.path.getsize(file_path)
    if file_bytes_size > max_size_bytes:
        percent_size = max_size_bytes / file_bytes_size
        for coord in list(ds.coords):
            if len(ds[coord]) > 1:
                ds = ds.isel({coord: slice(0, int(ds[coord].size * percent_size))})

    return ds

class CompressorThread(QThread):
    progress = pyqtSignal(str)
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd

    def run(self):
        with subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        ) as proc:
            for line in proc.stdout:
                self.progress.emit(line)
            proc.wait()

        where_am_i = subprocess.run(["uname", "-a"], capture_output=True, text=True)
        file_name = self.cmd[3] if "santis" in where_am_i.stdout.strip() else self.cmd[5]
        scored_results = load_scored_results(os.path.basename(file_name))
        scored_results_pd = pd.DataFrame(scored_results)

        numeric_cols = scored_results_pd.select_dtypes(include=[np.number]).columns
        mask = np.isfinite(scored_results_pd[numeric_cols]).all(axis=1)
        scored_results_pd = scored_results_pd[mask].dropna()

        clean_arr_l1 = utils.slice_array(scored_results_pd, [0, 1, 5, 6, 7])
        clean_arr_l2 = utils.slice_array(scored_results_pd, [0, 2, 5, 6, 7])
        clean_arr_linf = utils.slice_array(scored_results_pd, [0, 3, 5, 6, 7])

        create_cluster_plots(
            clean_arr_l1, clean_arr_l2, clean_arr_linf
        )

        self.finished.emit()


class CompressionAnalysisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Evaluate and compress netCDF files")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        self.open_button = QPushButton("Upload netCDF file")
        self.open_button.clicked.connect(self.open_file)
        self.main_layout.addWidget(self.open_button)

        self.file_label = QLabel("No file selected")
        self.main_layout.addWidget(self.file_label)

        self.field_selection = QComboBox()
        self.main_layout.addWidget(QLabel("Select field to compress:"))
        self.main_layout.addWidget(self.field_selection)

        self.toggle_button = QToolButton(self)
        self.toggle_button.setText("Advanced Options")
        self.toggle_button.setCheckable(True)
        self.main_layout.addWidget(self.toggle_button)

        self.panel_frame = QFrame(self)
        self.panel_frame.setVisible(False)
        self.panel_layout = QVBoxLayout(self.panel_frame)
        self.panel_layout.setContentsMargins(0, 0, 0, 0)

        self.panel_layout.addWidget(QLabel("Choose a compressor:"))
        self.options_compressor = QComboBox()
        self.options_compressor.addItems(["", "Blosc", "LZ4", "Zstd", "Zlib", "GZip", "BZ2", "LZMA", "None"])
        self.panel_layout.addWidget(self.options_compressor)

        self.panel_layout.addWidget(QLabel("Choose a filter:"))
        self.options_filter = QComboBox()
        self.options_filter.addItems(["", "Delta", "BitRound", "Quantize", "Asinh", "FixedOffsetScale", "None"])
        self.panel_layout.addWidget(self.options_filter)

        self.panel_layout.addWidget(QLabel("Choose a serializer:"))
        self.options_serializer = QComboBox()
        self.options_serializer.addItems(["", "PCodec", "ZFPY", "EBCCZarrFilter", "Zfp", "Sperr", "Sz3", "None"])
        self.panel_layout.addWidget(self.options_serializer)

        self.panel_layout.addWidget(QLabel("Set max L1 error:"))
        self.options_l1_error = QComboBox()
        self.options_l1_error.addItems(["Pre-defined"] + [str(x) for x in np.arange(0., 1., 0.1)])
        self.panel_layout.addWidget(self.options_l1_error)

        self.main_layout.addWidget(self.panel_frame)
        self.toggle_button.toggled.connect(self.toggle_compression)

        self.analyze_button = QPushButton("Analyze Compressors")
        self.analyze_button.clicked.connect(self.analyze_compressors)
        self.analyze_button.setEnabled(False)
        self.main_layout.addWidget(self.analyze_button)

        self.progress_bar = QProgressBar()
        self.main_layout.addWidget(self.progress_bar)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.main_layout.addWidget(self.log)

        form_layout = QFormLayout()

        self.comp_idx_spin = QSpinBox()
        self.comp_idx_spin.setRange(0, 79)
        self.comp_idx_spin.setValue(10)

        self.filt_idx_spin = QSpinBox()
        self.filt_idx_spin.setRange(0, 16)
        self.filt_idx_spin.setValue(10)

        self.ser_idx_spin = QSpinBox()
        self.ser_idx_spin.setRange(0, 34)
        self.ser_idx_spin.setValue(10)

        form_layout.addRow("Compression index:", self.comp_idx_spin)
        form_layout.addRow("Filter index:", self.filt_idx_spin)
        form_layout.addRow("Serializer index:", self.ser_idx_spin)
        self.main_layout.addLayout(form_layout)

        self.compress_button = QPushButton("Compress File")
        self.compress_button.clicked.connect(self.compress_file)
        self.compress_button.setEnabled(False)
        self.main_layout.addWidget(self.compress_button)

        self.file_path = None
        self.modified_file_path = None
        self.thread = None
        self.file_name = None

    def set_advanced_option_max(self, advnced_option_input, func):
        selected_field = self.field_selection.currentText()
        da = xr.open_dataset(self.file_path)[selected_field]
        return len(func(da, advnced_option_input))

    def toggle_compression(self, checked):
        self.panel_frame.setVisible(checked)

    def retrieve_file(self):
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            self.modified_file_path = tmp.name
            ds = xr.open_dataset(self.file_path)
            ds.to_netcdf(self.modified_file_path)
            ds.close()

    def open_file(self):
        file_dialog = QFileDialog()
        path, _ = file_dialog.getOpenFileName(self, "Upload netCDF file", "", "NetCDF files (*.nc)")
        if path:
            self.file_path = path
            self.file_label.setText(f"Selected file: {path}")
            self.log.append(f"Opened file: {path}")

            ds = xr.open_dataset(self.file_path)
            self.file_name = os.path.basename(self.file_path)
            variables = list(ds.data_vars.keys())
            ds.close()
            self.field_selection.clear()
            if variables:
                self.field_selection.addItems(variables)
                self.analyze_button.setEnabled(True)
                self.compress_button.setEnabled(True)
            else:
                self.log.append("No variables found.")
                self.analyze_button.setEnabled(False)
                self.compress_button.setEnabled(False)

    def analyze_compressors(self):
        selected_var = self.field_selection.currentText()
        self.retrieve_file()
        ds = load_and_resize_netcdf(xr.open_dataset(self.file_path), self.file_path)
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            path_to_modified_file = tmp.name
            ds.to_netcdf(path_to_modified_file)

        compressor_class = self.options_compressor.currentText()
        filter_class = self.options_filter.currentText()
        serializer_class = self.options_serializer.currentText()
        l1_error_class = "" if self.options_l1_error.currentText() == "Pre-defined" else self.options_l1_error.currentText()


        cmd = [
            "mpirun",
            "-n",
            "8",
            "dc_toolkit",
            "evaluate_combos",
            self.modified_file_path,
            os.getcwd(),
            "--field-to-compress=" + selected_var,
            "--compressor-class=" + compressor_class,
            "--filter-class=" + filter_class,
            "--serializer-class=" + serializer_class,
            "--override-existing-l1-error=" + l1_error_class
        ]

        self.thread = CompressorThread(cmd)
        self.thread.progress.connect(self.update_progress)
        self.thread.log.connect(self.log.append)
        self.main_dir_files = os.listdir(os.getcwd())
        self.thread.finished.connect(self.analysis_finished)
        self.thread.start()

        self.analyze_button.setEnabled(False)
        self.compress_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log.append("Initializing compressors anaylsis...")

    def update_progress(self, pct):
        self.log.append(f"{pct}")

    def analysis_finished(self):
        self.log.append("Analysis finished.")
        new_files = set(os.listdir(os.getcwd())) - set(self.main_dir_files)
        for file in new_files:
            if "scored_results" not in file:
                os.remove(file)
        self.main_dir_files = new_files
        self.analyze_button.setEnabled(True)
        self.compress_button.setEnabled(True)

    def show_popup(self, param, input_value, max_value):
        msg_box = QMessageBox(self)
        msg_box.setText(f"{param} input value {input_value} is higher than the max allowed {max_value}")
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def compress_file(self):
        self.retrieve_file()
        max_compressor_value = self.set_advanced_option_max(self.options_compressor.currentText(), utils.compressor_space)
        check_compx_val = self.comp_idx_spin.value() <= max_compressor_value

        max_filter_value = self.set_advanced_option_max(self.options_filter.currentText(), utils.filter_space)
        check_filter_val = self.filt_idx_spin.value() <= max_filter_value

        max_serializer_value = self.set_advanced_option_max(self.options_serializer.currentText(), utils.serializer_space)
        check_serializer_val = self.ser_idx_spin.value() <= max_serializer_value

        if not check_compx_val:
            param = "Compressor"
            self.show_popup(param, self.comp_idx_spin.value(), max_compressor_value)
        elif not check_filter_val:
            param = "Filter"
            self.show_popup(param, self.filt_idx_spin.value(), max_filter_value)
        elif not check_serializer_val:
            param = "Serializer"
            self.show_popup(param, self.ser_idx_spin.value(), max_serializer_value)
        else:
            selected_field = self.field_selection.currentText()

            da = xr.open_dataset(self.modified_file_path)[selected_field]

            compressor_class = self.options_compressor.currentText()
            filter_class = self.options_filter.currentText()
            serializer_class = self.options_serializer.currentText()

            compressors = utils.compressor_space(da, compressor_class)
            filters = utils.filter_space(da, filter_class)
            serializers = utils.serializer_space(da, serializer_class)
            self.max_value = len(compressors)
            self.comp_idx_spin.setMaximum(len(compressors))
            self.filt_idx_spin.setMaximum(len(filters))
            self.ser_idx_spin.setMaximum(len(serializers))

            comp_idx = self.comp_idx_spin.value()
            filt_idx = self.filt_idx_spin.value()
            ser_idx = self.ser_idx_spin.value()

            temp_dir = os.path.dirname(self.modified_file_path)

            cmd = [
                "dc_toolkit",
                "compress_with_optimal",
                self.modified_file_path,
                temp_dir,
                selected_field,
                str(comp_idx), str(filt_idx), str(ser_idx),
                "--compressor-class=" + compressor_class,
                "--filter-class=" + filter_class,
                "--serializer-class=" + serializer_class

            ]

            self.log.append(f"Running compression with parameters: comp_idx={comp_idx}, filt_idx={filt_idx}, ser_idx={ser_idx}")
            before = set(os.listdir(temp_dir))
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.log.append("Compression completed successfully.")
                after = set(os.listdir(temp_dir))
                generated_files = list(after - before)
                output_file = os.path.join(temp_dir, generated_files[0])

                split_tmp_name = os.path.basename(output_file).split(".=.", 1)
                compressed_file_name = f"{self.file_name}.=.{split_tmp_name[1]}"
                self.log.append(f"Generated file: {compressed_file_name}")

                save_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "save as zip file",
                    compressed_file_name,
                    "ZIP Archives (*.zip);;All Files (*)"
                )

                with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(output_file, arcname=os.path.basename(output_file))
                self.log.append(f"zip file saved to: {save_path}")
                os.remove(output_file)

            except subprocess.CalledProcessError as e:
                self.log.append(f"Compression failed:\n{e.stderr}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CompressionAnalysisUI()
    window.resize(700, 600)
    window.show()
    sys.exit(app.exec())
