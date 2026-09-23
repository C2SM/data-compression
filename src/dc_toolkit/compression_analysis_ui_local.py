"""
Desktop (Qt) UI: open a netCDF file, sweep the codec space on one field
(under mpirun, one rank per physical core), inspect the results in the
browser, write the field with a chosen pipeline and save the store as a zip.
Launched by `dc_toolkit run_local_ui`.
"""
import json
import os
import shutil
import subprocess
import sys

# PyQt6 is not a declared dependency (it does not build everywhere); install
# a wheel on first use.
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "PyQt6", "--only-binary", ":all:"])

import xarray as xr  # noqa: E402
from PyQt6.QtCore import QLocale, QThread, pyqtSignal  # noqa: E402
from PyQt6.QtGui import QValidator  # noqa: E402
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,  # noqa: E402
                             QLabel, QMainWindow, QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget)

from dc_toolkit import utils_cli  # noqa: E402

OUT_DIR = "out"


class CommandThread(QThread):
    """Run a command in the background, streaming its output lines."""
    line = pyqtSignal(str)
    done = pyqtSignal(int)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd

    def run(self):
        with subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                              env=utils_cli.ui_env()) as proc:
            for line in proc.stdout:
                self.line.emit(line.rstrip())
        self.done.emit(proc.returncode)


class ScientificSpinBox(QDoubleSpinBox):
    def __init__(self):
        super().__init__(None)
        self.setLocale(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
        self.setDecimals(10)
        self.setRange(1e-10, 1.0)
        self.setSingleStep(1e-4)
        self.setValue(utils_cli.UI_DEFAULT_L1)

    def textFromValue(self, value: float) -> str:
        return f"{value:.2e}"

    def valueFromText(self, text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return utils_cli.UI_DEFAULT_L1

    def validate(self, text: str, pos: int):
        """Accept what float() reads within the range; the stock validator rejects '1e-4'."""
        try:
            ok = self.minimum() <= float(text) <= self.maximum()
        except ValueError:
            ok = False
        return (QValidator.State.Acceptable if ok else QValidator.State.Intermediate), text, pos


class CompressionAnalysisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Evaluate and compress netCDF fields")
        self.dataset_path = None
        self.results = None
        self.swept_field = None  # the field the pipeline list belongs to
        self.thread = None
        self.launcher = utils_cli.ui_launcher()

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.open_button = QPushButton("Open netCDF file")
        self.open_button.clicked.connect(self.open_file)
        layout.addWidget(self.open_button)
        self.file_label = QLabel("No file selected")
        layout.addWidget(self.file_label)
        layout.addWidget(QLabel("Field to compress:"))
        self.field_box = QComboBox()
        self.field_box.currentTextChanged.connect(self.forget_results)
        layout.addWidget(self.field_box)

        grid = QGridLayout()
        self.class_boxes = {}
        for col, kind in enumerate(("compressor", "filter", "serializer")):
            grid.addWidget(QLabel(f"{kind.capitalize()} class:"), 0, col)
            box = QComboBox()
            box.addItems(utils_cli.UI_CLASS_OPTIONS[kind])
            grid.addWidget(box, 1, col)
            self.class_boxes[kind] = box
        layout.addLayout(grid)
        self.lossy_check = QCheckBox("Include lossy codecs")
        self.lossy_check.setChecked(True)
        layout.addWidget(self.lossy_check)
        self.ebcc_check = QCheckBox("Add EBCC (optional package; float lat/lon frames only)")
        layout.addWidget(self.ebcc_check)
        layout.addWidget(QLabel("Relative L1 error budget:"))
        self.l1_box = ScientificSpinBox()
        layout.addWidget(self.l1_box)

        self.analyze_button = QPushButton("Evaluate combos")
        self.analyze_button.clicked.connect(self.evaluate_combos)
        self.analyze_button.setEnabled(False)
        layout.addWidget(self.analyze_button)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        layout.addWidget(QLabel("Pipeline to write (best ratio first):"))
        self.pipeline_box = QComboBox()
        layout.addWidget(self.pipeline_box)
        self.compress_button = QPushButton("Compress field and save as zip")
        self.compress_button.clicked.connect(self.compress_field)
        self.compress_button.setEnabled(False)
        layout.addWidget(self.compress_button)

    def forget_results(self, *_):
        """Another field or file: the pipeline list no longer applies."""
        self.swept_field = self.results = None
        self.pipeline_box.clear()
        self.compress_button.setEnabled(False)

    def set_busy(self, busy: bool):
        """No file, field or command change while a command runs."""
        for widget in (self.open_button, self.field_box, self.analyze_button):
            widget.setEnabled(not busy)
        self.compress_button.setEnabled(not busy and self.pipeline_box.count() > 0)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open netCDF file", "", "NetCDF files (*.nc)")
        if not path:
            return
        try:  # an unreadable file must not take the window down
            with xr.open_dataset(path) as ds:
                variables = list(ds.data_vars)
        except Exception as e:
            QMessageBox.warning(self, "Cannot open file", f"{path}: {e}")
            return
        if not variables:
            QMessageBox.warning(self, "Nothing to compress", f"{path} has no data variables.")
            return
        self.dataset_path = path
        self.file_label.setText(f"Selected file: {path}")
        self.field_box.clear()
        self.field_box.addItems(variables)
        self.forget_results()
        self.analyze_button.setEnabled(bool(variables))

    def evaluate_combos(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        classes = {kind: box.currentText() for kind, box in self.class_boxes.items()}
        self.forget_results()
        self.swept_field = self.field_box.currentText()
        cmd = utils_cli.ui_sweep_command(self.launcher, self.dataset_path, OUT_DIR, self.swept_field,
                                         classes, self.lossy_check.isChecked(), self.ebcc_check.isChecked(),
                                         self.l1_box.value())
        self.log.append("Sweeping the codec space ...")
        self.set_busy(True)
        self.thread = CommandThread(cmd)
        self.thread.line.connect(self.log.append)
        self.thread.done.connect(self.sweep_finished)
        self.thread.start()

    def sweep_finished(self, returncode):
        if returncode != 0:
            self.set_busy(False)
            QMessageBox.warning(self, "evaluate_combos failed", f"exit code {returncode}; see the log.")
            return
        self.results = utils_cli.ui_results(OUT_DIR, self.swept_field)
        if self.results.empty:
            self.set_busy(False)
            self.log.append("No combination passed the gates; loosen the error budget and sweep again.")
            return
        self.pipeline_box.addItems(list(self.results["name"]))
        self.set_busy(False)
        self.log.append(f"{len(self.results)} combinations passed the gates; best: {self.results['name'].iloc[0]}")
        utils_cli.clustering_figure(self.results).show()

    def compress_field(self):
        field = self.swept_field
        name = self.pipeline_box.currentText()
        pipeline = json.loads(self.results.loc[self.results["name"] == name, "pipeline"].iloc[0])
        cmd = utils_cli.ui_compress_command(self.launcher, self.dataset_path, OUT_DIR, field, pipeline)
        self.log.append(f"Compressing {field} with {name} ...")
        self.set_busy(True)
        self.thread = CommandThread(cmd)
        self.thread.line.connect(self.log.append)
        self.thread.done.connect(self.compress_finished)
        self.thread.start()

    def compress_finished(self, returncode):
        self.set_busy(False)
        store = utils_cli.merged_store_path(OUT_DIR, self.dataset_path)
        if returncode != 0 or not os.path.isdir(store):
            QMessageBox.warning(self, "compress failed", f"exit code {returncode}; see the log.")
            return
        archive = utils_cli.zip_directory(store)
        save_path, _ = QFileDialog.getSaveFileName(self, "Save the compressed store", os.path.basename(archive),
                                                   "ZIP archives (*.zip)")
        if save_path:
            shutil.move(archive, save_path)
            self.log.append(f"Saved {save_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CompressionAnalysisUI()
    window.resize(700, 700)
    window.show()
    sys.exit(app.exec())
