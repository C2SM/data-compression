"""Every command end to end on the bundled TIGGE file, one rank, in-process."""
import json
import re
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from conftest import invoke

SWEEP = ("--l1-threshold", "0.005", "--max-evals", "30", "--extremes-sensitive")
STORE = "tigge_pl_t_q_dx=2_2024_08_02.zarr"


@pytest.fixture(scope="module")
def swept(tigge, tmp_path_factory):
    out = tmp_path_factory.mktemp("sweep")
    invoke("evaluate_combos", tigge, "--where-to-write", out, *SWEEP)
    return out


@pytest.fixture
def sweep_copy(swept, tmp_path):
    return shutil.copytree(swept, tmp_path / "sweep")


def test_help_of_every_command():
    from dc_toolkit.cli import cli
    assert "evaluate_combos" in invoke("--help").output
    invoke("help")
    for name in cli.commands:
        invoke(name, "--help")


@pytest.mark.parametrize("args", [
    ("evaluate_combos", "{tigge}", "--where-to-write", "{tmp}", "--l1-threshold", "nan"),
    ("evaluate_combos", "{tigge}", "--where-to-write", "{tmp}", "--l1-threshold", "0.01", "--phys-min", "inf"),
    ("compress", "{tigge}", "{tmp}", "--vars", "t", "--pipeline", "{{}}", "--linf-threshold", "-inf"),
    ("evaluate_combos", "{tigge}", "--where-to-write", "{tmp}", "--l1-threshold", "0.01", "--filter-class", "zfp"),
])
def test_bad_options_exit_2(tigge, tmp_path, args):
    invoke(*[a.format(tigge=tigge, tmp=tmp_path) for a in args], code=2)


def test_inverted_bounds_are_refused(tigge, tmp_path):
    out = invoke("evaluate_combos", tigge, "--where-to-write", tmp_path, "--l1-threshold", "0.01", "--phys-min", "5",
                 "--phys-max", "1", code=1).output
    assert "--phys-min 5 is above --phys-max 1" in out


def test_sweep_writes_the_documented_files(swept):
    """README.md, "Output files": nothing undocumented, nothing missing (one rank, no crash)."""
    names = {p.name for p in swept.iterdir()}
    for var in ("t", "q"):
        assert {f"config_space_{var}.csv", f"config_space_{var}_rank0.csv", f"failures_{var}_rank0.csv",
                f"results_{var}.parquet", f"sweep_state_{var}.json", f"manifest_{var}.json"} <= names
    assert all(re.fullmatch(r"(config_space_[tq](_rank0)?\.csv|failures_[tq]_rank0\.csv|results_[tq]\.parquet|"
                            r"sweep_state_[tq]\.json|manifest_[tq]\.json)", n) for n in names), names


def test_manifest_records_provenance_and_the_state(swept):
    m = json.loads((swept / "manifest_t.json").read_text())
    state = json.loads((swept / "sweep_state_t.json").read_text())
    from dc_toolkit import utils_cli
    assert m["sweep_state_digest"] == utils_cli.state_digest(state)
    assert m["provenance"]["dc_toolkit"] and m["num_rows"] == 30 and m["num_filtered"] == 30 - m["num_passed"]
    assert {"code", "bounds", "metric_definitions", "row_columns", "env"} <= set(state)


def test_resume_reuses_every_row(tigge, sweep_copy, swept):
    out = invoke("evaluate_combos", tigge, "--where-to-write", sweep_copy, *SWEEP).output
    assert "[resume] 30 of 30 combo(s) of 't' are already recorded" in out
    for var in ("t", "q"):
        a, b = (pd.read_parquet(d / f"results_{var}.parquet").sort_values("pipeline").reset_index(drop=True)
                for d in (swept, sweep_copy))
        pd.testing.assert_frame_equal(a, b)


def test_changed_chunk_setting_restarts_the_field(tigge, sweep_copy):
    """Rows are reused only while sweep_state_{var}.json matches (here the chunk settings); the previous
    results stay beside the new ones as *.previous."""
    out = invoke("evaluate_combos", tigge, "--where-to-write", sweep_copy, *SWEEP, "--inner-chunk-mib", "8").output
    assert "measured with another inner_chunk_mib; starting this field from scratch" in out
    for name in ("results_t.parquet", "manifest_t.json", "config_space_t.csv"):
        assert (sweep_copy / f"{name}.previous").is_file() and (sweep_copy / name).is_file()


def test_compress_verify_merge_inspect(tigge, sweep_copy):
    out = invoke("compress", tigge, sweep_copy, "--no-consolidate").output
    assert "[verify-gate] t: PASS" in out and "[verify-gate] q: PASS" in out
    store = sweep_copy / STORE
    assert "consolidated_metadata" not in json.loads((store / "zarr.json").read_text())
    invoke("merge_compressed_fields", tigge, sweep_copy)
    assert "consolidated_metadata" in json.loads((store / "zarr.json").read_text())
    invoke("open_zarr_and_inspect", store, "--head", "2")
    src, g = xr.open_dataset(tigge), zarr.open_group(str(store), mode="r")
    for var in ("t", "q"):
        a, b = src[var].values, g[var][...]
        assert np.array_equal(np.isfinite(a), np.isfinite(b))
        m = json.loads((sweep_copy / f"manifest_{var}.json").read_text())
        linf = m["effective_thresholds"]["linf"]
        assert np.nanmax(np.abs(a - b)) <= linf * np.nanmax(np.abs(a)) * (1 + 1e-9)
        record = json.loads(g[var].attrs["dc_toolkit"])
        assert record["request"]["pipeline"] == m["best"]["pipeline"] and record["verify_gate"] == "pass"
    batch = json.loads((sweep_copy / "batch_manifest.json").read_text())
    assert not batch["any_error"] and {r["verify_gate"] for r in batch["results"].values()} == {"pass"}


def test_compress_refuses_an_empty_vars(tigge, sweep_copy):
    assert "--vars names no field" in invoke("compress", tigge, sweep_copy, "--vars", " , ", code=1).output


def test_no_continue_on_error_stops_at_a_missing_field(tigge, sweep_copy, tmp_path):
    pipe = '{"compressor": {"name": "numcodecs.zstd", "configuration": {"level": 3}}, "filter": null, "serializer": null}'
    out = invoke("compress", tigge, tmp_path / "o", "--vars", "aa,t", "--pipeline", pipe, "--no-continue-on-error",
                 code=1).output
    batch = json.loads((tmp_path / "o" / "batch_manifest.json").read_text())["results"]
    assert batch["aa"]["status"] == "missing-from-dataset" and batch["t"]["status"] == "not-attempted", out


def test_compress_failures_exit_1(tigge, sweep_copy, tmp_path):
    invoke("compress", tigge, sweep_copy, "--vars", "t,zz", code=1)
    assert json.loads((sweep_copy / "batch_manifest.json").read_text())["results"]["zz"]["status"] == "no-pipeline"
    invoke("merge_compressed_fields", tigge, tmp_path / "nowhere", code=1)
    (tmp_path / "empty").mkdir()
    invoke("compress", tigge, tmp_path / "empty", code=1)


def test_stock_codecs_only(tigge, sweep_copy):
    invoke("compress", tigge, sweep_copy, "--stock-codecs-only", "--no-continue-on-error")
    from dc_toolkit import utils_cli
    store = str(sweep_copy / STORE)
    assert all(utils_cli.array_is_stock(store, v) for v in ("t", "q"))


def test_clustering_and_plots(tigge, swept, tmp_path, monkeypatch):
    import plotly.basedatatypes
    monkeypatch.setattr(plotly.basedatatypes.BaseFigure, "show", lambda self, *a, **k: None)
    invoke("perform_clustering", swept / "results_t.parquet", "L1")
    pd.read_parquet(swept / "results_q.parquet").head(3).to_parquet(tmp_path / "tiny.parquet")
    assert "need >= 4" in invoke("perform_clustering", tmp_path / "tiny.parquet", "L2").output
    invoke("analyze_clustering", swept / "results_t.parquet")
    invoke("plot_compression_errors", tigge, tmp_path / "plot", "t", "--manifest-dir", swept)
    assert (tmp_path / "plot" / "t_compression_errors.pdf").stat().st_size > 0


def test_plot_refuses_a_field_that_is_not_lat_lon(fields, swept, tmp_path):
    invoke("plot_compression_errors", fields["edge"], tmp_path / "p", "const",
           "--pipeline", swept / "manifest_q.json", code=1)


def test_conversions_round_trip(tigge, tmp_path):
    z, nc = tmp_path / "t.zarr", tmp_path / "back.nc"
    invoke("from_nc_to_zarr", tigge, "--out", z)
    invoke("from_nc_to_zarr", tigge, "--out", z, code=1)  # exists, no --overwrite
    invoke("from_zarr_to_netcdf", z, "--out", nc)
    src, back = xr.open_dataset(tigge), xr.open_dataset(nc)
    for var in ("t", "q"):
        assert np.array_equal(src[var].values, back[var].values, equal_nan=True)


def test_compressed_store_back_to_netcdf(tigge, sweep_copy, tmp_path):
    invoke("compress", tigge, sweep_copy)
    invoke("from_zarr_to_netcdf", sweep_copy / STORE, "--out", tmp_path / "c.nc")
    assert set(xr.open_dataset(tmp_path / "c.nc").data_vars) == {"t", "q"}


def test_sweep_of_a_zarr_input(tigge, tmp_path):
    z = tmp_path / "in.zarr"
    invoke("from_nc_to_zarr", tigge, "--out", z)
    invoke("evaluate_combos", z, "--where-to-write", tmp_path / "o", "--field-to-compress", "t",
           "--l1-threshold", "0.005", "--max-evals", "10")
    assert (tmp_path / "o" / "manifest_t.json").is_file()
    invoke("compress", z, tmp_path / "o")


@pytest.mark.parametrize("path_or_field", ["field", "suffix"])
def test_unknown_field_or_format(tigge, tmp_path, path_or_field):
    if path_or_field == "field":
        args = (tigge, "--field-to-compress", "nope")
    else:
        bad = tmp_path / "x.txt"
        bad.write_text("x")
        args = (bad,)
    invoke("evaluate_combos", *args, "--where-to-write", tmp_path / "o", "--l1-threshold", "0.01", code=1)


def test_netcdf_export_refuses_above_max_size(tigge, tmp_path):
    invoke("from_nc_to_zarr", tigge, "--out", tmp_path / "t.zarr")
    out = invoke("from_zarr_to_netcdf", tmp_path / "t.zarr", "--out", tmp_path / "x.nc", "--max-size", "1KB",
                 code=1).output
    assert "exceeds --max-size" in out and not (tmp_path / "x.nc").exists()
