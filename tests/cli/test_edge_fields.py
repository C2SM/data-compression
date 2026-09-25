"""Sweep-level scenarios on tiny synthetic fields, one rank, in-process."""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from conftest import invoke


def test_constant_zero_integer_and_all_nan_fields_are_stored_losslessly(fields, tmp_path):
    out = invoke("evaluate_combos", fields["edge"], "--where-to-write", tmp_path, "--l1-threshold", "0.01").output
    for var in ("const", "zeros", "u8", "allnan"):
        best = json.loads((tmp_path / f"manifest_{var}.json").read_text())["best"]
        assert best["name"] == "zstd(level=6) | - | -", (var, best)
    assert "the field has no finite value; the search is skipped" in out
    invoke("compress", fields["edge"], tmp_path)
    src, g = xr.open_dataset(fields["edge"]), zarr.open_group(str(tmp_path / "edge.zarr"), mode="r")
    for var in ("const", "zeros", "u8", "allnan"):
        assert np.array_equal(src[var].values, g[var][...], equal_nan=True), var


def test_a_field_with_nan_leaves_fso_out(fields, tmp_path):
    """FixedScaleOffset cannot keep NaN fill; the class then has nothing for the field, which is skipped."""
    out = invoke("evaluate_combos", fields["nan"], "--where-to-write", tmp_path, "--l1-threshold", "0.01",
                 "--filter-class", "fixedscaleoffset", "--serializer-class", "none").output
    assert "skipping sst: the filter class has nothing for this field: it holds NaN/Inf" in out
    assert not (tmp_path / "results_sst.parquet").exists()
    invoke("compress", fields["nan"], tmp_path, code=1)  # nothing to compress


def test_a_field_with_nan_keeps_its_fill(fields, tmp_path):
    invoke("evaluate_combos", fields["nan"], "--where-to-write", tmp_path, "--l1-threshold", "0.01",
           "--max-evals", "60")
    df = pd.read_parquet(tmp_path / "results_sst.parquet")
    assert (df["n_corrupt"] == 0).all() and not df["filter"].str.contains("fixedscaleoffset|delta").any()
    assert not df["serializer"].str.contains("zfpy").any()
    invoke("compress", fields["nan"], tmp_path)
    src = xr.open_dataset(fields["nan"])["sst"].values
    assert np.array_equal(np.isnan(src), np.isnan(zarr.open_group(str(tmp_path / "nan.zarr"), mode="r")["sst"][...]))


@pytest.mark.parametrize("kind", ["fso", "zfp"])
def test_verify_gate_refuses_a_pipeline_that_writes_the_fill(fields, tmp_path, kind):
    sst = xr.open_dataset(fields["nan"])["sst"].values
    lo, hi = float(np.nanmin(sst)), float(np.nanmax(sst))
    pipe = {"compressor": None, "serializer": None, "filter": {"name": "numcodecs.fixedscaleoffset", "configuration": {
        "offset": lo, "scale": 65535 / (hi - lo), "dtype": "float32", "astype": "uint16"}}} if kind == "fso" else {
        "compressor": None, "filter": None, "serializer": {"name": "numcodecs.zfpy", "configuration": {
            "mode": 4, "tolerance": 0.5}}}
    (tmp_path / "p.json").write_text(json.dumps(pipe))
    out = invoke("compress", fields["nan"], tmp_path, "--vars", "sst", "--pipeline", tmp_path / "p.json", code=1).output
    assert "pass_finite" in out  # enforced without any threshold
    assert "sst" not in zarr.open_group(str(tmp_path / "nan.zarr"), mode="r").array_keys()


def test_sample_skips_the_levels_that_do_not_vary(fields, tmp_path):
    """Qc is 0 on its top 10 levels; the 3 levels are block midpoints of the other 20."""
    out = invoke("evaluate_combos", fields["icon"], "--where-to-write", tmp_path, "--field-to-compress", "qc",
                 "--l1-threshold", "0.01", "--eval-data-size-limit", "1KB", "--extremes-sensitive",
                 "--phys-min", "0", "--max-evals", "40").output
    assert "raised the sample budget of 'qc'" in out
    assert "strided time=3/8 [1, 4, 6], height=3/30 [13, 20, 26] (of the 20 that vary)" in out
    assert "99th percentile of the non-zero |values|" in out


def test_a_field_varying_on_few_levels_is_sampled_there(fields, tmp_path):
    out = invoke("evaluate_combos", fields["icon"], "--where-to-write", tmp_path, "--field-to-compress", "w",
                 "--l1-threshold", "0.01", "--eval-data-size-limit", "1KB", "--max-evals", "10").output
    assert "height=3/30 [0, 1, 2] (of the 3 that vary)" in out


def test_named_field_whose_sample_does_not_vary_is_an_error(fields, tmp_path):
    out = invoke("evaluate_combos", fields["diag"], "--where-to-write", tmp_path, "--field-to-compress", "d",
                 "--l1-threshold", "0.01", "--eval-data-size-limit", "1KB", code=1).output
    assert "the sample holds the single value 0" in out


def test_ensemble_members_share_the_time_minimum(fields, tmp_path):
    out = invoke("evaluate_combos", fields["ens"], "--where-to-write", tmp_path, "--l1-threshold", "0.01",
                 "--eval-data-size-limit", "1KB", "--max-evals", "10").output
    assert "member_id=1 x time=3 x 3 levels" in out


def test_smallest_sample_that_does_not_fit_stops_the_sweep(fields, tmp_path):
    out = invoke("evaluate_combos", fields["icon"], "--where-to-write", tmp_path, "--field-to-compress", "qc",
                 "--l1-threshold", "0.01", "--inner-chunk-mib", "1000000", code=1).output
    assert "[memcheck] FATAL: the smallest sample of 'qc'" in out


@pytest.mark.parametrize("var", ["u10", "orog"])
def test_gradient_gate_records_the_metric(fields, tmp_path, var):
    invoke("evaluate_combos", fields["map"], "--where-to-write", tmp_path, "--field-to-compress", var,
           "--l1-threshold", "0.001", "--gradient-gate", "--no-gradient-shortcircuit", "--max-evals", "30")
    df = pd.read_parquet(tmp_path / f"results_{var}.parquet")
    assert len(df) == 30 and df["grad_rel"].notna().all() and (df["grad_rel"] >= 0).all()
    assert json.loads((tmp_path / f"manifest_{var}.json").read_text())["gradient_threshold"] == 0.1
