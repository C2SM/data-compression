"""evaluate_combos --requirements on the bundled TIGGE file: an entry of the community list as a gate, with
and without an L1 budget, its records, resume, and the inputs it refuses."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import invoke

pytest.importorskip("compression_recommendations")

T = ("--field-to-compress", "t", "--requirements", "cf-short-name=t,level-kind=pressure")  # 0.05 K or 1 % of the range
EVALS = ("--max-evals", "60")


@pytest.fixture(scope="module")
def swept(tigge, tmp_path_factory):
    out = tmp_path_factory.mktemp("requirement")
    return out, invoke("evaluate_combos", tigge, "--where-to-write", out, *T, *EVALS).output


@pytest.fixture
def sweep_copy(swept, tmp_path):
    return shutil.copytree(swept[0], tmp_path / "sweep")


def test_the_requirement_alone_gates_the_sweep(swept):
    out, log = swept
    m = json.loads((out / "manifest_t.json").read_text())
    req, df = m["requirements"], pd.read_parquet(out / "results_t.parquet")
    assert m["args"]["l1_threshold"] is None and set(m["effective_thresholds"].values()) == {None}
    assert req["markers"] == {"cf-short-name": "t", "level-kind": "pressure"} and "CfShortName(t)" in req["entry"]
    assert req["list"]["version"] and req["range"] > 50
    bound = max(0.05, 0.01 * req["range"])  # any of two bounds on every cell: the looser one
    assert req["resolved"] != req["conditions"] and f"MaxPointwiseAbsoluteErrorBound({0.01 * req['range']:g})" in log
    assert (df["pass_req"] == (df["max_abs_err"] <= bound)).all() and (df["pass_req"] == (df["n_req"] == 0)).all()
    assert (df["keep"] == (df["pass_req"] & df["pass_finite"])).all() and 0 < df["keep"].sum() < len(df)
    assert m["best"]["ratio"] == df.loc[df["keep"], "ratio"].max()
    assert "[gates] thresholds (relative): L1=off L2=off Linf=off bias=off q99=off" in log
    assert "[requirements] t (units=K) is checked against" in log


def test_an_l1_budget_gates_beside_the_requirement(tigge, tmp_path):
    invoke("evaluate_combos", tigge, "--where-to-write", tmp_path, *T, *EVALS, "--l1-threshold", "1e-6")
    df = pd.read_parquet(tmp_path / "results_t.parquet")
    assert (df["keep"] == (df["pass_req"] & df["pass_l1"] & df["pass_l2"] & df["pass_linf"] & df["pass_bias"])).all()
    assert (df["pass_req"] & ~df["pass_l1"]).any()


def test_resume_reuses_the_rows_of_the_same_requirement(tigge, sweep_copy, swept):
    out = invoke("evaluate_combos", tigge, "--where-to-write", sweep_copy, *T, *EVALS).output
    assert "[resume] 60 of 60 combo(s) of 't' are already recorded" in out
    a, b = (pd.read_parquet(d / "results_t.parquet").sort_values("pipeline").reset_index(drop=True)
            for d in (swept[0], sweep_copy))
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("gates", [("--l1-threshold", "0.005"),  # no requirement any more
                                   ("--requirements", "cf-short-name=q,level-kind=pressure")])
def test_another_requirement_restarts_the_field(tigge, sweep_copy, gates):
    """N_Req was counted against the conditions in sweep_state_t.json, as N_Bounds against the bounds."""
    out = invoke("evaluate_combos", tigge, "--where-to-write", sweep_copy, "--field-to-compress", "t", *EVALS,
                 *gates).output
    assert "measured with another requirement; starting this field from scratch" in out
    assert (sweep_copy / "results_t.parquet.previous").is_file()


def test_limits_of_an_entry_the_field_exceeds_are_flagged(tigge, tmp_path):
    """Total cloud cover is a fraction in [0, 1]; a temperature in K is a field in other units."""
    out = invoke("evaluate_combos", tigge, "--where-to-write", tmp_path, "--field-to-compress", "t",
                 "--requirements", "cf-short-name=tcc,level-kind=single", "--max-evals", "5").output
    assert "beyond the entry's limits [0, 1]: is the field in the units of the list?" in out


def test_a_sentinel_of_the_entry_is_not_taken_for_other_units(tmp_path):
    """Soil type is a category in [0, 11] with the sentinel 255, which the entry names beside its limits."""
    codes = np.random.default_rng(0).integers(0, 8, (2, 40, 40)).astype("u1")
    codes[:, :3] = 255
    xr.Dataset({"slt": (("time", "lat", "lon"), codes)}).to_netcdf(tmp_path / "slt.nc")
    out = invoke("evaluate_combos", tmp_path / "slt.nc", "--where-to-write", tmp_path / "o", "--field-to-compress",
                 "slt", "--requirements", "cf-short-name=slt,level-kind=single", "--max-evals", "10").output
    assert "MissingValue(255)" in out and "WARNING" not in out
    assert json.loads((tmp_path / "o" / "manifest_slt.json").read_text())["best"] is not None  # a lossless pipeline


def test_a_limit_of_the_entry_is_sampled_as_a_bound_is(fields, tmp_path):
    """As under --phys-min 0 (test_edge_fields): the level where the field comes nearest 0 joins the sample."""
    out = invoke("evaluate_combos", fields["geo"], "--where-to-write", tmp_path, "--field-to-compress", "geo",
                 "--requirements", "cf-short-name=tp,level-kind=single", "--eval-data-size-limit", "1KB",
                 "--compressor-class", "none", "--filter-class", "none", "--serializer-class", "zfpy").output
    assert "DataLimits(minimum=0)" in out and "height=3/12 [2, 6, 11]" in out


@pytest.mark.parametrize("args, code, message", [
    ((), 2, "give --l1-threshold, --requirements, or both"),
    (("--requirements", "cf-short-name=t,level-kind=pressure"), 1, "name it with --field-to-compress"),
    ((*T, "--extremes-sensitive"), 1, "--extremes-sensitive needs --q99-threshold"),
    (("--field-to-compress", "t", "--requirements", "t"), 1, "one name and the level kind"),
    (("--field-to-compress", "t", "--requirements", "cf-short-name=t,level-kind=single"), 1, "no entry matches"),
])
def test_refused_inputs(tigge, tmp_path, args, code, message):
    assert message in invoke("evaluate_combos", tigge, "--where-to-write", tmp_path, *args, code=code).output
    assert not list(tmp_path.glob("manifest_*.json"))
