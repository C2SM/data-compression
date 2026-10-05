"""evaluate_combos --requirements on the bundled TIGGE file: an entry of the community list as a gate, with
and without an L1 budget, its records, resume, and the inputs it refuses; compress holds the whole field to it."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from conftest import invoke

pytest.importorskip("compression_recommendations")

T = ("--field-to-compress", "t", "--requirements", "cf-short-name=t,level-kind=pressure")  # 0.05 K or 1 % of the range
EVALS = ("--max-evals", "60")
STORE = "tigge_pl_t_q_dx=2_2024_08_02.zarr"


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


def test_a_unit_factor_converts_the_entry_to_the_units_of_the_field(tigge, tmp_path, swept):
    """As if t were in units 400 times smaller than the entry's kelvin: its 0.05 becomes 20, the looser bound
    now that the one relative to the range (0.86) has no units to convert."""
    out = invoke("evaluate_combos", tigge, "--where-to-write", tmp_path, *T, *EVALS,
                 "--requirements-unit-factor", "400").output
    m = json.loads((tmp_path / "manifest_t.json").read_text())
    req, df = m["requirements"], pd.read_parquet(tmp_path / "results_t.parquet")
    assert req["unit_factor"] == m["args"]["requirements_unit_factor"] == 400
    assert req["conditions"] == json.loads((swept[0] / "manifest_t.json").read_text())["requirements"]["conditions"]
    assert (f"is checked against (MaxPointwiseAbsoluteErrorBound(20) or MaxPointwiseAbsoluteErrorBound("
            f"{0.01 * req['range']:g})), the range-relative bounds x the field's range {req['range']:g}, "
            f"the entry's bounds and limits x 400 (--requirements-unit-factor)") in out
    assert (df["pass_req"] == (df["max_abs_err"] <= 20)).all()
    assert pd.read_parquet(swept[0] / "results_t.parquet")["pass_req"].sum() < df["pass_req"].sum() < len(df)


def test_a_unit_factor_leaves_an_entry_without_units_as_it_is(tigge, tmp_path):
    """q is held within 1 % of its own value: the factor has no number to convert, and the recorded rows stay."""
    q = ("--field-to-compress", "q", "--requirements", "cf-short-name=q,level-kind=pressure", "--max-evals", "10")
    invoke("evaluate_combos", tigge, "--where-to-write", tmp_path, *q)
    out = invoke("evaluate_combos", tigge, "--where-to-write", tmp_path, *q, "--requirements-unit-factor", "1000").output
    assert "MaxPointwiseRelativeErrorBound(0.01); --requirements-unit-factor 1000 changes none of the entry's numbers" in out
    assert "[resume] 10 of 10 combo(s) of 'q' are already recorded" in out


@pytest.mark.parametrize("gates", [("--l1-threshold", "0.005"),  # no requirement any more
                                   ("--requirements", "cf-short-name=q,level-kind=pressure"),
                                   (*T[2:], "--requirements-unit-factor", "400")])  # the entry in other units
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
    (("--field-to-compress", "t", "--l1-threshold", "0.005", "--requirements-unit-factor", "1000"), 1,
     "--requirements-unit-factor converts the entry of --requirements"),
    ((*T, "--requirements-unit-factor", "0"), 2, "Invalid value for '--requirements-unit-factor'"),
    (("--field-to-compress", "t", "--requirements", "cf-short-name=slt,level-kind=single",
      "--requirements-unit-factor", "1000"), 1, "a unit factor cannot convert the sentinel"),
])
def test_refused_inputs(tigge, tmp_path, args, code, message):
    assert message in invoke("evaluate_combos", tigge, "--where-to-write", tmp_path, *args, code=code).output
    assert not list(tmp_path.glob("manifest_*.json"))


def _tighten(where, value):
    """Rewrite the manifest's resolved conditions to one bound on every cell, as another sweep would record it."""
    path = where / "manifest_t.json"
    m = json.loads(path.read_text())
    m["requirements"]["resolved"] = [{"kind": "max-pointwise-absolute-error-bound", "value": value}]
    path.write_text(json.dumps(m))


def test_compress_holds_the_whole_field_to_the_requirement(tigge, sweep_copy):
    out = invoke("compress", tigge, sweep_copy).output
    assert "[verify-gate] t: PASS, the whole field meets the requirement." in out
    asked = json.loads((sweep_copy / "manifest_t.json").read_text())["requirements"]
    stored = zarr.open_group(str(sweep_copy / STORE), mode="r")["t"]
    record = json.loads(stored.attrs["dc_toolkit"])
    assert record["request"]["requirement"] == asked["resolved"] and record["errors"]["N_Req"] == 0
    batch = json.loads((sweep_copy / "batch_manifest.json").read_text())["results"]["t"]
    assert record["requirements"] == batch["requirements"] == asked and batch["errors"]["N_Req"] == 0
    # the list's own checker, given the field and what the store gives back
    from compression_recommendations import Recommendations
    check = pytest.importorskip("compression_requirement_checks").check_safety_requirements
    assert check(original=xr.open_dataset(tigge)["t"].values, reconstructed=stored[...],
                 requirements=Recommendations.provide.search(markers=asked["markers"]))


def test_a_field_that_breaks_the_requirement_stays_out_of_the_store(tigge, sweep_copy):
    _tighten(sweep_copy, 1e-9)  # the winner is lossy
    out = invoke("compress", tigge, sweep_copy, code=1).output
    assert "[verify-gate] FAIL: t: verify gate FAILED (pass_req)" in out
    assert "requirement: MaxPointwiseAbsoluteErrorBound(1e-09) fails at" in out
    assert not (sweep_copy / STORE / "t").exists()
    assert json.loads((sweep_copy / "batch_manifest.json").read_text())["results"]["t"]["status"] == "error"
    out = invoke("compress", tigge, sweep_copy, "--no-verify-gate").output  # written, and marked
    assert "(advisory: --no-verify-gate set)" in out
    assert json.loads(zarr.open_group(str(sweep_copy / STORE), mode="r")["t"].attrs["dc_toolkit"])["errors"]["N_Req"] > 0


def test_a_stored_field_is_rewritten_for_another_requirement(tigge, sweep_copy):
    invoke("compress", tigge, sweep_copy)
    assert "already in" in invoke("compress", tigge, sweep_copy).output
    _tighten(sweep_copy, 5.0)
    out = invoke("compress", tigge, sweep_copy).output
    assert "its verify gate changed; rewriting it" in out and "[verify-gate] t: PASS" in out
