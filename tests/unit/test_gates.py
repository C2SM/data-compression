"""Gates, thresholds, the verify gate, CR drift, resume reuse and the choice of the winner."""
import csv
import json
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from dc_toolkit import utils_cli

THR = {"l1": 0.01, "l2": 0.02, "linf": 0.1, "bias": 0.005, "q99": math.inf}
OK = {"Relative_Error_L1": 0.001, "Relative_Error_L2": 0.002, "Relative_Error_Linf": 0.01, "Bias_Rel": 1e-4,
      "Q99_Rel": None, "Grad_Rel": None, "Decoded_Min": 0.0, "Decoded_Max": 1.0, "N_Corrupt": 0, "N_Bounds": 0}


def test_all_gates_pass():
    keep, reasons = utils_cli.evaluate_gates(OK, THR)
    assert keep and all(reasons.values())


@pytest.mark.parametrize("metric, value, gate", [
    ("Relative_Error_L1", 0.02, "pass_l1"), ("Relative_Error_L2", 0.03, "pass_l2"),
    ("Relative_Error_Linf", 0.2, "pass_linf"), ("Bias_Rel", 0.01, "pass_bias"), ("N_Corrupt", 1, "pass_finite"),
    ("N_Bounds", 1, "pass_bounds")])
def test_each_gate_fails_on_its_own(metric, value, gate):
    keep, reasons = utils_cli.evaluate_gates({**OK, metric: value}, THR)
    assert not keep and [k for k, ok in reasons.items() if not ok] == [gate]


def test_missing_metric_and_disabled_gate_pass():
    keep, _ = utils_cli.evaluate_gates({**OK, "Relative_Error_L2": None, "Relative_Error_Linf": 5.0},
                                       {**THR, "linf": math.inf})
    assert keep


def test_gate_bounds_carry_the_slack():
    assert utils_cli.gate_bounds(None, None, 1.0) is None
    assert utils_cli.gate_bounds(0.0, None, 0.5) == (-0.5, math.inf)
    assert utils_cli.gate_bounds(None, 10.0) == (-math.inf, 10.0)


@pytest.mark.parametrize("enabled, ok", [(False, True), (True, False)])
def test_gradient_gate_only_when_enabled(enabled, ok):
    keep, _ = utils_cli.evaluate_gates({**OK, "Grad_Rel": 0.5}, THR, grad_threshold=0.1, grad_gate=enabled)
    assert keep is ok


def test_derived_thresholds():
    opts = SimpleNamespace(l1_threshold=0.01, l2_threshold=None, linf_threshold=None, bias_threshold=None,
                           q99_threshold=None, l2_gate=True, linf_gate=False, bias_gate=True, extremes_sensitive=True)
    assert utils_cli.derive_thresholds(opts) == {"l1": 0.01, "l2": 0.02, "linf": math.inf, "bias": 0.005, "q99": 0.02}


def test_verify_thresholds_without_a_manifest_follow_the_sweep_multiples():
    assert utils_cli.verify_thresholds(None, {"l1": 0.01, "l2": None}) == {
        "l1": 0.01, "l2": 0.02, "linf": 0.1, "bias": 0.005, "q99": math.inf}
    assert utils_cli.verify_thresholds(None, {}) == dict.fromkeys(utils_cli.GATE_KEYS, math.inf)


@pytest.mark.parametrize("errors, status", [
    (OK, "no-thresholds"), ({**OK, "N_Corrupt": 5}, "fail"), ({**OK, "N_Bounds": 2}, "fail")])
def test_the_finite_and_bounds_gates_need_no_threshold(errors, status):
    assert utils_cli.verify_against_manifest("t", errors, None)[0] == status


@pytest.mark.parametrize("errors, overrides, status", [
    (OK, {"l1": 0.01}, "pass"),
    ({**OK, "Relative_Error_L1": 0.05}, {"l1": 0.01}, "fail"),
    ({**OK, "N_Corrupt": 3}, {"l1": 0.01}, "fail"),
    ({**OK, "Relative_Error_L1": 0.05}, {"l1": None}, "fail"),  # None keeps the manifest's value
])
def test_verify_gate(errors, overrides, status):
    manifest = {"effective_thresholds": {"l1": 0.01, "l2": 0.02, "linf": 0.1, "bias": 0.005, "q99": None},
                "phys_min": None, "phys_max": None, "phys_slack": 0.0}
    assert utils_cli.verify_against_manifest("t", errors, manifest, overrides)[0] == status


@pytest.mark.parametrize("achieved, predicted, status", [(5.0, None, "skip"), (5.0, 5.0, "ok"), (3.0, 5.0, "under"),
                                                         (8.0, 5.0, "over"), (5.0, math.inf, "skip")])
def test_cr_drift(achieved, predicted, status):
    assert utils_cli.evaluate_cr_drift(achieved, predicted, 0.25)[2] == status


def _row(pipeline, ratio, l1, **kw):
    row = {c: None for c in utils_cli.PARTIAL_CSV_COLUMNS}
    row.update(name=pipeline, compressor="-", filter="-", serializer="-", pipeline=json.dumps({"p": pipeline}),
               ratio=ratio, l1_rel=l1, l2_rel=l1, linf_rel=l1, bias_rel=0.0, n_corrupt=0, n_bounds=0, eucd=1.0,
               decoded_min=0.0, decoded_max=1.0, keep=True)
    row.update(kw)
    return row


def _write_rank(where, var, rank, rows):
    with open(where / f"config_space_{var}_rank{rank}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=utils_cli.PARTIAL_CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _gate(errors):
    return utils_cli.evaluate_gates(errors, THR)


def _no_missing(df):
    return {}


def test_winner_is_best_ratio_then_lower_l1_then_pipeline(tmp_path):
    _write_rank(tmp_path, "t", 0, [_row("a", 10.0, 0.005), _row("b", 10.0, 0.001)])
    _write_rank(tmp_path, "t", 1, [_row("c", 12.0, 0.02), _row("d", 9.0, 0.0)])  # c fails L1 when re-gated
    planned = {json.dumps({"p": p}) for p in "abcd"}
    best, n_passed, n_rows, parquet = utils_cli.sweep_select_best(str(tmp_path), "t", _gate, planned, _no_missing)
    assert best["name"] == "b" and (n_passed, n_rows) == (3, 4)
    assert set(pd.read_parquet(parquet)["name"]) == set("abcd")


def test_rows_outside_the_planned_space_are_dropped(tmp_path):
    _write_rank(tmp_path, "t", 0, [_row("a", 10.0, 0.005), _row("old", 50.0, 0.0)])
    best, _, _, parquet = utils_cli.sweep_select_best(str(tmp_path), "t", _gate, {json.dumps({"p": "a"})},
                                                      _no_missing)
    assert best["name"] == "a" and list(pd.read_parquet(parquet)["name"]) == ["a"]


def test_a_row_lacking_a_needed_metric_is_not_kept(tmp_path):
    """A row whose re-evaluation for a newly needed metric failed must not win on its old verdict."""
    _write_rank(tmp_path, "t", 0, [_row("a", 20.0, 0.001, grad_rel=None), _row("b", 10.0, 0.001, grad_rel=0.01)])
    opts = SimpleNamespace(extremes_sensitive=False, gradient_gate=True, gradient_shortcircuit=True)
    best, n_passed, n_rows, parquet = utils_cli.sweep_select_best(
        str(tmp_path), "t", _gate, {json.dumps({"p": p}) for p in "ab"},
        lambda df: utils_cli.missing_metrics(df, None, opts, THR))
    assert best["name"] == "b" and (n_passed, n_rows) == (1, 2)
    df = pd.read_parquet(parquet).set_index("name")
    assert not df.loc["a", "pass_grad"] and not df.loc["a", "keep"]


def test_a_reevaluated_row_merges_its_metrics(tmp_path):
    _write_rank(tmp_path, "t", 0, [_row("a", 20.0, 0.001, grad_rel=None)])
    _write_rank(tmp_path, "t", 1, [_row("a", 20.0, 0.001, grad_rel=0.01)])
    opts = SimpleNamespace(extremes_sensitive=False, gradient_gate=True, gradient_shortcircuit=True)
    best, n_passed, n_rows, _ = utils_cli.sweep_select_best(
        str(tmp_path), "t", _gate, {json.dumps({"p": "a"})}, lambda df: utils_cli.missing_metrics(df, None, opts, THR))
    assert best["name"] == "a" and (n_passed, n_rows) == (1, 1)


def test_resume_reuses_only_rows_with_the_metrics_the_gates_need():
    prev = pd.DataFrame([_row("a", 10, 0.001, q99_rel=0.001, grad_rel=None),
                         _row("b", 10, 0.001, q99_rel=None, grad_rel=0.01),
                         _row("c", 10, 0.5, q99_rel=0.001, grad_rel=None)])  # fails L1: no gradient needed
    opts = SimpleNamespace(extremes_sensitive=True, gradient_gate=True, gradient_shortcircuit=True)
    ok = utils_cli.reusable_rows(prev, 0.3, opts, THR)
    assert list(ok) == [False, False, True]
    assert list(utils_cli.reusable_rows(prev, None, opts, THR)) == [False, True, True]  # no q99 cut: not needed
