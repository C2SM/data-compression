"""The requirement count (N_Req): each kind of condition, mean conditions decided over the whole field, and
the verdict of the community list's own checker on every entry of the list."""
import dask.array as dsa
import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.codecs import numcodecs as nc

from dc_toolkit import utils

MAX_ABS, MAX_REL = "max-pointwise-absolute-error-bound", "max-pointwise-relative-error-bound"
MEAN_ABS, MEAN_REL = "mean-absolute-error-bound", "mean-relative-error-bound"


def errors(orig, dec, conditions, chunks=None):
    orig, dec = np.asarray(orig, "f4"), np.asarray(dec, "f4")
    return utils._errors_from_sums(utils._error_sums(orig, dec, chunks or orig.shape, requirement=conditions),
                                   False, conditions)[0]


def n_req(orig, dec, *conditions, chunks=None):
    return errors(orig, dec, list(conditions), chunks)["N_Req"]


@pytest.mark.parametrize("condition, orig, dec, failing", [
    ({"kind": MAX_ABS, "value": 0.1}, [1, 2, 3], [1.05, 2.2, 3], 1),
    ({"kind": MAX_REL, "value": 0.1}, [1, 10, 0, 0], [1.05, 12, 0, 1e-9], 2),       # a zero has no relative slack
    ({"kind": "data-limits", "minimum": 0, "maximum": 1}, [-1, 0, 0.5, 1, 2], [-2, -0.1, 0.5, 1.2, 0.5], 2),
    ({"kind": "data-limits", "minimum": 0}, [-1, 0, 0.5, 1, 2], [-2, -0.1, 0.5, 1.2, 0.5], 1),
    ({"kind": "isovalue", "value": 0.5}, [0.2, 0.5, 0.8, 0.8], [0.6, 0.51, 0.7, 0.5], 3),  # crossed, left, reached
    ({"kind": "missing-value", "value": 255}, [255, 255, 3, 4], [255, 254, 255, 4], 2),
    ({"kind": "lossless"}, [1, 0.0, 2, 3], [1, -0.0, 2.0000002, 3], 2),
])
def test_cells_failing_a_condition(condition, orig, dec, failing):
    assert n_req(orig, dec, condition) == failing
    assert n_req(orig, orig, condition) == 0


def test_a_failed_mean_condition_fails_every_cell():
    assert n_req(np.zeros(4), [0.5, 0, 0, 0], {"kind": MEAN_ABS, "value": 0.125}) == 0
    assert n_req(np.zeros(4), [0.75, 0, 0, 0], {"kind": MEAN_ABS, "value": 0.125}) == 4


def test_a_relative_mean_keeps_every_zero():
    condition = {"kind": MEAN_REL, "value": 0.1}
    assert n_req([10, 10, 0], [10.5, 10.5, 0], condition) == 0
    assert n_req([10, 10, 0], [10.5, 10.5, 0.001], condition) == 1  # the mean holds, the zero moved
    assert n_req([10, 10, 0], [15, 15, 0], condition) == 2          # the mean fails, the zero is kept


def test_mean_conditions_are_decided_over_the_whole_field():
    """Chunk by chunk and block by block (compress), the first cell alone would break the mean."""
    orig, dec, condition = np.zeros(4, "f4"), np.array([0.5, 0, 0, 0], "f4"), [{"kind": MEAN_ABS, "value": 0.125}]
    assert n_req(orig, dec, *condition, chunks=(1,)) == 0
    blocks = [utils._error_sums(orig[s], dec[s], (1,), requirement=condition) for s in (slice(0, 1), slice(1, 4))]
    assert utils._errors_from_sums(utils._merge_sums(blocks), False, condition)[0]["N_Req"] == 0


def test_any_and_all_combine_cell_by_cell():
    loose, tight = {"kind": MAX_ABS, "value": 0.25}, {"kind": MAX_REL, "value": 0.01}
    orig, dec = [100, 100, 1, 1], [100.5, 102, 1.125, 1.5]        # ok for: tight; neither; loose; neither
    assert n_req(orig, dec, {"kind": "any", "requirements": [loose, tight]}) == 2
    assert n_req(orig, dec, {"kind": "all", "requirements": [loose, tight]}) == 4
    assert n_req(orig, dec, loose, tight) == 4                     # a list of conditions holds together
    mean = {"kind": MEAN_ABS, "value": 0.5}                        # fails: the mean error is 0.78
    assert n_req(orig, dec, {"kind": "any", "requirements": [mean, loose]}) == 3
    assert n_req(orig, dec, {"kind": "any", "requirements": []}) == 4
    assert n_req(orig, dec, {"kind": "all", "requirements": []}) == 0


def test_missing_cells_are_left_to_the_finite_gate():
    exact = {"kind": MAX_ABS, "value": 0}
    assert n_req([np.nan, 1], [np.nan, 1], exact) == 0
    err = errors([np.nan, 1, 2], [5, 1, np.nan], [exact])
    assert (err["N_Req"], err["N_Corrupt"]) == (0, 2)
    assert n_req([np.nan] * 3, [np.nan] * 3, {"kind": MEAN_REL, "value": 0.1}, {"kind": MEAN_ABS, "value": 1}) == 0


def test_no_requirement_no_count():
    assert errors([1, 2], [1, 3], None)["N_Req"] is None


def test_absolute_errors_are_in_the_units_of_the_field():
    err = errors([1, 2, 3, np.nan], [1.5, 2, 2, np.nan], None)
    assert (err["Mean_Abs_Error"], err["Max_Abs_Error"]) == (0.5, 1.0)


def test_range_relative_bounds_become_absolute():
    conditions = [{"kind": "any", "requirements": [
        {"kind": "max-pointwise-range-relative-error-bound", "value": 0.01},
        {"kind": "mean-range-relative-error-bound", "value": 0.02}]}, {"kind": "data-limits", "minimum": 0.0}]
    assert utils.resolve_requirement(conditions, 50.0) == [{"kind": "any", "requirements": [
        {"kind": MAX_ABS, "value": 0.5}, {"kind": MEAN_ABS, "value": 1.0}]}, {"kind": "data-limits", "minimum": 0.0}]
    assert conditions[0]["requirements"][0]["value"] == 0.01       # the list's own conditions stay as they are
    assert utils.requirement_text(utils.resolve_requirement(conditions, 50.0)) == \
        "(MaxPointwiseAbsoluteErrorBound(0.5) or MeanAbsoluteErrorBound(1)) and DataLimits(minimum=0)"


@pytest.mark.parametrize("condition", [
    {"kind": "max-pointwise-quadratic-error-bound", "value": 1, "minimum": 0, "maximum": 1},  # a kind the list has, unused
    {"kind": "median-range-relative-error-bound", "value": 0.1},
    {"kind": "data-limits", "minimum": -np.inf}, {"kind": "missing-value", "value": np.nan}])
def test_a_condition_the_gate_cannot_check_is_refused(condition):
    with pytest.raises(ValueError, match=condition["kind"]):
        utils.resolve_requirement([{"kind": "all", "requirements": [condition]}], 50.0)


def test_persist_and_sweep_count_the_same_cells(tmp_path):
    """compress's blocks, merged, and the sweep's chunks describe the same round trip."""
    x = (280 + 10 * np.random.default_rng(0).standard_normal((4, 32, 32))).astype("f4")
    conditions = [{"kind": "any", "requirements": [{"kind": MEAN_ABS, "value": 0.01}, {"kind": MAX_ABS, "value": 0.5}]}]
    kwargs = utils.codec_pipeline_kwargs(nc.Zstd(level=6), nc.BitRound(keepbits=7), None)
    dims = ("time", "lat", "lon")
    _, sweep, _ = utils.evaluate_codec_pipeline(x, dims, kwargs, chunks=(1, 32, 32), requirement=conditions)
    da = xr.DataArray(dsa.from_array(x, chunks=(2, 32, 32)), dims=dims)
    _, verify, _ = utils.persist_with_codec_pipeline(da, zarr.storage.LocalStore(str(tmp_path / "s.zarr")), "x", kwargs,
                                                     (1, 32, 32), (2, 32, 32), requirement=conditions)
    assert 0 < sweep["N_Req"] == verify["N_Req"] < x.size


def _field(rng, dtype, scale, mode: int, n: int = 300):
    """A field to compare verdicts on: noise around 0 or an offset, in turn with exact zeros, all positive, as
    categories with the sentinel 255, as fractions of 1, and with NaN and Inf cells."""
    x = scale * rng.choice([0, 0, 3, 50]) + scale * rng.standard_normal(n)
    if mode in (0, 3):
        x[rng.random(n) < 0.3] = 0
    if mode == 1:
        x = np.abs(x)
    if mode == 2:
        x = rng.integers(0, 15, n).astype(float)
        x[rng.random(n) < 0.05] = 255
    if mode == 4:
        x = rng.random(n)
    x = (np.rint(x) if dtype == "i4" else x).astype(dtype)
    if mode == 5 and dtype != "i4":
        x[::37], x[5] = np.nan, np.inf
    return x


def _reconstruction(rng, x, amplitude, keep_zeros: bool):
    """`x` with noise of `amplitude` (0: an exact copy); NaN and Inf stay, and with `keep_zeros` the zeros."""
    y = x.astype("f8") + amplitude * rng.standard_normal(x.size)
    if keep_zeros:
        y[x == 0] = 0
    return (np.rint(y) if x.dtype.kind == "i" else y).astype(x.dtype)


def _holds(x, y, conditions) -> bool:
    """The requirement as dc_toolkit gates it: range-relative bounds on the range of the field, the cells
    counted over two blocks in chunks, and the finite gate."""
    low, high = utils.finite_range(x)
    resolved = utils.resolve_requirement(conditions, high - low)
    blocks = [utils._error_sums(x[s], y[s], (97,), requirement=resolved) for s in (slice(0, 120), slice(120, None))]
    err = utils._errors_from_sums(utils._merge_sums(blocks), False, resolved)[0]
    return err["N_Req"] == 0 and err["N_Corrupt"] == 0


def test_every_entry_of_the_list_gets_the_verdict_of_its_own_checker():
    """dc_toolkit counts in float64, chunk by chunk; compression-requirement-checks decides in exact fractions
    over the whole array.  They must agree on every entry, from an exact copy to a reconstruction far off, so
    that each entry is seen to hold and to fail."""
    provided = pytest.importorskip("compression_recommendations").Recommendations.provide
    check = pytest.importorskip("compression_requirement_checks").check_safety_requirements
    rng = np.random.default_rng(11)
    for i, entry in enumerate(provided.recommendations):
        conditions = [r.get_config() for r in entry.requirements]
        bound = max((leaf["value"] for leaf in utils.requirement_leaves(conditions)
                     if leaf["kind"] in (MAX_ABS, MEAN_ABS)), default=0.0)  # a field some decades above it, if any
        scale = bound * 10.0 ** rng.uniform(0, 3) if bound else 10.0 ** rng.uniform(-6, 5)
        verdicts = set()
        for k, amplitude in enumerate([0.0, *(scale * 10.0 ** np.arange(-8.0, 3.0))]):
            x = _field(rng, ("f4", "f8")[i % 2], scale, mode=k % 6)
            y = _reconstruction(rng, x, amplitude, keep_zeros=k % 2 == 0)
            theirs = check(original=x, reconstructed=y, requirements=entry.requirements)
            assert _holds(x, y, conditions) == theirs, (entry.humanise(), k)
            verdicts.add(theirs)
        assert verdicts == {True, False}, entry.humanise()


def _condition(rng, x, depth: int = 0) -> dict:
    """A random condition within reach of the field `x`: any kind of leaf, or an any/all of up to three."""
    finite = x[np.isfinite(x)].astype("f8")
    scale = float(np.abs(finite).mean()) or 1.0
    kind = str(rng.choice([MAX_ABS, MAX_REL, MEAN_ABS, MEAN_REL, "max-pointwise-range-relative-error-bound",
                           "mean-range-relative-error-bound", "data-limits", "isovalue", "missing-value", "lossless"]
                          + ["any", "all"] * (3 if depth < 2 else 0)))
    if kind in ("any", "all"):
        return {"kind": kind, "requirements": [_condition(rng, x, depth + 1) for _ in range(rng.integers(0, 4))]}
    value = float(rng.choice(finite))  # a value the field holds
    if kind == "data-limits":
        limits = {"minimum": value - scale * rng.random(), "maximum": value + scale * rng.random()}
        return {"kind": kind, **{k: v for k, v in limits.items() if rng.random() < 0.7}}
    if kind in ("isovalue", "missing-value"):
        return {"kind": kind, "value": value if rng.random() < 0.7 else 255.0}
    if kind == "lossless":
        return {"kind": kind}
    return {"kind": kind, "value": 10.0 ** rng.uniform(-5, 0) * (scale if "absolute" in kind else 1.0)}


def test_any_tree_of_conditions_gets_the_verdict_of_the_checker():
    """Every kind of condition the list's format has, nested at random, on float32, float64 and int32 fields:
    the entries of the list never let an isovalue, a sentinel or a limit decide on its own."""
    Requirement = pytest.importorskip("compression_recommendations.requirements.abc").Requirement
    check = pytest.importorskip("compression_requirement_checks").check_safety_requirements
    rng, verdicts = np.random.default_rng(5), {True: 0, False: 0}
    for i in range(400):
        dtype = ("f4", "f8", "i4")[i % 3]
        scale = 10.0 ** rng.uniform(1 if dtype == "i4" else -6, 5)
        x = _field(rng, dtype, scale, mode=i % 6, n=150)
        condition = _condition(rng, x)
        for amplitude in (0.0, scale * 10.0 ** rng.uniform(-6, -2), scale * 10.0 ** rng.uniform(-2, 1)):
            y = _reconstruction(rng, x, amplitude, keep_zeros=i % 2 == 0)
            theirs = check(original=x, reconstructed=y, requirements=[Requirement.from_config(**condition)])
            assert _holds(x, y, [condition]) == theirs, (condition, dtype, amplitude)
            verdicts[theirs] += 1
    assert min(verdicts.values()) > 300, verdicts
