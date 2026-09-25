"""compress: --skip-existing against the recorded request, the FixedScaleOffset range check, the physical bounds in
the verify gate, and the store lock."""
import json
import os
import socket

import numpy as np
import pytest
import xarray as xr
import zarr

from conftest import invoke

SWEEP = ("--l1-threshold", "0.005", "--max-evals", "20", "--field-to-compress", "t")


@pytest.fixture
def swept_copy(tigge_copy, tmp_path):
    out = tmp_path / "o"
    invoke("evaluate_combos", tigge_copy, "--where-to-write", out, *SWEEP)
    return tigge_copy, out


def test_skip_existing_compares_the_request(swept_copy):
    """An array is skipped only when it was written from the same source, pipeline, chunks and gate."""
    src, out = swept_copy
    invoke("compress", src, out)
    assert "already in" in invoke("compress", src, out).output
    st = os.stat(src)
    os.utime(src, (st.st_atime, st.st_mtime + 100))  # the source changed (or was replaced)
    assert "its source changed; rewriting it" in invoke("compress", src, out).output
    assert "already in" in invoke("compress", src, out).output
    other = {"compressor": {"name": "numcodecs.zstd", "configuration": {"level": 3}}, "filter": None, "serializer": None}
    assert "its pipeline changed" in invoke("compress", src, out, "--vars", "t", "--pipeline", json.dumps(other)).output
    assert "already in" in invoke("compress", src, out, "--vars", "t", "--pipeline", json.dumps(other)).output
    assert "its verify gate changed" in invoke("compress", src, out, "--vars", "t", "--pipeline", json.dumps(other),
                                               "--l1-threshold", "0.001").output


def test_skip_existing_under_the_cr_drift_gate(swept_copy):
    """An array written with a drift warning is not skipped once the gate is on (it is rewritten, and fails)."""
    src, out = swept_copy
    m = json.loads((out / "manifest_t.json").read_text())
    m["best"]["ratio"] *= 20  # a prediction the field cannot reach
    (out / "manifest_t.json").write_text(json.dumps(m))
    assert "[cr-drift] WARNING" in invoke("compress", src, out).output
    res = invoke("compress", src, out, "--cr-drift-gate", code=1)
    assert "short of the sweep's; rewriting it" in res.output and "cr-drift gate FAILED" in res.output


def test_skip_existing_checks_the_drift_against_this_runs_prediction(swept_copy):
    """An array written with --pipeline (no prediction) is not skipped under --cr-drift-gate when the
    manifest's prediction is far off."""
    src, out = swept_copy
    m = json.loads((out / "manifest_t.json").read_text())
    invoke("compress", src, out, "--vars", "t", "--pipeline", json.dumps(m["best"]["pipeline"]))
    m["best"]["ratio"] *= 20
    (out / "manifest_t.json").write_text(json.dumps(m))
    assert "short of the sweep's; rewriting it" in invoke("compress", src, out, "--cr-drift-gate", code=1).output


def test_an_array_without_a_record_is_rewritten(swept_copy):
    src, out = swept_copy
    store = out / (os.path.basename(src)[:-3] + ".zarr")
    g = zarr.open_group(str(store), mode="a", zarr_format=3)
    g.create_array("t", shape=(2,), dtype="f4")[...] = np.zeros(2, "f4")
    assert "it records no dc_toolkit request; rewriting it" in invoke("compress", src, out).output
    assert zarr.open_group(str(store), mode="r")["t"].shape == xr.open_dataset(src)["t"].shape


def test_fso_that_cannot_hold_the_field_is_refused(swept_copy):
    """Values beyond a FixedScaleOffset's range wrap; the check holds even with --no-verify-gate."""
    src, out = swept_copy
    t = xr.open_dataset(src)["t"].values
    lo, hi = float(t.min()), float(t.max())
    mid = (lo + hi) / 2
    fso = {"compressor": None, "serializer": None, "filter": {"name": "numcodecs.fixedscaleoffset", "configuration": {
        "offset": lo, "scale": 65535 / (mid - lo), "dtype": str(t.dtype), "astype": "uint16"}}}
    res = invoke("compress", src, out, "--vars", "t", "--pipeline", json.dumps(fso), "--no-verify-gate", code=1)
    assert "its FixedScaleOffset to uint16 codes as" in res.output


def test_bounds_the_source_already_crosses_do_not_fail_lossless(tigge_copy, tmp_path):
    """Bounds inside the field's own range count only the cells a pipeline moves across them."""
    t = xr.open_dataset(tigge_copy)["t"].values
    inside = float(np.quantile(t, 0.5))
    out = tmp_path / "o"
    log = invoke("evaluate_combos", tigge_copy, "--where-to-write", out, *SWEEP, "--phys-max", inside,
                 "--compressor-class", "zstd", "--filter-class", "none", "--serializer-class", "none").output
    assert "the field itself spans" in log
    m = json.loads((out / "manifest_t.json").read_text())
    assert m["best"] is not None and m["phys_max"] == inside
    assert "[verify-gate] t: PASS" in invoke("compress", tigge_copy, out).output


def test_a_live_compress_lock_is_respected(swept_copy):
    src, out = swept_copy
    lock = out / (os.path.basename(src)[:-3] + ".zarr.lock")
    lock.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))
    assert "another compress is writing" in invoke("compress", src, out, code=1).output
    lock.unlink()
    invoke("compress", src, out)
    assert not lock.exists()
