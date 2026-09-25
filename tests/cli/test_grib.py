"""The .grib input path (cfgrib), which no validation script exercised: a 4-message GRIB2 file written from an
ecCodes sample in tmp_path, swept, compressed and verified in-process."""
import json

import numpy as np
import pytest

from conftest import invoke


@pytest.fixture(scope="module")
def grib(tmp_path_factory):
    eccodes = pytest.importorskip("eccodes")
    pytest.importorskip("cfgrib")
    path = tmp_path_factory.mktemp("grib") / "t2m.grib"
    with open(path, "wb") as fh:
        for step in range(4):
            gid = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
            lat = np.linspace(-1, 1, nj)[:, None]
            lon = np.linspace(0, 2 * np.pi, ni)[None, :]
            eccodes.codes_set(gid, "step", 6 * step)
            eccodes.codes_set_values(gid, (280 + 10 * np.cos(lat) * np.sin(lon + step)).ravel())
            eccodes.codes_write(gid, fh)
            eccodes.codes_release(gid)
    return path


def test_grib_sweep_and_compress(grib, tmp_path):
    out = invoke("evaluate_combos", grib, "--where-to-write", tmp_path, "--l1-threshold", "0.005",
                 "--max-evals", "20").output
    manifests = list(tmp_path.glob("manifest_*.json"))
    assert manifests, out[-2000:]
    assert all(json.loads(m.read_text())["best"] is not None for m in manifests)
    out = invoke("compress", grib, tmp_path).output
    assert "[verify-gate]" in out and "PASS" in out
