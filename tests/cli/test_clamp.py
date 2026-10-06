"""evaluate_combos --clamp-to-bounds: the clip zfp and EBCC decode through, from --phys-min or a requirement's
data limits; in the sweep's rows, the manifest, the store, and a reader without dc_toolkit imported."""
import importlib.util
import json
import subprocess
import sys

import pandas as pd
import pytest
import zarr

from conftest import invoke

ZFP = ("--serializer-class", "zfpy", "--filter-class", "none", "--compressor-class", "zstd", "--max-evals", "12")
OROG = ("--field-to-compress", "orog", "--l1-threshold", "0.05", "--phys-min", "0")  # orog >= 0, exactly 0 at the poles
CLAMP_LINE = "[clamp] orog: 12 combo(s) with zfp or EBCC decode through clamp(minimum=0.0)."


def codec_names(array):
    codecs = array.metadata.to_dict()["codecs"]
    if codecs[0]["name"] == "sharding_indexed":
        codecs = codecs[0]["configuration"]["codecs"]
    return [c["name"] for c in codecs]


def test_the_clamp_needs_bounds(fields, tmp_path):
    r = invoke("evaluate_combos", fields["map"], "--where-to-write", tmp_path, "--field-to-compress", "orog",
               "--l1-threshold", "0.05", "--clamp-to-bounds", *ZFP, code=1)
    assert "--clamp-to-bounds needs --phys-min, --phys-max, or a --requirements entry with data limits" in r.output


def test_the_clamp_keeps_zfp_within_the_bounds(fields, tmp_path):
    """The same combos as without it, each with the clamp in front: no cell below 0, no error larger."""
    invoke("evaluate_combos", fields["map"], "--where-to-write", tmp_path / "plain", *OROG, *ZFP)
    plain = pd.read_parquet(tmp_path / "plain" / "results_orog.parquet")
    out = invoke("evaluate_combos", fields["map"], "--where-to-write", tmp_path / "clamped", *OROG, *ZFP,
                 "--clamp-to-bounds").output
    df = pd.read_parquet(tmp_path / "clamped" / "results_orog.parquet")
    assert CLAMP_LINE in out and "| clamp=on" in out
    assert (plain["decoded_min"] < 0).any() and (~plain["pass_bounds"] & plain["pass_l1"]).any()
    assert (df["decoded_min"] >= 0).all() and (df["n_bounds"] == 0).all() and df["keep"].sum() > plain["keep"].sum()
    assert df["name"].str.contains("| clamp(minimum=0.0) |", regex=False).all()
    both = plain.merge(df, on=["compressor", "serializer"], suffixes=("", "_c"))
    assert len(both) == 12 and (both["max_abs_err_c"] <= both["max_abs_err"]).all()
    assert (both["mean_abs_err_c"] <= both["mean_abs_err"]).all()
    m = json.loads((tmp_path / "clamped" / "manifest_orog.json").read_text())
    clamp = {"name": "numcodecs.clamp", "configuration": {"minimum": 0.0}}
    assert m["args"]["clamp_to_bounds"] is True and m["clamp"] == clamp
    assert m["best"]["pipeline"]["filter"] == clamp
    assert json.loads((tmp_path / "clamped" / "sweep_state_orog.json").read_text())["clamp"] == clamp
    out = invoke("evaluate_combos", fields["map"], "--where-to-write", tmp_path / "clamped", *OROG, *ZFP).output
    assert "measured with another clamp; starting this field from scratch" in out


def test_compress_writes_the_clamp_and_a_bare_reader_applies_it(fields, tmp_path):
    invoke("evaluate_combos", fields["map"], "--where-to-write", tmp_path, *OROG, *ZFP, "--clamp-to-bounds")
    assert "[verify-gate] orog: PASS" in invoke("compress", fields["map"], tmp_path).output
    store = str(tmp_path / "map.zarr")
    a = zarr.open_group(store, mode="r")["orog"]
    assert codec_names(a)[0] == "numcodecs.clamp"
    rec = json.loads(a.attrs["dc_toolkit"])
    assert rec["errors"]["N_Bounds"] == 0 and rec["errors"]["Decoded_Min"] >= 0
    read = f"import zarr; print(float(zarr.open_group({store!r}, mode='r')['orog'][...].min()))"
    r = subprocess.run([sys.executable, "-c", read], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and float(r.stdout) >= 0, r.stderr  # the entry point resolves numcodecs.clamp
    r = invoke("compress", fields["map"], tmp_path, "--stock-codecs-only", "--no-skip-existing", code=1)
    assert "keeps no stock row" in r.output


@pytest.mark.skipif(importlib.util.find_spec("compression_recommendations") is None,
                    reason="needs the recommendations extra")
def test_a_requirements_limits_set_the_clamp(fields, tmp_path):
    """DataLimits(minimum=0) of the tp entry is a clamp bound too; with --phys-min the tighter side counts."""
    out = invoke("evaluate_combos", fields["map"], "--where-to-write", tmp_path, "--field-to-compress", "orog",
                 "--requirements", "grib-short-name=tp,level-kind=single", "--phys-min", "-5",
                 "--clamp-to-bounds", *ZFP).output
    assert CLAMP_LINE in out
    df = pd.read_parquet(tmp_path / "results_orog.parquet")
    assert (df["decoded_min"] >= 0).all() and (df["n_bounds"] == 0).all()


@pytest.mark.ebcc
def test_ebcc_decodes_through_the_clamp(fields, tigge, tmp_path):
    """EBCC keeps the clamp as its one filter besides AsType: on a float32 frame alone, on a float64 one in
    a chain with the cast."""
    pytest.importorskip("ebcc")
    invoke("evaluate_combos", fields["map"], "--where-to-write", tmp_path / "f32", *OROG, "--serializer-class", "ebcc",
           "--clamp-to-bounds")
    df = pd.read_parquet(tmp_path / "f32" / "results_orog.parquet")
    assert len(df) == 7 and df["name"].str.startswith("- | clamp(minimum=0.0) | EBCC(").all()
    assert (df["decoded_min"] >= 0).all()
    assert "[verify-gate] orog: PASS" in invoke("compress", fields["map"], tmp_path / "f32").output
    invoke("evaluate_combos", tigge, "--where-to-write", tmp_path / "f64", "--field-to-compress", "t",
           "--l1-threshold", "0.005", "--phys-min", "200", "--serializer-class", "ebcc", "--clamp-to-bounds")
    df = pd.read_parquet(tmp_path / "f64" / "results_t.parquet")
    assert df["name"].str.startswith("- | clamp(minimum=200.0)+astype(decode_dtype=float64, encode_dtype=float32) | EBCC(").all()
    best = json.loads((tmp_path / "f64" / "manifest_t.json").read_text())["best"]["pipeline"]
    assert [c["name"] for c in best["filter"]] == ["numcodecs.clamp", "numcodecs.astype"]
    out = invoke("compress", tigge, tmp_path / "f64").output
    assert "[verify-gate] t: PASS" in out
    a = zarr.open_group(str(tmp_path / "f64" / "tigge_pl_t_q_dx=2_2024_08_02.zarr"), mode="r")["t"]
    assert codec_names(a)[:3] == ["numcodecs.clamp", "numcodecs.astype", "numcodecs.ebcc_filter"]
