"""Codec spaces, pairing rules and pipeline identity (the JSON --resume, the manifests and compress key on)."""
import json

import click
import numpy as np
import pytest
import xarray as xr
from zarr.codecs import numcodecs as nc

from dc_toolkit import utils, utils_cli

ALL = dict(compressor_class="all", filter_class="all", serializer_class="all", with_lossy=True, with_ebcc=False)


def space(dtype, data_range=(0.0, 1.0), nonfinite=0, **overrides):
    da = xr.DataArray(np.zeros((4, 64, 64), dtype), dims=("time", "lat", "lon"))
    return utils_cli.codec_spaces(da, {**ALL, **overrides}, data_range, nonfinite=nonfinite)


@pytest.mark.parametrize("dtype, total", [("float32", 10494), ("float64", 17127)])
def test_space_sizes_quoted_in_the_docs(dtype, total):
    """docs/PARALLELIZATION.md: 'up to 10,494 (float32) or 17,127 (float64) codec configurations'."""
    cs = utils_cli.sweep_config_space(*space(dtype), None, 1, np.dtype(dtype), True)
    assert len(cs) == total


def test_without_lossy_space():
    """docs/intro.md section 9: a float field --without-lossy keeps compressors and lossless serializers."""
    comps, filts, sers = space("float64", with_lossy=False)
    assert (len(comps), filts, len(sers)) == (33, [None], 9)


def test_without_lossy_refuses_a_float_filter_class():
    with pytest.raises(click.ClickException, match="nothing lossless for a float field"):
        space("float32", with_lossy=False, filter_class="delta")


def test_without_lossy_is_bit_exact():
    """Every --without-lossy pipeline of a float field round-trips exactly (Delta on floats does not)."""
    rng = np.random.default_rng(3)  # signs and magnitudes vary: a difference of two floats then rounds
    x = (rng.standard_normal((4, 64, 64)) * np.exp(rng.normal(0, 3, (4, 64, 64)))).astype("f4")
    comps, filts, sers = space("float32", with_lossy=False)
    for cfg in [(c, None, None) for c in comps[:5]] + [(None, f, None) for f in filts] + [(None, None, s) for s in sers]:
        dec, _ = utils._zarr_roundtrip(x, ("time", "lat", "lon"), utils.codec_pipeline_kwargs(*cfg), (1, 64, 64))
        assert np.array_equal(dec, x), utils.pipeline_name(*cfg)


def test_integer_fields_keep_delta():
    comps, filts, sers = space("int16", data_range=None, with_lossy=False)
    assert any(isinstance(f, nc.Delta) for f in filts)


def test_nan_unsafe_codecs_leave_a_field_with_nan():
    """FixedScaleOffset, zfp and Delta (floats) cannot give NaN back; a field holding any loses them."""
    comps, filts, sers = space("float32", nonfinite=3)
    assert not any(isinstance(f, (nc.FixedScaleOffset, nc.Delta)) for f in filts if f is not None)
    assert not any(isinstance(s, nc.ZFPY) for s in sers if s is not None)
    assert any(isinstance(f, nc.BitRound) for f in filts if f is not None)
    with pytest.raises(click.ClickException, match="holds NaN/Inf"):
        space("float32", nonfinite=3, filter_class="fixedscaleoffset")


def test_config_space_order_is_deterministic_and_capped():
    a = utils_cli.sweep_config_space(*space("float32"), 50, 1, np.dtype("float32"), True)
    b = utils_cli.sweep_config_space(*space("float32"), 50, 1, np.dtype("float32"), True)
    assert [utils.pipeline_json(*c) for c in a] == [utils.pipeline_json(*c) for c in b] and len(a) == 50


def test_max_evals_spans_the_space():
    """--max-evals takes a uniform subset, not the first compressor's combos."""
    cs = utils_cli.sweep_config_space(*space("float32"), 300, 1, np.dtype("float32"), True)
    assert len({type(c[0]).__name__ for c in cs}) >= 5
    assert len({type(c[2]).__name__ for c in cs}) >= 2


@pytest.mark.parametrize("dtype", ["float32", "float64", "int16", "uint8"])
def test_every_codec_round_trips_through_its_json(dtype):
    """A pipeline's identity is its JSON: from_dict(to_dict(codec)) must give back the same JSON, or --resume
    re-evaluates rows and compress rebuilds another codec than the sweep measured."""
    comps, filts, sers = space(dtype)
    for cfg in [(c, None, None) for c in comps] + [(None, f, None) for f in filts] + [(None, None, s) for s in sers]:
        key = utils.pipeline_json(*cfg)
        assert utils.pipeline_json(*utils.pipeline_from_dict(json.loads(key))) == key


@pytest.mark.ebcc
def test_ebcc_round_trips_through_its_json():
    pytest.importorskip("ebcc")
    ebcc = utils.EBCC.from_params(46, 90, 0.1)
    key = utils.pipeline_json(None, None, ebcc)
    assert utils.pipeline_json(*utils.pipeline_from_dict(json.loads(key))) == key
    assert not utils.pipeline_is_stock(json.loads(key))


@pytest.mark.parametrize("filt, ser, dtype, ok", [
    (nc.FixedScaleOffset(offset=0, scale=1, dtype="float32", astype="uint16"), utils.ZFPYRank(mode=2, rate=8), "float32", False),
    (nc.BitRound(keepbits=7), utils.ZFPYRank(mode=2, rate=8), "float32", False),
    (nc.BitRound(keepbits=23), utils.ZFPYRank(mode=2, rate=8), "float32", True),
    (nc.FixedScaleOffset(offset=0, scale=1, dtype="float32", astype="uint8"), nc.PCodec(level=8), "float32", False),
    (nc.FixedScaleOffset(offset=0, scale=1, dtype="float32", astype="uint16"), nc.PCodec(level=8), "float32", True),
    (None, None, "float32", True),
])
def test_pairing_rules(filt, ser, dtype, ok):
    assert utils.combo_is_valid(filt, ser, None, dtype=np.dtype(dtype)) is ok


@pytest.mark.ebcc
def test_ebcc_pairing_rules():
    pytest.importorskip("ebcc")
    ebcc = utils.EBCC.from_params(46, 90, 0.1)
    assert utils.combo_is_valid(None, ebcc, None)
    assert utils.combo_is_valid(nc.AsType(encode_dtype="float32", decode_dtype="float64"), ebcc, None)
    assert not utils.combo_is_valid(None, ebcc, nc.Zstd(level=3))
    assert not utils.combo_is_valid(nc.BitRound(keepbits=7), ebcc, None)


@pytest.mark.parametrize("shape, want", [((1, 3, 1398101), (3, 1398101)), ((70000, 3, 3), (210000, 3)),
                                         ((1, 1, 1), (1,)), ((2, 3, 4, 5, 6), (6, 4, 5, 6))])
def test_zfpy_rank_folds_to_what_zfp_accepts(shape, want):
    assert utils.ZFPYRank.encode_shape(shape) == want


def test_fso_configs_need_the_full_range():
    da = xr.DataArray(np.zeros(3, "f4"))
    assert utils.fixed_scale_offset_configs(da, None) == []
    assert utils.fixed_scale_offset_configs(da, (1.0, 1.0)) == []
    assert [c["astype"] for c in utils.fixed_scale_offset_configs(da, (0.0, 10.0))] == ["uint16"]
    assert [c["astype"] for c in utils.fixed_scale_offset_configs(da.astype("f8"), (0.0, 10.0))] == ["uint16", "uint32"]


@pytest.mark.parametrize("smin, smax, bad", [(0.0, 10.0, False), (-0.00005, 10.0, False), (-0.001, 10.0, True),
                                             (0.0, 10.5, True)])
def test_fso_range_problem(smin, smax, bad):
    """Compress refuses a FixedScaleOffset that cannot hold the field it wrote (values beyond wrap)."""
    fso = nc.FixedScaleOffset(offset=0.0, scale=65535 / 10.0, dtype="float32", astype="uint16")
    problem = utils_cli.fso_range_problem((None, fso, None), {"Source_Min": smin, "Source_Max": smax})
    assert bool(problem) is bad


@pytest.mark.parametrize("offset, bad", [(250.0, False), (200.0, True)])
def test_fso_range_problem_signed(offset, bad):
    """CF packing into int16 centres the range on the offset; an offset at the minimum wraps above 250."""
    fso = nc.FixedScaleOffset(offset=offset, scale=655.34, dtype="float32", astype="int16")
    problem = utils_cli.fso_range_problem((None, fso, None), {"Source_Min": 200.0, "Source_Max": 300.0})
    assert bool(problem) is bad
    assert utils_cli.fso_range_problem((nc.Zstd(level=3), None, None), {"Source_Min": -1e9, "Source_Max": 1e9}) is None


def test_validate_pipeline_refuses_another_dtype():
    """A filter built for float64 would reinterpret a float32 field's bytes."""
    da = xr.DataArray(np.zeros((4, 8), "f4"), dims=("time", "ncells"))
    fso = nc.FixedScaleOffset(offset=0.0, scale=1.0, dtype="float64", astype="uint16")
    with pytest.raises(click.ClickException, match="built for float64"):
        utils_cli.validate_pipeline((None, fso, None), da, "x")
