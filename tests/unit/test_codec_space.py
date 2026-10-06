"""Codec spaces, pairing rules and pipeline identity (the JSON --resume, the manifests and compress key on)."""
import json
import os
import subprocess
import sys

import click
import numpy as np
import pytest
import xarray as xr
import zarr
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


def test_max_evals_subsets_nest():
    """A quick test with a small cap, then a larger one: the larger sweep reuses the smaller one's rows."""
    small = {utils.pipeline_json(*c) for c in utils_cli.sweep_config_space(*space("float32"), 10, 1, np.dtype("float32"), True)}
    large = {utils.pipeline_json(*c) for c in utils_cli.sweep_config_space(*space("float32"), 40, 1, np.dtype("float32"), True)}
    assert small < large


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


@pytest.mark.ebcc
def test_a_process_that_only_reads_ebcc_logs_nothing(tmp_path):
    """EBCC's decoder ignores EBCC_LOG_LEVEL, so dc_toolkit.codecs applies it (default 4, errors only)."""
    pytest.importorskip("ebcc")
    path = str(tmp_path / "s.zarr")
    x = np.linspace(0, 1, 2 * 46 * 90, dtype="f4").reshape(2, 46, 90)
    zarr.create_array(path, shape=x.shape, dtype=x.dtype, chunks=(1, 46, 90), compressors=None,
                      serializer=utils.EBCC.from_params(46, 90, 0.01))[...] = x
    read = f"import zarr; zarr.open_array({path!r}, mode='r')[...]"  # the codec comes through the entry point
    env = {k: v for k, v in os.environ.items() if k != "EBCC_LOG_LEVEL"}
    r = subprocess.run([sys.executable, "-c", read], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0 and "ebcc_codec.c" not in r.stderr, r.stderr  # every EBCC log line names its source


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
def test_a_range_beyond_float32_leaves_only_ebcc_out():
    pytest.importorskip("ebcc")
    da = xr.DataArray(np.zeros((4, 64, 64), "f8"), dims=("time", "lat", "lon"))
    sers = utils.serializer_space(da, with_ebcc=True, data_range=(0.0, 1e300))
    assert sers and not any(isinstance(s, utils.EBCC) for s in sers)
    with pytest.raises(ValueError, match="float32 range"):
        utils.serializer_space(da, serializer_class="ebcc", data_range=(0.0, 1e300))


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


def test_fso_range_problem_uses_the_codecs_arithmetic():
    """A maximum just past the last code: exact arithmetic keeps it, float32 scaling rounds it to 65536."""
    fso = nc.FixedScaleOffset(offset=200.0, scale=546.125, dtype="float32", astype="uint16")
    assert utils_cli.fso_range_problem((None, fso, None), {"Source_Min": 200.0, "Source_Max": 320.00092})
    assert utils_cli.fso_range_problem((None, fso, None), {"Source_Min": 200.0, "Source_Max": 320.0}) is None


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


def test_clamp_clips_on_decode_only():
    """Clamp is the identity on encode and clips on decode, in the array's dtype; NaN stays NaN."""
    c = utils.Clamp(minimum=0)
    x = np.array([[-1.0, 0.0, 0.5, np.nan, 2.0]], dtype="f4")
    assert np.array_equal(c._codec.encode(x), x, equal_nan=True)
    d = c._codec.decode(x.copy())
    assert d.dtype == x.dtype and np.array_equal(d, [[0.0, 0.0, 0.5, np.nan, 2.0]], equal_nan=True)
    assert utils.Clamp(maximum=1)._codec.decode(x.copy())[0, -1] == 1.0
    assert c.to_dict() == {"name": "numcodecs.clamp", "configuration": {"minimum": 0.0}}
    assert utils.codec_label(utils.Clamp(minimum=0, maximum=100)) == "clamp(maximum=100.0, minimum=0.0)"
    for bad in ({}, {"minimum": float("nan")}, {"minimum": 1, "maximum": 0}):
        with pytest.raises(ValueError):
            utils.Clamp(**bad)
    with pytest.raises(TypeError, match="float arrays"):
        c._codec.decode(np.array([1, 2], dtype="i4"))


def test_clamp_chain_round_trips_through_zarr_and_its_json():
    """The clamp is a codec of the pipeline: zarr.json carries it first in the filter chain, so it runs last
    on decode; no cell's error grows; the JSON identity survives from_dict(to_dict), a chain as a list."""
    rng = np.random.default_rng(0)
    f = np.where(rng.random((4, 46, 90)) < 0.3, 0.0, rng.random((4, 46, 90)) * 0.01).astype("f4")
    zf, zstd = utils.ZFPYRank(mode=4, tolerance=0.01), nc.Zstd(level=3)

    def roundtrip(filt):
        z = zarr.create_array(store=zarr.storage.MemoryStore(), shape=f.shape, dtype=f.dtype, chunks=(1, 46, 90),
                              **utils.codec_pipeline_kwargs(zstd, filt, zf))
        z[...] = f
        return z[...], [m["name"] for m in z.metadata.to_dict()["codecs"]]

    plain, _ = roundtrip(None)
    assert plain.min() < 0  # zfp rings below the zeros
    chains = ((utils.Clamp(minimum=0),), (utils.Clamp(minimum=0), nc.Quantize(digits=3, dtype="float32")))
    for filt in chains:
        d, names = roundtrip(filt)
        assert d.min() == 0 and names[:len(filt)] == [c.to_dict()["name"] for c in filt]
        key = utils.pipeline_json(zstd, filt, zf)
        assert utils.pipeline_json(*utils.pipeline_from_dict(json.loads(key))) == key
        assert not utils.pipeline_is_stock(json.loads(key))
    d, _ = roundtrip(chains[0])
    assert (np.abs(d - f) <= np.abs(plain - f)).all()
    one, two = (json.loads(utils.pipeline_json(zstd, filt, zf))["filter"] for filt in chains)
    assert one == {"name": "numcodecs.clamp", "configuration": {"minimum": 0.0}}  # a chain of one is that codec
    assert [c["name"] for c in two] == ["numcodecs.clamp", "numcodecs.quantize"]
    assert utils.pipeline_name(zstd, chains[1], zf).startswith("zstd(level=3) | clamp(minimum=0.0)+quantize(")


def test_clamp_pairing_rules():
    c, zf, q = utils.Clamp(minimum=0), utils.ZFPYRank(mode=2, rate=8), nc.Quantize(digits=3, dtype="float32")
    assert utils.combo_is_valid((c,), zf, None) and utils.combo_is_valid((c, q), zf, nc.Zstd(level=3))
    assert not utils.combo_is_valid((c,), nc.PCodec(level=8), None)  # nothing to clip behind a lossless serializer
    assert not utils.combo_is_valid((c,), None, None)
    assert not utils.combo_is_valid((q, c), zf, None)                 # last on decode means first in the chain
    assert not utils.combo_is_valid((c, c), zf, None)
    assert not utils.combo_is_valid((q, q), zf, None)                 # no other chain
    assert not utils.combo_is_valid((c, nc.BitRound(keepbits=7)), zf, None, dtype=np.dtype("f4"))  # the old rules hold behind it
    assert utils.combo_is_valid((c, nc.BitRound(keepbits=23)), zf, None, dtype=np.dtype("f4"))


@pytest.mark.ebcc
def test_ebcc_clamp_pairing_rules():
    pytest.importorskip("ebcc")
    c, ebcc = utils.Clamp(minimum=0), utils.EBCC.from_params(46, 90, 0.1)
    cast = nc.AsType(encode_dtype="float32", decode_dtype="float64")
    assert utils.combo_is_valid((c,), ebcc, None) and utils.combo_is_valid((c, cast), ebcc, None)
    assert not utils.combo_is_valid((c, nc.BitRound(keepbits=7)), ebcc, None)
    assert not utils.combo_is_valid((c,), ebcc, nc.Zstd(level=3))


def test_config_space_clamps_the_lossy_serializers_only():
    """--clamp-to-bounds puts the clamp in front of every zfp/EBCC combo and nowhere else, after the
    --max-evals subset is drawn: the same combos, with and without it."""
    comps, filts, sers = space("float32")
    plain = utils_cli.sweep_config_space(comps, filts, sers, 50, 0, np.dtype("float32"), True)
    clamped = utils_cli.sweep_config_space(comps, filts, sers, 50, 0, np.dtype("float32"), True,
                                           clamp=utils.Clamp(minimum=0))
    assert len(plain) == len(clamped) == 50
    for (c, f, s), (c2, f2, s2) in zip(plain, clamped):
        assert c is c2 and s is s2
        if utils.lossy_serializer(s):
            chain = utils.filter_codecs(f2)
            assert isinstance(chain[0], utils.Clamp) and chain[1:] == utils.filter_codecs(f)
        else:
            assert f2 is f
    kinds = {utils.lossy_serializer(s) for _, _, s in plain}
    assert kinds == {True, False}


def test_validate_pipeline_refuses_a_clamp_on_an_integer_field():
    da = xr.DataArray(np.zeros((4, 8), "i4"), dims=("time", "ncells"))
    with pytest.raises(click.ClickException, match="float field"):
        utils_cli.validate_pipeline((None, (utils.Clamp(minimum=0),), utils.ZFPYRank(mode=2, rate=8)), da, "x")
