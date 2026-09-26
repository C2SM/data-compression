"""Error metrics of the sweep and of compress's verify pass, the bounds count, the gradient pieces, and golden
values that change only with METRIC_DEFINITIONS."""
import tracemalloc

import dask.array as dsa
import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.codecs import numcodecs as nc

from dc_toolkit import utils, utils_cli

DIMS = ("time", "lat", "lon")


@pytest.fixture(scope="module")
def sst():
    x = (280 + 10 * np.random.default_rng(0).standard_normal((4, 32, 32))).astype("f4")
    x[0, :4, :] = np.nan
    return x


def pipelines(x):
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    fso = nc.FixedScaleOffset(offset=lo, scale=(2**16 - 1) / (hi - lo), dtype="float32", astype="uint16")
    return {"fso": (None, fso, None), "zfp": (None, None, utils.ZFPYRank(mode=4, tolerance=0.5)),
            "lossless": (nc.Zstd(level=6), None, None)}


@pytest.mark.parametrize("name, corrupt", [("fso", 128), ("zfp", 128), ("lossless", 0)])
def test_sweep_counts_fill_turned_into_data(sst, name, corrupt):
    _, err, _ = utils.evaluate_codec_pipeline(sst, DIMS, utils.codec_pipeline_kwargs(*pipelines(sst)[name]),
                                              chunks=(1, 32, 32))
    assert err["N_Corrupt"] == corrupt
    assert err["N_Valid"] == sst.size - 128


def test_corrupt_counts_a_changed_kind_of_non_finite_value():
    orig = np.array([np.nan, np.inf, -np.inf, np.nan, 1.0], "f4")
    dec = np.array([np.inf, np.nan, np.inf, np.nan, np.nan], "f4")  # NaN->Inf, Inf->NaN, -Inf->+Inf, kept, lost
    assert utils._error_sums(orig, dec, (5,))["n_corrupt"] == 4


def _chunk_bytes(codec, x, pipeline):
    with zarr.config.set({"codec_pipeline.path": f"zarr.core.codec_pipeline.{pipeline}"}):
        store = zarr.storage.MemoryStore()
        z = zarr.create_array(store=store, name="x", shape=x.shape, dtype=x.dtype, chunks=x.shape, serializer=codec,
                              compressors=None)
        z[...] = x
        return zarr.core.sync.sync(store.get("x/c/0/0/0", zarr.core.buffer.default_buffer_prototype())).to_bytes()


@pytest.mark.parametrize("codec", [utils.ZFPYRank(mode=2, rate=8), utils.ZFPYFlat(mode=2, rate=8)])
def test_zfp_encoders_fold_on_both_zarr_pipelines(codec):
    """zarr's synchronous codec pipeline (an opt-in) encodes through _encode_sync: the fold must hold there too."""
    x = np.random.default_rng(4).normal(size=(1, 3, 70000)).astype("f4")  # zfp cannot take 70000 cells at 3-D
    assert _chunk_bytes(codec, x, "FusedCodecPipeline") == _chunk_bytes(codec, x, "BatchedCodecPipeline")


def test_bounds_count_only_cells_the_round_trip_moves_out():
    orig = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], "f4")   # -1 and 2 lie outside [0, 1] already
    dec = np.array([-2.0, -0.1, 0.5, 1.2, 0.5], "f4")   # 0 -> -0.1 and 1 -> 1.2 leave the bounds
    acc = utils._error_sums(orig, dec, (5,), bounds=(0.0, 1.0))
    assert acc["n_bounds"] == 2
    assert utils._error_sums(orig, dec, (5,), bounds=(-0.2, 1.0))["n_bounds"] == 1  # the slack keeps -0.1
    assert utils._error_sums(orig, dec, (5,))["n_bounds"] == 0


def test_source_range_spans_the_finite_original(sst):
    errors, _ = utils._errors_from_sums(utils._error_sums(sst, sst, (1, 32, 32)), False)
    assert (errors["Source_Min"], errors["Source_Max"]) == (float(np.nanmin(sst)), float(np.nanmax(sst)))


def _persisted(x, cfg, block, tmp_path, inner=(1, 32, 32), shards=None, **kw):
    da = xr.DataArray(dsa.from_array(x, chunks=block), dims=DIMS)
    store = zarr.storage.LocalStore(str(tmp_path / "s.zarr"))
    return utils.persist_with_codec_pipeline(da, store, "x", utils.codec_pipeline_kwargs(*cfg), inner, shards, **kw)


def test_persist_and_sweep_agree(sst, tmp_path):
    """compress's fused write + verify and the sweep describe the same round trip."""
    cfg = (nc.Zstd(level=6), nc.BitRound(keepbits=7), None)
    ratio_s, sweep, _ = utils.evaluate_codec_pipeline(sst, DIMS, utils.codec_pipeline_kwargs(*cfg), chunks=(1, 32, 32),
                                                      q99_abs=285.0, bounds=(270.0, 300.0))
    ratio_p, verify, _ = _persisted(sst, cfg, (2, 32, 32), tmp_path, shards=(2, 32, 32), q99_abs=285.0,
                                    bounds=(270.0, 300.0))
    for key in ("Relative_Error_L1", "Relative_Error_L2", "Relative_Error_Linf", "Bias_Rel", "Q99_Rel",
                "N_Corrupt", "N_Bounds", "Source_Min", "Source_Max"):
        assert verify[key] == pytest.approx(sweep[key], rel=1e-9, abs=1e-12), key
    assert ratio_p > 1


def test_persist_refuses_blocks_that_split_a_shard(sst, tmp_path):
    with pytest.raises(AssertionError, match="whole write units"):
        _persisted(sst, (nc.Zstd(level=6), None, None), (1, 32, 32), tmp_path, shards=(2, 32, 32))


@pytest.mark.parametrize("shape, unit, block", [
    ((8, 120, 83886080), (8, 120, 34952), (8, 120, 139808)),   # native R02B10: 4 shards of 128 MiB per block
    ((8, 120, 1000), (8, 120, 1000), (8, 120, 1000)),           # one unit covers the field
    ((96, 83886080), (1, 4194304), (1, 83886080)),              # 2-D: 16 MiB units, widened along the cells
])
def test_read_blocks_hold_whole_units(shape, unit, block):
    got = utils_cli.read_blocks(shape, "float32", unit)
    assert got == block and all(g % u == 0 or g == s for g, u, s in zip(got, unit, shape))


def test_q99_cut_over_nonzero_values():
    rng = np.random.default_rng(0)
    z = np.zeros((4, 64, 64), "f4")
    flat = z.reshape(-1)
    idx = np.arange(0, flat.size, 400)
    flat[idx] = (rng.random(idx.size) * 5).astype("f4")
    cut, over_nonzero = utils_cli.q99_cut(z)
    assert over_nonzero and cut == float(np.quantile(np.abs(z[z != 0]), 0.99)) and cut > 0
    kw = utils.codec_pipeline_kwargs(nc.Zstd(level=6), nc.Quantize(digits=1, dtype="float32"), None)
    _, err, _ = utils.evaluate_codec_pipeline(z, DIMS, kw, chunks=(1, 64, 64), q99_abs=cut)
    dec, _ = utils._zarr_roundtrip(z, DIMS, kw, (1, 64, 64))
    tail = np.abs(z) >= cut
    manual = float(np.abs(dec[tail].astype("f8") - z[tail]).sum() / np.abs(z[tail].astype("f8")).sum())
    assert err["Q99_Rel"] == pytest.approx(manual, rel=1e-9)


def test_q99_cut_edge_cases():
    assert utils_cli.q99_cut(np.zeros(10, "f4")) == (0.0, False)
    assert utils_cli.q99_cut(np.full(10, np.nan)) == (None, False)
    assert utils_cli.q99_cut(np.array([-128] * 50 + [3] * 50, "i1")) == (128.0, False)


def _gradient_reference(o, d, axes):
    o, d = o.astype("f8"), d.astype("f8")
    err = ori = 0.0
    for ax in axes:
        do, dd = np.diff(o, axis=ax), np.diff(d, axis=ax)
        m = np.isfinite(do) & np.isfinite(dd)
        err += np.abs(dd[m] - do[m]).sum()
        ori += np.abs(do[m]).sum()
    return err / ori


@pytest.mark.parametrize("shape, axes, max_elems", [((500, 100), (0, 1), 4000), ((500, 100), (0,), 4000),
                                                    ((500, 100), (1,), 150), ((3, 7, 400), (1, 2), 64),
                                                    ((5000, 1000), (0, 1), 4 << 20)])
def test_gradient_in_pieces_equals_whole_array(shape, axes, max_elems):
    rng = np.random.default_rng(1)
    o = rng.normal(size=shape).cumsum(axis=0).astype("f4")
    o[(slice(10, 14),) + (slice(3, 5),) * (o.ndim - 1)] = np.nan
    d = (o + rng.normal(scale=0.01, size=o.shape)).astype("f4")
    got = utils._gradient_rel_l1(o, d, axes=axes, max_elems=max_elems)
    assert got == pytest.approx(_gradient_reference(o, d, axes), rel=1e-12)


@pytest.mark.parametrize("shape, axis, max_elems", [((100, 7), 0, 50), ((3, 7, 400), 2, 64), ((10, 10), 1, 5),
                                                   ((6, 9, 11), 1, 30)])
def test_gradient_pieces_cover_every_pair_once(shape, axis, max_elems):
    """Every neighbouring pair along `axis` lies in exactly one piece, and no piece exceeds max_elems."""
    covered = np.zeros(shape, int)  # counts pairs by their first index
    for sl in utils._gradient_pieces(shape, axis, max_elems):
        assert np.prod([s.stop - s.start for s in sl]) <= max_elems
        first = list(sl)
        first[axis] = slice(sl[axis].start, sl[axis].stop - 1)
        covered[tuple(first)] += 1
    pairs = [slice(None)] * len(shape)
    pairs[axis] = slice(0, shape[axis] - 1)
    assert (covered[tuple(pairs)] == 1).all()


def test_gradient_memory_stays_in_pieces():
    big = np.random.default_rng(2).normal(size=(3, 3, 2_000_000)).astype("f4")  # 3 x 3 slabs of an ICON field
    tracemalloc.start()
    utils._gradient_rel_l1(big, big, axes=(2,))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 200 * 2**20


def test_finite_range():
    assert utils.finite_range(np.full(5, np.nan)) is None
    assert utils.finite_range(np.array([np.inf, 1.0, -np.inf, 3.0, np.nan])) == (1.0, 3.0)
    assert utils.finite_range(np.full(7, 280.0, "f4")) == (280.0, 280.0)


def test_golden_metrics():
    """Fixed input, fixed pipeline: the values change only with the code that measures them, which
    utils.METRIC_DEFINITIONS and the measurement digest in sweep_state_{var}.json then track."""
    x = (280 + 10 * np.sin(np.linspace(0, 20, 4 * 32 * 32))).reshape(4, 32, 32).astype("f4")
    x[1, 0, :5] = np.nan
    ratio, err, eucd = utils.evaluate_codec_pipeline(
        x, DIMS, utils.codec_pipeline_kwargs(nc.Zstd(level=6), nc.BitRound(keepbits=6), None), chunks=(1, 32, 32),
        q99_abs=289.0, bounds=(273.0, 287.0), compute_gradient=True, gradient_axes=(1, 2))
    got = {k: err[k] for k in ("Relative_Error_L1", "Relative_Error_L2", "Relative_Error_Linf", "Bias_Rel",
                               "Q99_Rel", "Grad_Rel", "N_Corrupt", "N_Bounds", "N_Valid")}
    assert got == pytest.approx(GOLDEN, rel=1e-6), got


GOLDEN = {"Relative_Error_L1": 0.003972591126180994, "Relative_Error_L2": 0.004527047414506878,
          "Relative_Error_Linf": 0.006896551724137931, "Bias_Rel": 3.165570113717022e-05,
          "Q99_Rel": 0.00572476455810689, "Grad_Rel": 1.2573106803576688, "N_Corrupt": 0, "N_Bounds": 352,
          "N_Valid": 4091}
