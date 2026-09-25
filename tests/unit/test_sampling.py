"""The sampler: dim kinds, minimum sample, cascade/balanced plans, block midpoints among the varying indices,
and the numbers docs/SAMPLING.md and docs/PARALLELIZATION.md quote."""
import numpy as np
import pytest
import xarray as xr

from dc_toolkit import utils, utils_cli

SLAB = 1000 * 4
GiB = 2**30


def da3(T=8, V=120, N=1000):
    return xr.DataArray(np.zeros((T, V, N), "f4"), dims=("time", "height", "ncells"), name="x",
                        coords={"time": ("time", np.arange(T), {"units": "hours since 2020-01-01"}),
                                "height": ("height", np.arange(V), {"standard_name": "height"})})


def kept(s):
    return s.sizes["time"], s.sizes["height"]


def field(**sizes):
    return xr.DataArray(np.zeros(tuple(sizes.values()) + (4, 4), "f4"), dims=tuple(sizes) + ("lat", "lon"), name="x")


def test_minimum_sample_is_3_by_3():
    assert utils.minimum_sample(da3()) == (9 * SLAB, "3 time steps x 3 levels")


def test_minimum_sample_singular():
    d = xr.DataArray(np.zeros((1, 10, 4), "f4"), dims=("time", "lev", "ncells"))
    assert utils.minimum_sample(d)[1] == "1 time step x 3 levels"


def test_fewer_than_three_keeps_all():
    small = da3(T=2, V=2)
    assert utils.minimum_sample(small)[0] == small.nbytes


@pytest.mark.parametrize("slabs, want", [(3.2, (3, 3)), (9, (3, 3)), (12.8, (3, 4)), (25.6, (3, 8)),
                                         (51.2, (7, 7)), (1000, (8, 120))])
def test_cascade(slabs, want):
    assert kept(utils.build_representative_sample(da3(), int(slabs * SLAB), rank=1)) == want


@pytest.mark.parametrize("slabs, want", [(3.2, (3, 3)), (12.8, (3, 4))])
def test_balanced(slabs, want):
    assert kept(utils.build_representative_sample(da3(), int(slabs * SLAB), rank=1, policy="balanced")) == want


def test_block_midpoints():
    s = utils.build_representative_sample(da3(), int(3.2 * SLAB), rank=1)
    assert list(s.time.values) == [1, 4, 6] and list(s.height.values) == [20, 60, 100]


def test_midpoints_among_the_levels_that_vary():
    """The model-top levels on which a field is constant are not sampled."""
    s = utils.build_representative_sample(da3(), int(3.2 * SLAB), rank=1, candidates={"height": np.arange(40, 120)})
    assert list(s.height.values) == [53, 80, 106] and list(s.time.values) == [1, 4, 6]


def test_all_candidates_kept_when_they_fit():
    s = utils.build_representative_sample(da3(), int(40 * SLAB), rank=1, candidates={"height": np.array([3, 7])})
    assert list(s.height.values) == [3, 7]


def test_vertical_floor_lowered_to_fit_three_steps():
    assert kept(utils.build_representative_sample(da3(), 9 * SLAB, rank=1, vertical_floor=5)) == (3, 3)


def test_time_only_field_keeps_three_midpoints():
    d = xr.DataArray(np.zeros((8, 50, 40), "f4"), dims=("time", "lat", "lon"),
                     coords={"time": ("time", np.arange(8), {"units": "hours since 2020-01-01"})})
    s = utils.build_representative_sample(d, 2 * 2000 * 4, rank=1)
    assert list(s.time.values) == [1, 4, 6]


def test_no_time_or_vertical_dim_is_its_own_minimum():
    d = xr.DataArray(np.zeros((1000,), "f4"), dims=("ncells",), name="x")
    assert utils.minimum_sample(d) == (4000, "the whole field, which has no time or vertical dim to thin")


@pytest.mark.parametrize("sizes, desc", [
    ({"member": 2, "time": 2, "lev": 90}, "member=2 x time=2 x 3 levels"),
    ({"time": 2, "step": 2, "lev": 90}, "time=2 x step=2 x 3 levels"),
    ({"time": 10, "lev": 2, "ilev": 2}, "3 time steps x lev=2 x ilev=2"),
    ({"member": 2, "time": 5, "lev": 30}, "member=1 x time=3 x 3 levels"),
    ({"member_id": 50, "time": 10, "lev": 20}, "member_id=3 x time=1 x 3 levels"),
    ({"time": 2}, "2 time steps"), ({"time": 100}, "3 time steps"), ({"lev": 50}, "3 levels")])
@pytest.mark.parametrize("policy", ["cascade", "balanced"])
@pytest.mark.parametrize("mult", [1.0, 1.3, 2.0])
def test_sampler_keeps_the_minimum_within_budget(sizes, desc, policy, mult):
    d = field(**sizes)
    floor, got = utils.minimum_sample(d)
    assert got == desc
    t, v, _ = utils._classify_sample_dims(d)
    s = utils.build_representative_sample(d, int(floor * mult), rank=1, policy=policy)
    prod = lambda arr, group: int(np.prod([arr.sizes[n] for _, n in group])) if group else 1  # noqa: E731
    assert prod(s, t) >= min(3, prod(d, t)) and prod(s, v) >= min(3, prod(d, v))
    assert s.nbytes <= max(floor, int(floor * mult))


@pytest.mark.parametrize("vfloor", [3, None])
@pytest.mark.parametrize("mult", [1.0, 1.2, 1.5, 2.0, 3.0])
def test_two_small_vertical_dims_stay_within_budget(vfloor, mult):
    d = field(time=10, lev=2, ilev=2)
    floor, _ = utils.minimum_sample(d)
    s = utils.build_representative_sample(d, int(floor * mult), rank=1, vertical_floor=vfloor)
    assert s.nbytes <= int(floor * mult) and s.sizes["time"] >= 3


@pytest.mark.parametrize("da, axes", [
    (da3(), (2,)),
    (xr.DataArray(np.zeros((8, 50, 40), "f4"), dims=("time", "lat", "lon")), (1, 2)),
    (xr.DataArray(np.zeros((50, 40), "f4"), dims=("lat", "lon")), (0, 1)),
    (xr.DataArray(np.zeros((3, 4, 5, 6), "f4"), dims=("member_id", "time", "lat", "lon")), (2, 3)),
    (xr.DataArray(np.zeros((3, 2, 5, 6), "f4"), dims=("number", "isobaricInhPa", "latitude", "longitude"),
                  coords={"number": ("number", [0, 1, 2], {"standard_name": "realization"}),
                          "isobaricInhPa": ("isobaricInhPa", [500, 850], {"standard_name": "air_pressure"})}),
     (2, 3)),
])
def test_horizontal_axes(da, axes):
    assert utils.horizontal_axes(da) == axes


# ---- numbers the docs quote ---------------------------------------------------------------------------

@pytest.mark.parametrize("slabs, cascade, balanced", [(9, (3, 3), (3, 3)), (16, (3, 5), (4, 4)),
                                                      (32, (4, 8), (5, 6)), (96, (8, 12), (8, 12))])
def test_sampling_md_qc_table(slabs, cascade, balanced):
    """docs/SAMPLING.md, "3. The plan": R02B10 qc (time 8 x height 120) at 2.8, 5, 10 and 30 GiB."""
    assert kept(utils.build_representative_sample(da3(), slabs * SLAB, rank=1)) == cascade
    assert kept(utils.build_representative_sample(da3(), slabs * SLAB, rank=1, policy="balanced")) == balanced


@pytest.mark.parametrize("slabs, steps", [(3.2, 3), (6.4, 6)])
def test_sampling_md_tot_prec(slabs, steps):
    """docs/SAMPLING.md: a 2-D field of 96 steps keeps 3 time steps at 1 GiB and 6 at 2 GiB (slab 320 MiB)."""
    d = xr.DataArray(np.zeros((96, 1000), "f4"), dims=("time", "ncells"),
                     coords={"time": ("time", np.arange(96), {"units": "minutes since 2020-01-01"})})
    assert utils.build_representative_sample(d, int(slabs * SLAB), rank=1).sizes["time"] == steps


@pytest.mark.parametrize("ranks, cap_gib", [(16, 20.5), (32, 10.4), (64, 5.2)])
def test_sampling_md_memory_caps(ranks, cap_gib):
    """docs/SAMPLING.md, "2. The budget": the Santis caps (849.6 GiB, threshold 0.8, 16 MiB chunks)."""
    budget = int(849.6 * GiB * 0.8)
    assert round(utils_cli.max_sample_bytes_for_ranks(budget, ranks, 16) / GiB, 1) == cap_gib


def test_sampling_md_30_gib_needs_at_most_10_ranks():
    budget = int(849.6 * GiB * 0.8)
    assert utils_cli.max_sample_bytes_for_ranks(budget, 10, 16) >= 30 * GiB
    assert utils_cli.max_sample_bytes_for_ranks(budget, 11, 16) < 30 * GiB


def test_parallelization_md_329_gb():
    """docs/PARALLELIZATION.md: 'about 329 GB for a 5 GB sample and 32 ranks'."""
    assert round(utils_cli.node_steady_estimate_bytes(5 * 10**9, 32, 16) / 1e9) == 329


def test_estimate_and_cap_are_inverse():
    for ranks in (1, 8, 32, 64):
        cap = utils_cli.max_sample_bytes_for_ranks(500 * GiB, ranks, 16)
        assert utils_cli.node_steady_estimate_bytes(cap, ranks, 16) <= 500 * GiB
        assert utils_cli.node_steady_estimate_bytes(cap + 1, ranks, 16) > 500 * GiB


def test_sampling_md_chunk_shapes():
    """docs/SAMPLING.md: the sample's 16 MiB chunks hold 1 time step x 3 levels x 1.4 M cells, the production
    chunks 1 time step x 120 levels x 35 k cells; zfp encodes both as 2-D (size-1 axes dropped)."""
    dims = ("time", "height", "ncells")
    sample = utils.compute_chunk_shape_for_eval((3, 3, 83886080), "float32", 16, dims)
    production = utils.compute_chunk_shape_for_eval((8, 120, 83886080), "float32", 16, dims)
    assert (sample, production) == ((1, 3, 1398101), (1, 120, 34952))
    assert [len(utils.ZFPYRank.encode_shape(c)) for c in (sample, production)] == [2, 2]
