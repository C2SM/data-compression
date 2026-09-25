"""scan_field: the one pass over the whole field (range, NaN count, the time steps and levels that vary), and
open_dataset's default fill for netCDF floats that declare none."""
import dask
import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

from dc_toolkit import utils


def cube(x):
    T, V = x.shape[:2]
    return xr.DataArray(dsa.from_array(x, chunks=(1, 2, x.shape[2])), dims=("time", "height", "ncells"),
                        coords={"time": ("time", np.arange(T), {"units": "hours since 2020-01-01"}),
                                "height": ("height", np.arange(V), {"standard_name": "height"})})


def test_range_nonfinite_and_varying():
    x = np.zeros((4, 6, 50), "f4")
    x[1, 3] = np.linspace(-2, 5, 50)
    x[2, 4, :3] = [np.nan, np.inf, -np.inf]
    scan = utils.scan_field(cube(x))
    assert scan.readable and scan.range == (-2.0, 5.0) and scan.nonfinite == 3
    assert list(scan.varying["time"]) == [False, True, False, False]
    assert list(scan.varying["height"]) == [False, False, False, True, False, False]


@pytest.mark.parametrize("fill, want", [(0.0, (0.0, 0.0)), (np.nan, None)])
def test_constant_and_empty_fields(fill, want):
    scan = utils.scan_field(cube(np.full((2, 2, 10), fill, "f4")))
    assert scan.range == want and not scan.varying["time"].any()


def test_a_failed_read_makes_the_field_unreadable():
    def bad():
        raise OSError("injected")
    data = dsa.concatenate([dsa.from_array(np.ones((1, 2, 4), "f4")),
                            dsa.from_delayed(dask.delayed(bad)(), (1, 2, 4), "f4")])
    scan = utils.scan_field(xr.DataArray(data, dims=("time", "height", "ncells")))
    assert not scan.readable and scan.range is None


def test_integer_field():
    scan = utils.scan_field(xr.DataArray(dsa.from_array(np.arange(12, dtype="i2").reshape(3, 4)), dims=("t", "x")))
    assert scan.range == (0.0, 11.0) and scan.nonfinite == 0


def test_undeclared_netcdf_fill_is_masked(tmp_path):
    """A float variable without _FillValue keeps netCDF's default fill in unwritten cells."""
    netCDF4 = pytest.importorskip("netCDF4")
    path = tmp_path / "f.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("x", 6)
        v = ds.createVariable("v", "f4", ("x",), fill_value=False)  # no _FillValue attribute
        v.set_auto_mask(False)
        v[:3] = [1.0, 2.0, 3.0]
        v[3:] = netCDF4.default_fillvals["f4"]  # what a writer that never filled them leaves
    da = utils.open_dataset(str(path), "v")["v"]
    assert "_FillValue" not in xr.open_dataset(path)["v"].encoding
    assert np.isnan(da.values[3:]).all() and list(da.values[:3]) == [1.0, 2.0, 3.0]
