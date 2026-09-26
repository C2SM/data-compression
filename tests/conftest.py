"""Shared fixtures: pinned thread pools, the bundled TIGGE file, tiny synthetic fields, the CLI, an MPI launcher."""
import os
import pathlib
import shlex
import shutil

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLOSC_NTHREADS",
               "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_THREAD_LIMIT")
for _v in THREAD_VARS:  # before numpy and dc_toolkit load: evaluate_combos and compress refuse to start otherwise
    os.environ[_v] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
TIGGE = REPO / "netCDF_files" / "tigge_pl_t_q_dx=2_2024_08_02.nc"


def invoke(*args, code=0):
    """Run `dc_toolkit ARGS` in-process (one MPI rank); assert its exit status unless `code` is None."""
    from click.testing import CliRunner
    from dc_toolkit.cli import cli
    result = CliRunner().invoke(cli, [str(a) for a in args])
    if code is not None:
        assert result.exit_code == code, (f"dc_toolkit {' '.join(map(str, args))}: exit {result.exit_code}, "
                                          f"expected {code}\n{result.output[-3000:]}\n{result.exception!r}")
    return result


@pytest.fixture(scope="session")
def tigge():
    if not TIGGE.is_file():
        pytest.skip(f"bundled file missing: {TIGGE}")
    return str(TIGGE)


def _tcoord(n):
    return ("time", np.arange(n, dtype="f8"), {"units": "hours since 2021-01-05", "standard_name": "time"})


def _hcoord(n):
    return ("height", np.arange(n, dtype="f8"), {"standard_name": "height", "axis": "Z"})


@pytest.fixture(scope="session")
def fields(tmp_path_factory):
    """Small synthetic files, written once per session: {name: path}."""
    out = tmp_path_factory.mktemp("fields")
    rng = np.random.default_rng(0)
    lat, lon = np.linspace(-90, 90, 46), np.linspace(0, 356, 90)  # sides >= 32: an EBCC tile fits
    ll = {"lat": ("lat", lat, {"units": "degrees_north", "standard_name": "latitude"}),
          "lon": ("lon", lon, {"units": "degrees_east", "standard_name": "longitude"})}
    LAT, LON = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")

    def smooth(t):
        return 285 + 15 * np.cos(LAT) + 3 * np.sin(2 * LON + 0.3 * t) * np.cos(LAT) ** 2

    paths = {k: str(out / f"{k}.nc") for k in ("edge", "nan", "icon", "map", "ens", "diag", "geo")}
    xr.Dataset({  # constant, all-zero, constant uint8, all-NaN
        "const": (("time", "height", "lat", "lon"), np.full((4, 5, 46, 90), 273.15, "f4")),
        "zeros": (("time", "height", "lat", "lon"), np.zeros((4, 5, 46, 90), "f4")),
        "u8": (("lat", "lon"), np.full((46, 90), 7, "u1")),
        "allnan": (("time", "lat", "lon"), np.full((4, 46, 90), np.nan, "f4")),
    }, coords={"time": _tcoord(4), "height": _hcoord(5), **ll}).to_netcdf(paths["edge"])

    sst = np.stack([smooth(t) for t in range(6)]).astype("f4")  # an SST whose land cells are NaN fill
    sst[:, (np.sin(3 * LON) * np.cos(LAT) > 0.35) | (np.abs(lat)[:, None] > 75)] = np.nan
    xr.Dataset({"sst": (("time", "lat", "lon"), sst, {"units": "K"})},
               coords={"time": _tcoord(6), **ll}).to_netcdf(paths["nan"], encoding={"sst": {"_FillValue": np.float32(-9.99e-08)}})

    T, H, N = 8, 30, 5000  # ICON-like columns; slab 20 kB, minimum sample 3 x 3 slabs
    qc = np.zeros((T, H, N), "f4")
    wet = rng.random((T, H, N)) < 0.005
    wet[:, :10] = False  # the top 10 levels hold no cloud: the sample skips them
    qc[wet] = np.exp(rng.normal(-11, 1.2, wet.sum())).astype("f4")
    w = np.zeros((T, H, N), "f4")
    w[:, :3] = rng.normal(0, 0.5, (T, 3, N))  # varies on 3 levels only
    x = np.linspace(0, 20 * np.pi, N)
    ta = (220 + 2.2 * np.arange(H)[None, :, None]
          + 5 * np.sin(x[None, None, :] + 0.2 * np.arange(T)[:, None, None])).astype("f4")
    xr.Dataset({"qc": (("time", "height", "ncells"), qc), "w": (("time", "height", "ncells"), w),
                "ta": (("time", "height", "ncells"), ta)},
               coords={"time": _tcoord(T), "height": _hcoord(H)}).to_netcdf(paths["icon"])

    xr.Dataset({"orog": (("lat", "lon"), (2000 * np.cos(LAT) ** 4 * (1 + np.sin(5 * LON))).astype("f4")),
                "u10": (("time", "lat", "lon"), np.stack([smooth(t) - 285 for t in range(4)]).astype("f4"))},
               coords={"time": _tcoord(4), **ll}).to_netcdf(paths["map"])

    v = (250 + 10 * rng.standard_normal((6, 10, 8, 2000))).astype("f4")
    xr.Dataset({"tens": (("member_id", "time", "lev", "ncells"), v)},
               coords={"time": ("time", np.arange(10, dtype="f8"), {"units": "hours since 2021-01-01"})}
               ).to_netcdf(paths["ens"])

    d = np.zeros((8, 8, 500), "f4")  # varies on slabs (t, t+1 mod 8) only: every time step and level varies,
    for t in range(8):               # yet the 3 x 3 block midpoints (1, 4, 6) x (1, 4, 6) hit none of them
        d[t, (t + 1) % 8] = rng.normal(0, 1, 500)
    xr.Dataset({"d": (("time", "height", "ncells"), d)},
               coords={"time": _tcoord(8), "height": _hcoord(8)}).to_netcdf(paths["diag"])

    geo = (1000.0 * (11 - np.arange(12))[None, :, None] + rng.normal(0, 3, (4, 12, 500))).astype("f4")
    geo[:, 11] = np.linspace(0.001, 5.0, 500, dtype="f4")  # geopotential-like: nearest 0 on the lowest level only
    xr.Dataset({"geo": (("time", "height", "ncells"), geo, {"units": "m2 s-2", "valid_min": np.nan, "code": 6})},
               coords={"time": _tcoord(4), "height": _hcoord(12)}).to_netcdf(paths["geo"])
    return paths


@pytest.fixture
def tigge_copy(tigge, tmp_path):
    """The TIGGE file in tmp_path, for tests that change its mtime."""
    return str(shutil.copy(tigge, tmp_path / pathlib.Path(tigge).name))


@pytest.fixture(scope="session")
def mpiexec():
    """launcher(n) -> argv prefix starting n ranks: DC_TOOLKIT_MPIEXEC with {n} (e.g. 'mpirun --oversubscribe
    -n {n}' in CI, 'srun --ntasks={n} --cpus-per-task=1' on Santis), else mpiexec -n {n}; skips without one."""
    spec = os.environ.get("DC_TOOLKIT_MPIEXEC")
    if not spec:
        exe = shutil.which("mpiexec") or shutil.which("mpirun")
        if exe is None:
            pytest.skip("no MPI launcher: set DC_TOOLKIT_MPIEXEC")
        spec = exe + " -n {n}"
    return lambda n: shlex.split(spec.format(n=n))
