"""Run under 2+ ranks: scan_field over a dask array one of whose block reads raises; every rank must find the
field unreadable and none may hang."""
import dask
import dask.array as dsa
import numpy as np
import xarray as xr
from mpi4py import MPI

from dc_toolkit import utils


def bad():
    raise OSError("injected read failure")


comm = MPI.COMM_WORLD
blocks = [dsa.from_array(np.ones((10, 10), "f4"))] * 3 + [dsa.from_delayed(dask.delayed(bad)(), (10, 10), "f4")]
scan = utils.scan_field(xr.DataArray(dsa.concatenate(blocks, axis=0), dims=("time", "ncells")), comm)
ok = comm.allreduce(int(not scan.readable and scan.range is None), op=MPI.MIN)
if comm.rank == 0:
    print("scan_fault:", "PASS" if ok else f"FAIL {scan}")
