"""README.md, "Reading a store without dc_toolkit", checked in a bare zarr client: the same site-packages minus
dc_toolkit (and its entry points), mpi4py, ebcc and zarr_any_numcodecs, started with python -S."""
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest
import zarr

from conftest import REPO
from dc_toolkit import utils

READ = ("import sys, zarr, numpy as np\n"
        "g = zarr.open_group(sys.argv[1], mode='r', use_consolidated=(sys.argv[3] == 'consolidated'))\n"
        "a = g[sys.argv[2]][...]\n"
        "print('READ', a.shape, repr(float(a[0, 0, 0])))\n")


@pytest.fixture(scope="module")
def bare_site(tmp_path_factory):
    site = pathlib.Path(zarr.__file__).resolve().parents[1]
    d = tmp_path_factory.mktemp("bare_site")
    for p in site.iterdir():
        if not p.name.startswith(("dc_toolkit", "__editable__", "ebcc", "zarr_any_numcodecs", "mpi4py")) \
                and p.suffix != ".pth":
            (d / p.name).symlink_to(p)
    return str(d)


def bare(code, site, *args):
    return subprocess.run([sys.executable, "-S", "-c", code, *map(str, args)], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": site}, timeout=300)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    path = tmp_path_factory.mktemp("stores") / "s.zarr"
    x = np.random.default_rng(0).normal(280, 5, (4, 32, 32)).astype("f4")
    g = zarr.open_group(str(path), mode="w", zarr_format=3)
    for name, ser in (("flat", utils.ZFPYFlat(mode=4, tolerance=0.01)), ("stock", utils.ZFPYRank(mode=4, tolerance=0.01))):
        g.create_array(name, shape=x.shape, dtype=x.dtype, chunks=(1, 32, 32), serializer=ser, compressors=None)[...] = x
    zarr.consolidate_metadata(str(path))
    return path, x


def readme_snippet():
    section = (REPO / "README.md").read_text().split("## Reading a store without dc_toolkit", 1)[1]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


def test_the_client_is_bare(bare_site):
    r = bare("import dc_toolkit", bare_site)
    assert r.returncode != 0 and "No module named 'dc_toolkit'" in r.stderr


def test_stock_array_reads_without_consolidated_metadata(bare_site, store):
    r = bare(READ, bare_site, store[0], "stock", "scan")
    assert r.returncode == 0 and "READ (4, 32, 32)" in r.stdout, r.stderr[-2000:]


def test_flat_array_needs_the_snippet(bare_site, store):
    r = bare(READ, bare_site, store[0], "flat", "scan")
    assert r.returncode != 0, r.stdout
    print(r.stderr.strip().splitlines()[-1])


def test_readme_claim_consolidated_store_fails_for_its_other_arrays(bare_site, store):
    """'A store holding either fails at zarr.open in a client without them, even for its other arrays'."""
    r = bare(READ, bare_site, store[0], "stock", "consolidated")
    print("exit", r.returncode, (r.stderr.strip().splitlines() or [""])[-1], r.stdout.strip())
    assert r.returncode != 0, r.stdout


def test_readme_snippet_decodes_zfpy_flat(bare_site, store):
    r = bare(readme_snippet() + READ, bare_site, store[0], "flat", "consolidated")
    assert r.returncode == 0, r.stderr[-2000:]
    assert abs(float(r.stdout.split()[-1]) - float(store[1][0, 0, 0])) <= 0.01
