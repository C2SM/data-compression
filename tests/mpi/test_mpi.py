"""Multi-rank runs: the sweep's results do not depend on the rank count, a failed read reaches every rank, and
single-process commands refuse several ranks."""
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

pytestmark = pytest.mark.mpi
HERE = pathlib.Path(__file__).parent
SWEEP = ("--l1-threshold", "0.005", "--max-evals", "40", "--extremes-sensitive")


def run(argv, timeout=300):
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def by_ranks(mpiexec, tigge, tmp_path_factory):
    out = {}
    for n in (1, 2, 4):
        d = tmp_path_factory.mktemp(f"ranks{n}")
        r = run(mpiexec(n) + ["dc_toolkit", "evaluate_combos", tigge, "--where-to-write", d, *SWEEP])
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        out[n] = d
    return out


@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("var", ["t", "q"])
def test_results_do_not_depend_on_the_rank_count(by_ranks, n, var):
    a, b = (pd.read_parquet(by_ranks[k] / f"results_{var}.parquet").sort_values("pipeline").reset_index(drop=True)
            for k in (1, n))
    pd.testing.assert_frame_equal(a, b)
    assert len(list(by_ranks[n].glob(f"config_space_{var}_rank*.csv"))) == n


def test_failed_read_reaches_every_rank(mpiexec):
    r = run(mpiexec(2) + [sys.executable, HERE / "scan_fault.py"], timeout=120)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout + r.stderr


def test_compress_refuses_several_ranks(mpiexec, tigge, by_ranks):
    r = run(mpiexec(2) + ["dc_toolkit", "compress", tigge, by_ranks[2]], timeout=120)
    assert r.returncode != 0 and "not meant to run in parallel" in r.stdout + r.stderr


def test_compress_after_a_multi_rank_sweep(mpiexec, tigge, by_ranks):
    r = run(mpiexec(1) + ["dc_toolkit", "compress", tigge, by_ranks[4]])
    assert r.returncode == 0 and "[verify-gate] t: PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
