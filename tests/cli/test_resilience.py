"""A combo that kills its rank is retried alone and left out when it kills it again; a cancelled run
blames no combo; two sweeps of one field do not run into one directory."""
import json
import os
import pathlib
import socket
import subprocess
import sys

import pandas as pd
import pytest

from conftest import invoke

DRIVER = pathlib.Path(__file__).with_name("crash_driver.py")
TARGET = "'level': 22"  # Zstd(level=22), one of the three combos below
ARGS = ("--field-to-compress", "t", "--l1-threshold", "0.01", "--compressor-class", "zstd", "--filter-class", "none",
        "--serializer-class", "none")


@pytest.fixture
def run(mpiexec, tigge):
    """run(out, how, marker): the crash driver as one rank; an exit status != 0 means it died."""
    def launch(out, how="KILL", marker="-"):
        return subprocess.run(mpiexec(1) + [sys.executable, str(DRIVER), TARGET, how, str(marker), "evaluate_combos",
                                            tigge, "--where-to-write", str(out), *ARGS],
                              capture_output=True, text=True, timeout=600)
    return launch


@pytest.mark.mpi
def test_a_combo_that_kills_its_rank_is_isolated_then_left_out(run, tmp_path):
    first = run(tmp_path)
    assert first.returncode != 0, first.stdout[-2000:] + first.stderr[-2000:]
    journal = (tmp_path / "inflight_t_rank0.csv").read_text().split("\t")
    assert journal[0] == "crowd" and "level=22" in journal[1]
    second = run(tmp_path)  # the suspect runs alone, after the others, and kills the rank again
    assert second.returncode != 0 and "evaluating one at a time the 1 combo(s)" in second.stdout
    assert (tmp_path / "inflight_t_rank0.csv").read_text().startswith("alone\t")
    third = run(tmp_path)
    assert third.returncode == 0, third.stdout[-2000:] + third.stderr[-2000:]
    assert "leaving out 1 combo(s) that killed their rank when evaluated alone" in third.stdout
    m = json.loads((tmp_path / "manifest_t.json").read_text())
    assert m["crashed"] == ["zstd(level=22) | - | -"] and m["best"] is not None and m["num_rows"] == 2
    assert not (tmp_path / "inflight_t_rank0.csv").exists()


@pytest.mark.mpi
def test_a_suspect_that_survives_alone_is_recorded(run, tmp_path):
    marker = tmp_path / "crashed-once"
    assert run(tmp_path, marker=marker).returncode != 0
    second = run(tmp_path, marker=marker)
    assert second.returncode == 0 and "evaluating one at a time the 1 combo(s)" in second.stdout
    m = json.loads((tmp_path / "manifest_t.json").read_text())
    assert m["crashed"] == [] and m["num_rows"] == 3
    assert "zstd(level=22) | - | -" in set(pd.read_parquet(tmp_path / "results_t.parquet")["name"])


@pytest.mark.mpi
@pytest.mark.parametrize("how", ["TERM", "USR1"])
def test_a_terminated_run_blames_no_combo(run, tmp_path, how):
    """A signal from outside (a cancel, the walltime, --signal=USR1@T) does not make the combo a suspect."""
    marker = tmp_path / "terminated-once"
    assert run(tmp_path, how=how, marker=marker).returncode != 0
    assert (tmp_path / "inflight_t_rank0.csv").read_text() == ""
    second = run(tmp_path, how=how, marker=marker)
    assert second.returncode == 0 and "one at a time" not in second.stdout
    assert json.loads((tmp_path / "manifest_t.json").read_text())["num_rows"] == 3


def test_a_live_sweep_lock_is_respected(tigge, tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "sweep_t.lock").write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))
    out = invoke("evaluate_combos", tigge, "--where-to-write", tmp_path, *ARGS, code=1).output
    assert "another sweep of t is writing" in out
    (tmp_path / "sweep_t.lock").unlink()
    invoke("evaluate_combos", tigge, "--where-to-write", tmp_path, *ARGS)
    assert not (tmp_path / "sweep_t.lock").exists()
