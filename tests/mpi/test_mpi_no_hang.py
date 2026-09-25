"""Multi-rank paths that end early: special fields, a named field whose sample does not vary, a sample that does
not fit, a held lock.  Every rank must reach the same end, and an error is printed once; a hang shows as
TimeoutExpired."""
import json
import os
import socket

import subprocess

import pytest

pytestmark = pytest.mark.mpi


def run(argv, timeout):
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=timeout)


def test_special_fields_end_on_every_rank(mpiexec, fields, tmp_path):
    r = run(mpiexec(4) + ["dc_toolkit", "evaluate_combos", fields["edge"], "--where-to-write", tmp_path,
                          "--l1-threshold", "0.01"], timeout=180)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert {p.name for p in tmp_path.glob("manifest_*.json")} == {f"manifest_{v}.json"
                                                                   for v in ("const", "zeros", "u8", "allnan")}


@pytest.mark.parametrize("file, extra, message", [
    ("diag", ("--field-to-compress", "d", "--eval-data-size-limit", "1KB"), "the sample holds the single value 0"),
    ("icon", ("--field-to-compress", "qc", "--inner-chunk-mib", "1000000"), "[memcheck] FATAL: the smallest sample"),
    ("icon", ("--field-to-compress", "qc", "--memory-threshold", "0.5"), "another sweep of qc is writing"),
])
def test_an_early_end_reaches_every_rank(mpiexec, fields, tmp_path, file, extra, message):
    if "another sweep" in message:  # a lock held by a live process outside Slurm
        (tmp_path / "sweep_qc.lock").write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))
    r = run(mpiexec(2) + ["dc_toolkit", "evaluate_combos", fields[file], "--where-to-write", tmp_path,
                          "--l1-threshold", "0.01", *extra], timeout=120)
    text = r.stdout + r.stderr
    assert r.returncode != 0 and message in text, text[-3000:]
    assert text.count(message) == 1, text[-3000:]
