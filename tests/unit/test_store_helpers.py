"""--pipeline parsing, per-rank files, locks, staging, promotion and consolidation of the merged store, and the
checks compress makes before it trusts a manifest."""
import json
import os
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import click
import numpy as np
import pytest
import zarr

from dc_toolkit import utils_cli

PIPE = {"compressor": {"name": "numcodecs.zstd", "configuration": {"level": 3}}, "filter": None, "serializer": None}


@pytest.mark.parametrize("form", ["inline", "at", "path", "manifest"])
def test_parse_pipeline_forms(tmp_path, form):
    (tmp_path / "p.json").write_text(json.dumps(PIPE))
    (tmp_path / "manifest_t.json").write_text(json.dumps({"best": {"name": "x", "pipeline": PIPE}}))
    text = {"inline": json.dumps(PIPE), "at": f"@{tmp_path / 'p.json'}", "path": str(tmp_path / "p.json"),
            "manifest": str(tmp_path / "manifest_t.json")}[form]
    assert utils_cli.parse_pipeline_arg(text) == PIPE


@pytest.mark.parametrize("text", ["", "   ", "[1, 2]", '{"best": null}', "/no/such/file.json", "{not json"])
def test_parse_pipeline_refuses(text):
    with pytest.raises(click.ClickException):
        utils_cli.parse_pipeline_arg(text)


def test_rank_files_match_exactly(tmp_path):
    for name in ("config_space_t_rank0.csv", "config_space_t_rank12.csv", "config_space_t_rank0_rank0.csv",
                 "config_space_t.csv", "config_space_tt_rank0.csv", "failures_t_rank0.csv", "inflight_t_rank3.csv"):
        (tmp_path / name).write_text("x\n")
    assert [p.name for p in utils_cli.rank_files(tmp_path, "config_space", "t")] == [
        "config_space_t_rank0.csv", "config_space_t_rank12.csv"]
    assert [p.name for p in utils_cli.rank_files(tmp_path, "inflight", "t")] == ["inflight_t_rank3.csv"]


@pytest.mark.xfail(strict=True, reason="config_space_{var}.csv of a variable '<x>_rank0' is a rank file of <x>")
def test_config_space_table_is_not_a_rank_file(tmp_path):
    (tmp_path / "config_space_t_rank0.csv").write_text("x\n")  # the planned-combo table of variable "t_rank0"
    assert utils_cli.rank_files(tmp_path, "config_space", "t") == []


def test_partial_last_line_is_cut(tmp_path):
    p = tmp_path / "config_space_t_rank0.csv"
    p.write_bytes(b"a,b\n1,2\n3,")
    utils_cli.drop_partial_last_line(p)
    assert p.read_bytes() == b"a,b\n1,2\n"


def test_json_safe_and_state_digest():
    assert utils_cli.json_safe({"a": (1.0, float("inf")), "b": np.float32(2.5), "c": np.int64(3)}) == {
        "a": [1.0, None], "b": 2.5, "c": 3}
    assert utils_cli.state_digest({"a": 1, "b": [2]}) == utils_cli.state_digest({"b": [2], "a": 1})


def test_header_only_rank_csvs_keep_the_metrics_numeric(tmp_path):
    """A rank that evaluated nothing leaves a header-only CSV; the metrics must stay numbers in the union."""
    import csv as csv_module
    import pandas as pd
    with open(tmp_path / "config_space_t_rank1.csv", "w", newline="") as fh:
        csv_module.writer(fh).writerow(utils_cli.PARTIAL_CSV_COLUMNS)
    row = dict.fromkeys(utils_cli.PARTIAL_CSV_COLUMNS, "")
    row.update(name="a", pipeline='{"p": "a"}', ratio=2.0, l1_rel=0.001, l2_rel=0.001, linf_rel=0.001, bias_rel=0.0,
               grad_rel=0.01, n_corrupt=0, n_bounds=0, eucd=1.0, keep=True)
    with open(tmp_path / "config_space_t_rank0.csv", "w", newline="") as fh:
        w = csv_module.DictWriter(fh, fieldnames=utils_cli.PARTIAL_CSV_COLUMNS)
        w.writeheader()
        w.writerow(row)
    prev = utils_cli.read_rank_csvs(tmp_path, "t")
    assert len(prev) == 1 and pd.api.types.is_float_dtype(prev["l1_rel"])
    opts = SimpleNamespace(extremes_sensitive=False, gradient_gate=True, gradient_shortcircuit=True)
    thr = {"l1": 0.01, "l2": 0.02, "linf": 0.1, "bias": 0.005, "q99": float("inf")}
    assert list(utils_cli.reusable_rows(prev, None, opts, thr)) == [True]


def test_measurement_digest_is_stable():
    assert utils_cli.measurement_digest() == utils_cli.measurement_digest() and len(utils_cli.measurement_digest()) == 16


# ---- locks ---------------------------------------------------------------------------------------------

def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_lock_is_exclusive_and_released(tmp_path):
    lock = tmp_path / "x.lock"
    assert utils_cli.acquire_lock(lock) is None
    owner = json.loads(lock.read_text())
    assert owner["pid"] == os.getpid() and owner["host"] == socket.gethostname()
    lock.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))  # a live owner, outside Slurm
    assert "pid" in utils_cli.acquire_lock(lock)
    utils_cli.release_lock(lock)
    assert not lock.exists()


def test_a_dead_owners_lock_is_taken_over(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text(json.dumps({"host": socket.gethostname(), "pid": _dead_pid()}))
    assert utils_cli.acquire_lock(lock) is None and json.loads(lock.read_text())["pid"] == os.getpid()


def test_a_lock_being_written_is_respected(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text("")
    assert "taken" in utils_cli.acquire_lock(lock)
    old = time.time() - 3600
    os.utime(lock, (old, old))
    assert utils_cli.acquire_lock(lock) is None  # an empty lock an hour old: its owner died while writing it


@pytest.mark.parametrize("owner, alive", [({"slurm_job_id": "1", "slurm_step_id": "0"}, None),
                                          ({"host": "elsewhere", "pid": 1}, True)])
def test_lock_owner_on_another_host(owner, alive, monkeypatch):
    if alive is None:  # a Slurm step: squeue decides; a job that does not exist is gone
        def fake(argv, **kw):
            return SimpleNamespace(returncode=1, stdout="", stderr="slurm_load_jobs error: Invalid job id specified")
        monkeypatch.setattr(utils_cli.subprocess, "run", fake)
        alive = False
    assert utils_cli._lock_owner_alive(owner) is alive


def test_squeue_lists_the_owners_step(monkeypatch):
    monkeypatch.setattr(utils_cli.subprocess, "run",
                        lambda argv, **kw: SimpleNamespace(returncode=0, stdout="7.0\n7.2\n", stderr=""))
    assert utils_cli._lock_owner_alive({"slurm_job_id": "7", "slurm_step_id": "2"})
    assert not utils_cli._lock_owner_alive({"slurm_job_id": "7", "slurm_step_id": "1"})
    monkeypatch.setattr(utils_cli.subprocess, "run",  # an array task's steps print as <array>_<task>.<step>
                        lambda argv, **kw: SimpleNamespace(returncode=0, stdout="878981_2.0\n", stderr=""))
    assert utils_cli._lock_owner_alive({"slurm_job_id": "878983", "slurm_step_id": "0"})


def test_a_lock_of_this_very_step_is_an_earlier_incarnation(monkeypatch):
    """A requeued job reruns its steps under the same ids: their old locks are stale unless the pid lives here."""
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "4")
    owner = {"slurm_job_id": "123", "slurm_step_id": "4", "host": socket.gethostname()}
    assert not utils_cli._lock_owner_alive({**owner, "pid": _dead_pid()})
    assert utils_cli._lock_owner_alive({**owner, "pid": os.getpid()})


def test_two_runs_taking_over_one_stale_lock(tmp_path, monkeypatch):
    """The stale lock is removed only if it is still the one judged, and a run keeps the lock only if it still
    names it after taking it."""
    lock = tmp_path / "x.lock"
    lock.write_text(json.dumps({"host": "gone", "pid": 999999}))
    other = {"host": "other", "pid": 12345, "time": "now"}

    def judged_while_another_takes_it(owner):
        if owner.get("pid") == 999999:
            lock.write_text(json.dumps(other))  # the other run replaced the stale lock meanwhile
            return False
        return True
    monkeypatch.setattr(utils_cli, "_lock_owner_alive", judged_while_another_takes_it)
    assert "pid 12345" in utils_cli.acquire_lock(lock)
    assert json.loads(lock.read_text())["pid"] == 12345
    lock.unlink()

    monkeypatch.setattr(utils_cli.time, "sleep", lambda s: lock.write_text(json.dumps(other)))
    assert "pid 12345" in utils_cli.acquire_lock(lock)  # replaced right after this run created it
    utils_cli.release_lock(lock)
    assert lock.exists()  # not this run's lock


# ---- store ---------------------------------------------------------------------------------------------

def _store_with(path, names):
    g = zarr.open_group(str(path), mode="a", zarr_format=3)
    for n in names:
        g.create_array(n, shape=(4,), dtype="f4", overwrite=True)[...] = np.arange(4, dtype="f4")
    return g


def test_promote_replaces_and_consolidate_discards_leftovers(tmp_path):
    merged = tmp_path / "d.zarr"
    _store_with(merged, ["t"])
    utils_cli.consolidate_store(str(merged))
    assert "consolidated_metadata" in json.loads((merged / "zarr.json").read_text())
    _store_with(utils_cli.staging_path(str(merged)), ["t"])  # a verified rewrite of t
    utils_cli.promote_staged(str(merged), "t")
    assert not utils_cli.staging_path(str(merged)).exists()
    assert "consolidated_metadata" not in json.loads((merged / "zarr.json").read_text())  # stale listing dropped
    _store_with(utils_cli.staging_path(str(merged)), ["q"])  # a killed run's unfinished write
    assert utils_cli.consolidate_store(str(merged)) == ["t"]
    assert not utils_cli.staging_path(str(merged)).exists()


def test_promote_of_a_new_array_drops_the_listing(tmp_path):
    """Consolidated readers would not see the new array until the next consolidation."""
    merged = tmp_path / "d.zarr"
    _store_with(merged, ["t"])
    utils_cli.consolidate_store(str(merged))
    _store_with(utils_cli.staging_path(str(merged)), ["q"])
    utils_cli.promote_staged(str(merged), "q")
    assert "consolidated_metadata" not in json.loads((merged / "zarr.json").read_text())


def test_an_array_set_aside_by_a_killed_promotion_comes_back(tmp_path):
    merged = tmp_path / "d.zarr"
    _store_with(merged, ["t"])
    staging = utils_cli.staging_path(str(merged))
    _store_with(staging, ["t"])
    os.replace(merged / "t", staging / "t.__replaced__")  # killed between the two renames
    utils_cli.remove_staged(str(merged))
    assert (merged / "t" / "zarr.json").is_file() and not staging.exists()


def test_pipeline_needs_vars(tmp_path):
    opts = SimpleNamespace(where_to_write=str(tmp_path), vars_filter=None, pipeline=json.dumps(PIPE),
                           stock_codecs_only=False, dataset_file="x.nc")
    with pytest.raises(click.ClickException, match="--pipeline needs --vars"):
        utils_cli.compress_candidates(opts)


def _sweep_dir(tmp_path, digest_ok=True, state=True):
    s = {"a": 1}
    if state:
        (tmp_path / "sweep_state_t.json").write_text(json.dumps(s))
    manifest = {"best": {"name": "z", "pipeline": PIPE, "ratio": 2.0}, "dataset_file": "x.nc",
                "sweep_state_digest": utils_cli.state_digest(s) if digest_ok else "other"}
    (tmp_path / "manifest_t.json").write_text(json.dumps(manifest))
    return SimpleNamespace(where_to_write=str(tmp_path), vars_filter=None, pipeline=None, stock_codecs_only=False,
                           dataset_file="x.nc")


@pytest.mark.parametrize("digest_ok, state, reason", [(True, True, None), (False, True, "does not match"),
                                                      (True, False, "is missing")])
def test_compress_trusts_only_the_last_sweeps_manifest(tmp_path, digest_ok, state, reason):
    """A manifest a later, unfinished sweep of the field superseded is not used."""
    candidates, _, dropped = utils_cli.compress_candidates(_sweep_dir(tmp_path, digest_ok, state))
    if reason is None:
        assert [c["var"] for c in candidates] == ["t"] and not dropped
    else:
        assert not candidates and reason in dropped["t"]


def test_an_unparsable_manifest_drops_the_field(tmp_path):
    opts = _sweep_dir(tmp_path)
    (tmp_path / "manifest_t.json").write_text("{truncated")
    candidates, _, dropped = utils_cli.compress_candidates(opts)
    assert not candidates and "cannot parse" in dropped["t"]


def test_results_without_a_manifest_are_reported(tmp_path):
    opts = _sweep_dir(tmp_path)
    (tmp_path / "results_q.parquet").write_bytes(b"")
    _, _, dropped = utils_cli.compress_candidates(opts)
    assert "has no manifest_q.json" in dropped["q"]


def test_chunk_geometry_precedence():
    opts = SimpleNamespace(inner_chunk_mib=None, max_inner_chunk_mib=64, spatial_split=None)
    geometry, sources = utils_cli.chunk_geometry(opts, {"args": {"inner_chunk_mib": 8}})
    assert geometry == {"inner_chunk_mib": 8, "max_inner_chunk_mib": 64, "spatial_split": True}
    assert sources == {"inner_chunk_mib": "manifest", "max_inner_chunk_mib": "cli", "spatial_split": "default"}


@pytest.mark.parametrize("avail, want", [(100 * 2**30, 8), (10 * 2**30, 3), (None, 8)])
def test_write_concurrency_fits_the_memory(avail, want, monkeypatch):
    monkeypatch.setattr(utils_cli, "available_memory", lambda threshold: avail)
    opts = SimpleNamespace(verify=True, memory_threshold=0.8, threads=8)
    assert utils_cli.write_concurrency(512 * 2**20, 16 * 2**20, 100, opts, "x") == want


def test_write_concurrency_refuses_a_block_that_does_not_fit(monkeypatch):
    monkeypatch.setattr(utils_cli, "available_memory", lambda threshold: 2**30)
    opts = SimpleNamespace(verify=True, memory_threshold=0.8, threads=8)
    with pytest.raises(SystemExit):
        utils_cli.write_concurrency(512 * 2**20, 16 * 2**20, 100, opts, "x")
