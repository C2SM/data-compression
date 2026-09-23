# Handoff: the pure-MPI sweep of `dc_toolkit`, from Santis to the laptop and back

You are Claude Code on Christos Kotsalos's laptop (or back on Santis), in a checkout of
`C2SM/data-compression` on the branch `mpi-shmem-experimental` (remote head `02b55d9`, four commits on top
of `main` at `760fdce`). Christos owns the repo and wrote the backend. This file travels with him between
machines: read it first, do the laptop tests of section 4, and **write your findings back into section 6 of
this same file** so that the final check on Santis (section 7) can read them. Keep sections 1-5 and 7 as
they are; edit only section 6 unless a fact above turns out wrong, then correct it and say so in section 6.

## 1. What the branch does

`evaluate_combos` runs as plain MPI ranks, one per core. The ranks of a node share one copy of the sample
through an MPI-3 shared-memory window (`utils_cli.shared_sample_window`: the node's first rank allocates,
rank 0 fills, the node leaders broadcast, everyone maps it read-only, a checksum confirms the fill).
Every rank evaluates one pipeline at a time with zarr's synchronous API; zarr's own pool is pinned to one
worker; dask runs synchronously. Work over the whole sample happens once, not once per rank: rank 0
computes the q99 cut and broadcasts it, and the full-field range for FixedScaleOffset is one pass over
the file's blocks split across all ranks (`utils.full_field_data_range(da, comm)`).

Deleted with the branch: the thread pool of the sweep, the per-thread zarr event loops and the shared
codec executor ("the bypass"), and the sweep's `--threads-per-rank`, `--codec-threads`,
`--bypass-zarr-sync` and `--allow-multi-rank-per-node` options. Kept by design: the three write commands
(`compress`, `from_nc_to_zarr`, `from_zarr_to_netcdf`) still use dask's threaded scheduler in one process;
the Qt desktop UI keeps its worker thread (it only keeps the window responsive); zarr's codec adapter still
calls codecs through zarr's pool.

Launching: `srun --ntasks-per-node=32 --cpus-per-task=1` on a cluster, `mpirun -n <cores>` on a laptop. A
single rank on a multi-core machine prints `[topology] NOTE: one rank on N cores ... mpirun -n N`. The UIs
start the sweep under `mpirun -n <physical cores>` when an MPI launcher is on the PATH (`ui_mpirun`), and
`compress` as one process; the environment they pass sets `OMPI_ALLOW_RUN_AS_ROOT` for containers.

The commits: `0c7804f` code, `33371be` driver and docs (santis.run, PARALLELIZATION.md, intro.md, README,
install script), `02b55d9` the UI launcher counts physical cores. `main` is untouched.

## 2. Why: the measurements behind the decision (Santis, GH200 nodes, 288 cores, ~850 GiB budget)

- **Threads vs ranks, same 32 cores, 2000 combos on a 0.9 GB sample:** 32 ranks 529 s, 1 rank x 32
  threads with the bypass 585 s, results identical row for row. Ranks with private sample copies used
  85 GB against 42 GB, which is why the sample is shared.
- **Shared window vs threads, steady state (cgroup of the step, anonymous memory):** equal or lower for
  the ranks in every case: 2 GiB sample 34 vs 41 GiB, full 4.6 GiB field at 32 wide 161 vs 165, the 300 GiB
  native-grid field at 2/4/8 GiB samples 36/66/133 vs 37/70/139. Peaks match within the cost of the N
  interpreters (0.1 to 6.5 GiB).
- **Two start-up transients found and removed:** the q99 cut (about three sample-sized temporaries) and a
  window checksum that copied the sample; both ran on every rank at once and took a 32-rank node to
  449 GiB and 197 GiB before the fixes. Rule that follows: nothing may pass over the whole sample per rank.
- **Time on big netCDF files:** the HDF5 library serialises reader threads on a global lock, so processes
  read in parallel and threads do not. The 300 GiB field at 2/4/8 GiB samples took 250/382/752 s under
  ranks against 509/632/1225 s under threads; in the production driver 117 s against 795 s.
- **The `[memory]` model** (`S + R x 2 x S + R x 32 MiB` per node) is about twice too conservative
  (predicted 298 GiB, measured 165 for the 32-wide full field). Left at 2.0 on purpose.
- **Straggler tail** of the static split: 13% at 63 combos per rank; expected ~20% at 35 per rank on
  8 nodes. Fix (post-merge): a `Fetch_and_op` work counter on rank 0, about ten lines.

## 3. Validated on Santis before this handoff (all against `main` in identical venvs)

| check | result |
|---|---|
| `santis.run`, 3 R02B06 fields, 2 nodes, `--max-evals 320`, `COMPRESS=1` | identical tables, same winners, verify gate PASS on both trees |
| `santis.run`, R02B10 tot_prec 35 GiB and out_15 qc 300 GiB at the 1 GiB / 16-rank tier | identical, same winners |
| two variables in one run (TIGGE `t` and `q`), window freed per variable | each identical to its single-variable run |
| 16 ranks for 10 combos | correct banner, idle ranks harmless, identical rows |
| memory guard auto-shrink (5 GB sample, `--memory-threshold 0.05`, 8 ranks) | shrank to 2.5 GiB per the per-node model, ran |
| EBCC under ranks (TIGGE `t`, 60 + 7 EBCC combos) | 8 ranks identical to 1 rank |
| resume from rows recorded by the threaded code, partial resume, `compress` on the new manifest | reused / identical / PASS |
| smoke on 1 node, 2 nodes, 2 ranks x 2 threads, T1 to T3 resume | identical |

Cray MPICH throughout: `Win.Allocate_shared` up to 8 GiB windows on 32 ranks, no `/dev/shm` use.

## 4. The laptop tests (Open MPI, macOS or Linux, and Docker)

Everything that follows is what Santis cannot show: a different MPI, no cgroups, a small memory budget,
and the container. Nothing here needs a cluster. Record every result in section 6.

### 4.1 Environment

`git checkout mpi-shmem-experimental`, then `docs/intro.md` section 1: an MPI with `mpicc` (Homebrew
`open-mpi`, or `libopenmpi-dev openmpi-bin`), `python3 -m venv venv && source venv/bin/activate && bash
install_dc_toolkit.sh`, and the thread pins of intro section 1.4 in every shell. Note the OS, the MPI
implementation and version (`mpirun --version`), Python, `numcodecs` and `zarr` versions, physical and
logical core counts, RAM.

### 4.2 The sweep under mpirun (must)

```bash
FILE=netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc
N=<physical cores>
mpirun -n $N dc_toolkit evaluate_combos "$FILE" --where-to-write out/mpi --field-to-compress t --l1-threshold 0.005 --max-evals 300 --no-resume
mpirun -n 1  dc_toolkit evaluate_combos "$FILE" --where-to-write out/one --field-to-compress t --l1-threshold 0.005 --max-evals 300 --no-resume
```

Expect: `[topology] 1 node(s) x N rank(s)/node = N parallel evaluations, one shared sample per node`, a
`[memory]` line whose budget source is host RAM (no cgroup), the `NOTE` about one rank on N cores in the
second run, 300 rows in each `results_t.parquet`, and the two tables identical:

```bash
python - <<'PY'
import pandas as pd
a = pd.read_parquet("out/mpi/results_t.parquet").set_index("pipeline").sort_index()
b = pd.read_parquet("out/one/results_t.parquet").set_index("pipeline").sort_index()
cols = ["ratio", "l1_rel", "l2_rel", "linf_rel", "eucd", "keep"]
print("identical" if a.index.equals(b.index) and all(a[c].equals(b[c]) for c in cols) else "DIFFER", len(a))
PY
```

If Open MPI says there are not enough slots, note it: `-n` must not exceed the physical cores it counts
(`--oversubscribe` is the escape hatch, Open MPI only). Then `dc_toolkit compress "$FILE" out/mpi` as one
process (no mpirun): expect `[verify-gate] t: PASS`. Then run the sweep a second time into `out/mpi`
without `--no-resume`: expect `[resume] 300 of 300 combo(s) ... already recorded`.

### 4.3 A sample that does not fit (should)

The window must behave when the guard shrinks the sample. On a laptop the TIGGE file is too small to
matter, so lower the threshold instead and read the lines: `--memory-threshold 0.05` with `-n $N` should
still run (the budget is 5% of host RAM); report the `[memory]` line and whether `[memcheck]` shrank or
refused. If a bigger netCDF is at hand (any field of a few GB), run it with `--eval-data-size-limit 2GiB`
and `-n $N` and report the peak RSS of the processes (`ps -o rss` or Activity Monitor) against the
`[memory]` estimate.

### 4.4 The UIs (must)

- `dc_toolkit run_local_ui`: pick `t`, run the sweep, then compress with a chosen pipeline. Expect the
  sweep's log to show the `[topology]` line with N ranks (the UI prefixes `mpirun -n <physical cores>`),
  and compress to run as one process and succeed.
- `dc_toolkit run_web_ui`: the same through the browser.

### 4.5 Docker (must, if Docker is available)

Build the image (`docker build -t dc-toolkit .`) and run the README's "Running with MPI" example, Mac and
Linux form, with `-n 4`. The open question is the shared window under Open MPI inside a container: Open MPI
backs it with `/dev/shm`, whose Docker default is 64 MB. Expect either success, or an
`MPI.Win.Allocate_shared` failure (`[shared-sample] FATAL ...`); in the second case repeat with
`--shm-size=1g` on the `docker run` line and, if that fixes it, record it: the README's Docker commands
then need `--shm-size` and the container section a sentence about it. Also run
`docker run -p 8501:8501 dc-toolkit run_web_ui` and start a sweep from the browser: the UI runs
`mpirun` as root inside the container with `OMPI_ALLOW_RUN_AS_ROOT=1` in the environment; report whether
the sweep starts and how many ranks the log shows.

### 4.6 Read-through (should)

Read `docs/PARALLELIZATION.md`, the intro's sections 1, 3 and 10 and the README's parallelism and Docker
sections as a laptop user would; note anything that does not match what you saw.

## 5. Working rules

- Christos is the commit author; **no `Co-Authored-By` trailer**. Compact code, few comments ("why", not
  narration), no changelog language in comments or docs. Never commit data paths, user names or accounts.
- Do not touch `main`. Fixes the laptop tests require (the `--shm-size` note is the likely one) go on
  `mpi-shmem-experimental` with Christos's go; say in section 6 what changed and why. Push the branch
  after committing so Santis can pull it.
- This file is committed on the branch as test scaffolding and is dropped before the merge, as the
  earlier test folders were; it holds no paths beyond `$SCRATCH` and `$DYAMOND_DATA_ROOT`. Commit your
  section 6 on the branch and push, so that Santis reads it with `git pull`.

## 6. Laptop findings (written by Claude Code on the laptop)

Written on 2026-09-23 on the branch at `bffc985` (`02b55d9` plus the two commits that added this file).
Nothing in sections 1 to 5 turned out wrong; two expectations of section 4 are sharpened in 6.7. Only this
section changed on the branch: no code, no docs (6.6 lists what should change, for Christos's go).

### 6.1 Environment

| | |
|---|---|
| machine | Apple M2 Pro, 12 physical = 12 logical cores (no SMT), 32 GiB; macOS 27.0 (Darwin 27.0.0, arm64) |
| MPI | Open MPI 5.0.9 (Homebrew), `mpicc` on the PATH; 12 slots: `-n 12` runs as is, `-n 13` is refused ("not enough slots") |
| Python | 3.14.4 (Homebrew); `python3 -m venv venv && bash install_dc_toolkit.sh` as in intro 1.2 worked first time, every dependency had a 3.14 wheel, only `mpi4py` 4.1.2 built from source |
| libraries | numcodecs 0.17.0, zarr 3.4.0, numpy 2.5.3, xarray 2026.7.0, dask 2026.8.0, h5py 3.16.0, netCDF4 1.7.4, streamlit 1.64.0 |
| Docker | Docker Desktop 29.6.1; its VM has 12 CPUs and 11.7 GiB |
| EBCC | not on the laptop (no cmake); built into the image |

Every run below had the thread pins of intro 1.4, a scratch directory as cwd (the `out/` dirs live there)
and the bundled TIGGE file by absolute path. No `--oversubscribe` was needed.

### 6.2 The sweep under mpirun: PASS

| step | decisive lines |
|---|---|
| `mpirun -n 12`, 300 combos, 6.0 s | `[topology] 1 node(s) x 12 rank(s)/node = 12 parallel evaluations, one shared sample per node.` / `[memory] rank-0 transient peak ~= 0 MiB (building the sample); per-node steady ~= 0 MiB (shared sample) + ~2 MiB (12 ranks x 2.0x decode/encode cache) + ~384 MiB (ranks x 2 x inner_chunk_mib) = 387.1 MiB total.` / 300 rows |
| `mpirun -n 1`, 2.3 s | `[topology] NOTE: one rank on 12 cores.  A rank evaluates one pipeline at a time; start one rank per core to use them all (mpirun -n 12 dc_toolkit ...).` / `[memory] ... = 32.4 MiB total.` / 300 rows |
| the comparison | `identical 300` |
| `compress`, one process | `[verify-gate] t: PASS, production error norms are within the sweep thresholds.` / `[cr-drift] t: PASS (achieved 12.89x vs predicted 12.89x, drift +0.0%)` |
| second sweep into `out/mpi` | `[resume] 300 of 300 combo(s) of 't' are already recorded; skipping those.` |

The budget source is printed by the memcheck line, which every 12-rank run on this laptop shows before the
sample line, default `--eval-data-size-limit` included: `[memcheck] auto-shrunk sample budget from 4.7 GiB
(--eval-data-size-limit) to 1.0 GiB to stay under 0.80 x 32.0 GiB per node (from host total RAM (sysconf))
with 12 rank(s) per node.` (the model: (0.8 x 32 GiB - 12 x 32 MiB) / 25). Harmless for a field below the
shrunk budget; see 6.6.

Extras: `mpirun -n 2 dc_toolkit compress ...` aborts with "compress is not meant to run in parallel. Launch
it with a single process." The full space (17127 combos) on 12 ranks takes 15 to 19 s (the UI runs of 6.4)
and 22 s on 4 ranks in the container (6.5); the container's table and the laptop's agree on `keep` for all
17127 rows and on the winner, and differ only in the last bits of the float metrics (Linux/Python 3.13
against macOS/Python 3.14, e.g. 1251 ratios at 1e-16 relative). On one platform the tables are identical
across rank counts and between the two UIs.

### 6.3 A sample that does not fit: PASS, with a laptop caveat

- `--memory-threshold 0.05`, `-n 12`: `[memcheck] auto-shrunk sample budget from 4.7 GiB
  (--eval-data-size-limit) to 50.2 MiB to stay under 0.05 x 32.0 GiB per node (from host total RAM (sysconf))
  with 12 rank(s) per node.`; the sweep ran (60 rows).
- The refuse path, `--oversubscribe -n 64` at 0.05: `[memcheck] FATAL: cannot fit any sample with 64 rank(s)
  per node, inner_chunk_mib=16, node memory budget 32.0 GiB (from host total RAM (sysconf)) at threshold
  0.05.  Start fewer ranks per node or request more RAM.`, then MPI_ABORT, exit 1. Clean.
- No bigger netCDF was at hand, so one was made: float32 `t(time=128, lat=1800, lon=3600)`, 3.1 GiB, a
  smooth field plus noise, uncompressed netCDF4. `-n 12 --eval-data-size-limit 2GiB --compressor-class blosc
  --max-evals 24`:
  `[memcheck] auto-shrunk sample budget from 2.0 GiB (--eval-data-size-limit) to 1.0 GiB ...` /
  `[sample] field is 3.1 GiB > limit 1.0 GiB; policy=cascade; strided time=41/128 | preserved spatial: lat,
  lon -> 1013.5 MiB.` / `[memory] rank-0 transient peak ~= 2026 MiB (building the sample); per-node steady ~=
  1013 MiB (shared sample) + ~24323 MiB (12 ranks x 2.0x decode/encode cache) + ~384 MiB (ranks x 2 x
  inner_chunk_mib) = 25.1 GiB total.` Four runs, 67 to 106 s wall, exit 0, 24 rows, no FATAL.
  Measured with `ps` RSS every 0.5 s: peak sum over the 12 ranks 25.96 GiB, peak single rank 4.44 GiB
  (RSS counts the 1 GiB window in every rank, so the sum over-counts by up to 11 GiB). The machine view is
  the honest one: available RAM fell by 14.7 GiB and swap grew by 1.7 GiB, a footprint of about 16.5 GiB
  = S + 12 x ~1.3 GiB (the same 16 to 17 GiB in three runs), against the model's 25.1 GiB. The factor 2.0
  is ~1.5x conservative here, as on Santis.
  The caveat: the budget is the host's **total** RAM (32 GiB), but only 16 to 20 GiB was free (Docker
  Desktop, editor, browser). The run that started with 16.4 GiB available pushed 5.9 GiB to swap (macOS
  swap 2.9 -> 8.8 GiB); the laptop stayed usable and the rows were correct, but the guard cannot see what
  other applications hold, so "fits under 0.80 x total" does not mean "fits". Reference footprint of the
  interpreters alone (TIGGE, full space, 12 ranks): sum 3.65 GiB, max 0.41 GiB, 0.3 GiB per rank as
  PARALLELIZATION.md says.
- Where the window lives on macOS: Open MPI's `osc/sm` (`osc_sm_backing_directory` empty) creates it in the
  session directory under `$TMPDIR` and unlinks the file once every rank has attached; only the twelve
  16 MiB `btl/sm` segments stay visible. There is no `/dev/shm` to size on macOS.

### 6.4 The UIs: PASS

Both UIs start the sweep as `mpirun -n 12 dc_toolkit evaluate_combos ...` (`ui_mpirun`: 12 physical cores)
and `compress` as one process; 12 `config_space_t_rank*.csv` in `out/` after each sweep; the failures files
hold their header only.

- `run_local_ui`: the PyQt6 wheel installed on first use in 4 s. The window was driven offscreen
  (`QT_QPA_PLATFORM=offscreen`, the open and save dialogs replaced by fixed paths, plotly's `show()` muted);
  the command itself was also launched and showed its window process. Log: `[topology] 1 node(s) x 12
  rank(s)/node = 12 parallel evaluations, one shared sample per node.`, 17127 combos in 19 s, "15774
  combinations passed the gates", 15774 pipelines offered; compress with the first, `bz2(level=6) |
  bitround(keepbits=52) | zfpy_flat(mode=2, rate=8)`: `[verify-gate] t: PASS, production error norms are
  within the sweep thresholds.`, `[cr-drift] t: no predicted ratio for this pipeline; skipped.`, the store
  zipped and saved (4669 bytes).
- `run_web_ui` (headless streamlit, driven in headless Chromium through Playwright): upload of the TIGGE
  file, field `t`, "Evaluate combos": `[sweep] 17127 combos: 17127 from the 33 x 23 x 33 grid (valid
  pairings) + 0 EBCC; split across 12 rank(s), ~1428 per rank.`, 16 s, "15774 combinations passed the
  gates", table and plots; "Compress field" with the offered default: "Wrote t with bz2(level=6) |
  bitround(keepbits=52) | zfpy_flat(mode=2, rate=8).", the zip downloaded (4669 bytes). The status
  placeholder shows only the latest line, so the `[topology]` line is visible for a moment; the rank files
  are the proof.

### 6.5 Docker: PASS; the README's MPI commands need `--shm-size`

- The `Dockerfile` clones `main`, so it cannot build the branch. The image came from a copy with `git clone
  --branch mpi-shmem-experimental` (the only change): 284 s, 3.28 GB, `python:3.13` -> Python 3.13.15,
  Debian Open MPI 5.0.7, EBCC built. Inside: `/dev/shm` 64 MB, user root, and
  `/etc/openmpi/openmpi-mca-params.conf` sets `osc = ^ucx,pt2pt`, so `osc/sm` (priority 100,
  `osc_sm_backing_directory = /dev/shm`) backs `Win.Allocate_shared`.
- README "Running with MPI", Mac and Linux form, verbatim with `-n 4` (cwd a scratch copy of
  `netCDF_files/`): `[topology] 1 node(s) x 4 rank(s)/node = 4 parallel evaluations, one shared sample per
  node.`, `[memcheck] ... to stay under 0.80 x 11.7 GiB per node (from host total RAM (sysconf)) with 4
  rank(s) per node.`, 17127 rows in 22 s, the outputs on the host owned by the caller. The Windows form was
  not run (no Windows).
- The open question, answered with a 395.5 MiB field (synthetic, 16 steps) mounted into `/mnt/data`,
  `-n 4`, default `/dev/shm`: the sweep dies at the window. Open MPI prints "It appears as if there is not
  enough space for /dev/shm/osc_sm.<host>.<id> (the shared-memory backing file) ... Space Requested:
  414724096 B / Space Available: 66736128 B", `Win.Allocate_shared` raises `MPI_ERR_INTERN`, and the
  branch's guard fires: `[shared-sample] FATAL: MPI.Win.Allocate_shared failed (MPI_ERR_INTERN: internal
  error); the node communicator is not a shared-memory one (MPI without COMM_TYPE_SHARED?).`, MPI_ABORT,
  exit 1, nothing written. With `--shm-size=1g` on the `docker run` line the same command runs: `[memory]
  ... 395 MiB (shared sample) + ~3164 MiB (4 ranks x 2.0x decode/encode cache) + ~128 MiB (ranks x 2 x
  inner_chunk_mib) = 3.6 GiB total.`, 8 rows, exit 0. `-n 1` runs the same field with the default
  `/dev/shm` (`osc/sm` keeps a one-process window in private memory). So the README's examples pass only
  because the TIGGE sample is 128 KiB: any real field under `mpirun -n >1` needs `--shm-size` of at least
  the sample size (the working sets are private memory and do not count).
- `docker run -p 8501:8501 dc-toolkit run_web_ui`, driven from the browser as in 6.4: the sweep starts as
  root (`OMPI_ALLOW_RUN_AS_ROOT` from `ui_env`) under `mpirun -n 12` (the VM's 12 CPUs): `[sweep] ...
  split across 12 rank(s), ~1428 per rank.`, 12 rank files in `/opt/data-compression/out`, 13 s, 15774
  kept, compress and download fine. The UI caps uploads at 10 MB (`load_and_resize_netcdf`), so a UI
  sweep's window never approaches 64 MB; `/dev/shm` only matters for the CLI in a container.

### 6.6 Read-through: what does not match

1. README, "Running with MPI" (both forms) and the sentence that introduces them: add `--shm-size` (for
   example `--shm-size=8g`) to the `docker run` lines and say why: inside a container Open MPI keeps the
   shared sample in `/dev/shm`, 64 MB by default in Docker; give it at least the sample size, or a
   multi-rank sweep aborts with `[shared-sample] FATAL` (the Open MPI "not enough space" message above it
   names the two sizes). The single-rank example needs nothing.
2. `utils_cli.shared_sample_window`: the FATAL hint blames COMM_TYPE_SHARED; in the container the cause was
   `/dev/shm`. Suggested wording: "... failed ({e}); the node communicator is not a shared-memory one, or
   /dev/shm is too small for the sample (Docker: --shm-size)."
3. The docstrings of `compression_analysis_ui_local.py` (line 3) and `compression_analysis_ui_web.py`
   (line 5) still say the sweep runs as "one local MPI rank"; stale since `02b55d9`.
4. intro section 3: on a laptop the first line after `[var]` is `[memcheck] auto-shrunk ...`, which the
   sample output does not show (with 8 ranks and the default 5 GB budget it appears on any machine below
   about 106 GB of RAM). One sentence there or in section 10 stops readers from worrying. Section 11 could
   list `[memcheck] FATAL: cannot fit any sample with N rank(s) per node` (too many ranks for the budget).
5. The budget on a host without a cgroup is total RAM, not free RAM (6.3). Either a sentence in intro
   section 10 ("the estimate is checked against the machine's total memory; close what you can, or start
   fewer ranks") or `detect_node_memory_budget` returning the smaller of total and available when no cgroup
   limit is found. Christos's call; here the swap did no harm.
6. The `Dockerfile` clones `main`: right after the merge, but a branch cannot be tested from it.
7. Confirmed as written: intro sections 1 (3.11 or newer: 3.14 works), 1.4, 3 (`[topology]`, `[memory]`,
   the `NOTE`), 10 (memory, interruptions, `compress` safe to repeat); PARALLELIZATION.md (the rank model,
   0.3 GiB per interpreter, "the model is conservative", the single-process guard of `compress`); the
   README's parallelism, UI and Docker sections apart from item 1.

### 6.7 Changes on the branch, corrections, open questions for Santis

- Changed on the branch: this file only (this section). Items 1 and 2 of 6.6 are the fixes the laptop
  tests call for in the sense of section 5; 3 to 5 are tidy-ups. None was applied without Christos's go.
- Corrections to section 4: the budget source is named by the `[memcheck]` line, not the `[memory]` line
  (4.2), and a 1-rank run names no source at all; the FATAL of 4.5 does happen, is caught cleanly, but its
  hint points elsewhere (item 2).
- For the Santis check: nothing found on the laptop calls for a re-run there. Every table was identical
  across rank counts and UIs on one platform; across platforms the metrics differ at 1e-16 with the same
  gate verdicts and winner. If items 1 to 5 are applied, the smoke covers 2 and 3; 5 touches only the
  no-cgroup branch of the budget (the cgroup path is untouched).
- Open: whether the laptop budget should be free rather than total RAM (item 5), and whether the README
  should recommend a `--shm-size` value or a rule ("at least the sample").

## 7. The final check on Santis (after the laptop)

Read section 6. If the branch gained commits on the laptop, `git pull --ff-only` in the clone (the Santis
clone tracks the branch; the test worktree `dc_toolkit_santis_test` and the `main` reference worktree
`dc_toolkit_main_ref` sit next to it with their venvs), re-run the debug smoke (`$SCRATCH/mpi_shmem_phaseB/
smoke3.sbatch`: 8 ranks, 1 rank, 2 nodes, two resumes, compress, all against the old threaded parquet) and,
if docs changed, re-read them. Then the merge is Christos's call: `main` at `760fdce` has not moved, so a
fast-forward is possible; his earlier practice is a merge commit. Post-merge follow-ups, in order: the work
counter for the static split, a production run of the full V3 field list under the new topology, moving
EBCC's per-rank finiteness check to rank 0, and a decision on the memory model's factor.

Santis records for the check: phase A job 879075 (`$SCRATCH/mpi_shmem_phaseA/879075/REPORT.md`), memory
matrix and stress 879171/879206/879215 (`$SCRATCH/mpi_shmem_phaseB/REPORT.md`), pure-MPI smoke 880122,
checks 880181, EBCC 880191, production pairs 880183/880185 and 880184/880186
(`$SCRATCH/mpi_shmem_phaseB/prod/VALIDATION.md`).
