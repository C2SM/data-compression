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
- This file is untracked on purpose (test scaffolding stays out of the repository). Carry it by hand or
  commit it on the branch temporarily and drop it before the merge; if you commit it, it holds no paths
  beyond `$SCRATCH` and `$DYAMOND_DATA_ROOT`.

## 6. Laptop findings (written by Claude Code on the laptop)

_Empty until the laptop tests run. Fill in: environment (4.1); one verdict per test 4.2 to 4.6 with the
decisive log lines (the `[topology]`, `[memory]`, `[resume]`, `[verify-gate]` and any `FATAL` or traceback);
what was changed on the branch, if anything, and the commit hashes; open questions for the Santis check._

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
