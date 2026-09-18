# Santis test: replace the thread pool with MPI ranks over a shared sample

You are on Santis (CSCS Alps, Grace nodes, 288 cores per node) in a checkout of `C2SM/data-compression`
(the `dc_toolkit` package), on a branch off `santis-production-test`. Christos Kotsalos owns the repo
and wrote the backend.

`evaluate_combos` currently gets its intra-node parallelism from a 32-thread `ThreadPoolExecutor`
plus `utils.AsyncBypass`, a per-thread asyncio event loop machinery that exists only to stop zarr's
single global event loop from serialising those threads. Christos is not comfortable maintaining
threaded Python at this depth. This folder tests whether **32 single-threaded MPI ranks per node,
sharing one sample through an MPI-3 shared-memory window**, can replace it at equal memory and equal
or better speed - which would make `AsyncBypass` deletable.

Your job: run section 3 (a measurement that needs **no code changes** and decides whether the rest is
worth doing), report the number to Christos, and only once Christos agrees implement section 4 and
validate it with sections 5 and 6. Sections 1 and 2 are context. **Do not write implementation code
before section 3 has produced a number and Christos has answered.**

## 1. Why this test exists

Three topologies are on the table. All numbers below are for one node, `S` = sample bytes,
`--inner-chunk-mib 16`, 32-wide parallelism, from `utils_cli.per_rank_steady_estimate_bytes` and
`PER_THREAD_WORKING_FACTOR = 2.0` (measured by tracemalloc, 1.9-2.5x at the transient peak):

| topology | node memory | sample `Bcast` per node | bypass needed | status |
|---|---|---|---|---|
| **T1** 1 rank x 32 threads (today) | `S + 32*2S` = **65 x S** | 1 | yes | in production |
| **T2** 32 ranks x 1 thread, plain | `32*(S + 2S)` = **96 x S** | 32 | no | rejected, see below |
| **T3** 32 ranks x 1 thread, shared window | `S + 32*2S` = **65 x S** | 1 | no | **what this tests** |

At a 5 GB sample on a Grace node with `--memory-threshold 0.80`, T1 needs ~325 GB and fits; T2 needs
~480 GB and does not. (That assumes ~500 GB per node, so ~400 GB usable. It is consistent with
Alt C's observed OOM, but confirm the real `memory.max` in phase A rather than trusting it.)
That is the whole of the T2 rejection recorded as Alt C in the repo-root `santis.run`:
*"Faster per combo, but each rank duplicates the sample buffer ... OOMs on R02B08 and R02B10."*
The gap is ~1.5x, not 32x, but 1.5x is exactly the fit/no-fit line at production sample sizes.

T3 keeps T1's footprint and T2's process model. The node-local communicator it needs already exists:
`utils.detect_node_topology` builds it with `Split_type(MPI.COMM_TYPE_SHARED)`.

**The number nobody has.** Alt C says T2 is "faster per combo" without quantifying it. Alt B in the
same file (7.6x slower) is a *different* configuration - one under-subscribed rank per node, 1 user
thread - and says nothing about T2 or T3. Section 3 measures the T1/T2 gap, which is also the T1/T3
gap, because T3 differs from T2 only in where the sample bytes live.

**Decision rule, agreed with Christos before the run:**

| T2 vs T1 sweep wall time | verdict |
|---|---|
| T2 slower, or faster by < 20% | **stop.** The bypass has earned its place; report and close. |
| T2 faster by 20-100% | report with the per-phase timings; Christos decides. |
| T2 faster by > 2x | implement T3 (section 4). |

Expect the middle or the first row. Both topologies saturate the same memory-bandwidth ceiling, which
is why `--codec-threads > 1` lost by 14% (Alt A), and encode/decode already parallelise under zarr's
own pool in either arm.

## 2. What the arms actually do at runtime

Read this before you interpret any log line.

**T1 (baseline).** `--ntasks-per-node=1 --cpus-per-task=32`. `detect_cores_available()` returns 32,
`compute_default_threads_per_rank(1, 32)` returns 32, `--bypass-zarr-sync` is on by default, and
`utils.AsyncBypass.enable(32)` gives every user thread its own event loop over one shared 32-worker
codec executor.

**T2 (the measurement arm).** `--ntasks-per-node=32 --cpus-per-task=1` plus
`--allow-multi-rank-per-node` (the flag already exists; without it `sweep_setup` aborts) and
`--no-bypass-zarr-sync`. Then:
- `detect_cores_available()` returns **1** per rank, so `compute_default_threads_per_rank(32, 1)`
  returns 1. `check_thread_product(1, 1)` passes. You do not need `--threads-per-rank`.
- With the bypass off and `ranks_on_node > 1`, `utils.check_thread_oversubscription` sets
  `zarr.config.set({"threading.max_workers": 1})`. **This is correct, do not "fix" it.** Each rank
  would otherwise spawn zarr's own ~32-thread pool, giving 1024 threads per node. Pinned, the node
  runs 32 processes x 1 zarr worker = 32 chunk encodes in flight, which is the parity point with T1.
- `sweep_run_rank` still opens a `ThreadPoolExecutor(max_workers=1)`. That is a one-thread pool, not
  a thread-parallel run; leave it.
- `sweep_setup` prints `[topology] NOTE: 32 MPI rank(s) per node; each holds its own sample copy.`
  That note is the thing T3 removes.

**T3 (the target, not yet built).** T2's geometry and flags, but `sweep_build_sample` puts one sample
per node in an `MPI.Win.Allocate_shared` window that every rank on the node maps read-only.

## 3. Runbook, phase A: measure the gap (no code changes)

Set up exactly as `test_sweep_analysis/CLAUDE.md` sections 3.1 to 3.4: an env file in
`$SCRATCH`, a **separate** worktree (never touch Christos's clone), the `prgenv-gnu/26.3:v1` uenv and
a venv. EBCC is **not** needed here - use `bash install_dc_toolkit.sh` without `WITH_EBCC=1` and save
10 minutes. Reuse that runbook's `DYAMOND_DATA_ROOT` and `ACCOUNT` discovery.

**Shortcut if the production test already ran on this machine.** `install_dc_toolkit.sh` does
`pip install -e .`, and the commits that added this folder touch no file under `src/`. So an existing
worktree and venv from `test_sweep_analysis` are still valid: fetch, `git checkout --detach
origin/santis-production-test`, and skip the uenv and venv build entirely. Confirm with
`dc_toolkit --help` and `git -C "$REPO" log --oneline -1` before relying on it.

**There is no submit script in this folder** - only this file. Phase A is two srun calls inside one
allocation; compose them from A.2 either in an interactive `salloc --nodes=1 --time=01:00:00` (the
simplest for a one-hour, two-command test) or in a small sbatch script you keep in `$SCRATCH`. Do not
add a `.run` file to the repo unless Christos asks.

### A.1 Pick the field and sample size

One node, a **pinned** sample of `--eval-data-size-limit 1GB`, and a field big enough to actually
reach it: R02B10 `out_1_2/remap_qc` (3D, 4.6 GiB), the field phase 5 of the other runbook uses at
this sample size. Its settings, from the `heavy|` line of `test_sweep_analysis/santis_test.run`:

```
FILE=$DYAMOND_DATA_ROOT/Data_Dyamond_PostProcessed/out_1_2/remap_qc_20220225T000000Z.nc
VAR=qc   L1=0.01   GATES="--extremes-sensitive --phys-min 0"
```

Pass `$GATES` in **both** arms: the q99 metric is an extra pass over the sample and belongs in a
bandwidth comparison. Note that the production list runs this field at 16 threads, not 32 - here you
deliberately use 32 in the T1 arm, because the question is 32-wide threads against 32-wide ranks.
Do not "correct" it to 16.

**Do not use a small R02B06 field for this.** A field whose whole sample fits in cache does not
reproduce the memory-bandwidth regime that decides this question, and the toolkit has already been
burned by exactly that: Alt A in `santis.run` measured +17% on a 120 MiB `tot_prec` and **-14%** when
the same setting met a production-size file. If the `[memory]` line reports a sample well under 1 GB,
you picked too small a field.

At 1 GB neither arm auto-shrinks and both are far inside the budget (T2 needs 32 x 3 GB = 96 GB), so
the arms stay comparable.

Cut the codec space with `--max-evals 2000` so a run is minutes, not hours. This is safe for an A/B:
`utils_cli.sweep_config_space` applies `--max-evals` **before** the shuffle and seeds the permutation
with the final combo count, so both arms evaluate the identical set in the identical order.

Pass `--no-resume` in both arms and give each its own output directory.

### A.2 Both arms, same node if you can

```
#SBATCH --nodes=1 --time=01:00:00
# each srun below passes --cpus-per-task explicitly, so do NOT export a job-wide
# SRUN_CPUS_PER_TASK here: the two arms need different values.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       BLOSC_NTHREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1
```

Arm T1:
```
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --cpu-bind=verbose \
    dc_toolkit evaluate_combos "$FILE" --where-to-write "$OUT/t1" --field-to-compress "$VAR" \
    --l1-threshold "$L1" $GATES --eval-data-size-limit 1GB --max-evals 2000 --no-resume
```

Arm T2:
```
srun --nodes=1 --ntasks-per-node=32 --cpus-per-task=1 --cpu-bind=verbose \
    dc_toolkit evaluate_combos "$FILE" --where-to-write "$OUT/t2" --field-to-compress "$VAR" \
    --l1-threshold "$L1" $GATES --eval-data-size-limit 1GB --max-evals 2000 --no-resume \
    --allow-multi-rank-per-node --no-bypass-zarr-sync
```

Run each arm twice, alternating, and report both times. A single pair is not a measurement.

### A.3 The comparison is only valid if the binding matches

**This is the most likely way to get a wrong answer.** A Grace node is 4 sockets. If T1's 32 threads
land spread across sockets and T2's 32 ranks get packed onto one, or the reverse, you are measuring
NUMA placement, not topology. Before trusting any timing, capture from inside both arms:

```
srun ... bash -c 'grep -H Cpus_allowed_list /proc/self/status' | sort
```

Both arms must cover the same set of CPUs, and T2's 32 ranks must be spread the way T1's 32 threads
are. `--cpu-bind=verbose` prints srun's own view. If they differ, fix T2 with
`--distribution=block:cyclic` or an explicit `--cpu-bind=map_cpu:...` and say in the report which
binding each timing used.

### A.4 What to record

Per arm: the `[topology]` line, the `[memory]` line, the `[sweep]` combo count, total wall time of the
srun step, and the sweep seconds from the tool's own output. Confirm in both that `[memcheck]` did
**not** auto-shrink - if it did in one arm only, the arms used different samples and the timing is
void.

Then confirm the two arms agree on the science: the parquets must match row for row.

```
python - <<'PY'
import pandas as pd
a = pd.read_parquet("t1/results_<var>.parquet").set_index("pipeline").sort_index()
b = pd.read_parquet("t2/results_<var>.parquet").set_index("pipeline").sort_index()
assert a.index.equals(b.index), (len(a), len(b))
for c in ["ratio", "l1_rel", "l2_rel", "linf_rel", "eucd", "keep"]:
    assert a[c].equals(b[c]), c
print("identical:", len(a), "rows")
PY
```

A mismatch here is more interesting than the timing: it means the topology changes the numbers, which
it must not. Report it before anything else.

**Stop here and report.** Section 4 starts only if Christos says so.

## 4. If it is a go: the T3 design

Not implementation instructions to follow blindly - the design Christos agreed to, for you to
implement and defend.

### 4.1 The shared window, in `utils_cli.sweep_build_sample`

Today: rank 0 builds the sample, `utils.broadcast_numpy` sends it to every rank, each allocating its
own buffer. Target:

1. `node_comm, ranks_on_node, local_rank = utils.detect_node_topology(comm)` (already there).
2. A leaders communicator: `comm.Split(0 if local_rank == 0 else MPI.UNDEFINED, key=rank)`.
3. Rank 0 builds the sample and broadcasts `meta` (shape, dtype, dims, coords, attrs) to **all** ranks
   as today.
4. Every rank calls `MPI.Win.Allocate_shared(nbytes if local_rank == 0 else 0, itemsize, comm=node_comm)`
   and then `win.Shared_query(0)` to get the leader's segment; wrap it with
   `np.ndarray(buffer=buf, dtype=dtype, shape=shape)`.
5. Node leaders `Bcast` the sample **into that buffer** over the leaders communicator - not into a
   fresh array. `utils.broadcast_numpy` allocates with `np.empty` on non-root, so it needs an
   optional `out=` parameter, or call `leaders.Bcast(view, root=0)` directly.
6. `node_comm.Barrier()`, then set `sample_np.flags.writeable = False` on every rank. Nothing in the
   sweep writes to the sample; making that explicit is free and turns a future aliasing bug into an
   exception.
7. `sample_da` wraps the same buffer with `xr.DataArray` - a view, no copy, as today.
8. **Return the window and free it.** The sweep loops over variables; a window leaked per variable
   leaks a whole sample per variable. `win.Free()` after the variable's `sweep_run_rank`, before the
   next `sweep_build_sample`.

### 4.2 An assertion worth its cost

The characteristic shared-window bug is a rank reading the segment before the leader filled it: no
crash, just wrong metrics on some ranks. After the barrier, once per variable:

```
h = float(np.nansum(sample_np, dtype=np.float64))
assert comm.allreduce(h, MPI.MIN) == comm.allreduce(h, MPI.MAX)
```

One pass over the sample, seconds at 5 GB, against a class of bug that would otherwise surface as
"some ratios look odd". Keep it unconditional.

### 4.3 The rest of the diff

- `sweep_setup`: the `ranks_on_node > 1` abort inverts - one rank per node becomes the special case.
  Keep `--allow-multi-rank-per-node` accepted as a no-op for one release so existing scripts do not
  break, and drop the `each holds its own sample copy` NOTE.
- **The memory model must be rewritten.** `per_rank_steady_estimate_bytes` is per rank and assumes a
  private sample; under T3 the node cost is `S + ranks_on_node * 2 * S + ranks * 2 * chunk`. The
  `[memory]` banner in `sweep_banner` and `max_sample_bytes_for_threads` follow it.
- **`check_node_memory_headroom` has a latent hazard here.** It treats a cgroup-sourced budget as
  per-task (`"cgroup is per-task under SLURM"`) and compares the *per-rank* requirement against the
  *whole* limit. If Santis's cgroup is per **step** rather than per task, 32 ranks each pass a check
  they collectively fail by 32x. Settle it empirically in phase A: print `/sys/fs/cgroup/memory.max`
  from inside a 1-task step and from inside a 32-task step on the same node. If they are equal, the
  guard is wrong for multi-rank and must divide by `ranks_on_node` for cgroup sources too.
- `sweep_run_rank`: `ThreadPoolExecutor(max_workers=1)` becomes a plain loop; `AsyncBypass.close_loops()`
  goes.
- `utils.AsyncBypass`: keep the class as a thin `zarr.core.sync.sync` wrapper at first (`run()`
  already falls back to it when disabled), so the diff is one concern at a time. Delete the per-thread
  loops, `_get_or_create_shared_executor` and `_thread_loops` in a **second** commit, after T3 has
  passed section 6.
- `utils.progress_bar` prints rank 0's own pending count. Still true, still per-rank, now 1/32 of a
  node - say so in the log line or leave it; do not make it collective.
- Per-rank CSVs go from 8 to 256 per variable at 8 nodes. `rank_files`' regex (`rank\d+`) already
  handles it, and `read_rank_csvs` concatenates whatever it finds, so resume still works across a
  changed rank count. Watch the Lustre small-file cost in the sweep timings.

### 4.4 Optional, measure separately: an RMA work counter

T3 keeps the static `config_space[rank::size]` split, so 8 nodes become 256 static slices of ~35
combos instead of 8 slices of ~1100 that are internally work-stealing. Per-combo cost varies by an
order of magnitude; the shuffle covers that at 1100 combos and will not at 35. If phase B shows a
straggler tail, replace the split with an `MPI.Win.Allocate` counter on rank 0 and `Fetch_and_op(1,
MPI.SUM)` to claim the next combo index - about ten lines, global dynamic balancing, and it removes
rank affinity from resume as a side effect. **Land it as its own commit with its own A/B.** Do not
bundle it with T3, or you will not know which change moved the number.

## 5. Runbook, phase B: validate T3

1. Repeat section 3's A/B with T3 as a third arm, same field, same `--max-evals`, same node. T3 must
   match T1 on memory (the `[memory]` line) and T2 on speed, within noise.
2. Then the memory claim, which is the whole point: one node, `--eval-data-size-limit 5GB`, no
   `--max-evals` cap needed beyond keeping it short. T1 and T3 must both run; T2 must be **refused**
   by the memcheck or OOM. If T2 quietly survives at 5 GB, `check_node_memory_headroom` is
   under-counting (section 4.3) and that is a finding in its own right.
3. Then scale: 2 nodes, one heavy field (R02B10 `out_1_2/remap_qc`, 1 GiB sample), T1 vs T3, to
   exercise the multi-node `Bcast` path with a real leaders communicator.
4. Then the full harness. `test_sweep_analysis/santis_test.run` is the existing production test; run
   it unchanged on the T3 branch with `HEAVY=0` first, then in full. Its `check_outputs.py` asserts
   25 parquet columns, recomputed gate verdicts, manifest bests, resume counts, staging and
   consolidated metadata - none of which should care about topology, which is exactly why it is the
   right oracle.

## 6. The correctness bar

T3 changes where bytes live, not what is computed. The bar is therefore exact equality, not
similarity:

- The section 3 parquet comparison must pass for **every** field tested, across T1, T2 and T3.
- `check_outputs.py` must exit 0 on a full `santis_test.run`.
- Resume must work across a topology change: sweep half a field under T1, resume it under T3, and the
  final parquet must equal an uninterrupted T1 run row for row. `sweep_state_{var}.json` fingerprints
  the sample and chunking but **not** the rank count or topology, so this must work - if it does not,
  the fingerprint or `reusable_rows` is wrong.

## 7. Hazards specific to shared windows

| hazard | how it shows | what to do |
|---|---|---|
| `/dev/shm` too small | `Win.Allocate_shared` fails, or the node OOMs at a size the model says fits | Many MPI builds back shared windows with POSIX shm, which counts against both `/dev/shm` and the cgroup. Check `df -h /dev/shm` on a compute node **in phase A**, before designing around a size. |
| `Split_type` fell back to hostnames | `Allocate_shared` fails on a communicator that is not shared-memory | `detect_node_topology` has a hostname fallback for old MPI. T3 must assert the `COMM_TYPE_SHARED` path was taken and abort with a clear message otherwise. |
| missing barrier after the fill | some ranks' metrics are garbage, no crash | the section 4.2 checksum |
| window not freed per variable | memory grows one sample per variable; variable 3 or 4 OOMs | `win.Free()`; watch a multi-variable run, not a single-field one |
| leader placement | the sample lands in one socket's memory, 32 ranks read it remotely | T1 already pays this (one buffer, 32 threads), so it is a wash - but confirm, do not assume, with the section 3.3 binding capture |
| cgroup accounting of shm | the node OOM-kills below the modelled limit | shared pages are charged to the cgroup that first touched them; the leader's cgroup carries the whole sample under a per-task cgroup |

## 8. Known, not bugs

- T2 printing `[topology] NOTE: ... each holds its own sample copy` and pinning
  `zarr.config threading.max_workers` to 1 are both correct behaviour (section 2).
- Blosc `typesize` means ratios differ from any pre-`b371897` sweep. Errors do not. Irrelevant to an
  A/B where both arms run the same code, but it will bite a comparison against v3 numbers.
- `PER_THREAD_WORKING_FACTOR = 2.0` is the per-*worker* factor and applies to a rank exactly as it
  does to a thread. T3 does not change it.

## 9. Working rules

- Christos wants compact, navigable code with few comments ("why", not narration), no test
  infrastructure in the repo beyond these test folders, and stable CLI options and output formats.
- Never modify Christos's existing clone; work in a worktree. Never push to `main`. Christos is the
  commit author: no Claude co-author trailer.
- Results are gitignored (`*.csv`, `*.parquet`, `*manifest*.json`, `*.zarr*`, `out-*`). Never commit
  the DYAMOND data path or user names: use `DYAMOND_DATA_ROOT`.
- Phase A costs about one node-hour. Phase B is a few. Ask before anything larger, before any
  resubmission, and before any code change in phase A.

## 10. What to report

**After phase A (the decision point):**
1. The two timings per arm, the binding capture for both, and which decision-rule row they land in.
2. The parquet comparison: identical, or the first differing pipelines and columns.
3. `/dev/shm` size and the `memory.max` seen by a 1-task step vs a 32-task step (section 4.3's open
   question - answer it while you have the allocation).
4. Your recommendation in one paragraph: stop, or build T3.

**After phase B, additionally:**
5. The three-way table: T1, T2, T3 on memory (`[memory]` line and observed RSS), sweep seconds, and
   whether the 5 GB case ran or was refused.
6. `check_outputs.py` verdict on a full `santis_test.run`, and the cross-topology resume result.
7. Whether a straggler tail appeared at 256 slices, with the per-rank completion spread, and your
   call on section 4.4.
8. What you would delete from `utils.py` in the follow-up commit, and what you would keep.
