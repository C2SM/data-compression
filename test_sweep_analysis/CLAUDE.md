# Santis production test of dc_toolkit

You are on Santis (CSCS Alps, Grace nodes, 288 cores per node) in a checkout of the
branch `santis-production-test` of `C2SM/data-compression` (the `dc_toolkit` package). The branch
holds three days of rework (2026-09-15 to 2026-09-17) done with Christos Kotsalos, who owns the repo
and wrote the backend. The rework was checked locally with 85 functional checks, a row-by-row
regression against commit `870943c`, and two adversarial review rounds (42 confirmed findings
fixed), but **it has never run at production scale**. This folder is that run: a Slurm job
(`santis_test.run`) that sweeps and compresses a few DYAMOND fields and exercises every changed
path. A checker (`check_outputs.py`) then validates the outputs against the invariants below.

Your job: set up, submit, monitor, read `test_report.md`, triage every failure and warning down to
a root cause with evidence, and report back to Christos. Do not apply non-trivial code fixes
without him.

## 1. What changed (branch vs `main` @ `c94b83b`)

**Layout.** `src/dc_toolkit/cli.py` holds only click commands and options. `utils_cli.py` (new)
holds the command helpers in numbered sections: 1 plumbing, 2 threads and memory guards, 3 gates,
4 pipelines and persistence, 5 sweep, 6 compress, 7 conversion, 8 analysis, 9 UI support.
`utils.py` is the library. The streamlit UI covers both the local and the vcluster case, so
`compression_analysis_ui_vcluster.py` is deleted. Together with the Qt UI, the five source files
are 3810 lines, down from 8029 at `870943c`.

**A codec combination is identified by its pipeline, not by indices.** The indices
`comp_idx/filt_idx/ser_idx` into rebuilt codec lists are gone, and with them
`sample_signature_{var}.json`, the `*.npy` score files and the codec-space rebuild at compress time.
- `utils.pipeline_to_dict` gives `{"compressor", "filter", "serializer"}`, each the codec's zarr
  `to_dict()` JSON or `null`. It is byte-for-byte what zarr writes in the array's `zarr.json`.
  `utils.pipeline_json` (sorted keys, compact) is the resume key and the `pipeline` column of the
  CSVs and the parquet. `pipeline_name` / `codec_label` are the readable forms.
- `manifest_{var}.json` has `best = {name, pipeline, ratio, l1_rel, eucd}`, plus
  `effective_thresholds`, `phys_min/max`, `q99_abs`, `args` (sweep flags incl. chunk geometry) and
  `env`.
- `compress DATASET WHERE_TO_WRITE [--vars a,b] [--pipeline JSON|@file]` replaces
  `compress_fields_from_results` and `merge_compressed_fields` (both kept as hidden aliases). It
  writes each field's manifest best (fallback: best kept parquet row) and takes the chunk geometry
  from the manifest unless the CLI flags are given. It consolidates the store at the end.
  `compress_with_optimal` only prints a pointer.
- A `null` compressor now means no compressor. Index `-1` used to mean zarr's default Zstd.
- This removed a real bug: at `870943c`, batch compress wrote Blosc instead of the sweep's Zstd
  winner when the class flags were not repeated.

**Resume (`--resume`, default on).**
- `sweep_state_{var}.json` fingerprints dataset path, sample shape and dtype, full-field range,
  sampling policy, vertical floor, `inner_chunk_mib`, `spatial_split` and `env` (library versions and
  EBCC env vars). If any of these changed, the field restarts from scratch.
- Rank 0 decides which recorded rows are reusable and broadcasts that. Rows are re-gated with the
  current thresholds at consolidation.
- A row that lacks a metric a newly enabled gate needs is evaluated again: `q99_rel` under
  `--extremes-sensitive`, or `grad_rel` under `--gradient-gate` for rows that pass the cheap gates.
- Rows outside the current codec space are dropped, and failed combos are always retried.
- A partial last CSV line from a killed run is cut. An unreadable CSV is renamed `*.unreadable`.
- Per-rank files are matched by exact regex, never a `rank*` glob.

**compress hardening.**
- Each field is written into `<store>.__staging__/<var>` (a sibling of the store), re-read and
  gated, and only then renamed into `<store>/<var>`. A SIGKILL, OOM or walltime kill leaves nothing
  a reader can see, and a failed rewrite keeps the old array. Consolidation discards staging
  leftovers first.
- The verify gate also checks physical bounds (`Decoded_Min/Max`). `--l1/--l2/--linf/--bias-threshold`
  override the manifest's values, which is the only gate for a `--pipeline` field without a manifest.
- Production norms are computed in float64. Integer fields used to overflow and fail their own gate.
- The exit status is 1 when any field failed or had no usable pipeline. `batch_manifest.json` records
  per field: `status`, `pipeline`, `source`, `verify_gate`, `cr_drift_status`, `ratio`,
  `predicted_ratio`, `cr_drift`, `errors`, and the chunk and shard geometry.
- `--no-consolidate` drops stale consolidated metadata. A stale listing used to silently hide fields
  or describe the wrong codecs.

**Codec space and numerics.**
- **Blosc gets `typesize` = item size.** Through zarr's wrapper, shuffle was a no-op and 9 of the 33
  compressors were duplicates. **Blosc ratios therefore differ from every earlier sweep.** Error
  metrics do not change.
- ZFPY runs only on floats, and on int32/int64 in fixed-rate mode. `ZFPYFlat` (registered as
  `numcodecs.zfpy`) flattens chunks to 1-D, because zfp's per-axis limit (2^24 in 2-D, 2^16 in 3-D)
  overflows on the R02B10 cell axis (83.9 M cells).
- BitRound and Quantize take floats only (`dtype.kind in "iu"` guards). Integer fields get Delta.
- Relative errors: 0/0 = 0 and x/0 = inf. **All-zero fields now pass the gates.** v3 kept 0 of 8580
  combos on R02B10 `out_8/cape`, which is all zeros.
- The gradient metric is computed slab-wise and reuses the decoded buffer.
- The memory model's `PER_THREAD_WORKING_FACTOR` is 2.0, a measured value; the v3 generator assumed
  1.5. The sweep can therefore auto-shrink a sample where v3 did not (`[memcheck] auto-shrunk ...`).
  That is not a failure.
- Non-numeric variables (CF bounds, datetimes, scalars like `crs`) are skipped.

**MPI and threads.**
- Under MPI, an uncaught exception on any rank calls `comm.Abort` through `sys.excepthook`, so an
  error no longer hangs the allocation.
- The bypass's shared executor ignores `shutdown`. A garbage-collected per-thread event loop used to
  shut it down and fail every combo of the following variables. Per-thread loops are closed after
  each variable.

**EBCC (optional serializer).**
- Install with `pip install -e ".[ebcc]"` or `WITH_EBCC=1 bash install_dc_toolkit.sh` (cmake and
  HDF5 headers, about 10 min).
- `--with-ebcc` appends 7 standalone combos `(no compressor, no filter or AsType->float32, EBCC)` to
  the sweep; `--max-evals` never cuts them. Their max-absolute-error targets are fraction x
  full-field range, with fractions 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4 and 1e-4.
- Constraints: float fields whose last two dims are a (lat, lon) frame (native ICON cell grids do not
  qualify), a tile of 32..2047 cells per side that divides the frame, and one tile per chunk.
- NaN/Inf or an invalid tile make the C library **exit the process**. The toolkit validates before
  encoding (`utils.ebcc_sweep_entries`, `utils_cli.validate_pipeline`), so a srun step that dies
  without a Python traceback during EBCC work is a validation gap.
- Encoding runs at about 1-2 MB/s per thread. The codec is stored as `numcodecs.ebcc_filter`, and the
  `zarr.codecs` entry point maps that name to `dc_toolkit.utils:EBCC`.

**Packaging.** `pyproject.toml` adds scikit-learn, pandas and pyarrow, drops tslearn and
zarr-any-numcodecs, and adds the `ebcc` extra and the entry point. The Dockerfile builds with EBCC.

## 2. What the test runs

Topology as in production: 8 nodes x 1 rank x 32 CPUs, `--bypass-zarr-sync`. `compress` runs as one
task. L1 budgets (relative), sample sizes, threads and gate flags come from the v3 production sweep.
All fields are float32 and finite.

| phase | steps (`steps.tsv`) | field(s) | what it proves |
|---|---|---|---|
| 0 | `preflight`, `nodes`, `cli_help` | - | dc_toolkit is imported from this checkout; each node shows its cores and cgroup limit; the CLI starts under srun |
| 1 | `sweep`, `compress` | R02B06 `out_7/remap_clct` (bounds 0..100), `out_3/t_2m` native (gradient gate), `out_2/tot_prec` native (q99 gate), `out_1_2/remap_qc` 3D (q99); R02B10 `out_8/remap_cape` (all zeros) | full codec space (8844 combos, 8580 for cape) on whole fields; compress with verify gate, staging, consolidation |
| 2 | `resume_identical`, `resume_gradient`, `compress_rerun`, `pipeline_compress` | R02B06 clct (`RESUME_KEY`) | copying the rank CSVs and re-running the same flags reuses every row; adding the gradient gate re-evaluates only the rows that pass the cheap gates; a second compress skips the existing array; `--pipeline @file` writes the best kept Delta+PCodec (lossless) pipeline |
| 3 | `kill_run`, `kill_resume` | R02B06 tot_prec (`KILL_KEY`) | SIGTERM after `KILL_AFTER` s, then resume on half the nodes; the final parquet must equal phase 1's row for row |
| 4 | `sweep`, `compress`, `pipeline_compress` | R02B06 `out_7/remap_t_2m` with `--with-ebcc` (only when ebcc imports) | EBCC combos mixed into a full sweep; the best kept EBCC pipeline written via `--pipeline` and read back |
| 5 | `sweep`, `compress` | R02B10 `out_1_2/remap_qc` (3D, 4.6 GiB, 1 GiB sample, 16 threads); R02B10 `out_3/t_2m` native (7.5 GiB, 1 GiB sample, 16 threads, gradient) | cascade sampling over time and levels, multi-GiB Bcast, ZFPYFlat on the 83.9 M-cell axis, spatial split of 320 MiB timesteps, multi-GiB staged writes |
| 6 | `check` | all | `check_outputs.py` writes `test_report.md`; its exit status is the job's |

Reference numbers from the v3 sweep (old code, 8 nodes; heavy fields used larger samples):

| field | v3 combos | v3 kept | v3 sweep time |
|---|---|---|---|
| R02B06 out_7 clct | 8844 | 4884 | 97 s |
| R02B06 out_3 t_2m | 8844 | 6402 | 97 s |
| R02B06 out_2 tot_prec | 8844 | 4587 | ~4 min |
| R02B06 out_1_2 qc | 8844 | 3432 | 228 s |
| R02B10 out_8 cape | 8580 | 0 | 100 s |
| R02B06 out_7 t_2m (without EBCC) | 8844 | 6402 | ~1.5 min |
| R02B10 out_1_2 qc | 8844 | 3432 | 1516 s (2 GiB sample) |
| R02B10 out_3 t_2m | 8844 | 5874 | 5924 s (1.5 GiB sample) |

On the whole-field R02B06 entries the kept counts should match v3. Blosc's `typesize` changes
ratios, not errors. If a count differs, find the gate columns that differ before calling it a
regression. For cape, expect most combos kept, a very large ratio, and a CR-drift warning at compress
(shard index overhead dominates an all-zero store).

Expected wall time is about 2 h 50 min: phases 0-4 about 45 min, R02B10 qc about 25 min, R02B10
t_2m about 85 min, check about 5 min. The limit is 4 h. A step that would not fit in the remaining
walltime is recorded as `skipped-walltime` (a warning), and heavy sweeps need 75 min left.

Before this branch was pushed, the harness itself was dry-run on a Mac. The whole
`santis_test.run` ran under bash 3.2 with a fake `srun` (2 MPI ranks via `mpirun`) and a fake
`timeout`, over the full codec space on the bundled tigge file, and passed. The kill landed
mid-sweep (3282 of 8844 rows recorded) and the resume on one rank matched the uninterrupted run row
for row. The checker was also shown to catch six planted corruptions: a flipped verdict, a wrong
best, dropped consolidated metadata, a drifted resume metric, a staging leftover and a tampered
codec config. **Not exercised before Santis:** real `srun` geometry and CPU binding,
`SRUN_CPUS_PER_TASK`, SIGTERM forwarding by `srun`, `squeue`-based walltime guards, cgroup memory
limits, the uenv, and anything at DYAMOND scale.

## 3. Setup and submission

```bash
uenv image pull prgenv-gnu/26.3:v1                  # once
uenv start --view=default prgenv-gnu/26.3:v1
git clone git@github.com:C2SM/data-compression.git dc_toolkit_test && cd dc_toolkit_test
git checkout santis-production-test
python -m venv venv && source venv/bin/activate
bash install_dc_toolkit.sh                          # WITH_EBCC=1 bash install_dc_toolkit.sh for phase 4
```

Use a **fresh checkout and venv**. The v3 sweep ran from a locally patched checkout: its
`--mask-abs-above` flag was never in the repo. The `preflight` step aborts the job when `dc_toolkit`
is not imported from the submitting checkout.

```bash
export DYAMOND_DATA_ROOT=<parent of Data_Dyamond_PostProcessed, ..._R02B06, ..._R02B08>   # ask Christos
sbatch --account=<account> test_sweep_analysis/santis_test.run                            # full test

# quicker variants
HEAVY=0 sbatch --account=<account> --time=01:15:00 test_sweep_analysis/santis_test.run   # skip phase 5
SWEEP_EXTRA_FLAGS="--max-evals 300" HEAVY=0 KILL_AFTER=40 \
    sbatch --account=<account> --time=00:40:00 test_sweep_analysis/santis_test.run       # plumbing only
```

sbatch exports the submitting environment, so the knobs travel with the job. Knobs: `RESULTS_BASE`
(default `$SCRATCH/dc_toolkit_test_sweep_analysis/<jobid>`), `VENV`, `HEAVY`,
`WITH_EBCC=auto|0|1`, `SWEEP_EXTRA_FLAGS`, `KILL_AFTER` (default 100 s), `RESUME_KEY`, `KILL_KEY`,
and `FIELDS_FILE` (one `CLASS|RES|STREAM|FILE|VAR|L1|SAMPLE|THREADS|GATE FLAGS` line per entry;
CLASS is `light`, `ebcc` or `heavy`). Re-run a subset with `FIELDS_FILE` rather than editing the
default list. Do not reuse a `RESULTS_BASE`: the phase 1 directories would resume and the phase 2
copies would be stale.

Monitor with `squeue --me`, `tail -f out-dc_test-<jobid>.out` in the submit directory, and the
per-step logs. To re-run only the checker (it needs the venv, and srun because reading an EBCC store
imports mpi4py):
`srun -A <account> --uenv=prgenv-gnu/26.3:v1 --view=default -p debug -N1 -n1 python test_sweep_analysis/check_outputs.py $RESULTS_BASE`.

## 4. Outputs

```
$RESULTS_BASE/
  steps.tsv  environment.log  preflight.log  nodes.log  cli_help.log  check.log  test_report.md
  fields/<RES>_<STREAM>_<VAR>/          phase 1 and 5: sweep.log compress.log
      config_space_{var}.csv  config_space_{var}_rank{N}.csv  failures_{var}_rank{N}.csv
      sweep_state_{var}.json  manifest_{var}.json  results_{var}.parquet
      batch_manifest.json  (batch_manifest.first.json for RESUME_KEY)  <file stem>.zarr/
  resume_identical/<key>/  resume_gradient/<key>/      phase 2 sweeps on copied rank CSVs
  pipeline_override/<key>/  pipeline.json  pick.log  <stem>.zarr/  batch_manifest.json
  kill_resume/<key>/        kill_run.log  kill_resume.log  + sweep outputs
  ebcc/<key>/               sweep + compress outputs, pipeline_ebcc/ (EBCC via --pipeline)
```

`steps.tsv` columns: `step key dir input var l1 rc seconds`. `rc` is an exit status, or
`missing-input`, `skipped-walltime` or `no-pipeline`.

## 5. What `check_outputs.py` asserts

It does not import dc_toolkit. It recomputes everything from the files. Any failed assertion means
FAIL and exit 1.

- **Every step:** exit status 0. `kill_run` is exempt. Status 0 there only means the sweep finished
  before the kill, so the kill/resume check is inconclusive (a warning). `missing-input` and
  `no-pipeline` fail.
- **Sweep directories:**
  - The parquet has exactly the 25 columns and unique pipelines. `config_space_{var}.csv` has
    `num_combos` unique rows, a superset of the parquet's pipelines.
  - Result rows + `num_failed_total` = `num_combos`, and the failures files hold `num_failed_total`
    rows.
  - Kept rows = `num_passed`, and `num_filtered` = combos - passed - failed.
  - `keep` equals the AND of the `pass_*` columns. It also equals a **recomputed verdict** from the
    metrics, the manifest's `effective_thresholds`, `phys_min/max` and `gradient_threshold`.
  - `manifest.best` is the top kept row by ratio desc, then L1 asc, then pipeline.
  - With the q99 gate on, every row has `q99_rel`. With the gradient gate on, every row passing the
    cheap gates has `grad_rel`.
  - Every Blosc compressor carries `typesize` = item size, and `--with-ebcc` plans exactly 7 EBCC
    combos.
- **compress:**
  - `batch_manifest.json` entry: `status=ok`, `verify_gate=pass`, `any_error=false`, and the pipeline
    equals the manifest best (or `pipeline.json`).
  - The production errors are within the thresholds and phys bounds, and `N_Corrupt=0`.
  - The store exists and no `.__staging__` sibling is left.
  - The consolidated metadata lists the array with the same codecs as the array's `zarr.json`. Those
    codecs (inside `sharding_indexed` when sharded) equal the pipeline:
    `[filter] + [serializer or bytes] + [compressor]`.
  - Dims, shape and dtype match the source, and the first slab reads back without new NaN/Inf.
- **`compress_rerun`:** status `skipped-existing`.
- **`resume_identical`:** the `[resume] X of N` line has X = rows of the phase 1 parquet. Every
  metric, `keep` and `best` are identical.
- **`resume_gradient`:** X = rows that fail a cheap gate or already have `grad_rel`. Every other
  metric is identical.
- **`kill_resume`:** every metric, `keep` and `best` equal phase 1's. X outside (0, N) only warns.
- **`pipeline_compress`:** as compress, with `source=--pipeline` and only the L1 threshold.

Warnings do not fail the run: CR drift above 25 %, failed combos, rows with `n_corrupt > 0`,
`grad_rel` on rows that fail a cheap gate, and skipped steps. Read every one of them anyway.

## 6. Triage

| symptom (log or report) | likely cause | look at |
|---|---|---|
| `[oversubscription] --threads * --codec-threads = 32 exceeds the visible cores (1)` | srun steps did not get 32 CPUs | `SRUN_CPUS_PER_TASK` in the job; `nodes.log`; `utils_cli.check_thread_product` |
| `[oversubscription-check] ... not pinned to 1` | thread env vars lost | the exports at the top of `santis_test.run` |
| `[topology] ERROR: detected N MPI rank(s) per node` | more than one task per node | sbatch/srun geometry |
| `[memcheck] REFUSING` / `FATAL: cannot fit any sample` | cgroup budget below the model | `nodes.log` memory.max; `utils_cli.sweep_sample_limit`, `PER_THREAD_WORKING_FACTOR` |
| `[memcheck] auto-shrunk sample budget` | model shrank the sample to fit | expected on tight nodes; note the size in the report |
| `[sample] FATAL: one horizontal slab ...` | sample budget below one horizontal slab | `--eval-data-size-limit` of that entry |
| job sits without progress | a rank died before a collective and was not aborted | Tracebacks on any rank; `utils_cli.sweep_setup` excepthook |
| srun step ends with no Python traceback during EBCC | EBCC C library exited the process | `utils.ebcc_sweep_entries`, `utils_cli.validate_pipeline`, the field's NaN/tile |
| many rows in `failures_*` | a codec pairing raises | `error` column; `utils.combo_is_valid` |
| `keep verdict disagrees with their metrics` | gate logic diverged (code or checker) | the rows; `utils_cli.evaluate_gates`, `regate`, `utils.within_limit` |
| `[resume] ... starting this field from scratch` in `resume_identical` | `sweep_state` fingerprint mismatch | diff the two `sweep_state_{var}.json`; `utils_cli.sweep_recorded_rows` |
| `resume_*`/`kill_resume` rows differ from phase 1 | non-deterministic codec or a resume merge bug | the differing pipelines; `utils_cli.sweep_select_best` (groupby last), `reusable_rows` |
| `[verify-gate] FAIL` on a heavy field | the sample under-represents the full field (not necessarily a bug) | sweep row vs `batch_manifest.json` errors; `Decoded_Min` vs `phys_min` |
| `[cr-drift] WARNING` | sample vs full field, or shard index overhead | informational unless `--cr-drift-gate` |
| store missing and `.__staging__` present | compress killed mid-write (walltime/OOM) | `compress.log` tail; re-run compress (it retries) |
| `stored codecs ... are not the pipeline` | pipeline dict and zarr metadata diverge | `utils.codec_pipeline_kwargs`, `pipeline_from_dict`, `_CODEC_CLASSES` |

## 7. Known, not bugs

- Blosc ratios, and so some best pipelines, differ from v3 results. Heavy fields use 1 GiB samples
  here, so their numbers are not directly comparable with v3.
- The repo-root `santis.run` is the old production driver. Its L1 values read like native units
  (1.0 would allow 100 % relative error) and it lacks `SRUN_CPUS_PER_TASK`. Do not reuse its list.
- `--mask-abs-above` does not exist. Fields with undeclared fill sentinels (`runoff_s`, `lhfl_s`,
  `qhfl_s`, `cin_ml`, `smi`) are deliberately left out.
- Open suggestion, not applied: PCodec `delta_spec="auto"` with an explicit `delta_encoding_order`
  pins that order. A true auto entry was about 10 % smaller on a test field.

## 8. Working rules

- Christos wants compact, navigable code with few comments ("why", not narration), no test
  infrastructure in the repo (this folder is the exception for this run), and stable CLI options and
  output formats.
- Never push to `main`. Work on this branch or a branch off it. Christos is the commit author: no
  Claude co-author trailer.
- Results are gitignored (`*.csv`, `*.parquet`, `*manifest*.json`, `*.zarr*`, `out-*`); keep them out
  of commits. Never commit the DYAMOND data path or user names: use `DYAMOND_DATA_ROOT`.
- For each failure, give the evidence (log line, file, row), the root cause, and a minimal fix
  proposal. Re-run the affected phase (`FIELDS_FILE`, `HEAVY=0`, `WITH_EBCC=0`, `RESUME_KEY`) before
  calling anything fixed.

## 9. What to report back

1. Job id, commit hash, total wall time, the checker verdict.
2. The `Failures` and `Warnings` sections of `test_report.md` verbatim.
3. Per field: combos, kept (vs v3), best pipeline, sweep ratio, store ratio, CR drift, verify
   verdict, sweep seconds (vs v3), and any `[memcheck]` or `[sample]` lines of note.
4. Resume counts (`[resume] X of N`) for `resume_identical`, `resume_gradient` and `kill_resume`,
   and whether the kill landed mid-sweep.
5. EBCC: kept EBCC combos, best EBCC ratio vs the overall best, and the EBCC encode time if visible.
6. For every failure: root cause with evidence and a proposed fix. Also list what you would change
   in the test itself.
