# Parallelization strategies in `dc_toolkit`

This document explains, in plain English, how each `dc_toolkit` command uses parallelism. It is intended both for new users trying to understand how the toolkit makes use of an HPC node, and as a reference for the maintainers about *why* the choices are what they are.

It is deliberately not exhaustive — for the full architectural details, read `src/dc_toolkit/utils_cli.py` and `src/dc_toolkit/utils.py`. This doc is the user-friendly tour.

---

## The big picture

`dc_toolkit` has three kinds of commands, and each kind uses a different parallelism style:

| Kind | Commands | What runs in parallel |
|---|---|---|
| **A. Big sweep** | `evaluate_combos` | Many independent codec configurations, across MPI ranks (one per core) on any number of nodes; the ranks of a node share one copy of the sample |
| **B. Big single write** | `compress`, `from_nc_to_zarr`, `from_zarr_to_netcdf` | The chunks of one big array, on one node, through dask's threads |
| **C. Light utilities** | `merge_compressed_fields`, `open_zarr_and_inspect`, `perform_clustering`, `analyze_clustering`, `plot_compression_errors`, all UI commands | Nothing to tune: small post-processing work |

The first two kinds are interesting and the rest of this document is about them. Kind C commands do small post-processing work (metadata consolidation, clustering of result tables, plot generation, UIs) where parallelism would not change anything meaningful.

---

## `evaluate_combos`: MPI ranks, one per core

This is the most parallelism-intensive command in the toolkit. It evaluates up to 10,494 (float32) or 17,127 (float64) codec configurations (EBCC aside; the `[sweep]` line prints the exact count) against a representative sample of a field, scores each one, and picks the best.

The work is *embarrassingly parallel*: each combo is independent. The toolkit runs it as a plain MPI program: one rank per core, each rank evaluating one pipeline at a time.

### Splitting the work

Run with N nodes and R ranks per node:

```bash
#SBATCH --nodes=8 --ntasks-per-node=32 --cpus-per-task=1
```

Every rank orders the combos the same way: any EBCC combos first (the slowest, so none is left to start when the rest are done), then the others shuffled with a seed derived from the total combo count (so every node gets a representative mix of cheap and expensive codecs, and `--resume` sees the same order). The combos not yet recorded are split across the N nodes deterministically (node k takes every N-th of them, starting at k). Inside a node, the R ranks share a counter in the node's shared memory (`node_counter`): a rank that finishes a combo claims the next one with an atomic fetch-and-add, which waits on no other process, so no rank idles while its node has work left. (The window carries the hint `disable_shm_accumulate=false`; without it MPICH routes the atomics through the rank that owns the counter, which answers only between two combos.) Combo costs differ by an order of magnitude; the counter keeps a rank that drew expensive ones from holding up the node, and on a fresh sweep each node's share is large enough (about 1,300 to 2,100 combos on 8 nodes) that the shuffle evens out the nodes. Every rank streams its results to its own CSV, and rank 0 consolidates them at the end.

The combo phase scales with the node count: two nodes evaluate a field's combos in about half the time of one. The sample build and the consolidation run on rank 0 alone, and the sample is sent to every node, so these steps do not shrink with more nodes.

On a laptop the same program is started with `mpirun -n <cores>`; started without `mpirun` it is one rank, and it says so at startup.

### Why processes and not threads

Each rank is an ordinary Python process. That buys three things at once:

- **Nothing is shared inside a node but the sample and the work counter.** The sample is read-only and the counter changes only through an atomic fetch-and-add; no interpreter lock, no memory allocator and no zarr event loop is contended by 32 workers, because each worker has its own.
- **File reads scale.** The netCDF library serialises threads behind a global lock, so reader threads inside one process take turns; separate processes read in parallel. The full-field range is read that way: its blocks are split across all ranks. The sample is read once, by rank 0, and reaches the other ranks through the shared-memory window.
- **One core per rank.** Each rank calls zarr's synchronous API one call at a time, with zarr's internal pool pinned to one worker (`threading.max_workers = 1`) and dask on its synchronous scheduler, because a rank owns one core.

The price is one Python interpreter per rank, about 0.3 GiB each, which on a 288-core node is noise.

### One sample per node: the shared-memory window

Every rank reads the same sample, and it must exist once per node, not once per rank. `sweep_build_sample` does this with an MPI-3 shared-memory window:

1. Rank 0 builds the representative sample (`build_representative_sample`).
2. On every node the first rank allocates a window of the sample's size (`MPI.Win.Allocate_shared` on the node communicator that `detect_node_topology` builds with `COMM_TYPE_SHARED`); the other ranks of the node allocate nothing and map the leader's segment (`Shared_query`).
3. Rank 0 copies its sample into its node's window; the node leaders broadcast it into theirs (one transfer per node over the interconnect).
4. A node barrier, then every rank marks its view of the same physical pages read-only, and a checksum of a strided probe of the window (every 4099th element), compared across all ranks, catches a rank that read it before the fill.
5. The window is freed after the variable, collectively, so a multi-variable sweep never holds two samples.

Under Open MPI on Linux the window is a file in `/dev/shm`, so a container must give that at least the sample size (`docker run --shm-size`); Cray MPICH on Santis does not use it.

Everything downstream reads views of that array: `z[...] = sample_np` hands zarr chunk-shaped views, the error norms slice it chunk by chunk, the gradient metric walks it in blocks of leading slabs of about 32 MiB. It is read-only for the whole sweep, so sharing it costs nothing and needs no lock.

Work that touches the whole sample happens once, not once per rank: rank 0 computes the q99 cut of the extremes gate, the sample's value range (a sample without variation is not searched) and EBCC's check for NaN/Inf and broadcasts the answers, and the full-field range for FixedScaleOffset is one pass over the file's blocks split across all ranks (without that range FixedScaleOffset is left out, never fitted to the sample). Anything added to the pre-sweep path should follow the same rule; a per-rank pass over the sample multiplies its temporaries by the rank count.

### Why 32 ranks and not 288?

A Grace node has 288 cores. Why does the production driver stop at 32 per node?

Because compression is **memory-bandwidth-bound**, not compute-bound. Each codec call streams data through memory rather than doing dense arithmetic, and the aggregate memory bandwidth saturates at roughly 32 well-distributed workers. Adding more beyond that does not speed anything up — they queue waiting for memory — and it adds a working set per rank. The same reason keeps the codec libraries' own thread pools pinned to one thread: inner threads would compete for the same bandwidth.

### Memory: one shared sample plus a working set per rank

What is *not* shared is the working set of the combo each rank has in flight:

| Per-rank allocation | Size | Lives until |
|---|---|---|
| its own `MemoryStore` holding the encoded bytes | sample / ratio | `store.clear()`, right after the decode |
| **the decoded array** from `z[...]` | **1 × sample, full size** | the end of the combo |
| float64 copies of a chunk's valid cells in the metric loop (original and error) | 2 × inner chunk for a float64 field, 4 × for float32 | per chunk |

So the shape of it is *one shared read-only sample per node, plus a private full-size decoded buffer per rank*. Only the metric loop and the codec calls work chunk-wise; `z[...]` materializes the whole sample. The model the toolkit prints as `[memory]` for each field, once its sample is built, is

```
node steady state  =  S + R × 2 × S + R × 32 MiB          (S = sample, R = ranks per node, 16 MiB inner chunks)
```

about 325 GB for a 5 GB sample and 32 ranks (the R × 32 MiB term counts float64 fields; a float32 field's extra 2 × chunk per rank sits inside the factor-2 headroom). The factor 2 is a rank's own peak (decoded buffer plus encoded bytes plus a filter's copy); the ranks of a node do not peak together, and a node's measured steady state is nearer 1.1 × S per rank, so the model is conservative. The sweep checks the model against `--memory-threshold` × the node's budget (the cgroup limit under SLURM, else host RAM) and shrinks the sample when it does not fit (`[memcheck] auto-shrunk ...`), but not below 3 time steps × 3 levels (3 indices across all time-like dims, such as ensemble members; all, where a field has fewer): a smaller `--eval-data-size-limit` is raised to that minimum (`[sample] raised the sample budget ...`), and a field whose minimum does not fit stops the sweep (`[memcheck] FATAL: the smallest sample ...`). At that minimum a 3-D field of the native R02B10 grid samples 2.8 GiB.

The decoded buffer is what decoding with one `z[...]` costs: R combos in flight on a node hold R full-size decoded copies, however the sample is shared. That is why the sample is shared and the working sets are not, and why the knobs are the sample size and the number of ranks (`santis.run` sets both per field).

### Summary for `evaluate_combos`

```
8 nodes × 32 ranks = 256 cores, 256 pipelines in flight, 8 copies of the sample in total
```

---

## `compress`: dask on chunks

This command takes the winning pipeline of each swept field (or the `--pipeline` you pass) and persists the real field. Unlike `evaluate_combos`, the work is not thousands of independent combos — it is *one* big array per field that needs to be encoded. So the parallelism strategy is different.

### One Python process, no MPI

The command explicitly runs as a single Python process. It aborts if accidentally launched with multiple MPI ranks, because the work does not decompose cleanly across ranks at the file level.

### Parallelism via dask's threaded scheduler

The field is rechunked into **write units**: one shard (~512 MiB by default), or one chunk when sharding is skipped. Dask's threaded scheduler runs one task per write unit on up to `--threads` workers; inside a task, zarr's sharding codec encodes the shard's inner chunks (one codec call per ~16 MiB chunk):

```
   Field (e.g., 5 GB)
         │
         ▼
   Rechunked into 10 write units (shards of 512 MiB)
         │
         ▼
   Dask scheduler: --threads workers, one task per shard
         │
         ▼
   Each task:  read the shard's data → zarr encodes its 32 inner chunks (in compiled code) → one shard file
         │
         ▼
   The field is written into a staging store, re-read and gated, then moved into place
```

Threads work here because the heavy work is compiled code that releases Python's interpreter lock: the codec calls and the numpy reductions. The dask workers read and rechunk; the codec calls they issue run in zarr's own thread pool (asyncio's default, min(32, cores + 4) threads, since compress leaves `threading.max_workers` unset). Reads of a netCDF input take turns behind the library's global lock (see "File reads scale" above). Give the process its cores through the launcher, since `--threads` defaults to the visible ones (`srun --ntasks=1 --cpus-per-task=32` in `santis.run`, or plain invocation on a laptop).

### Chunks vs shards (a frequent confusion)

These are two different cuts of the same data. Different jobs, different sizes:

- **Chunk** (`--inner-chunk-mib`, default: the sweep's value, else 16 MiB): the unit the codec works on. One codec call = one chunk. Smaller chunks have higher per-call overhead but more parallelism granularity; larger chunks compress more efficiently but use more memory.
- **Shard** (`--shard-mib`, default 512 MiB): the unit zarr writes to disk. One shard = one file. Many chunks bundle into one shard so we do not end up with millions of tiny files (which would be a disaster on shared HPC filesystems).

Dask operates on shards; zarr encodes the chunks inside each one. So `--shard-mib` sets the file size, the number of dask tasks (field / shard) and the memory per worker, while `--inner-chunk-mib` sets the size of a codec call (and of a partial read later).

### Memory model

Peak memory ≈ `--threads × (max(source block, --shard-mib) + 3 × --shard-mib)`, capped at about three times the field: a write task holds its share of the source data, the rechunked shard, zarr's encode copy and the encoded bytes. `--threads` defaults to the visible core count: on a full 288-core Grace node with 512 MiB shards the per-thread term comes to about 576 GiB, more than the node has, so the three-times-the-field cap decides and the memory guard refuses any field larger than about a quarter of the available memory (0.8 / 3 at the default `--memory-threshold`); with `--threads 32` (or `srun --cpus-per-task=32`) and 512 MiB shards it is 64 GiB, though on a field smaller than about 21 GiB the three-times-the-field cap binds first. If you bump `--shard-mib`, memory scales linearly.

### Threads inside a codec

A codec call runs on one thread: zarr turns Blosc's internal threads off, and the other codecs have none. A write gets its parallelism from running many codec calls at once, one per thread of zarr's pool (at most min(32, cores + 4) threads), fed by the `--threads` dask workers; threads inside a single call would compete for the same memory bandwidth anyway.

### Several fields

Fields are processed one after another, not in parallel — the dataset is opened once, and the per-field memory and CPU profile is that of a single write. The metadata consolidation at the end is single-threaded and takes seconds (`--no-consolidate` skips it; `merge_compressed_fields` runs it later).

---

## `from_nc_to_zarr` and `from_zarr_to_netcdf`: format conversion

These convert between NetCDF and zarr. Same architecture as `compress`: single Python process, dask threaded scheduler with `--threads` workers (default: visible cores) operating on chunks.

The differences from `compress`:

- `from_nc_to_zarr` writes uncompressed zarr (no codecs).
- `from_zarr_to_netcdf` decodes the zarr (which involves running the codec stack in reverse). It has a `--max-size` safety guard against accidentally producing huge `.nc` files from compressed zarr stores.
- Neither has the memory guard or the thread-variable check (`--oversubscription-check`) of `compress`.

Both should be launched single-process: `srun -n 1 ...` or no srun.

---

## Light commands (Kind C, summarized)

- **`merge_compressed_fields`** — consolidates the metadata of a store that `compress --no-consolidate` wrote. Single-threaded.
- **`open_zarr_and_inspect`** — read-only diagnostic that prints array shapes, dtypes, codec config. Single-threaded.
- **`perform_clustering` / `analyze_clustering`** — k-means clustering on the small results table from a sweep. Single-threaded Python (with optional BLAS-internal threading if you do not pin `OMP_NUM_THREADS=1`).
- **`plot_compression_errors`** — round-trips one (lat, lon) field of at most 2.5 GiB twice (as is and shifted by 180 degrees) through one pipeline for diagnostic plotting. Not compute-intensive; dask's default threaded scheduler reads it and zarr's pool runs the codec calls, with no `--threads` knob.
- **All UI commands** (`run_web_ui` and `run_web_ui_vcluster`: Streamlit web servers; `run_local_ui`: a Qt desktop app). Not compute-intensive; single-threaded themselves, they start the sweep under `mpirun` (one rank per physical core) when `mpirun` or `mpiexec` is on the PATH, else as one rank, or, on a vcluster, under `srun`.

These commands are intentionally simple. They run on a login node or a small interactive allocation, complete quickly, and do not need parallelism.

---

## A few key invariants

To keep the parallelism behaving correctly, the toolkit relies on some invariants: the thread variables, `--threads` and the single-rank guard are checked at startup, the others are launch conventions and code rules. Most of the time you do not need to think about them, but they are worth being aware of:

- **One rank per core** for `evaluate_combos` (a launch convention, not checked): `--ntasks-per-node` with `--cpus-per-task=1` is the number of ranks and of cores you give a node. A single rank on a multi-core machine is legal and prints a note suggesting `mpirun -n <physical cores>`, the number Open MPI accepts by default.
- **Codec env vars must be pinned to 1**:
  ```bash
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
         BLOSC_NTHREADS=1 NUMBA_NUM_THREADS=1 \
         VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1
  ```
  This prevents the codec libraries and BLAS from spawning their own thread pools next to a rank that owns one core (or, in the write commands, next to dask's workers). `evaluate_combos` and `compress` abort at startup if any of these are unset or not 1 (`--no-oversubscription-check` turns the abort into a warning); the conversion commands do not check them.
- **`--threads ≤ visible cores`** (the CPUs in the process's affinity mask) is checked at startup by `compress` and `from_zarr_to_netcdf`.
- **Single-rank commands** (`compress`, the store, conversion and plot commands) abort if launched with `mpirun -n >1` or `srun -n >1`; the clustering and UI commands have no MPI guard.
- **An error on one rank ends the job.** An uncaught exception on any rank of a sweep calls `MPI_Abort`, which stops every rank. No collective call may sit on an error path (an `except` or `finally` block): the failing rank would wait there for ranks that never arrive, and the job would hang until its time limit.

---

## Where to read further

- `santis.run` — the production driver for Santis: this topology, a per-field list of budgets, gates, sample sizes and ranks per node, and resume through `RESULTS_BASE`.
- `src/dc_toolkit/cli.py` — the commands and their options only.
- `src/dc_toolkit/utils_cli.py` — what each command does, step by step, from the function the command calls (the sweep: `sweep_dataset` in section 5, then `sweep_setup`, `sweep_sample_limit`, `sweep_build_sample`, `shared_sample_window`, `node_counter`, `sweep_run_rank`; compress: `compress_fields` in section 6).
- `src/dc_toolkit/utils.py` — `detect_node_topology` and `check_thread_oversubscription` (section 7); the memory guards (`check_memory_headroom`, `check_node_memory_headroom`) and the `--threads` check (`check_thread_count`) live in section 2 of `utils_cli.py`.

---

## TL;DR

| Command | Parallelism in one line |
|---|---|
| `evaluate_combos` | N nodes × R ranks/node, one rank per core, each rank evaluating one pipeline at a time and claiming the next from its node's counter in shared memory; one shared copy of the sample per node |
| `compress` | One process, dask's threaded scheduler with `--threads` workers (default: visible cores), each worker writes one shard; fields done sequentially |
| `from_nc_to_zarr` | One process, dask's threaded scheduler with `--threads` workers (default: visible cores), one task per source chunk (`--preserve-source-chunks`, the default); every variable in one uncompressed, unsharded write |
| `from_zarr_to_netcdf` | One process, dask's threaded scheduler with `--threads` workers decoding the zarr chunks; every variable in one NetCDF write |
| `merge_compressed_fields` | Single-threaded metadata consolidation |
| `open_zarr_and_inspect` | Single-threaded read-only inspection |
| `perform_clustering`, `analyze_clustering` | Single-threaded clustering |
| `plot_compression_errors` | Diagnostic plotting of one field; dask's default threads read it, zarr's pool runs the codec calls |
| UI commands | Single-threaded Streamlit (`run_web_ui`, `run_web_ui_vcluster`) or Qt (`run_local_ui`) front ends that start the sweep under mpirun (when on the PATH, else one rank) or srun |

The interesting commands are `evaluate_combos` (MPI ranks sharing one sample per node) and the three dask-driven write commands (single process + dask threaded scheduler on chunks). Everything else is intentionally simple.
