# Parallelization strategies in `dc_toolkit`

This document explains, in plain English, how each `dc_toolkit` command uses parallelism. It's intended both for new users trying to understand how the toolkit makes use of an HPC node, and as a future-reference note for the maintainers about *why* we made the choices we did.

It is deliberately not exhaustive — for the full architectural details, read `src/dc_toolkit/cli.py` and `src/dc_toolkit/utils.py`. This doc is the user-friendly tour.

---

## The big picture

`dc_toolkit` has three kinds of commands, and each kind uses a different parallelism style:

| Kind | Commands | What runs in parallel |
|---|---|---|
| **A. Big sweep** | `evaluate_combos` | Many independent codec configurations, across nodes and threads |
| **B. Big single write** | `compress_with_optimal`, `compress_fields_from_results`, `from_nc_to_zarr`, `from_zarr_to_netcdf` | The chunks of one big array, on one node |
| **C. Light single-threaded utilities** | `merge_compressed_fields`, `open_zarr_and_inspect`, `perform_clustering`, `analyze_clustering`, `plot_compression_errors`, all UI commands | Nothing — pure single-threaded Python |

The first two kinds are interesting and the rest of this document is about them. Kind C commands do small post-processing work (metadata consolidation, clustering of result tables, plot generation, web UIs) where parallelism would not change anything meaningful.

---

## Why threading works at all in Python (the GIL question)

A common worry: "Python has a Global Interpreter Lock (the GIL), so threads can't actually run in parallel — right?"

This is *technically* true but *practically* irrelevant for our workload. The GIL only blocks Python code from running on multiple threads at the same time. It does **not** block code written in C/C++/Rust — and the heavy work in `dc_toolkit` is all in such libraries:

- Compression itself (Blosc, zfp, Quantize, BitRound, …) is C/C++ code that releases the GIL while compressing.
- numpy reductions used to compute error norms (L1/L2/L∞) are also C code that releases the GIL.
- HDF5 / NetCDF / zarr file I/O releases the GIL during reads and writes.

Roughly **99% of each compression operation's wall time** is spent inside such GIL-releasing C code. So 32 threads in a Python process can keep 32 cores genuinely busy doing real compression work, all in parallel, with the GIL only briefly serializing the small Python orchestration glue between calls.

The rule of thumb: Python threading works well when the heavy work is in compiled libraries. Compression is one of those workloads. Pure-Python loops (think: parsing a big JSON file in Python code) would be a different story — but that's not what we do.

---

## `evaluate_combos`: parallelism in two layers

This is the most parallelism-intensive command in the toolkit. It evaluates ~13,000 different codec configurations against a representative sample of a field, scores each one, and picks the best.

The work is *embarrassingly parallel*: each combo is independent. The toolkit takes advantage of this on two layers.

### Layer 1: across nodes (MPI)

Run with N nodes, one MPI rank per node:

```bash
SBATCH --nodes=8 --ntasks-per-node=1 --cpus-per-task=32
```

The 13,000 combos get split across the 8 ranks deterministically (rank 0 takes every 8th combo starting at 0, rank 1 takes every 8th starting at 1, etc). Each rank works on its slice independently — no coordination during the sweep, just a final result-gather at the end.

Adding more nodes gives a clean linear speedup at this layer. Two nodes process combos roughly 2× faster than one; eight nodes process them 8× faster than one.

### Layer 2: within one node (threads + the bypass)

Now zoom into one rank — one Python process on one node. It has ~1,625 combos to evaluate. To use the node's 32 cores, the rank uses a `ThreadPoolExecutor` with 32 worker threads. Each worker thread grabs a combo, evaluates it end to end, returns the result, then grabs the next.

This is where things get interesting. The codec libraries we use are wrapped by zarr (a chunked-array storage library), and zarr has an internal scheduling mechanism (technically: an *asyncio event loop* — a single-threaded scheduler that runs *coroutines*, async functions that pause and resume cooperatively) that — in its standard form — assumes only one thread is calling it at a time. With our 32 threads all calling zarr concurrently, this scheduler becomes a bottleneck.

We measured this directly: 32 threads with the standard zarr scheduling was ~5× slower than 32 separate MPI ranks doing the same total work. The 5× slowdown was *not* the GIL — it was zarr's internal queuing serializing all 32 threads through one scheduler (the single event loop running on one OS thread, processing one coroutine at a time).

**The bypass** (`--bypass-zarr-sync`, default on) fixes this. Instead of all 32 threads sharing zarr's one scheduler, the bypass gives each thread its own scheduler (technically: a *per-thread event loop* — every user thread runs its own asyncio loop on its own OS thread, so the 32 loops progress in parallel with no shared chokepoint). Now all 32 threads can call zarr concurrently with no shared bottleneck. Coupled with one shared, bounded pool of 32 codec worker threads (technically: a `concurrent.futures.ThreadPoolExecutor` registered as the *default executor* on every per-thread loop, so the loops don't each spawn their own pool), the rank's parallelism becomes:

```
   32 user threads (one combo each, mostly waiting)
                        │
                        ▼
   32 codec worker threads (the actual compression work)
                        │
                        ▼
   32 cores busy doing real work in parallel
```

In other words: the bypass removes a serialization point that exists in zarr's defaults. It is *not* a workaround for Python or for the GIL — it's an adaptation to the specific multi-threaded usage pattern we need.

### Why 32 threads and not 288?

A Grace node has 288 cores. Why do we cap at 32?

Because compression is **memory-bandwidth-bound**, not compute-bound. Each codec call streams data through memory rather than doing dense arithmetic. A Grace node has 4 sockets, and the aggregate memory bandwidth saturates at roughly 32 well-distributed worker threads. Adding more threads beyond that doesn't speed anything up — they just queue waiting for memory. Direct testing on Santis confirmed this: a 9× larger thread budget came out 14% *slower* on production-size files due to cache pressure, and the result holds across multiple configurations.

So 32 is the right number for this hardware and workload. It's also coincidentally Python's default thread-pool cap, so things line up nicely.

### Codec-internal threading (`--codec-threads`, default off)

The codec libraries themselves can use multiple threads inside a single encode call (Blosc has built-in support; zfp uses OpenMP). The `--codec-threads N` flag exposes this. We tested it; it doesn't help on production-size files for the same memory-bandwidth reason. Leave at the default (1) unless you have a specific reason and can A/B-test the change.

### Summary for `evaluate_combos`

```
8 nodes × 32 threads × 1 codec-thread/encode = 256 cores of effective parallelism
```

Memory: each node holds one copy of the 5 GB sample (the bypass is what made this possible — the alternative of 32 ranks per node would duplicate the sample 32 times and OOM).

---

## `compress_with_optimal` and `compress_fields_from_results`: dask on chunks

These commands take the winning codec from a sweep and apply it to persist a real field (or all fields). Unlike `evaluate_combos`, the work isn't 13,000 independent combos — it's *one* big array that needs to be encoded. So the parallelism strategy is different.

### One Python process, no MPI

Both commands explicitly run as a single Python process. They abort if accidentally launched with multiple MPI ranks, because the work doesn't decompose cleanly across ranks at the file level.

### Parallelism via dask's threaded scheduler

The compression of one big field is split into work units called **chunks** (one chunk = one codec call, ~16 MiB by default). Dask's threaded scheduler distributes those chunks across worker threads:

```
   Field (e.g., 5 GB)
         │
         ▼
   Split into 320 chunks of 16 MiB each
         │
         ▼
   Dask scheduler: 32 worker threads pull chunks from a queue
         │
         ▼
   Each worker:  read chunk → encode chunk (in C) → emit encoded bytes
         │
         ▼
   Encoded chunks bundled into shards (~512 MiB each, default)
         │
         ▼
   Shards written to disk
```

Same parallelism mechanics as `evaluate_combos` at the per-thread level — codec calls in C release the GIL, 32 threads in 32 codec calls in parallel, etc. The difference is what the threads are *doing*: in `evaluate_combos` each thread drives a whole combo end-to-end; here each thread processes one chunk.

### Chunks vs shards (a frequent confusion)

These are two different cuts of the same data. Different jobs, different sizes:

- **Chunk** (`--inner-chunk-mib`, default 16 MiB): the unit the codec works on. One codec call = one chunk. Smaller chunks have higher per-call overhead but more parallelism granularity; larger chunks compress more efficiently but use more memory.
- **Shard** (`--shard-mib`, default 512 MiB): the unit zarr writes to disk. One shard = one file. Many chunks bundle into one shard so we don't end up with millions of tiny files (which would be a disaster on shared HPC filesystems).

Dask operates on chunks. Shards are an output-side concern handled by zarr's writer. The two sizes are independent knobs.

### Memory model

Peak memory ≈ `--threads × --shard-mib`. With defaults that's `32 × 512 MiB = 16 GiB`, well under the per-node budget. If you bump `--shard-mib`, memory scales linearly.

### `--codec-threads` here too

Same flag as in `evaluate_combos`, same default (1), same guidance: leave at 1 unless you've measured otherwise.

### Difference between the two commands

| | `compress_with_optimal` | `compress_fields_from_results` |
|---|---|---|
| Reads the manifest from a sweep | Yes — for one field | Yes — for all fields |
| Runs the encode for | One field | All fields, sequentially (one at a time) |
| When to use | One-off persist | Bulk persist after a full sweep |

The "batch" version processes fields one after another, not in parallel — so the per-field memory and CPU profile is the same as the single-field version. The benefit is opening the dataset once and reading manifests automatically.

---

## `from_nc_to_zarr` and `from_zarr_to_netcdf`: format conversion

These convert between NetCDF and zarr. Same architecture as `compress_with_optimal`: single Python process, dask threaded scheduler with `--threads` workers operating on chunks.

The differences from `compress_with_optimal`:

- `from_nc_to_zarr` writes uncompressed zarr (no codecs), so there's no `--codec-threads` flag.
- `from_zarr_to_netcdf` decodes the zarr (which involves running the codec stack in reverse), so it does have `--codec-threads`. It also has a `--max-size` safety guard against accidentally producing huge `.nc` files from compressed zarr stores.

Both should be launched single-process: `srun -n 1 ...` or no srun.

---

## Light commands (Class C, summarized)

- **`merge_compressed_fields`** — consolidates metadata on a `.zarr` store. Single-threaded, runs in seconds, just walks zarr's metadata structure.
- **`open_zarr_and_inspect`** — read-only diagnostic that prints array shapes, dtypes, codec config. Single-threaded.
- **`perform_clustering` / `analyze_clustering`** — k-means clustering on the small results table from a sweep. Single-threaded Python (with optional BLAS-internal threading if you don't pin `OMP_NUM_THREADS=1`).
- **`plot_compression_errors`** — runs one codec configuration once for diagnostic plotting. Single-threaded.
- **All UI commands** (`run_web_ui`, `run_local_ui`, `run_web_ui_vcluster`) — Streamlit web servers and SLURM submission helpers. Not compute-intensive; single-threaded.

These commands are intentionally simple. They run on a login node or a small interactive allocation, complete quickly, and don't need parallelism.

---

## A few key invariants

To keep the parallelism behaving correctly, the toolkit enforces some invariants at startup. Most of the time you don't need to think about them, but they're worth being aware of:

- **One MPI rank per node** is the default for `evaluate_combos`. Multi-rank-per-node would duplicate the sample buffer once per rank and OOM on large fields.
- **Codec env vars must be pinned to 1**:
  ```bash
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
         BLOSC_NTHREADS=1 NUMBA_NUM_THREADS=1 \
         VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1
  ```
  This prevents the codec libraries from spawning their own internal thread pools that would compete with our outer threads. The toolkit aborts at startup if any of these are unset or non-1.
- **`--threads × --codec-threads ≤ physical cores`** is enforced by a built-in product check.
- **Single-rank commands** (everything except `evaluate_combos`) abort if launched with `mpirun -n >1` or `srun -n >1`.

---

## Where to read further

- `santis.run` — the production driver script, with an inline comment block summarizing the experiments on Santis that informed the topology choices.
- `src/dc_toolkit/cli.py` — the actual implementation of every command.
- `src/dc_toolkit/utils.py` — the bypass machinery (`_AsyncBypass`, `_get_or_create_shared_executor`, `_get_thread_event_loop`) and the various safety checks (`check_thread_oversubscription`, `_check_thread_product`, the memory-headroom checks).

---

## TL;DR

| Command | Parallelism in one line |
|---|---|
| `evaluate_combos` | N nodes × 32 threads/node, each thread handles one full combo, with the bypass to avoid zarr's internal scheduler bottleneck |
| `compress_with_optimal` | One process, dask's threaded scheduler with 32 workers, each worker processes one chunk |
| `compress_fields_from_results` | Same as above, fields done sequentially (one at a time) |
| `from_nc_to_zarr` | Same as above, no codec stack (uncompressed write) |
| `from_zarr_to_netcdf` | Same as above, with codec stack to decode |
| `merge_compressed_fields` | Single-threaded metadata consolidation |
| `open_zarr_and_inspect` | Single-threaded read-only inspection |
| `perform_clustering`, `analyze_clustering` | Single-threaded clustering |
| `plot_compression_errors` | Single-threaded diagnostic plotting |
| UI commands | Single-threaded Streamlit / SLURM helpers |

The interesting commands are `evaluate_combos` (uses MPI + threads + the bypass) and the four dask-driven write commands (single process + dask threaded scheduler on chunks). Everything else is intentionally simple.
