 
<div align="center">
  <img src="./data-compression_logo.png" alt="Logo" width="300"/>
</div>

Set of tools for compressing netCDF files with Zarr.

The tools use the following compression libraries:

- [Numcodecs](https://github.com/zarr-developers/numcodecs): Zarr native library [[documentation](https://numcodecs.readthedocs.io/en/stable/)]

## Installation

**System Prerequisites**

- C/C++ compiler toolchain (required to build mpi4py)
- MPI implementation (required for mpi4py)
- ecCodes library for GRIB files

On Santis@ALPS:

 ```commandline
export UENV_NAME="prgenv-gnu/24.11:v2"
```

On Balfrin@ALPS:
 ```commandline
export UENV_NAME="netcdf-tools/2024:v1"
```

Then:
```
uenv image pull $UENV_NAME
uenv start --view=default $UENV_NAME
```

once the above is complete (just for Santis, locally it is not needed):

```commandline
git clone git@github.com:C2SM/data-compression.git dc_toolkit
cd dc_toolkit
rm -rf venv
python -m venv venv
source venv/bin/activate
bash install_dc_toolkit.sh
```

## Usage

```
--------------------------------------------------------------------------------

Usage: dc_toolkit --help           # List of available commands
Usage: dc_toolkit COMMAND --help   # Documentation per command

Example:

dc_toolkit \                                                # CLI-tool
  evaluate_combos \                                         # command
  netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc \            # netCDF file
  --where-to-write ./dump \                                 # output directory
  --field-to-compress t \                                   # field to sweep
  --eval-data-size-limit 5GB                                # sample size

--------------------------------------------------------------------------------
```

### End-to-end workflow

The typical pipeline is three commands:

1. **`evaluate_combos`** — sweep `(compressor × filter × serializer)` combinations on a representative sample of the field and record compression ratio / error metrics per combo.
2. **`compress_with_optimal`** (one field at a time) or **`compress_fields_from_results`** (batch: all fields at once, dataset opened once) — persist the field(s) into a shared `.zarr` store using the winning combo from step 1, at production chunk/shard sizes.
3. **`merge_compressed_fields`** — consolidate metadata on the shared store so downstream readers can open it quickly without scanning every array.


> **Important:** pass the **same `--eval-data-size-limit`** to step 2 as you used in step 1. The `(comp_idx, filt_idx, ser_idx)` tuple from the sweep indexes into a codec space whose dtype-dependent parameters (e.g. the BitRound/Quantize grids) are derived from the sample — change the sample size and the tuple can resolve to a slightly different codec object. Symptom: worse compression ratio at persist time than the sweep reported, no error.

### Output files

`evaluate_combos` writes the following per variable `{var}` into `--where-to-write`:

| File | What it is |
|------|------------|
| `config_space_{var}.csv` | Full planning space — the Cartesian product that was going to be evaluated. Input to `analyze_clustering`. |
| `config_space_{var}_rank{N}.csv` | Per-rank streaming audit trail, flushed per row. Useful to tail during long sweeps or to inspect after a crash. |
| `results_{var}.parquet` | Consolidated results across all ranks, with a `keep` column distinguishing passing and filtered-out combos. The canonical file for analysis. |
| `*_scored_results_with_names.npy` | Kept-only scored configs in numpy structured-array form. Input to `perform_clustering` / `analyze_clustering`. |
| `manifest_{var}.json` | Best combo per variable. Read by `compress_fields_from_results` to drive the batch persist. |

`compress_with_optimal` and `compress_fields_from_results` write the compressed data into `{where_to_write}/{dataset_basename}.zarr`, under one group per variable. `batch_manifest.json` summarises a batch run.

### HPC parallelism (SLURM / MPI)

> For a thorough walkthrough of how every command parallelizes work — including how the `--bypass-zarr-sync` machinery actually works, why we cap at 32 threads on a 288-core node, and the chunk-vs-shard distinction — see [`docs/PARALLELIZATION.md`](docs/PARALLELIZATION.md).

`evaluate_combos` runs as **one MPI rank per node**, with each rank driving 32 user threads via the `--bypass-zarr-sync` machinery (default on).  Scale out by increasing `--nodes` and keeping `--ntasks-per-node=1`:

```bash
#SBATCH --nodes=8 --ntasks-per-node=1 --cpus-per-task=32

srun --unbuffered dc_toolkit evaluate_combos input.nc \
    --where-to-write ./out \
    --field-to-compress t \
    --eval-data-size-limit 5GB \
    --threads-per-rank 32
```

This topology was selected over multi-rank-per-node (32 ranks × 1 thread, the original design) to avoid OOM on large fields — the latter duplicates the sample buffer once per rank.  See `santis.run` for the validated production driver and the inline comment block summarising the experiments behind the choice.

Codec-internal thread pools must be pinned to 1 to avoid nested oversubscription (the tool checks this at startup and aborts by default; `--no-oversubscription-check` disables the guard):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       BLOSC_NTHREADS=1 NUMBA_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1
```

The `--codec-threads N` flag (default 1) is available on `evaluate_combos`, `compress_with_optimal`, `compress_fields_from_results`, and `from_zarr_to_netcdf` for use cases where codec-internal threading is genuinely needed (e.g. very large chunks on workloads that aren't memory-bandwidth-bound).  Direct testing on Santis with the production Dyamond data showed `--codec-threads > 1` does **not** help on this workload; leave it at the default unless you have a specific reason and can A/B test the change.

`compress_with_optimal`, `compress_fields_from_results`, and `merge_compressed_fields` are single-process commands — launch with `srun -n 1 ...` or plain invocation. Parallelism inside the write comes from dask's threaded scheduler, tuned via `--threads` (default: auto-detected from visible cores), `--inner-chunk-mib` (default: 16), and `--shard-mib` (default: 512). `--verify/--no-verify` (default on) re-reads the store to compute error norms — skip with `--no-verify` on re-compression runs where the combo is already trusted.

## UI implementation

Two User Interfaces have been implemented to make the file compression process more user-friendly.
Both UIs provide functionalities for compressors similarity metrics and file compression.

Outside of the mutual UI functionalities, this UI allows users to download similarity metrics plots and tweak parameters more dynamically.

If launched from santis, make sure to ssh correctly:
```
ssh -L 8501:localhost:8501 santis
```
```
dc_toolkit run_web_ui_vcluster \
  --user_account "YOUR_USER_ACCOUNT" \
  --uenv_image UENV_NAME \
  --uploaded_file "PATH_TO_FILE" \
  --time "00:15:00" \
  --nodes "1" --ntasks-per-node "72"
```
Local web-versions and non are also available:
```
dc_toolkit run_local_ui
```
````
dc_toolkit run_web_ui
````

## Docker

A self-contained image has been setup in the `Dockerfile`. You can copy the file locally, the run:

```commandline
docker build -t dc-toolkit .
```
The image contains all dependencies and automatically clones the repository.
Once this build is complete, you can run commands with docker. An example:

```commandline
docker run \
  -u $(id -u):$(id -g) \
  -w /mnt/data/docker_saved_files \
  -v "$(pwd)/netCDF_files":/mnt/data \
  -e XDG_CACHE_HOME=/tmp/.cache \
  --entrypoint /bin/bash \
  dc-toolkit \
  -c 'mkdir -p docker_saved_files && dc_toolkit evaluate_combos /opt/data-compression/netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc --where-to-write /mnt/data/docker_saved_files --field-to-compress t'
```

**Command Breakdown:**

* **`-u $(id -u):$(id -g)`**: Runs the container using your local machine's User and Group IDs rather than the Docker default `root`. This guarantees that compressed files output to your machine, are fully owned by you and aren't locked behind root permissions.
* **`-w /mnt/data/docker_saved_files`**: Sets the Working Directory.
* **`-v "$(pwd)/netCDF_files":/mnt/data`**: The volume mount. This creates a bridge between your local computer and the container so the toolkit can read your input data and write the results back to your hard drive.
* **`-e XDG_CACHE_HOME=/tmp/.cache`**: Sets the cache directory to a temporary location inside the container.
* **`--entrypoint /bin/bash`**: Forces Docker to start with a Bash shell instead of the default program (dc_toolkit).
* **`dc-toolkit`**: The name of the Docker image to run.
* **`-c '...'`**: Executes a custom shell command to handle the complex environment setup:
  * **`mkdir -p docker_saved_files`**: Creates an output directory on your host.
  * **`dc_toolkit evaluate_combos ...`**: Executes the actual compression tool, using a file inside the container and saving the results (under `--where-to-write`) to your mounted volume.

Single-machine runs (Docker included) get their parallelism from the node-local `ThreadPoolExecutor` inside a single MPI rank — no multi-rank `mpirun` is needed. Also make sure to pin codec-internal thread pools so they don't fight the outer threads:

```bash
-e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
-e BLOSC_NTHREADS=1 -e NUMBA_NUM_THREADS=1 \
-e VECLIB_MAXIMUM_THREADS=1 -e OMP_THREAD_LIMIT=1
```

Or for the web UI:

```commandline
docker run -p 8501:8501 dc-toolkit run_web_ui
```

### Running with MPI (single-container, exercises the MPI code path)

OpenMPI + Docker requires specific file permission and cache handling. Note that on a single container `evaluate_combos` runs with **one** MPI rank (`-n 1`) — the rank-per-node invariant means multi-rank on one node is not supported. Parallel work inside the single rank is done by the `ThreadPoolExecutor`; the `mpirun` launch is useful for exercising the MPI code path in CI or smoke tests. For real multi-node speedup, use SLURM (see the HPC section above).

---

#### Mac and Linux

```bash
docker run \
  -u $(id -u):$(id -g) \
  -w /mnt/data/docker_saved_files \
  -v $(pwd)/netCDF_files:/mnt/data \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e BLOSC_NTHREADS=1 -e NUMBA_NUM_THREADS=1 \
  -e VECLIB_MAXIMUM_THREADS=1 -e OMP_THREAD_LIMIT=1 \
  --entrypoint mpirun \
  dc-toolkit \
  -n 1 \
  bash -c 'HOME=/tmp/$OMPI_COMM_WORLD_RANK exec dc_toolkit evaluate_combos /opt/data-compression/netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc --where-to-write /mnt/data/docker_saved_files --field-to-compress t --eval-data-size-limit 5GB'
```

**Command Breakdown:**

* **`-u $(id -u):$(id -g)`**: Runs the container as your local user so outputs aren't locked behind `root` permissions.
* **`-w /mnt/data/docker_saved_files`**: Sets the Working Directory.
* **`-v $(pwd)/netCDF_files:/mnt/data`**: Volume mount bridging local and container filesystems.
* **`-e OMP_NUM_THREADS=1 ...`**: Pins codec-internal thread pools to 1 so they don't nest against the `ThreadPoolExecutor` inside the rank.
* **`--entrypoint mpirun`**: Bypasses the default entrypoint to launch via OpenMPI.
* **`dc-toolkit`**: The image name.
* **`-n 1`**: One MPI rank per node; on a Docker container that's one rank total. Parallelism inside the rank comes from threads, not from multiple ranks.
* **`bash -c '...'`**: Executes the dc_toolkit command:
  * **`HOME=/tmp/$OMPI_COMM_WORLD_RANK`**: Assigns a unique `$HOME` per rank — harmless with `-n 1`, kept for parity with multi-rank launches.
  * **`exec dc_toolkit evaluate_combos ... --where-to-write /mnt/data/docker_saved_files ...`**: Runs the sweep, writing all outputs into the mounted volume.

---

#### Windows (PowerShell)

When using Docker Desktop on Windows via WSL 2, Docker handles file permissions differently. You don't need to pass your user ID (Docker Desktop handles the translation automatically), but you do need to explicitly allow OpenMPI to run as root and format your paths for PowerShell.

```powershell
docker run `
  -e HOME=/tmp `
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 `
  -e BLOSC_NTHREADS=1 -e NUMBA_NUM_THREADS=1 `
  -e VECLIB_MAXIMUM_THREADS=1 -e OMP_THREAD_LIMIT=1 `
  -w /mnt/data/docker_saved_files `
  -v "${PWD}\netCDF_files:/mnt/data" `
  --entrypoint mpirun `
  dc-toolkit `
  --allow-run-as-root `
  -n 1 `
  bash -c "HOME=/tmp/`$OMPI_COMM_WORLD_RANK exec dc_toolkit evaluate_combos /mnt/data/tigge_pl_t_q_dx=2_2024_08_02.nc --where-to-write /mnt/data/docker_saved_files --field-to-compress t --eval-data-size-limit 5GB"
```

**Command Breakdown:**

* **`-e HOME=/tmp`**: Sets a base temporary home directory for the container environment.
* **`-e OMP_NUM_THREADS=1 ...`**: Pins codec-internal thread pools to 1 (prevents nested oversubscription).
* **`-w /mnt/data/docker_saved_files`**: Sets the Working Directory inside the container so output files (like `config_space_{var}.csv` and `results_{var}.parquet`) drop exactly into your mounted folder.
* **`-v "${PWD}\netCDF_files:/mnt/data"`**: Windows equivalent of the volume mount. `${PWD}` dynamically grabs your current PowerShell directory to link your local files to the container.
* **`--entrypoint mpirun`**: Bypasses the default container start command to run OpenMPI.
* **`dc-toolkit`**: The image name.
* **`--allow-run-as-root`**: The container defaults to `root` on Windows; this flag bypasses OpenMPI's built-in safety restrictions against running parallel jobs as root.
* **`-n 1`**: One rank per node; on a Docker container that's one rank total.
* **`bash -c "..."`**: Executes the parallel command. Note double-quotes for PowerShell, with an escaped backtick (` `$ `) in front of the MPI variable to prevent PowerShell from evaluating it on your host before it reaches the container.

## Slides

### [Click here to view slides](https://c2sm.github.io/data-compression/)
