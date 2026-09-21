 
<div align="center">
  <img src="./data-compression_logo.png" alt="Logo" width="300"/>
</div>

Set of tools for compressing netCDF files with Zarr.

The tools use the following compression libraries:

- [Numcodecs](https://github.com/zarr-developers/numcodecs): Zarr native library [[documentation](https://numcodecs.readthedocs.io/en/stable/)]
- [EBCC](https://github.com/spcl/EBCC) (optional): Error Bounded Climate Compressor, see [EBCC](#ebcc-optional) below

## Installation

**System Prerequisites**

- C/C++ compiler toolchain (required to build mpi4py)
- MPI implementation (required for mpi4py)
- ecCodes library for GRIB files

On Santis@ALPS:

 ```commandline
export UENV_NAME="prgenv-gnu/26.3:v1"
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

once the above is complete (the uenv steps are for ALPS only; locally they are not needed):

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
  --l1-threshold 0.005 \                                    # relative L1 budget (0.5 %), required
  --eval-data-size-limit 5GB                                # sample size

--------------------------------------------------------------------------------
```

### End-to-end workflow

The typical pipeline is two commands:

1. **`evaluate_combos`** — sweep `(compressor × filter × serializer)` combinations on a representative sample of each field and record the compression ratio and error metrics of every combo.  `--l1-threshold` (a relative L1 error budget) is mandatory; the L2, Linf and bias gates default to 2x, 10x and 0.5x of it.  The winner of each field goes to `manifest_{var}.json`.  With `--resume` (the default) combos already recorded in the output directory are not evaluated again: their metrics are reused (as long as the sample and chunk settings are unchanged), the gates are re-applied with the current thresholds, a combo that lacks a metric a newly enabled gate needs is evaluated again, and rows outside the current codec space are left out of the results.  `--no-resume` starts the field from scratch.  `--compressor-class` / `--filter-class` / `--serializer-class` accept a fixed list of names (a typo is refused when the command line is parsed); a field whose dtype the chosen class cannot take is skipped with a message (an error when it is the `--field-to-compress`), except integer fields, which have Delta as their only filter and fall back to it (with a message) for any `--filter-class` but `none`; variables that are not numeric arrays (datetimes, strings, scalars such as `crs`) are skipped too, and so are CF bounds such as `clon_bnds` (grid geometry, which a lossy codec would move) unless one is named with `--field-to-compress`.
2. **`compress`** — persist the fields into one shared `.zarr` store (dataset opened once) with the winning pipeline of each field, then consolidate the store's metadata so readers open it quickly.  The chunk geometry (`--inner-chunk-mib`, `--max-inner-chunk-mib`, `--spatial-split`) defaults to what the sweep used, as recorded in the manifest, so the store matches what was measured; the command prints where each value came from.  After each write the field is re-read and gated against the sweep's thresholds and physical bounds (`--l1-threshold` ... `--bias-threshold` override them, e.g. for a field without a manifest) and its ratio is compared with the sweep's.  Each field is written under a staging name and renamed into place only after its gates passed, so an interrupted or failed write never counts as done and never replaces an earlier good array; a failed field, or one without a usable pipeline, is recorded in `batch_manifest.json` and makes the command exit with status 1, so a later run retries it.

A combination is identified by its **pipeline**: the zarr JSON of its three codecs, as stored in `zarr.json` (`{"compressor": {...}, "filter": {...}, "serializer": {...}}`, `null` for an absent codec).  It appears in every result row, in the manifests and in the store itself, so nothing has to be rebuilt or re-sampled between the sweep and the write.  `compress --vars t --pipeline '{...}'` (or `--pipeline file.json`, a `manifest_{var}.json` included) writes a field with a pipeline of your own, for example one picked from `results_{var}.parquet` or from a UI.

### Output files

`evaluate_combos` writes the following per variable `{var}` into `--where-to-write`:

| File | What it is |
|------|------------|
| `config_space_{var}.csv` | The planned combos in sweep order (name, codec labels, pipeline JSON): the valid `(compressor, filter, serializer)` triples after the pairing rules, `--max-evals` and the EBCC entries, shuffled with a count-dependent seed. |
| `config_space_{var}_rank{N}.csv` | Per-rank streaming audit trail, flushed every 100 rows (plus `failures_{var}_rank{N}.csv` for combos that raised).  Useful to tail during long sweeps, to inspect after a crash, and read back by `--resume`. |
| `results_{var}.parquet` | Consolidated results across all ranks: one row per combo with its `name`, codec labels, `pipeline` JSON, ratio, error metrics, the per-gate verdicts and a `keep` column marking the combos that passed every gate.  The canonical file for analysis (`perform_clustering`, `analyze_clustering`) and the fallback of `compress` when a manifest is missing. |
| `sweep_state_{var}.json` | What the recorded rows were measured on (dataset, sample shape, value range, sampling policy, chunk settings).  `--resume` reuses rows only while it matches; otherwise the field restarts from scratch. |
| `manifest_{var}.json` | The best kept combo (`best.name`, `best.pipeline`, its ratio, relative L1 error and Euclidean distance), the effective thresholds, the sweep arguments (chunk geometry, codec-space settings, ...), the q99 cut and the environment.  Read by `compress` and `plot_compression_errors`. |

`compress` writes the compressed data into `{where_to_write}/{dataset_stem}.zarr` (the input filename without extension), one zarr array per variable at the root of the store, and `batch_manifest.json` summarising the run (per field: status, pipeline, ratio, predicted ratio and CR drift, error norms, the verify-gate and CR-drift verdicts, chunk/shard geometry).

### HPC parallelism (SLURM / MPI)

> How every command parallelizes work (the `--bypass-zarr-sync` machinery, the 32-thread cap on a 288-core node, chunks vs shards): [`docs/PARALLELIZATION.md`](docs/PARALLELIZATION.md).

`evaluate_combos` runs as **one MPI rank per node**, with each rank driving 32 user threads via the `--bypass-zarr-sync` machinery (default on).  Scale out by increasing `--nodes` and keeping `--ntasks-per-node=1`:

```bash
#SBATCH --nodes=8 --ntasks-per-node=1 --cpus-per-task=32

srun --unbuffered dc_toolkit evaluate_combos input.nc \
    --where-to-write ./out \
    --field-to-compress t \
    --l1-threshold 0.005 \
    --eval-data-size-limit 5GB \
    --threads-per-rank 32
```

One rank per node with 32 threads holds about 65 × the sample (one copy plus each thread's decoded buffers); 32 single-threaded ranks (`--allow-multi-rank-per-node`) each hold their own copy, about 96 × the sample per node: ~1.5× more, not 32×, because the per-thread decoded buffer dominates either way.  `docs/PARALLELIZATION.md` has the arithmetic.  `santis.run` is the production driver; it reads the input location from the environment: `DYAMOND_DATA_ROOT=/path/to/parent sbatch santis.run`, where the parent directory holds the `Data_Dyamond_PostProcessed*` trees.

Codec-internal thread pools must be pinned to 1 to avoid nested oversubscription (the tool checks this at startup and aborts by default; `--no-oversubscription-check` disables the guard):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       BLOSC_NTHREADS=1 NUMBA_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1
```

`--codec-threads N` (default 1, on `evaluate_combos`, `compress` and `from_zarr_to_netcdf`) enables codec-internal threading.  Compression here is memory-bandwidth-bound, so codec-internal threads compete with the outer threads for the same bandwidth and gain nothing on this workload; leave it at 1 unless an A/B test says otherwise.

`compress` is a single-process command — launch with `srun -n 1 ...` or plain invocation. Parallelism inside the write comes from dask's threaded scheduler, tuned via `--threads` (default: auto-detected from visible cores), `--inner-chunk-mib` (default: the sweep's value from the manifest, else 16), and `--shard-mib` (default: 512). `--verify/--no-verify` (default on) re-reads the store to compute error norms — skip with `--no-verify` on re-compression runs where the combo is already trusted.

## EBCC (optional)

[EBCC](https://github.com/spcl/EBCC) compresses each `(lat, lon)` frame with a JPEG 2000 base layer plus an
error-bounded residual.  At loose error bounds (0.1 to 1 % of the field's range) it reaches 2 to 4x the ratio of
zfp; at tight bounds the advantage disappears.  It is **off by default** because it is slow to encode (about
1 to 2 MB/s per thread, 200x slower than zfp; decoding is 30 to 200 MB/s) and because of its constraints:

- float fields whose last two dims are a `(lat, lon)` frame (native ICON grids do not qualify); float64 is
  down-cast to float32 through the `AsType` filter;
- no `NaN`/`Inf` anywhere in the field: the EBCC library terminates the process on them, so the toolkit checks
  first and skips or refuses instead;
- one frame (or an exact tile of it, 32 to 2047 cells per side) per inner chunk; `--inner-chunk-mib` and `--spatial-split` are ignored, `--shard-mib` still groups frames into shards;
- runs alone: a filter in front breaks its error bound and a compressor after it gains nothing;
- a store written with EBCC can only be read where `dc_toolkit[ebcc]` is installed (see [Reading a store without dc_toolkit](#reading-a-store-without-dc_toolkit)).

Install (needs `cmake`, a C/C++ toolchain and HDF5 headers; the Docker image includes it):

```commandline
pip install -e ".[ebcc]"          # or: WITH_EBCC=1 bash install_dc_toolkit.sh
```

Use `--with-ebcc` on `evaluate_combos` to add seven EBCC combos next to the regular sweep, or
`--serializer-class ebcc` to sweep EBCC alone.  EBCC is lossy, so both forms need `--with-lossy` (the default).
Each combo bounds the maximum absolute error at a fraction of the field's value range, from 10 % down to
0.01 % (EBCC's own floor is range/65535, its base layer being 16-bit).  They appear in the results with
no compressor, no filter (or the `AsType` cast to float32 for float64, which EBCC brings along even when `--filter-class` excludes it) and are never cut by `--max-evals`.  A winning EBCC
pipeline persists like any other (its tile size and error target travel in the pipeline JSON); `compress`
refuses it up front when the field has `NaN`/`Inf`, is not float32 without the `AsType` filter, or has a
frame the tile does not divide.  Keep `--eval-data-size-limit` small on EBCC sweeps: a 5 GB sample takes
about an hour per EBCC combo per thread.

## Reading a store without dc_toolkit

Every codec `compress` writes decodes in a zarr client without dc_toolkit (given numcodecs, `pcodec` and `zfpy`) except two, which exist only through
dc_toolkit's `zarr.codecs` entry point: `numcodecs.zfpy_flat` (the flattening ZFPY encoder; its bytes are
plain zfp, only the name is ours) and `numcodecs.ebcc_filter` (which also needs the `ebcc` package).  A store
holding either fails at `zarr.open` in a client without them, even for its other arrays, because zarr resolves
every array's codec chain when it opens the group's consolidated metadata.

Two ways round it.  `compress --stock-codecs-only` skips such winners and writes the best kept row of
`results_{var}.parquet` whose codecs are all stock, so the store opens anywhere (the sweep still evaluates every
pipeline).  Or register the name in the reader; for `zfpy_flat` this needs only `zfpy` and zarr, no dc_toolkit:

```python
import numcodecs, numcodecs.zfpy
from zarr.codecs.numcodecs import ZFPY
from zarr.registry import register_codec

class _ZFPYFlat(numcodecs.zfpy.ZFPY):          # numcodecs side: the same zfp codec under a second id
    codec_id = "zfpy_flat"
numcodecs.register_codec(_ZFPYFlat)

class ZFPYFlat(ZFPY, codec_name="zfpy_flat"):  # zarr side: the wrapper zarr instantiates from zarr.json
    pass
register_codec("numcodecs.zfpy_flat", ZFPYFlat)
```

Both registrations are needed: zarr v3 keeps its own codec registry on top of numcodecs'.  Decoding needs no
reshape logic, because a zfp stream carries its own shape.  EBCC has no such shortcut: `pip install "dc_toolkit[ebcc]"`.

## UI implementation

Two user interfaces wrap the same workflow for one field of a netCDF file: choose the codec space and the
relative L1 budget, run `evaluate_combos`, look at the combinations that passed the gates as KMeans scatter
plots of L1 / L2 / LInf vs ratio, pick a pipeline by name, run `compress` with it and save the store as a zip.
The web UI also shows the combinations as a table and exports the plots as HTML; the desktop UI opens the
plots in the browser.  Outputs go to `./out`; the UIs pin the codec thread variables for the commands they
launch.

The web UI (streamlit) runs its sweeps as one local process:
```
dc_toolkit run_web_ui
```
On a vcluster the same UI launches them under `srun` with the given allocation and works on a file that is
already on the cluster (`--uploaded_file`, required; `--partition` defaults to `debug`); forward the port
first (`ssh -L 8501:localhost:8501 santis`):
```
dc_toolkit run_web_ui_vcluster \
  --user_account "YOUR_USER_ACCOUNT" \
  --uenv_image "$UENV_NAME" \
  --uploaded_file "PATH_TO_FILE" \
  --time "00:15:00" \
  --nodes "1" --ntasks-per-node "1"
```
`evaluate_combos` runs one MPI rank per node (threads provide the intra-node parallelism); a higher `--ntasks-per-node` aborts at startup.

The desktop UI (Qt; installs `PyQt6` on first use) runs locally:
```
dc_toolkit run_local_ui
```

## Docker

The `Dockerfile` builds a self-contained image (all dependencies, the repository cloned inside):

```commandline
docker build -t dc-toolkit .
```
An example run:

```commandline
docker run \
  -u $(id -u):$(id -g) \
  -w /mnt/data/docker_saved_files \
  -v "$(pwd)/netCDF_files":/mnt/data \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e BLOSC_NTHREADS=1 -e NUMBA_NUM_THREADS=1 \
  -e VECLIB_MAXIMUM_THREADS=1 -e OMP_THREAD_LIMIT=1 \
  --entrypoint /bin/bash \
  dc-toolkit \
  -c 'mkdir -p docker_saved_files && dc_toolkit evaluate_combos /opt/data-compression/netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc --where-to-write /mnt/data/docker_saved_files --field-to-compress t --l1-threshold 0.005'
```

**Command Breakdown:**

* **`-u $(id -u):$(id -g)`**: Runs the container using your local machine's User and Group IDs rather than the Docker default `root`. The files written to your machine are then owned by you and aren't locked behind root permissions.
* **`-w /mnt/data/docker_saved_files`**: Sets the Working Directory.
* **`-v "$(pwd)/netCDF_files":/mnt/data`**: The volume mount. This creates a bridge between your local computer and the container so the toolkit can read your input data and write the results back to your hard drive.
* **`-e XDG_CACHE_HOME=/tmp/.cache`**: Sets the cache directory to a temporary location inside the container.
* **`-e OMP_NUM_THREADS=1 ...`**: Pins the codec-internal thread pools to 1; `evaluate_combos` aborts at startup otherwise (`--no-oversubscription-check` disables the guard).
* **`--entrypoint /bin/bash`**: Forces Docker to start with a Bash shell instead of the default program (dc_toolkit).
* **`dc-toolkit`**: The name of the Docker image to run.
* **`-c '...'`**: The shell command the container runs:
  * **`mkdir -p docker_saved_files`**: Creates an output directory on your host.
  * **`dc_toolkit evaluate_combos ...`**: Executes the actual compression tool, using a file inside the container and saving the results (under `--where-to-write`) to your mounted volume.

Or for the web UI:

```commandline
docker run -p 8501:8501 dc-toolkit run_web_ui
```

### Running with MPI (single-container, exercises the MPI code path)

OpenMPI + Docker requires specific file permission and cache handling. On a single container `evaluate_combos` runs with **one** MPI rank (`-n 1`): several ranks on one node abort at startup unless `--allow-multi-rank-per-node` is passed, and the parallelism comes from the rank's threads. The `mpirun` launch exercises the MPI code path in CI or smoke tests; for real multi-node speedup use SLURM (see the HPC section above).

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
  bash -c 'HOME=/tmp/$OMPI_COMM_WORLD_RANK exec dc_toolkit evaluate_combos /opt/data-compression/netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc --where-to-write /mnt/data/docker_saved_files --field-to-compress t --l1-threshold 0.005 --eval-data-size-limit 5GB'
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
  * **`HOME=/tmp/$OMPI_COMM_WORLD_RANK`**: A `$HOME` per rank under the writable `/tmp` (`/tmp/0` with `-n 1`), so ranks do not share caches.
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
  bash -c "HOME=/tmp/`$OMPI_COMM_WORLD_RANK exec dc_toolkit evaluate_combos /mnt/data/tigge_pl_t_q_dx=2_2024_08_02.nc --where-to-write /mnt/data/docker_saved_files --field-to-compress t --l1-threshold 0.005 --eval-data-size-limit 5GB"
```

**Command Breakdown:**

* **`-e HOME=/tmp`**: Sets a base temporary home directory for the container environment.
* **`-e OMP_NUM_THREADS=1 ...`**: Pins codec-internal thread pools to 1 (prevents nested oversubscription).
* **`-w /mnt/data/docker_saved_files`**: Sets the Working Directory inside the container.
* **`-v "${PWD}\netCDF_files:/mnt/data"`**: Windows equivalent of the volume mount. `${PWD}` dynamically grabs your current PowerShell directory to link your local files to the container.
* **`--entrypoint mpirun`**: Bypasses the default container start command to run OpenMPI.
* **`dc-toolkit`**: The image name.
* **`--allow-run-as-root`**: The container defaults to `root` on Windows; this flag bypasses OpenMPI's built-in safety restrictions against running parallel jobs as root.
* **`-n 1`**: One rank per node; on a Docker container that's one rank total.
* **`bash -c "..."`**: Executes the parallel command. Note double-quotes for PowerShell, with an escaped backtick (` `$ `) in front of the MPI variable to prevent PowerShell from evaluating it on your host before it reaches the container.

## Slides

### [Click here to view slides](https://c2sm.github.io/data-compression/)
