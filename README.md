 
<div align="center">
  <img src="./data-compression_logo.png" alt="Logo" width="300"/>
</div>

Set of tools for compressing netCDF files with Zarr. 

The tools use the following compression libraries:

- [Numcodecs](https://github.com/zarr-developers/numcodecs): Zarr native library [[documentation](https://numcodecs.readthedocs.io/en/stable/)]
- [numcodecs-wasm](https://github.com/juntyr/numcodecs-rs): Compression for codecs compiled to WebAssembly [[documentation](https://numcodecs-wasm.readthedocs.io/en/latest/)]
- [EBCC](https://github.com/spcl/EBCC): Error Bounded Climate Compressor [[documentation](https://github.com/spcl/EBCC/blob/master/README.md)]

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
git clone git@github.com:C2SM/data-compression.git
python -m venv venv
source venv/bin/activate
bash install_dc_toolkit.sh
```

## Usage

```
--------------------------------------------------------------------------------

Usage: dc_toolkit --help # List of available commands

Usage: dc_toolkit COMMAND --help # Documentation per command

Example:

dc_toolkit \ # CLI-tool
  evaluate_combos \ # command
  netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc \ # netCDF file to compress
  ./dump \ # where to write the compressed file(s)
  --field-to-compress t # field of netCDF to compress

--------------------------------------------------------------------------------
```

## UI implementation

Two User Interfaces have been implemented to make the file compression process more user-friendly.
Both UIs provide functionlaities for compressors similarity metrics and file compression.

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
docker run dc-toolkit \
  evaluate_combos \
  /opt/data-compression/netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc \
  /opt/data-compression/dump \
  --field-to-compress t
```

Or for the web UI:

```commandline
docker run -p 8501:8501 dc-toolkit run_web_ui
```

### Running with MPI (Parallel Processing)

To drastically speed up the evaluation process, you can run the toolkit in parallel using OpenMPI. 

Because running MPI inside Docker requires some specific file permission and cache handling, use the following commands to securely mount your directories and isolate the process caches depending on your operating system.

---

#### Mac and Linux

For Unix-based systems, we map your local user ID to the container to avoid permission issues and assign unique temporary directories to isolate caches.

```bash
docker run \
  -u $(id -u):$(id -g) \
  -w /mnt/data/docker_saved_files \
  -v $(pwd)/netCDF_files:/mnt/data \
  --entrypoint mpirun \
  dc-toolkit \
  -n 8 \
  bash -c 'HOME=/tmp/$OMPI_COMM_WORLD_RANK exec dc_toolkit evaluate_combos /mnt/data/tigge_pl_t_q_dx=2_2024_08_02.nc /mnt/data/docker_saved_files --field-to-compress t'
```

**Command Breakdown:**

* **`-u $(id -u):$(id -g)`**: Runs the container using your local machine's User and Group IDs rather than the Docker default `root`. This guarantees that compressed files output to your machine, are fully owned by you and aren't locked behind root permissions.
* **`-w /mnt/data/docker_saved_files`**: Sets the Working Directory.
* **`-v $(pwd)/netCDF_files:/mnt/data`**: The volume mount. This creates a bridge between your local computer and the container so the toolkit can read your input data and write the results back to your hard drive.
* **`--entrypoint mpirun`**: Tells Docker to bypass the image's default entrypoint and boot up using OpenMPI's runner instead.
* **`dc-toolkit`**: The name of the Docker image to run.
* **`-n 8`**: Tells `mpirun` to spin up 8 parallel processes.
* **`bash -c '...'`**: Executes a custom shell command across all 8 processes to handle the complex environment setup:
  * **`HOME=/tmp/$OMPI_COMM_WORLD_RANK`**: Assigns a mathematically unique, temporary "Home" directory to each process. This completely eliminates race conditions where multiple processes try to write to the exact same  cache simultaneously.
  * **`exec dc_toolkit evaluate_combos ...`**: Executes the actual compression tool, passing the paths (as they appear *inside* the container's `/mnt/data` mount) to the input NetCDF file and the designated output directory.

---

#### Windows (PowerShell)

When using Docker Desktop on Windows via WSL 2, Docker handles file permissions differently. You do not need to pass your user ID (as Docker Desktop handles the translation automatically), but you do need to explicitly allow OpenMPI to run as root and format your paths for PowerShell.

```powershell
docker run `
  -e HOME=/tmp `
  -w /mnt/data/docker_saved_files `
  -v "${PWD}\netCDF_files:/mnt/data" `
  --entrypoint mpirun `
  dc-toolkit `
  --allow-run-as-root `
  -n 8 `
  bash -c "HOME=/tmp/`$OMPI_COMM_WORLD_RANK exec dc_toolkit evaluate_combos /mnt/data/tigge_pl_t_q_dx=2_2024_08_02.nc /mnt/data/docker_saved_files --field-to-compress t"
```

**Command Breakdown:**

* **`-e HOME=/tmp`**: Sets a base temporary home directory for the container environment.
* **`-w /mnt/data/docker_saved_files`**: Sets the Working Directory inside the container so output files (like `config_space.csv`) drop exactly into your mounted folder.
* **`-v "${PWD}\netCDF_files:/mnt/data"`**: The Windows equivalent of the volume mount. `${PWD}` dynamically grabs your current PowerShell directory to link your local files to the container.
* **`--entrypoint mpirun`**: Bypasses the default container start command to run OpenMPI.
* **`dc-toolkit`**: The name of the Docker image.
* **`--allow-run-as-root`**: Because the container defaults to the `root` user on Windows, this flag is required to bypass OpenMPI's built-in safety restrictions against running parallel jobs as root.
* **`-n 8`**: Tells `mpirun` to spin up 8 parallel processes.
* **`bash -c "..."`**: Executes the parallel command. Notice that double-quotes are used here for PowerShell, with an escaped backtick (` `$ `) in front of the MPI variable to prevent PowerShell from prematurely evaluating it on your host machine before it reaches the container.

## Slides

### [Click here to view slides](https://c2sm.github.io/data-compression/)
