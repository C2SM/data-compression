 
<div align="center">
  <img src="./data-compression_logo.png" alt="Logo" width="200"/>
</div>

# Data Compression Project 

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

In Santis@ALPS:

 ```commandline
export UENV_NAME="prgenv-gnu/24.11:v2"
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
dc_toolkit run_web_ui_santis --user_account "d75" --uploaded_file "./netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc" --time "00:15:00" --nodes "1" --ntasks-per-node "72"
```
Local web-versions and non are also available:
```
dc_toolkit run_local_ui
```
````
dc_toolkit run_web_ui
````
