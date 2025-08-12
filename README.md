# Data Compression Project

TODO

## Installation

In Santis@ALPS:

 ```commandline
export UENV_NAME="prgenv-gnu/25.06:rc5"
uenv image pull $UENV_NAME
uenv start --view=default $UENV_NAME
```

once the above is complete (just for Santis, locally it is not needed):

```commandline
git clone git@github.com:C2SM/data-compression.git
python -m venv venv
source venv/bin/activate
bash install_data_compression.sh
```

## Usage

```
--------------------------------------------------------------------------------

Usage: data_compression_cscs_exclaim linear_quantization_zlib_compressors
           [OPTIONS] NETCDF_FILE FIELD_TO_COMPRESS PARAMETERS_FILE

Example:

data_compression_cscs_exclaim linear_quantization_zlib_compressors netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc t parameters.yaml

Options:
  --help  Show this message and exit.

--------------------------------------------------------------------------------
```

## UI implementation

Two User Interfaces have been implemented to make the file compression process more user-friendly.
Both UIs provide functionlaities for compressors similarity metrics and file compression.

compression_analysis_ui_web.py is the web app.
Outside of the mutual UI functionalities, this UI allows users to download similarity metrics plots and tweak parameters more dynamically, though it is a bit slower.

```
streamlit run ./src/data_compression_cscs_exclaim/compression_analysis_ui_web.py [OPTIONAL] --server.maxUploadSize=FILE_SIZE_MB --server.maxMessageSize=FILE_SIZE_MB

```
if launched from santis, make sure to ssh correctly:
```
ssh -L 8501:localhost:8501 santis
```

A local version is also available:
````
python ./src/data_compression_cscs_exclaim/compression_analysis_ui_local.py
````