# Data Compression Project

TODO

## Installation

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

A local version is also available:
````
python ./src/data_compression_cscs_exclaim/compression_analysis_ui_local.py
````