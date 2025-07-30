#!/bin/bash

pip install -e .
pip install -r requirements.txt

# Install EBCC with Zarr support
git clone --recursive https://github.com/spcl/EBCC.git
pushd EBCC
pip install -e ".[zarr]"
popd
