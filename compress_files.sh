#!/bin/bash

#SBATCH --account=cwp03
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=out-%j.out
#SBATCH --error=out-%j.out

export PYTHONUNBUFFERED=1
source venv/bin/activate

INPUT_DIR="/capstor/store1/cscs/userlab/cwp03/zemanc/Data_Dyamond_PostProcessed/out_1_1"
for filepath in "$INPUT_DIR"/*.nc; do
    filename="$(basename "$filepath")"
    echo "Processing: $filename"
    srun dc_toolkit \
        evaluate_combos \
        "$filepath" \
        ./temp \
        --field-to-compress qv \
        --override-existing-l1-error 0.002 \
        --without-lossy \
        --without-numcodecs-wasm \
        --without-ebcc \
    || echo "qv var not in file: $filename"
    rm -rf ./temp
done

