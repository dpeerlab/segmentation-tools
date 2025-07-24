#!/bin/bash

set -euo pipefail

TSV_FILE="/data1/peerd/ghoshr/alignment/xenium_if_pairs.tsv"
SCRIPT="/data1/peerd/ghoshr/segmentation_tools/src/segmentation_tools/scripts/convert_to_ometiff.sh"
OUTPUT_DIR="/data1/peerd/ghoshr/alignment/if_tif_files"
CONDA_ENV="segmentation-tools"
PARTITION_NAME="peerd"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# === Load Conda and activate environment ===
# source ~/.bashrc
# conda activate $CONDA_ENV

# Skip header and read image_path column
tail -n +2 "$TSV_FILE"  | cut -f2 | while read -r nd2_path; do
    if [[ ! -f "$nd2_path" ]]; then
        echo "WARNING: $nd2_path does not exist, skipping"
        continue
    fi


    base_name="$(basename "$nd2_path")"
    parent_dir="$(basename "$(dirname "$nd2_path")")"
    base_no_ext="${base_name%.*}"  # strips final extension
    output_path="${OUTPUT_DIR}/${parent_dir}_${base_no_ext}.tiff"
    job_name="convert_${parent_dir}_${base_no_ext}"

    echo "$job_name"

    sbatch --job-name="$job_name" \
           --output="logs/${job_name}.out" \
           --error="logs/${job_name}.err" \
           --time=01:00:00 \
           --mem=32G \
           --cpus-per-task=2 \
           --partition=$PARTITION_NAME \
           --wrap="bash \"$SCRIPT\" \"$nd2_path\" \"$OUTPUT_DIR\""
done
