#!/bin/bash

set -euo pipefail

convert_file() {
    input_file="$1"
    output_file="$2"

    base_name="$(basename "$input_file")"
    base_name="${base_name%.*}"
    raw_dir=$(mktemp -d)

    echo "Processing: $input_file"

    if ! bioformats2raw "$input_file" "$raw_dir" --overwrite --progress; then
        echo "bioformats2raw failed on $input_file"
        rm -rf "$raw_dir"
        return
    fi

    if [[ ! -f "$raw_dir/.zgroup" ]]; then
        echo "bioformats2raw output missing .zgroup for $input_file"
        rm -rf "$raw_dir"
        return
    fi

    if ! raw2ometiff "$raw_dir" "$output_file" --progress;  then
        echo "raw2ometiff failed on $input_file"
        rm -rf "$raw_dir"
        return
    fi

    echo "Saved: $output_file"
    rm -rf "$raw_dir"
}

# === Main ===

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input_file_or_dir> <output_dir>"
    exit 1
fi

INPUT="$1"
OUTPUT_ROOT="$2"
mkdir -p "$OUTPUT_ROOT"

if [[ -f "$INPUT" ]]; then
    parent_dir="$(basename "$(dirname "$INPUT")")"
    base_name="$(basename "$INPUT")"
    base_no_ext="${base_name%.*}"  # strips final extension
    output_path="${OUTPUT_ROOT}/${parent_dir}_${base_no_ext}.tiff"
    convert_file "$INPUT" "$output_path"
elif [[ -d "$INPUT" ]]; then
    find "$INPUT" -type f | while read -r file; do
        rel_path="${file#$INPUT/}"  # get path relative to input dir
        output_subdir="${OUTPUT_ROOT}/$(dirname "$rel_path")"
        mkdir -p "$output_subdir"
        output_path="${output_subdir}/$(basename "${file%.*}").tiff"
        convert_file "$file" "$output_path"
    done

else
    echo "Input is not a valid file or directory"
    exit 1
fi
