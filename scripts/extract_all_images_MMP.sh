#!/bin/bash
input_folder="${1?"Usage: $0 <input/folder> <output/folder>"}"
output_base_folder="${2?"Usage: $0 <input/folder> <output/folder>"}"
for input in "$input_folder"/*/; do
    sequence_name=$(basename "$input")
    output_folder="${output_base_folder}/${sequence_name}"
    python "${EVENT_CNN_BASE}/utils/extract_images_MMP.py" \
        "${input}/memmaps" \
        "$output_folder"
done
