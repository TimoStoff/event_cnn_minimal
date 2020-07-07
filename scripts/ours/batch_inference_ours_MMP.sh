#!/bin/bash
model="${1?"Usage: $0 <model.pth> <input/folder> <output/folder>"}"
input_folder="${2?"Usage: $0 <model.pth> <input/folder> <output/folder>"}"
output_base_folder="${3?"Usage: $0 <model.pth> <input/folder> <output/folder>"}"
for input in "$input_folder"/*/; do
    sequence_name=$(basename "$input")
    output_folder="${output_base_folder}/${sequence_name}"
    python "${EVENT_CNN_BASE}/inference.py" \
        --checkpoint_path "$model" \
        --events_file_path "${input}/memmaps" \
        --output_folder "$output_folder" \
        --loader_type MMP
done
