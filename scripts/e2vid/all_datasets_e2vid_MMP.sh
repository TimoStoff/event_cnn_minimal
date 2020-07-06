#!/bin/bash
model="${1?"Usage: $0 <model.pth> dataset_paths.txt <output/base/folder>"}"
data_file="${2?"Usage: $0 <model.pth> dataset_paths.txt <output/base/folder>"}"
output_base_folder="${3?"Usage: $0 <model.pth> dataset_paths.txt <output/base/folder>"}"
while read -r dataset; do
    ./batch_inference_e2vid_MMP.sh $model $dataset "${output_base_folder}/$(basename $dataset)"
done < "$data_file"
