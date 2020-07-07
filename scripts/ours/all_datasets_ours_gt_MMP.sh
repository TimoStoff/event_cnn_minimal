#!/bin/bash
model="${1?"Usage: $0 <model.pth> dataset_paths.txt <output/base/folder> <output/base/folder/gt>"}"
data_file="${2?"Usage: $0 <model.pth> dataset_paths.txt <output/base/folder> <output/base/folder/gt>"}"
output_base_folder="${3?"Usage: $0 <model.pth> dataset_paths.txt <output/base/folder> <output/base/folder/gt>"}"
output_base_folder_gt="${4?"Usage: $0 <model.pth> dataset_paths.txt <output/base/folder> <output/base/folder/gt>"}"
while read -r dataset; do
    ./batch_inference_ours_MMP.sh $model $dataset "${output_base_folder}/$(basename $dataset)" "${output_base_folder_gt}/$(basename $dataset)"
done < "$data_file"
