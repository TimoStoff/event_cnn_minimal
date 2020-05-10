# event_cnn_minimal
Minimal code for running inference on models trained for ECCV'20.

# Running with [Anaconda](https://docs.anaconda.com/anaconda/install/)
```
cuda_version=10.1

conda create -y -n event_cnn python=3.7
conda activate event_cnn
conda install -y pytorch torchvision cudatoolkit=$cuda_version -c pytorch
conda install -y -c conda-forge opencv
conda install -y -c conda-forge tqdm
conda install -y -c anaconda h5py 
```
# Usage

Clone this repo and submodules:
```
git clone -b inference git@github.com:TimoStoff/event_cnn_minimal.git --recursive
```
Download the pretrained models:
```
cd event_cnn_minimal
mkdir pretrained
cd pretrained
wget <url>
```
This code processes the events in HDF5 format. To convert the rosbags to this format, run
```
python events_contrast_maximization/tools/rosbag_to_h5.py <path/to/rosbag/or/dir/with/rosbags> --output_dir <path/to/save_h5_events> --event_topic <event_topic> --image_topic <image_topic>
```
As an example, using [`slider_depth`](http://rpg.ifi.uzh.ch/datasets/davis/slider_depth.bag) from "The event camera dataset and simulator":
```
wget http://rpg.ifi.uzh.ch/datasets/davis/slider_depth.bag /tmp/
python events_contrast_maximization/tools/rosbag_to_h5.py /tmp/slider_depth.bag --output_dir /tmp/h5_events --event_topic /dvs/events --image_topic /dvs/image_raw
```
To estimate reconstruction:
```
python inference.py --checkpoint_path <path/to/model.pth> --device 0 --h5_file_path </path/to/events.h5> --output_folder </path/to/output/dir>
```
For example:
```
python inference.py --checkpoint_path pretrained/reconstruction/reconstruction_model.pth --device 0 --h5_file_path /tmp/h5_events/slider_depth.h5 --output_folder /tmp/reconstruction --legacy
```
To estimate flow:
```
python inference.py --checkpoint_path <path/to/model.pth> --device 0 --h5_file_path </path/to/events.h5> --output_folder </path/to/output/dir> --is_flow
```
For example:
```
python inference.py --checkpoint_path pretrained/flow/flow_model.pth --device 0 --h5_file_path /tmp/h5_events/slider_depth.h5 --output_folder /tmp/reconstruction --legacy --is_flow
```
Flow is saved as both a png showing HSV color as slow vectors and as npy files.

Note that the models reported on in the preprint ["How to Train Your Event Camera Neural Network](https://arxiv.org/abs/2003.09078) loaded the voxels in a slightly different way to the updated version. Hence when running inference on those models, the `--legacy` flag is necessary. Updated models are denoted by the prefix `update_` and don't require this flag to be set.
