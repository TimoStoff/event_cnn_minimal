import argparse
import cv2
import os
import numpy as np
from util import setup_output_folder, append_timestamp
from os.path import join
from tqdm import tqdm


def load_data(data_path, timestamp_fname='timestamps.npy', image_fname='images.npy'):

    assert os.path.isdir(data_path), '%s is not a valid data_pathectory' % data_path

    data = {}
    for subroot, _, fnames in sorted(os.walk(data_path)):
        for fname in sorted(fnames):
            path = os.path.join(subroot, fname)
            if fname.endswith('.npy'):
                if fname.endswith(timestamp_fname):
                    frame_stamps = np.load(path)
                    data['frame_stamps'] = frame_stamps
                elif fname.endswith(image_fname):
                    data['images'] = np.load(path, mmap_mode='r')  # N x H x W x C
    return data


def save_images(data, output_folder, ts_path):
    for i, (image, ts) in enumerate(zip(tqdm(data['images']), data['frame_stamps'])):
        fname = 'frame_{:010d}.png'.format(i)
        cv2.imwrite(join(output_folder, fname), image)
        append_timestamp(ts_path, fname, ts)


def main(args):
    data = load_data(args.data_path)
    ts_path = setup_output_folder(args.output_folder)
    save_images(data, args.output_folder, ts_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('output_folder', type=str)
    args = parser.parse_args()
    main(args)
