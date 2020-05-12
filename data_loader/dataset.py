from torch.utils.data import Dataset
import numpy as np
import random
import torch
import h5py
# local modules
from utils.data_augmentation import Compose, RobustNorm
from utils.data import data_sources
from events_contrast_maximization.utils.event_utils import events_to_voxel_torch, \
    events_to_neg_pos_voxel_torch, binary_search_torch_tensor, events_to_image_torch


class DynamicH5Dataset(Dataset):
    """
    Loads time-synchronized, event voxel grids, optic flow and
    standard frames from a hdf5 file containing events, optic flow and
    frames. Voxel grids are formed on-the-fly.

    This Dataset class iterates through all the event tensors and returns,
    for each tensor, a dictionary where:

    * frame is a H x W tensor containing the first frame whose
      timestamp >= event tensor
    * events is a C x H x W tensor containing the event data
    * flow is a 2 x H x W tensor containing the flow (displacement) from
      the current frame to the last frame

    Parameters:
        h5_path Path to the h5 file containing the event/image data
        transforms Dict containing the desired augmentations on the voxel grid
        sensor_size The size of the image sensor from which the events originate
        num_bins The number of bins desired in the voxel grid
        voxel_method Which method should be used to form the voxels.
            Currently supports:
            * "k_events" (new voxels are formed every k events)
            * "t_seconds" (new voxels are formed every t seconds)
            * "between_frames" (all events between frames are taken, requires frames to exist)
            A sliding window width must be given for k_events and t_seconds,
            which determines overlap (no overlap if set to -1). Eg:
            method={'method':'k_events', 'k':10000, 'sliding_window_w':10000}
    """

    def __init__(self, h5_path, transforms=None, sensor_size=None, num_bins=5,
                 voxel_method=None, max_length=None, combined_voxel_channels=True):
        if transforms is None:
            transforms = {}
        if voxel_method is None:
            voxel_method = {'method': 'between_frames'}
        self.h5_path = h5_path
        try:
            self.h5_file = h5py.File(h5_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(h5_path, err))

        self.num_bins = num_bins
        self.voxel_method = voxel_method
        if sensor_size is None:
            self.sensor_size = self.h5_file.attrs['sensor_resolution'][0:2]
            print("sensor size = {}".format(self.sensor_size))
        else:
            self.sensor_size = sensor_size[0:2]

        self.num_events = self.h5_file.attrs['num_events']
        self.duration = self.h5_file.attrs['duration']
        self.t0 = self.h5_file.attrs['t0']
        self.np_ts = np.array(self.h5_file['events/ts'])

        data_source = self.h5_file.attrs.get('source', 'unknown')
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

        if self.voxel_method['method'] == 'k_events':
            self.length = max(int((self.num_events - voxel_method['k']) / voxel_method['sliding_window_w']) + 1, 0)
        elif self.voxel_method['method'] == 't_seconds':
            self.length = max(int((self.duration - voxel_method['t']) / voxel_method['sliding_window_t']) + 1, 0)
        elif self.voxel_method['method'] == 'between_frames':
            self.length = self.h5_file.attrs['num_imgs'] - 1
        else:
            raise Exception("Invalid voxel forming method chosen ({})".format(self.voxel_method))
        if self.length == 0:
            raise Exception("Current voxel generation parameters lead to sequence length of zero")

        self.normalize_voxels = False
        if 'RobustNorm' in transforms.keys():
            vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]
            del (transforms['RobustNorm'])
            self.normalize_voxels = True
            self.vox_transform = Compose(vox_transforms_list)

        transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]

        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)
        if not self.normalize_voxels:
            self.vox_transform = self.transform

        if max_length is not None:
            self.length = min(self.length, max_length + 1)
        self.combined_voxel_channels = combined_voxel_channels

    def __len__(self):
        return self.length

    def transform_frame(self, frame, seed):
        frame = torch.from_numpy(frame).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)
        return frame

    def transform_voxel(self, voxel, seed):
        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel

    def transform_flow(self, flow, seed):
        flow = torch.from_numpy(flow)  # should end up [2 x H x W]
        if self.transform:
            random.seed(seed)
            flow = self.transform(flow, is_flow=True)
        return flow

    def __getitem__(self, i, seed=None):
        assert (i >= 0)
        assert (i < self.length)

        if self.voxel_method['method'] == 'between_frames':
            try:
                dset = self.h5_file
                img_dset = self.h5_file['images']['image{:09d}'.format(i + 1)]
            except Exception as e:
                print("Incorrect index: {}".format(e))
                raise e

            timestamp = img_dset.attrs['timestamp']

            events_start_idx = self.h5_file['images']['image{:09d}'.format(i)].attrs['event_idx'] + 1
            events_end_idx = img_dset.attrs['event_idx']
        elif self.voxel_method['method'] == 'k_events':
            events_start_idx = i * self.voxel_method['sliding_window_w']
            events_end_idx = self.voxel_method['k'] + events_start_idx

            timestamp = self.h5_file['events/ts'][events_start_idx]
        elif self.voxel_method['method'] == 't_seconds':
            events_start_idx = np.searchsorted(self.np_ts,
                                               i * self.voxel_method['sliding_window_t'] + self.t0)
            events_end_idx = np.searchsorted(self.np_ts,
                                             self.voxel_method['t'] + i * self.voxel_method['sliding_window_t'] + self.t0)

            timestamp = self.h5_file['events/ts'][events_start_idx]
        else:
            raise Exception("Unsupported voxel_method")

        assert (events_end_idx <= self.num_events)

        xs = torch.from_numpy((self.h5_file['events/xs'][events_start_idx:events_end_idx]).astype(np.float32))  # H x W
        ys = torch.from_numpy((self.h5_file['events/ys'][events_start_idx:events_end_idx]).astype(np.float32))  # H x W
        ts = torch.from_numpy(
            (self.h5_file['events/ts'][events_start_idx:events_end_idx] - self.t0).astype(np.float32))  # H x W
        ps = torch.from_numpy(
            (self.h5_file['events/ps'][events_start_idx:events_end_idx] * 2 - 1).astype(np.float32))  # H x W
        if self.combined_voxel_channels:
            voxel = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_size).float()
        else:
            voxel = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_size)
            voxel = torch.cat([voxel[0], voxel[1]], dim=0).float()

        if seed is None:
            seed = random.randint(0, 2 ** 32)

        voxel = self.transform_voxel(voxel, seed)

        if self.voxel_method['method'] == 'between_frames':
            frame = img_dset[:]  # H x W

            frame = self.transform_frame(frame, seed)

            if dset.attrs['num_flow'] == dset.attrs['num_imgs']:
                flow = self.h5_file['flow']['flow{:09d}'.format(i + 1)][:]
                flow = self.transform_flow(flow, seed)
                # convert to displacement (pix)
                dt = events_end_idx - events_start_idx
                flow *= dt
            else:
                flow = torch.zeros((2, frame.shape[-2], frame.shape[-1]),
                                   dtype=frame.dtype, device=frame.device)

            item = {'frame': frame,
                    'flow': flow,
                    'events': voxel,
                    'timestamp': timestamp,
                    'data_source_idx': self.data_source_idx}
        else:
            item = {'events': voxel,
                    'timestamp': timestamp,
                    'data_source_idx': self.data_source_idx}

        return item
