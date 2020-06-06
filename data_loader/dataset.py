from torch.utils.data import Dataset
import numpy as np
import random
import torch
import h5py
# local modules
from utils.data_augmentation import Compose, RobustNorm
from utils.data import data_sources
from events_contrast_maximization.utils.event_utils import events_to_voxel_torch, \
    events_to_neg_pos_voxel_torch, binary_search_torch_tensor, events_to_image_torch, \
    binary_search_h5_dset

class BaseVoxelDataset(Dataset):

    def __init__(self, data_path, transforms=None, sensor_resolution=None, num_bins=5,
                 voxel_method=None, max_length=None, combined_voxel_channels=True):

        self.num_bins = num_bins
        self.data_path = data_path
        self.combined_voxel_channels = combined_voxel_channels
        self.sensor_resolution = sensor_resolution
        self.data_source_idx = -1
        self.has_flow = False

        self.sensor_resolution, self.t0, self.tk, self.num_events, self.frame_ts, self.num_frames = \
                None, None, None, None, None, None

        self.load_data(data_path)

        if self.sensor_resolution is None or self.has_flow is None or self.t0 is None \
                or self.tk is None or self.num_events is None or self.frame_ts is None:
            raise Exception("Dataloader failed to initialize")
        self.num_pixels = self.sensor_resolution[0] * self.sensor_resolution[1]
        self.duration = self.tk-self.t0

        if voxel_method is None:
            voxel_method = {'method': 'between_frames'}
        self.set_voxel_method(voxel_method)

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
    
    def __getitem__(self, index, seed=None):
        assert 0 <= index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        seed = random.randint(0, 2 ** 32) if seed is None else seed

        xs, ys, ts, ps = self.get_events(index)
        dt = ts[-1]-ts[0] 

        voxel = self.get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels=self.combined_voxel_channels)
        voxel = self.transform_voxel(voxel, seed)

        if self.voxel_method['method'] == 'between_frames':
            frame = self.get_frame(index+1)
            frame = self.transform_frame(frame, seed)

            if self.has_flow:
                flow = self.get_flow(index+1)
                # convert to displacement (pix)
                flow = flow * dt
                flow = self.transform_flow(flow, seed)
            else:
                flow = torch.zeros((2, frame.shape[-2], frame.shape[-1]), dtype=frame.dtype, device=frame.device)

            item = {'frame': frame,
                    'flow': flow,
                    'events': voxel,
                    'timestamp': self.frame_ts[index+1],
                    'data_source_idx': self.data_source_idx,
                    'dt': dt}
        else:
            item = {'events': voxel,
                    'timestamp': self.frame_ts[index+1],
                    'data_source_idx': self.data_source_idx,
                    'dt': dt}
        return item

    def get_frame(self, index):
        raise NotImplementedError

    def get_flow(self, index):
        raise NotImplementedError

    def get_events(self, index):
        raise NotImplementedError

    def load_data(self, data_path):
        """
        Load data from files. Must set the members:
            self.sensor_resolution, self.num_pixels, self.length, self.has_flow,
            self.t0, self.tk, self.duration, self.num_events, self.event_frame_indices
        """
        raise NotImplementedError

    def find_ts_index(self, timestamp):
        """
        Given a timestamp, find the event index
        """
        raise NotImplementedError

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for ts in self.frame_ts:
            end_index = self.find_ts_index(ts)
            frame_indices.append(start_idx, end_index)
            start_idx = end_index
        return frame_indices

    def compute_timeblock_indices(self):
        timeblock_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            start_time = (voxel_method['t']-voxel_method['sliding_window_t'])*i
            end_time = start_time + voxel_method['t']
            end_idx = self.find_ts_index(end_time)
            timeblock_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return timeblock_indices

    def compute_k_indices(self):
        k_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            idx0 = (voxel_method['k']-voxel_method['sliding_window_w'])*index
            idx1 = idx0 + voxel_method['k']
            k_indices.append(idx0, idx1)
        return k_indices

    def set_voxel_method(self, voxel_method):
        self.voxel_method = voxel_method
        if self.voxel_method['method'] == 'k_events':
            self.length = max(int(self.num_events/(voxel_method['k']-voxel_method['sliding_window_w'])), 0)
            self.event_indices = self.compute_k_indices()
        elif self.voxel_method['method'] == 't_seconds':
            self.length = max(int(self.duration/(voxel_method['t']-voxel_method['sliding_window_t'])), 0)
            self.event_indices = self.compute_timeblock_indices()
        elif self.voxel_method['method'] == 'between_frames':
            self.length = self.num_frames-1
            self.event_indices = self.compute_frame_indices()
        else:
            raise Exception("Invalid voxel forming method chosen ({})".format(self.voxel_method))
        if self.length == 0:
            raise Exception("Current voxel generation parameters lead to sequence length of zero")

    def __len__(self):
        return self.length

    def get_event_indices(self, index):
        idx0, idx1 = self.event_indices[index]
        if not (idx0 >= 0 and idx1 <= self.num_events):
            raise Exception("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))
        return idx0, idx1

    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
        # generate voxel grid which has size C x H x W
        if combined_voxel_channels:
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_resolution)
        else:
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_resolution)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)

        return voxel_grid

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

class DynamicH5Dataset(BaseVoxelDataset):
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
        data_path Path to the h5 file containing the event/image data
        transforms Dict containing the desired augmentations on the voxel grid
        sensor_resolution The size of the image sensor from which the events originate
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
    def get_frame(self, index):
        return self.h5_file['images']['image{:09d}'.format(index)][:]

    def get_flow(self, index):
        return self.h5_file['flow']['flow{:09d}'.format(index)][:]

    def get_events(self, index):
        idx0, idx1 = self.get_event_indices(index)
        xs = torch.from_numpy((self.h5_file['events/xs'][idx0:idx1]).astype(np.float32))
        ys = torch.from_numpy((self.h5_file['events/ys'][idx0:idx1]).astype(np.float32))
        ts = torch.from_numpy((self.h5_file['events/ts'][idx0:idx1] - self.t0).astype(np.float32))
        ps = torch.from_numpy((self.h5_file['events/ps'][idx0:idx1] * 2 - 1).astype(np.float32))
        return xs, ys, ts, ps

    def load_data(self, data_path):
        """
        Load data from files. Must set the members:
            self.sensor_resolution, self.num_pixels, self.length, self.has_flow,
            self.t0, self.tk, self.duration, self.num_events, self.event_frame_indices
        """
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
        else:
            self.sensor_resolution = sensor_resolution[0:2]
        print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['flow']) > 0
        self.t0 = self.h5_file['events/ts'][0]
        self.tk = self.h5_file['events/ts'][-1]
        self.num_events = self.h5_file.attrs["num_events"]
        self.num_frames = self.h5_file.attrs["num_imgs"]

        self.frame_ts = []
        for img_name in self.h5_file['images']:
            self.frame_ts.append(self.h5_file['images/{}'.format(img_name)].attrs['timestamp'])

        data_source = self.h5_file.attrs.get('source', 'unknown')
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        """
        Given a timestamp, find the event index
        """
        idx = binary_search_h5_dset(self.h5_file['events/ts'], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for img_name in self.h5_file['images']:
            end_idx = self.h5_file['images/{}'.format(img_name)].attrs['event_idx']
            frame_indices.append([start_idx, end_idx])
        return frame_indices

    #def __init__(self, data_path, transforms=None, sensor_resolution=None, num_bins=5,
    #             voxel_method=None, max_length=None, combined_voxel_channels=True):

    #    if transforms is None:
    #        transforms = {}
    #    if voxel_method is None:
    #        voxel_method = {'method': 'between_frames'}
    #    self.data_path = data_path

    #    try:
    #        self.h5_file = h5py.File(data_path, 'r')
    #    except OSError as err:
    #        print("Couldn't open {}: {}".format(data_path, err))

    #    self.num_bins = num_bins
    #    self.voxel_method = voxel_method
    #    if sensor_resolution is None:
    #        self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
    #        print("sensor size = {}".format(self.sensor_resolution))
    #    else:
    #        self.sensor_resolution = sensor_resolution[0:2]

    #    self.num_events = self.h5_file.attrs['num_events']
    #    self.duration = self.h5_file.attrs['duration']
    #    self.t0 = self.h5_file.attrs['t0']
    #    self.np_ts = np.array(self.h5_file['events/ts'])

    #    data_source = self.h5_file.attrs.get('source', 'unknown')
    #    try:
    #        self.data_source_idx = data_sources.index(data_source)
    #    except ValueError:
    #        self.data_source_idx = -1

    #    if self.voxel_method['method'] == 'k_events':
    #        self.length = max(int((self.num_events - voxel_method['k']) / voxel_method['sliding_window_w']) + 1, 0)
    #    elif self.voxel_method['method'] == 't_seconds':
    #        self.length = max(int((self.duration - voxel_method['t']) / voxel_method['sliding_window_t']) + 1, 0)
    #    elif self.voxel_method['method'] == 'between_frames':
    #        self.length = self.h5_file.attrs['num_imgs'] - 1
    #    else:
    #        raise Exception("Invalid voxel forming method chosen ({})".format(self.voxel_method))
    #    if self.length == 0:
    #        raise Exception("Current voxel generation parameters lead to sequence length of zero")

    #    self.normalize_voxels = False
    #    if 'RobustNorm' in transforms.keys():
    #        vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]
    #        del (transforms['RobustNorm'])
    #        self.normalize_voxels = True
    #        self.vox_transform = Compose(vox_transforms_list)

    #    transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]

    #    if len(transforms_list) == 0:
    #        self.transform = None
    #    elif len(transforms_list) == 1:
    #        self.transform = transforms_list[0]
    #    else:
    #        self.transform = Compose(transforms_list)
    #    if not self.normalize_voxels:
    #        self.vox_transform = self.transform

    #    if max_length is not None:
    #        self.length = min(self.length, max_length + 1)
    #    self.combined_voxel_channels = combined_voxel_channels

    #def __len__(self):
    #    return self.length

    #def transform_frame(self, frame, seed):
    #    frame = torch.from_numpy(frame).float().unsqueeze(0) / 255
    #    if self.transform:
    #        random.seed(seed)
    #        frame = self.transform(frame)
    #    return frame

    #def transform_voxel(self, voxel, seed):
    #    if self.vox_transform:
    #        random.seed(seed)
    #        voxel = self.vox_transform(voxel)
    #    return voxel

    #def transform_flow(self, flow, seed):
    #    flow = torch.from_numpy(flow)  # should end up [2 x H x W]
    #    if self.transform:
    #        random.seed(seed)
    #        flow = self.transform(flow, is_flow=True)
    #    return flow

    #def __getitem__(self, i, seed=None):
    #    assert (i >= 0)
    #    assert (i < self.length)

    #    if self.voxel_method['method'] == 'between_frames':
    #        try:
    #            dset = self.h5_file
    #            img_dset = self.h5_file['images']['image{:09d}'.format(i + 1)]
    #        except Exception as e:
    #            print("Incorrect index: {}".format(e))
    #            raise e

    #        timestamp = img_dset.attrs['timestamp']

    #        events_start_idx = self.h5_file['images']['image{:09d}'.format(i)].attrs['event_idx'] + 1
    #        events_end_idx = img_dset.attrs['event_idx']
    #    elif self.voxel_method['method'] == 'k_events':
    #        events_start_idx = i * self.voxel_method['sliding_window_w']
    #        events_end_idx = self.voxel_method['k'] + events_start_idx

    #        timestamp = self.h5_file['events/ts'][events_start_idx]
    #    elif self.voxel_method['method'] == 't_seconds':
    #        events_start_idx = np.searchsorted(self.np_ts,
    #                                           i * self.voxel_method['sliding_window_t'] + self.t0)
    #        events_end_idx = np.searchsorted(self.np_ts,
    #                                         self.voxel_method['t'] + i * self.voxel_method['sliding_window_t'] + self.t0)

    #        if events_end_idx == events_start_idx:
    #            events_end_idx = events_start_idx + 1
    #            print('WARNING! Set events_end_idx to events_start_idx + 1 at i={}'.format(i))

    #        timestamp = self.h5_file['events/ts'][events_start_idx]
    #    else:
    #        raise Exception("Unsupported voxel_method")

    #    assert (events_end_idx <= self.num_events)

    #    xs = torch.from_numpy((self.h5_file['events/xs'][events_start_idx:events_end_idx]).astype(np.float32))  # H x W
    #    ys = torch.from_numpy((self.h5_file['events/ys'][events_start_idx:events_end_idx]).astype(np.float32))  # H x W
    #    ts = torch.from_numpy(
    #        (self.h5_file['events/ts'][events_start_idx:events_end_idx] - self.t0).astype(np.float32))  # H x W
    #    ps = torch.from_numpy(
    #        (self.h5_file['events/ps'][events_start_idx:events_end_idx] * 2 - 1).astype(np.float32))  # H x W
    #    if self.combined_voxel_channels:
    #        voxel = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_resolution=self.sensor_resolution).float()
    #    else:
    #        voxel = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_resolution=self.sensor_resolution)
    #        voxel = torch.cat([voxel[0], voxel[1]], dim=0).float()

    #    if seed is None:
    #        seed = random.randint(0, 2 ** 32)

    #    voxel = self.transform_voxel(voxel, seed)
    #    dt = self.h5_file['events/ts'][events_end_idx] - self.h5_file['events/ts'][events_start_idx]

    #    if self.voxel_method['method'] == 'between_frames':
    #        frame = img_dset[:]  # H x W
    #        frame = self.transform_frame(frame, seed)

    #        if dset.attrs['num_flow'] == dset.attrs['num_imgs']:
    #            flow = self.h5_file['flow']['flow{:09d}'.format(i + 1)][:]
    #            flow = self.transform_flow(flow, seed)
    #            # convert to displacement (pix)
    #            flow *= dt
    #        else:
    #            flow = torch.zeros((2, frame.shape[-2], frame.shape[-1]),
    #                               dtype=frame.dtype, device=frame.device)

    #        item = {'frame': frame,
    #                'flow': flow,
    #                'events': voxel,
    #                'timestamp': timestamp,
    #                'data_source_idx': self.data_source_idx,
    #                'dt': torch.tensor(dt)}
    #    else:
    #        item = {'events': voxel,
    #                'timestamp': timestamp,
    #                'data_source_idx': self.data_source_idx,
    #                'dt': torch.tensor(dt)}
    #    return item


class MemMapDataset(Dataset):
    """
    Loads time-synchronized, event voxel grids, optic flow and
    standard frames from numpy memmaps containing events, optic flow and
    frames. Voxel grids are formed on-the-fly.

    This Dataset class iterates through all the event tensors and returns,
    for each tensor, a dictionary where:

    * frame is a H x W tensor containing the first frame whose
      timestamp >= event tensor
    * events is a C x H x W tensor containing the event data
    * flow is a 2 x H x W tensor containing the flow (displacement) from
      the current frame to the last frame

    Parameters:
        data_path Path to the h5 file containing the event/image data
        transforms Dict containing the desired augmentations on the voxel grid
        sensor_resolution The size of the image sensor from which the events originate
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

    def __init__(self,
                 data_path,
                 transforms={},
                 num_bins=5,
                 max_length=None,
                 voxel_method=None,
                 config=None):

        if isinstance(config, dict):
            self.config = config
            data_source = 'unknown'
        else:
            if config is None:
                config = os.path.join(data_path, "dataset_config.json")
            assert (os.path.exists(config))
            self.config = read_json(config)
            data_source = self.config['data_source']
        assert "sensor_resolution" in self.config

        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

        self.sensor_resolution = self.config["sensor_resolution"]
        self.num_pixels = self.sensor_resolution[0] * self.sensor_resolution[1]
        self.num_bins = num_bins
        self.data_path = data_path

        self.filehandle = self.load_files(data_path)

        if voxel_method is None:
            voxel_method = {'method': 'between_frames'}
        self.set_voxel_method(voxel_method)

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

        self.length = self.__len__()
        if max_length is not None:
            self.length = min(self.length, max_length + 1)

    def set_voxel_method(self, voxel_method):
        self.voxel_method = voxel_method
        if self.voxel_method['method'] == 'k_events':
            self.length = max(int(self.num_events/(voxel_method['k']-voxel_method['sliding_window_w'])), 0)
        elif self.voxel_method['method'] == 't_seconds':
            self.length = max(int(self.duration/(voxel_method['t']-voxel_method['sliding_window_t'])), 0)
        elif self.voxel_method['method'] == 'between_frames':
            self.length = self.filehandle['num_imgs'] - 1
        else:
            raise Exception("Invalid voxel forming method chosen ({})".format(self.voxel_method))
        if self.length == 0:
            raise Exception("Current voxel generation parameters lead to sequence length of zero")

    def compute_sequence_stats(self):
        # if index is not in dict compute them from frame_stamps
        if "length" not in data:
            if self.voxel_method is 'between_frames':
                data["length"] = len(data["index"])
            elif self.voxel_method is 'k_events':
                data["length"] = int((len(data['p']) - 2 * self.events_per_voxel_grid) // (
                            self.events_per_voxel_grid * (1.0 - self.overlap_perc)))
            elif self.voxel_method is 't_seconds':
                t0, tk = data['t'][0][0], data['t'][-1][0]
                seq_dur = tk - t0
                data["length"] = int((seq_dur + self.period - self.duration) / self.period)
                starting_indices = []
                end_indices = []
                for i in range(data["length"]):
                    t_start = t0 + self.period * i
                    starting_indices.append(np.searchsorted(np.squeeze(data['t']), t_start))
                    end_indices.append(np.searchsorted(np.squeeze(data['t']), t_start + self.duration) - 1)
                data["start_indices"] = starting_indices
                data["end_indices"] = end_indices
            # print("{}: Enough for {} unique voxel grids".format(subroot, data["length"]))

    def load_files(self, rootdir):
        assert os.path.isdir(rootdir), '%s is not a valid rootdirectory' % rootdir

        data = {}
        self.has_flow = False
        for subroot, _, fnames in sorted(os.walk(rootdir)):
            for fname in sorted(fnames):
                path = os.path.join(subroot, fname)

                if fname.endswith(".npy"):
                    if fname.endswith("index.npy"):  # index mapping image index to event idx
                        indices = np.load(path)  # N x 2
                        assert len(indices.shape) == 2 and indices.shape[1] == 2
                        indices = indices.astype("int64")  # ignore event indices which are 0 (before first image)
                        data["index"] = indices.T
                    elif fname.endswith("timestamps.npy"):
                        frame_stamps = np.load(path)
                        data["frame_stamps"] = frame_stamps
                    elif fname.endswith("images.npy"):
                        data["images"] = np.load(path, mmap_mode="r")
                    elif fname.endswith("optic_flow.npy"):
                        data["optic_flow"] = np.load(path, mmap_mode="r")
                        self.has_flow = True
                    elif fname.endswith("optic_flow_timestamps.npy"):
                        optic_flow_stamps = np.load(path)
                        data["optic_flow_stamps"] = optic_flow_stamps

                    handle = np.load(path, mmap_mode="r")
                    if fname.endswith("t.npy"):  # timestamps
                        data["t"] = handle
                    elif fname.endswith("xy.npy"):  # coordinates
                        data["xy"] = handle
                    elif fname.endswith("p.npy"):  # polarity
                        data["p"] = handle
            if len(data) > 0:
                data['path'] = subroot
                if "t" not in data:
                    print(f"Ignoring rootdirectory {subroot} since no events")
                    continue
                assert (len(data['p']) == len(data['xy']) and len(data['p']) == len(data['t']))
                self.num_events = len(data['p'])

                if "index" not in data and "frame_stamps" in data:
                    data["index"] = find_event_indices_for_frame(data["t"], data['frame_stamps'])

                self.t0, self.tk = data['t'][0][0], data['t'][-1][0]
                self.duration = self.tk-self.t0
        return data

    def __getitem__(self, index, seed=None):
        assert 0 <= index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        voxel = self.get_voxel_grid(self.filehandle, index)
        frame = self.filehandle['images'][index + 1][:, :, 0]

        if seed is None:
            # if no specific random seed was passed, generate our own.
            seed = random.randint(0, 2 ** 32)

        voxel = self.transform_voxel(voxel, seed)
        dt = self.filehandle['frame_stamps'][index + 1] - self.filehandle['frame_stamps'][index]

        if self.voxel_method['method'] == 'between_frames':
            frame = self.transform_frame(frame, seed)
            if self.has_flow:
                flow = self.filehandle['optic_flow'][index + 1]
                # convert to displacement (pix)
                flow = flow * dt
                flow = self.transform_flow(flow, seed)
            else:
                flow = torch.zeros((2, frame.shape[-2], frame.shape[-1]), dtype=frame.dtype, device=frame.device)

            item = {'frame': frame,
                    'flow': flow,
                    'events': voxel,
                    'timestamp': self.filehandle["frame_stamps"][1 + index],
                    'data_source_idx': self.data_source_idx,
                    'dt': dt}
        else:
            item = {'events': voxel,
                    'timestamp': self.filehandle["frame_stamps"][1 + index],
                    'data_source_idx': self.data_source_idx,
                    'dt': dt}
        return item

    def get_voxel_grid(self, filehandle, index, combined_voxel_channels=True):
        if self.voxel_method is 'between_frames':
            indices = filehandle["index"]
            idx1, idx0 = indices[index]
        elif self.voxel_method is 'k_events':
            idx0 = int((1.0 - self.overlap_perc) * self.events_per_voxel_grid * index)
            idx1 = int(idx0 + self.events_per_voxel_grid)
        elif self.voxel_method is 't_seconds':
            idx0 = filehandle["start_indices"][index]
            idx1 = filehandle["end_indices"][index]
        else:
            raise Exception("voxel_method is not supported")

        if not (idx0 >= 0 and idx1 <= filehandle["num_events"]):
            print("WARNING: Either {}<0s or {}>{}".format(idx0, idx1, filehandle["num_events"]))
        assert (idx0 >= 0 and idx1 <= filehandle["num_events"])
        # select events with indices between current frame at index and next frame at index+1

        events_xy = filehandle["xy"][idx0:idx1]
        events_t = filehandle["t"][idx0:idx1] - filehandle["t"][idx0]
        events_p = filehandle["p"][idx0:idx1]

        # generate voxel grid which has size C x H x W
        H, W = self.sensor_resolution
        channels = self.num_bins

        if len(events_xy) < 2:
            return np.zeros((2 * channels, H, W))

        if combined_voxel_channels:
            voxel_grid = er.events_to_voxel_torch(events_xy[:, 0], events_xy[:, 1],
                    events_t, events_p, channels, (H, W))
        else:
            voxel_grid = er.events_to_neg_pos_voxel(events_xy[:, 0], events_xy[:, 1],
                    events_t, events_p, channels, (H, W))
            voxel_grid = np.concatenate([voxel_grid[0], voxel_grid[1]], 0)

        return voxel_grid

    def transform_frame(self, frame, seed):
        frame = torch.from_numpy(frame).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)
        return frame

    def transform_voxel(self, voxel, seed):
        voxel = torch.from_numpy(voxel).type(torch.FloatTensor)  # [C x H x W]
        if self.transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel

    def transform_flow(self, flow, seed):
        flow = torch.from_numpy(flow)  # should end up [2 x H x W]
        if self.transform:
            random.seed(seed)
            flow = self.transform(flow, is_flow=True)
        return flow

    @staticmethod
    def find_event_indices_for_frame(event_stamps, frame_stamps):
        # find the event index corresponding to the frame ts
        indices_first = np.searchsorted(event_stamps[:, 0], frame_stamps[1:])
        indices_last = np.searchsorted(event_stamps[:, 0], frame_stamps[:-1])
        index = np.stack([indices_first, indices_last], -1)
        return index

    def __len__(self):
        return self.length
