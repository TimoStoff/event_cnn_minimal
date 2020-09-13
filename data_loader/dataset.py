from torch.utils.data import Dataset
import numpy as np
import random
import torch
import h5py
import os
# local modules
from utils.data_augmentation import *
from utils.data import data_sources
from events_contrast_maximization.utils.event_utils import events_to_voxel_torch, \
    events_to_neg_pos_voxel_torch, binary_search_torch_tensor, events_to_image_torch, \
    binary_search_h5_dset, get_hot_event_mask, save_image
from utils.util import read_json, write_json


class BaseVoxelDataset(Dataset):
    """
    Dataloader for voxel grids given file containing events.
    Also loads time-synchronized frames and optic flow if available.
    Voxel grids are formed on-the-fly.
    For each index, returns a dict containing:
        * frame is a H x W tensor containing the first frame whose
          timestamp >= event tensor
        * events is a C x H x W tensor containing the voxel grid
        * flow is a 2 x H x W tensor containing the flow (displacement) from
          the current frame to the last frame
        * dt is the time spanned by 'events'
        * data_source_idx is the index of the data source (simulated, IJRR, MVSEC etc)
    Subclasses must implement:
        - get_frame(index) method which retrieves the frame at index i
        - get_flow(index) method which retrieves the optic flow at index i
        - get_events(idx0, idx1) method which gets the events between idx0 and idx1
            (in format xs, ys, ts, ps, where each is a np array
            of x, y positions, timestamps and polarities respectively)
        - load_data() initialize the data loading method and ensure the following
            members are filled:
            sensor_resolution - the sensor resolution
            has_flow - if this dataset has optic flow
            t0 - timestamp of first event
            tk - timestamp of last event
            num_events - the total number of events
            frame_ts - list of the timestamps of the frames
            num_frames - the number of frames
        - find_ts_index(timestamp) given a timestamp, find the index of
            the corresponding event

    Parameters:
        data_path Path to the file containing the event/image data
        transforms Dict containing the desired augmentations
        sensor_resolution The size of the image sensor from which the events originate
        num_bins The number of bins desired in the voxel grid
        voxel_method Which method should be used to form the voxels.
            Currently supports:
            * "k_events" (new voxels are formed every k events)
            * "t_seconds" (new voxels are formed every t seconds)
            * "between_frames" (all events between frames are taken, requires frames to exist)
            A sliding window width must be given for k_events and t_seconds,
            which determines overlap (no overlap if set to 0). Eg:
            method={'method':'k_events', 'k':10000, 'sliding_window_w':100}
            method={'method':'t_events', 't':0.5, 'sliding_window_t':0.1}
            method={'method':'between_frames'}
            Default is 'between_frames'.
    """

    def get_frame(self, index):
        """
        Get frame at index
        """
        raise NotImplementedError

    def get_flow(self, index):
        """
        Get optic flow at index
        """
        raise NotImplementedError

    def get_events(self, idx0, idx1):
        """
        Get events between idx0, idx1
        """
        raise NotImplementedError

    def load_data(self, data_path):
        """
        Perform initialization tasks and ensure essential members are populated.
        Required members are:
            members are filled:
            self.sensor_resolution - the sensor resolution
            self.has_flow - if this dataset has optic flow
            self.t0 - timestamp of first event
            self.tk - timestamp of last event
            self.num_events - the total number of events
            self.frame_ts - list of the timestamps of the frames
            self.num_frames - the number of frames
        """
        raise NotImplementedError

    def find_ts_index(self, timestamp):
        """
        Given a timestamp, find the event index
        """
        raise NotImplementedError


    def __init__(self, data_path, transforms={}, sensor_resolution=None, num_bins=5,
                 voxel_method=None, max_length=None, combined_voxel_channels=True,
                 filter_hot_events=False):
        """
        self.transform applies to event voxels, frames and flow.
        self.vox_transform applies to event voxels only.
        """

        self.num_bins = num_bins
        self.data_path = data_path
        self.combined_voxel_channels = combined_voxel_channels
        self.sensor_resolution = sensor_resolution
        self.data_source_idx = -1
        self.has_flow = False
        self.channels = self.num_bins if combined_voxel_channels else self.num_bins*2

        self.sensor_resolution, self.t0, self.tk, self.num_events, self.frame_ts, self.num_frames = \
            None, None, None, None, None, None

        self.load_data(data_path)

        if self.sensor_resolution is None or self.has_flow is None or self.t0 is None \
                or self.tk is None or self.num_events is None or self.frame_ts is None \
                or self.num_frames is None:
            raise Exception("Dataloader failed to intialize all required members ({})".format(self.data_path))

        self.num_pixels = self.sensor_resolution[0] * self.sensor_resolution[1]
        self.duration = self.tk - self.t0

        if filter_hot_events:
            secs_for_hot_mask = 0.2
            hot_pix_percent = 0.01
            hot_num = min(self.find_ts_index(secs_for_hot_mask+self.t0), self.num_events)
            xs, ys, ts, ps = self.get_events(0, hot_num)
            self.hot_events_mask = get_hot_event_mask(xs.astype(np.int), ys.astype(np.int), ps, self.sensor_resolution, num_hot=int(self.num_pixels*hot_pix_percent))
            self.hot_events_mask = np.stack([self.hot_events_mask]*self.channels, axis=2).transpose(2,0,1)
        else:
            self.hot_events_mask = np.ones([self.channels, *self.sensor_resolution])
        self.hot_events_mask = torch.from_numpy(self.hot_events_mask).float()

        if voxel_method is None:
            voxel_method = {'method': 'between_frames'}
        self.set_voxel_method(voxel_method)

        if 'LegacyNorm' in transforms.keys() and 'RobustNorm' in transforms.keys():
            raise Exception('Cannot specify both LegacyNorm and RobustNorm')

        self.normalize_voxels = False
        for norm in ['RobustNorm', 'LegacyNorm']:
            if norm in transforms.keys():
                vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]
                del (transforms[norm])
                self.normalize_voxels = True
                self.vox_transform = Compose(vox_transforms_list)
                break

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
        """
        Get data at index.
            :param index: index of data
            :param seed: random seed for data augmentation
        """
        assert 0 <= index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        seed = random.randint(0, 2 ** 32) if seed is None else seed

        idx0, idx1 = self.get_event_indices(index)
        xs, ys, ts, ps = self.get_events(idx0, idx1)
        try:
            ts_0, ts_k  = ts[0], ts[-1]
        except:
            ts_0, ts_k = 0, 0
        if len(xs) < 3:
            voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
        else:
            xs = torch.from_numpy(xs.astype(np.float32))
            ys = torch.from_numpy(ys.astype(np.float32))
            ts = torch.from_numpy((ts-ts_0).astype(np.float32))
            ps = torch.from_numpy(ps.astype(np.float32))
            voxel = self.get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels=self.combined_voxel_channels)

        voxel = self.transform_voxel(voxel, seed).float()
        dt = ts_k - ts_0
        if dt == 0:
            dt = np.array(0.0)

        #print("Get voxel: event_t0={}, event_tk={}, image_ts={}".format(ts_0, ts_k, self.frame_ts[index]))
        if self.voxel_method['method'] == 'between_frames':
            frame = self.get_frame(index)
            frame = self.transform_frame(frame, seed)

            if self.has_flow:
                flow = self.get_flow(index)
                # convert to displacement (pix)
                flow = flow * dt
                flow = self.transform_flow(flow, seed)
            else:
                flow = torch.zeros((2, frame.shape[-2], frame.shape[-1]), dtype=frame.dtype, device=frame.device)

            timestamp = torch.tensor(self.frame_ts[index], dtype=torch.float64)
            item = {'frame': frame,
                    'flow': flow,
                    'events': voxel,
                    'timestamp': timestamp,
                    'data_source_idx': self.data_source_idx,
                    'dt': torch.tensor(dt, dtype=torch.float64)}
        else:
            print("Not between")
            item = {'events': voxel,
                    'timestamp': torch.tensor(ts_k, dtype=torch.float64),
                    'data_source_idx': self.data_source_idx,
                    'dt': torch.tensor(dt, dtype=torch.float64)}
        return item

    def compute_frame_indices(self):
        """
        For each frame, find the start and end indices of the
        time synchronized events
        """
        frame_indices = []
        start_idx = 0
        for ts in self.frame_ts:
            end_index = self.find_ts_index(ts)
            frame_indices.append([start_idx, end_index])
            start_idx = end_index
        return frame_indices

    def compute_timeblock_indices(self):
        """
        For each block of time (using t_events), find the start and
        end indices of the corresponding events
        """
        timeblock_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            start_time = ((self.voxel_method['t'] - self.voxel_method['sliding_window_t']) * i) + self.t0
            end_time = start_time + self.voxel_method['t']
            end_idx = self.find_ts_index(end_time)
            timeblock_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return timeblock_indices

    def compute_k_indices(self):
        """
        For each block of k events, find the start and
        end indices of the corresponding events
        """
        k_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            idx0 = (self.voxel_method['k'] - self.voxel_method['sliding_window_w']) * i
            idx1 = idx0 + self.voxel_method['k']
            k_indices.append([idx0, idx1])
        return k_indices

    def set_voxel_method(self, voxel_method):
        """
        Given the desired method of computing voxels,
        compute the event_indices lookup table and dataset length
        """
        self.voxel_method = voxel_method
        if self.voxel_method['method'] == 'k_events':
            self.length = max(int(self.num_events / (voxel_method['k'] - voxel_method['sliding_window_w'])), 0)
            self.event_indices = self.compute_k_indices()
        elif self.voxel_method['method'] == 't_seconds':
            self.length = max(int(self.duration / (voxel_method['t'] - voxel_method['sliding_window_t'])), 0)
            self.event_indices = self.compute_timeblock_indices()
        elif self.voxel_method['method'] == 'between_frames':
            self.length = self.num_frames - 1
            self.event_indices = self.compute_frame_indices()
        else:
            raise Exception("Invalid voxel forming method chosen ({})".format(self.voxel_method))
        if self.length == 0:
            raise Exception("Current voxel generation parameters lead to sequence length of zero")

    def __len__(self):
        return self.length

    def get_event_indices(self, index):
        """
        Get start and end indices of events at index
        """
        idx0, idx1 = self.event_indices[index]
        if not (idx0 >= 0 and idx1 <= self.num_events):
            raise Exception("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))
        return idx0, idx1

    def get_empty_voxel_grid(self, combined_voxel_channels=True):
        """Return an empty voxel grid filled with zeros"""
        if combined_voxel_channels:
            size = (self.num_bins, *self.sensor_resolution)
        else:
            size = (2*self.num_bins, *self.sensor_resolution)
        return torch.zeros(size, dtype=torch.float32)

    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        :param combined_voxel_channels: if True, create voxel grid merging positive and
            negative events (resulting in NUM_BINS x H x W tensor). Otherwise, create
            voxel grid for positive and negative events separately
            (resulting in 2*NUM_BINS x H x W tensor)
        """
        if combined_voxel_channels:
            # generate voxel grid which has size self.num_bins x H x W
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_resolution)
        else:
            # generate voxel grid which has size 2*self.num_bins x H x W
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.num_bins,
                                                       sensor_size=self.sensor_resolution)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)

        voxel_grid = voxel_grid*self.hot_events_mask 

        return voxel_grid

    def transform_frame(self, frame, seed):
        """
        Augment frame and turn into tensor
        """
        frame = torch.from_numpy(frame).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)
        return frame

    def transform_voxel(self, voxel, seed):
        """
        Augment voxel and turn into tensor
        """
        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel

    def transform_flow(self, flow, seed):
        """
        Augment flow and turn into tensor
        """
        flow = torch.from_numpy(flow)  # should end up [2 x H x W]
        if self.transform:
            random.seed(seed)
            flow = self.transform(flow, is_flow=True)
        return flow


class DynamicH5Dataset(BaseVoxelDataset):
    """
    Dataloader for events saved in the Monash University HDF5 events format
    (see https://github.com/TimoStoff/event_utils for code to convert datasets)
    """

    def get_frame(self, index):
        return self.h5_file['images']['image{:09d}'.format(index)][:]

    def get_flow(self, index):
        return self.h5_file['flow']['flow{:09d}'.format(index)][:]

    def get_events(self, idx0, idx1):
        xs = self.h5_file['events/xs'][idx0:idx1]
        ys = self.h5_file['events/ys'][idx0:idx1]
        ts = self.h5_file['events/ts'][idx0:idx1]
        ps = self.h5_file['events/ps'][idx0:idx1] * 2.0 - 1.0
        return xs, ys, ts, ps

    def load_data(self, data_path):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
        else:
            self.sensor_resolution = self.sensor_resolution[0:2]
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
        idx = binary_search_h5_dset(self.h5_file['events/ts'], timestamp)
        return idx

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for img_name in self.h5_file['images']:
            end_idx = self.h5_file['images/{}'.format(img_name)].attrs['event_idx']
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class MemMapDataset(BaseVoxelDataset):
    """
    Dataloader for events saved in the MemMap events format used at RPG.
    (see https://github.com/TimoStoff/event_utils for code to convert datasets)
    """

    def get_frame(self, index):
        frame = self.filehandle['images'][index][:, :, 0]
        return frame

    def get_flow(self, index):
        flow = self.filehandle['optic_flow'][index]
        return flow

    def get_events(self, idx0, idx1):
        xy = self.filehandle["xy"][idx0:idx1]
        xs = xy[:, 0].astype(np.float32)
        ys = xy[:, 1].astype(np.float32)
        ts = self.filehandle["t"][idx0:idx1]
        ps = self.filehandle["p"][idx0:idx1] * 2.0 - 1.0
        return xs, ys, ts, ps

    def load_data(self, data_path, timestamp_fname="timestamps.npy", image_fname="images.npy",
                  optic_flow_fname="optic_flow.npy", optic_flow_stamps_fname="optic_flow_stamps.npy",
                  t_fname="t.npy", xy_fname="xy.npy", p_fname="p.npy"):

        assert os.path.isdir(data_path), '%s is not a valid data_pathectory' % data_path

        data = {}
        self.has_flow = False
        for subroot, _, fnames in sorted(os.walk(data_path)):
            for fname in sorted(fnames):
                path = os.path.join(subroot, fname)
                if fname.endswith(".npy"):
                    if fname.endswith(timestamp_fname):
                        frame_stamps = np.load(path)
                        data["frame_stamps"] = frame_stamps
                    elif fname.endswith(image_fname):
                        data["images"] = np.load(path, mmap_mode="r")
                    elif fname.endswith(optic_flow_fname):
                        data["optic_flow"] = np.load(path, mmap_mode="r")
                        self.has_flow = True
                    elif fname.endswith(optic_flow_stamps_fname):
                        optic_flow_stamps = np.load(path)
                        data["optic_flow_stamps"] = optic_flow_stamps

                    try:
                        handle = np.load(path, mmap_mode="r")
                    except Exception as err:
                        print("Couldn't load {}:".format(path))
                        raise err
                    if fname.endswith(t_fname):  # timestamps
                        data["t"] = handle.squeeze()
                    elif fname.endswith(xy_fname):  # coordinates
                        data["xy"] = handle.squeeze()
                    elif fname.endswith(p_fname):  # polarity
                        data["p"] = handle.squeeze()
            if len(data) > 0:
                data['path'] = subroot
                if "t" not in data:
                    print("Ignoring root {} since no events".format(subroot))
                    continue
                assert (len(data['p']) == len(data['xy']) and len(data['p']) == len(data['t']))

                self.t0, self.tk = data['t'][0], data['t'][-1]
                self.num_events = len(data['p'])
                self.num_frames = len(data['images'])

                self.frame_ts = []
                for ts in data["frame_stamps"]:
                    self.frame_ts.append(ts)
                data["index"] = self.frame_ts

        self.filehandle = data
        self.find_config(data_path)

    def find_ts_index(self, timestamp):
        index = np.searchsorted(self.filehandle["t"], timestamp)
        return index

    def infer_resolution(self):
        if len(self.filehandle["images"]) > 0:
            return self.filehandle["images"][0].shape[:2]
        else:
            print('WARNING: sensor resolution inferred from maximum event coordinates - highly not recommended')
            return [np.max(self.filehandle["xy"][:, 1]) + 1, np.max(self.filehandle["xy"][:, 0]) + 1]

    def find_config(self, data_path):
        if self.sensor_resolution is None:
            config = os.path.join(data_path, "dataset_config.json")
            if os.path.exists(config):
                self.config = read_json(config)
                self.data_source = self.config['data_source']
                self.sensor_resolution = self.config["sensor_resolution"]
            else:
                data_source = 'unknown'
                self.sensor_resolution = self.infer_resolution()
                print("Inferred sensor resolution: {}".format(self.sensor_resolution))


class SequenceDataset(Dataset):
    """Load sequences of time-synchronized {event tensors + frames} from a folder."""
    def __init__(self, data_root, sequence_length, dataset_type='MemMapDataset',
            step_size=None, proba_pause_when_running=0.0,
            proba_pause_when_paused=0.0, normalize_image=False,
            noise_kwargs={}, hot_pixel_kwargs={}, dataset_kwargs={}):
        self.L = sequence_length
        self.step_size = step_size if step_size is not None else self.L
        self.proba_pause_when_running = proba_pause_when_running
        self.proba_pause_when_paused = proba_pause_when_paused
        self.normalize_image = normalize_image
        self.noise_kwargs = noise_kwargs
        self.hot_pixel_kwargs = hot_pixel_kwargs

        assert(self.L > 0)
        assert(self.step_size > 0)

        self.dataset = eval(dataset_type)(data_root, **dataset_kwargs)
        if self.L >= self.dataset.length:
            self.length = 0
        else:
            self.length = (self.dataset.length - self.L) // self.step_size + 1

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        """ Returns a list containing synchronized events <-> frame pairs
            [e_{i-L} <-> I_{i-L},
             e_{i-L+1} <-> I_{i-L+1},
            ...,
            e_{i-1} <-> I_{i-1},
            e_i <-> I_i]
        """
        assert(i >= 0)
        assert(i < self.length)

        # generate a random seed here, that we will pass to the transform function
        # of each item, to make sure all the items in the sequence are transformed
        # in the same way
        seed = random.randint(0, 2**32)

        # data augmentation: add random, virtual "pauses",
        # i.e. zero out random event tensors and repeat the last frame
        sequence = []

        # add the first element (i.e. do not start with a pause)
        k = 0
        j = i * self.step_size
        item = self.dataset.__getitem__(j, seed)
        sequence.append(item)

        paused = False
        for n in range(self.L - 1):

            # decide whether we should make a "pause" at this step
            # the probability of "pause" is conditioned on the previous state (to encourage long sequences)
            u = np.random.rand()
            if paused:
                probability_pause = self.proba_pause_when_paused
            else:
                probability_pause = self.proba_pause_when_running
            paused = (u < probability_pause)

            if paused:
                # add a tensor filled with zeros, paired with the last frame
                # do not increase the counter
                item = self.dataset.__getitem__(j + k, seed)
                item['events'].fill_(0.0)
                if 'flow' in item:
                    item['flow'].fill_(0.0)
                sequence.append(item)
            else:
                # normal case: append the next item to the list
                k += 1
                item = self.dataset.__getitem__(j + k, seed)
                sequence.append(item)
            # add noise
            if self.noise_kwargs:
                item['events'] = add_noise_to_voxel(item['events'], **self.noise_kwargs)

        # add hot pixels
        if self.hot_pixel_kwargs:
            add_hot_pixels_to_sequence_(sequence, **self.hot_pixel_kwargs)

        # normalize image
        if self.normalize_image:
            normalize_image_sequence_(sequence, key='frame')
        return sequence

