import os.path
import numpy as np
import torch.utils.data as data
from utils.data import data_sources
import torch
import random
import events_contrast_maximization.utils.event_utils as er
from utils import read_json, write_json
from utils.data_augmentation import add_noise_to_voxel, \
        add_hot_pixels_to_sequence_, normalize_image_sequence_, \
        RandomCrop, CenterCrop, RandomFlip, RobustNorm, Compose


def compute_indices(event_stamps, frame_stamps, event_window_ms=0, event_window_nr=0):
    # computes the event indices
    indices_first = np.searchsorted(event_stamps[:,0], frame_stamps[1:])

    if event_window_nr > 0:
        indices_last = indices_first - event_window_nr
    elif event_window_ms > 0:
        indices_last = np.searchsorted(event_stamps[:,0], frame_stamps[1:] - event_window_ms * 1e-3)
    else:
        indices_last = np.searchsorted(event_stamps[:,0], frame_stamps[:-1])

    index = np.stack([indices_first, indices_last], -1)

    return index


class MemMapDataset(data.Dataset):

    def __init__(self, 
                 root,
                 transforms = {},
                 num_temporal_bins = 5,
                 event_window_ms=0,
                 event_window_nr=0,
                 max_length=None,
                 generation_mode='between_frames',
                 generation_params={},
                 config = None):

        if isinstance(config, dict):
            self.config = config
            data_source = 'unknown'
        else:
            if config is None:
                config = os.path.join(root, "dataset_config.json")
            assert(os.path.exists(config))
            self.config = read_json(config)
            data_source = self.config['data_source']
        assert "sensor_resolution" in self.config

        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

        self.sensor_resolution = self.config["sensor_resolution"]
        self.num_pixels = self.sensor_resolution[0]*self.sensor_resolution[1]
        self.num_temporal_bins = num_temporal_bins

        self.event_window_ms = event_window_ms
        self.event_window_nr = event_window_nr
        self.set_generation_mode(generation_mode, generation_params)

        self.filehandle = self.load_files(root,
                               event_window_ms,
                               event_window_nr)

        self.root = root


        self.normalize_voxels = False
        if 'RobustNorm' in transforms.keys():
            vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]
            del(transforms['RobustNorm'])
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

        self.timestamps = None
        self.length = self.__len__()
        if max_length is not None:
            self.length = min(self.length, max_length + 1)

    def set_generation_mode(self, generation_mode, generation_params):
        self.generation_mode = generation_mode 
        self.generation_params = generation_params
        if generation_mode is 'between_frames':
            pass
        elif generation_mode is 'k_events':
            self.events_per_pixel = generation_params['events_per_pixel']
            self.events_per_voxel_grid = self.num_pixels*self.events_per_pixel 
            self.overlap_perc = generation_params['overlap_perc'] 
        elif generation_mode is 't_seconds':
            self.duration = generation_params['duration'] 
            if self.duration is None or self.duration is '':
                self.duration = self.period
            self.period = 1.0/generation_params['frequency'] 
            pass
        else:
            self.generation_mode = 'between_frames'
            self.generation_params = {}

    def load_files(self, rootdir, event_window_ms, event_window_nr):
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
                    elif fname.endswith("xy.npy"): # coordinates
                        data["xy"] = handle
                    elif fname.endswith("p.npy"): # polarity
                        data["p"] = handle

            if len(data) > 0:
                data['path'] = subroot

                if "t" not in data:
                    print(f"Ignoring rootdirectory {subroot} since no events")
                    continue
                assert(len(data['p']) == len(data['xy']) and len(data['p']) == len(data['t']))
                data["num_events"] = len(data['p'])

                # if index is not in dict compute them from frame_stamps
                if "index" not in data and "frame_stamps" in data:
                    data["index"] = compute_indices(data["t"], data['frame_stamps'], self.event_window_ms)
                if "length" not in data:
                    if self.generation_mode is 'between_frames':
                        data["length"] = len(data["index"])
                    elif self.generation_mode is 'k_events':
                        data["length"] = int((len(data['p'])-2*self.events_per_voxel_grid)//(self.events_per_voxel_grid*(1.0-self.overlap_perc)))
                    elif self.generation_mode is 't_seconds':
                        t0, tk = data['t'][0][0], data['t'][-1][0]
                        seq_dur = tk-t0
                        data["length"] = int((seq_dur+self.period-self.duration)/self.period)
                        starting_indices = []
                        end_indices = []
                        for i in range(data["length"]):
                            t_start = t0 + self.period*i
                            starting_indices.append(np.searchsorted(np.squeeze(data['t']), t_start))
                            end_indices.append(np.searchsorted(np.squeeze(data['t']), t_start+self.duration)-1)
                        data["start_indices"] = starting_indices
                        data["end_indices"] = end_indices
                    #print("{}: Enough for {} unique voxel grids".format(subroot, data["length"]))
        return data

    #    * frame is a H x W tensor containing the first frame whose
    #      timestamp >= event tensor
    #    * events is a C x H x W tensor containing the event data
    #    * flow is a 2 x H x W tensor containing the flow (displacement) from
    #      the current frame to the last frame
    def __getitem__(self, index, seed=None):
        assert index >= 0 and index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        voxel = self.get_voxel_grid(self.filehandle, index)
        frame = self.filehandle['images'][index+1][:,:,0]

        if seed is None:
            # if no specific random seed was passed, generate our own.
            seed = random.randint(0, 2**32)

        voxel = self.transform_voxel(voxel, seed)
        frame = self.transform_frame(frame, seed)

        if self.has_flow:
            flow = self.filehandle['optic_flow'][index+1]
            dt = self.filehandle['frame_stamps'][index+1]-self.filehandle['frame_stamps'][index]
            # convert to displacement (pix)
            flow = flow*dt
            flow = self.transform_flow(flow, seed)
        else:
            flow = torch.zeros((2, frame.shape[-2], frame.shape[-1]), dtype=frame.dtype, device=frame.device)

        item = {'frame': frame,
                'flow': flow,
                'events': voxel,
                'timestamp': self.filehandle["frame_stamps"][1+index],
#                'data_source': [self.filehandle['path']],
                'data_source_idx': self.data_source_idx}
        return item

    def get_voxel_grid(self, filehandle, index):
        if self.generation_mode is 'between_frames':
            indices = filehandle["index"]
            idx1, idx0 = indices[index]
        elif self.generation_mode is 'k_events':
            idx0 = int((1.0-self.overlap_perc)*self.events_per_voxel_grid*index)
            idx1 = int(idx0 + self.events_per_voxel_grid)
        elif self.generation_mode is 't_seconds':
            idx0 = filehandle["start_indices"][index]
            idx1 = filehandle["end_indices"][index]
        if not (idx0>=0 and idx1<=filehandle["num_events"]): 
            print("WARNING: Either {}<0s or {}>{}".format(idx0, idx1, filehandle["num_events"]))
        assert(idx0>=0 and idx1<=filehandle["num_events"])
        # select events with indices between current frame at index and next frame at index+1

        events_xy = filehandle["xy"][idx0:idx1]
        events_t = filehandle["t"][idx0:idx1]-filehandle["t"][idx0]
        events_p = filehandle["p"][idx0:idx1]

        # generate voxel grid which has size C x H x W
        H, W = self.sensor_resolution
        channels = self.num_temporal_bins

        if len(events_xy) < 2:
            return np.zeros((2*channels, H, W) )

        voxel_grid = er.events_to_neg_pos_voxel(events_xy[:, 0], events_xy[:, 1],
                events_t, events_p, channels, (H,W))
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

    def __len__(self):
        return self.filehandle["length"]
