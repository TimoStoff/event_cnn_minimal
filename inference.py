import argparse
import torch
import numpy as np
from os.path import join
import os
import cv2
from thop import profile
from tqdm import tqdm

from utils.util import ensure_dir, flow2bgr_np
from model import model as model_arch
from utils.parse_config import ConfigParser
from data_loader.data_loaders import InferenceDataLoader
from utils.util import CropParameters
from utils.timers import CudaTimer

model_info = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(args, checkpoint):
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']

    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    logger.info(model)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    print(model)
    for param in model.parameters():
        param.requires_grad = False
    return model


def main(args, model):

    dataset_kwargs = {'transforms': {},
                      'max_length': None,
                      'sensor_size': None,
                      'num_bins': 5,
                      'voxel_method': {'method':'between_frames'}
                      }
    if not args.legacy:
        dataset_kwargs['transforms'] = {'RobustNorm':{}}
        dataset_kwargs['combined_voxel_channels'] = False

    data_loader = InferenceDataLoader(args.h5_file_path, dataset_kwargs=dataset_kwargs)

    for d in data_loader:
        height, width = d['events'].shape[-2:]
        break
    model_info['input_shape'] = height, width
    crop = CropParameters(width, height, model.num_encoders)

    # count FLOPs
    tmp_voxel = crop.pad(torch.randn(1, model_info['num_bins'], height, width).to(device))
    model_info['FLOPs'], model_info['Params'] = profile(model, inputs=(tmp_voxel, ))

    ensure_dir(args.output_folder)
    print('Saving to: {}'.format(args.output_folder))
    with open(join(args.output_folder, 'timestamps.txt'), 'w+') as ts_file:
        model.reset_states()
        for i, item in enumerate(tqdm(data_loader)):
            voxel = item['events'].to(device)
            voxel = crop.pad(voxel)
            with CudaTimer('Inference'):
                output = model(voxel)
            # save sample images, or do something with output here
            if args.is_flow:
                flow_t = torch.squeeze(crop.crop(output['flow']))
                flow = flow_t.cpu().numpy()
                ts = item['timestamp'].cpu().numpy()
                flow_dict = {'flow':flow, 'ts':ts}
                fname = 'flow_{:010d}.npy'.format(i)
                np.save(os.path.join(args.output_folder, fname), flow_dict)
                with open(os.path.join(args.output_folder, fname), "a") as myfile:
                    myfile.write("\n")
                    myfile.write("timestamp: {:.10f}".format(ts[0]))
                flow_img = flow2bgr_np(flow[0,:,:], flow[1,:,:])
                fname = 'flow_{:010d}.png'.format(i)
                cv2.imwrite(os.path.join(args.output_folder, fname), flow_img)
            else:
                image = crop.crop(output['image'])
                image = torch.squeeze(image)  # H x W
                image = image.cpu().numpy()  # normalize here
                image = np.clip(image, 0, 1)  # normalize here
                image = (image * 255).astype(np.uint8)
                fname = 'frame_{:010d}.png'.format(i)
                cv2.imwrite(join(args.output_folder, fname), image)
            ts_file.write('{} {:.15f}\n'.format(fname, item['timestamp'].item()))


def print_model_info():
    print('Input shape: {} x {} x {}'.format(model_info.pop('num_bins'), *model_info.pop('input_shape')))
    print('== Model statistics ==')
    for k, v in model_info.items():
        print('{}: {:.2f} {}'.format(k, *format_power(v)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--checkpoint_path', required=True, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('--h5_file_path', required=True, type=str,
                      help='path to hdf5 events')
    parser.add_argument('--output_folder', default="/tmp/output", type=str,
                      help='where to save outputs to')
    parser.add_argument('--device', default='0', type=str,
                      help='indices of GPUs to enable')
    parser.add_argument('--is_flow', action='store_true',
            help='If true, save output to flow npy file')
    parser.add_argument('--legacy', action='store_true',
            help='Set this if using any of the original networks from ECCV20 paper')

    args = parser.parse_args()
    
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    kwargs = {}
    print('Loading checkpoint: {} ...'.format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path)
    kwargs['checkpoint'] = checkpoint

    #import h5py
    #dataset_kwargs = {'transforms': {},
    #                  'max_length': None,
    #                  'sensor_size': None,
    #                  'num_bins': 5,
    #                  'legacy': True,
    #                  'voxel_method': {'method':'between_frames'}
    #                  }
    #h5_path = "/home/timo/Data2/preprocessed_datasets/h5_voxels/slider_depth_cut.h5"
    #data_loader = InferenceDataLoader(args.h5_file_path, dataset_kwargs=dataset_kwargs)
    #h5_file = h5py.File(h5_path, 'r')
    #for i, item in enumerate(data_loader):
    #    data_name = "frame_{:09d}".format(i)
    #    dset = h5_file[data_name]
    #    voxel = np.stack([bin[:] for bin in dset['voxels'].values()], axis=0)  # C x H x W
    #    new_voxel = item['events']

    #    if True:
    #        print(np.sum(voxel))
    #        print(torch.sum(new_voxel))
    #        #print(voxel)
    #        #print(new_voxel)
    model = load_model(args, **kwargs)
    main(args, model)
