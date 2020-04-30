import argparse
import torch
import numpy as np
from os.path import join

from .model import model as model_arch
from .utils.parse_config import ConfigParser


def load_model(args, checkpoint):
    checkpoint_path = checkpoint is not None
    print(checkpoint)
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']

    # build model architecture
    model = config.init_obj('arch', model_arch)
    model.load_state_dict(state_dict)
    print(model)

    return model

def save_model(model, savepath):
    script = torch.jit.script(model)
    print("save to {}".format(savepath))
    torch.jit.save(script, savepath)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--checkpoint_path', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-s', '--savepath', default="/tmp/save_model.pth", type=str,
                      help='where to save model to')

    args = parser.parse_args()
    
    kwargs = {}
    if args.checkpoint_path is not None:
        print('Loading checkpoint: {} ...'.format(args.checkpoint_path))
        checkpoint = torch.load(args.checkpoint_path)
        kwargs['checkpoint'] = checkpoint
    elif args.config is not None:
        print('Using config: {} '.format(args.config))
        kwargs['config'] = ConfigParser.from_args(parser)

    model = load_model(args, **kwargs)
    #save_model(model, args.savepath)
