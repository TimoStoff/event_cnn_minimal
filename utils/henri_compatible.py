from utils.default_config import default_config
import copy
from parse_config import ConfigParser


def make_henri_compatible(checkpoint, final_activation=''):
    """
    Checkpoints have ConfigParser type configs, whereas Henri checkpoints have
    dictionary type configs or "arch, model" dicts.
    We will generate and add a ConfigParser to the checkpoint and return it.
    """
    assert ('config' in checkpoint or ('arch' in checkpoint and 'model' in checkpoint))
    check_config = checkpoint['config'] if 'config' in checkpoint else checkpoint
    new_config = copy.deepcopy(default_config)
    new_config['arch']['type'] = check_config['arch']
    new_config['arch']['args']['unet_kwargs'] = check_config['model']
    if final_activation:
        new_config['arch']['args']['unet_kwargs']['final_activation'] = final_activation
    config = ConfigParser(new_config)
    checkpoint['config'] = config
    print(new_config)
    return checkpoint
