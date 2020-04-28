default_config = {
    'name': 'inference',
    'n_gpu': 1,
    'arch': {
        'args': {}
    },
    'valid_data_loader': {
        'type': 'HDF5DataLoader',
        'args': {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 1,
            'pin_memory': True,
            'sequence_kwargs': {
                'dataset_type': 'HDF5Dataset',
                'normalize_image': True,
                'dataset_kwargs': {
                    'transforms': {
                        'CenterCrop': {
                            'size': 160
                        }
                    }
                }
            }
        }
    },
    'optimizer': {
        'type': 'Adam',
        'args': {
            'lr': 0.0001,
            'weight_decay': 0,
            'amsgrad': True
        }
    },
    'loss_ftns': {
        'perceptual_loss': {
            'weight': 1.0
        },
        'temporal_consistency_loss': {
            'weight': 1.0
        }
    },
    'lr_scheduler': {
        'type': 'StepLR',
        'args': {
            'step_size': 50,
            'gamma': 1.0
        }
    },
    'trainer': {
        'epochs': 1,
        'save_dir': '/tmp/inference',
        'save_period': 1,
        'verbosity': 2,
        'monitor': 'min val_loss',
        'num_previews': 4,
        'val_num_previews': 8,
        'tensorboard': True
    }
}

