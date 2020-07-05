import torch
import torch.nn as nn
import torch.nn.functional as f
from os.path import join
import numpy as np
# local modules
from model.submodules import ConvLSTM, ResidualBlock, ConvLayer, \
        UpsampleConvLayer, TransposedConvLayer, RecurrentConvLayer, RecurrentResidualLayer


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


def identity(x1, x2=None):
    return x1


class BaseUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=True,
                 kernel_size=5):
        super(BaseUNet, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        if self.skip_type == 'sum':
            self.apply_skip_connection = skip_sum
        elif self.skip_type == 'concat':
            self.apply_skip_connection = skip_concat
        elif self.skip_type == 'no_skip' or self.skip_type is None:
            self.apply_skip_connection = identity
        else:
            raise KeyError('Could not identify skip_type, please add "skip_type":'
                           ' "sum", "concat" or "no_skip" to config["model"]')
        self.activation = identity if activation is None or activation == 'identity' else getattr(torch, activation)
        self.norm = norm
        self.kernel_size = kernel_size

        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert(self.num_input_channels > 0)
        assert(self.num_output_channels > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        self.activation = identity if activation is None or activation == 'identity' else getattr(torch, activation)

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm)



class UNetFire(BaseUNet):
    """
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convgru', base_num_channels=32,
                 num_residual_blocks=2, norm=None, kernel_size=3,
                 recurrent_blocks={'resblock': [0]}, BN_momentum=0.1):
        super(UNetFire, self).__init__(num_input_channels=num_input_channels,
                                       num_output_channels=num_output_channels,
                                       skip_type=skip_type,
                                       base_num_channels=base_num_channels,
                                       num_residual_blocks=num_residual_blocks,
                                       norm=norm,
                                       kernel_size=kernel_size)

        self.recurrent_blocks = recurrent_blocks
        self.num_recurrent_units = 0
        self.head = RecurrentConvLayer(self.num_input_channels,
                                       self.base_num_channels,
                                       kernel_size=self.kernel_size,
                                       padding=self.kernel_size // 2,
                                       recurrent_block_type=recurrent_block_type,
                                       norm=self.norm,
                                       BN_momentum=BN_momentum)
        self.num_recurrent_units += 1
        self.resblocks = nn.ModuleList()
        recurrent_indices = self.recurrent_blocks.get('resblock', [])
        for i in range(self.num_residual_blocks):
            if i in recurrent_indices or -1 in recurrent_indices:
                self.resblocks.append(RecurrentResidualLayer(
                    in_channels=self.base_num_channels,
                    out_channels=self.base_num_channels,
                    recurrent_block_type=recurrent_block_type,
                    norm=self.norm,
                    BN_momentum=BN_momentum))
                self.num_recurrent_units += 1
            else:
                self.resblocks.append(ResidualBlock(self.base_num_channels,
                                                    self.base_num_channels,
                                                    norm=self.norm,
                                                    BN_momentum=BN_momentum))

        self.pred = ConvLayer(2 * self.base_num_channels if self.skip_type == 'concat' else self.base_num_channels,
                              self.num_output_channels, kernel_size=1, padding=0, activation=None, norm=None)
        self.pred.conv2d.bias.data.fill_(0.5)

    def forward(self, x, prev_states):
        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """

        if prev_states is None:
            prev_states = [None] * (self.num_recurrent_units)

        states = []
        state_idx = 0

        # head
        x, state = self.head(x, prev_states[state_idx])
        state_idx += 1
        states.append(state)

        head_feature_map = x

        # residual blocks
        recurrent_indices = self.recurrent_blocks.get('resblock', [])
        for i, resblock in enumerate(self.resblocks):
            if i in recurrent_indices or -1 in recurrent_indices:
                x, state = resblock(x, prev_states[state_idx])
                state_idx += 1
                states.append(state)
            else:
                x = resblock(x)

        # tail
        img = self.pred(self.apply_skip_connection(x, head_feature_map))
        return img, states


class BaseE2VID(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert('num_bins' in config)
        self.num_bins = int(config['num_bins'])  # number of bins in the voxel grid event tensor

        try:
            self.skip_type = str(config['skip_type'])
        except KeyError:
            self.skip_type = 'sum'

        try:
            self.num_encoders = int(config['num_encoders'])
        except KeyError:
            self.num_encoders = 4

        try:
            self.base_num_channels = int(config['base_num_channels'])
        except KeyError:
            self.base_num_channels = 32

        try:
            self.num_residual_blocks = int(config['num_residual_blocks'])
        except KeyError:
            self.num_residual_blocks = 2

        try:
            self.norm = str(config['norm'])
        except KeyError:
            self.norm = None

        try:
            self.use_upsample_conv = bool(config['use_upsample_conv'])
        except KeyError:
            self.use_upsample_conv = True

        self.kernel_size = int(config.get('kernel_size', 5))


class FireNet_legacy(BaseE2VID):
    """
    Model from the paper: "Fast Image Reconstruction with an Event Camera", Scheerlinck et. al., 2019.
    The model is essentially a lighter version of E2VID, which runs faster (~2-3x faster) and has considerably less parameters (~200x less).
    However, the reconstructions are not as high quality as E2VID: they suffer from smearing artefacts, and initialization takes longer.
    """
    def __init__(self, config={}, unet_kwargs={}):
        if unet_kwargs:
            config = unet_kwargs
        super().__init__(config)
        self.recurrent_block_type = str(config.get('recurrent_block_type', 'convgru'))
        recurrent_blocks = config.get('recurrent_blocks', {'resblock': [0]})
        BN_momentum = config.get('BN_momentum', 0.1)
        self.net = UNetFire(self.num_bins,
                            num_output_channels=1,
                            skip_type=self.skip_type,
                            recurrent_block_type=self.recurrent_block_type,
                            base_num_channels=self.base_num_channels,
                            num_residual_blocks=self.num_residual_blocks,
                            norm=self.norm,
                            kernel_size=self.kernel_size,
                            recurrent_blocks=recurrent_blocks,
                            BN_momentum=BN_momentum)
        self.num_recurrent_units = self.net.num_recurrent_units
        self.reset_states()

    def reset_states(self):
        self.states = [None] * self.num_recurrent_units

    def forward(self, event_tensor):
        img, self.states = self.net.forward(event_tensor, self.states)
        return {'image': img}
