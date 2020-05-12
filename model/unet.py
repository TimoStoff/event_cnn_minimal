import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from .submodules import \
    ConvLayer, UpsampleConvLayer, TransposedConvLayer, \
    RecurrentConvLayer, ResidualBlock, ConvLSTM, \
    ConvGRU, RecurrentResidualLayer

from .model_util import *


class BaseUNet(nn.Module):
    def __init__(self, base_num_channels, num_encoders, num_residual_blocks,
                 num_output_channels, skip_type, norm, use_upsample_conv,
                 num_bins, recurrent_block_type=None, kernel_size=5,
                 channel_multiplier=2):
        super(BaseUNet, self).__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type

        self.encoder_input_sizes = [int(self.base_num_channels * pow(channel_multiplier, i)) for i in range(self.num_encoders)]
        self.encoder_output_sizes = [int(self.base_num_channels * pow(channel_multiplier, i + 1)) for i in range(self.num_encoders)]
        self.max_num_channels = self.encoder_output_sizes[-1]
        self.skip_ftn = eval('skip_' + skip_type)
        print('Using skip: {}'.format(self.skip_ftn))
        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer
        assert(self.num_output_channels > 0)
        print(f'Kernel size {self.kernel_size}')
        print(f'Skip type {self.skip_type}')
        print(f'norm {self.norm}')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(self.UpsampleLayer(
                input_size if self.skip_type == 'sum' else 2 * input_size,
                output_size, kernel_size=self.kernel_size,
                padding=self.kernel_size // 2, norm=self.norm))
        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                         num_output_channels, 1, activation=None, norm=norm)


class WNet(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    One decoder for flow and one for image.
    """

    def __init__(self, unet_kwargs):
        unet_kwargs['num_output_channels'] = 3
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()
        self.image_decoders = self.build_decoders()
        self.flow_decoders = self.build_decoders()
        self.image_pred = self.build_prediction_layer(num_output_channels=1)
        self.flow_pred = self.build_prediction_layer(num_output_channels=2)
        self.states = [None] * self.num_encoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        flow_activations = x
        for i, decoder in enumerate(self.flow_decoders):
            flow_activations = decoder(self.skip_ftn(flow_activations, blocks[self.num_encoders - i - 1]))
        image_activations = x
        for i, decoder in enumerate(self.image_decoders):
            image_activations = decoder(self.skip_ftn(image_activations, blocks[self.num_encoders - i - 1]))

        # tail
        flow = self.flow_pred(self.skip_ftn(flow_activations, head))
        image = self.image_pred(self.skip_ftn(image_activations, head))

        output_dict = {'image': image, 'flow': flow}

        return output_dict


class UNetFlow(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, unet_kwargs):
        unet_kwargs['num_output_channels'] = 3
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(num_output_channels=3)
        self.states = [None] * self.num_encoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img_flow = self.pred(self.skip_ftn(x, head))

        output_dict = {'image': img_flow[:, 0:1, :, :], 'flow': img_flow[:, 1:3, :, :]}

        return output_dict


class UNetFlowNoRecur(BaseUNet):
    """
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, unet_kwargs):
        unet_kwargs['num_output_channels'] = 3
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(ConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(num_output_channels=3)

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img_flow = self.pred(self.skip_ftn(x, head))

        output_dict = {'image': img_flow[:, 0:1, :, :], 'flow': img_flow[:, 1:3, :, :]}

        return output_dict


class UNetRecurrent(BaseUNet):
    """
    Compatible with E2VID_lightweight
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """
    def __init__(self, unet_kwargs):
        final_activation = unet_kwargs.pop('final_activation', 'none')
        self.final_activation = getattr(torch, final_activation, None)
        print(f'Using {self.final_activation} final activation')
        unet_kwargs['num_output_channels'] = 1
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.states = [None] * self.num_encoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.pred(self.skip_ftn(x, head))
        if self.final_activation is not None:
            img = self.final_activation(img)
        return {'image': img}


class UNet(BaseUNet):
    """
    UNet architecture. Symmetric, skip connections on every encoding layer.
    """
    def __init__(self, unet_kwargs):
        super().__init__(**unet_kwargs)
        self.encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins  # since there is no self.head!
            self.encoders.append(ConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = ConvLayer(self.base_num_channels, self.num_output_channels, kernel_size=1, activation=None)

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        return self.pred(x)
