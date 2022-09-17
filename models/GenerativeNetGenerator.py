import torch
import torch.nn as nn

"""This class will generate the discriminator networks as
proposed in the original paper. While doing so it tries to stay as flexible as possible
to allow for improvements later on.

-> the implementation of the paper can be found under: 
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/eb6ae80412e23c09b4317b04d889f1af27526d2d/models/networks.py#L542

"""


class GenerativeNetGenerator:
    """
           The generative networks are created in accordance with the notes in the original paper.
    """

    def __init__(self, is_nine_block_model):
        self.is_nine_block_model = is_nine_block_model

    def create_generator(self):
        return GenerativeModel(self.is_nine_block_model)


class GenerativeModel(nn.Module):
    def __init__(self, is_nine_block_model):
        super(GenerativeModel, self).__init__()
        self.is_nine_block_model = is_nine_block_model
        model = [nn.ReflectionPad2d(3),
                 ConvInstNormReluLayer(3, 64, 7, 1, 0),
                 ConvInstNormReluLayer(64, 128, 3, 2, 1),
                 ConvInstNormReluLayer(128, 256, 3, 2, 1),
                 ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False),
                 ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False),
                 ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False),
                 ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False),
                 ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False),
                 ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False)]

        if self.is_nine_block_model:
            model += [ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False),
                      ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False),
                      ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False)]

        model += [UppsamplingLayer(256, 128, 3, 2, 1, 1),
                  UppsamplingLayer(128, 64, 3, 2, 1, 1)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(64, 3, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ConvInstNormReluLayer(nn.Module):
    def __init__(self, size_in, size_out, kernel_size, stride, padding):
        super(ConvInstNormReluLayer, self).__init__()
        self.size_in = size_in
        self.size_out = size_out

        self.convolution = nn.Conv2d(size_in, size_out, kernel_size, stride, padding, padding_mode='reflect')

        self.inst_norm = nn.BatchNorm2d(size_out, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(True)

    def forward(self, x):
        x = self.convolution(x)
        x = self.inst_norm(x)
        return self.activation(x)


class UppsamplingLayer(nn.Module):
    def __init__(self, size_in, size_out, kernel_size, stride, padding, output_padding):
        super(UppsamplingLayer, self).__init__()
        self.size_in = size_in
        self.size_out = size_out

        self.convolution = nn.ConvTranspose2d(size_in, size_out, kernel_size, stride, padding,
                                              output_padding=output_padding)
        self.inst_norm = nn.BatchNorm2d(size_out, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(True)

    def forward(self, x):
        x = self.convolution(x)
        x = self.inst_norm(x)
        return self.activation(x)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
