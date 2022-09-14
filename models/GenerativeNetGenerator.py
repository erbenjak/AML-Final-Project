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

class GenerativeModel(nn.Module):

    def __init__(self, is_six_block_model):
        self.is_six_block_model = is_six_block_model

        if is_six_block_model:
            self.firstLayer = ConvInstNormReluLayer(1, 64, 7, 1)
            self.secondLayer = ConvInstNormReluLayer(64, 128, 3, 2)
            self.thirdLayer = ConvInstNormReluLayer(128, 256, 3, 2)
            self.fourthLayer = ResnetBlock(256,'reflect', nn.BatchNorm2d, False, False)
            self.fifthLayer = ResnetBlock(256,'reflect', nn.BatchNorm2d, False, False)
            self.sixthLayer = ResnetBlock(256,'reflect', nn.BatchNorm2d, False, False)
            self.seventhLayer = ResnetBlock(256,'reflect', nn.BatchNorm2d, False, False)
            self.eightLayer = ResnetBlock(256,'reflect', nn.BatchNorm2d, False, False)
            self.ninthLayer = ResnetBlock(256,'reflect', nn.BatchNorm2d, False, False)
            self.tenthLayer = ConvInstNormReluLayer(128, 64, 3, 0.5)
            self.eleventhLayer = ConvInstNormReluLayer(64, 32, 3, 0.5)
            self.twelfthLayer = ConvInstNormReluLayer(32, 3, 7, 1)
        else:
            self.firstLayer = ConvInstNormReluLayer(1, 64, 7, 1)
            self.secondLayer = ConvInstNormReluLayer(64, 128, 3, 2)
            self.thirdLayer = ConvInstNormReluLayer(128, 256, 3, 2)
            self.fourthLayer = ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False)
            self.fifthLayer = ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False)
            self.sixthLayer = ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False)
            self.seventhLayer = ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False)
            self.eightLayer = ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False)
            self.ninthLayer = ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False)
            self.tenthLayer = ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False)
            self.eleventhLayer = ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False)
            self.twelfthLayer = ResnetBlock(256, 'reflect', nn.BatchNorm2d, False, False)
            self.thirteenthLayer = ConvInstNormReluLayer(128, 64, 3, 0.5)
            self.fourteenthLayer = ConvInstNormReluLayer(64, 32, 3, 0.5)
            self.fifteenthLayer = ConvInstNormReluLayer(32, 3, 7, 1)

    def forward(self, x):
        x = self.firstLayer(x)
        x = self.secondLayer(x)
        x = self.thirdLayer(x)
        x = self.fouthLayer(x)
        x = self.fifthLayer(x)
        x = self.sixthLayer(x)
        x = self.eightLayer(x)
        x = self.ninthLayer(x)
        x = self.tenthLayer(x)
        x = self.eleventhLayer(x)
        x = self.twelfthLayer(x)
        if self.is_six_block_model:
            x = self.thirteenthLayer(x)
            x = self.fourteenthLayer(x)
            x = self.fifteenthLayer(x)

        return x


class ConvInstNormReluLayer(nn.Module):
    """

    """
    def __init__(self, size_in, size_out, kernel_size, stride, instance_norm_active):
        super().__init__()
        self.instance_norm_active = instance_norm_active

        self.size_in = size_in
        self.size_out = size_out

        self.convolution = nn.Conv2d(size_in, size_out, kernel_size, stride, padding_mode='reflect')

        self.inst_norm = nn.InstanceNorm1d(size_out)
        self.activation = nn.ReLU(0.2)

    def forward(self, x):
        x = self.convolution(x)

        if self.instance_norm_active:
            x = self.inst_norm(x)

        return self.activation(x)


"""The implementation of a resnet-block is taken from the original paper"""

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
