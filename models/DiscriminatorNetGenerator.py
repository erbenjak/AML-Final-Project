import torch
import torch.nn as nn

"""This class will generate the discriminator networks as
proposed in the original paper. While doing so it tries to stay as flexible as possible
to allow for improvements later on.

-> the implementation of the paper can be found under: 
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/eb6ae80412e23c09b4317b04d889f1af27526d2d/models/networks.py#L542

"""


class DiscriminatorNetGenerator:
    """
           The discriminator network is described in the original paper
           it uses PatchGAN-architecture. It uses the proposed amounts
           of filters.
    """

    def __init__(self):
        self.created = True

    @staticmethod
    def create_discriminator():
        return DiscriminatorModel();


class DiscriminatorModel(nn.Module):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()
        # the first layer produces 64 filters
        model = [ConvInstNormLeakyReluLayer(3, 64, False),
                 ConvInstNormLeakyReluLayer(64, 128, True),
                 ConvInstNormLeakyReluLayer(128, 256, True),
                 ConvInstNormLeakyReluLayer(256, 512, True),
                 nn.Conv2d(512, 1, 4, 2)
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ConvInstNormLeakyReluLayer(nn.Module):
    def __init__(self, size_in, size_out, instance_norm_active):
        super(ConvInstNormLeakyReluLayer, self).__init__()
        self.instance_norm_active = instance_norm_active

        self.size_in = size_in
        self.size_out = size_out

        self.convolution = nn.Conv2d(size_in, size_out, 4, 2, padding=1)
        # this could also be solved with a batchNorm
        self.inst_norm = nn.InstanceNorm2d(size_out, affine=False, track_running_stats=False)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.convolution(x)

        if self.instance_norm_active:
            x = self.inst_norm(x)

        return self.activation(x)
