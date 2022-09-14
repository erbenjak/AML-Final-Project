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

class DiscriminatorModel(nn.Module):

    def __init__(self):
        # the first layer produces 64 filters
        self.firstLayer  = ConvInstNormLeakyReluLayer(1,   64, False)
        self.secondLayer = ConvInstNormLeakyReluLayer(64, 128, True )
        self.thirdLayer  = ConvInstNormLeakyReluLayer(128,256, True )
        self.fouthLayer  = ConvInstNormLeakyReluLayer(256,512, True )
        self.fifthLayer  = nn.Conv2d(512,1,4,stride=2)

    def forward(self,x):
        x = self.firstLayer(x)
        x = self.secondLayer(x)
        x = self.thirdLayer(x)
        x = self.fouthLayer(x)
        x = self.fifthLayer(x)
        return x

class ConvInstNormLeakyReluLayer(nn.Module):
    """

    """
    def __init__(self, size_in, size_out, instance_norm_active):
        super().__init__()
        self.instance_norm_active = instance_norm_active

        self.size_in  = size_in
        self.size_out = size_out

        self.convolution = nn.Conv2d(size_in,size_out,4,stride=2)
        self.inst_norm   = nn.InstanceNorm1d(size_out)
        self.activation  = nn.LeakyReLU(0.2)


    def forward(self,x):
        x = self.convolution(x)

        if self.instance_norm_active:
            x = self.inst_norm(x)

        return self.activation(x)