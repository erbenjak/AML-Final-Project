import torch
import torch.nn as nn
import functools


class applesToOrangesModel:
    """This is a first model on a way to a cyclic GAN
        it will not be the most flexible but will define
        the most important things and will give a overview
        of how the project is supposed to work
    """

    def __init__(self, params):
        # the model actually requires 4 different networks

        # the two generator networks
        self.nw_gen_apple_to_orange = self.defineGeneratorNetwork()
        self.nw_gen_orange_to_apple = self.defineGeneratorNetwork()


        self.gpu_ids = params.gpu_ids
        self.isTrain = params.isTrain
        self.device  = torch.device()


    def defineGeneratorNetwork(self,in_channels,out_channels,):
