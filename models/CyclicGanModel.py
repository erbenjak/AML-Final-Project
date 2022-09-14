import torch.nn as nn
import numpy as np

from models.GenerativeNetGenerator import GenerativeNetGenerator


class CyclicGanModel:

    def __init__(self, generators_are_nine_block):
        gen_net_producer = GenerativeNetGenerator(generators_are_nine_block)
        self.netG_A = gen_net_producer.create_generator()
        self.netG_B = gen_net_producer.create_generator()
