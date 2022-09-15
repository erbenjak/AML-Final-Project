import torch.nn as nn
import numpy as np
import torch.optim
import itertools
from PIL import Image
from random import randrange

from models.DiscriminatorNetGenerator import DiscriminatorNetGenerator
from models.GenerativeNetGenerator import GenerativeNetGenerator


class CyclicGanModel:

    def __init__(self, opt):
        # initialize the device currently under use
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)

        # initialize the kept data empty
        self.realA = None
        self.realB = None

        gen_net_producer = GenerativeNetGenerator(opt.generators_are_nine_block)
        self.netG_A = gen_net_producer.create_generator()
        self.netG_B = gen_net_producer.create_generator()

        if opt.is_training:
            # the discriminator networks are only required during training
            dis_net_producer = DiscriminatorNetGenerator()
            self.netD_A = dis_net_producer.create_discriminator()
            self.netD_B = dis_net_producer.create_discriminator()

            # default with pytorch Adam implementation
            betas = (0.9, 0.999)
            if opt.betaOne != 0.9 or opt.betaTwo != 0.999:
                betas = (opt.betaOne, opt.betaTwo)

            # furthermore one needs to define optimizers
            self.optimizer_generator = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.learning_rate,
                betas=betas)
            self.optimizer_discriminator = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.learning_rate,
                betas=betas)

            # we also need to define the correct schedulers which control our learning rate
            # - for now a step-function is used
            self.scheduler_generators = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer_generator,
                                                                        step_size=opt.num_epoches_till_lr_decay,
                                                                        gamma=0.1)
            self.scheduler_discriminators = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer_discriminator,
                                                                            step_size=opt.num_epoches_till_lr_decay,

                                                                            gamma=0.1)

    def load_input(self, inputImages):
        """The images are loaded and necessary preprocessing is performed on the data"""
        # preprocessing  would be done here
        # 1. crop the image for style-focused tasks
        # -OR-
        # 2. transform the image to the correct pixel dimensions
        size_after_pre_pro = 250
        if self.requiresRandomCrop:
            for key, image in inputImages.items():
                x,y = image.size
                random_x_start = randrange(0, x-size_after_pre_pro)
                random_y_start = randrange(0, y-size_after_pre_pro)
                inputImages[key] = image.crop((random_x_start, random_y_start,
                                               random_x_start+size_after_pre_pro,
                                               random_y_start+size_after_pre_pro))
        if self.requiresCorrectScalling:
            inputImages['image_A'] = inputImages['image_A'].resize((size_after_pre_pro, size_after_pre_pro))
            inputImages['image_B'] = inputImages['image_B'].resize((size_after_pre_pro, size_after_pre_pro))

        self.realA = inputImages['image_A'].to(self.device)
        self.realB = inputImages['image_B'].to(self.device)

