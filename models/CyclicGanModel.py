import torch.nn as nn
import numpy as np
import torch.optim
import itertools
from PIL import Image
from random import randrange

from models.DiscriminatorNetGenerator import DiscriminatorNetGenerator
from models.GANLoss import GANLoss
from models.GenerativeNetGenerator import GenerativeNetGenerator


class CyclicGanModel:

    def __init__(self, opt):
        # store all training options
        self.opt = opt

        # initialize the device currently under use
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)

        # initialize the kept data empty
        self.realA = None
        self.realB = None
        self.fakeA = None
        self.fakeB = None
        self.reconA = None
        self.reconB = None

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
                x, y = image.size
                random_x_start = randrange(0, x - size_after_pre_pro)
                random_y_start = randrange(0, y - size_after_pre_pro)
                inputImages[key] = image.crop((random_x_start, random_y_start,
                                               random_x_start + size_after_pre_pro,
                                               random_y_start + size_after_pre_pro))
        if self.requiresCorrectScalling:
            inputImages['image_A'] = inputImages['image_A'].resize((size_after_pre_pro, size_after_pre_pro))
            inputImages['image_B'] = inputImages['image_B'].resize((size_after_pre_pro, size_after_pre_pro))

        self.realA = inputImages['image_A'].to(self.device)
        self.realB = inputImages['image_B'].to(self.device)

    def forward(self):
        # the following tasks need to be performed:
        # 1. create the 'fake' images
        self.fakeA = self.netG_B(self.realB)
        self.fakeB = self.netG_A(self.realA)

        # 2. use those to create the reconstructions
        self.reconA = self.netG_B(self.fakeB)
        self.reconB = self.netG_A(self.fakeA)

    def train_parameter(self):
        # I. Do a forward pass -please note that the images need to be loaded beforehand
        self.forward()

        # II. Optimize the generator networks
        # 0.5 since the discriminators do not require gradient during training of the generators they are turned off
        self.netD_A.requires_grad_(False)
        self.netD_B.requires_grad_(False)
        # 1. reset the gradients
        self.optimizer_generator.zero_grad()
        # 2. calculate the actual losses of the generator networks
        total_generative_loss = self.calculate_loss_generators()
        total_generative_loss.backwards()
        # 3. perform the actual optimization step
        self.optimizer_generator.step()
        # 3.5 turn gradients back on
        self.netD_A.requires_grad_(True)
        self.netD_B.requires_grad_(True)

        # Note here one could further improve as in the project from the paper by using an image buffer for actually
        # choosing old fake images sometimes

        # III. Optimize the discriminator networks
        # 1. reset the gradients
        self.optimizer_discriminator.zero_grad()
        # 2. calculate the actual losses of two discriminator networks
        discriminator_A_loss = self.calculate_loss_discriminator(self.netD_A, self.realB, self.fakeB)
        discriminator_B_loss = self.calculate_loss_discriminator(self.netD_B, self.realA, self.fakeA)
        discriminator_A_loss.backwards()
        discriminator_B_loss.backwards()
        # 3. perform the actual optimization step
        self.optimizer_discriminator.step()

    def calculate_loss_generators(self):
        factor_gan_loss = self.opt.gan_factor
        factor_cyclic_loss = self.opt.cyclic_factor

        # GAN losses
        GAN_loss_netG_A = self.calculateGanLoss(self.netD_A(self.fakeB), True)
        GAN_loss_netG_B = self.calculateGanLoss(self.netD_B(self.fakeA), True)

        # cyclic losses
        cyclic_los_rec_A = self.calculateCyclicLoss(self.realA, self.reconA)
        cyclic_los_rec_B = self.calculateCyclicLoss(self.realB, self.reconB)

        # summing up the losses
        loss_total = (GAN_loss_netG_A + GAN_loss_netG_B) * factor_gan_loss + (
                    cyclic_los_rec_A + cyclic_los_rec_B) * factor_cyclic_loss

        return loss_total

    def calculate_gan_loss(self, prediction, real):
        loss_methode = GANLoss().to(self.device)
        return loss_methode(prediction, real)

    @staticmethod
    def calculate_cyclic_loss(real, fake):
        loss_methode = torch.nn.L1Loss()
        return loss_methode(fake, real)

    def calculate_loss_discriminator(self, discriminator, real, fake):
        loss_methode = GANLoss.to(self.device)
        loss_real = loss_methode(discriminator(real), True)
        loss_fake = loss_methode(discriminator(fake), False)

        complete_loss = (loss_real+loss_fake) * 0.5
        return complete_loss