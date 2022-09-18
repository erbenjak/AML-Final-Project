import torch.nn as nn
import numpy as np
import torch.optim
import itertools
from PIL import Image
from random import randrange
from torchvision import transforms
from torch.nn import init

from models.DiscriminatorNetGenerator import DiscriminatorNetGenerator
from models.GANLoss import GANLoss
from models.GenerativeNetGenerator import GenerativeNetGenerator


def lambda_rule(epoch):
    # This is hardcoded for now 100 stable and 100 decaying epochs - FOR NOW ;)
    lr_l = 1.0 - max(0, epoch + 0 - 100) / float(100 + 1)
    return lr_l


def set_requires_grad(net, requires_grad):
    # A custom methode is necessary as requires_grad_ is an inplace operation and
    # changes the version making a second backward pass illegal
    for param in net.parameters():
        param.requires_grad = requires_grad


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

        # set correct preprocessing
        if opt.isRandomCrop:
            self.requiresRandomCrop = True
            self.requiresCorrectScaling = False
        else:
            self.requiresRandomCrop = False
            self.requiresCorrectScaling = True

        # initialize the kept data empty
        self.realA = None
        self.realB = None
        self.fakeA = None
        self.fakeB = None
        self.reconA = None
        self.reconB = None

        gen_net_producer = GenerativeNetGenerator(opt.GeneratorIsNineBlock)
        self.netG_A = gen_net_producer.create_generator()
        self.netG_B = gen_net_producer.create_generator()

        if opt.isTrain:
            # the discriminator networks are only required during training
            dis_net_producer = DiscriminatorNetGenerator()
            self.netD_A = dis_net_producer.create_discriminator()
            self.netD_B = dis_net_producer.create_discriminator()

            # default with pytorch Adam implementation
            betas = (0.9, 0.999)
            if opt.Beta1 != 0.9 or opt.Beta2 != 0.999:
                betas = (opt.beta1, opt.beta2)

            # furthermore one needs to define optimizers
            self.optimizer_generator = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.LearningRate,
                betas=betas)
            self.optimizer_discriminator = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.LearningRate,
                betas=betas)

            # we also need to define the correct schedulers which control our learning rate
            # we followed the proposed in the reference paper and keep the lr constant for 100 epochs and then decline
            # linearly over the next 100 epochs
            self.scheduler_generators = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_generator,
                                                                          lr_lambda=lambda_rule)
            self.scheduler_discriminators = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_discriminator,
                                                                              lr_lambda=lambda_rule)

    def load_input(self, inputImages):
        """The images are loaded and necessary preprocessing is performed on the data"""
        # preprocessing  would be done here
        # 1. crop the image for style-focused tasks
        # -OR-
        # 2. transform the image to the correct pixel dimensions
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
        set_requires_grad(self.netD_A, False)
        set_requires_grad(self.netD_B, False)
        # 1. reset the gradients
        self.optimizer_generator.zero_grad()
        # 2. calculate the actual losses of the generator networks
        total_generative_loss = self.calculate_loss_generators()
        total_generative_loss.backward(retain_graph=True)
        # 3. perform the actual optimization step
        self.optimizer_generator.step()
        # 3.5 turn gradients back on
        set_requires_grad(self.netD_A, True)
        set_requires_grad(self.netD_B, True)

        # Note here one could further improve as in the project from the paper by using an image buffer for actually
        # choosing old fake images sometimes

        # III. Optimize the discriminator networks
        # 1. reset the gradients
        self.optimizer_discriminator.zero_grad()
        # 2. calculate the actual losses of two discriminator networks
        discriminator_A_loss = self.calculate_loss_discriminator(self.netD_A, self.realB, self.fakeB)
        discriminator_B_loss = self.calculate_loss_discriminator(self.netD_B, self.realA, self.fakeA)
        discriminator_A_loss.backward()
        discriminator_B_loss.backward()
        # 3. perform the actual optimization step
        self.optimizer_discriminator.step()

    def calculate_loss_generators(self):
        factor_gan_loss = self.opt.GanFactor
        factor_cyclic_loss = self.opt.CyclicFactor

        # GAN losses
        GAN_loss_netG_A = self.calculate_gan_loss(self.netD_A(self.fakeB), True)
        GAN_loss_netG_B = self.calculate_gan_loss(self.netD_B(self.fakeA), True)

        # cyclic losses
        cyclic_los_rec_A = self.calculate_cyclic_loss(self.realA, self.reconA)
        cyclic_los_rec_B = self.calculate_cyclic_loss(self.realB, self.reconB)

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
        loss_methode = GANLoss().to(self.device)

        loss_real = loss_methode(discriminator(real), True)
        loss_fake = loss_methode(discriminator(fake.detach()), False)

        complete_loss = (loss_real + loss_fake) * 0.5
        return complete_loss

    def update_learning_rate(self, current_epoch):
        self.scheduler_generators.step(current_epoch)
        self.scheduler_discriminators.step(current_epoch)

    def load_to_device(self):
        self.netG_A.to(self.device)
        self.netG_B.to(self.device)
        self.netD_A.to(self.device)
        self.netD_B.to(self.device)

    def init_weights(self):
        def init_weights_gaussian(m):
            classname = m.__class__.__name__
            # all linear and convolution layers need to have their weights initialized
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.normal_(m.weight.data, 0.0, std=0.02)
                # special case for the bias
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                # special case batch-norm
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

        self.netG_A.apply(init_weights_gaussian)
        self.netG_B.apply(init_weights_gaussian)
        self.netD_A.apply(init_weights_gaussian)
        self.netD_B.apply(init_weights_gaussian)

    def get_latest_images(self):
        return {"realA": self.realA, "fakeA": self.fakeA, "reconA": self.reconA,
                "realB": self.realB, "fakeB": self.fakeB, "reconB": self.reconB}
