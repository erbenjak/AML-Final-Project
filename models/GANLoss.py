import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    This class allows for the calculation of the GAN-Loss. Having a specialized class
    allows to not have the need to create a tensor with correct target labels of the correct size.
    """
    def __init__(self, real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))

        # as a loss function the mean squared error function is used
        self.loss = nn.MSELoss()

    def __call__(self, prediction, target_is_real):
        # creating a comparison tensor
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        # expanding the comparison tensor
        target_tensor = target_tensor.expand_as(prediction)
        return self.loss(prediction, target_tensor)
