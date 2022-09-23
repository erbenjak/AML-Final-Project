import random
import torch


class ImageBuffer:
    """This Buffer is used to reduce oscillation. The paper introduces Shrivastava et al.’s strategy,
    which this class will provide an implementation of."""

    def __init__(self, opt):
        """Initialize the buffer and set its size the recommended default is 50"""
        self.size_total = opt.ImageBufferSize
        # current size is set to zero
        self.size_current = 0
        # the buffer is simply a
        self.buffer = []

    def create_mini_batch_from_buffer(self, input_images):
        # This is a bit hilarious when working with batch_size 1 - admittedly
        mini_batch = []

        for image in input_images:
            if self.size_current < self.size_total:
                # buffer is not full image is added to buffer and to the minibatch
                self.buffer.append(image)
                mini_batch.append(image)
                self.size_current += 1
            else:
                # if the buffer is filled either an image from the pool is picked OR
                # the current image is selected the chance for this should be 50/50
                if random.randint(0, 100 - 1) < 50:
                    # case A: do not use buffer:
                    mini_batch.append(image)
                else:
                    # case B: pick random image from buffer and replace
                    chosenImageIdx = random.randint(0, self.size_total - 1)
                    chosenImage = self.buffer[chosenImageIdx].clone()
                    self.buffer[chosenImageIdx] = image
                    mini_batch.append(chosenImage)
        # the minibatch still needs to be bundles to a tensor
        if input_images.size()[0] == 1:
            return torch.cat(mini_batch, 0).unsqueeze(0)

        idx = 0
        for image in mini_batch:
            mini_batch[idx] = image.unsqueeze(0)
            idx += 1

        return torch.cat(mini_batch, 0)
