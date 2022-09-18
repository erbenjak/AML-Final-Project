import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
"""
    Therefore the Dataloader also needs to always provide 2 images. One from Dataset A and one from Dataset B.
    This implementation in specific is meant to get images of oranges and apples to tranform in between the two kind.
"""


class ApplesOrangesDataset(Dataset):
    """Apples and Oranges Dataset"""
    def __init__(self, opt, transform=None):
        # the root dir needs to be modified to have both paths
        rootDir = opt.PathToData
        if opt.isTrain:
            self.root_dir_a = os.path.join(rootDir, "trainA")
            self.root_dir_b = os.path.join(rootDir, "trainB")
        else:
            self.root_dir_a = os.path.join(rootDir, "testA")
            self.root_dir_b = os.path.join(rootDir, "testB")

        self.image_paths_A = self.find_images_in_directory(self.root_dir_a)
        self.image_paths_B = self.find_images_in_directory(self.root_dir_b)

        self.size_A = len(self.image_paths_A)
        self.size_B = len(self.image_paths_B)

        self.requiresRandomCrop = False
        self.requiresCorrectScaling = True

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        """
        :return: the size of the bigger dataset -  if
        """
        return max(len(self.image_paths_A), len(self.image_paths_B))

    def __getitem__(self, idx):
        """
        needs to return an image from A and a random image from B
        + both their paths for debugging purposes
        :param idx: index of the image to be picked
        :return:
        """
        A_image_path = self.image_paths_A[idx % self.size_A]

        """the image from b is choosen at random to not couple images up too much"""
        index_B = random.randint(0, self.size_B)
        B_image_path = self.image_paths_B[index_B]

        A_image = Image.open(A_image_path).convert('RGB')
        B_image = Image.open(B_image_path).convert('RGB')

        inputImages = {'image_A': A_image, 'image_B': B_image}

        size_after_pre_pro = 256
        if self.requiresRandomCrop:
            for key, image in inputImages.items():
                x, y = image.size
                random_x_start = random.randrange(0, x - size_after_pre_pro)
                random_y_start = random.randrange(0, y - size_after_pre_pro)
                inputImages[key] = image.crop((random_x_start, random_y_start,
                                               random_x_start + size_after_pre_pro,
                                               random_y_start + size_after_pre_pro))
        if self.requiresCorrectScaling:
            inputImages['image_A'] = inputImages['image_A'].resize((size_after_pre_pro, size_after_pre_pro))
            inputImages['image_B'] = inputImages['image_B'].resize((size_after_pre_pro, size_after_pre_pro))

        """the transform is not yet realized as for now the unconverted images can be used"""
        return [self.transform(inputImages['image_A']), self.transform(inputImages['image_B'])]

    def find_images_in_directory(self, directory):
        pathsToFoundImages=[]

        """check if the given path is a directoy"""
        if not os.path.isdir(directory):
            return pathsToFoundImages

        """list all the found images"""
        listOfFiles = os.listdir(directory)

        """now all the items need to get converted to specific paths"""
        for pathToFile in listOfFiles:
            if pathToFile.__contains__(".jpg"):
                pathsToFoundImages.append(os.path.join(directory, pathToFile))
        return pathsToFoundImages
