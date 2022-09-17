import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
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

        self.transform = transform

    def __len__(self):
        """
        :return: the size of the bigger dataset -  if
        """
        return max(len(self.image_paths_A),len(self.image_paths_B))

    def __getitem__(self, idx):
        """
        needs to return an image from A and a random image from B
        + both their paths for debugging purposes
        :param idx: index of the image to be picked
        :return:
        """
        A_image_path = self.image_paths_A[idx % self.size_A]

        """the image from b is choosen at random to not couple images up too much"""
        index_B = random.randint(0,self.size_B)
        B_image_path = self.image_paths_B[index_B]

        A_image=Image.open(A_image_path).convert('RGB')
        B_image=Image.open(B_image_path).convert('RGB')

        """the transform is not yet realized as for now the unconverted images can be used"""
        return A_image, B_image, A_image_path, B_image_path

    def find_images_in_directory(self, directory):
        pathsToFoundImages=[]

        """check if the given path is a directoy"""
        if os.path.isdir(directory)== False:
            return pathsToFoundImages

        """list all the found images"""
        listOfFiles = os.listdir(directory)

        """now all the items need to get converted to specific paths"""
        for pathToFile in listOfFiles:
            if pathToFile.__contains__(".jpg"):
                pathsToFoundImages.append(pathToFile)

        return pathsToFoundImages