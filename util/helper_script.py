import os
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance

"""
This script helps with some tasks around the project. 
1st --> finds high contrast images
2nd --> lowers the contrast of images
3rd --> allows to pick some random images from the image pool
"""

def find_highest_contrast_images(path, destPath, amount):
    files = os.listdir(path)
    contrast_values = {}

    for file in files:
        image_path_complete = os.path.join(path, file)
        img = cv2.imread(image_path_complete)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = img_grey.std()
        contrast_values[str(file)] = contrast

    contrast_values = dict(sorted(contrast_values.items(), key=lambda item: item[1]))
    for i in range(len(contrast_values)-amount, len(contrast_values)):
        picked_file = list(contrast_values)[i]
        dest_path_complete = os.path.join(destPath, picked_file)
        source_path_complete = os.path.join(path, picked_file)
        os.rename(source_path_complete, dest_path_complete)

    print(contrast_values)

def lower_contrast_of_images(path):
    files = os.listdir(path)
    index = 1

    for file in files:
        image_path_complete = os.path.join(path, file)
        im = Image.open(image_path_complete)

        enhancer = ImageEnhance.Contrast(im)

        factor = 0.5  # decrease constrast
        im_output = enhancer.enhance(factor)

        image_path_storage = os.path.join(path, "lowcontrast_"+str(index)+".jpg")
        im_output.save(image_path_storage)

        index += 1

def getRandomFile(path, destPath, amount):
    """
    Returns a random filename, chosen among the files of the given path.
    """
    files = os.listdir(path)

    random_indecies = []

    while len(random_indecies) < amount:
        index = random.randrange(0, len(files))
        if not random_indecies.__contains__(index):
            random_indecies.append(index)

    picked_files = []
    for index in random_indecies:
        picked_files.append(files[index])

    for picked_file in picked_files:
        source_path_complete= os.path.join(path, picked_file)
        dest_path_complete = os.path.join(destPath, picked_file)
        os.rename(source_path_complete, dest_path_complete)


if __name__ == '__main__':
    lower_contrast_of_images("C:\\Users\\Nutzer\\PycharmProjects\\AML-Final-Project\\currentData\\testB")
