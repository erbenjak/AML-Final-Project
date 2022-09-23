import os
import random
from torchvision import transforms

from data.ApplesOrangesDataset import ApplesOrangesDataset
from data.PhotoMonetDataset import PhotoMonetDataset
from data.PhotoVanGoghDataset import PhotoVanGoghDataset
from models.CyclicGanModel import CyclicGanModel
from util.OptionsManager import OptionsManager


def store_images(images, path, index_image):
    for key, image in images.items():
        real_image = transforms.ToPILImage()(image).convert("RGB")
        storage_path = path + "\\" + str(key) + str(index_image) + ".jpg"
        real_image.save(storage_path)


def create_dataset_training(options):
    if options.DatasetName == 'Apples2Oranges':
        return ApplesOrangesDataset(options)
    if options.DatasetName == 'Photo2Monet':
        return PhotoMonetDataset(options)
    if options.DatasetName == 'Photo2VanGogh':
        return PhotoVanGoghDataset(options)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    opt = OptionsManager().load_all_options()
    training_dataset = create_dataset_training(opt)

    model = CyclicGanModel(opt)
    model.load_progress(200)

    for i, data in enumerate(training_dataset):
        if i > 20:
            break

        model.load_input({'image_A': data[0], 'image_B': data[1]})
        model.forward()

        images = model.get_latest_images()
        store_images(images, opt.ImageStoragePath, i)
