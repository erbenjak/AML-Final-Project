import random
import time
from torchvision import transforms

from data.ApplesOrangesDataset import ApplesOrangesDataset
from data.BatchDataLoader import BatchDataLoader
from data.PhotoMonetDataset import PhotoMonetDataset
from models.CyclicGanModel import CyclicGanModel
from util.OptionsManager import OptionsManager


def find_dataset(options):
    if options.DatasetName == 'Apples2Oranges':
        return ApplesOrangesDataset(options)
    if options.DatasetName == 'Photo2Monet':
        return PhotoMonetDataset(options)
    else:
        raise NotImplementedError


def store_images(images, path, current_epoch):
    # images are actually batches
    batch_size = images['A_real'].size()[0]
    chosen_image = random.randint(1, batch_size) - 1

    for key, image in images.items():
        image = image[chosen_image]
        real_image = transforms.ToPILImage()(image).convert("RGB")
        storage_path = path + "\\" + str(key) + str(current_epoch) + ".jpg"
        real_image.save(storage_path)


if __name__ == '__main__':
    # 0.5 passing the arguments that were given
    opt = OptionsManager().load_all_options()

    # 1st we need to create the dataloader
    dataset = find_dataset(opt)
    dataset_size = len(dataset)
    print("The size of the dataset is as follows: " + str(dataset_size))

    # 1.5 create a dataloader which is able to load the data in batches in parallel
    dataloader = BatchDataLoader(dataset, batch_size=1, require_shuffle=False)

    # 2nd we need to create the models
    model = CyclicGanModel(opt)
    device = model.load_to_device()

    # 2.5 initialize the weights of the model to a gaussian distribution
    model.init_weights()

    # 3rd we need to do the actual training
    count_iterations = 0

    # hard coded for 200 epochs - FOR NOW ;)
    for epoch in range(1, 201):
        time_start = time.time()
        iteration_round = 0

        for index, collection in enumerate(dataloader):
            time_start_iteration = time.time()
            model.load_input({'image_A': collection[0], 'image_B': collection[1]})
            model.train_parameter()
            iteration_round += 1
            if iteration_round % 250 == 1:
                print("completed a run of iteration #" + str(iteration_round))

        # adapt learning rate:
        model.update_learning_rate(epoch)

        end_time = time.time()
        print("epoch took:" + str(end_time-time_start) + " seconds")

        # 4th the model will be stored by at every 5th and the images will be stored for further inspection
        if epoch % 10 == 1:
            print("Saving some images at epoch #" + str(epoch))
            latest_images = model.get_latest_images()
            store_images(latest_images, opt.ImageStoragePath, epoch)

        if epoch == 5:
            # one early storage call - for debugging purposes
            print("Saving the model at epoch #" + str(epoch))
            model.save_progress(epoch)

        if epoch % 25 == 0:
            print("Saving the model at epoch #" + str(epoch))
            model.save_progress(epoch)
