import argparse
import time
import torch
from torch.nn import init

from data.ApplesOrangesDataset import ApplesOrangesDataset
from data.BatchDataLoader import BatchDataLoader
from models.CyclicGanModel import CyclicGanModel
from util.OptionsManager import OptionsManager


def find_dataset(options):
    if options.DatasetName == 'Apples2Oranges':
        return ApplesOrangesDataset(options)
    else:
        raise NotImplementedError

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
    device = model.get_device()
    model.to(device)

    # 2.5 initialize the weights of the model to a gaussian distribution
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
    model.apply(init_weights_gaussian)

    # 3rd we need to do the actual training
    count_iterations = 0

    # hard coded for 200 epochs - FOR NOW ;)
    for epoch in (0, 200):
        time_start = time.time()
        # adapt learning rate:
        model.update_learning_rate(epoch)

        for index, collection in enumerate(dataloader):
            time_start_iteration = time.time()
            print(collection)
            model.load_input({'image_A': collection[0], 'image_B': collection[1]})
            model.train_parameter()

            # 4th storing the model and so on
