import argparse
import time

from data.ApplesOrangesDataset import ApplesOrangesDataset
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

    # 2nd we need to create the models
    model = CyclicGanModel(opt)

    # 3rd we need to do the actual training
    count_iterations = 0

    for epoch in (0, 200):
        time_start = time.time()

    # 4th storing the model and so on
