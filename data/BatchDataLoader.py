import torch


class BatchDataLoader:
    """Wrapper class used for loading a batch of images from a given dataset. Allows for parallel loading for batches > 1"""

    def __init__(self, dataset, batch_size, require_shuffle):
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=require_shuffle,
                                                      num_workers=4)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data
