import torch.utils.data
from pytorch_skeleton.data import SkeletonData


class SkeletonDataset(torch.utils.data.Dataset):
    data: SkeletonData
    split: str
    indexes: list

    def __init__(self, data: SkeletonData, split: str):
        self.data = data
        self.split = split
        self.indexes = self.data.train_indexes if split == 'train' else \
            self.data.validation_indexes if split == 'validation' else self.data.test_indexes

    def __getitem__(self, item):
        return self.data.samples[self.indexes[item], 0], self.data.samples[self.indexes[item], 1]

    def __len__(self):
        return len(self.indexes)
