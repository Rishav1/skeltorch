import torch.utils.data
from pytorch_skeleton.data import SkeletonData


class SkeletonDataset(torch.utils.data.Dataset):
    data: SkeletonData
    split: str

    def __init__(self, data: SkeletonData, split: str):
        self.data = data
        self.split = split

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
