import logging
import torch
import torch.optim
import torch.utils.data
from pytorch_skeleton.execution import SkeletonExecution
from pytorch_skeleton.experiment import SkeletonExperiment
from pytorch_skeleton.dataset import SkeletonDataset
from pytorch_skeleton.model import SkeletonModel


class Skeleton:
    execution: SkeletonExecution
    experiment: SkeletonExperiment
    train_dataset: SkeletonDataset
    validation_dataset: SkeletonDataset
    test_dataset: SkeletonDataset
    train_loader: torch.utils.data.DataLoader
    validation_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    model: SkeletonModel
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler
    train_it_n: int = 0
    val_it_n: int = 0
    epoch_n: int = 0
    logger: logging.Logger

    def __init__(self, execution: SkeletonExecution, experiment: SkeletonExperiment, logger: logging.Logger):
        self.execution = execution
        self.experiment = experiment
        self.logger = logger

        # Initialize train and validation Dataset objects
        self.train_dataset = SkeletonDataset(self.experiment.data, split='train')
        self.validation_dataset = SkeletonDataset(self.experiment.data, split='validation')
        self.test_dataset = SkeletonDataset(self.experiment.data, split='test')

        # Initialize train and validation DataLoaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.experiment.conf.get('training', 'batch_size')
        )
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.experiment.conf.get('training', 'batch_size')
        )

    def train(self):
        pass

    def test(self):
        pass

    def _step_train(self, input_data):
        pass

    def _step_validation(self, input_data):
        pass

    def _step_test(self, input_data):
        pass
