import logging
import numpy as np
import random
import torch
import torch.optim
import torch.utils.data
from pytorch_skeleton.execution import SkeletonExecution
from pytorch_skeleton.experiment import SkeletonExperiment
from pytorch_skeleton.dataset import SkeletonDataset
from pytorch_skeleton.model import SkeletonModel


class Skeleton:
    """

    """
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
    counters: dict
    losses: dict
    logger: logging.Logger

    def __init__(self, execution: SkeletonExecution, experiment: SkeletonExperiment, logger: logging.Logger):
        self.execution = execution
        self.experiment = experiment
        self.logger = logger
        self._init_datasets()
        self._init_loaders()
        self._init_model()
        self._init_optimizer()
        self._init_lr_scheduler()
        self.counters = {'epoch': 0, 'train_it': 0, 'validation_it': 0}
        self.losses = {'train': {}, 'validation': {}, 'test': {}}
        self._load_checkpoint()

    def train(self):
        """

        Returns:

        """
        for self.counters['epoch'] in \
                range(self.counters['epoch'] + 1, self.experiment.conf.get('training', 'max_epochs') + 1):

            self.logger.info('Initiating Epoch {}'.format(self.counters['epoch']))

            e_train_losses = []
            e_validation_losses = []

            self.model.train()
            for self.counters['train_it'], (it_data, it_target) in \
                    enumerate(self.train_loader, self.counters['train_it'] + 1):
                self.optimizer.zero_grad()
                it_prediction = self.model(it_data.float())
                it_loss = torch.nn.functional.l1_loss(it_prediction, it_target.float())
                it_loss.backward()
                self.optimizer.step()
                e_train_losses.append(it_loss.item())

            self.model.eval()
            for self.counters['validation_it'], (it_data, it_target) in \
                    enumerate(self.validation_loader, self.counters['validation_it'] + 1):
                with torch.no_grad():
                    it_prediction = self.model(it_data.float())
                it_loss = torch.nn.functional.l1_loss(it_prediction, it_target.float())
                e_validation_losses.append(it_loss.item())

            self.losses['train'][self.counters['epoch']] = np.mean(e_train_losses)
            self.losses['validation'][self.counters['epoch']] = np.mean(e_validation_losses)

            self.logger.info(
                'Epoch: {} | Average Training Loss: {} | Average Validation Loss: {}'.format(
                    self.counters['epoch'],
                    self.losses['train'][self.counters['epoch']],
                    self.losses['validation'][self.counters['epoch']]
                )
            )

            self._apply_lr_decay()
            self._save_checkpoint()
            self._apply_early_stopping()

    def test(self):

        for test_epoch in self.losses['train'].keys():
            print(test_epoch)
        exit()

        test_losses = []
        for it_data, it_target in self.test_loader:
            with torch.no_grad():
                it_prediction = self.model(it_data.float())
            it_loss = torch.nn.functional.l1_loss(it_prediction, it_target.float())
            test_losses.append(it_loss.item())

        self.logger.info('Average Test Loss: {}'.format(np.mean(test_losses)))

    def _init_datasets(self):
        self.train_dataset = SkeletonDataset(self.experiment.data, split='train')
        self.validation_dataset = SkeletonDataset(self.experiment.data, split='validation')
        self.test_dataset = SkeletonDataset(self.experiment.data, split='test')

    def _init_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.experiment.conf.get('training', 'batch_size'))
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset, batch_size=self.experiment.conf.get('training', 'batch_size'))
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.experiment.conf.get('training', 'batch_size'))

    def _init_model(self):
        self.model = SkeletonModel(self.experiment.conf.get('model', 'use_bias'))

    def _init_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.experiment.conf.get('training', 'lr'))

    def _init_lr_scheduler(self):
        self.lr_scheduler = None
        if self.experiment.conf.get('training', 'lr_decay_strategy') == 'plateau':
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1)

    def _apply_lr_decay(self):
        if self.lr_scheduler:
            self.lr_scheduler.step(self.losses['validation'][self.counters['epoch']])

    def _apply_early_stopping(self):
        if self.experiment.conf.get('training', 'early_stopping_strategy') == 'no_reduction':
            no_reduction_epochs = self.experiment.conf.get('training', 'early_stopping_no_reduction_epochs')
            no_reduction_tolerance = self.experiment.conf.get('training', 'early_stopping_no_reduction_tolerance')
            if no_reduction_epochs >= self.counters['epoch']:
                return
            previous_losses = [self.losses['validation'][i] for i in
                               range(self.counters['epoch'] - 1, self.counters['epoch'] - no_reduction_epochs - 1, - 1)]
            if all(previous_losses < self.losses['validation'][self.counters['epoch']]) + no_reduction_tolerance:
                self.logger.info('No reduction in the validation loss for {} epochs (tolerance of {}). Early stopping '
                                 'applied.'.format(no_reduction_epochs, no_reduction_tolerance))
                exit()

    def _save_checkpoint(self):
        checkpoint_data = dict()
        checkpoint_data['model'] = self.model.state_dict()
        checkpoint_data['optimizer'] = self.optimizer.state_dict()
        checkpoint_data['lr_scheduler'] = self.lr_scheduler.state_dict() if self.lr_scheduler else None
        checkpoint_data['random_states'] = (
            random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state() if
            torch.cuda.is_available() else None
        )
        checkpoint_data['counters'] = self.counters
        checkpoint_data['losses'] = self.losses
        self.experiment.save_checkpoint(checkpoint_data, self.counters['epoch'])

    def _load_checkpoint(self, epoch_n: int = None):
        checkpoint_data = self.experiment.load_checkpoint(epoch_n, self.execution.device)
        if checkpoint_data:
            self.model.load_state_dict(checkpoint_data['model'])
            self.optimizer.load_state_dict(checkpoint_data['optimizer'])
            if self.lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint_data['model'])
            random.setstate(checkpoint_data['random_states'][0])
            np.random.set_state(checkpoint_data['random_states'][1])
            torch.set_rng_state(checkpoint_data['random_states'][2])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint_data['random_states'][3])
            self.counters = checkpoint_data['counters']
            self.losses = checkpoint_data['losses']
