import logging
import numpy as np
import random
import torch
import torch.optim
import torch.utils.data
from skeltorch import Execution, Experiment


class Runner:
    execution: Execution
    experiment: Experiment
    datasets: dict
    loaders: dict
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler
    counters: dict
    losses: dict
    logger: logging.Logger

    def __init__(self, execution: Execution, experiment: Experiment, logger: logging.Logger):
        self.execution = execution
        self.experiment = experiment
        self.logger = logger

    def init(self):
        self.datasets = self.experiment.data.get_datasets()
        self.loaders = self.experiment.data.get_loaders()
        self.init_model()
        self.init_optimizer()
        self.init_lr_scheduler()
        self.counters = {'epoch': 1, 'train_it': 1, 'validation_it': 1}
        self.losses = {'train': {}, 'validation': {}, 'test': {}}
        self._load_checkpoint()

    def init_model(self):
        raise NotImplementedError

    def init_optimizer(self):
        raise NotImplementedError

    def init_lr_scheduler(self):
        self.lr_scheduler = None
        if self.experiment.conf.get('training', 'lr_decay_strategy') == 'plateau':
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1)

    def train(self):
        """

        Returns:

        """
        for self.counters['epoch'] in \
                range(self.counters['epoch'], self.experiment.conf.get('training', 'max_epochs') + 1):
            self.logger.info('Initiating Epoch {}'.format(self.counters['epoch']))

            e_train_losses = []
            e_validation_losses = []

            self.model.train()
            for it_data, it_target in self.train_loader:
                self.optimizer.zero_grad()
                it_loss = self._step_train(it_data, it_target)
                it_loss.backward()
                self.optimizer.step()
                e_train_losses.append(it_loss.item())
                self.counters['train_it'] += 1

            self.model.eval()
            for it_data, it_target in self.validation_loader:
                it_loss = self._step_validation(it_data, it_target)
                e_validation_losses.append(it_loss.item())
                self.counters['validation_it'] += 1

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

    def _step_train(self, it_data: any, it_target: any):
        raise NotImplementedError

    def _step_validation(self, it_data: any, it_target: any):
        raise NotImplementedError

    def test(self):
        test_losses = []
        for it_data, it_target in self.test_loader:
            it_loss = torch.nn.functional.l1_loss(it_prediction, it_target.float())
            test_losses.append(it_loss.item())
        self.logger.info('Average Test Loss: {}'.format(np.mean(test_losses)))

    def _step_test(self, it_data: any, it_target: any):
        pass

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
