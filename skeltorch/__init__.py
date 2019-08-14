import logging
import argparse
import random
import torch
import numpy as np
from skeltorch.experiment import Experiment
from skeltorch.execution import Execution
from skeltorch.configuration import Configuration
from skeltorch.data import Data
from skeltorch.runner import Runner
from typing import Type, Callable


class Skeltorch:
    parser: argparse.ArgumentParser
    subparsers: dict
    execution: Execution
    experiment: Experiment
    runner: Runner
    logger: logging.Logger
    _commandHandlers: dict

    def __init__(self, conf: Type[Configuration], data: Type[Data], runner: Type[Runner]):
        self._init_parser()
        self._init_subparsers()
        self._init_logger()
        self._create_execution()
        self._set_random_seed()
        self._create_experiment(conf, data)
        self._create_runner(runner)
        self._init_command_handlers()

    def _init_parser(self):
        self.parser = argparse.ArgumentParser(description='Run Skeleton')
        self.parser.add_argument('--exp-name', required=True, help='Name of the experiment')
        self.parser.add_argument('--cuda', action='store_true', help='Whether you want to run in GPU')
        self.parser.add_argument('--cuda-device', default=0, type=int,
                                 help='GPU device to run into. Starting from index 0.')
        self.parser.add_argument('--seed', type=int, default=0, help='Seed for the generation of random values')
        self.parser.add_argument('--verbose', action='store_true',
                                 help='Whether to output the log using the standard output')
        self.parser.add_argument('--exp-folder-path', help='Basepath to store/load the experiment from')
        self.parser.add_argument('--data-folder-path', help='Basepath to store/load the data from')

    def _init_subparsers(self):
        self.subparsers = dict()
        self.subparsers['_creator'] = self.parser.add_subparsers(dest='command', required=True)
        self.subparsers['init'] = self.subparsers['_creator'].add_parser(name='init')
        self.subparsers['train'] = self.subparsers['_creator'].add_parser(name='train')
        self.subparsers['test'] = self.subparsers['_creator'].add_parser(name='test')
        self.subparsers['init'].add_argument('--config-file-path', type=str, help='Path to the config file')
        self.subparsers['init'].add_argument('--config-schema-path', type=str,
                                             help='Path to the schema file used to validate the configuration')

    def _init_logger(self):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger('skeltorch')

    def _create_execution(self):
        self.execution = Execution(self.parser.parse_args(), self.logger)

    def _set_random_seed(self):
        random.seed(self.execution.seed)
        np.random.seed(self.execution.seed)
        torch.manual_seed(self.execution.seed)
        torch.cuda.manual_seed_all(self.execution.seed)

    def _create_experiment(self, conf_class: Type[Configuration], data_class: Type[Data]):
        conf = conf_class()
        self.experiment = Experiment(self.execution, conf, data_class(conf), self.logger)

    def _create_runner(self, runner_class: Type[Runner]):
        self.runner = runner_class(self.execution, self.experiment, self.logger)

    def _init_command_handlers(self):
        self._commandHandlers = dict()
        self._commandHandlers['init'] = self.experiment.create
        self._commandHandlers['train'] = self.runner.train
        self._commandHandlers['test'] = self.runner.test

    def add_command_handler(self, command_name: str, command_handler: Callable):
        self._commandHandlers[command_name] = command_handler

    def run(self):
        if self.execution.command != 'init':
            self.logger.propagate = self.execution.verbose
            self.logger.addHandler(logging.FileHandler(self.experiment.paths['experiment_log']))
            self.experiment.load()
            self.runner.init()
        self._commandHandlers[self.execution.command]()
