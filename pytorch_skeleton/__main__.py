import argparse
import numpy as np
import random
import torch
import torch.cuda
from pytorch_skeleton import Skeleton
from pytorch_skeleton.execution import SkeletonExecution
from pytorch_skeleton.experiment import SkeletonExperiment
from pytorch_skeleton.utils import SkeletonUtils

# Create the main parser and add global parameters to it
parser = argparse.ArgumentParser(description='Run Skeleton')
parser.add_argument('--exp-name', required=True, help='Name of the experiment')
parser.add_argument('--cuda', action='store_true', help='Whether you want to run in GPU')
parser.add_argument('--cuda-device', default=0, type=int, help='GPU device to run into. Starting from index 0.')
parser.add_argument('--seed', type=int, default=0, help='Seed for the generation of random values')
parser.add_argument('--verbose', action='store_true', help='Whether to output the log using the standard output')
parser.add_argument('--exp-folder-path', help='Basepath to store/load the experiment from')
parser.add_argument('--data-folder-path', help='Basepath to store/load the data from')

# Create subparsers for the different commands
subparsers = parser.add_subparsers(dest='command', required=True)
subparsers_init = subparsers.add_parser(name='init')
subparsers_train = subparsers.add_parser(name='train')
subparsers_test = subparsers.add_parser(name='test')

# Add parameters to the subparsers
subparsers_init.add_argument('--config-file-path', type=str, help='Path to the config file')
subparsers_init.add_argument('--config-schema-path', type=str, help='Path to the schema file used to validate the '
                                                                    'configuration')

# Create the Logger object for the execution
logger = SkeletonUtils.get_logger()

# Create the Execution and Experiment object
execution = SkeletonExecution(parser.parse_args(), logger)
experiment = SkeletonExperiment(execution.exp_name, execution.paths['experiments'], logger)

# Extend the Logger to fulfill execution arguments if the command is not init
if execution.command and execution.command != 'init':
    SkeletonUtils.extend_logger(logger, experiment.paths['experiment_log'], execution.verbose)

# Initialize random seeds
random.seed(execution.seed)
np.random.seed(execution.seed)
torch.manual_seed(execution.seed)
torch.cuda.manual_seed_all(execution.seed)

# Execute actions associated to the 'init' command
if execution.command == 'init':
    experiment.create(execution.paths['config_file'], execution.paths['config_schema'])

# Execute actions associated to the 'train' command
elif execution.command == 'train':
    experiment.load()
    Skeleton(execution, experiment, logger).train()

# Execute actions associated to the 'train' command
elif execution.command == 'test':
    experiment.load()
    Skeleton(execution, experiment, logger).test()
