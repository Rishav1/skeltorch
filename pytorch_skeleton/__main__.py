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
parser.add_argument('--data-folder-path', type=str, default='data', help='Folder to store data')
parser.add_argument('--cuda', action='store_true', help='Whether you want to run in GPU')
parser.add_argument('--cuda-device', default=0, type=int, help='GPU device to run into')
parser.add_argument('--seed', type=int, default=0, help='Seed for the generation of random values')
parser.add_argument('--verbose', action='store_true', help='Whether to output the log using the standard output')

# Create subparsers for the different commands
subparsers = parser.add_subparsers(dest='command')
subparsers_init = subparsers.add_parser(name='init')
subparsers_train = subparsers.add_parser(name='train')
subparsers_test = subparsers.add_parser(name='test')

# Add parameters to the subparsers
subparsers_init.add_argument('--config-file-path', type=str, default='config.json', help='Path to the config.json file')
subparsers_init.add_argument('--config-schema-file-path', type=str, default='config.schema.json',
                             help='Path to the config.schema.json file')

# Parse the Arguments
args = parser.parse_args()

# Create the Execution and Experiment object
execution = SkeletonExecution(args)
experiment = SkeletonExperiment(execution.exp_name)

# Create a logger object for the execution
logger = SkeletonUtils.get_logger(experiment.paths['experiment_log'],
                                  execution.verbose) if execution.command != 'init' else None

# Initialize random seeds
random.seed(execution.seed)
np.random.seed(execution.seed)
torch.manual_seed(execution.seed)
torch.cuda.manual_seed_all(execution.seed)

# Execute actions associated to the 'init' command
if args.command == 'init':
    experiment.create(execution.config_file_path, execution.config_schema_file_path)
    logger.info('Experiment {} created successfully.'.format(experiment.exp_name))

# Execute actions associated to the 'train' command
elif args.command == 'train':
    experiment.load()
    logger.info('Experiment {} loaded successfully.'.format(experiment.exp_name))
    Skeleton(execution, experiment, logger).train()

# Execute actions associated to the 'train' command
elif args.command == 'test':
    experiment.load()
    logger.info('Experiment {} loaded successfully.'.format(experiment.exp_name))
    Skeleton(execution, experiment, logger).test()
