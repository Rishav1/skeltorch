import argparse
import os
import torch
import logging


class Execution:
    """Execution class handler.

    Class used to store and validate the arguments of the execution. Other objects must access execution arguments
    directly from a created instance of this class, instead of parsing the arguments of `argparse.ArgumentParser`.

    Attributes:
        command (str): string identifying the command called in the execution.
        exp_name (str): name of the experiment associated to the execution.
        paths (str): dictionary containing useful paths associated to the execution.
        cuda (bool): whether or not to use CUDA to run the experiment.
        cuda_device (int): number of CUDA device to use.
        device (str): identifier of the device to use to run the experiment.
        seed (int): random seed used in the experiment.
        verbose (bool): whether or not to log execution information
        logger (logging.Logger): logger object.
    """
    command: str
    exp_name: str
    paths: dict = {}
    cuda: bool
    cuda_device: int
    device: str
    seed: int
    verbose: bool
    logger: logging.Logger

    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        """Constructor of the execution class.

        Uses the arguments extracted from `argparse.ArgumentParser` to fill the class parameters and validate the
        execution.

        Args:
            args (argparse.Namespace): object containing the non-validated arguments of the execution.
            logger (logging.Logger): logger object.
        """
        self.command = args.command
        self.exp_name = args.exp_name
        self.cuda = args.cuda
        self.cuda_device = args.cuda_device
        self.device = 'cpu' if not self.cuda else 'cuda:{}'.format(self.cuda_device)
        self.seed = args.seed
        self.verbose = args.verbose
        self.logger = logger
        self._initialize_paths(args)
        self._validate_execution()

    def _initialize_paths(self, args: argparse.Namespace):
        """Initializes the paths of the execution.

        It checks the relevant arguments of `args` to check if they are valid. If not, it sets the default paths
        which are:
            * Experiment Folder Path: `./experiments/`
            * Data Folder Path: `./data/`
            * Configuration File Path: `./config.json`
            * Configuration Schema Path: `./config.schema.json`

        Args:
            args (argparse.Namespace): object containing the non-validated arguments of the execution.
        """
        self.paths['package'] = os.path.dirname(os.path.dirname(__file__)) + os.sep

        self.paths['experiments'] = os.path.join(self.paths['package'], 'experiments') \
            if 'exp_folder_path' not in args or not args.exp_folder_path else args.exp_folder_path

        self.paths['data'] = os.path.join(self.paths['package'], 'data') \
            if 'data_folder_path' not in args or not args.data_folder_path else args.data_folder_path

        self.paths['config_file'] = os.path.join(self.paths['package'], 'config.json') \
            if 'config_file_path' not in args or not args.config_file_path else args.config_file_path

        self.paths['config_schema'] = os.path.join(self.paths['package'], 'config.schema.json') \
            if 'config_schema_path' not in args or not args.config_schema_path else args.config_schema_path

    def _validate_execution(self):
        """Validates the attributes of the execution

        Validates the different attributes of the execution:
            1. Verifies that CUDA is available if it is requested. If not, it sets the CPU mode.
            2. Verify that the requested CUDA device is available. If not, sets the GPU with index 0.
        """
        if self.cuda and not torch.cuda.is_available():
            self.logger.warning('CUDA requested but not available. Switching to CPU mode.')
            self.cuda = False
            self.cuda_device = 0
            self.device = 'cpu'

        if self.cuda and self.cuda_device >= torch.cuda.device_count():
            self.logger.warning('Requested GPU device {} but not available. Switching to GPU device 0.'
                                .format(self.cuda_device))
            self.cuda_device = 0
            self.device = 'cuda:0'
