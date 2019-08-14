import logging
import os
import re
import torch
import tensorboardX
from skeltorch.configuration import Configuration
from skeltorch.data import Data
from skeltorch.execution import Execution


class Experiment:
    """Experiment class handler.

    Class used to validate the process of creating and loading an experiment. Also used to create a reference to
    experiment-specific objects, such as its configuration or its associated data.

    Attributes:
        exp_name (str): name of the experiment, which is equal to the folder name.
        paths (str): dictionary containing useful paths associated to the experiment.
        conf (SkeletonConfiguration): configuration object of the experiment.
        data (SkeletonData): data object of the experiment.
        tbx (tensorboardX.SummaryWriter): object to log data using TensorBoard.
        logger (logging.Logger): logger object.
    """
    execution: Execution
    exp_name: str
    paths: dict = {}
    conf: Configuration
    data: Data
    tbx: tensorboardX.SummaryWriter
    logger: logging.Logger

    def __init__(self, execution: Execution, conf: Configuration, data: Data, logger: logging.Logger):
        """Constructor of the experiment class.

        Args:
            exp_name (str): name of the experiment.
            logger (logging.Logger): logger object.
        """
        self.execution = execution
        self.exp_name = self.execution.exp_name
        self.logger = logger
        self.conf = conf
        self.data = data
        self._initialize_paths(self.execution.paths['experiments'])

    def create(self):
        """Creates a new experiment.

        The process of creating an experiment follows this steps:
            1. Create an instance of `SkeletonConfiguration` and `SkeletonData`.
            2. Create the folders associated to an experiment.
            3. Save both instances inside the experiment folder.
            4. Create an empty `verbose.log` inside the experiment folder.
        """
        self.conf.create(self.execution.paths['config_file'], self.execution.paths['config_schema'])
        self.data.create(self.execution.paths['data'])
        os.makedirs(self.paths['experiment'])
        os.makedirs(self.paths['experiment_tensorboard'])
        os.makedirs(self.paths['experiment_checkpoints'])
        self.conf.save(self.paths['experiment_config'])
        self.data.save(self.paths['experiment_data'])
        open(self.paths['experiment_log'], 'a').close()
        self.logger.info('Experiment {} created successfully.'.format(self.exp_name))

    def load(self):
        """Loads an existing experiment.

        The process of loading an experiment follows this steps:
            1. Loads both the `SkeletonConfiguration` and `SkeletonData` instances saved when creating the experiment.
            2. Initializes a new instance of `tensorboardX.SummaryWriter` inside `self.tbx`.
        """
        self.conf.load(self.paths['experiment_config'])
        self.data.load(self.paths['experiment_data'])
        self._initialize_tensorboard()
        self.logger.info('Experiment {} loaded successfully.'.format(self.exp_name))

    def save_checkpoint(self, checkpoint_data: dict, epoch_n: int):
        """Saves a checkpoint.

        Saves `checkpoint_data` in disk and associates it to `epoch_n`.

        Args:
            checkpoint_data (dict): data associated to the epoch.
            epoch_n (int): epoch number used to identify the checkpoint.
        """
        with open(os.path.join(self.paths['experiment_checkpoints'], 'epoch-{}.checkpoint.pkl'.format(epoch_n)),
                  'wb') as checkpoint_file:
            torch.save(checkpoint_data, checkpoint_file)
            self.logger.info('Checkpoint of epoch {} saved.'.format(epoch_n))

    def load_checkpoint(self, epoch_n: int = None, device: str = None):
        """Loads a previously-saved checkpoint.

        Loads the checkpoint associated to `epoch_n`. If it is None, it loads the last available checkpoint. If
        `epoch_n` is not available or there are no checkpoints in disk, it returns None.

        Args:
            epoch_n (int): epoch number used to identify the checkpoint.
            device (str): device to load the checkpoint into.

        Returns:
            checkpoint_data (dict): data associated to the epoch or None.
        """
        available_checkpoints = self._get_available_checkpoints()
        if epoch_n is None:
            if len(available_checkpoints) == 0:
                return None
            else:
                epoch_n = max(available_checkpoints)

        if epoch_n is not None:
            if epoch_n not in available_checkpoints:
                self.logger.error('Epoch {} is not available. Try loading another epoch.'.format(epoch_n))
                exit()

        with open(os.path.join(self.paths['experiment_checkpoints'], 'epoch-{}.checkpoint.pkl'.format(epoch_n)),
                  'rb') as checkpoint_file:
            self.logger.info('Checkpoint of epoch {} restored.'.format(epoch_n))
            return torch.load(checkpoint_file, map_location=device)

    def _initialize_paths(self, exp_folder_path: str):
        basepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments') if not exp_folder_path \
            else exp_folder_path
        self.paths['experiment'] = os.path.join(basepath, self.exp_name)
        self.paths['experiment_tensorboard'] = os.path.join(self.paths['experiment'], 'tensorboard')
        self.paths['experiment_checkpoints'] = os.path.join(self.paths['experiment'], 'checkpoints')
        self.paths['experiment_config'] = os.path.join(self.paths['experiment'], 'config.pkl')
        self.paths['experiment_data'] = os.path.join(self.paths['experiment'], 'data.pkl')
        self.paths['experiment_log'] = os.path.join(self.paths['experiment'], 'verbose.log')

    def _initialize_tensorboard(self):
        self.tbx = tensorboardX.SummaryWriter(self.paths['experiment_tensorboard'], flush_secs=9999)

    def _get_available_checkpoints(self):
        available_checkpoints = []
        for checkpoint_file in os.listdir(self.paths['experiment_checkpoints']):
            modeling_regex = re.search(r'epoch-(\d+).checkpoint.pkl', checkpoint_file)
            if modeling_regex:
                available_checkpoints.append(int(modeling_regex.group(1)))
        return available_checkpoints
