import logging
import os
import re
import torch
import tensorboardX
from pytorch_skeleton.configuration import SkeletonConfiguration
from pytorch_skeleton.data import SkeletonData


class SkeletonExperiment:
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
    exp_name: str
    paths: dict = {}
    conf: SkeletonConfiguration
    data: SkeletonData
    tbx: tensorboardX.SummaryWriter
    logger: logging.Logger

    def __init__(self, exp_name: str, exp_folder_path: str, logger: logging.Logger):
        """Constructor of the experiment class.

        Args:
            exp_name (str): name of the experiment.
            logger (logging.Logger): logger object.
        """
        self.exp_name = exp_name
        self.logger = logger
        self.conf = SkeletonConfiguration()
        self.data = SkeletonData()
        self._initialize_paths(exp_folder_path)

    def create(self, config_file_path: str, config_schema_file_path: str):
        """Creates a new experiment.

        The process of creating an experiment follows this steps:
            1. Create an instance of `SkeletonConfiguration` and `SkeletonData`.
            2. Create the folders associated to an experiment.
            3. Save both instances inside the experiment folder.
            4. Create an empty `verbose.log` inside the experiment folder.

        Args:
            config_file_path (str): path to the configuration file to be used in the experiment.
            config_schema_file_path (str): path to the schema file to be used to validate the configuration file.
        """
        self.conf.create(config_file_path, config_schema_file_path)
        self.data.create(self.conf)
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
