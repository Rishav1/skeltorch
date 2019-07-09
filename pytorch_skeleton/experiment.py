import os
import tensorboardX
from pytorch_skeleton.configuration import SkeletonConfiguration
from pytorch_skeleton.data import SkeletonData


class SkeletonExperiment:
    exp_name: str
    paths: dict = {}
    conf: SkeletonConfiguration
    data: SkeletonData
    tbx: tensorboardX.SummaryWriter

    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.paths['package'] = os.path.dirname(os.path.dirname(__file__)) + os.sep
        self.paths['experiment'] = self.paths['package'] + 'experiments' + os.sep + self.exp_name + os.sep
        self.paths['experiment_tensorboard'] = self.paths['experiment'] + 'tensorboard' + os.sep
        self.paths['experiment_checkpoints'] = self.paths['experiment'] + 'checkpoints' + os.sep
        self.paths['experiment_config'] = self.paths['experiment'] + 'config.json'
        self.paths['experiment_data'] = self.paths['experiment'] + 'data.pkl'
        self.paths['experiment_log'] = self.paths['experiment'] + 'verbose.log'
        self.conf = SkeletonConfiguration()
        self.data = SkeletonData()

    def create(self, config_file_path: str, config_schema_file_path: str):
        self.conf.create(config_file_path, config_schema_file_path)
        self.data.create()
        os.makedirs(self.paths['experiment'])
        os.makedirs(self.paths['experiment_tensorboard'])
        os.makedirs(self.paths['experiment_checkpoints'])
        self.conf.save(self.paths['experiment_config'])
        self.data.save(self.paths['experiment_data'])
        open(self.paths['experiment_log'], 'a').close()

    def load(self):
        self.conf.load(self.paths['experiment_config'])
        self.data.load(self.paths['experiment_data'])
        self.tbx = tensorboardX.SummaryWriter(self.paths['experiment_tensorboard'], flush_secs=9999)

