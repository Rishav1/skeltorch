import argparse
import os


class SkeletonExecution:
    path: str
    command: str
    exp_name: str
    data_folder_path: str
    config_file_path: str
    config_schema_file_path: str
    cuda: bool
    cuda_device: int
    device: str
    seed: int
    verbose: bool

    def __init__(self, args: argparse.Namespace):
        self.path = os.path.dirname(os.path.dirname(__file__)) + os.sep
        self.command = args.command
        self.exp_name = args.exp_name
        self.data_folder_path = args.data_folder_path
        self.config_file_path = args.config_file_path if 'config_file_path' in args else None
        self.config_schema_file_path = args.config_schema_file_path if 'config_schema_file_path' in args else None
        self.cuda = args.cuda
        self.cuda_device = args.cuda_device
        self.seed = args.seed
        self.verbose = args.verbose
        self._validate_execution()

    def _validate_execution(self):
        pass
