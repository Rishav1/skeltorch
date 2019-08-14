import numpy as np
import random
import pickle
from skeltorch.configuration import Configuration


class Data:
    """Data class handler.

    Class used to store data-related information such as path references, data features or even the raw data itself.
    Use this class and its methods/attributes as an intermediate point between the project data and the file system
    to provide an abstracted implementation. Do not return `torch.Tensor` objects, but raw data.
    """
    conf: Configuration
    samples: list
    train_indexes: list
    validation_indexes: list
    test_indexes: list

    def __init__(self, conf: Configuration):
        self.conf = conf

    def create(self, data_folder_path: str):
        """Creates the data object associated to the experiment.

        In this toy example, we generate samples that fulfills y = 2x +1 if the configuration parameter
        `generate_with_bias` is true or y = 2x if not. We split the data with the percentages of the configuration.

        Note: As the data is light-weight, we store it as an object. For heavy files such as thousands of images,
        it would be better to only store the file paths and to create a function to load them dynamically or to
        follow an hybrid approach.

        Args:
            conf (SkeletonConfiguration): configuration object of the experiment.
        """
        raise NotImplementedError

    def save(self, data_file_path: str):
        """Saves the data object attributes inside the experiment.

        Stores all relevant data attributes inside `data_file_path` to be retrieved in a future execution.

        Args:
            data_file_path (str): path to the data file inside the experiment.
        """
        with open(data_file_path, 'wb') as data_file:
            data = dict()
            for attr, value in self.__dict__.items():
                data[attr] = value
            pickle.dump(data, data_file)

    def load(self, data_file_path: str):
        """Loads the data attributes from the experiment.

        Retrieves data object attributes previously-stored using `self.save()` inside `data_file_path`.

        Args:
            data_file_path (str): path to the data file inside the experiment.
        """
        with open(data_file_path, 'rb') as data_file:
            data = pickle.load(data_file)
            for attr, value in data.items():
                setattr(self, attr, value)

    def get_datasets(self):
        raise NotImplementedError

    def get_loaders(self):
        raise NotImplementedError
