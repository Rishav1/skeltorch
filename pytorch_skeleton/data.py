import numpy as np
import random
import pickle
from pytorch_skeleton.configuration import SkeletonConfiguration


class SkeletonData:
    """Data class handler.

    Class used to store data-related information such as path references, data features or even the raw data itself.
    Use this class and its methods/attributes as an intermediate point between the project data and the file system
    to provide an abstracted implementation. Do not return `torch.Tensor` objects, but raw data.
    """

    samples: np.ndarray
    train_indexes: list
    validation_indexes: list
    test_indexes: list

    def __init__(self):
        pass

    def create(self, conf: SkeletonConfiguration):
        """Creates the data object associated to the experiment.

        In this toy example, we generate samples that fulfills y = 2x +1 if the configuration parameter
        `generate_with_bias` is true or y = 2x if not. We split the data with the percentages of the configuration.

        Note: As the data is light-weight, we store it as an object. For heavy files such as thousands of images,
        it would be better to only store the file paths and to create a function to load them dynamically or to
        follow an hybrid approach.

        Args:
            conf (SkeletonConfiguration): configuration object of the experiment.
        """
        self._generate_fake_data(conf.get('data', 'total_samples'), conf.get('data', 'generate_with_bias'))
        self._split_data(conf.get('split', 'validation_percentage'), conf.get('split', 'test_percentage'))

    def save(self, data_file_path: str):
        """Saves the data object attributes inside the experiment.

        Stores all relevant data attributes inside `data_file_path` to be retrieved in a future execution.

        Args:
            data_file_path (str): path to the data file inside the experiment.
        """
        with open(data_file_path, 'wb') as data_file:
            data = dict()
            data['samples'] = self.samples
            data['train_indexes'] = self.train_indexes
            data['validation_indexes'] = self.validation_indexes
            data['test_indexes'] = self.test_indexes
            pickle.dump(data, data_file)

    def load(self, data_file_path: str):
        """Loads the data attributes from the experiment.

        Retrieves data object attributes previously-stored using `self.save()` inside `data_file_path`.

        Args:
            data_file_path (str): path to the data file inside the experiment.
        """
        with open(data_file_path, 'rb') as data_file:
            data = pickle.load(data_file)
            self.samples = data['samples']
            self.train_indexes = data['train_indexes']
            self.validation_indexes = data['validation_indexes']
            self.test_indexes = data['test_indexes']

    def _generate_fake_data(self, total_samples: int, generate_with_bias: bool):
        self.samples = np.ndarray((total_samples, 2))
        self.samples[:, 0] = np.random.randint(0, 100, total_samples)
        self.samples[:, 1] = 2 * self.samples[:, 0] + 1 if generate_with_bias else 2 * self.samples[:, 0]

    def _split_data(self, validation_percentage: int, test_percentage: int):
        num_samples = self.samples.shape[0]
        all_indexes = set(range(0, num_samples))
        self.test_indexes = random.sample(all_indexes, num_samples * test_percentage // 100)
        self.validation_indexes = random.sample(all_indexes - set(self.test_indexes), num_samples *
                                                validation_percentage // 100)
        self.train_indexes = list(all_indexes - set(self.test_indexes) - set(self.validation_indexes))
