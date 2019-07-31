import json
import jsonschema
import pickle


class SkeletonConfiguration:
    """Class to handle project configuration parameters.

    Configuration parameters given in `config.json` are stored as class attributes dynamically. Configuration
    parameters must be grouped in 4 main categories:

        * Data: configuration parameters that have anything to do with the data itself, such as the quantization
        depth for images or the dataset to use.
        * Split: configuration parameters used to split the data, such as the number of images to use for each class
        or the number of images allocated to the validation and test splits.
        * Model: configuration parameters used to tune the model.
        * Training: configuration parameters related with the training process itself, such as the batch size,
        the learning rate strategy or the early-stopping condition.

    Other configuration categories may be included, depending on the specific task. Remember to declare the category
    in the class attributes so the `set()` method does not raises an exception.
    """
    data: dict
    split: dict
    model: dict
    training: dict

    def __init__(self):
        self.data = {}
        self.split = {}
        self.model = {}
        self.training = {}

    def create(self, config_file_path: str, config_schema_path: str):
        """Loads and validates configuration file with schema validation.

        Called when creating the experiment to validate the configuration file.

        Args:
            config_file_path (str): path to the `config.json` file to be used in the experiment.
            config_schema_path (str): path to the `config.schema.json` file used to validate the configuration.

        Raises:
            json.decoder.JSONDecodeError: error decoding one of the `.json` files.
            jsonschema.exceptions.ValidationError: raised when the configuration file does not match the schema.
        """
        with open(config_file_path, 'r') as config_file:
            config_content = json.load(config_file)
            with open(config_schema_path, 'r') as schema_file:
                schema_content = json.load(schema_file)
                jsonschema.validate(config_content, schema_content)
            for config_cat, config_cat_items in config_content.items():
                for config_param, config_value in config_cat_items.items():
                    self.set(config_cat, config_param, config_value)
        self._init_default_parameters()
        self._init_computed_parameters()

    def save(self, config_file_path: str):
        """Saves the configuration object attributes inside the experiment.

        Stores all relevant configuration attributes inside `config_file_path` to be retrieved in a future execution.

        Args:
            config_file_path (str): path to the configuration file inside the experiment.
        """
        with open(config_file_path, 'wb') as config_file:
            config = dict()
            config['data'] = self.data
            config['split'] = self.split
            config['model'] = self.model
            config['training'] = self.training
            pickle.dump(config, config_file)

    def load(self, config_file_path: str):
        """Loads the configuration attributes from the experiment.

        Retrieves configuration object attributes previously-stored using `self.save()` inside `config_file_path`.

        Args:
            config_file_path (str): path to the configuration file inside the experiment.
        """
        with open(config_file_path, 'rb') as config_file:
            config = pickle.load(config_file)
            self.data = config['data']
            self.split = config['split']
            self.model = config['model']
            self.training = config['training']

    def get(self, config_cat: str, config_param: str):
        """Gets a dynamically-loaded attribute.

        Given a configuration category `config_cat` and an identifier `config_param`, it retrieves `self.config_cat[
        config_param]`.

        Args:
            config_cat (str): category of the configuration parameter.
            config_param (str): identifier of the configuration parameter.

        Return:
            config_value (any): retrieved configuration value.
        """
        return getattr(self, config_cat)[config_param]

    def set(self, config_cat: str, config_param: str, config_value: any):
        """Sets a dynamically-loaded attribute.

        Given a configuration category `config_cat` and an identifier `config_param`, it sets `self.config_cat[
        config_param] = config_value`.

        Args:
            config_cat (str): category of the configuration parameter.
            config_param (str): identifier of the configuration parameter.
            config_value (any): configuration value to set.
        """
        getattr(self, config_cat)[config_param] = config_value

    def _init_default_parameters(self):
        """Initializes optional configuration parameters.

        Initializes all those configuration parameters which are not strictly relevant and can have a default value.
        Only initialized if no other value has explicitly been set in the configuration file.
        """
        if 'early_stopping_no_reduction_epochs' not in self.training:
            self.set('training', 'early_stopping_no_reduction_epochs', 10)
        if 'early_stopping_no_reduction_tolerance' not in self.training:
            self.set('training', 'early_stopping_no_reduction_tolerance', 0.01)

    def _init_computed_parameters(self):
        """Initializes computed configuration parameters.

        Used to remove redundant parameters that depend on other lower-level configuration parameters. Stores inside
        the configuration object in order to have direct access from other parts of the project.
        """
        self.set('data', 'test_samples', self.get('data', 'total_samples') *
                 self.get('split', 'test_percentage') // 100)
        self.set('data', 'validation_samples', self.get('data', 'total_samples') *
                 self.get('split', 'validation_percentage') // 100)
        self.set('data', 'train_samples', self.get('data', 'total_samples') - self.get('data', 'test_samples') -
                 self.get('data', 'validation_samples'))
