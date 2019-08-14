import json
import jsonschema
import pickle


class Configuration:
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
                setattr(self, config_cat, dict())
                for config_param, config_value in config_cat_items.items():
                    self.set(config_cat, config_param, config_value)

    def save(self, config_file_path: str):
        """Saves the configuration object attributes inside the experiment.

        Stores all relevant configuration attributes inside `config_file_path` to be retrieved in a future execution.

        Args:
            config_file_path (str): path to the configuration file inside the experiment.
        """
        with open(config_file_path, 'wb') as config_file:
            config = dict()
            for attr, value in self.__dict__.items():
                config[attr] = value
            pickle.dump(config, config_file)

    def load(self, config_file_path: str):
        """Loads the configuration attributes from the experiment.

        Retrieves configuration object attributes previously-stored using `self.save()` inside `config_file_path`.

        Args:
            config_file_path (str): path to the configuration file inside the experiment.
        """
        with open(config_file_path, 'rb') as config_file:
            config = pickle.load(config_file)
            for attr, value in config.items():
                setattr(self, attr, value)

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
        try:
            return getattr(self, config_cat)[config_param]
        except KeyError:
            return None

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
