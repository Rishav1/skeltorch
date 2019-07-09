import json


class SkeletonConfiguration:

    def __init__(self):
        pass

    def create(self, config_file_path: str, config_schema_path: str):
        # Open the config.json file placed in config_file_path
        with open(config_file_path, 'r') as config_file:

            # Load the content in a dict
            config_content = json.load(config_file)

            # Validate the content of the config file if a schema is provided
            if config_schema_path is not None:
                self._validate_conf(config_content, config_schema_path)

            # Iterate over the config parameters and set them as class attributes
            for conf_key, conf_val in config_content.items():
                setattr(self, conf_key, conf_val)

    def load(self, config_file_path: str):

        # Initialize redundant parameters
        self._init_computed_parameters()

    def save(self, config_file_path: str):
        pass

    def _validate_conf(self, config_content: dict, config_schema_path: str):
        pass

    def _init_computed_parameters(self):
        pass
