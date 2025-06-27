import yaml
from typing import Any, Dict
from pathlib import Path


class ConfigLoader:
    def __init__(self, file_name: str = "config.yml"):
        config_path = Path(__file__).parent / file_name

        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file '{config_path}' not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")

    def get(self, key: str, default, required: bool = True):
        value = self.config.get(key, default)
        if required and value is None:
            raise KeyError(f"Missing required config variable: {key}")
        return value


# def load_config(file_name: str = 'config.yml', required: bool = True) -> Dict[str, Any]:
#     """
#     Load configuration from a YAML file

#     Args:
#         file_path (str): Path to the YAML config file.

#     Returns:
#         dict: Parsed configuration.

#     Raises:
#         FileNotFoundError: If the file does not exist.
#         yaml.YAMLError: If the YAML content is invalid.
#     """
#     config_path = Path(__file__).parent / file_name

#     try:
#         with open(config_path, 'r') as file:
#             return yaml.safe_load(file)
#     except FileNotFoundError:
#         raise FileNotFoundError(
#             f"Configuration file '{config_path}' not found.")
#     except yaml.YAMLError as e:
#         raise ValueError(f"Error parsing YAML config: {e}")
