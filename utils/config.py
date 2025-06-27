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

    def get(self, key: str, default, required: bool = True) -> str:
        value = self.config.get(key, default)
        if required and value is None:
            raise KeyError(f"Missing required config variable: {key}")
        return value