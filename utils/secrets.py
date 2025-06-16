from dotenv import load_dotenv
import os
from pathlib import Path

class SecretsLoader:
    def __init__(self, env_path:str ='.env'):
        """
        Initializes the loader and loads environment variables from the given .env file.
        """
        env_file = Path(env_path)
        if not env_file.exists():
            raise FileNotFoundError(f"{env_path} not found.")
        load_dotenv(dotenv_path=env_file)
    
    def get(self, key:str, default=None, required:bool = True) -> str:
        """
        Retrieve the value for the given environment variable key.
        
        Args:
            key (str): The name of the environment variable.
            default: The default value if not found.
            required (bool): Whether to raise an error if the key is missing.

        Returns:
            str: The value of the environment variable.

        Raises:
            KeyError: If the key is required and not found.
        """
        value = os.getenv(key, default)
        if required and value is None:
            raise KeyError(f"Missing required environment variable: {key}")
        return value
            
        