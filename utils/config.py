'''This module contains the ConfigManager class for managing configuration settings.'''

import tomllib
import sys
import os
from typing import Any

class ConfigManager:
    def __init__(self, config_path:str, logger:Any):
        self.config_path = config_path
        self.logger = logger
        self.settings = self._load_config()
        

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, "rb") as f:
                        data = tomllib.load(f)
                        self.logger.info(f"Configuration succesfully loaded : {self.config_path}")
                        return data
            
        except FileNotFoundError:
              self.logger.error(f"Config file not found! path : {self.config_path}")
              sys.exit(1)
        except tomllib.TOMLDecodeError as e:
              self.logger.error(f"Invalid toml format! details {e}")
              sys.exit(1)
        except Exception as e:
              self.logger.error(f"An issue occurred while reading the configuration file: {e}")
              sys.exit(1)