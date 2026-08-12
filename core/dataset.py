"""This module handles dataset validation and preparation for YOLO training."""

import os
import yaml
from typing import Any


class DatasetHandler:
    def __init__(self, config_dict: dict, logger: Any):
        self.config_dict = config_dict or {}
        self.logger = logger
        self.yaml_path = self.config_dict.get("yaml_path")

    def validate(self) -> bool:
        """
        Validates the existence and structural integrity of the YOLO dataset YAML file.
        Returns True if the dataset is valid, False otherwise.
        """
        if not self.yaml_path:
            self.logger.error("Dataset YAML path is not defined in the configuration.")
            return False

        if not os.path.isfile(self.yaml_path):
            self.logger.error(f"Dataset YAML file not found: {os.path.abspath(self.yaml_path)}")
            return False

        try:
            with open(self.yaml_path, "r", encoding="utf-8") as file:
                yaml_data = yaml.safe_load(file)

            # Check for strictly required YOLO format keys
            required_keys = ["train", "val", "nc", "names"]
            missing_keys = [key for key in required_keys if key not in yaml_data]

            if missing_keys:
                self.logger.error(f"Invalid dataset YAML format. Missing required keys: {missing_keys}")
                return False

            num_classes = yaml_data.get("nc")
            self.logger.info(f"Dataset successfully validated: {self.yaml_path} ({num_classes} classes detected)")
            return True

        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error in dataset file: {e}")
            return False
            
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during dataset validation: {e}")
            return False