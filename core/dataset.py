"""This module handles dataset validation and preparation for YOLO training."""

import os
import yaml
from typing import Any, Dict


class DatasetHandler:
    def __init__(self, config_dict: Dict, logger: Any):
        self.config_dict = config_dict or {}
        self.logger = logger
        self.yaml_path = self.config_dict.get("yaml_path")
        self.dataset_name = "unknown_dataset"

    def validate(self) -> bool:
        if not self.yaml_path or not os.path.isfile(self.yaml_path):
            self.logger.error(f"Dataset YAML file not found: {self.yaml_path}")
            return False

        try:
            with open(self.yaml_path, "r", encoding="utf-8") as file:
                yaml_data = yaml.safe_load(file)

            required_keys = ["train", "val", "nc", "names"]
            missing_keys = [key for key in required_keys if key not in yaml_data]

            if missing_keys:
                self.logger.error(f"Invalid dataset YAML format. Missing keys: {missing_keys}")
                return False

            if "path" in yaml_data:
                yaml_dir = os.path.dirname(os.path.abspath(self.yaml_path))
                dataset_root = os.path.abspath(os.path.join(yaml_dir, yaml_data["path"]))

                if not os.path.isdir(dataset_root):
                    self.logger.error(f"Dataset root folder does not exist: {dataset_root}")
                    return False

                self.dataset_name = os.path.basename(dataset_root)
                self.logger.debug(f"Dataset root validated at: {dataset_root}")

            num_classes = yaml_data.get("nc")
            self.logger.info(f"Dataset successfully validated: {self.yaml_path} ({num_classes} classes)")
            return True

        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error in dataset file: {e}")
            return False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during dataset validation: {e}")
            return False