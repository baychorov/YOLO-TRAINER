'''this module contains the trainer class for training the model'''
from core.base_trainer import BaseTrainer
import torch
import os
from ultralytics import YOLO
from typing import Dict, Any
from dataclasses import dataclass, field
from ultralytics.utils import LOGGER
import logging
"""This module provides the core training engine for YOLO models."""

from ultralytics import YOLO
from typing import Any, Dict


class YoloTrainer:
    def __init__(self, model_config: Dict, training_config: Dict, dataset_path: str, logger: Any):
        self.model_config = model_config or {}
        self.training_config = training_config or {}
        self.dataset_path = dataset_path
        self.logger = logger
        self.model = None

    def setup(self) -> bool:
        """
        Initializes the YOLO model with the specified base weights.
        Returns True if successful, False otherwise.
        """
        base_weights = self.model_config.get("base_weights", "yolov8n.pt")
        task = self.model_config.get("task", "detect")
        
        try:
            self.logger.info(f"Initializing YOLO model with weights: {base_weights} (Task: {task})")
            self.model = YOLO(base_weights, task=task)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO model: {e}")
            return False

    def train(self) -> None:
        """
        Executes the YOLO training loop using the provided configuration.
        Catches and logs any critical runtime exceptions.
        """
        if self.model is None:
            self.logger.error("Model is not initialized. Please call setup() before train().")
            return

        try:
            self.logger.info("Starting YOLO training process...")
            
            # Prepare training arguments, falling back to safe defaults if missing
            train_args = {
                "data": self.dataset_path,
                "epochs": self.training_config.get("epochs", 100),
                "batch": self.training_config.get("batch_size", 16),
                "imgsz": self.training_config.get("imgsz", 640),
                "device": self.training_config.get("device", "0"),
                "workers": self.training_config.get("workers", 8),
            }
            
            self.logger.debug(f"Applied training arguments: {train_args}")
            
            # Unpack the dictionary directly into the YOLO train method
            self.model.train(**train_args)
            
            self.logger.info("YOLO training process completed successfully.")
            
        except RuntimeError as e:
            # specifically catch hardware/memory errors which are common in ML
            self.logger.critical(f"Hardware or Memory Error during training: {e}")
            
        except Exception as e:
            self.logger.critical(f"An unexpected critical error occurred during training: {e}")