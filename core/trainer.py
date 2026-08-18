"""This module provides the core training engine for YOLO models."""

import os
import re
import logging
from typing import Any, Dict
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER
from core.base_trainer import BaseTrainer


def clean_ansi_escape_codes(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return text


def setup_yolo_logging(custom_logger: Any) -> None:
    class GenericLoggerWrapper(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_level = record.levelname.lower()
            log_entry = self.format(record)
            clean_log_entry = clean_ansi_escape_codes(log_entry)

            if not clean_log_entry:
                return

            if hasattr(custom_logger, log_level):
                log_method = getattr(custom_logger, log_level)
                log_method(clean_log_entry)

    for handler in list(ULTRALYTICS_LOGGER.handlers):
        ULTRALYTICS_LOGGER.removeHandler(handler)

    handler = GenericLoggerWrapper()
    ULTRALYTICS_LOGGER.addHandler(handler)
    
    if hasattr(custom_logger, "file_log_level"):
        ULTRALYTICS_LOGGER.setLevel(custom_logger.file_log_level)


class YoloTrainer(BaseTrainer):
    def __init__(
        self,
        model_config: Dict,
        training_config: Dict,
        dataset_path: str,
        dataset_name: str,
        logger: Any,
    ):
        self.model_config = model_config or {}
        self.training_config = training_config or {}
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.logger = logger
        self.model = None
        self.weights_path = ""
        
        setup_yolo_logging(self.logger)

    def setup(self) -> bool:
        base_weights = self.model_config.get("base_weights", "yolov8n.pt")
        task = self.model_config.get("task", "detect")

        base_models_dir = os.path.abspath(os.path.join("models", "base"))
        os.makedirs(base_models_dir, exist_ok=True)
        self.weights_path = os.path.join(base_models_dir, base_weights)

        try:
            self.logger.info(f"Initializing YOLO model with weights: {self.weights_path} (Task: {task})")
            self.model = YOLO(self.weights_path, task=task)
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO model: {e}")
            return False

    def train(self) -> None:
        if self.model is None:
            self.logger.error("Model is not initialized. Please call setup() before train().")
            return

        try:
            self.logger.info("Starting YOLO training process...")

            model_name = os.path.splitext(os.path.basename(self.weights_path))[0]
            date_str = datetime.now().strftime("%Y%m%d")
            run_name = f"{self.dataset_name}-{date_str}-{model_name}"

            trained_dir = os.path.abspath(os.path.join("models", "trained"))
            os.makedirs(trained_dir, exist_ok=True)

            epochs = self.training_config.get("epochs") or self.training_config.get("epoch") or 100
            batch_size = self.training_config.get("batch_size") or self.training_config.get("batch") or 16

            train_args = {
                "data": os.path.abspath(self.dataset_path),
                "epochs": epochs,
                "batch": batch_size,
                "imgsz": self.training_config.get("imgsz", 640),
                "device": str(self.training_config.get("device", "0")),
                "workers": self.training_config.get("workers", 8),
                "project": trained_dir,
                "name": run_name,
                "exist_ok": True,
                "verbose": True,
                "amp": False,
            }

            self.logger.debug(f"Applied training arguments: {train_args}")
            results = self.model.train(**train_args)
            self.logger.info(f"YOLO training process completed successfully. Artifacts saved in: {results.save_dir}")

        except RuntimeError as e:
            self.logger.critical(f"Hardware or Memory Error during training: {e}")
        except Exception as e:
            self.logger.critical(f"An unexpected critical error occurred during training: {e}")