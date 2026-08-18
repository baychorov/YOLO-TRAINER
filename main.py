"""Main entry point for the YOLO Training Pipeline."""

import argparse
import sys
from dotenv import load_dotenv

from utils.config import ConfigManager
from utils.logger import Logger
from utils.sender import EmailSender
from core.dataset import DatasetHandler
from core.trainer import YoloTrainer


class BootstrapLogger:
    """A minimal logger for the initialization phase to break dependency loops."""
    
    def info(self, message: str) -> None:
        print(f"[BOOTSTRAP - INFO] {message}")

    def error(self, message: str) -> None:
        print(f"[BOOTSTRAP - ERROR] {message}")


def main():
    parser = argparse.ArgumentParser(description="YOLO Training Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.local.toml", 
        help="Path to the configuration file"
    )
    args = parser.parse_args()

    load_dotenv()

    temp_logger = BootstrapLogger()
    config_manager = ConfigManager(config_path=args.config, logger=temp_logger)
    config_data = config_manager.settings

    notifier_settings = config_data.get("notifier", {})
    email_sender = EmailSender(config_dict=notifier_settings)

    logger_settings = config_data.get("logger", {})
    app_logger = Logger(
        **logger_settings,
        notifier=email_sender
    )
    
    app_logger.info("Pipeline architecture successfully initialized!")

    dataset_settings = config_data.get("dataset", {})
    dataset_handler = DatasetHandler(config_dict=dataset_settings, logger=app_logger)
    
    if not dataset_handler.validate():
        app_logger.error("Dataset validation failed. Halting execution.")
        sys.exit(1)

    model_settings = config_data.get("model", {})
    training_settings = config_data.get("training", {})
    
    trainer = YoloTrainer(
        model_config=model_settings,
        training_config=training_settings,
        dataset_path=dataset_handler.yaml_path,
        dataset_name=dataset_handler.dataset_name,
        logger=app_logger
    )
    
    if not trainer.setup():
        app_logger.error("Model setup failed. Halting execution.")
        sys.exit(1)
        
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Execution interrupted by the user. Exiting safely...")
        sys.exit(0)