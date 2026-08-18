"""This module defines the abstract base class for all model trainers."""

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Abstract interface for machine learning model trainers.
    Enforces a standard contract (setup and train methods) for any model architecture 
    added to the pipeline.
    """
    
    @abstractmethod
    def setup(self) -> bool:
        """
        Initializes the model architecture, loads base weights, and prepares the environment.
        Must return True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Executes the main training loop using the dataset and hyperparameter configuration.
        """
        pass