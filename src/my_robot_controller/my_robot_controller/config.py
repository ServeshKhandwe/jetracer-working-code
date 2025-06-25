"""
config.py

This module defines the Config class which loads, validates, and stores configuration parameters
from an external YAML file (default "config.yaml"). These parameters include training hyperparameters,
agent settings, simulation details, safety parameters, environment settings, and data collection settings.
The Config class provides a get_param() method for retrieving nested configuration values.
"""

import os
import logging
from typing import Any, Dict

import yaml

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Configuration loader and storage class.

    Loads configuration parameters from a YAML file, sets default values for any missing parameters,
    and provides access to these parameters via a dot-separated key lookup.

    Attributes:
        file_path (str): Path to the YAML configuration file.
        parameters (Dict[str, Any]): Dictionary containing all configuration parameters.
    """

    def __init__(self, file_path: str = "config.yaml") -> None:
        """
        Initialize the Config object by loading and validating the configuration from the YAML file.
        Default values are applied to parameters that are not specified.

        Args:
            file_path (str): The path to the YAML configuration file. Defaults to "config.yaml".
        """
        self.file_path: str = file_path
        self.parameters: Dict[str, Any] = self._load_config(self.file_path)
        self._set_defaults()
        logger.info("Configuration successfully loaded.") # Removed self.parameters from log for brevity

    def _load_config(self, file_path: str) -> Dict[str, Any]:
        """
        Load the configuration from a YAML file.

        Args:
            file_path (str): The file path to the YAML configuration file.

        Returns:
            Dict[str, Any]: A dictionary containing the configuration parameters.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If any required configuration section is missing (unless defaults cover it).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file '{file_path}' not found.")

        with open(file_path, "r") as file:
            config_data: Dict[str, Any] = yaml.safe_load(file)

        # Validate that essential sections are present.
        # Other sections like 'data_collection' will get defaults if missing.
        required_sections = ["training", "agent", "simulation", "safety", "environment"]
        for section in required_sections:
            if section not in config_data:
                raise ValueError(f"Missing required configuration section: '{section}'")

        # Ensure 'data_collection' section exists, even if empty, for default setting
        if "data_collection" not in config_data:
            config_data["data_collection"] = {}
        
        # Ensure 'environment' section has 'env_type' or set a default
        if "environment" not in config_data:
            config_data["environment"] = {} # Should not happen due to required_sections check
        if config_data["environment"].get("env_type") is None:
            logger.warning("`environment.env_type` not specified in config.yaml, defaulting to 'turtlebot'.")
            config_data["environment"]["env_type"] = "turtlebot"


        return config_data

    def _set_defaults(self) -> None:
        """
        Set default values for configuration parameters that are not provided (i.e., have a None value).
        """
        # Set defaults for training parameters.
        training_defaults: Dict[str, Any] = {
            "learning_rate": 1e-3,
            "batch_size": 256,
            "epochs": 100,
            "total_episodes": 1000,
            "steps_per_episode": 200,
            "replay_buffer_size": 100000,
        }
        if "training" not in self.parameters: # Should exist due to _load_config validation
            self.parameters["training"] = {}
        for key, default_value in training_defaults.items():
            if self.parameters["training"].get(key) is None:
                self.parameters["training"][key] = default_value

        # Set default for safety planning horizon.
        if "safety" not in self.parameters: # Should exist
             self.parameters["safety"] = {}
        if self.parameters["safety"].get("planning_horizon") is None:
            self.parameters["safety"]["planning_horizon"] = 10
            
            
        if self.parameters["safety"].get("max_generators_in_reach_set") is None:
            self.parameters["safety"]["max_generators_in_reach_set"] = 20 # Default value
            logger.info("Set default for safety.max_generators_in_reach_set: 20")
            
            
        # Set defaults for data_collection parameters
        data_collection_defaults: Dict[str, Any] = {
            "num_trajectories": 100,
            "max_steps_per_trajectory": 200,
            "output_file_path": "data/collected_offline_data.npz",
            "policy_type": "random",
            "dataset_path": "data/collected_offline_data.npz" # Default path for DatasetLoader
        }
        if "data_collection" not in self.parameters:
            self.parameters["data_collection"] = {} # Ensure section exists
        for key, default_value in data_collection_defaults.items():
            if self.parameters["data_collection"].get(key) is None:
                self.parameters["data_collection"][key] = default_value
        
        # Ensure dataset_path for DatasetLoader is properly set in main config if not in data_collection
        if self.parameters.get("dataset_path") is None:
             self.parameters["dataset_path"] = self.parameters["data_collection"]["output_file_path"]

        # Set defaults for safety.noise parameters
        safety_noise_defaults: Dict[str, Any] = {
            "process_noise_scale": 0.01,
            "measurement_noise_scale": 0.005,
            "av_noise_scale": 0.001,
        }
        if "safety" not in self.parameters: # Should exist
             self.parameters["safety"] = {}
        if self.parameters["safety"].get("noise") is None:
            self.parameters["safety"]["noise"] = {} # Ensure 'noise' sub-dictionary exists

        for key, default_value in safety_noise_defaults.items():
            if self.parameters["safety"]["noise"].get(key) is None:
                self.parameters["safety"]["noise"][key] = default_value

        # Set defaults for safety.vector_noise parameters
        env_type = self.parameters.get("environment", {}).get("env_type", "turtlebot")
        agent_config = self.parameters.get("agent", {})
        if env_type.lower() == "quadrotor":
            phys_dim_for_noise_default = agent_config.get("physical_state_dim_quadrotor", 9)
            default_z_w_diag = [0.01] * phys_dim_for_noise_default
            default_z_v_diag = [0.005] * phys_dim_for_noise_default
            default_z_av_diag = [0.002] * phys_dim_for_noise_default
        else:  # turtlebot
            phys_dim_for_noise_default = agent_config.get("physical_state_dim_turtlebot", 8)
            default_z_w_diag = [0.01] * phys_dim_for_noise_default
            default_z_v_diag = [0.005] * phys_dim_for_noise_default
            default_z_av_diag = [0.002] * phys_dim_for_noise_default

        safety_vector_noise_defaults: Dict[str, Any] = {
            "z_w_generators_diag": default_z_w_diag,
            "z_v_generators_diag": default_z_v_diag,
            "z_av_generators_diag": default_z_av_diag,
        }

        if self.parameters["safety"].get("vector_noise") is None:
            self.parameters["safety"]["vector_noise"] = {}

        for key, default_value in safety_vector_noise_defaults.items():
            current_val = self.parameters["safety"]["vector_noise"].get(key)
            if current_val is None:
                self.parameters["safety"]["vector_noise"][key] = default_value
                logger.info(f"Set default for safety.vector_noise.{key}: {default_value}")
            elif isinstance(current_val, list) and len(current_val) != phys_dim_for_noise_default:
                logger.warning(
                    f"Configured 'safety.vector_noise.{key}' has length {len(current_val)}, "
                    f"but physical_state_dim is {phys_dim_for_noise_default}. Using default: {default_value}."
                )
                self.parameters["safety"]["vector_noise"][key] = default_value

    def get_param(self, key: str, default_value: Any = None) -> Any:
        """
        Retrieve a configuration parameter using a dot-separated key.

        Example:
            To retrieve the learning rate from the training section:
                learning_rate = config.get_param("training.learning_rate")

        Args:
            key (str): A dot-separated key string for nested configuration (e.g., "training.learning_rate").
            default_value (Any): The value to return if the specified key is not found.

        Returns:
            Any: The configuration value if found; otherwise, the default_value.
        """
        keys = key.split(".")
        value: Any = self.parameters
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                logger.warning("Key '%s' not found in config. Returning default: %s", key, default_value)
                return default_value
        return value
