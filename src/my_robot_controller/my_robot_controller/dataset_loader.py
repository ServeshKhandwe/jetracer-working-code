"""
dataset_loader.py

This module implements the DatasetLoader class, responsible for loading
offline trajectory data from .npz files and preparing it for system identification.
The DatasetLoader processes state-action pairs and provides methods to extract
training data for learning the system dynamics model.
"""

import os
import logging
import numpy as np
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads and processes offline trajectory data for system identification.
    """

    def __init__(self, dataset_path: str) -> None:
        """
        Initialize the DatasetLoader.

        Args:
            dataset_path (str): Path to the .npz file containing collected trajectories.
        """
        self.dataset_path = dataset_path
        self.state_trajectories: List[np.ndarray] = []
        self.action_trajectories: List[np.ndarray] = []
        self.loaded = False
        
        if os.path.exists(dataset_path):
            self._load_data()
        else:
            logger.warning(f"Dataset file not found: {dataset_path}")

    def _load_data(self) -> None:
        """Load trajectory data from the .npz file."""
        try:
            data = np.load(self.dataset_path, allow_pickle=True)
            
            # Extract trajectories
            self.state_trajectories = list(data['state_trajs'])
            self.action_trajectories = list(data['action_trajs'])
            
            logger.info(f"Loaded {len(self.state_trajectories)} state trajectories")
            logger.info(f"Loaded {len(self.action_trajectories)} action trajectories")
            
            if self.state_trajectories:
                logger.info(f"Example state trajectory shape: {self.state_trajectories[0].shape}")
            if self.action_trajectories:
                logger.info(f"Example action trajectory shape: {self.action_trajectories[0].shape}")
            
            self.loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load dataset from {self.dataset_path}: {e}")
            self.loaded = False

    def get_training_data(self, n_physical_state: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract training data for system identification.
        
        Args:
            n_physical_state (int): Dimension of physical state (first n components of observation)
            
        Returns:
            Tuple containing:
            - X_data: Current states (N, n_physical_state)
            - U_data: Actions (N, n_action) 
            - X_next_data: Next states (N, n_physical_state)
        """
        if not self.loaded or not self.state_trajectories:
            logger.warning("No data loaded. Returning empty arrays.")
            return np.array([]).reshape(0, n_physical_state), np.array([]).reshape(0, 2), np.array([]).reshape(0, n_physical_state)
        
        X_list = []
        U_list = []
        X_next_list = []
        
        for i, (state_traj, action_traj) in enumerate(zip(self.state_trajectories, self.action_trajectories)):
            if len(state_traj) < 2 or len(action_traj) < 1:
                logger.warning(f"Trajectory {i} too short, skipping")
                continue
                
            # Extract physical states (first n_physical_state dimensions)
            physical_states = state_traj[:, :n_physical_state]
            
            # Ensure we have matching dimensions
            min_len = min(len(physical_states) - 1, len(action_traj))
            if min_len <= 0:
                continue
                
            # Current states and actions
            X_curr = physical_states[:-1][:min_len]  # s_t
            U_curr = action_traj[:min_len]           # u_t
            X_next = physical_states[1:][:min_len]   # s_{t+1}
            
            X_list.append(X_curr)
            U_list.append(U_curr)
            X_next_list.append(X_next)
        
        if not X_list:
            logger.warning("No valid trajectory data found")
            return np.array([]).reshape(0, n_physical_state), np.array([]).reshape(0, 2), np.array([]).reshape(0, n_physical_state)
        
        # Concatenate all trajectories
        X_data = np.vstack(X_list)
        U_data = np.vstack(U_list)
        X_next_data = np.vstack(X_next_list)
        
        logger.info(f"Training data shapes: X={X_data.shape}, U={U_data.shape}, X_next={X_next_data.shape}")
        
        return X_data, U_data, X_next_data

    def get_data_statistics(self) -> dict:
        """Get basic statistics about the loaded data."""
        if not self.loaded:
            return {}
            
        stats = {
            'num_trajectories': len(self.state_trajectories),
            'total_transitions': 0,
            'avg_trajectory_length': 0,
            'state_dim': 0,
            'action_dim': 0
        }
        
        if self.state_trajectories:
            trajectory_lengths = [len(traj) for traj in self.state_trajectories]
            stats['total_transitions'] = sum(max(0, length - 1) for length in trajectory_lengths)
            stats['avg_trajectory_length'] = np.mean(trajectory_lengths)
            stats['state_dim'] = self.state_trajectories[0].shape[1] if len(self.state_trajectories[0].shape) > 1 else 1
            
        if self.action_trajectories:
            stats['action_dim'] = self.action_trajectories[0].shape[1] if len(self.action_trajectories[0].shape) > 1 else 1
            
        return stats
