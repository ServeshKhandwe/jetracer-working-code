"""
goal_manager.py

This module implements goal generation and management for the safe LLM navigation task.
Goals can be set from motion capture or generated automatically in safe areas.
"""

import numpy as np
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class GoalManager:
    """
    Manages goal generation and tracking for safe navigation tasks.
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Goal generation parameters (will be mostly unused due to fixed goal)
        self.goal_area_bounds = config.get("goal_manager", {}).get("goal_area_bounds", {
            "x_min": -3.0, "x_max": 3.0,
            "y_min": -3.0, "y_max": 3.0
        })
        
        self.min_goal_distance = config.get("goal_manager", {}).get("min_goal_distance", 1.0)
        self.goal_tolerance = config.get("goal_manager", {}).get("goal_tolerance", 0.2)
        self.goal_timeout = config.get("goal_manager", {}).get("goal_timeout", 100)  # steps
        
        # Current goal state - Fixed at (0,0)
        self.current_goal = np.array([0.0, 0.0])
        self.goal_start_time = 0
        self.goals_reached = 0
        self.total_goals = 1 # Start with 1 as the fixed goal is set
        
        logger.info(f"Goal is fixed at: {self.current_goal}")
        
        # Motion capture goal setting (will be ignored for fixed goal)
        self.mocap_goal_enabled = True

    def set_goal(self, goal_position: np.ndarray):
        """
        Attempt to set goal position directly. This will be ignored if the goal is fixed.
        
        Args:
            goal_position: Goal position [x, y]
        """
        logger.info(f"Attempted to set goal to {goal_position}, but goal is fixed at {self.current_goal}. Ignoring.")
        # Do not update self.current_goal to maintain the fixed goal.
        # If mocap tries to set a new goal, we ensure the fixed goal remains.
        if self.current_goal is None or not np.array_equal(self.current_goal, np.array([0.0, 0.0])):
            self.current_goal = np.array([0.0, 0.0]) # Ensure it's reset if it was somehow changed
            logger.info(f"Re-asserting fixed goal: {self.current_goal}")


    def generate_new_goal(self, robot_position: np.ndarray, 
                         obstacle_zonotopes: List = None) -> np.ndarray:
        """
        Ensures the fixed goal at (0,0) is set.
        
        Args:
            robot_position: Current robot position [x, y, ...] (unused for fixed goal)
            obstacle_zonotopes: List of obstacle zonotopes (unused for fixed goal)
            
        Returns:
            The fixed goal position [0.0, 0.0]
        """
        if self.current_goal is None or not np.array_equal(self.current_goal, np.array([0.0, 0.0])):
            self.current_goal = np.array([0.0, 0.0])
            self.goal_start_time = 0 # Reset timer if goal is "re-generated"
            # Avoid incrementing total_goals here if it's just re-asserting the fixed goal
            logger.info(f"Fixed goal re-asserted/generated: {self.current_goal}")
        else:
            logger.info(f"Using existing fixed goal: {self.current_goal}")
            
        return self.current_goal
    
    def _is_goal_in_obstacle(self, goal_position: np.ndarray, obstacle) -> bool:
        """Check if goal position collides with an obstacle."""
        try:
            if not hasattr(obstacle, 'center') or not isinstance(obstacle.center, np.ndarray):
                return False
            
            if obstacle.center.shape[0] < 2:
                return False
            
            # Get obstacle bounds in x,y dimensions
            obs_lower, obs_upper = obstacle.get_interval_bounds()
            
            # Add small buffer around obstacles
            buffer = 0.3
            if (obs_lower[0] - buffer <= goal_position[0] <= obs_upper[0] + buffer and
                obs_lower[1] - buffer <= goal_position[1] <= obs_upper[1] + buffer):
                return True
        except Exception as e:
            logger.warning(f"Error checking goal-obstacle collision: {e}")
        
        return False
    
    def is_goal_reached(self, robot_position: np.ndarray) -> bool:
        """Check if the robot has reached the current goal."""
        if self.current_goal is None:
            return False
        
        distance = np.linalg.norm(robot_position[:2] - self.current_goal)
        return distance < self.goal_tolerance
    
    def is_goal_timeout(self, current_step: int) -> bool:
        """Check if the current goal has timed out."""
        if self.current_goal is None:
            return False
        
        return (current_step - self.goal_start_time) > self.goal_timeout
    
    def goal_reached(self):
        """Mark current goal as reached."""
        if self.current_goal is not None: # Check if there is a goal to be reached
            self.goals_reached += 1
            logger.info(f"Fixed goal (0,0) reached! ({self.goals_reached}/{self.total_goals} overall fixed goal reaches)")
            
            # For a fixed goal system, we can either:
            # 1. Keep the goal active (robot stays at goal)
            # 2. Reset for potential re-targeting
            # 
            # Since the requirement is to stop when goal is reached,
            # we'll keep the goal active so the robot knows to stay stopped
            logger.info("Goal remains active - robot should stay stopped at goal location")
            # Do NOT set self.current_goal = None to avoid re-generation
    
    def get_goal_vector(self, robot_position: np.ndarray) -> np.ndarray:
        """
        Get vector from robot to goal for state representation.
        
        Returns:
            Goal vector [dx, dy, distance, angle]
        """
        if self.current_goal is None:
            return np.zeros(4)
        
        robot_pos = robot_position[:2]
        goal_vector = self.current_goal - robot_pos
        distance = np.linalg.norm(goal_vector)
        angle = np.arctan2(goal_vector[1], goal_vector[0])
        
        return np.array([goal_vector[0], goal_vector[1], distance, angle])
    
    def get_success_rate(self) -> float:
        """Get goal success rate."""
        if self.total_goals == 0:
            return 0.0
        return self.goals_reached / self.total_goals
    
    def reset_stats(self):
        """Reset goal statistics."""
        self.goals_reached = 0
        self.total_goals = 0
        self.current_goal = None
