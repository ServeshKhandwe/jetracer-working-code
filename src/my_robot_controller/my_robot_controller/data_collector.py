"""
data_collector.py

This module implements the DataCollector class, responsible for generating
state and action trajectories by interacting with the SimulationEnv using a specified policy
(e.g., random actions). The collected data is then saved in a .npz format suitable
for the DatasetLoader.

This script is intended to be run to generate an offline dataset for system identification.
"""

import os
import logging
import random
from typing import Any, Dict, List

import numpy as np

from config import Config
# For ROS2 integration, we'll use the real robot data instead of SimulationEnv
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from collections import deque
import time

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataCollector(Node):
    """
    Collects state and action trajectories from the real Turtlebot.
    """

    def __init__(self, config_obj: Config) -> None:
        """
        Initialize the DataCollector.

        Args:
            config_obj (Config): Configuration object containing all parameters.
        """
        super().__init__('data_collector')
        
        self.config = config_obj.parameters
        
        # Get data collection parameters
        self.num_trajectories: int = int(self.config["data_collection"]["num_trajectories"])
        self.max_steps_per_trajectory: int = int(self.config["data_collection"]["max_steps_per_trajectory"])
        self.output_file_path: str = self.config["data_collection"]["output_file_path"]
        self.policy_type: str = self.config["data_collection"]["policy_type"]

        # Robot state variables
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.yaw = 0.0
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.Lidar_readings = np.zeros((37,))

        # Data storage
        self.trajectory_data = []
        self.current_trajectory_states = []
        self.current_trajectory_actions = []
        self.trajectory_count = 0
        self.step_count = 0

        # ROS2 setup
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        self.scan_subscription = self.create_subscription(
            LaserScan, '/scan', self.laser_scan_callback, 10
        )
        self.odom_subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Timer for data collection
        self.timer = self.create_timer(0.1, self.collection_timer_callback)

        logger.info(
            f"DataCollector initialized: policy_type='{self.policy_type}', "
            f"num_trajectories={self.num_trajectories}, "
            f"max_steps_per_trajectory={self.max_steps_per_trajectory}, "
            f"output_path='{self.output_file_path}'"
        )

    def laser_scan_callback(self, msg):
        """Process LiDAR data."""
        self.Lidar_readings = np.array(msg.ranges)
        
        # Filter invalid readings
        for i in range(len(self.Lidar_readings)):
            if self.Lidar_readings[i] == float('Inf'):
                self.Lidar_readings[i] = 3.5
            elif np.isnan(self.Lidar_readings[i]):
                self.Lidar_readings[i] = 0.0001

    def odom_callback(self, msg):
        """Update robot pose."""
        self.x_pos = msg.pose.pose.position.x
        self.y_pos = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def _get_current_state(self) -> np.ndarray:
        """Get current robot state including physical state and LiDAR."""
        # Physical state: [x, y, theta, v, ex1, ex2, ex3, ex4]
        physical_state = np.array([
            self.x_pos, self.y_pos, self.yaw, self.linear_x, 0, 0, 0, 0
        ])
        
        # Pad or truncate LiDAR to 18 readings
        if len(self.Lidar_readings) >= 18:
            lidar_subset = self.Lidar_readings[:18]
        else:
            lidar_subset = np.pad(self.Lidar_readings, (0, 18 - len(self.Lidar_readings)), 
                                 'constant', constant_values=3.5)
        
        # Full observation state
        full_state = np.concatenate([physical_state, lidar_subset])
        return full_state

    def _get_random_action(self) -> np.ndarray:
        """
        Generates a random action for the turtlebot.
        Uses action ranges from the 'safety' section of the config.

        Returns:
            np.ndarray: A randomly sampled action [linear_vel, angular_vel].
        """
        action_constraints = self.config.get("safety", {}).get("action_constraints", {})
        turtlebot_constraints = action_constraints.get("turtlebot", {})
        
        lv_bounds = turtlebot_constraints.get("longitudinal_velocity", [0.0, 0.25])
        av_bounds = turtlebot_constraints.get("angular_velocity", [-0.5, 0.5])
        
        action_dim1 = random.uniform(lv_bounds[0], lv_bounds[1])
        action_dim2 = random.uniform(av_bounds[0], av_bounds[1])
        action = np.array([action_dim1, action_dim2], dtype=np.float64)
        return action

    def collection_timer_callback(self):
        """Main data collection loop."""
        if self.trajectory_count >= self.num_trajectories:
            # Data collection complete
            self._save_data()
            self.get_logger().info("Data collection completed!")
            rclpy.shutdown()
            return

        # Start new trajectory if needed
        if self.step_count == 0:
            self.current_trajectory_states = []
            self.current_trajectory_actions = []
            self.get_logger().info(f"Starting trajectory {self.trajectory_count + 1}/{self.num_trajectories}")
            
            # Store initial state
            current_state = self._get_current_state()
            self.current_trajectory_states.append(current_state.copy())

        # Generate and execute action
        if self.policy_type == "random":
            action = self._get_random_action()
        else:
            # Default to random if policy not implemented
            self.get_logger().warning(f"Policy type '{self.policy_type}' not implemented. Using random.")
            action = self._get_random_action()

        # Execute action
        msg = Twist()
        msg.linear.x = float(action[0])
        msg.angular.z = float(action[1])
        self.cmd_vel_pub.publish(msg)
        
        # Store action and current velocities
        self.linear_x = action[0]
        self.angular_z = action[1]
        self.current_trajectory_actions.append(action.copy())

        # Wait a bit for action to take effect, then store next state
        time.sleep(0.05)  # Small delay
        next_state = self._get_current_state()
        self.current_trajectory_states.append(next_state.copy())

        self.step_count += 1

        # Check if trajectory is complete
        if self.step_count >= self.max_steps_per_trajectory:
            # End trajectory
            if len(self.current_trajectory_states) > 0 and len(self.current_trajectory_actions) > 0:
                self.trajectory_data.append({
                    'states': np.array(self.current_trajectory_states, dtype=np.float64),
                    'actions': np.array(self.current_trajectory_actions, dtype=np.float64)
                })
                self.get_logger().info(
                    f"Completed trajectory {self.trajectory_count + 1} "
                    f"with {len(self.current_trajectory_states)} states and {len(self.current_trajectory_actions)} actions."
                )
            
            self.trajectory_count += 1
            self.step_count = 0
            
            # Stop robot between trajectories
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)

    def _save_data(self) -> None:
        """
        Saves the collected state and action trajectories to a .npz file.
        """
        if not self.trajectory_data:
            self.get_logger().error("No data collected to save.")
            return

        output_dir = os.path.dirname(self.output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created directory: {output_dir}")

        try:
            # Extract states and actions
            all_state_trajs = [traj['states'] for traj in self.trajectory_data]
            all_action_trajs = [traj['actions'] for traj in self.trajectory_data]
            
            # Save as object arrays because trajectories can have different lengths
            np.savez(
                self.output_file_path, 
                state_trajs=np.array(all_state_trajs, dtype=object), 
                action_trajs=np.array(all_action_trajs, dtype=object)
            )
            logger.info(f"Data successfully saved to {self.output_file_path}")
            logger.info(f"Number of state trajectories: {len(all_state_trajs)}")
            logger.info(f"Number of action trajectories: {len(all_action_trajs)}")
            if all_state_trajs:
                logger.info(f"Example state trajectory 0 shape: {all_state_trajs[0].shape}")
            if all_action_trajs:
                logger.info(f"Example action trajectory 0 shape: {all_action_trajs[0].shape if len(all_action_trajs[0]) > 0 else 'empty'}")

        except Exception as e:
            logger.error(f"Failed to save data to {self.output_file_path}: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Load configuration
        config_path = r"/home/amr/SafeLLMRA/Turtlebot/my_robot_controller/config.yaml"
        config = Config(config_path)
        
        # Instantiate DataCollector
        collector = DataCollector(config_obj=config)
        
        # Collect data
        rclpy.spin(collector)
        
    except FileNotFoundError as fnf_error:
        logger.error(f"Configuration file not found. {fnf_error}")
    except ValueError as val_error:
        logger.error(f"Configuration error. {val_error}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during data collection: {e}", exc_info=True)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
