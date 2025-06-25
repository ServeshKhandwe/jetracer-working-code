#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from collections import deque
import numpy as np
import time
import os
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import logging

# Import SafeLLMRA components with error handling
try:
    from .config import Config
    from .zonotope import Zonotope, MatrixZonotope
    from .safety_controller import SafetyController
    from .dataset_loader import DatasetLoader
    from .system_identification import SystemIdentification
    from .td3_agent import TD3Agent
    from .goal_manager import GoalManager
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import SafeLLMRA components: {e}")
    raise

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SafeLLMRATurtlebotController(Node):
    def __init__(self):
        super().__init__('safellmra_turtlebot_controller')
        
        # Initialize SafeLLMRA components
        self._initialize_safellmra()
        
        # ROS2 Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Subscriptions
        self.scan_subscription = self.create_subscription(
            LaserScan, '/scan', self.laser_scan_callback, 10
        )
        # Remove cmd_vel_subscription since we're using RL agent instead of LLM
        self.odom_subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Robot state variables
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.yaw = 0.0

        # Sensor data
        self.ranges_buffer = deque(maxlen=100)
        self.Lidar_readings = np.zeros((37,))
        
        # Timing variables
        self.time = 0.0
        self.step_count = 0
        
        # State history for model
        self.X_0T = np.zeros((8, 100))
        
        # RL Training variables
        self.training_mode = self.config_dict.get("run_training", True)
        self.episode_count = 0
        self.episode_step = 0
        self.episode_reward = 0.0
        self.max_episode_steps = self.config_dict.get("training", {}).get("steps_per_episode", 200)
        self.max_episodes = self.config_dict.get("training", {}).get("total_episodes", 1000)
        
        # Episode data for training
        self.episode_start_time = time.time()
        self.prev_state = None
        self.prev_action = None
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.safety_interventions = 0
        self.total_steps = 0
        
        # Timer for control loop (20Hz for RL training)
        self.timer = self.create_timer(0.05, self.rl_control_loop)
        
        # Visualization
        self._setup_visualization()
        
        self.get_logger().info('SafeLLMRA TD3 Turtlebot Controller Node started.')

    def _initialize_safellmra(self):
        """Initialize SafeLLMRA configuration and components."""
        try:
            # Load configuration
            config_path = r"/home/amr/SafeLLMRA/Turtlebot/my_robot_controller/config.yaml"
            self.config = Config(config_path)
            self.config_dict = self.config.parameters
            
            # Load and learn model
            self._load_and_learn_model()
            
            # Initialize SafetyController
            self.safety_controller = SafetyController(self.config_dict, self.modelSet)
            
            # Initialize Goal Manager
            self.goal_manager = GoalManager(self.config_dict)
            
            # Initialize TD3 Agent
            self._initialize_td3_agent()
            
            logger.info("SafeLLMRA TD3 components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SafeLLMRA components: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Full traceback: ", exc_info=True)
            self._create_fallback_config()

    def _initialize_td3_agent(self):
        """Initialize the TD3 reinforcement learning agent."""
        try:
            # State dimension: physical_state (8) + lidar (18) + goal_vector (4) = 30
            state_dim = 8 + 18 + 4  # physical + lidar + goal info
            action_dim = 2  # [linear_vel, angular_vel]
            
            # Action bounds from config with proper type checking
            action_constraints = self.config_dict.get("safety", {}).get("action_constraints", {}).get("turtlebot", {})
            
            # Safely extract bounds with type checking
            linear_bounds = action_constraints.get("longitudinal_velocity", [0.0, 0.25])
            angular_bounds = action_constraints.get("angular_velocity", [-0.5, 0.5])
            
            # Ensure numeric values
            if isinstance(linear_bounds, list) and len(linear_bounds) == 2:
                max_linear = float(linear_bounds[1])
                min_linear = float(linear_bounds[0])
            else:
                max_linear, min_linear = 0.25, 0.0
                
            if isinstance(angular_bounds, list) and len(angular_bounds) == 2:
                max_angular = float(angular_bounds[1])
                min_angular = float(angular_bounds[0])
            else:
                max_angular, min_angular = 0.5, -0.5
            
            max_action = np.array([max_linear, max_angular])
            min_action = np.array([min_linear, min_angular])
            
            self.td3_agent = TD3Agent(state_dim, action_dim, max_action, min_action, self.config_dict)
            
            # Try to load existing model
            model_path = "/home/amr/SafeLLMRA/Turtlebot/models/td3_safe_nav.pt"
            if os.path.exists(model_path) and not self.training_mode:
                self.td3_agent.load(model_path)
                logger.info("Loaded existing TD3 model for evaluation")
            else:
                logger.info("Starting with fresh TD3 model for training")
                
        except Exception as e:
            logger.error(f"Failed to initialize TD3 agent: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            # Create minimal fallback agent
            self._create_fallback_td3_agent()

    def _create_fallback_td3_agent(self):
        """Create a minimal fallback TD3 agent."""
        try:
            state_dim = 30
            action_dim = 2
            max_action = np.array([0.25, 0.5])
            min_action = np.array([0.0, -0.5])
            
            fallback_config = {
                "training": {
                    "learning_rate": 3e-4,
                    "batch_size": 256,
                    "replay_buffer_size": 10000
                }
            }
            
            self.td3_agent = TD3Agent(state_dim, action_dim, max_action, min_action, fallback_config)
            logger.warning("Created fallback TD3 agent")
        except Exception as e:
            logger.error(f"Failed to create fallback TD3 agent: {e}")
            self.td3_agent = None

    def _load_and_learn_model(self):
        """Load collected data and learn the dynamics model."""
        # Get dataset path from config
        dataset_path = self.config_dict.get("data_collection", {}).get("dataset_path", 
                                           "data/collected_offline_data.npz")
        
        logger.info(f"Loading dataset from: {dataset_path}")
        
        # Load dataset
        dataset_loader = DatasetLoader(dataset_path)
        
        if not dataset_loader.loaded:
            logger.warning("No dataset loaded, using fallback model")
            self._create_turtlebot_model()
            return
        
        # Print dataset statistics
        stats = dataset_loader.get_data_statistics()
        logger.info(f"Dataset statistics: {stats}")
        
        # Get training data
        n_physical_state = self.config_dict.get("agent", {}).get("physical_state_dim_turtlebot", 8)
        X_data, U_data, X_next_data = dataset_loader.get_training_data(n_physical_state)
        
        if X_data.shape[0] == 0:
            logger.warning("No valid training data found, using fallback model")
            self._create_turtlebot_model()
            return
        
        # Split data for validation (80/20 split)
        n_train = int(0.8 * X_data.shape[0])
        X_train, X_val = X_data[:n_train], X_data[n_train:]
        U_train, U_val = U_data[:n_train], U_data[n_train:]
        X_next_train, X_next_val = X_next_data[:n_train], X_next_data[n_train:]
        
        logger.info(f"Training with {n_train} samples, validating with {X_val.shape[0]} samples")
        
        # Learn model using system identification
        system_id = SystemIdentification(self.config_dict)
        self.modelSet = system_id.learn_linear_model(X_train, U_train, X_next_train)
        
        # Validate model
        if X_val.shape[0] > 0:
            validation_metrics = system_id.validate_model(X_val, U_val, X_next_val)
            logger.info(f"Model validation metrics: {validation_metrics}")
            
            # Check if model quality is reasonable
            r2_mean = validation_metrics.get("r2_mean", 0)
            if r2_mean < 0.1:  # Very poor fit
                logger.warning(f"Poor model fit (R²={r2_mean:.3f}), consider collecting more data")
            else:
                logger.info(f"Model fit quality: R²={r2_mean:.3f}")
        
        logger.info(f"Learned MatrixZonotope model: {self.modelSet}")

    def _create_turtlebot_model(self):
        """Create a simple linear model for the turtlebot dynamics."""
        # Turtlebot physical state: [x, y, theta, v, ex1, ex2, ex3, ex4]
        # Action: [v_cmd, omega_cmd]
        n_phys = 8
        n_act = 2
        dt = 0.1  # Time step
        
        # Simple kinematic model
        A = np.eye(n_phys)
        B = np.zeros((n_phys, n_act))
        
        # Linearized dynamics (assuming small angles)
        # x(t+1) = x(t) + v_cmd * dt * cos(theta) ≈ x(t) + v_cmd * dt
        # y(t+1) = y(t) + v_cmd * dt * sin(theta) ≈ y(t) + 0 (for theta ≈ 0)
        # theta(t+1) = theta(t) + omega_cmd * dt
        # v(t+1) = v_cmd
        
        B[0, 0] = dt  # dx/dv_cmd
        B[2, 1] = dt  # dtheta/domega_cmd  
        B[3, 0] = 1.0  # dv/dv_cmd
        
        # Create ModelSet as MatrixZonotope
        C_M = np.hstack([A, B])  # Shape: (8, 10)
        
        # Add small uncertainty
        G_M = 0.01 * np.ones((n_phys, n_phys + n_act))  # Small uncertainty matrix
        
        self.modelSet = MatrixZonotope(C_M, G_M)
        logger.info(f"Created turtlebot model: {self.modelSet}")

    def _create_fallback_config(self):
        """Create minimal fallback configuration if config loading fails."""
        self.config_dict = {
            "safety": {
                "planning_horizon": 3,
                "max_generators_in_reach_set": 20,
                "action_constraints": {
                    "turtlebot": {
                        "longitudinal_velocity": [0.0, 0.25],
                        "angular_velocity": [-0.5, 0.5]
                    }
                },
                "safe_state_bounds": {
                    "lower": [-100.0] * 8,
                    "upper": [100.0] * 8
                },
                "vector_noise": {
                    "z_w_generators_diag": [0.01] * 8,
                    "z_v_generators_diag": [0.005] * 8,
                    "z_av_generators_diag": [0.002] * 8
                }
            },
            "environment": {"env_type": "turtlebot"},
            "agent": {"physical_state_dim_turtlebot": 8},
            "training": {
                "steps_per_episode": 200,
                "total_episodes": 1000
            }
        }
        
        # Create simple model
        self._create_turtlebot_model()
        self.safety_controller = SafetyController(self.config_dict, self.modelSet)
        self.goal_manager = GoalManager(self.config_dict)
        self._initialize_td3_agent()
        logger.warning("Using fallback SafeLLMRA configuration")

    def _setup_visualization(self):
        """Setup visualization for reachable sets and obstacles."""
        try:
            # Set matplotlib backend for GUI display
            import matplotlib
            matplotlib.use('TkAgg')  # Use TkAgg backend for GUI
            
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)  # Non-blocking show
            
            # Initial plot setup
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            self.ax.set_title('SafeLLMRA TD3 Controller - Initializing...')
            
            plt.draw()
            plt.pause(0.1)  # Small pause to ensure window opens
            
            self.visualization_enabled = True
            logger.info("Visualization window opened successfully")
            
        except Exception as e:
            logger.warning(f"Visualization setup failed: {e}")
            logger.info("Running without visualization. Install 'python3-tk' if you want GUI: sudo apt install python3-tk")
            self.visualization_enabled = False

    def laser_scan_callback(self, msg):
        """Process LiDAR data for obstacle detection."""
        self.Lidar_readings = np.array(msg.ranges)

        # Filter invalid readings
        for i in range(len(self.Lidar_readings)):
            if self.Lidar_readings[i] == float('Inf'):
                self.Lidar_readings[i] = 3.5  # Replace infinite readings
            elif np.isnan(self.Lidar_readings[i]):
                self.Lidar_readings[i] = 0.0001  # Replace NaN values

    def odom_callback(self, msg):
        """Update robot pose and state history."""
        self.x_pos = msg.pose.pose.position.x
        self.y_pos = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        # Update state history - shift and add new state
        self.X_0T[:, :-1] = self.X_0T[:, 1:]  # Shift old values
        self.X_0T[:, -1] = [self.x_pos, self.y_pos, self.yaw, self.linear_x, 0, 0, 0, 0]

    def _lidar_to_obstacle_zonotopes(self) -> list:
        """Convert LiDAR readings to obstacle zonotopes using SafeLLMRA format."""
        obstacles = []
        max_range = 3.5
        min_obstacle_distance = 0.1
        obstacle_size = 0.15  # Conservative obstacle radius
        
        # Match the LiDAR processing from existing code
        num_rays = len(self.Lidar_readings)
        
        for i, distance in enumerate(self.Lidar_readings):
            if distance <= min_obstacle_distance or distance >= max_range:
                continue
                
            # Calculate obstacle position (matching existing angle calculation)
            relative_angle = -math.pi/2 + (math.pi * i / (num_rays - 1))
            ray_angle = self.yaw + relative_angle
            
            obs_x = self.x_pos + distance * math.cos(ray_angle)
            obs_y = self.y_pos + distance * math.sin(ray_angle)
            
            # Create zonotope for obstacle (8D to match physical state)
            center = np.zeros(8)
            center[0] = obs_x  # x position
            center[1] = obs_y  # y position
            
            # Create generators matrix for bounding box
            generators = np.zeros((8, 2))
            generators[0, 0] = obstacle_size  # x-dimension extent
            generators[1, 1] = obstacle_size  # y-dimension extent
            
            obstacle_zono = Zonotope(center, generators)
            obstacles.append(obstacle_zono)
        
        return obstacles

    def _get_state_representation(self) -> np.ndarray:
        """Get current state representation for RL agent."""
        # Physical state (8D)
        current_physical_state = self.X_0T[:, -1]
        
        # LiDAR readings (18D) - subsample or pad to fixed size
        if len(self.Lidar_readings) >= 18:
            lidar_subset = self.Lidar_readings[:18]
        else:
            lidar_subset = np.pad(self.Lidar_readings, (0, 18 - len(self.Lidar_readings)), 'constant', constant_values=3.5)
        
        # Goal vector (4D)
        robot_position = current_physical_state[:2]
        goal_vector = self.goal_manager.get_goal_vector(current_physical_state)
        
        # Combine all state components
        full_state = np.concatenate([current_physical_state, lidar_subset, goal_vector])
        
        return full_state

    def rl_control_loop(self):
        """Main RL control loop with safety filtering."""
        if self.td3_agent is None:
            # Fallback behavior if TD3 agent failed to initialize
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.cmd_vel_pub.publish(msg)
            return
            
        self.step_count += 1
        
        # Get current state
        current_state = self._get_state_representation()
        current_physical_state = self.X_0T[:, -1]
        
        # Generate goal if needed
        obstacle_zonotopes = self._lidar_to_obstacle_zonotopes()
        if self.goal_manager.current_goal is None:
            self.goal_manager.generate_new_goal(current_physical_state, obstacle_zonotopes)
            self.td3_agent.set_goal(self.goal_manager.current_goal)
        
        # Get action from TD3 agent with error handling
        try:
            add_exploration_noise = self.training_mode
            candidate_action = self.td3_agent.select_action(current_state, add_noise=add_exploration_noise)
        except Exception as e:
            logger.error(f"TD3 action selection failed: {e}")
            candidate_action = np.array([0.0, 0.0])  # Safe fallback
        
        # Apply safety filter
        is_safe, safe_plan_actions, visualizable_zonotopes = self.safety_controller.enforce_safety(
            current_state, candidate_action, obstacle_zonotopes
        )
        
        if is_safe and safe_plan_actions:
            final_action = safe_plan_actions[0]
            safety_intervention = False
        else:
            # Safety intervention - use conservative action
            final_action = np.array([0.0, 0.0])
            safety_intervention = True
            self.safety_interventions += 1
            logger.warning(f"Safety intervention #{self.safety_interventions}")
        
        # Check for collisions (simplified check)
        collision_occurred = False
        robot_pos = current_physical_state[:2]
        for obs in obstacle_zonotopes:
            if isinstance(obs.center, np.ndarray) and obs.center.shape[0] >= 2:
                obs_pos = obs.center[:2]
                distance = np.linalg.norm(robot_pos - obs_pos)
                if distance < 0.2:  # Collision threshold
                    collision_occurred = True
                    break
        
        # Check goal reached
        goal_reached = self.goal_manager.is_goal_reached(current_physical_state)
        if goal_reached:
            self.goal_manager.goal_reached()
        
        # Training logic
        if self.training_mode and self.prev_state is not None:
            # Compute reward
            reward = self.td3_agent.compute_reward(
                self.prev_state[:8], self.prev_action, current_physical_state, 
                goal_reached, collision_occurred
            )
            
            # Add bonus for safety
            if not safety_intervention:
                reward += 1.0
            
            self.episode_reward += reward
            
            # Check episode termination
            episode_done = (
                goal_reached or 
                collision_occurred or 
                self.episode_step >= self.max_episode_steps or
                self.goal_manager.is_goal_timeout(self.episode_step)
            )
            
            # Store transition
            self.td3_agent.store_transition(
                self.prev_state, self.prev_action, reward, current_state, episode_done
            )
            
            # Train agent
            if len(self.td3_agent.replay_buffer) > self.td3_agent.batch_size:
                training_info = self.td3_agent.train()
                if training_info and self.step_count % 100 == 0:
                    logger.info(f"Training info: {training_info}")
            
            # Episode management
            if episode_done:
                self._end_episode()
        
        # Update episode tracking
        if self.training_mode:
            self.episode_step += 1
            
        # Store current state/action for next iteration
        self.prev_state = current_state.copy()
        self.prev_action = candidate_action.copy()  # Store original action for learning
        
        # Publish safe command
        msg = Twist()
        msg.linear.x = float(np.clip(final_action[0], 0.0, 0.25))
        msg.angular.z = float(np.clip(final_action[1], -0.5, 0.5))
        self.cmd_vel_pub.publish(msg)
        
        # Update visualization
        if self.visualization_enabled:
            self._update_visualization(obstacle_zonotopes, visualizable_zonotopes, is_safe, goal_reached)

    def _end_episode(self):
        """Handle end of training episode."""
        self.episode_rewards.append(self.episode_reward)
        
        success_rate = self.goal_manager.get_success_rate()
        avg_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else 0
        
        logger.info(f"Episode {self.episode_count} completed:")
        logger.info(f"  Reward: {self.episode_reward:.2f}")
        logger.info(f"  Steps: {self.episode_step}")
        logger.info(f"  Success rate: {success_rate:.2f}")
        logger.info(f"  Avg reward (last 100): {avg_reward:.2f}")
        logger.info(f"  Safety interventions: {self.safety_interventions}")
        
        # Reset episode
        self.episode_count += 1
        self.episode_step = 0
        self.episode_reward = 0.0
        self.goal_manager.current_goal = None  # Force new goal generation
        
        # Save model periodically
        if self.episode_count % 50 == 0:
            model_dir = "/home/amr/SafeLLMRA/Turtlebot/models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/td3_safe_nav_episode_{self.episode_count}.pt"
            self.td3_agent.save(model_path)
        
        # Check if training is complete
        if self.episode_count >= self.max_episodes:
            logger.info("Training completed!")
            self.training_mode = False
            final_model_path = "/home/amr/SafeLLMRA/Turtlebot/models/td3_safe_nav_final.pt"
            self.td3_agent.save(final_model_path)

    def _update_visualization(self, obstacle_zonotopes, reachable_sets, is_safe, goal_reached):
        """Update the visualization with obstacles, reachable sets, and goal."""
        try:
            self.ax.cla()
            
            # Plot obstacles (red)
            for obs_zono in obstacle_zonotopes:
                if isinstance(obs_zono.center, np.ndarray) and obs_zono.center.shape[0] >= 2:
                    l_bounds, u_bounds = obs_zono.get_interval_bounds()
                    x_min, y_min = l_bounds[0], l_bounds[1]
                    x_max, y_max = u_bounds[0], u_bounds[1]
                    width, height = x_max - x_min, y_max - y_min
                    
                    rect = plt.Rectangle((x_min, y_min), width, height,
                                       facecolor='red', alpha=0.7, edgecolor='darkred')
                    self.ax.add_patch(rect)

            # Plot reachable sets
            color = 'green' if is_safe else 'red'
            for i, reach_zono in enumerate(reachable_sets):
                if isinstance(reach_zono.center, np.ndarray) and reach_zono.center.shape[0] >= 2:
                    l_bounds, u_bounds = reach_zono.get_interval_bounds()
                    x_min, y_min = l_bounds[0], l_bounds[1]
                    x_max, y_max = u_bounds[0], u_bounds[1]
                    width, height = x_max - x_min, y_max - y_min
                    
                    alpha = 0.8 - i * 0.1
                    rect = plt.Rectangle((x_min, y_min), width, height,
                                       facecolor=color, alpha=max(alpha, 0.1), 
                                       edgecolor='black', linewidth=1)
                    self.ax.add_patch(rect)

            # Plot robot position
            self.ax.plot(self.x_pos, self.y_pos, 'bo', markersize=10, label='Robot')
            
            # Plot heading direction
            head_x = self.x_pos + 0.3 * math.cos(self.yaw)
            head_y = self.y_pos + 0.3 * math.sin(self.yaw)
            self.ax.plot([self.x_pos, head_x], [self.y_pos, head_y], 'b-', linewidth=3)

            # Plot goal
            if self.goal_manager.current_goal is not None:
                goal_color = 'gold' if goal_reached else 'orange'
                self.ax.plot(self.goal_manager.current_goal[0], self.goal_manager.current_goal[1], 
                           'o', color=goal_color, markersize=15, label='Goal')
                
                # Draw line to goal
                self.ax.plot([self.x_pos, self.goal_manager.current_goal[0]], 
                           [self.y_pos, self.goal_manager.current_goal[1]], 
                           '--', color='gray', alpha=0.5)

            # Set plot properties
            self.ax.set_xlim(self.x_pos - 5, self.x_pos + 5)
            self.ax.set_ylim(self.y_pos - 5, self.y_pos + 5)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            
            status = "SAFE" if is_safe else "UNSAFE"
            mode = "TRAINING" if self.training_mode else "EVALUATION"
            title = f'SafeLLMRA TD3 Controller - {status} - {mode}'
            if self.training_mode:
                title += f' - Episode {self.episode_count} - Reward: {self.episode_reward:.1f}'
            self.ax.set_title(title)
            self.ax.legend()

            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            logger.warning(f"Visualization update failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SafeLLMRATurtlebotController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
