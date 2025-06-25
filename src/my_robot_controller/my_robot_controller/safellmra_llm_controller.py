#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import math
from collections import deque
import numpy as np
import time
import os
import json
import re
import threading
import asyncio
import websockets
import logging

import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
import random  # Add random import for fallback direction selection

# Import SafeLLMRA components
try:
    from config import Config
    from zonotope import Zonotope, MatrixZonotope
    from safety_controller import SafetyController
    from dataset_loader import DatasetLoader
    from system_identification import SystemIdentification
    from llm_agent import LLMAgent
    from goal_manager import GoalManager
except ImportError as e:
    print(f"Failed to import SafeLLMRA components: {e}")
    raise

# Set up logging with better visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)
logger = logging.getLogger(__name__)

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return math.atan2(math.sin(angle), math.cos(angle))

def angle_difference(angle1, angle2):
    """Calculate the smallest angle difference between two angles"""
    return normalize_angle(angle1 - angle2)

class SafeLLMRAController:
    def __init__(self):
        # Initialize ROS node - This will be moved to main()
        # rospy.init_node('safellmra_llm_controller', anonymous=True)
        
        # Print startup banner
        rospy.loginfo("="*60)
        rospy.loginfo("SafeLLMRA LLM Controller Starting on Jetracer")
        rospy.loginfo("="*60)
        
        # Initialize SafeLLMRA components
        self._initialize_safellmra()
        
        # ROS Publishers and Subscribers
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.plan_pub = rospy.Publisher("/llm_plan", Float64MultiArray, queue_size=10)

        # Subscriptions
        self.scan_subscription = rospy.Subscriber('/scan', LaserScan, self.laser_scan_callback)
        self.odom_subscription = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # Robot state variables
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.yaw = 0.0

        # Motion capture variables
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_qx = 0.0
        self.robot_qy = 0.0
        self.robot_qz = 0.0
        self.robot_qw = 0.0
        self.valid_mocap_received = False

        # Position scaling factor - convert from millimeters to meters
        self.position_scale_factor = 0.001  # 1 mm = 0.001 m

        # Sensor data
        self.ranges_buffer = deque(maxlen=100)
        self.Lidar_readings = np.zeros((18,))
        
        # State history for model
        self.X_0T = np.zeros((8, 100))
        
        # LLM control variables
        self.last_plan_time = rospy.Time.now()
        self.plan_timeout = rospy.Duration(0.5)  # Request new plan every second
        self.current_plan = []
        self.plan_step = 0
        self.last_step_time = rospy.Time.now()
        self.step_duration = rospy.Duration(2.0)  # Will be updated from config
        
        # Safety tracking
        self.safety_interventions = 0
        self.total_steps = 0
        
        # Visualization variables
        self.visualizable_zonotopes = []
        self.is_safe = True
        self.goal_reached = False
        
        # Control enabling flags
        self.visualization_ready = False
        self.system_ready = False

        # Visualization update control from main thread
        self.visualization_update_pending = False
        self.current_obstacle_zonotopes_for_viz = []
        
        # Setup visualization and wait for it to be ready
        self._setup_and_wait_for_visualization()
        
        # Only start other components after visualization is ready
        self._start_system_components()
        
        rospy.loginfo('SafeLLMRA LLM Controller Node started with visualization ready.')

    def _initialize_safellmra(self):
        """Initialize SafeLLMRA configuration and components."""
        rospy.loginfo("Initializing SafeLLMRA components...")
        
        try:
            # Load configuration - updated path for Jetracer
            config_path = "/home/jetson/catkin_ws/src/my_robot_controller/my_robot_controller/config.yaml"
            rospy.loginfo(f"Loading configuration from: {config_path}")
            
            if not os.path.exists(config_path):
                rospy.logwarn(f"Config file not found at {config_path}, using fallback configuration")
                self._create_fallback_config()
                return
            
            self.config = Config(config_path)
            self.config_dict = self.config.parameters
            rospy.loginfo("Configuration loaded successfully")
            
            # Load LiDAR scaling factor from config if available
            self.lidar_scaling_factor = self.config_dict.get("sensor", {}).get("lidar_scaling_factor", 1.0)
            rospy.loginfo(f"LiDAR scaling factor set to: {self.lidar_scaling_factor}")
            
            # Load position scaling factor from config if available
            self.position_scale_factor = self.config_dict.get("sensor", {}).get("position_scale_factor", 0.001)
            rospy.loginfo(f"Position scaling factor set to: {self.position_scale_factor}")
            
            # Load step duration from config
            step_duration_seconds = self.config_dict.get("planning", {}).get("step_duration", 0.3)
            self.step_duration = rospy.Duration(step_duration_seconds)
            rospy.loginfo(f"Step duration set to: {step_duration_seconds} seconds")
            
            # Load plan timeout from config
            plan_frequency = self.config_dict.get("planning", {}).get("plan_frequency", 1.0)
            self.plan_timeout = rospy.Duration(plan_frequency)
            rospy.loginfo(f"Plan frequency set to: {plan_frequency} seconds")
            
            # Load and learn model
            self._load_and_learn_model()
            
            # Initialize SafetyController
            rospy.loginfo("Initializing Safety Controller...")
            self.safety_controller = SafetyController(self.config_dict, self.modelSet)
            
            # Initialize Goal Manager
            rospy.loginfo("Initializing Goal Manager...")
            self.goal_manager = GoalManager(self.config_dict)
            
            # Initialize LLM Agent
            rospy.loginfo("Initializing LLM Agent...")
            self.llm_agent = LLMAgent(self.config_dict)
            
            rospy.loginfo("? All SafeLLMRA components initialized successfully")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize SafeLLMRA components: {e}")
            rospy.logwarn("Falling back to default configuration...")
            self._create_fallback_config()

    def _load_and_learn_model(self):
        """Load collected data and learn the dynamics model."""
        rospy.loginfo("Loading dynamics model...")
        
        dataset_path = self.config_dict.get("data_collection", {}).get("dataset_path", 
                                           "data/collected_offline_data.npz")
        
        # Make path relative to SafeLLMRA directory if not absolute
        if not os.path.isabs(dataset_path):
            base_path = "/home/amr/SafeLLMRA/Turtlebot/my_robot_controller/my_robot_controller/"
            dataset_path = os.path.join(base_path, dataset_path)
        
        rospy.loginfo(f"Dataset path: {dataset_path}")
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            rospy.logwarn(f"Dataset file not found at: {dataset_path}")
            rospy.loginfo("Using fallback turtlebot model")
            self._create_turtlebot_model()
            return
        
        # Load dataset
        rospy.loginfo("Loading dataset...")
        dataset_loader = DatasetLoader(dataset_path)
        
        if not dataset_loader.loaded:
            rospy.logwarn("Dataset could not be loaded, using fallback model")
            self._create_turtlebot_model()
            return
        
        rospy.loginfo(f"? Dataset loaded successfully from {dataset_path}")
        
        # Get training data
        n_physical_state = self.config_dict.get("agent", {}).get("physical_state_dim_turtlebot", 8)
        rospy.loginfo(f"Physical state dimension: {n_physical_state}")
        
        X_data, U_data, X_next_data = dataset_loader.get_training_data(n_physical_state)
        
        if X_data.shape[0] == 0:
            rospy.logwarn("No valid training data found in dataset")
            rospy.loginfo("Using fallback turtlebot model")
            self._create_turtlebot_model()
            return
        
        rospy.loginfo(f"Training data shapes - X: {X_data.shape}, U: {U_data.shape}, X_next: {X_next_data.shape}")
        
        # Learn model using system identification
        rospy.loginfo("Learning dynamics model from data...")
        system_id = SystemIdentification(self.config_dict)
        self.modelSet = system_id.learn_linear_model(X_data, U_data, X_next_data)
        
        rospy.loginfo(f"? Learned MatrixZonotope model successfully")
        logger.info(f"Model details: {self.modelSet}")

    def _create_turtlebot_model(self):
        """Create a simple linear model for the turtlebot dynamics."""
        rospy.loginfo("Creating fallback turtlebot dynamics model...")
        
        n_phys = 8
        n_act = 2
        dt = 0.1
        
        A = np.eye(n_phys)
        B = np.zeros((n_phys, n_act))
        
        B[0, 0] = dt  # dx/dv_cmd
        B[2, 1] = dt  # dtheta/domega_cmd  
        B[3, 0] = 1.0  # dv/dv_cmd
        
        C_M = np.hstack([A, B])
        G_M = 0.01 * np.ones((n_phys, n_phys + n_act))
        
        self.modelSet = MatrixZonotope(C_M, G_M)
        rospy.loginfo("? Fallback turtlebot model created")
        logger.info(f"Model details: {self.modelSet}")

    def _create_fallback_config(self):
        """Create minimal fallback configuration."""
        rospy.loginfo("Creating fallback configuration...")
        
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
            "llm": {
                "api_key": "your_openai_api_key",
                "model": "gpt-4",
                "server_ip": "192.168.64.147"
            },
            "planning": {
                "plan_horizon": 3,
                "plan_frequency": 1.0,
                "step_duration": 0.3
            },
            "sensor": {
                "lidar_scaling_factor": 1.5,  # Default scaling factor for LiDAR
                "position_scale_factor": 0.001  # Convert mm to meters
            }
        }
        
        # Update step duration and plan timeout from fallback config
        self.step_duration = rospy.Duration(self.config_dict["planning"]["step_duration"])
        self.plan_timeout = rospy.Duration(self.config_dict["planning"]["plan_frequency"])
        
        # Load LiDAR scaling factor from fallback config
        self.lidar_scaling_factor = self.config_dict.get("sensor", {}).get("lidar_scaling_factor", 1.5)
        rospy.loginfo(f"Using fallback LiDAR scaling factor: {self.lidar_scaling_factor}")
        
        # Load position scaling factor
        self.position_scale_factor = self.config_dict.get("sensor", {}).get("position_scale_factor", 0.001)
        rospy.loginfo(f"Using position scaling factor: {self.position_scale_factor}")
        
        rospy.loginfo("? Fallback configuration created")
        
        self._create_turtlebot_model()
        
        rospy.loginfo("Initializing components with fallback config...")
        self.safety_controller = SafetyController(self.config_dict, self.modelSet)
        self.goal_manager = GoalManager(self.config_dict)
        self.llm_agent = LLMAgent(self.config_dict)
        rospy.loginfo("? Components initialized with fallback config")

    async def connect_to_mocap(self, server_ip="192.168.64.147"):
        """Connect to motion capture system via WebSocket."""
        uri = f"ws://{server_ip}:8765"
        rospy.loginfo(f"Connecting to motion capture system at {uri}")

        while not rospy.is_shutdown():
            try:
                async with websockets.connect(uri) as websocket:
                    rospy.loginfo("Connected to motion capture system")
                    while not rospy.is_shutdown():
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)

                            if "Player" in data["objects"]:
                                # Store raw mocap data (assumed to be in millimeters)
                                self.robot_x = data["objects"]["Player"]["x"]
                                self.robot_y = data["objects"]["Player"]["y"]
                                self.robot_qx = data["objects"]["Player"]["qx"]
                                self.robot_qy = data["objects"]["Player"]["qy"]
                                self.robot_qz = data["objects"]["Player"]["qz"]
                                self.robot_qw = data["objects"]["Player"]["qw"]

                                # Update yaw from quaternion - CORRECTED CALCULATION
                                siny_cosp = 2 * (self.robot_qw * self.robot_qz + self.robot_qx * self.robot_qy)
                                cosy_cosp = 1 - 2 * (self.robot_qy * self.robot_qy + self.robot_qz * self.robot_qz)
                                raw_yaw = math.atan2(siny_cosp, cosy_cosp)
                                
                                # Apply 180-degree correction for yaw
                                corrected_yaw = raw_yaw + 0.31
                                
                                # Normalize to [-pi, pi]
                                # self.yaw = math.atan2(math.sin(corrected_yaw), math.cos(corrected_yaw))
                                self.yaw = raw_yaw - 1.87006
                                self.yaw = self.yaw * 180 / math.pi + 31
                                # convert to radians
                                self.yaw = self.yaw * math.pi / 180
                                self.yaw += math.pi / 2
                                
                                # Convert mocap position from millimeters to meters - UPDATE IMMEDIATELY
                                self.x_pos = self.robot_x * self.position_scale_factor
                                self.y_pos = self.robot_y * self.position_scale_factor
                                
                                # Log position updates for debugging lag
                                rospy.logdebug(f"MoCap update: raw=({self.robot_x:.1f},{self.robot_y:.1f}mm) -> pos=({self.x_pos:.3f},{self.y_pos:.3f}m)")

                            if "Goal" in data["objects"]:
                                # Convert goal position from millimeters to meters
                                goal_x = data["objects"]["Goal"]["x"] * self.position_scale_factor
                                goal_y = data["objects"]["Goal"]["y"] * self.position_scale_factor
                                self.goal_manager.set_goal(np.array([goal_x, goal_y]))

                            self.valid_mocap_received = True

                        except websockets.exceptions.ConnectionClosed:
                            rospy.logwarn("Motion capture connection lost, attempting to reconnect...")
                            break
            except Exception as e:
                rospy.logerr(f"Motion capture connection error: {e}")
            await asyncio.sleep(0.5)

    def start_mocap_client(self):
        """Start motion capture client in separate thread."""
        rospy.loginfo("Starting motion capture client...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server_ip = self.config_dict.get("llm", {}).get("server_ip", "192.168.64.147")
        rospy.loginfo(f"Motion capture server IP: {server_ip}")
        loop.run_until_complete(self.connect_to_mocap(server_ip))

    def laser_scan_callback(self, data):
        """Process laser scan data"""
        # Store the raw ranges and angle information
        # Extract ranges from laser scan
        ranges = np.array(data.ranges)
        angles = np.arange(
            data.angle_min,
            data.angle_min + len(ranges) * data.angle_increment,
            data.angle_increment,
        )

        # Downsample to 5 degree increments (~0.087 radians)
        downsample_factor = int(np.radians(5) / data.angle_increment)
        if downsample_factor < 1:
            downsample_factor = 1
        angles = angles[::downsample_factor]
        ranges = ranges[::downsample_factor]

        # Filter angles for rear-facing view using the specified mask
        mask = (angles >= 3 * np.pi / 4) | (angles <= -3 * np.pi / 4)

        filtered_ranges = ranges[mask]
        filtered_angles = angles[mask]
        
        # Store both ranges and their corresponding angles
        self.lidar_scan = filtered_ranges.tolist()
        self.lidar_angles = filtered_angles.tolist()

        # Process readings - replace inf/nan values and apply scaling
        for i in range(len(self.lidar_scan)):
            if np.isnan(self.lidar_scan[i]):
                self.lidar_scan[i] = -1  # Indicate no reading
            elif np.isinf(self.lidar_scan[i]):
                self.lidar_scan[i] = -1  # Indicate no reading
            else:
                # Apply scaling factor to correct distance perception
                self.lidar_scan[i] *= self.lidar_scaling_factor
                
        self.Lidar_readings = np.array(self.lidar_scan)
        rospy.logdebug(f"Processed {len(self.lidar_scan)} rear-facing LiDAR readings with scaling factor {self.lidar_scaling_factor}")

    def odom_callback(self, msg):
        """Update robot pose and state history."""
        # Only use odometry if mocap is not available
        if not self.valid_mocap_received:
            # Odometry is typically already in meters, but apply scaling if needed
            self.x_pos = msg.pose.pose.position.x
            self.y_pos = msg.pose.pose.position.y

            q = msg.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            self.yaw = math.atan2(siny_cosp, cosy_cosp)

        # Remove automatic state history update here - it's now done in control_loop
        # This prevents double updates and ensures we use the most current data
        # self.X_0T[:, :-1] = self.X_0T[:, 1:]
        # self.X_0T[:, -1] = [self.x_pos, self.y_pos, self.yaw, self.linear_x, 0, 0, 0, 0]

    def _lidar_to_obstacle_zonotopes(self):
        """Convert LiDAR readings to obstacle zonotopes using raw sensor data."""
        obstacles = []
        # Adjust max range to account for scaling factor
        max_range = 3.5 * self.lidar_scaling_factor
        min_obstacle_distance = 0.1 * self.lidar_scaling_factor
        obstacle_size = 0.15
        min_cluster_size = 2  # Require at least 2 adjacent readings for an obstacle
        max_cluster_gap = 0.3 * self.lidar_scaling_factor  # Scale the gap based on our scaling factor
        
        # Make sure we have angle data
        if not hasattr(self, 'lidar_angles') or len(self.lidar_angles) != len(self.Lidar_readings):
            rospy.logwarn("LiDAR angles not available, using estimated rear-facing angles")
            num_rays = len(self.Lidar_readings)
            # Estimate angles for rear-facing view
            self.lidar_angles = [3 * np.pi / 4 + i * (np.pi/2) / max(1, num_rays-1) for i in range(num_rays)]
        
        # First pass: identify valid readings
        valid_readings = []
        for i, distance in enumerate(self.Lidar_readings):
            # Note: distance is already scaled in laser_scan_callback
            if min_obstacle_distance < distance < max_range:
                # Use the raw sensor angle directly
                sensor_angle = self.lidar_angles[i] if i < len(self.lidar_angles) else (3 * np.pi / 4 + i * np.pi/2 / len(self.Lidar_readings))
                
                # Calculate obstacle position using raw sensor data
                obs_x = self.x_pos + distance * math.cos(sensor_angle)
                obs_y = self.y_pos + distance * math.sin(sensor_angle)
                
                valid_readings.append({
                    'index': i,
                    'distance': distance,
                    'sensor_angle': sensor_angle,
                    'x': obs_x,
                    'y': obs_y
                })
                
                rospy.logdebug(f"Reading {i}: dist={distance:.2f}, sensor_angle={math.degrees(sensor_angle):.1f}, pos=({obs_x:.2f},{obs_y:.2f})")
        
        if not valid_readings:
            return obstacles
        
        # Second pass: cluster nearby readings
        clusters = []
        current_cluster = [valid_readings[0]]
        
        for i in range(1, len(valid_readings)):
            curr_reading = valid_readings[i]
            prev_reading = valid_readings[i-1]
            
            # Calculate distance between consecutive obstacle points
            dx = curr_reading['x'] - prev_reading['x']
            dy = curr_reading['y'] - prev_reading['y']
            spatial_distance = math.sqrt(dx*dx + dy*dy)
            
            # Check if readings are part of the same obstacle
            # Consider both spatial distance and index continuity
            index_gap = curr_reading['index'] - prev_reading['index']
            
            if spatial_distance <= max_cluster_gap and index_gap <= 2:
                current_cluster.append(curr_reading)
            else:
                # Finish current cluster and start new one
                if len(current_cluster) >= min_cluster_size:
                    clusters.append(current_cluster)
                current_cluster = [curr_reading]
        
        # Don't forget the last cluster
        if len(current_cluster) >= min_cluster_size:
            clusters.append(current_cluster)
        
        # Third pass: create zonotopes from valid clusters
        for cluster in clusters:
            if len(cluster) < min_cluster_size:
                continue
                
            # Calculate cluster center
            center_x = sum(reading['x'] for reading in cluster) / len(cluster)
            center_y = sum(reading['y'] for reading in cluster) / len(cluster)
            
            # Calculate cluster size based on spread of readings
            x_coords = [reading['x'] for reading in cluster]
            y_coords = [reading['y'] for reading in cluster]
            
            x_size = max(max(x_coords) - min(x_coords), obstacle_size)
            y_size = max(max(y_coords) - min(y_coords), obstacle_size)
            
            # Limit maximum obstacle size to prevent huge obstacles from noise
            x_size = min(x_size, 0.5)
            y_size = min(y_size, 0.5)
            
            center = np.zeros(8)
            center[0] = center_x
            center[1] = center_y
            
            generators = np.zeros((8, 2))
            generators[0, 0] = x_size / 2
            generators[1, 1] = y_size / 2
            
            obstacle_zono = Zonotope(center, generators)
            obstacles.append(obstacle_zono)
            
            rospy.logdebug(f"Created obstacle from {len(cluster)} readings at ({center_x:.2f}, {center_y:.2f}) size {x_size:.2f}x{y_size:.2f}")
        
        rospy.logdebug(f"LiDAR processing: {len(valid_readings)} valid readings -> {len(clusters)} clusters -> {len(obstacles)} obstacles")
        return obstacles

    def _get_state_representation(self):
        """Get current state representation for LLM."""
        current_physical_state = self.X_0T[:, -1]
        
        # LiDAR readings (18D)
        if len(self.Lidar_readings) >= 18:
            lidar_subset = self.Lidar_readings[:18]
        else:
            lidar_subset = np.pad(self.Lidar_readings, (0, 18 - len(self.Lidar_readings)), 'constant', constant_values=3.5)
        
        # Goal vector
        robot_position = current_physical_state[:2]
        goal_vector = self.goal_manager.get_goal_vector(current_physical_state)
        
        # Modify the goal angle component to use normalized relative angle
        if hasattr(self, 'goal_manager') and self.goal_manager.current_goal is not None:
            goal_pos = self.goal_manager.current_goal
            # Calculate absolute goal angle in world frame
            abs_goal_angle = math.atan2(goal_pos[1] - self.y_pos, goal_pos[0] - self.x_pos)
            # Calculate relative angle from robot's perspective
            rel_goal_angle = normalize_angle(abs_goal_angle - self.yaw)
            # Update the goal angle in the goal vector
            if len(goal_vector) >= 4:
                goal_vector[3] = rel_goal_angle
                rospy.logdebug(f"Goal angle updated: abs={abs_goal_angle:.2f}, rel={rel_goal_angle:.2f}")
        
        full_state = np.concatenate([current_physical_state, lidar_subset, goal_vector])
        return full_state

    def _setup_and_wait_for_visualization(self):
        """Setup visualization and block until window is visible and ready."""
        rospy.loginfo("Setting up visualization system...")
        
        try:
            # Set matplotlib backend for GUI display
            matplotlib.use('TkAgg')  # Use TkAgg backend for GUI
            
            rospy.loginfo("Creating visualization window...")
            
            # Create figure and axis
            plt.ioff()  # Turn off interactive mode initially
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            
            # Initial plot setup
            self.ax.set_xlim(-5, 5) # Adjusted initial limits
            self.ax.set_ylim(-5, 5) # Adjusted initial limits
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel('X Position (m)')
            self.ax.set_ylabel('Y Position (m)')
            self.ax.set_title('SafeLLMRA LLM Controller - Initializing...')
            
            # Add coordinate axes for reference
            self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
            self.ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
            
            # Show window and force rendering
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            # Wait for window to be fully rendered
            rospy.loginfo("Waiting for visualization window to be ready...")
            for i in range(30):  # Wait up to 3 seconds
                plt.pause(0.1)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
                # Check if figure window exists and is visible
                if plt.get_fignums() and self.fig.canvas.get_tk_widget().winfo_viewable():
                    break
                if rospy.is_shutdown():
                    return
            
            self.visualization_enabled = True
            self.visualization_ready = True
            rospy.loginfo("? Visualization window is ready and visible")
            
        except Exception as e:
            rospy.logwarn(f"Visualization setup failed: {e}")
            rospy.loginfo("Running without visualization. Install 'python3-tk' if you want GUI: sudo apt install python3-tk")
            self.visualization_enabled = False
            self.visualization_ready = True  # Allow system to continue without visualization

    def _start_system_components(self):
        """Start all system components after visualization is ready."""
        if not self.visualization_ready:
            rospy.logwarn("Visualization not ready, delaying system startup...")
            return
        
        rospy.loginfo("Starting system components...")
        
        # Start motion capture client
        self.mocap_thread = threading.Thread(target=self.start_mocap_client)
        self.mocap_thread.daemon = True
        self.mocap_thread.start()
        rospy.loginfo("? Motion capture client started")
        
        # Wait a moment for initial data
        rospy.sleep(0.5)
        
        # Start control timers
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        self.plan_execution_timer = rospy.Timer(rospy.Duration(0.05), self.execute_plan_step)
        rospy.loginfo("? Control timers started")
        
        if self.visualization_enabled:
            self.visualization_timer = rospy.Timer(rospy.Duration(0.5), self.update_visualization_timer) # Changed 0.2 to 0.5
            rospy.loginfo("? Visualization timer started")
        
        self.system_ready = True
        rospy.loginfo("="*60)
        rospy.loginfo("? ALL SYSTEM COMPONENTS STARTED SUCCESSFULLY")
        rospy.loginfo("SafeLLMRA LLM Controller is now ready for operation")
        rospy.loginfo("="*60)

    def update_visualization_timer(self, event):
        """Timer callback for visualization updates."""
        if self.visualization_enabled and self.system_ready:
            # Prepare data for visualization update by the main thread
            self.current_obstacle_zonotopes_for_viz = self._lidar_to_obstacle_zonotopes()
            # self.visualizable_zonotopes, self.is_safe, self.goal_reached are updated
            # directly as instance attributes by other parts of the code (e.g., control_loop)
            self.visualization_update_pending = True

    def execute_plan_step(self, event):
        """Execute steps from the current plan at appropriate intervals."""
        if not self.system_ready:
            return
            
        # Check if goal is reached and stop robot
        if self.goal_reached:
            if self.current_plan:  # Clear any existing plan
                self.current_plan = []
                self.plan_step = 0
            # Send stop command
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            self.linear_x = 0.0
            self.angular_z = 0.0
            rospy.loginfo_throttle(5, "Goal reached - robot stopped")
            return
            
        if not self.current_plan:
            return

        current_time = rospy.Time.now()
        time_since_last_step = current_time - self.last_step_time

        if time_since_last_step >= self.step_duration:
            # Check if there's a step to execute at current plan_step
            if self.plan_step < len(self.current_plan):
                step = self.current_plan[self.plan_step] # Use current plan_step
                linear_vel = float(step.get("linear_velocity", 0.0))
                angular_vel = float(step.get("angular_velocity", 0.0))

                # Apply velocity limits
                linear_vel = max(-0.5, min(0.6, linear_vel))
                angular_vel = max(-0.6, min(0.6, angular_vel))

                # Store current velocities for visualization
                self.linear_x = linear_vel
                self.angular_z = angular_vel

                # Create and publish command
                cmd = Twist()
                cmd.linear.x = linear_vel
                cmd.angular.z = angular_vel
                self.cmd_vel_pub.publish(cmd)

                self.last_step_time = current_time # Mark time of this step execution
                self.plan_step += 1 # Increment for the *next* potential step
            
            # After attempting to execute a step (or if plan_step was already at/beyond len),
            # check if the plan is now completed.
            # The `and self.current_plan` ensures we only act if there was a plan to complete.
            if self.plan_step >= len(self.current_plan) and self.current_plan:
                rospy.loginfo("LLM Plan execution complete.")
                self.current_plan = [] # Clear the plan
                # Stop robot when plan is complete
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd)
                self.linear_x = 0.0
                self.angular_z = 0.0

    def control_loop(self, event):
        """Main control loop that requests LLM control at regular intervals."""
        if not self.system_ready:
            return
            
        if not self.valid_mocap_received:
            rospy.logwarn_throttle(10, "Waiting for valid mocap data...")
            return

        self.total_steps += 1
        
        # Update state history with MOST CURRENT position data immediately
        self.X_0T[:, :-1] = self.X_0T[:, 1:]
        self.X_0T[:, -1] = [self.x_pos, self.y_pos, self.yaw, self.linear_x, 0, 0, 0, 0]
        
        # Get current state representation using the just-updated state
        current_state = self._get_state_representation()
        current_physical_state = self.X_0T[:, -1]
        
        # Generate goal if needed
        obstacle_zonotopes = self._lidar_to_obstacle_zonotopes()
        if self.goal_manager.current_goal is None:
            self.goal_manager.generate_new_goal(current_physical_state, obstacle_zonotopes)

        # Check if goal is reached
        self.goal_reached = self.goal_manager.is_goal_reached(current_physical_state)
        if self.goal_reached:
            # Stop the robot immediately when goal is reached
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            self.linear_x = 0.0
            self.angular_z = 0.0
            
            # Clear any current plan
            if self.current_plan:
                self.current_plan = []
                self.plan_step = 0
                rospy.loginfo("Goal reached - stopping robot and clearing plan")
            
            self.goal_manager.goal_reached()
            rospy.loginfo_throttle(2, "Goal reached! Robot stopped.")
            return  # Exit early, don't request new plans

        # ALWAYS ensure we have reachable sets for visualization - USE CURRENT POSITION
        if not hasattr(self, 'visualizable_zonotopes') or not self.visualizable_zonotopes:
            # Create initial reachable set around CURRENT position (not buffered)
            current_center = np.zeros(8)
            current_center[0] = self.x_pos  # Use current position directly
            current_center[1] = self.y_pos  # Use current position directly
            current_generators = np.zeros((8, 4))  # More generators for better visualization
            current_generators[0, 0] = 0.2  # x range
            current_generators[1, 1] = 0.2  # y range
            current_generators[0, 2] = 0.1  # additional x uncertainty
            current_generators[1, 3] = 0.1  # additional y uncertainty
            initial_zonotope = Zonotope(current_center, current_generators)
            self.visualizable_zonotopes = [initial_zonotope]
            self.is_safe = True
            rospy.loginfo("Created initial reachable set for visualization")

        # Check if we need new plan
        current_time = rospy.Time.now()
        time_since_last_plan = current_time - self.last_plan_time
        
        if (not self.current_plan or self.plan_step >= len(self.current_plan) or 
            time_since_last_plan >= self.plan_timeout):
            
            # Get LLM plan
            candidate_plan = self.llm_agent.get_plan(current_state, obstacle_zonotopes, self.yaw)
            
            if candidate_plan:
                # Apply safety filter to first action using CURRENT state
                first_action = np.array([candidate_plan[0]["linear_velocity"], 
                                       candidate_plan[0]["angular_velocity"]])
                
                rospy.loginfo(f"Calling safety controller with current state: pos=({self.x_pos:.3f},{self.y_pos:.3f}), action: {first_action}")
                is_safe, safe_plan_actions, visualizable_zonotopes = self.safety_controller.enforce_safety(
                    current_state, first_action, obstacle_zonotopes
                )
                
                # Store zonotopes for visualization - ALWAYS ensure we have something
                self.is_safe = is_safe
                if visualizable_zonotopes and len(visualizable_zonotopes) > 0:
                    self.visualizable_zonotopes = visualizable_zonotopes
                    rospy.loginfo(f"Received {len(visualizable_zonotopes)} reachable sets from safety controller at pos=({self.x_pos:.3f},{self.y_pos:.3f})")
                else:
                    # Keep existing zonotopes or create new ones CENTERED ON CURRENT POSITION
                    rospy.logwarn("No reachable sets from safety controller, creating fallback")
                    current_center = np.zeros(8)
                    current_center[0] = self.x_pos  # Use current position
                    current_center[1] = self.y_pos  # Use current position
                    current_generators = np.zeros((8, 3))
                    current_generators[0, 0] = 0.15
                    current_generators[1, 1] = 0.15
                    current_generators[0, 2] = 0.1
                    current_generators[1, 2] = 0.1
                    fallback_zonotope = Zonotope(current_center, current_generators)
                    if not hasattr(self, 'visualizable_zonotopes') or not self.visualizable_zonotopes:
                        self.visualizable_zonotopes = [fallback_zonotope]
                
                if is_safe and safe_plan_actions:
                    # Use safe plan
                    self.current_plan = []
                    for action in safe_plan_actions:
                        self.current_plan.append({
                            "linear_velocity": action[0],
                            "angular_velocity": action[1]
                        })
                    self.plan_step = 0
                    self.last_step_time = current_time
                    rospy.loginfo(f"? New safe plan generated with {len(self.current_plan)} steps")
                else:
                    # Safety intervention with directional avoidance
                    self.safety_interventions += 1
                    
                    # Analyze obstacle direction and generate appropriate avoidance action
                    obstacle_direction = self._analyze_obstacle_direction(obstacle_zonotopes)
                    safe_linear_vel, safe_angular_vel = self._generate_avoidance_action(obstacle_direction)
                    
                    cmd = Twist()
                    cmd.linear.x = safe_linear_vel
                    cmd.angular.z = safe_angular_vel
                    self.cmd_vel_pub.publish(cmd)
                    self.linear_x = safe_linear_vel
                    self.angular_z = safe_angular_vel
                    rospy.logwarn(f"? Safety intervention #{self.safety_interventions} - Directional avoidance: {obstacle_direction}")
                
                self.last_plan_time = current_time
                
                # Publish plan for debugging
                plan_msg = Float64MultiArray()
                plan_data = []
                for step in candidate_plan:
                    plan_data.extend([step["linear_velocity"], step["angular_velocity"]])
                plan_msg.data = plan_data
                self.plan_pub.publish(plan_msg)
                
                # Queue visualization update
                if self.visualization_enabled:
                    self.current_obstacle_zonotopes_for_viz = obstacle_zonotopes
                    self.visualization_update_pending = True
            else:
                # No plan from LLM - ensure we have reachable sets for visualization at CURRENT position
                self.is_safe = False
                current_center = np.zeros(8)
                current_center[0] = self.x_pos  # Use current position
                current_center[1] = self.y_pos  # Use current position
                current_generators = np.zeros((8, 2))
                current_generators[0, 0] = 0.1  # Small x range
                current_generators[1, 1] = 0.1  # Small y range
                fallback_zonotope = Zonotope(current_center, current_generators)
                if not hasattr(self, 'visualizable_zonotopes') or not self.visualizable_zonotopes:
                    self.visualizable_zonotopes = [fallback_zonotope]
                rospy.logwarn("? No plan from LLM")

    def _update_visualization(self):
        """Update the visualization with obstacles, reachable sets, and goal."""
        # Fetch data from instance attributes
        obstacle_zonotopes = self.current_obstacle_zonotopes_for_viz
        reachable_sets = self.visualizable_zonotopes
        is_safe = self.is_safe
        goal_reached = self.goal_reached

        if not self.visualization_ready or not self.visualization_enabled:
            return
            
        try:
            # Clear previous plot
            self.ax.cla()
            
            # Set up plot properties - journal quality settings
            view_range = 2.0
            if hasattr(self, 'x_pos') and hasattr(self, 'y_pos') and np.isfinite(self.x_pos) and np.isfinite(self.y_pos):
                self.ax.set_xlim(self.x_pos - view_range, self.x_pos + view_range)
                self.ax.set_ylim(self.y_pos - view_range, self.y_pos + view_range)
            else:
                self.ax.set_xlim(-view_range, view_range)
                self.ax.set_ylim(-view_range, view_range)
            
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3, linewidth=0.5)
            self.ax.set_xlabel('X Position (m)', fontsize=12)
            self.ax.set_ylabel('Y Position (m)', fontsize=12)
            
            # Check robot data availability
            robot_data_available = False
            if hasattr(self, 'x_pos') and hasattr(self, 'y_pos') and \
               np.isfinite(self.x_pos) and np.isfinite(self.y_pos) and \
               (self.valid_mocap_received or (abs(self.x_pos) > 0.001 or abs(self.y_pos) > 0.001)):
                robot_data_available = True
            
            if not robot_data_available:
                self.ax.text(0.5, 0.5, 'Waiting for robot data...', 
                           transform=self.ax.transAxes, ha='center', va='center', 
                           fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
                self.ax.set_title('SafeLLMRA Controller - Initializing', fontsize=14)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                return

            # Plot obstacles (red rectangles)
            obstacle_count = 0
            for i, obs_zono in enumerate(obstacle_zonotopes):
                if isinstance(obs_zono.center, np.ndarray) and obs_zono.center.shape[0] >= 2:
                    try:
                        l_bounds, u_bounds = obs_zono.get_interval_bounds()
                        x_min, y_min = l_bounds[0], l_bounds[1]
                        x_max, y_max = u_bounds[0], u_bounds[1]
                        width, height = x_max - x_min, y_max - y_min
                        
                        if width > 0 and height > 0:
                            rect = Rectangle((x_min, y_min), width, height,
                                           facecolor='red', alpha=0.7, edgecolor='darkred', linewidth=1.5)
                            self.ax.add_patch(rect)
                            obstacle_count += 1
                    except Exception:
                        pass

            # ALWAYS plot reachable sets - this is critical for journal publication
            reach_color = 'lightgreen' if is_safe else 'orange'
            edge_color = 'darkgreen' if is_safe else 'darkorange'
            reach_count = 0
            
            # FORCE creation of reachable sets if none exist
            if not reachable_sets or len(reachable_sets) == 0:
                rospy.logwarn("Creating emergency reachable set for visualization")
                current_center = np.zeros(8)
                current_center[0] = self.x_pos
                current_center[1] = self.y_pos
                current_generators = np.zeros((8, 3))
                current_generators[0, 0] = 0.2  # x range
                current_generators[1, 1] = 0.2  # y range
                current_generators[0, 2] = 0.1  # additional uncertainty
                current_generators[1, 2] = 0.1
                emergency_zonotope = Zonotope(current_center, current_generators)
                reachable_sets = [emergency_zonotope]
                # Update the instance variable too
                self.visualizable_zonotopes = reachable_sets
            
            rospy.loginfo(f"Plotting {len(reachable_sets)} reachable sets")
            
            for i, reach_zono in enumerate(reachable_sets):
                try:
                    if hasattr(reach_zono, 'center') and isinstance(reach_zono.center, np.ndarray) and reach_zono.center.shape[0] >= 2:
                        l_bounds, u_bounds = reach_zono.get_interval_bounds()
                        if len(l_bounds) >= 2 and len(u_bounds) >= 2:
                            x_min, y_min = l_bounds[0], l_bounds[1]
                            x_max, y_max = u_bounds[0], u_bounds[1]
                            width, height = x_max - x_min, y_max - y_min
                            
                            # Calculate the center of the reachable set
                            center_x = (x_min + x_max) / 2
                            center_y = (y_min + y_max) / 2
                            
                            # Calculate displacement from robot position
                            dx = center_x - self.x_pos
                            dy = center_y - self.y_pos
                            
                            # Flip the displacement direction (to opposite side)
                            flipped_center_x = self.x_pos - dx
                            flipped_center_y = self.y_pos - dy
                            
                            # Recalculate the bounds based on the flipped center
                            flipped_x_min = flipped_center_x - width/2
                            flipped_x_max = flipped_center_x + width/2
                            flipped_y_min = flipped_center_y - height/2
                            flipped_y_max = flipped_center_y + height/2
                            
                            rospy.loginfo(f"Reachable set {i}: original=({center_x:.3f},{center_y:.3f}), flipped=({flipped_center_x:.3f},{flipped_center_y:.3f}), size=({width:.3f},{height:.3f})")
                            
                            if width > 0.001 and height > 0.001:  # Minimum size threshold
                                # Use decreasing alpha for future reachable sets
                                alpha_val = max(0.4, 0.8 - i * 0.1)
                                
                                # Use flipped coordinates for the rectangle
                                rect = Rectangle((flipped_x_min, flipped_y_min), width, height,
                                               facecolor=reach_color, alpha=alpha_val, 
                                               edgecolor=edge_color, linewidth=2)
                                self.ax.add_patch(rect)
                                reach_count += 1
                                rospy.loginfo(f"? Plotted reachable set {i}")
                            else:
                                rospy.logwarn(f"Reachable set {i} too small: {width}x{height}")
                        else:
                            rospy.logwarn(f"Reachable set {i} bounds issue: {len(l_bounds)} lower, {len(u_bounds)} upper")
                    else:
                        rospy.logwarn(f"Reachable set {i} invalid center")
                except Exception as e:
                    rospy.logwarn(f"Failed to plot reachable set {i}: {e}")

            rospy.loginfo(f"Successfully plotted {reach_count} reachable sets")

            # Plot robot position and heading
            # Robot body (blue circle)
            self.ax.plot(self.x_pos, self.y_pos, 'o', color='blue', markersize=12, 
                        label='Robot', markeredgecolor='navy', markeredgewidth=2)
            
            
            # Plot goal position
            goal_plotted = False
            if hasattr(self, 'goal_manager') and self.goal_manager.current_goal is not None:
                goal_pos_arr = self.goal_manager.current_goal
                goal_color = 'gold' if goal_reached else 'orange'
                self.ax.plot(goal_pos_arr[0], goal_pos_arr[1], 
                           '*', color=goal_color, markersize=20, label='Goal',
                           markeredgecolor='black', markeredgewidth=2)
                
                

            

            # Clean title for publication
            status_str = "Safe" if is_safe else "Unsafe"
            self.ax.set_title(f'SafeLLMRA Controller - Status: {status_str}', fontsize=14, fontweight='bold')
            
            # Add clean legend
            handles, labels = self.ax.get_legend_handles_labels()
            # Always add reachable set to legend manually since it's a patch
            from matplotlib.patches import Patch
            reach_patch = Patch(color=reach_color, alpha=0.6, label='Reachable Set')
            handles.append(reach_patch)
            if obstacle_count > 0:
                obstacle_patch = Patch(color='red', alpha=0.7, label='Obstacles')
                handles.append(obstacle_patch)
            self.ax.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.9)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            rospy.logerr(f"Visualization error: {e}")
            try:
                self.ax.cla()
                self.ax.text(0.5, 0.5, f'Visualization Error\nSystem Running', 
                           ha='center', va='center', transform=self.ax.transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except Exception:
                pass

    def _analyze_obstacle_direction(self, obstacle_zonotopes):
        """Sophisticated analysis of obstacle directions using weighted proximity."""
        if not obstacle_zonotopes:
            return None

        # Weighted sums for left/right/front
        left_weight = 0.0
        right_weight = 0.0
        front_weight = 0.0
        total_weight = 0.0

        # For debugging: count obstacles per region
        left_count = 0
        right_count = 0
        front_count = 0

        # Parameters
        front_angle = math.radians(60)  # +/- 30 deg
        side_angle = math.radians(100)  # +/- 50 deg for sides
        min_distance = 0.05  # Ignore obstacles closer than this (sensor noise)
        max_distance = 2.0   # Ignore obstacles farther than this

        for obs_zono in obstacle_zonotopes:
            if isinstance(obs_zono.center, np.ndarray) and obs_zono.center.shape[0] >= 2:
                obs_x = obs_zono.center[0]
                obs_y = obs_zono.center[1]
                rel_x = obs_x - self.x_pos
                rel_y = obs_y - self.y_pos

                # Robot frame
                cos_yaw = math.cos(self.yaw)
                sin_yaw = math.sin(self.yaw)
                local_x = rel_x * cos_yaw + rel_y * sin_yaw
                local_y = -rel_x * sin_yaw + rel_y * cos_yaw

                distance = math.hypot(local_x, local_y)
                if distance < min_distance or distance > max_distance:
                    continue

                angle = math.atan2(local_y, local_x)  # In robot frame

                # Weight: closer obstacles are more important (inverse distance)
                weight = 1.0 / (distance + 1e-2)
                total_weight += weight

                # Classify region by angle
                if abs(angle) < front_angle / 2:
                    front_weight += weight
                    front_count += 1
                elif angle > 0 and abs(angle) < side_angle:
                    left_weight += weight
                    left_count += 1
                elif angle < 0 and abs(angle) < side_angle:
                    right_weight += weight
                    right_count += 1
                # else: ignore obstacles behind

        rospy.loginfo(f"Weighted obstacle analysis: front={front_weight:.2f}({front_count}), left={left_weight:.2f}({left_count}), right={right_weight:.2f}({right_count})")

        # Decision logic: prioritize front, then compare sides
        if front_weight > 0.1:
            if left_weight <= right_weight:
                return "avoid_left"
            else:
                return "avoid_right"
        elif left_weight > 0.05 and right_weight < 0.05:
            return "avoid_right"
        elif right_weight > 0.05 and left_weight < 0.05:
            return "avoid_left"
        elif left_weight > 0.05 and right_weight > 0.05:
            if left_weight <= right_weight:
                return "avoid_left"
            else:
                return "avoid_right"
        else:
            return "random"  # No significant obstacles detected

    def _generate_avoidance_action(self, direction):
        """Generate an avoidance action based on the specified direction."""
        # Conservative velocities for safety intervention
        safe_linear_vel = 0.12  # Very slow forward movement
        
        if direction == "avoid_left":
            safe_angular_vel = 0.3   # Turn left
            rospy.loginfo("Safety intervention: Turning left to avoid obstacles on right")
        elif direction == "avoid_right":
            safe_angular_vel = -0.3  # Turn right
            rospy.loginfo("Safety intervention: Turning right to avoid obstacles on left")
        else:  # random or fallback
            safe_angular_vel = -0.3  # Randomly choose left or right
            direction_str = "left" if safe_angular_vel > 0 else "right"
            rospy.loginfo(f"Safety intervention: Randomly turning {direction_str}")
        
        return safe_linear_vel, safe_angular_vel

def main():
    """Main function to start the SafeLLMRA LLM Controller."""
    rospy.init_node('safellmra_llm_controller', anonymous=True) # Initialize ROS node here
    
    controller = SafeLLMRAController()
    
    if controller.visualization_enabled:
        rospy.loginfo("Visualization enabled. Starting main loop with plt.pause().")
        while not rospy.is_shutdown():
            if controller.visualization_update_pending:
                try:
                    controller._update_visualization()
                except Exception as e:
                    rospy.logerr(f"Error in scheduled visualization update: {e}", exc_info=True)
                finally:
                    # Always reset the flag, even if update failed, to avoid spamming errors
                    # for the same failed update.
                    controller.visualization_update_pending = False
            
            try:
                # plt.pause processes GUI events, updates plot, and allows ROS to spin.
                # A short pause is needed to keep GUI responsive.
                plt.pause(0.05)  # Adjust timing as needed (e.g., 20 FPS)
            except Exception as e:
                # Handle cases where the plot window might be closed by the user or other Tkinter errors
                # Common errors include TclError or errors indicating the canvas is gone.
                if 'TclError' in str(e) or \
                   (hasattr(e, 'message') and isinstance(e.message, str) and 'main thread is not in main loop' in e.message.lower()) or \
                   'cannot be None' in str(e) or \
                   'application has been destroyed' in str(e): # More robust check for closed window
                    rospy.logwarn(f"Matplotlib window closed or error: {e}. Stopping visualization loop.")
                    break 
                else:
                    rospy.logerr(f"Unhandled error during plt.pause: {e}", exc_info=True)
                    break # Exit loop on other critical plt.pause errors
        rospy.loginfo("Visualization loop ended.")
    else:
        rospy.loginfo("Visualization not enabled. Calling rospy.spin().")
        rospy.spin()
    
    rospy.loginfo("Controller shutdown.")

if __name__ == '__main__':
    main()