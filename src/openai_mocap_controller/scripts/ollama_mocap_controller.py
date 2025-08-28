#!/usr/bin/env python3

import rospy
import asyncio
import websockets
import json
import math
import threading
import time
import numpy as np
import requests
from datetime import datetime
from collections import deque

# ROS imports
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

class OllamaMocapController:
    def __init__(self):
        rospy.init_node('ollama_mocap_controller', anonymous=True)
        
        rospy.loginfo("="*60)
        rospy.loginfo("Ollama Mocap Controller Starting")
        rospy.loginfo("="*60)
        
        # Load parameters
        self.load_parameters()
        
        # Initialize Ollama client
        self.setup_ollama()
        
        # Robot state variables
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.robot_yaw = 0.0
        self.robot_qx = 0.0
        self.robot_qy = 0.0
        self.robot_qz = 0.0
        self.robot_qw = 0.0
        self.valid_mocap_received = False
        
        # Goal position (always 0,0,0)
        self.goal_x = 0.0
        self.goal_y = 0.0
        
        # LiDAR data
        self.lidar_ranges = []
        self.lidar_angles = []
        
        # Control variables
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.last_decision_time = time.time()
        
        # Performance tracking
        self.performance_data = []
        self.test_start_time = time.time()
        self.distance_traveled = 0.0
        self.last_position = None
        self.goal_reached = False
        
        # Chain of thought tracking
        self.action_history = []
        self.max_history_length = 5
        
        # API failure tracking
        self.api_failures = 0
        self.api_successes = 0
        
        # Response time tracking
        self.response_times = []  # List to store all response times
        self.total_response_time = 0.0
        self.min_response_time = float('inf')
        self.max_response_time = 0.0
        
        # ROS Publishers and Subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.performance_pub = rospy.Publisher('/performance_data', String, queue_size=10)
        
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # Start mocap client
        self.start_mocap_client()
        
        # Start control loop
        self.control_timer = rospy.Timer(rospy.Duration(1.0/self.control_frequency), self.control_loop)
        
        rospy.loginfo("Ollama Mocap Controller initialized successfully")
        
    def load_parameters(self):
        """Load ROS parameters"""
        self.ollama_server_ip = rospy.get_param('~ollama_server_ip', '192.168.137.54')
        self.ollama_server_port = rospy.get_param('~ollama_server_port', 11434)
        self.ollama_model = rospy.get_param('~ollama_model', 'llama3:latest')
        self.mocap_server_ip = rospy.get_param('~mocap_server_ip', '192.168.137.54')
        self.mocap_server_port = rospy.get_param('~mocap_server_port', 8765)
        self.control_frequency = rospy.get_param('~control_frequency', 10.0)
        self.decision_frequency = rospy.get_param('~decision_frequency', 2.0)
        self.position_scale_factor = rospy.get_param('~position_scale_factor', 0.001)
        self.max_linear_velocity = rospy.get_param('~max_linear_velocity', 0.3)
        self.max_angular_velocity = rospy.get_param('~max_angular_velocity', 0.8)
        
        # Single model configuration
        self.log_performance = rospy.get_param('~log_performance', True)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.1)  # meters
        self.request_timeout = rospy.get_param('~request_timeout', 10.0)  # seconds
        
        rospy.loginfo(f"Using Ollama server: {self.ollama_server_ip}:{self.ollama_server_port}")
        rospy.loginfo(f"Using Ollama model: {self.ollama_model}")
        rospy.loginfo(f"Goal tolerance: {self.goal_tolerance} meters")
        
    def setup_ollama(self):
        """Initialize Ollama client and test connection"""
        self.ollama_url = f"http://{self.ollama_server_ip}:{self.ollama_server_port}"
        
        # Test connection to Ollama server
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                rospy.loginfo(f"Connected to Ollama server. Available models: {model_names}")
                
                # Check if requested model is available
                if not any(self.ollama_model in name for name in model_names):
                    rospy.logwarn(f"Model '{self.ollama_model}' not found. Available: {model_names}")
                    rospy.logwarn("Continuing anyway - Ollama will try to pull the model if needed")
            else:
                rospy.logwarn(f"Ollama server responded with status {response.status_code}")
        except Exception as e:
            rospy.logerr(f"Failed to connect to Ollama server at {self.ollama_url}: {e}")
            rospy.logerr("Make sure Ollama is running and accessible")
            
        rospy.loginfo(f"Ollama client initialized with model: {self.ollama_model}")
        
    def start_mocap_client(self):
        """Start motion capture client in separate thread"""
        self.mocap_thread = threading.Thread(target=self.run_mocap_client)
        self.mocap_thread.daemon = True
        self.mocap_thread.start()
        rospy.loginfo("Motion capture client started")
        
    def run_mocap_client(self):
        """Run the mocap client asyncio loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.connect_to_mocap())        

    async def connect_to_mocap(self):
        """Connect to motion capture system via WebSocket"""
        uri = f"ws://{self.mocap_server_ip}:{self.mocap_server_port}"
        rospy.loginfo(f"Connecting to motion capture system at {uri}")
        
        while not rospy.is_shutdown():
            try:
                async with websockets.connect(uri) as websocket:
                    rospy.loginfo("Connected to motion capture system")
                    while not rospy.is_shutdown():
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            
                            if "objects" in data and "Player" in data["objects"]:
                                player_data = data["objects"]["Player"]
                                
                                # Store raw mocap data (in millimeters)
                                self.robot_x = player_data["x"] * self.position_scale_factor
                                self.robot_y = player_data["y"] * self.position_scale_factor
                                self.robot_z = player_data["z"] * self.position_scale_factor
                                
                                # Store quaternion
                                self.robot_qx = player_data["qx"]
                                self.robot_qy = player_data["qy"]
                                self.robot_qz = player_data["qz"]
                                self.robot_qw = player_data["qw"]
                                
                                # Calculate yaw from quaternion
                                self.robot_yaw = self.quaternion_to_yaw(
                                    self.robot_qx, self.robot_qy, self.robot_qz, self.robot_qw
                                )
                                
                                self.valid_mocap_received = True
                                
                                # Minimal mocap logging
                                distance_to_goal = math.sqrt(self.robot_x**2 + self.robot_y**2)
                                rospy.logdebug(f"Mocap: pos=({self.robot_x:.3f}, {self.robot_y:.3f}), dist_to_goal={distance_to_goal:.3f}m")
                                
                                # Update distance traveled for performance tracking
                                if self.last_position is not None:
                                    dx = self.robot_x - self.last_position[0]
                                    dy = self.robot_y - self.last_position[1]
                                    self.distance_traveled += math.sqrt(dx*dx + dy*dy)
                                
                                self.last_position = (self.robot_x, self.robot_y)
                                
                        except websockets.exceptions.ConnectionClosed:
                            rospy.logwarn("Motion capture connection lost, attempting to reconnect...")
                            break
                        except json.JSONDecodeError as e:
                            rospy.logwarn(f"Failed to parse mocap data: {e}")
                        except Exception as e:
                            rospy.logwarn(f"Error processing mocap data: {e}")
                            
            except Exception as e:
                rospy.logerr(f"Motion capture connection error: {e}")
                await asyncio.sleep(1.0)
                
    def quaternion_to_yaw(self, qx, qy, qz, qw):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Debug: Log quaternion and resulting yaw
        rospy.logdebug(f"🔍 QUATERNION DEBUG: q=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f}) → yaw={math.degrees(yaw):.1f}°")
        
        return yaw
        
    def lidar_callback(self, data):
        """Process LiDAR scan data"""
        self.lidar_ranges = list(data.ranges)
        
        # Calculate angles
        self.lidar_angles = []
        for i in range(len(data.ranges)):
            angle = data.angle_min + i * data.angle_increment
            self.lidar_angles.append(angle)
            
        # Clean up invalid readings
        for i in range(len(self.lidar_ranges)):
            if math.isnan(self.lidar_ranges[i]) or math.isinf(self.lidar_ranges[i]):
                self.lidar_ranges[i] = data.range_max
                
    def odom_callback(self, msg):
        """Odometry callback - used as fallback if mocap not available"""
        if not self.valid_mocap_received:
            self.robot_x = msg.pose.pose.position.x
            self.robot_y = msg.pose.pose.position.y
            
            # Convert quaternion to yaw
            q = msg.pose.pose.orientation
            self.robot_yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
            
    def calculate_navigation_parameters(self):
        """Calculate deterministic navigation parameters using formulas"""
        # Basic calculations
        distance_to_goal = math.sqrt(self.robot_x**2 + self.robot_y**2)
        
        # COORDINATE SYSTEM FIX: Try different angle calculation
        # Standard: angle_to_goal = math.atan2(-self.robot_y, -self.robot_x)
        # Alternative: Maybe the coordinate system is rotated 180°
        angle_to_goal = math.atan2(-self.robot_y, -self.robot_x)  # Angle to reach (0,0)
        
        # EXPERIMENTAL FIX: Based on the logs showing robot moving away from goal
        # The coordinate system might be flipped. Let's try the opposite direction:
        angle_to_goal = math.atan2(self.robot_y, self.robot_x)  # FLIPPED: Try opposite direction
        
        # Normalize angle relative to robot's heading
        relative_goal_angle = angle_to_goal - self.robot_yaw
        while relative_goal_angle > math.pi:
            relative_goal_angle -= 2 * math.pi
        while relative_goal_angle < -math.pi:
            relative_goal_angle += 2 * math.pi
            
        # Formula-based calculations
        calculations = {
            'distance_to_goal': distance_to_goal,
            'angle_to_goal_world': angle_to_goal,
            'relative_goal_angle': relative_goal_angle,
            'relative_goal_angle_deg': math.degrees(relative_goal_angle),
            'robot_heading_deg': math.degrees(self.robot_yaw)
        }
        
        # Debug logging for angle calculations
        rospy.logdebug(f"🔍 ANGLE DEBUG: pos=({self.robot_x:.3f},{self.robot_y:.3f}), "
                      f"robot_yaw={math.degrees(self.robot_yaw):.1f}°, "
                      f"angle_to_goal={math.degrees(angle_to_goal):.1f}°, "
                      f"relative_angle={math.degrees(relative_goal_angle):.1f}°")
        
        # Deterministic linear velocity calculation
        if distance_to_goal < self.goal_tolerance:
            calculations['recommended_linear'] = 0.0
            calculations['linear_reasoning'] = "Goal reached - stop"
        elif distance_to_goal < 0.2:
            calculations['recommended_linear'] = 0.1
            calculations['linear_reasoning'] = "Very close - slow approach"
        elif distance_to_goal < 0.5:
            calculations['recommended_linear'] = 0.15
            calculations['linear_reasoning'] = "Close - moderate speed"
        elif distance_to_goal < 1.0:
            calculations['recommended_linear'] = 0.25
            calculations['linear_reasoning'] = "Medium distance - good speed"
        else:
            calculations['recommended_linear'] = 0.3
            calculations['linear_reasoning'] = "Far - maximum speed"
            
        # Deterministic angular velocity calculation
        abs_angle_error = abs(relative_goal_angle)
        if abs_angle_error < math.radians(5):  # < 5 degrees
            calculations['recommended_angular'] = 0.0
            calculations['angular_reasoning'] = "Well aligned - no turn needed"
        elif abs_angle_error < math.radians(15):  # < 15 degrees
            angular_magnitude = 0.2
            calculations['recommended_angular'] = angular_magnitude if relative_goal_angle > 0 else -angular_magnitude
            calculations['angular_reasoning'] = f"Small correction - gentle turn"
        elif abs_angle_error < math.radians(45):  # < 45 degrees
            angular_magnitude = 0.4
            calculations['recommended_angular'] = angular_magnitude if relative_goal_angle > 0 else -angular_magnitude
            calculations['angular_reasoning'] = f"Moderate turn needed"
        else:  # > 45 degrees
            angular_magnitude = 0.6
            calculations['recommended_angular'] = angular_magnitude if relative_goal_angle > 0 else -angular_magnitude
            calculations['angular_reasoning'] = f"Large turn needed"
            
        # Adjust linear velocity based on turning requirement
        if abs_angle_error > math.radians(30):  # If turning > 30 degrees, reduce forward speed
            calculations['recommended_linear'] *= 0.7
            calculations['linear_reasoning'] += " (reduced due to large turn)"
            
        return calculations

    def get_robot_state_description(self):
        """Generate a detailed text description with formula-based calculations"""
        # Get deterministic calculations
        calc = self.calculate_navigation_parameters()
        
        # Process LiDAR data for obstacles
        obstacle_description = self.get_obstacle_description()
        
        # Get action history for chain of thought
        history_description = self.get_action_history_description()
        
        state_description = f"""
CURRENT ROBOT STATE:
- Position: ({self.robot_x:.3f}, {self.robot_y:.3f}) meters
- Robot heading: {calc['robot_heading_deg']:.1f} degrees
- Goal position: (0.0, 0.0) meters

FORMULA-BASED CALCULATIONS:
- Distance to goal: {calc['distance_to_goal']:.3f} meters
- Angle to goal (world frame): {math.degrees(calc['angle_to_goal_world']):.1f} degrees
- Relative angle error: {calc['relative_goal_angle_deg']:.1f} degrees
- Turn direction: {'LEFT' if calc['relative_goal_angle'] > 0 else 'RIGHT' if calc['relative_goal_angle'] < 0 else 'STRAIGHT'}

DETERMINISTIC RECOMMENDATIONS:
- Recommended linear velocity: {calc['recommended_linear']:.3f} m/s
  Reasoning: {calc['linear_reasoning']}
- Recommended angular velocity: {calc['recommended_angular']:.3f} rad/s
  Reasoning: {calc['angular_reasoning']}

CALCULATION FORMULAS USED:
1. Distance = sqrt(x² + y²) = sqrt({self.robot_x:.3f}² + {self.robot_y:.3f}²) = {calc['distance_to_goal']:.3f}m
2. Goal angle = atan2(-y, -x) = atan2({-self.robot_y:.3f}, {-self.robot_x:.3f}) = {math.degrees(calc['angle_to_goal_world']):.1f}°
3. Relative angle = goal_angle - robot_heading = {math.degrees(calc['angle_to_goal_world']):.1f}° - {calc['robot_heading_deg']:.1f}° = {calc['relative_goal_angle_deg']:.1f}°
4. Linear velocity based on distance thresholds: <0.1m→stop, <0.2m→0.1, <0.5m→0.15, <1.0m→0.25, ≥1.0m→0.3
5. Angular velocity based on angle error: <5°→0.0, <15°→±0.2, <45°→±0.4, ≥45°→±0.6

COORDINATE SYSTEM CHECK:
- Robot position: ({self.robot_x:.3f}, {self.robot_y:.3f})
- Goal position: (0.000, 0.000)
- Vector to goal: ({-self.robot_x:.3f}, {-self.robot_y:.3f})
- Robot heading: {calc['robot_heading_deg']:.1f}° (0° = +X axis, 90° = +Y axis)
- Required heading to goal: {math.degrees(calc['angle_to_goal_world']):.1f}°
- Turn needed: {calc['relative_goal_angle_deg']:.1f}° ({'LEFT' if calc['relative_goal_angle'] > 0 else 'RIGHT' if calc['relative_goal_angle'] < 0 else 'STRAIGHT'})

OBSTACLES FROM LIDAR:
{obstacle_description}

PREVIOUS ACTIONS (Chain of Thought):
{history_description}

CONSTRAINTS:
- Maximum linear velocity: {self.max_linear_velocity} m/s
- Maximum angular velocity: {self.max_angular_velocity} rad/s
- Goal tolerance: {self.goal_tolerance} meters
"""
        return state_description
        
    def get_obstacle_description(self):
        """Analyze LiDAR data and describe obstacles"""
        if not self.lidar_ranges:
            return "No LiDAR data available"
            
        obstacles = []
        min_obstacle_distance = 0.5  # meters
        
        # Group nearby obstacles
        for i, distance in enumerate(self.lidar_ranges):
            if distance < min_obstacle_distance:
                angle = self.lidar_angles[i] if i < len(self.lidar_angles) else 0
                angle_deg = math.degrees(angle)
                
                # Determine direction relative to robot
                if -45 <= angle_deg <= 45:
                    direction = "front"
                elif 45 < angle_deg <= 135:
                    direction = "left"
                elif -135 <= angle_deg < -45:
                    direction = "right"
                else:
                    direction = "rear"
                    
                obstacles.append(f"- {direction}: {distance:.2f}m at {angle_deg:.0f}°")
                
        if not obstacles:
            return "No close obstacles detected (all obstacles > 0.5m away)"
        else:
            return "\n".join(obstacles[:10])  # Limit to 10 closest obstacles
            
    def get_action_history_description(self):
        """Get description of previous actions for chain of thought"""
        if not self.action_history:
            return "No previous actions recorded."
            
        history_text = []
        for i, action in enumerate(self.action_history[-self.max_history_length:]):
            step_num = len(self.action_history) - len(self.action_history[-self.max_history_length:]) + i + 1
            history_text.append(f"Step {step_num}: {action}")
            
        return "\n".join(history_text)
        
    def add_action_to_history(self, linear_vel, angular_vel, reasoning, distance_to_goal):
        """Add an action to the history for chain of thought"""
        action_description = f"Linear: {linear_vel:.2f} m/s, Angular: {angular_vel:.2f} rad/s, Distance: {distance_to_goal:.3f}m - {reasoning}"
        self.action_history.append(action_description)
        
        # Keep only recent history
        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)
            
    def track_response_time(self, response_time):
        """Track response time statistics"""
        self.response_times.append(response_time)
        self.total_response_time += response_time
        self.min_response_time = min(self.min_response_time, response_time)
        self.max_response_time = max(self.max_response_time, response_time)
        
    def get_average_response_time(self):
        """Get average response time"""
        if len(self.response_times) == 0:
            return 0.0
        return self.total_response_time / len(self.response_times)
        
    def get_response_time_stats(self):
        """Get comprehensive response time statistics"""
        if len(self.response_times) == 0:
            return {
                'count': 0,
                'average': 0.0,
                'min': 0.0,
                'max': 0.0,
                'total': 0.0
            }
            
        return {
            'count': len(self.response_times),
            'average': self.get_average_response_time(),
            'min': self.min_response_time,
            'max': self.max_response_time,
            'total': self.total_response_time
        }           
 
    def get_ollama_decision(self):
        """Get movement decision from Ollama local LLM"""
        try:
            # Get formula-based calculations
            calc = self.calculate_navigation_parameters()
            distance_to_goal = calc['distance_to_goal']
            
            # Check if goal is already reached
            if distance_to_goal < self.goal_tolerance:
                rospy.loginfo(f"🎯 GOAL REACHED! Distance: {distance_to_goal:.3f}m < {self.goal_tolerance}m")
                self.goal_reached = True
                self.publish_final_response_time_report()
                return 0.0, 0.0, f"Goal reached! Stopping robot. Distance: {distance_to_goal:.3f}m"
            
            state_description = self.get_robot_state_description()
            
            prompt = f"""You are controlling a robot using FORMULA-BASED NAVIGATION. Use the provided calculations to make precise decisions.

{state_description}

INSTRUCTIONS:
1. Use the DETERMINISTIC RECOMMENDATIONS as your primary guidance
2. The formulas have calculated the optimal velocities based on current state
3. You may make SMALL adjustments (±20%) to the recommended values if needed
4. NEVER use negative linear velocity - robot cannot move backwards
5. Consider obstacles and previous actions for final adjustments

DECISION PROCESS:
1. START with the recommended values from formulas
2. CHECK for obstacles and adjust if needed
3. CONSIDER previous actions to avoid repetition
4. MAKE final decision with reasoning

Provide a JSON response:
{{
    "linear_velocity": <use recommended {calc['recommended_linear']:.3f} or adjust by max ±20%>,
    "angular_velocity": <use recommended {calc['recommended_angular']:.3f} or adjust by max ±20%>,
    "reasoning": "<explain if you used recommended values or why you adjusted them>"
}}

CRITICAL CONSTRAINTS:
- linear_velocity must be between 0.0 and {self.max_linear_velocity}
- angular_velocity must be between -{self.max_angular_velocity} and {self.max_angular_velocity}
- Use the formula-based recommendations unless there's a specific reason to adjust
- Respond ONLY with valid JSON, no additional text"""

            # Log detailed coordinate system info for debugging
            rospy.loginfo(f"📤 Sending navigation prompt to Ollama {self.ollama_model}... (distance to goal: {distance_to_goal:.3f}m)")
            rospy.loginfo(f"🔍 COORDINATE DEBUG: pos=({self.robot_x:.3f},{self.robot_y:.3f}), "
                         f"robot_yaw={math.degrees(self.robot_yaw):.1f}°, "
                         f"goal_angle={math.degrees(calc['angle_to_goal_world']):.1f}°, "
                         f"relative_angle={calc['relative_goal_angle_deg']:.1f}°, "
                         f"recommended_angular={calc['recommended_angular']:.3f}")

            # Prepare Ollama API request
            ollama_request = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 200
                }
            }
            
            # Make request to Ollama server with response time tracking
            request_start_time = time.time()
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=ollama_request,
                timeout=self.request_timeout
            )
            request_end_time = time.time()
            response_time = request_end_time - request_start_time
            
            if response.status_code != 200:
                raise Exception(f"Ollama server returned status {response.status_code}: {response.text}")
            
            # Parse the response
            response_data = response.json()
            response_text = response_data.get('response', '').strip()
            
            # Log the raw response from Ollama
            rospy.loginfo("="*80)
            rospy.loginfo("📥 RECEIVED RESPONSE FROM OLLAMA:")
            rospy.loginfo("="*80)
            rospy.loginfo(f"Model: {self.ollama_model}")
            rospy.loginfo(f"Raw Response:")
            rospy.loginfo(response_text)
            rospy.loginfo("="*80)
            
            # Extract JSON from response (handle cases where there might be extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                decision = json.loads(json_text)
                
                # Validate and clamp values - FORCE linear velocity to be non-negative
                raw_linear = float(decision.get('linear_velocity', 0))
                raw_angular = float(decision.get('angular_velocity', 0))
                
                # CRITICAL: Linear velocity must NEVER be negative
                linear_vel = max(0.0, min(self.max_linear_velocity, raw_linear))
                angular_vel = max(-self.max_angular_velocity, 
                                min(self.max_angular_velocity, raw_angular))
                
                # Validate against formula recommendations
                recommended_linear = calc['recommended_linear']
                recommended_angular = calc['recommended_angular']
                
                # Check if Ollama deviated significantly from recommendations
                linear_deviation = abs(linear_vel - recommended_linear) / max(0.01, recommended_linear)
                angular_deviation = abs(angular_vel - recommended_angular) / max(0.01, abs(recommended_angular)) if recommended_angular != 0 else abs(angular_vel)
                
                # Log corrections and deviations
                if raw_linear < 0:
                    rospy.logwarn(f"⚠️  CORRECTED: Ollama suggested negative linear velocity {raw_linear:.3f} -> {linear_vel:.3f}")
                
                if linear_deviation > 0.3:  # More than 30% deviation
                    rospy.logwarn(f"⚠️  LARGE DEVIATION: Ollama linear {linear_vel:.3f} vs recommended {recommended_linear:.3f} ({linear_deviation*100:.1f}% diff)")
                    
                if angular_deviation > 0.3:  # More than 30% deviation
                    rospy.logwarn(f"⚠️  LARGE DEVIATION: Ollama angular {angular_vel:.3f} vs recommended {recommended_angular:.3f} ({angular_deviation*100:.1f}% diff)")
                
                reasoning = decision.get('reasoning', 'No reasoning provided')
                
                rospy.loginfo("="*80)
                rospy.loginfo("🤖 PARSED OLLAMA DECISION:")
                rospy.loginfo("="*80)
                rospy.loginfo(f"Model: {self.ollama_model}")
                rospy.loginfo(f"Linear velocity: {linear_vel:.3f} m/s")
                rospy.loginfo(f"Angular velocity: {angular_vel:.3f} rad/s")
                rospy.loginfo(f"Reasoning: {reasoning}")
                
                # Add to action history for chain of thought
                distance_to_goal = math.sqrt(self.robot_x**2 + self.robot_y**2)
                self.add_action_to_history(linear_vel, angular_vel, reasoning, distance_to_goal)
                
                rospy.loginfo(f"Current position: ({self.robot_x:.3f}, {self.robot_y:.3f})")
                rospy.loginfo(f"Distance to goal: {distance_to_goal:.3f}m")
                rospy.loginfo("="*80)
                
                # Track API success and response time
                self.api_successes += 1
                self.track_response_time(response_time)
                
                rospy.loginfo(f"⏱️  Response time: {response_time:.3f}s (avg: {self.get_average_response_time():.3f}s)")
                
                return linear_vel, angular_vel, reasoning
                
            else:
                rospy.logwarn("="*80)
                rospy.logwarn("❌ FAILED TO PARSE JSON FROM OLLAMA RESPONSE")
                rospy.logwarn("="*80)
                rospy.logwarn(f"Raw response: {response_text}")
                rospy.logwarn("="*80)
                return 0.0, 0.0, "Failed to parse response"
                
        except Exception as e:
            rospy.logerr("="*80)
            rospy.logerr("❌ OLLAMA API ERROR - USING FALLBACK NAVIGATION")
            rospy.logerr("="*80)
            rospy.logerr(f"Error: {e}")
            
            # Get formula-based calculations for fallback
            calc = self.calculate_navigation_parameters()
            distance_to_goal = calc['distance_to_goal']
            
            # Use formula-based fallback - same calculations as provided to Ollama
            if distance_to_goal < self.goal_tolerance:
                fallback_decision = (0.0, 0.0, "Fallback: Goal reached")
            else:
                # Use the same formula calculations
                fallback_linear = calc['recommended_linear']
                fallback_angular = calc['recommended_angular']
                fallback_decision = (fallback_linear, fallback_angular, 
                                   f"Fallback: Using formula-based navigation - {calc['linear_reasoning']}, {calc['angular_reasoning']}")
            
            rospy.logerr(f"Fallback Decision: linear={fallback_decision[0]:.3f}, angular={fallback_decision[1]:.3f}")
            rospy.logerr(f"Fallback Reasoning: {fallback_decision[2]}")
            rospy.logerr("="*80)
            
            # Track API failure
            self.api_failures += 1
            
            return fallback_decision
            
    def control_loop(self, event):
        """Main control loop"""
        if not self.valid_mocap_received:
            rospy.logwarn_throttle(5, "⏳ Waiting for mocap data...")
            return
            
        # Check if goal is reached and stop
        if self.goal_reached:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            rospy.loginfo_throttle(2, "🎯 Goal reached! Robot stopped.")
            
            # Show periodic response time summary
            if len(self.response_times) > 0:
                stats = self.get_response_time_stats()
                rospy.loginfo_throttle(5, f"📊 Response Time Summary: avg={stats['average']:.3f}s, "
                                      f"min={stats['min']:.3f}s, max={stats['max']:.3f}s, calls={stats['count']}")
            return
            
        current_time = time.time()
        
        # Get new decision from Ollama at specified frequency
        time_since_last_decision = current_time - self.last_decision_time
        decision_interval = 1.0 / self.decision_frequency
        
        rospy.logdebug(f"Decision timing: {time_since_last_decision:.2f}s since last, interval: {decision_interval:.2f}s")
        
        if time_since_last_decision >= decision_interval:
            linear_vel, angular_vel, reasoning = self.get_ollama_decision()
            
            self.current_linear_vel = linear_vel
            self.current_angular_vel = angular_vel
            self.last_decision_time = current_time
            
            # Log performance data
            if self.log_performance:
                self.log_performance_data(reasoning)
        
        # Publish control commands
        cmd = Twist()
        cmd.linear.x = self.current_linear_vel
        cmd.angular.z = self.current_angular_vel
        self.cmd_vel_pub.publish(cmd)
        
        # Verbose logging of commands sent
        rospy.logdebug(f"📤 Sending cmd_vel: linear={cmd.linear.x:.3f}, angular={cmd.angular.z:.3f}")
        
    def log_performance_data(self, reasoning):
        """Log performance data for current model"""
        current_time = time.time()
        distance_to_goal = math.sqrt(
            (self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2
        )
        
        performance_entry = {
            'timestamp': current_time,
            'model': self.ollama_model,
            'robot_x': self.robot_x,
            'robot_y': self.robot_y,
            'robot_yaw': self.robot_yaw,
            'distance_to_goal': distance_to_goal,
            'distance_traveled': self.distance_traveled,
            'linear_velocity': self.current_linear_vel,
            'angular_velocity': self.current_angular_vel,
            'reasoning': reasoning,
            'goal_reached': self.goal_reached,
            'elapsed_time': current_time - self.test_start_time,
            'action_count': len(self.action_history),
            'response_time_stats': self.get_response_time_stats()
        }
        
        self.performance_data.append(performance_entry)
        
        # Publish performance data
        performance_msg = String()
        performance_msg.data = json.dumps(performance_entry)
        self.performance_pub.publish(performance_msg)
        
    def publish_final_response_time_report(self):
        """Publish final response time report when goal is reached"""
        stats = self.get_response_time_stats()
        
        rospy.loginfo("="*80)
        rospy.loginfo("🏁 FINAL RESPONSE TIME REPORT")
        rospy.loginfo("="*80)
        rospy.loginfo(f"Model: {self.ollama_model}")
        rospy.loginfo(f"Server: {self.ollama_server_ip}:{self.ollama_server_port}")
        rospy.loginfo(f"Total API calls: {stats['count']}")
        rospy.loginfo(f"Average response time: {stats['average']:.3f} seconds")
        rospy.loginfo(f"Minimum response time: {stats['min']:.3f} seconds")
        rospy.loginfo(f"Maximum response time: {stats['max']:.3f} seconds")
        rospy.loginfo(f"Total response time: {stats['total']:.3f} seconds")
        rospy.loginfo(f"API success rate: {self.api_successes}/{self.api_successes + self.api_failures} ({100*self.api_successes/(self.api_successes + self.api_failures):.1f}%)")
        
        # Calculate performance metrics
        if len(self.response_times) > 0:
            # Calculate percentiles
            sorted_times = sorted(self.response_times)
            p50 = sorted_times[len(sorted_times)//2]
            p95 = sorted_times[int(len(sorted_times)*0.95)] if len(sorted_times) > 1 else sorted_times[0]
            
            rospy.loginfo(f"Response time P50 (median): {p50:.3f} seconds")
            rospy.loginfo(f"Response time P95: {p95:.3f} seconds")
            
            # Performance classification
            if stats['average'] < 1.0:
                performance_class = "EXCELLENT"
            elif stats['average'] < 2.0:
                performance_class = "GOOD"
            elif stats['average'] < 5.0:
                performance_class = "ACCEPTABLE"
            else:
                performance_class = "SLOW"
                
            rospy.loginfo(f"Performance classification: {performance_class}")
        
        rospy.loginfo("="*80)
        
        # Publish detailed report as JSON
        final_report = {
            'type': 'final_response_time_report',
            'model': self.ollama_model,
            'server': f"{self.ollama_server_ip}:{self.ollama_server_port}",
            'statistics': stats,
            'api_success_rate': self.api_successes/(self.api_successes + self.api_failures) if (self.api_successes + self.api_failures) > 0 else 0,
            'timestamp': time.time(),
            'elapsed_time': time.time() - self.test_start_time
        }
        
        if len(self.response_times) > 0:
            sorted_times = sorted(self.response_times)
            final_report['percentiles'] = {
                'p50': sorted_times[len(sorted_times)//2],
                'p95': sorted_times[int(len(sorted_times)*0.95)] if len(sorted_times) > 1 else sorted_times[0]
            }
        
        report_msg = String()
        report_msg.data = json.dumps(final_report)
        self.performance_pub.publish(report_msg)

if __name__ == '__main__':
    try:
        controller = OllamaMocapController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Ollama Mocap Controller shutting down")
    except Exception as e:
        rospy.logerr(f"Ollama Mocap Controller failed: {e}")