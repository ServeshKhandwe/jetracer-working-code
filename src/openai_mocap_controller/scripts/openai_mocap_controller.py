#!/usr/bin/env python3

import rospy
import asyncio
import websockets
import json
import math
import threading
import time
import numpy as np
from datetime import datetime
from collections import deque

# ROS imports
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

# OpenAI imports
import openai
from openai import OpenAI

class OpenAIMocapController:
    def __init__(self):
        rospy.init_node('openai_mocap_controller', anonymous=True)
        
        rospy.loginfo("="*60)
        rospy.loginfo("OpenAI Mocap Controller Starting")
        rospy.loginfo("="*60)
        
        # Load parameters
        self.load_parameters()
        
        # Initialize OpenAI client
        self.setup_openai()
        
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
        self.current_model_index = 0
        self.test_start_time = None
        self.distance_traveled = 0.0
        self.last_position = None
        
        # ROS Publishers and Subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.performance_pub = rospy.Publisher('/performance_data', String, queue_size=10)
        
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # Start mocap client
        self.start_mocap_client()
        
        # Start control loop
        self.control_timer = rospy.Timer(rospy.Duration(1.0/self.control_frequency), self.control_loop)
        
        rospy.loginfo("OpenAI Mocap Controller initialized successfully")
        
    def load_parameters(self):
        """Load ROS parameters"""
        self.openai_api_key = rospy.get_param('~openai_api_key', '')
        self.mocap_server_ip = rospy.get_param('~mocap_server_ip', '192.168.64.147')
        self.mocap_server_port = rospy.get_param('~mocap_server_port', 8765)
        self.control_frequency = rospy.get_param('~control_frequency', 10.0)
        self.decision_frequency = rospy.get_param('~decision_frequency', 2.0)
        self.position_scale_factor = rospy.get_param('~position_scale_factor', 0.001)
        self.max_linear_velocity = rospy.get_param('~max_linear_velocity', 0.3)
        self.max_angular_velocity = rospy.get_param('~max_angular_velocity', 0.8)
        
        # Model comparison parameters
        models_str = rospy.get_param('~models_to_test', 'gpt-4,gpt-3.5-turbo')
        self.models_to_test = [model.strip() for model in models_str.split(',')]
        self.test_duration = rospy.get_param('~test_duration', 60)  # seconds per model
        self.log_performance = rospy.get_param('~log_performance', True)
        
        rospy.loginfo(f"Models to test: {self.models_to_test}")
        rospy.loginfo(f"Test duration per model: {self.test_duration} seconds")
        
    def setup_openai(self):
        """Initialize OpenAI client"""
        if not self.openai_api_key:
            rospy.logfatal("OpenAI API key not provided! Set OPENAI_API_KEY environment variable or parameter.")
            rospy.signal_shutdown("Missing OpenAI API key")
            return
            
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.current_model = self.models_to_test[0] if self.models_to_test else "gpt-3.5-turbo"
        
        rospy.loginfo(f"OpenAI client initialized with model: {self.current_model}")
        
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
        return math.atan2(siny_cosp, cosy_cosp)
        
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
            
    def get_robot_state_description(self):
        """Generate a text description of the robot's current state"""
        # Calculate distance and angle to goal
        distance_to_goal = math.sqrt(
            (self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2
        )
        
        angle_to_goal = math.atan2(
            self.goal_y - self.robot_y, 
            self.goal_x - self.robot_x
        )
        
        # Normalize angle relative to robot's heading
        relative_goal_angle = angle_to_goal - self.robot_yaw
        while relative_goal_angle > math.pi:
            relative_goal_angle -= 2 * math.pi
        while relative_goal_angle < -math.pi:
            relative_goal_angle += 2 * math.pi
            
        # Process LiDAR data for obstacles
        obstacle_description = self.get_obstacle_description()
        
        state_description = f"""
Robot State:
- Position: ({self.robot_x:.2f}, {self.robot_y:.2f}) meters
- Heading: {math.degrees(self.robot_yaw):.1f} degrees
- Goal: (0.0, 0.0) meters
- Distance to goal: {distance_to_goal:.2f} meters
- Angle to goal: {math.degrees(relative_goal_angle):.1f} degrees relative to robot heading
- Current velocities: linear={self.current_linear_vel:.2f} m/s, angular={self.current_angular_vel:.2f} rad/s

Obstacles detected by LiDAR:
{obstacle_description}

Constraints:
- Maximum linear velocity: {self.max_linear_velocity} m/s
- Maximum angular velocity: {self.max_angular_velocity} rad/s
- Goal is always at (0, 0)
"""
        print(state_description)
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
            
    def get_openai_decision(self):
        """Get movement decision from OpenAI"""
        try:
            state_description = self.get_robot_state_description()
            
            prompt = f"""
You are controlling a robot that needs to navigate to the goal at (0, 0). 
Based on the current state, provide movement commands.

{state_description}

Provide a JSON response with linear and angular velocities:
{{
    "linear_velocity": <value between -{self.max_linear_velocity} and {self.max_linear_velocity}>,
    "angular_velocity": <value between -{self.max_angular_velocity} and {self.max_angular_velocity}>,
    "reasoning": "<brief explanation of your decision>"
}}

Consider:
1. Move towards the goal at (0, 0)
2. Avoid obstacles detected by LiDAR
3. Use appropriate speeds for safe navigation
4. If very close to goal (< 0.1m), use very slow speeds or stop
"""

            response = self.openai_client.chat.completions.create(
                model=self.current_model,
                messages=[
                    {"role": "system", "content": "You are a robot navigation controller. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response (handle cases where there might be extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                decision = json.loads(json_text)
                
                # Validate and clamp values
                linear_vel = max(-self.max_linear_velocity, 
                               min(self.max_linear_velocity, 
                                   float(decision.get('linear_velocity', 0))))
                angular_vel = max(-self.max_angular_velocity, 
                                min(self.max_angular_velocity, 
                                    float(decision.get('angular_velocity', 0))))
                
                reasoning = decision.get('reasoning', 'No reasoning provided')
                
                rospy.loginfo(f"OpenAI Decision ({self.current_model}): "
                            f"linear={linear_vel:.2f}, angular={angular_vel:.2f}")
                rospy.loginfo(f"Reasoning: {reasoning}")
                
                return linear_vel, angular_vel, reasoning
                
            else:
                rospy.logwarn("Could not parse JSON from OpenAI response")
                return 0.0, 0.0, "Failed to parse response"
                
        except Exception as e:
            rospy.logerr(f"Error getting OpenAI decision: {e}")
            return 0.0, 0.0, f"Error: {str(e)}"
            
    def control_loop(self, event):
        """Main control loop"""
        if not self.valid_mocap_received:
            rospy.logwarn_throttle(5, "Waiting for mocap data...")
            return
            
        current_time = time.time()
        
        # Check if we need to switch models
        self.check_model_switching()
        
        # Get new decision from OpenAI at specified frequency
        if current_time - self.last_decision_time >= (1.0 / self.decision_frequency):
            linear_vel, angular_vel, reasoning = self.get_openai_decision()
            
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
        
    def check_model_switching(self):
        """Check if it's time to switch to the next model"""
        current_time = time.time()
        
        if self.test_start_time is None:
            self.test_start_time = current_time
            rospy.loginfo(f"Starting test with model: {self.current_model}")
            
        # Check if current model test duration is complete
        if current_time - self.test_start_time >= self.test_duration:
            self.current_model_index += 1
            
            if self.current_model_index < len(self.models_to_test):
                # Switch to next model
                self.current_model = self.models_to_test[self.current_model_index]
                self.test_start_time = current_time
                self.distance_traveled = 0.0  # Reset for new model
                
                rospy.loginfo(f"Switching to model: {self.current_model}")
                
            else:
                # All models tested
                rospy.loginfo("All models tested. Publishing final performance report.")
                self.publish_final_performance_report()
                
                # Reset for another round if desired
                self.current_model_index = 0
                self.current_model = self.models_to_test[0]
                self.test_start_time = current_time
                
    def log_performance_data(self, reasoning):
        """Log performance data for current model"""
        current_time = time.time()
        distance_to_goal = math.sqrt(
            (self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2
        )
        
        performance_entry = {
            'timestamp': current_time,
            'model': self.current_model,
            'robot_x': self.robot_x,
            'robot_y': self.robot_y,
            'robot_yaw': self.robot_yaw,
            'distance_to_goal': distance_to_goal,
            'distance_traveled': self.distance_traveled,
            'linear_velocity': self.current_linear_vel,
            'angular_velocity': self.current_angular_vel,
            'reasoning': reasoning
        }
        
        self.performance_data.append(performance_entry)
        
        # Publish performance data
        performance_msg = String()
        performance_msg.data = json.dumps(performance_entry)
        self.performance_pub.publish(performance_msg)
        
    def publish_final_performance_report(self):
        """Publish a summary of all model performances"""
        if not self.performance_data:
            return
            
        # Group data by model
        model_performance = {}
        for entry in self.performance_data:
            model = entry['model']
            if model not in model_performance:
                model_performance[model] = []
            model_performance[model].append(entry)
            
        # Calculate metrics for each model
        report = {"model_comparison": {}}
        
        for model, data in model_performance.items():
            if not data:
                continue
                
            distances_to_goal = [entry['distance_to_goal'] for entry in data]
            total_distance_traveled = data[-1]['distance_traveled'] if data else 0
            
            avg_distance_to_goal = np.mean(distances_to_goal)
            min_distance_to_goal = np.min(distances_to_goal)
            final_distance_to_goal = distances_to_goal[-1] if distances_to_goal else float('inf')
            
            report["model_comparison"][model] = {
                'average_distance_to_goal': avg_distance_to_goal,
                'minimum_distance_to_goal': min_distance_to_goal,
                'final_distance_to_goal': final_distance_to_goal,
                'total_distance_traveled': total_distance_traveled,
                'efficiency_ratio': total_distance_traveled / max(0.1, avg_distance_to_goal),
                'data_points': len(data)
            }
            
        # Publish report
        report_msg = String()
        report_msg.data = json.dumps(report, indent=2)
        self.performance_pub.publish(report_msg)
        
        rospy.loginfo("Performance Report:")
        rospy.loginfo(json.dumps(report, indent=2))
        
    def run(self):
        """Main run loop"""
        rospy.loginfo("OpenAI Mocap Controller running...")
        rospy.spin()

def main():
    try:
        controller = OpenAIMocapController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("OpenAI Mocap Controller shutting down...")
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")

if __name__ == '__main__':
    main()