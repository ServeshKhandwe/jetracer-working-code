#!/usr/bin/env python3

import rospy
import json
import math
import numpy as np
import asyncio
import websockets
import threading
import openai
from datetime import datetime
from geometry_msgs.msg import Twist
import os

class LLMController:
    def __init__(self):
        rospy.init_node('llm_controller', anonymous=True)
        
        # Robot state
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.yaw = 0.0
        self.robot_mocap_received = False
        
        # Control parameters
        self.max_linear_velocity = 0.3  # m/s
        self.max_angular_velocity = 1.0  # rad/s
        self.position_scale_factor = 0.001  # Convert mm to m
        
        # Goal position
        self.goal_x = 0.0
        self.goal_y = 0.0
        
        # Publisher for robot commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Performance logging
        self.log_dir = os.path.join(os.path.dirname(__file__), '../logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, 
                                    f'performance_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        # OpenAI setup from config file
        try:
            from llm_controller_comparison.config import api_config
            openai.api_key = api_config.OPENAI_API_KEY
            self.default_model = api_config.DEFAULT_MODEL
            if openai.api_key == "your-api-key-here":
                rospy.logerr("Please set your actual API key in config/api_config.py")
                return
        except ImportError:
            rospy.logerr("Could not import api_config. Please ensure config/api_config.py exists with OPENAI_API_KEY")
            return
        except Exception as e:
            rospy.logerr(f"Error setting up OpenAI API: {e}")
            return
            
        # Start mocap client in a separate thread
        self.mocap_thread = threading.Thread(target=self.start_mocap_client)
        self.mocap_thread.daemon = True
        self.mocap_thread.start()
        
        # Control loop timer
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)  # 10Hz control loop

    def start_mocap_client(self):
        """Start the mocap client in a new event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.connect_to_mocap())

    async def connect_to_mocap(self, server_ip="192.168.64.147"):
        """Connect to motion capture system via WebSocket"""
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
                                # Convert mocap position from millimeters to meters
                                self.x_pos = data["objects"]["Player"]["x"] * self.position_scale_factor
                                self.y_pos = data["objects"]["Player"]["y"] * self.position_scale_factor
                                
                                # Get orientation quaternion
                                qx = data["objects"]["Player"]["qx"]
                                qy = data["objects"]["Player"]["qy"]
                                qz = data["objects"]["Player"]["qz"]
                                qw = data["objects"]["Player"]["qw"]
                                
                                # Calculate yaw from quaternion
                                siny_cosp = 2 * (qw * qz + qx * qy)
                                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                                self.yaw = math.atan2(siny_cosp, cosy_cosp)
                                
                                self.robot_mocap_received = True
                                
                        except websockets.exceptions.ConnectionClosed:
                            rospy.logwarn("Motion capture connection lost, attempting to reconnect...")
                            break
            except Exception as e:
                rospy.logerr(f"Motion capture connection error: {e}")
                await asyncio.sleep(1.0)

    async def get_llm_control(self, model=None):
        """Get control commands from LLM"""
        try:
            # Calculate distance and angle to goal
            dx = self.goal_x - self.x_pos
            dy = self.goal_y - self.y_pos
            distance = math.sqrt(dx*dx + dy*dy)
            angle_to_goal = math.atan2(dy, dx)
            angle_diff = angle_to_goal - self.yaw
            
            # Normalize angle difference to [-pi, pi]
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            
            # Create prompt for LLM
            prompt = f"""You are controlling a differential drive robot.
Current state:
- Position: ({self.x_pos:.2f}, {self.y_pos:.2f})
- Orientation: {self.yaw:.2f} rad
- Distance to goal: {distance:.2f} m
- Angle to goal: {angle_diff:.2f} rad

The goal is at (0, 0). Provide linear and angular velocities as a JSON:
{{
    "linear_velocity": float (-0.3 to 0.3 m/s),
    "angular_velocity": float (-1.0 to 1.0 rad/s)
}}

Focus on smooth, efficient movement."""

            response = await openai.ChatCompletion.acreate(
                model=model or self.default_model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )
            
            try:
                result = json.loads(response.choices[0].message.content)
                return result["linear_velocity"], result["angular_velocity"]
            except (json.JSONDecodeError, KeyError) as e:
                rospy.logerr(f"Error parsing LLM response: {e}")
                return 0.0, 0.0
                
        except Exception as e:
            rospy.logerr(f"Error getting LLM control: {e}")
            return 0.0, 0.0

    def control_loop(self, event):
        """Main control loop"""
        if not self.robot_mocap_received:
            return
            
        try:
            # Get control commands from LLM
            linear_vel, angular_vel = asyncio.run(self.get_llm_control())
            
            # Clamp velocities
            linear_vel = np.clip(linear_vel, -self.max_linear_velocity, self.max_linear_velocity)
            angular_vel = np.clip(angular_vel, -self.max_angular_velocity, self.max_angular_velocity)
            
            # Create and publish command
            cmd = Twist()
            cmd.linear.x = linear_vel
            cmd.angular.z = angular_vel
            self.cmd_vel_pub.publish(cmd)
            
            # Log performance
            distance_to_goal = math.sqrt(self.x_pos**2 + self.y_pos**2)
            with open(self.log_file, 'a') as f:
                f.write(f"{rospy.Time.now().to_sec()},{self.x_pos},{self.y_pos},{self.yaw}," + 
                       f"{linear_vel},{angular_vel},{distance_to_goal}\n")
                
        except Exception as e:
            rospy.logerr(f"Error in control loop: {e}")

    def run(self):
        """Main run loop"""
        rospy.spin()

if __name__ == '__main__':
    try:
        controller = LLMController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
