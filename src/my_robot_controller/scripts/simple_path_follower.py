#!/home/jetson/ros_venv/bin/python3
import rospy
from geometry_msgs.msg import Twist
import math
import numpy as np
import asyncio
import websockets
import json
import threading
import time
from std_msgs.msg import String
import csv
import os
from datetime import datetime

class PathFollowerNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("simple_path_follower", anonymous=True)
        
        # Publisher for velocity commands
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        
        # Path selector publisher (for visualization)
        self.path_pub = rospy.Publisher("/current_path", String, queue_size=10)
        
        # Motion capture variables
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_qx = 0.0
        self.robot_qy = 0.0
        self.robot_qz = 0.0
        self.robot_qw = 0.0
        self.yaw = 0.0
        self.valid_mocap_received = False
        
        # Current control inputs for recording
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        
        # Data recording variables
        self.recording = False
        self.data_buffer = []
        self.start_time = None
        self.recording_frequency = 20.0  # Hz - constant frequency for data recording
        self.recording_period = 1.0 / self.recording_frequency
        
        # Create data directory
        self.data_dir = "/home/jetson/trajectory_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Velocity-based path following variables
        self.current_path = None
        self.velocity_commands = []
        self.current_command_idx = 0
        self.path_completed = False
        
        # Tick-based timing
        self.control_loop_period = 0.05  # Corresponds to rospy.Duration(0.05)
        self.current_command_ticks_elapsed = 0
        
        # Control parameters - scaled for real robot with velocity variations
        self.very_slow_linear_velocity = 0.05   # m/s - very slow speed
        self.slow_linear_velocity = 0.08        # m/s - slower forward speed
        self.base_linear_velocity = 0.15        # m/s - moderate forward speed
        self.fast_linear_velocity = 0.25        # m/s - faster forward speed
        self.very_fast_linear_velocity = 0.35   # m/s - very fast speed
        self.base_angular_velocity = 0.3        # rad/s - moderate turning speed
        
        # Start mocap client in a separate thread
        self.mocap_thread = threading.Thread(target=self.start_mocap_client)
        self.mocap_thread.daemon = True
        self.mocap_thread.start()
        
        # Control loop timer - higher frequency for smooth execution
        self.control_timer = rospy.Timer(rospy.Duration(self.control_loop_period), self.control_loop)
        
        # Data recording timer - constant frequency
        self.recording_timer = rospy.Timer(rospy.Duration(self.recording_period), self.record_data)
        
        # Initialize with path 1 by default
        self.select_path(1)
        
        rospy.loginfo("Velocity-based path follower node initialized with data recording")

    def start_recording(self, trajectory_name):
        """Start data recording for a specific trajectory"""
        self.recording = True
        self.data_buffer = []
        self.start_time = rospy.Time.now()
        rospy.loginfo(f"Started recording data for trajectory: {trajectory_name}")

    def stop_recording(self, trajectory_name):
        """Stop data recording and save to file"""
        if self.recording:
            self.recording = False
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_dir}/trajectory_{trajectory_name}_{timestamp}.csv"
            
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['time_sec', 'x_pos', 'y_pos', 'psi_rad', 'linear_vel', 'angular_vel']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for data_point in self.data_buffer:
                    writer.writerow(data_point)
            
            rospy.loginfo(f"Data saved to: {filename} ({len(self.data_buffer)} data points)")
            self.data_buffer = []

    def record_data(self, event):
        """Record data at constant frequency"""
        if self.recording and self.valid_mocap_received:
            current_time = rospy.Time.now()
            elapsed_time = (current_time - self.start_time).to_sec()
            
            data_point = {
                'time_sec': elapsed_time,
                'x_pos': self.robot_x,
                'y_pos': self.robot_y,
                'psi_rad': self.yaw,
                'linear_vel': self.current_linear_vel,
                'angular_vel': self.current_angular_vel  # This represents wheel angle control
            }
            
            self.data_buffer.append(data_point)

    def start_mocap_client(self):
        """Start the mocap client in a new event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.connect_to_mocap())
    
    async def connect_to_mocap(self, server_ip="192.168.96.147"):
        """Connect to motion capture system via websocket"""
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
                                self.robot_x = data["objects"]["Player"]["x"]
                                self.robot_y = data["objects"]["Player"]["y"]
                                self.robot_qx = data["objects"]["Player"]["qx"]
                                self.robot_qy = data["objects"]["Player"]["qy"]
                                self.robot_qz = data["objects"]["Player"]["qz"]
                                self.robot_qw = data["objects"]["Player"]["qw"]
                                
                                # Calculate yaw from quaternion
                                siny_cosp = 2 * (
                                    self.robot_qw * self.robot_qz
                                    + self.robot_qx * self.robot_qy
                                )
                                cosy_cosp = 1 - 2 * (
                                    self.robot_qy * self.robot_qy
                                    + self.robot_qz * self.robot_qz
                                )
                                self.yaw = math.atan2(siny_cosp, cosy_cosp)
                                
                                self.valid_mocap_received = True
                                
                        except websockets.exceptions.ConnectionClosed:
                            rospy.logwarn("Motion capture connection lost, attempting to reconnect...")
                            break
            except Exception as e:
                rospy.logerr(f"Motion capture connection error: {e}")
                await asyncio.sleep(1.0)
    
    def select_path(self, path_num):
        """Select which velocity pattern to follow"""
        self.current_path = path_num
        self.current_command_idx = 0
        self.current_command_ticks_elapsed = 0
        self.path_completed = False
        
        # Generate velocity commands based on selection
        if path_num == 1:
            self.velocity_commands = self.get_vertical_s_commands()
            path_name = "vertical_s_shape"
        elif path_num == 2:
            self.velocity_commands = self.get_square_wave_commands()
            path_name = "square_wave"
        elif path_num == 3:
            self.velocity_commands = self.get_p_shape_commands()
            path_name = "p_shape_loop"
        elif path_num == 4:
            self.velocity_commands = self.get_horizontal_s_commands()
            path_name = "horizontal_s_shape"
        elif path_num == 8:
            self.velocity_commands = self.get_smooth_trajectory_commands()
            path_name = "smooth_trajectory"
        else:
            self.velocity_commands = self.get_vertical_s_commands()
            path_name = "default_vertical_s"
        
        self.path_pub.publish(path_name)
        
        # Start recording when path is selected
        self.start_recording(path_name)

    def control_loop(self, event):
        """Main control loop for velocity-based path following"""
        if not self.valid_mocap_received:
            rospy.logwarn_throttle(5.0, "Waiting for valid mocap data...")
            return
        
        if self.path_completed:
            # If path is completed, stop the robot and recording
            self.send_velocity(0.0, 0.0)
            if self.recording:
                path_name = f"path_{self.current_path}"
                self.stop_recording(path_name)
            return
        
        if not self.velocity_commands:
            rospy.logwarn_throttle(5.0, "No velocity commands available")
            return
        
        # Check if all commands in the current path are completed
        if self.current_command_idx >= len(self.velocity_commands):
            if not self.path_completed:
                rospy.loginfo("All velocity commands completed for the current path (idx check).")
                self.path_completed = True
            self.send_velocity(0.0, 0.0)
            return
        
        linear_vel, angular_vel, duration = self.velocity_commands[self.current_command_idx]
        
        # Increment ticks for the current command
        self.current_command_ticks_elapsed += 1
        elapsed_seconds = self.current_command_ticks_elapsed * self.control_loop_period
        
        # Check if current command duration has elapsed
        if elapsed_seconds >= duration:
            # Move to next command
            self.current_command_idx += 1
            self.current_command_ticks_elapsed = 0
            
            if self.current_command_idx < len(self.velocity_commands):
                rospy.loginfo(f"Moving to command {self.current_command_idx + 1}/{len(self.velocity_commands)}")
            else:
                rospy.loginfo("All velocity commands completed (duration check).")
                self.path_completed = True
                self.send_velocity(0.0, 0.0)
                return 
        else:
            # Execute current command
            self.send_velocity(linear_vel, angular_vel)
            
            remaining = duration - elapsed_seconds
            # Debug output (throttled)
            rospy.loginfo_throttle(1.0, 
                f"Cmd {self.current_command_idx + 1}/{len(self.velocity_commands)}: "
                f"Lin: {linear_vel:.2f} Ang: {angular_vel:.2f} "
                f"Elapsed: {elapsed_seconds:.2f}s / {duration:.2f}s. Remaining: {remaining:.2f}s")

    def send_velocity(self, linear_vel, angular_vel):
        """Send velocity commands to the robot"""
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)
        
        # Store current velocities for data recording
        self.current_linear_vel = linear_vel
        self.current_angular_vel = angular_vel

    # Velocity command generation functions with velocity variations
    def get_vertical_s_commands(self):
        """Generate velocity commands for vertical S-shape with velocity variations"""
        commands = [
            # Start moving forward - slow (longer time for same distance)
            (self.slow_linear_velocity, 0.0, 1.9),  # 0.08 * 1.9 = 0.152m vs 0.15 * 1.0 = 0.15m
            # First curve - turn right while moving forward - moderate speed
            (self.base_linear_velocity, -self.base_angular_velocity, 4.0),
            # Straight section - fast (shorter time for same distance)
            (self.fast_linear_velocity, 0.0, 1.2),  # 0.25 * 1.2 = 0.3m vs 0.15 * 2.0 = 0.3m
            # Second curve - turn left while moving forward - slow speed (longer time)
            (self.slow_linear_velocity, self.base_angular_velocity, 10.5),  # Compensate for slower speed during turn
            # Final straight section - very fast (shorter time)
            (self.very_fast_linear_velocity, 0.0, 0.43),  # 0.35 * 0.43 = 0.15m vs 0.15 * 1.0 = 0.15m
            # Stop
            (0.0, 0.0, 1.0)
        ]
        return commands

    def get_square_wave_commands(self):
        """Generate velocity commands for square wave pattern with velocity variations"""
        commands = [
            # Move forward - moderate speed
            (self.base_linear_velocity, 0.0, 2.0),
            # Turn left 90 degrees with forward motion - slow
            (self.slow_linear_velocity, self.base_angular_velocity, 12.6),  # ~90 degrees with forward motion
            # Move forward (up) - fast
            (self.fast_linear_velocity, 0.0, 2.0),
            # Turn left 90 degrees with forward motion - very slow
            (self.very_slow_linear_velocity, self.base_angular_velocity, 12.6),
            # Move forward - very fast
            (self.very_fast_linear_velocity, 0.0, 2.0),
            # Turn right 90 degrees with forward motion - moderate
            (self.base_linear_velocity, -self.base_angular_velocity, 12.6),
            # Move forward (down) - slow
            (self.slow_linear_velocity, 0.0, 2.0),
            # Turn right 90 degrees with forward motion - fast
            #(self.fast_linear_velocity, -self.base_angular_velocity, 12.6),
            # Final forward movement - moderate
            (self.base_linear_velocity, 0.0, 2.0),
            # Stop
            (0.0, 0.0, 1.0)
        ]
        return commands

    def get_p_shape_commands(self):
        """Generate velocity commands for P-shape/Loop with velocity variations"""
        commands = [
            # Move forward (stem of P) - start slow, increase speed
            (self.slow_linear_velocity, 0.0, 3.75),    # 0.08 * 3.75 = 0.3m vs 0.15 * 2.0 = 0.3m
            (self.base_linear_velocity, 0.0, 3.0),     # Baseline
            (self.fast_linear_velocity, 0.0, 1.8),     # 0.25 * 1.8 = 0.45m vs 0.15 * 3.0 = 0.45m
            # Start the loop - turn right - moderate speed
            (self.base_linear_velocity, -self.base_angular_velocity, 10.5),
            # Continue loop - more right turn - slow speed (longer time)
            (self.slow_linear_velocity, -self.base_angular_velocity, 19.7),  # 0.08 vs 0.15 baseline
            # Continue loop - more right turn - fast speed (shorter time)
            (self.fast_linear_velocity, -self.base_angular_velocity, 6.3),   # 0.25 vs 0.15 baseline
            # Complete the loop - final right turn - moderate speed
            #(self.base_linear_velocity, -self.base_angular_velocity, 12.5),
            # Move back to stem connection - slow (longer time)
            (self.slow_linear_velocity, 0.0, 1.9),     # 0.08 vs 0.15 baseline
            # Stop
            (0.0, 0.0, 1.0)
        ]
        return commands

    def get_horizontal_s_commands(self):
        """Generate velocity commands for horizontal S-shape with velocity variations"""
        commands = [
            # Initial turn left - slow (longer time)
            (self.slow_linear_velocity, self.base_angular_velocity, 3.75),  # 0.08 vs 0.15 baseline
            # Straight section - fast (shorter time)
            (self.fast_linear_velocity, 0.0, 1.2),     # 0.25 vs 0.15 baseline
            # Turn right (opposite direction) - very slow (much longer time)
            (self.very_slow_linear_velocity, -self.base_angular_velocity, 12.0),  # 0.05 vs 0.15 baseline
            # Straight section - very fast (shorter time)
            (self.very_fast_linear_velocity, 0.0, 0.86),  # 0.35 vs 0.15 baseline
            # Final turn left - moderate
            (self.base_linear_velocity, self.base_angular_velocity, 2.0),
            # Stop
            (0.0, 0.0, 1.0)
        ]
        return commands

    def get_smooth_trajectory_commands(self):
        """Generate velocity commands for smooth random trajectory with velocity variations"""
        commands = [
            # Complex curved path with varying speeds and turns
            (self.slow_linear_velocity, 0.1, 3.75),           # Slight left curve - slow (longer time)
            (self.very_fast_linear_velocity, -0.2, 0.86),     # Right curve, very fast (shorter time)
            (self.very_slow_linear_velocity, 0.4, 9.0),       # Sharp left turn - very slow (much longer time)
            (self.fast_linear_velocity, -0.1, 1.2),           # Slight right - fast (shorter time)
            (self.base_linear_velocity, 0.3, 2.0),            # Left curve - moderate (baseline)
            (self.slow_linear_velocity, -0.4, 3.75),          # Sharp right turn - slow (longer time)
            (self.very_fast_linear_velocity, 0.0, 0.86),      # Straight - very fast (shorter time)
            (self.base_linear_velocity, 0.2, 3.0),            # Gentle left curve - moderate (baseline)
            # Stop
            (0.0, 0.0, 1.0)
        ]
        return commands

def main():
    try:
        node = PathFollowerNode()
        
        rospy.loginfo("Velocity-based path follower node with data recording is running. Press Ctrl+C to stop.")
        
        # Execute path 2 once and wait for completion
        path_executed = False
        
        while not rospy.is_shutdown():
            try:
                # Execute path 2 only once
                if not path_executed and not node.path_completed:
                    user_input = 4
                    path_num = int(user_input)
                    if path_num in [1, 2, 3, 4, 8]:
                        node.select_path(path_num)
                        path_executed = True
                        rospy.loginfo(f"Path {path_num} started with data recording. Waiting for completion...")
                
                # Wait for path completion
                if path_executed and node.path_completed:
                    rospy.loginfo("Path execution completed. Robot stopped. Data saved.")
                    break
                
                rospy.sleep(0.5)

            except ValueError:
                print("Please enter a valid number.")
            except Exception as e:
                print(f"Error in main loop: {e}")
                break
        
        rospy.signal_shutdown("Path execution completed")
    except rospy.ROSInterruptException:
        pass
    finally:
        # Ensure robot stops if node is shut down
        if 'node' in locals() and isinstance(node, PathFollowerNode):
            node.send_velocity(0.0, 0.0)
        rospy.loginfo("Path follower node shut down.")

if __name__ == "__main__":
    main()
