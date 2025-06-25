#!/home/jetson/ros_venv/bin/python3
import rospy
from geometry_msgs.msg import Twist
import math
import time
import numpy as np

class SimplePathFollower:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('simple_path_follower', anonymous=True)
        
        # Publisher for cmd_vel
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Control parameters
        self.linear_speed = 0.3  # m/s
        self.max_angular_speed = 0.5  # rad/s
        
        # Path following parameters
        self.lookahead_distance = 0.5  # meters
        self.path_scale = 0.5  # Scale down paths for indoor use
        self.waypoint_tolerance = 0.2  # meters
        
        # Current position simulation (in real robot, get from odometry)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # Path selection
        self.path_type = "path1"  # Options: "path1", "path2", "path3", "path4", "path8"
        
        # Timer for control loop
        self.rate = rospy.Rate(20)  # 20 Hz for smoother control
        
        rospy.loginfo(f"Simple Path Follower initialized - Path: {self.path_type}")

    # Path generation functions from test.py
    def get_path1(self):
        """Generates coordinates for Path 1 (vertical S-shape)."""
        t = np.linspace(0, 2 * np.pi, 100)  # Reduced points for real-time
        x = np.sin(t) * self.path_scale
        y = t * self.path_scale * 0.5  # Scale down y direction
        return x, y

    def get_path2(self):
        """Generates coordinates for Path 2 (Square wave) - modified for car."""
        # Simplified square wave with smoother transitions
        points_per_segment = 20
        x = []
        y = []
        
        # Create a smoother square wave pattern
        segments = [
            (0, 0, 1*self.path_scale, 0),  # horizontal
            (1*self.path_scale, 0, 1*self.path_scale, 1*self.path_scale),  # vertical
            (1*self.path_scale, 1*self.path_scale, 2*self.path_scale, 1*self.path_scale),  # horizontal
            (2*self.path_scale, 1*self.path_scale, 2*self.path_scale, 0),  # vertical down
        ]
        
        for x1, y1, x2, y2 in segments:
            x_seg = np.linspace(x1, x2, points_per_segment)
            y_seg = np.linspace(y1, y2, points_per_segment)
            x.extend(x_seg)
            y.extend(y_seg)
        
        return np.array(x), np.array(y)

    def get_path3(self):
        """Generates coordinates for Path 3 (P-shape) - modified for car."""
        Y_start = 0.0
        Y_loop_connect = 1.0 * self.path_scale
        X_start = 0.0
        R = 0.3 * self.path_scale  # Smaller radius
        
        points_per_segment = 30
        points_in_loop = 50

        # Stem
        x_line = np.full(points_per_segment, X_start)
        y_line = np.linspace(Y_start, Y_loop_connect, points_per_segment)

        # Loop - made larger for car turning radius
        center_x, center_y = X_start + R, Y_loop_connect
        t_loop = np.linspace(np.pi, -np.pi, points_in_loop) 
        x_loop = center_x + R * np.cos(t_loop)
        y_loop = center_y + R * np.sin(t_loop)
        
        x = np.concatenate((x_line, x_loop))
        y = np.concatenate((y_line, y_loop))
        
        return x, y

    def get_path4(self):
        """Generates coordinates for Path 4 (horizontal S-shape)."""
        t = np.linspace(0, 2 * np.pi, 100)
        x = t * self.path_scale * 0.5
        y = np.sin(t) * self.path_scale
        return x, y

    def get_path8(self):
        """Generates coordinates for Path 8 (smooth trajectory)."""
        x = np.linspace(0, 3 * np.pi * self.path_scale, 100)
        y = (1.5 * np.sin(x * 0.7 / self.path_scale + np.pi/6) + 
             1.0 * np.cos(x * 1.3 / self.path_scale - np.pi/3) + 
             0.8 * np.sin(x * 2.2 / self.path_scale + np.pi/2)) * self.path_scale * 0.3
        return x, y

    def get_current_path(self):
        """Get the coordinates for the selected path."""
        if self.path_type == "path1":
            return self.get_path1()
        elif self.path_type == "path2":
            return self.get_path2()
        elif self.path_type == "path3":
            return self.get_path3()
        elif self.path_type == "path4":
            return self.get_path4()
        elif self.path_type == "path8":
            return self.get_path8()
        else:
            rospy.logwarn(f"Unknown path type: {self.path_type}")
            return self.get_path1()  # Default

    def find_nearest_waypoint(self, path_x, path_y):
        """Find the nearest waypoint on the path."""
        distances = np.sqrt((path_x - self.current_x)**2 + (path_y - self.current_y)**2)
        nearest_idx = np.argmin(distances)
        return nearest_idx

    def find_lookahead_waypoint(self, path_x, path_y, start_idx):
        """Find a waypoint at lookahead distance from current position."""
        for i in range(start_idx, len(path_x)):
            distance = np.sqrt((path_x[i] - self.current_x)**2 + (path_y[i] - self.current_y)**2)
            if distance >= self.lookahead_distance:
                return i
        return len(path_x) - 1  # Return last waypoint if none found

    def calculate_steering_angle(self, target_x, target_y):
        """Calculate steering angle to reach target point."""
        # Calculate angle to target
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        target_angle = math.atan2(dy, dx)
        
        # Calculate steering angle (difference from current heading)
        angle_diff = target_angle - self.current_yaw
        
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        return angle_diff

    def update_position(self, linear_vel, angular_vel, dt):
        """Update simulated position (replace with odometry in real robot)."""
        self.current_yaw += angular_vel * dt
        self.current_x += linear_vel * math.cos(self.current_yaw) * dt
        self.current_y += linear_vel * math.sin(self.current_yaw) * dt

    def execute_path_following(self):
        """Execute path following with lookahead control."""
        rospy.loginfo(f"Starting path following for {self.path_type}...")
        
        # Get path coordinates
        path_x, path_y = self.get_current_path()
        
        if len(path_x) == 0:
            rospy.logwarn("Empty path!")
            return
            
        current_waypoint_idx = 0
        dt = 1.0 / 20.0  # Control loop time step
        
        while current_waypoint_idx < len(path_x) - 1 and not rospy.is_shutdown():
            # Find nearest waypoint
            nearest_idx = self.find_nearest_waypoint(path_x, path_y)
            
            # Find lookahead waypoint
            lookahead_idx = self.find_lookahead_waypoint(path_x, path_y, nearest_idx)
            
            # Calculate steering
            steering_angle = self.calculate_steering_angle(path_x[lookahead_idx], path_y[lookahead_idx])
            
            # Limit angular velocity based on steering angle
            angular_vel = np.clip(steering_angle * 2.0, -self.max_angular_speed, self.max_angular_speed)
            
            # Reduce linear speed for sharp turns
            speed_factor = 1.0 - min(abs(angular_vel) / self.max_angular_speed, 0.7)
            linear_vel = self.linear_speed * speed_factor
            
            # Create and publish command
            cmd = Twist()
            cmd.linear.x = linear_vel
            cmd.angular.z = angular_vel
            self.cmd_vel_pub.publish(cmd)
            
            # Update position simulation
            self.update_position(linear_vel, angular_vel, dt)
            
            # Check if we reached the current waypoint
            distance_to_target = np.sqrt((path_x[lookahead_idx] - self.current_x)**2 + 
                                       (path_y[lookahead_idx] - self.current_y)**2)
            
            if distance_to_target < self.waypoint_tolerance:
                current_waypoint_idx = lookahead_idx
                
            rospy.loginfo_throttle(1.0, f"Following waypoint {current_waypoint_idx}/{len(path_x)-1}, "
                                       f"pos: ({self.current_x:.2f}, {self.current_y:.2f})")
            
            self.rate.sleep()

    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def run(self):
        """Main execution function"""
        rospy.loginfo("Waiting 3 seconds before starting path...")
        rospy.sleep(3.0)
        
        try:
            # Use new path following for all path types
            self.execute_path_following()
                
            rospy.loginfo("Path execution completed!")
            
        except rospy.ROSInterruptException:
            rospy.loginfo("Path execution interrupted")
        finally:
            # Always stop the robot when done
            self.stop_robot()
            rospy.loginfo("Robot stopped")

def main():
    try:
        path_follower = SimplePathFollower()
        path_follower.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Simple path follower interrupted")

if __name__ == '__main__':
    main()
