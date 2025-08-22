"""
llm_agent.py

This module implements the LLMAgent class that interfaces with ChatGPT API
to generate navigation plans for the robot. The agent takes robot state,
sensor data, and goal information to generate multi-step velocity commands.
"""

import openai
import json
import re
import numpy as np
import math
import logging
import rospy
from typing import List, Dict, Any, Optional

# Enhanced logging setup
logger = logging.getLogger(__name__)

class LLMAgent:
    """
    LLM agent that generates navigation plans using ChatGPT API.
    """
    
    def __init__(self, config: dict):
        rospy.loginfo("Initializing LLM Agent...")
        self.config = config
        
        # LLM configuration
        llm_config = config.get("llm", {})
        self.api_key = llm_config.get("api_key", "")
        self.model = llm_config.get("model", "gpt-4")
        self.max_retries = llm_config.get("max_retries", 3)
        self.base_url = llm_config.get("base_url", "https://api.groq.com/openai/v1")
        
        rospy.loginfo(f"LLM Configuration:")
        rospy.loginfo(f"  - Model: {self.model}")
        rospy.loginfo(f"  - Base URL: {self.base_url}")
        rospy.loginfo(f"  - Max retries: {self.max_retries}")
        
        # Set OpenAI API key
        if self.api_key and self.api_key != "your_openai_api_key":
            openai.api_key = self.api_key
            rospy.loginfo("✓ OpenAI API key configured")
        else:
            rospy.logwarn("⚠ No valid OpenAI API key provided in config")
            rospy.logwarn("  LLM functionality may be limited")
        
        # Current goal
        self.current_goal = None
        
        # Action constraints
        env_type = config.get("environment", {}).get("env_type", "turtlebot")
        action_constraints = config.get("safety", {}).get("action_constraints", {})
        turtlebot_constraints = action_constraints.get("turtlebot", {})
        
        self.linear_vel_bounds = turtlebot_constraints.get("longitudinal_velocity", [0.0, 0.3])
        self.angular_vel_bounds = turtlebot_constraints.get("angular_velocity", [-0.3, 0.3])
        
        rospy.loginfo(f"Action constraints:")
        rospy.loginfo(f"  - Linear velocity: {self.linear_vel_bounds} m/s")
        rospy.loginfo(f"  - Angular velocity: {self.angular_vel_bounds} rad/s")
        
        # Statistics
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        
        rospy.loginfo("✓ LLM Agent initialized successfully")

    def set_goal(self, goal_position: np.ndarray):
        """Set the current goal position."""
        self.current_goal = goal_position.copy()
        rospy.loginfo(f"LLM Agent goal updated: ({self.current_goal[0]:.3f}, {self.current_goal[1]:.3f})")

    def get_plan(self, robot_state: np.ndarray, obstacle_zonotopes: List, theta: float) -> List[Dict[str, float]]:
        """
        Generate a navigation plan using the LLM.
        
        Args:
            robot_state: Current robot state [physical_state, lidar, goal_vector]
            obstacle_zonotopes: List of obstacle zonotopes
            
        Returns:
            List of velocity commands [{"linear_velocity": v, "angular_velocity": w}, ...]
        """
        self.total_queries += 1
        
        try:
            rospy.logdebug(f"LLM Agent generating plan (query #{self.total_queries})")
            
            # Prepare sensor data
            sensor_data = self._prepare_sensor_data(robot_state, obstacle_zonotopes)
            rospy.logdebug(f"Sensor data prepared: robot at ({sensor_data['robot_position']['x']:.3f}, {sensor_data['robot_position']['y']:.3f})")
            
            # Create prompt
            prompt = self._create_prompt(sensor_data, theta)
            rospy.logdebug("LLM prompt created")
            
            # Get response from LLM
            response = self._query_llm(prompt)
            rospy.logdebug("LLM response received")
            
            # Parse response
            plan = self._parse_response(response)
            
            if plan and len(plan) > 0:
                self.successful_queries += 1
                rospy.logdebug(f"✓ LLM plan generated successfully ({len(plan)} steps)")
                return plan
            else:
                rospy.logwarn("⚠ LLM returned empty plan, using emergency plan")
                self.failed_queries += 1
                return self._get_emergency_plan()
            
        except Exception as e:
            self.failed_queries += 1
            rospy.logerr(f"✗ LLM Agent plan generation failed: {e}")
            rospy.logerr(f"  Query #{self.total_queries} failed")
            rospy.logerr(f"  Success rate: {self.successful_queries}/{self.total_queries} ({100*self.successful_queries/self.total_queries:.1f}%)")
            logger.error(f"Full error details: {e}", exc_info=True)
            return self._get_emergency_plan()

    def _prepare_sensor_data(self, robot_state: np.ndarray, obstacle_zonotopes: List) -> Dict[str, Any]:
        """Prepare sensor data for the LLM prompt."""
        try:
            # Extract components from robot state
            if len(robot_state) < 30:
                rospy.logwarn(f"⚠ Robot state vector too short: {len(robot_state)} (expected ≥30)")
                # Pad with zeros if necessary
                padded_state = np.zeros(30)
                padded_state[:len(robot_state)] = robot_state
                robot_state = padded_state
            
            physical_state = robot_state[:8]  # [x, y, theta, v, ex1, ex2, ex3, ex4]
            lidar_readings = robot_state[8:26]  # 18 LiDAR readings
            goal_vector = robot_state[26:30]  # [dx, dy, distance, angle]
            
            x_pos, y_pos, theta, velocity = physical_state[:4]
            goal_dx, goal_dy, goal_distance, goal_angle = goal_vector
            
            # Validate values
            if not all(np.isfinite([x_pos, y_pos, theta, velocity])):
                rospy.logwarn("⚠ Invalid values in robot state, using defaults")
                x_pos = y_pos = theta = velocity = 0.0
            
            if not np.isfinite(goal_distance) or goal_distance < 0:
                rospy.logwarn("⚠ Invalid goal distance, using default")
                goal_distance = 1.0
                
            # Process LiDAR data
            lidar_list = []
            for reading in lidar_readings:
                if np.isfinite(reading) and reading > 0:
                    lidar_list.append(float(reading))
                else:
                    lidar_list.append(3.5)  # Default max range
            
            # Calculate goal position
            goal_x = x_pos + goal_dx
            goal_y = y_pos + goal_dy
            
            LOS = math.atan2(goal_y - y_pos, goal_x - x_pos)
            
            data = {
                "robot_position": {"x": float(x_pos), "y": float(y_pos)},
                "robot_orientation": {"theta": float(theta)},
                "robot_velocity": {"linear": float(velocity)},
                "goal_position": {"x": float(goal_x), "y": float(goal_y)},
                "goal_distance": float(goal_distance),
                "goal_angle": float(goal_angle),
                "line_of_sight": float(LOS),
                "lidar_readings": lidar_list,
                "velocity_bounds": {
                    "linear": self.linear_vel_bounds,
                    "angular": self.angular_vel_bounds
                }
            }
            
            return data
            
        except Exception as e:
            rospy.logerr(f"✗ Failed to prepare sensor data: {e}")
            logger.error(f"Sensor data preparation error: {e}", exc_info=True)
            raise

    def _create_prompt(self, sensor_data: Dict[str, Any], theta: float) -> str:
        """Create the prompt for the LLM."""
        robot_pos = sensor_data["robot_position"]
        goal_pos = sensor_data["goal_position"]
        
        goal_distance = sensor_data["goal_distance"]
        LOS = sensor_data["line_of_sight"]
        lidar_readings = sensor_data["lidar_readings"]
        linear_bounds = sensor_data["velocity_bounds"]["linear"]
        angular_bounds = sensor_data["velocity_bounds"]["angular"]
        
        print(f"theta: {theta}, LOS: {LOS}")
        
        prompt = f"""You are the motion controller of a 2D differential drive robot operating in a SafeLLMRA framework. 
Generate control inputs (linear and angular velocities) to move the robot to the target position.

Current robot state:
- Position: x={robot_pos['x']:.3f} mm, y={robot_pos['y']:.3f} mm
- Orientation: theta={theta:.3f} radians
- Goal position: x={goal_pos['x']:.3f} mm, y={goal_pos['y']:.3f} mm
- Distance to goal: {goal_distance:.3f} mm
- Line of Sight: {LOS:.3f} radians

For example, 
the goal position is always at the origin (0, 0) in the robot's local frame, and the robot's current position is given in millimeters.
for example if the robot is at (0.561, 0.307) and since the goal is at (0, 0), the robot should move towards the origin. For that it needs to compute the angle to the goal and adjust its orientation accordingly. It should ideal move forward and turn left towards the origin.
just for an example:
if the robot is at (-0.853,0.338) then you need to move forward and turn right towards the origin.
if the robot is at (-0.903, -0.329) then you need to move forward and turn left towards the origin.
if the robot is at (0.650, 0.3385) then you need to move forward and turn right towards the origin.

Remeber that angular velocity is negative for left turns and positive for right turns.

The robot is controlled by:
- Linear velocity in [{linear_bounds[0]}, {linear_bounds[1]}] m/s
- Angular velocity in [{angular_bounds[0]}, {angular_bounds[1]}] rad/s (if angular value is positive = robot will do right turn, and if the angular velocity is negative = The robot will do left turn)

# IMPORTANT: To align the robot with the goal, compute the angular error as (LOS - theta). If this value is negative, turn left (negative angular velocity). If positive, turn right (positive angular velocity).

Generate a 3-step plan to reach the goal efficiently while avoiding obstacles.
Each step will be executed for approximately 1.0 seconds.

Guidelines:
1. Move towards the goal
2. Balance angular adjustments with forward progress
3. When the LOS and orientations is within ±0.3 radians, prioritize forward movement
4. Use smooth velocity changes between steps
5. Consider the robot dynamics and momentum
6. Be as verbose as possible in your reasoning and specifiy directions that each action should make the car move in

Output format (JSON only, no additional text):
{{
  "plan": [
    {{"linear_velocity": <value1>, "angular_velocity": <value1>}},
    {{"linear_velocity": <value2>, "angular_velocity": <value2>}},
    {{"linear_velocity": <value3>, "angular_velocity": <value3>}}
  ],
  "reasoning": "Brief explanation of the plan",
}}"""
        rospy.logdebug(f"LLM prompt created:\n{prompt[:500]}...")
        return prompt

      
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM API."""
        for attempt in range(self.max_retries):
            try:
                rospy.logdebug(f"LLM query attempt {attempt + 1}/{self.max_retries}")
                
                # Add format instruction to system message
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert robot navigation controller. Respond with compact, single-line JSON only - no additional text, formatting, or explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=5000,
                    temperature=0.01,
                    api_base=self.base_url
                )
                
                content = response.choices[0].message.content.strip()
                # Clean up any potential problematic characters
                content = content.replace('\n', ' ').replace('\r', '')
                rospy.loginfo(f"Full LLM Raw output: {content}")
                rospy.logdebug(f"✓ LLM query successful on attempt {attempt + 1}")
                return content
                
            except Exception as e:
                rospy.logwarn(f"⚠ LLM query attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                if "Connection" in str(e) or "timeout" in str(e).lower():
                    rospy.logwarn(f"  Network issue connecting to {self.base_url}")
                elif "API" in str(e) or "key" in str(e).lower():
                    rospy.logwarn(f"  API authentication issue")
                else:
                    rospy.logwarn(f"  General error: {type(e).__name__}")
                
                if attempt == self.max_retries - 1:
                    rospy.logerr(f"✗ All {self.max_retries} LLM query attempts failed")
                    raise
                    
                # Brief delay before retry
                import time
                time.sleep(0.5)

    def _parse_response(self, response: str) -> List[Dict[str, float]]:
        """Parse the LLM response to extract the velocity plan."""
        try:
            rospy.logdebug("Parsing LLM response...")
            
            # Try to extract JSON from the response
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                rospy.logdebug("Found JSON in code block")
            else:
                # Try to find JSON directly
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    rospy.logdebug("Found JSON directly")
                else:
                    json_str = response
                    rospy.logdebug("Using entire response as JSON")
            
            # Parse JSON
            parsed = json.loads(json_str)
            plan = parsed.get("plan", [])
            reasoning = parsed.get("reasoning", "No reasoning provided")
            
            rospy.loginfo(f"LLM reasoning: {reasoning}")
            
            if not plan:
                rospy.logwarn("⚠ No plan found in LLM response")
                return self._get_emergency_plan()
            
            # Validate and clamp velocities
            validated_plan = []
            for i, step in enumerate(plan):
                try:
                    linear_vel = float(step.get("linear_velocity", 0.0))
                    angular_vel = float(step.get("angular_velocity", 0.0))
                    
                    # Check for invalid values
                    if not np.isfinite(linear_vel):
                        rospy.logwarn(f"⚠ Invalid linear velocity in step {i}: {linear_vel}")
                        linear_vel = 0.0
                    if not np.isfinite(angular_vel):
                        rospy.logwarn(f"⚠ Invalid angular velocity in step {i}: {angular_vel}")
                        angular_vel = 0.0
                    
                    # Clamp to bounds
                    linear_vel_clamped = np.clip(linear_vel, self.linear_vel_bounds[0], self.linear_vel_bounds[1])
                    angular_vel_clamped = np.clip(angular_vel, self.angular_vel_bounds[0], self.angular_vel_bounds[1])
                    
                    # Log if clamping occurred
                    if abs(linear_vel - linear_vel_clamped) > 1e-6:
                        rospy.logdebug(f"Clamped linear velocity: {linear_vel:.3f} → {linear_vel_clamped:.3f}")
                    if abs(angular_vel - angular_vel_clamped) > 1e-6:
                        rospy.logdebug(f"Clamped angular velocity: {angular_vel:.3f} → {angular_vel_clamped:.3f}")
                    
                    validated_plan.append({
                        "linear_velocity": linear_vel_clamped,
                        "angular_velocity": angular_vel_clamped
                    })
                    
                except (KeyError, ValueError, TypeError) as e:
                    rospy.logwarn(f"⚠ Invalid step {i} in plan: {e}")
                    # Add safe step
                    validated_plan.append({"linear_velocity": 0.0, "angular_velocity": 0.0})
            
            if not validated_plan:
                rospy.logwarn("⚠ No valid steps after validation")
                return self._get_emergency_plan()
            
            rospy.logdebug(f"✓ Validated plan with {len(validated_plan)} steps")
            return validated_plan
            
        except json.JSONDecodeError as e:
            rospy.logerr(f"✗ JSON parsing failed: {e}")
            rospy.logerr(f"  Raw response: {response}...")
            return self._get_emergency_plan()
            
        except Exception as e:
            rospy.logerr(f"✗ Response parsing failed: {e}")
            rospy.logerr(f"  Response type: {type(response)}")
            rospy.logerr(f"  Response length: {len(response) if response else 0}")
            logger.error(f"Full parsing error: {e}", exc_info=True)
            return self._get_emergency_plan()

    def _get_emergency_plan(self) -> List[Dict[str, float]]:
        """Return a safe emergency plan (stop the robot)."""
        rospy.logwarn("⚠ Using emergency plan (stop robot)")
        return [
            {"linear_velocity": 0.0, "angular_velocity": 0.0},
            {"linear_velocity": 0.0, "angular_velocity": 0.0},
            {"linear_velocity": 0.0, "angular_velocity": 0.0}
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get LLM agent statistics."""
        success_rate = (self.successful_queries / self.total_queries * 100) if self.total_queries > 0 else 0
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": success_rate
        }
    
    def log_statistics(self):
        """Log current statistics."""
        stats = self.get_statistics()
        rospy.loginfo(f"LLM Agent Statistics:")
        rospy.loginfo(f"  - Total queries: {stats['total_queries']}")
        rospy.loginfo(f"  - Successful: {stats['successful_queries']}")
        rospy.loginfo(f"  - Failed: {stats['failed_queries']}")
        rospy.loginfo(f"  - Success rate: {stats['success_rate']:.1f}%")
