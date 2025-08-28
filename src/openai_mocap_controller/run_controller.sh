#!/bin/bash

# OpenAI Mocap Controller Launch Script

echo "=========================================="
echo "OpenAI Mocap Controller for Jetracer"
echo "=========================================="

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set!"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

echo "✓ OpenAI API key found"

# Source ROS environment
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash

echo "✓ ROS environment sourced"

# Build the package if needed
echo "Building openai_mocap_controller package..."
cd ~/catkin_ws
catkin_make --pkg openai_mocap_controller

if [ $? -eq 0 ]; then
    echo "✓ Package built successfully"
else
    echo "✗ Package build failed"
    exit 1
fi

# Launch the controller
echo "Launching OpenAI Mocap Controller..."
echo "Models to test: gpt-4, gpt-3.5-turbo, gpt-4-turbo"
echo "Test duration per model: 60 seconds"
echo "Goal position: (0, 0)"
echo ""

roslaunch openai_mocap_controller openai_mocap_controller.launch