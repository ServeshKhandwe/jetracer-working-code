#!/bin/bash

# Store original values
OLD_ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH
OLD_PYTHONPATH=$PYTHONPATH

# Source ROS and workspace
source /opt/ros/melodic/setup.bash
source ./devel/setup.bash

# Add project specific paths
export ROS_PACKAGE_PATH="$ROS_PACKAGE_PATH:$(pwd)/src"
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
