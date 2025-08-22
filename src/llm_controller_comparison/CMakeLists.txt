cmake_minimum_required(VERSION 3.0.2)
project(llm_controller_comparison)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  std_msgs
  tf
)

catkin_package(
  CATKIN_DEPENDS rospy geometry_msgs nav_msgs std_msgs tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/llm_controller_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
