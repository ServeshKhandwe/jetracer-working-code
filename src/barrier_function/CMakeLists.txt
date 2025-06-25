cmake_minimum_required(VERSION 2.8.3)
project(barrier_function)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  custom_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy geometry_msgs nav_msgs sensor_msgs std_msgs custom_msgs
)

install(PROGRAMS
  motion_capture.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)