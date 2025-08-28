cmake_minimum_required(VERSION 3.0.2)
project(openai_mocap_controller)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/openai_mocap_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)