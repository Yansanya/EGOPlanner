cmake_minimum_required(VERSION 3.0.2)
project(uav_ros_http_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/uav_ros_http_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
