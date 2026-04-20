cmake_minimum_required(VERSION 3.0.2)
project(ego_airsim_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  quadrotor_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/ego_to_airsim_bridge.py
  scripts/lidar_to_world.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
