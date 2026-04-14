cmake_minimum_required(VERSION 2.8.3)
project(map_generator)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  pcl_conversions
)

# 明确指定 Boost 库路径
set(BOOST_ROOT /usr)
set(BOOST_LIBRARYDIR /usr/lib/x86_64-linux-gnu)
set(Boost_NO_SYSTEM_PATHS OFF)
set(Boost_USE_STATIC_LIBS OFF)

# 查找 Boost
find_package(Boost 1.65.1 EXACT REQUIRED COMPONENTS system filesystem thread chrono)

# 然后找 PCL
find_package(PCL 1.7 REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_executable (random_forest src/random_forest_sensing.cpp )
target_link_libraries(random_forest
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)