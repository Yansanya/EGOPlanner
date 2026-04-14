cmake_minimum_required(VERSION 2.8.3)
project(plan_env)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  visualization_msgs
  cv_bridge
  message_filters
)

find_package(Eigen3 REQUIRED)

# Boost configuration for PCL（添加这部分）
set(BOOST_ROOT /usr)
set(BOOST_LIBRARYDIR /usr/lib/x86_64-linux-gnu)
set(Boost_NO_SYSTEM_PATHS OFF)
set(Boost_USE_STATIC_LIBS OFF)
find_package(Boost 1.65.1 EXACT REQUIRED COMPONENTS system filesystem thread chrono)

find_package(PCL 1.7 REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES plan_env
 CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}  # 添加这行
)

link_directories(${PCL_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS})  # 添加 ${Boost_LIBRARY_DIRS}

add_library( plan_env 
    src/grid_map.cpp 
    src/raycast.cpp
    )
target_link_libraries( plan_env
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}  # 添加这行
    )