#!/bin/bash

# 完全清除 conda 环境变量
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset PYTHONPATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Source ROS 环境
source /opt/ros/melodic/setup.bash

cd /media/tongji/data/uavbot/EGOPlanner

# 清理
rm -rf build devel

# 编译
catkin_make \
  -DBOOST_ROOT=/usr \
  -DBoost_NO_SYSTEM_PATHS=OFF \
  -DBoost_NO_BOOST_CMAKE=ON \
  -DBoost_DEBUG=OFF \
  -DCMAKE_BUILD_TYPE=Release

