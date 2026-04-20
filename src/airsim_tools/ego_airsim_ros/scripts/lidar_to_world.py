#!/usr/bin/env python3
"""
Transform LiDAR point cloud from sensor local frame to world frame.

Uses odometry data directly (no TF dependency) to transform LiDAR points
from the drone's body/sensor frame into the world frame that EGO-Planner
expects in its cloudCallback.

Transformation: p_world = R(q_odom) * p_lidar + t_odom
"""

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header


class LidarToWorld:
    def __init__(self):
        rospy.init_node("lidar_to_world")

        self.odom_pos = None
        self.odom_rot = None  # 3x3 rotation matrix

        odom_topic = rospy.get_param(
            "~odom_topic", "/airsim_node/Drone1/odom_local_enu"
        )
        lidar_topic = rospy.get_param(
            "~input_topic", "/airsim_node/Drone1/lidar/Lidar1"
        )
        self.target_frame = rospy.get_param("~target_frame", "world")

        self.pub = rospy.Publisher("/lidar_world", PointCloud2, queue_size=5)

        rospy.Subscriber(odom_topic, Odometry, self._odom_cb, queue_size=5)
        rospy.Subscriber(lidar_topic, PointCloud2, self._lidar_cb, queue_size=5)

        rospy.loginfo("LiDAR transformer ready (no TF, using odom directly)")
        rospy.loginfo("  Odom : %s", odom_topic)
        rospy.loginfo("  LiDAR: %s", lidar_topic)
        rospy.loginfo("  Out  : /lidar_world [frame: %s]", self.target_frame)

    @staticmethod
    def _quat_to_rot(q):
        """Quaternion (x,y,z,w) to 3x3 rotation matrix."""
        x, y, z, w = q
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ])

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_pos = np.array([p.x, p.y, p.z])
        self.odom_rot = self._quat_to_rot([q.x, q.y, q.z, q.w])

    def _lidar_cb(self, msg):
        if self.odom_pos is None:
            return

        points_local = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32,
        )

        if len(points_local) == 0:
            return

        # p_world = R * p_local + t
        points_world = (self.odom_rot @ points_local.T).T + self.odom_pos

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.target_frame

        cloud_msg = pc2.create_cloud_xyz32(header, points_world.tolist())
        self.pub.publish(cloud_msg)


if __name__ == "__main__":
    try:
        LidarToWorld()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
