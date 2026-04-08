#!/usr/bin/env python3
"""
Trajectory Visualizer Node

Subscribes to odometry and publishes:
  1. /odom_visualization/path   - accumulated flight path (nav_msgs/Path)
  2. /odom_visualization/robot  - drone marker at current position (visualization_msgs/Marker)

These topics match the default RViz config so the trajectory is displayed automatically.
"""

import rospy
import math
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker


class TrajVisualizer:
    def __init__(self):
        rospy.init_node("traj_visualizer")

        odom_topic = rospy.get_param("~odom_topic", "/airsim_node/Drone1/odom_local_enu")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.min_dist = rospy.get_param("~min_distance", 0.05)

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.world_frame
        self.last_pos = None

        self.path_pub = rospy.Publisher("/odom_visualization/path", Path, queue_size=2)
        self.marker_pub = rospy.Publisher("/odom_visualization/robot", Marker, queue_size=2)

        rospy.Subscriber(odom_topic, Odometry, self._odom_cb, queue_size=5)
        rospy.loginfo("TrajVisualizer started, subscribing to %s", odom_topic)

    def _odom_cb(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        if self.last_pos is not None:
            dx = px - self.last_pos[0]
            dy = py - self.last_pos[1]
            dz = pz - self.last_pos[2]
            if math.sqrt(dx*dx + dy*dy + dz*dz) < self.min_dist:
                self._publish_marker(msg)
                return

        self.last_pos = (px, py, pz)

        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = self.world_frame
        pose.pose = msg.pose.pose
        self.path_msg.poses.append(pose)

        self.path_msg.header.stamp = msg.header.stamp
        self.path_pub.publish(self.path_msg)
        self._publish_marker(msg)

    def _publish_marker(self, odom_msg):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = odom_msg.header.stamp
        m.ns = "mesh"
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose = odom_msg.pose.pose
        m.scale.x = 0.8
        m.scale.y = 0.15
        m.scale.z = 0.15
        m.color.r = 1.0
        m.color.g = 0.2
        m.color.b = 0.0
        m.color.a = 1.0
        self.marker_pub.publish(m)


if __name__ == "__main__":
    try:
        node = TrajVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
