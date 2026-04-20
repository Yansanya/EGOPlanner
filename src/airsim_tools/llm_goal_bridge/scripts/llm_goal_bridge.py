#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Subscribe goals from an LLM / external agent, publish nav_msgs/Path for EGO-Planner
(fsm/flight_type=1 subscribes to /waypoint_generator/waypoints).

Inputs:
  ~goal_json  (std_msgs/String)  JSON object, e.g. {"x":1.0,"y":2.0,"z":1.5}
                                z optional; if omitted or null, z is set to 0.0
                                (EGO uses odom height when z <= 0.1).
  ~goal       (geometry_msgs/PoseStamped)  optional direct goal (same as RViz goal).

Output:
  ~output_topic (nav_msgs/Path, default /waypoint_generator/waypoints)
"""

import json
import math
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class LlmGoalBridge(object):
    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.output_topic = rospy.get_param("~output_topic", "/waypoint_generator/waypoints")
        self.x_min = float(rospy.get_param("~x_min", -500.0))
        self.x_max = float(rospy.get_param("~x_max", 500.0))
        self.y_min = float(rospy.get_param("~y_min", -500.0))
        self.y_max = float(rospy.get_param("~y_max", 500.0))
        self.z_min = float(rospy.get_param("~z_min", -50.0))
        self.z_max = float(rospy.get_param("~z_max", 200.0))

        self.pub = rospy.Publisher(self.output_topic, Path, queue_size=1, latch=False)
        rospy.Subscriber("~goal_json", String, self._on_goal_json, queue_size=10)
        rospy.Subscriber("~goal", PoseStamped, self._on_goal_pose, queue_size=10)

        rospy.loginfo(
            "[llm_goal_bridge] output=%s frame=%s (JSON on ~goal_json, PoseStamped on ~goal)",
            self.output_topic,
            self.frame_id,
        )

    def _publish_path(self, x, y, z, log_prefix=""):
        x = clamp(float(x), self.x_min, self.x_max)
        y = clamp(float(y), self.y_min, self.y_max)
        z = clamp(float(z), self.z_min, self.z_max)

        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame_id
        ps = PoseStamped()
        ps.header = path.header
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.w = 1.0
        path.poses.append(ps)

        self.pub.publish(path)
        rospy.loginfo("%sPublished goal Path (x,y,z)=(%.3f, %.3f, %.3f) -> %s", log_prefix, x, y, z, self.output_topic)

    def _on_goal_json(self, msg):
        raw = (msg.data or "").strip()
        if not raw:
            rospy.logwarn("[llm_goal_bridge] empty goal_json")
            return
        try:
            data = json.loads(raw)
        except ValueError as e:
            rospy.logwarn("[llm_goal_bridge] invalid JSON: %s", e)
            return

        if not isinstance(data, dict):
            rospy.logwarn("[llm_goal_bridge] JSON root must be an object")
            return

        if "goal" in data and isinstance(data["goal"], dict):
            data = data["goal"]

        try:
            x = data["x"]
            y = data["y"]
        except KeyError as e:
            rospy.logwarn("[llm_goal_bridge] missing key: %s", e)
            return

        z = data.get("z", 0.0)
        if z is None or (isinstance(z, float) and math.isnan(z)):
            z = 0.0

        self._publish_path(x, y, z, log_prefix="[json] ")

    def _on_goal_pose(self, msg):
        p = msg.pose.position
        self._publish_path(p.x, p.y, p.z, log_prefix="[pose] ")


def main():
    rospy.init_node("llm_goal_bridge")
    LlmGoalBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
