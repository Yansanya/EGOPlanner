#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTP bridge for UAVbot (nanobot tools in UAVbot/nanobot/agent/tools/uav.py).

Implements the endpoints FlyToTool / GetPositionTool / GetDroneStateTool expect:
  POST /fly_to     JSON {"x","y"} optional "z" -> publishes std_msgs/String JSON on goal_json_topic
  GET  /position   -> last odometry (ENU), yaw from quaternion
  GET  /state      -> status + arrived + optional ego_planner (JSON from /ego_planner/fsm_status)
  GET  /capture    -> not implemented here (use AirSim / separate node); returns ok: false

Run alongside: roslaunch llm_goal_bridge llm_goal_bridge.launch
Set odom_topic to match EGO (e.g. /odom_world or AirSim odom remap).
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional, Tuple

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def _yaw_from_orientation(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RosHttpBridge:
    def __init__(self) -> None:
        rospy.init_node("uav_ros_http_bridge", disable_signals=False)
        self._lock = threading.Lock()
        self._odom = None  # type: Optional[Odometry]
        self._goal_xyz = None  # type: Optional[Tuple[float, float, float]]
        self._goal_active = False
        self._ego_planner_fsm = None  # type: Optional[dict[str, Any]]
        self._arrival_radius = float(rospy.get_param("~arrival_radius", 0.75))
        odom_topic = rospy.get_param("~odom_topic", "/odom_world")
        goal_topic = rospy.get_param("~goal_json_topic", "/llm_goal_bridge/goal_json")
        planner_fsm_topic = rospy.get_param(
            "~planner_fsm_topic", "/ego_planner/fsm_status"
        )
        self._host = rospy.get_param("~listen_host", "0.0.0.0")
        self._port = int(rospy.get_param("~listen_port", 8765))

        self._pub_goal = rospy.Publisher(goal_topic, String, queue_size=2)
        rospy.Subscriber(odom_topic, Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber(
            planner_fsm_topic, String, self._on_planner_fsm, queue_size=2
        )

        rospy.loginfo(
            "uav_ros_http_bridge: HTTP http://%s:%s -> goal %s, odom %s, planner_fsm %s",
            self._host,
            self._port,
            goal_topic,
            odom_topic,
            planner_fsm_topic,
        )

    def _on_odom(self, msg: Odometry) -> None:
        with self._lock:
            self._odom = msg

    def _on_planner_fsm(self, msg: String) -> None:
        raw = (msg.data or "").strip()
        if not raw:
            return
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                with self._lock:
                    self._ego_planner_fsm = parsed
        except ValueError:
            rospy.logwarn_throttle(
                5.0, "uav_ros_http_bridge: invalid planner FSM JSON: %s", raw[:200]
            )

    def fly_to(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            x = float(body["x"])
            y = float(body["y"])
        except (KeyError, TypeError, ValueError) as e:
            return 400, {"error": "need_numeric_x_y", "detail": str(e)}

        z_val = body.get("z", None)
        with self._lock:
            if z_val is None:
                if self._odom is None:
                    return 503, {"error": "no_odom_cannot_infer_z"}
                z = float(self._odom.pose.pose.position.z)
            else:
                z = float(z_val)
            self._goal_xyz = (x, y, z)
            self._goal_active = True

        payload = json.dumps({"x": x, "y": y, "z": z})
        self._pub_goal.publish(String(data=payload))
        rospy.loginfo("fly_to published goal_json: %s", payload)
        return 200, {"goal": {"x": x, "y": y, "z": z}}

    def position(self) -> dict[str, Any]:
        with self._lock:
            if self._odom is None:
                return {"has_odom": False, "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
            p = self._odom.pose.pose.position
            q = self._odom.pose.pose.orientation
            return {
                "has_odom": True,
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "yaw": _yaw_from_orientation(q),
            }

    def state(self) -> dict[str, Any]:
        with self._lock:
            ego = self._ego_planner_fsm
            if self._odom is None:
                return {
                    "status": "NO_ODOM",
                    "arrived": False,
                    "goal_active": self._goal_active,
                    "ego_planner": ego,
                }
            if not self._goal_active or self._goal_xyz is None:
                return {
                    "status": "IDLE",
                    "arrived": False,
                    "goal_active": False,
                    "ego_planner": ego,
                }
            p = self._odom.pose.pose.position
            gx, gy, gz = self._goal_xyz
            dx = p.x - gx
            dy = p.y - gy
            dz = p.z - gz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            arrived = dist < self._arrival_radius
            st = "ARRIVED" if arrived else "FLYING"
            return {
                "status": st,
                "arrived": arrived,
                "goal_active": True,
                "distance_m": dist,
                "ego_planner": ego,
            }


def _make_handler(bridge: RosHttpBridge):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            rospy.logdebug(fmt, *args)

        def _send_json(self, code: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/position":
                self._send_json(200, bridge.position())
            elif path == "/state":
                self._send_json(200, bridge.state())
            elif path == "/capture":
                self._send_json(
                    200,
                    {
                        "ok": False,
                        "error": "capture_not_implemented_in_uav_ros_http_bridge",
                    },
                )
            else:
                self._send_json(404, {"error": "not_found", "path": path})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != "/fly_to":
                self._send_json(404, {"error": "not_found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = 0
            raw = self.rfile.read(n) if n else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except ValueError:
                self._send_json(400, {"error": "invalid_json"})
                return
            code, resp = bridge.fly_to(body)
            self._send_json(code, resp)

    return H


def main() -> None:
    bridge = RosHttpBridge()
    handler = _make_handler(bridge)
    server = HTTPServer((bridge._host, bridge._port), handler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    rospy.loginfo("HTTP server thread started (daemon).")
    rospy.spin()


if __name__ == "__main__":
    main()
