#!/usr/bin/env python
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

import base64
import json
import math
import threading
try:
    from http.server import BaseHTTPRequestHandler, HTTPServer
except ImportError:
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def _yaw_from_orientation(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RosHttpBridge:
    def __init__(self):
        rospy.init_node("uav_ros_http_bridge", disable_signals=False)
        self._lock = threading.Lock()
        self._odom = None
        self._goal_xyz = None
        self._goal_active = False
        self._ego_planner_fsm = None
        self._arrival_radius = float(rospy.get_param("~arrival_radius", 0.75))
        odom_topic = rospy.get_param("~odom_topic", "/odom_world")
        goal_topic = rospy.get_param("~goal_json_topic", "/llm_goal_bridge/goal_json")
        planner_fsm_topic = rospy.get_param(
            "~planner_fsm_topic", "/ego_planner/fsm_status"
        )
        self._host = rospy.get_param("~listen_host", "0.0.0.0")
        self._port = int(rospy.get_param("~listen_port", 8765))
        self._airsim_host = rospy.get_param("~airsim_host", "localhost")
        self._airsim_vehicle = rospy.get_param("~airsim_vehicle", "Drone1")
        self._airsim_camera = rospy.get_param("~airsim_camera", "front_center")
        self._airsim_client = None

        self._pub_goal = rospy.Publisher(goal_topic, String, queue_size=2)
        rospy.Subscriber(odom_topic, Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber(
            planner_fsm_topic, String, self._on_planner_fsm, queue_size=2
        )

        rospy.loginfo(
            "uav_ros_http_bridge: HTTP http://%s:%s -> goal %s, odom %s, planner_fsm %s, airsim %s/%s",
            self._host,
            self._port,
            goal_topic,
            odom_topic,
            planner_fsm_topic,
            self._airsim_host,
            self._airsim_vehicle,
        )

    def _get_airsim_client(self):
        if self._airsim_client is not None:
            return self._airsim_client
        try:
            import airsim
        except Exception as e:
            return None, "airsim_import_failed: %s" % e
        try:
            c = airsim.MultirotorClient(ip=self._airsim_host)
            c.confirmConnection()
            self._airsim_client = c
            return c, None
        except Exception as e:
            return None, "airsim_connect_failed: %s" % e

    def capture(self):
        client, err = self._get_airsim_client()
        if client is None:
            return {"ok": False, "error": err or "airsim_unavailable"}
        try:
            import airsim
            # Request compressed bytes so image_data_uint8 can be opened as PNG directly.
            req = airsim.ImageRequest(self._airsim_camera, airsim.ImageType.Scene, False, True)
            res = client.simGetImages([req], vehicle_name=self._airsim_vehicle)
            if not res:
                return {"ok": False, "error": "airsim_empty_response"}
            img = res[0]
            data = img.image_data_uint8
            if not data:
                return {"ok": False, "error": "airsim_empty_image"}
            b64 = base64.b64encode(bytearray(data)).decode("ascii")
            return {
                "ok": True,
                "width": int(getattr(img, "width", 0)),
                "height": int(getattr(img, "height", 0)),
                "image_base64": b64,
                "source": "airsim_http_bridge",
            }
        except Exception as e:
            return {"ok": False, "error": "airsim_capture_failed: %s" % e}

    def _on_odom(self, msg):
        with self._lock:
            self._odom = msg

    def _on_planner_fsm(self, msg):
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

    def fly_to(self, body):
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

    def position(self):
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

    def state(self):
        with self._lock:
            ego = self._ego_planner_fsm
            if self._odom is None:
                return {
                    "status": "NO_ODOM",
                    "arrived": False,
                    "goal_active": self._goal_active,
                    "ego_planner": ego,
                }
            v = self._odom.twist.twist.linear
            speed_mps = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
            if not self._goal_active or self._goal_xyz is None:
                return {
                    "status": "HOVERING" if speed_mps < 0.2 else "MOVING",
                    "arrived": False,
                    "goal_active": False,
                    "speed_mps": speed_mps,
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
            if arrived:
                # Mark goal complete so subsequent /state reflects task completion.
                self._goal_active = False
            return {
                "status": st,
                "arrived": arrived,
                "goal_active": self._goal_active,
                "distance_m": dist,
                "speed_mps": speed_mps,
                "ego_planner": ego,
            }


def _make_handler(bridge):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            rospy.logdebug(fmt, *args)

        def _send_json(self, code, obj):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/position":
                self._send_json(200, bridge.position())
            elif path == "/state":
                self._send_json(200, bridge.state())
            elif path == "/capture":
                self._send_json(200, bridge.capture())
            else:
                self._send_json(404, {"error": "not_found", "path": path})

        def do_POST(self):
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


def main():
    bridge = RosHttpBridge()
    handler = _make_handler(bridge)
    server = HTTPServer((bridge._host, bridge._port), handler)
    th = threading.Thread(target=server.serve_forever)
    th.daemon = True
    th.start()
    rospy.loginfo("HTTP server thread started (daemon).")
    rospy.spin()


if __name__ == "__main__":
    main()
