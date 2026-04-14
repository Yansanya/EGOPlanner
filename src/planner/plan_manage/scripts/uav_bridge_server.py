#!/usr/bin/env python3
"""
UAV Bridge Server

A lightweight HTTP API running in the ROS Python 3.8 environment that bridges
nanobot (Python 3.11 conda) with ROS + AirSim.

Endpoints:
  POST /fly_to          Send a navigation goal to EGO-Planner
  GET  /position        Get current drone position (ENU)
  GET  /state           Get drone FSM state and movement status
  GET  /arrived         Check if drone has arrived at goal
  GET  /capture         Capture camera image from AirSim (returns base64 PNG)

Usage:
  source ~/ego-planner/devel/setup.bash
  python3 uav_bridge_server.py [--port 8765] [--airsim-host 172.24.96.1]
"""

import argparse
import base64
import errno
import json
import math
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

try:
    import airsim
except ImportError:
    airsim = None
    print("[WARN] airsim not installed, /capture endpoint disabled")


class UAVState:
    """Shared state updated by ROS callbacks."""

    def __init__(self):
        self.lock = threading.Lock()
        self.position = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.velocity = [0.0, 0.0, 0.0]
        self.has_odom = False

        self.goal = None
        self.goal_time = 0
        self.arrived = False
        self.arrive_threshold = 1.0

    def update_odom(self, msg):
        with self.lock:
            self.position = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
            self.velocity = [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ]
            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.yaw = math.atan2(siny_cosp, cosy_cosp)
            self.has_odom = True

            if self.goal is not None:
                dx = self.position[0] - self.goal[0]
                dy = self.position[1] - self.goal[1]
                dz = self.position[2] - self.goal[2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                speed = math.sqrt(sum(v*v for v in self.velocity))
                self.arrived = dist < self.arrive_threshold and speed < 0.3

    def get_snapshot(self):
        with self.lock:
            return {
                "x": round(self.position[0], 3),
                "y": round(self.position[1], 3),
                "z": round(self.position[2], 3),
                "yaw": round(self.yaw, 3),
                "has_odom": self.has_odom,
            }

    def get_state(self):
        with self.lock:
            speed = math.sqrt(sum(v*v for v in self.velocity))
            if not self.has_odom:
                status = "NO_ODOM"
            elif self.goal is None:
                status = "IDLE"
            elif self.arrived:
                status = "ARRIVED"
            elif speed > 0.2:
                status = "FLYING"
            else:
                status = "HOVERING"
            return {
                "status": status,
                "has_goal": self.goal is not None,
                "arrived": self.arrived,
                "speed": round(speed, 3),
                "goal": self.goal,
            }


uav = UAVState()
goal_pub = None
airsim_client = None


class BridgeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the UAV bridge."""

    def log_message(self, format, *args):
        rospy.logdebug("[HTTP] " + format % args)

    def _send_json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/position":
            self._send_json(uav.get_snapshot())

        elif self.path == "/state":
            self._send_json(uav.get_state())

        elif self.path == "/arrived":
            with uav.lock:
                self._send_json({"arrived": uav.arrived})

        elif self.path == "/capture":
            self._handle_capture()

        elif self.path == "/health":
            self._send_json({"ok": True})

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/fly_to":
            self._handle_fly_to()
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_fly_to(self):
        body = self._read_body()
        x = body.get("x")
        y = body.get("y")
        z = body.get("z")
        if x is None or y is None:
            self._send_json({"error": "x and y are required"}, 400)
            return

        if z is None:
            with uav.lock:
                z = uav.position[2] if uav.has_odom else 3.0

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0

        goal_pub.publish(msg)

        with uav.lock:
            uav.goal = [float(x), float(y), float(z)]
            uav.goal_time = time.time()
            uav.arrived = False

        rospy.loginfo("Goal sent: (%.1f, %.1f, %.1f)", x, y, z)
        self._send_json({
            "ok": True,
            "goal": {"x": float(x), "y": float(y), "z": float(z)},
        })

    def _handle_capture(self):
        if airsim_client is None:
            self._send_json({"error": "AirSim not connected"}, 503)
            return
        try:
            responses = airsim_client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
            ], vehicle_name="Drone1")
            if not responses or responses[0].width == 0:
                self._send_json({"error": "empty image"}, 500)
                return
            img = responses[0]
            img_bytes = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
            img_rgba = img_bytes.reshape(img.height, img.width, -1)

            try:
                import cv2
                _, png = cv2.imencode(".png", img_rgba[:, :, :3])
                b64 = base64.b64encode(png.tobytes()).decode()
            except ImportError:
                from PIL import Image
                import io
                pil_img = Image.fromarray(img_rgba[:, :, :3])
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()

            self._send_json({
                "ok": True,
                "width": img.width,
                "height": img.height,
                "format": "png",
                "image_base64": b64,
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 500)


def ros_thread():
    """Run ROS spin in a background thread."""
    rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="UAV Bridge Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--airsim-host", type=str, default="172.24.96.1")
    parser.add_argument("--vehicle", type=str, default="Drone1")
    args = parser.parse_args()

    global goal_pub, airsim_client

    rospy.init_node("uav_bridge_server", anonymous=True)

    odom_topic = "/airsim_node/{}/odom_local_enu".format(args.vehicle)
    rospy.Subscriber(odom_topic, Odometry, uav.update_odom, queue_size=5)
    goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)

    if airsim is not None:
        try:
            airsim_client = airsim.MultirotorClient(ip=args.airsim_host)
            airsim_client.confirmConnection()
            rospy.loginfo("AirSim connected at %s", args.airsim_host)
        except Exception as e:
            rospy.logwarn("AirSim connection failed: %s (capture disabled)", e)
            airsim_client = None

    t = threading.Thread(target=ros_thread, daemon=True)
    t.start()

    try:
        server = HTTPServer(("0.0.0.0", args.port), BridgeHandler)
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, 98):
            rospy.logerr(
                "Port %d is already in use. Stop the other bridge (or: ss -tlnp | grep %d) "
                "or use another port: --port 8766",
                args.port,
                args.port,
            )
            sys.exit(1)
        raise
    rospy.loginfo("UAV Bridge Server running on http://0.0.0.0:%d", args.port)
    rospy.loginfo("  Odom: %s", odom_topic)
    rospy.loginfo("  Goal: /move_base_simple/goal")
    rospy.loginfo("  AirSim: %s", "connected" if airsim_client else "disabled")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
