#!/usr/bin/env python3
"""
EGO-Planner <-> AirSim Bridge Node

Converts EGO-Planner's PositionCommand output to AirSim velocity commands,
and plots the flight trajectory in AirSim's 3D view.

Data flow:
  traj_server (/planning/pos_cmd) --> this node --> AirSim Python API

Usage:
  rosrun ego_airsim_ros ego_to_airsim_bridge.py [_param:=value]

Parameters:
  ~vehicle_name    (string, default: "Drone1")  - AirSim vehicle name
  ~airsim_host     (string, default: "localhost") - AirSim host IP
  ~odom_topic      (string, default: "/airsim_node/Drone1/odom_local_enu")
  ~takeoff_height  (float,  default: 3.0) - Takeoff altitude in meters
  ~Kp              (float,  default: 1.8) - Position tracking P gain
"""

import signal
import sys
import math
import threading
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

try:
    from quadrotor_msgs.msg import PositionCommand
except ImportError:
    print("Cannot import quadrotor_msgs. Source devel/setup.bash first.")
    sys.exit(1)

try:
    import airsim
except ImportError:
    print("Cannot import airsim. Install: pip install airsim")
    sys.exit(1)


def force_exit(signum, frame):
    """Hard exit on second Ctrl+C."""
    print("\nForce exit.")
    import os
    os._exit(1)


class EgoAirSimBridge:
    def __init__(self):
        rospy.init_node("ego_airsim_bridge")

        self.vehicle_name = rospy.get_param("~vehicle_name", "Drone1")
        host = rospy.get_param("~airsim_host", "localhost")
        self.takeoff_height = rospy.get_param("~takeoff_height", 3.0)
        self.Kp = rospy.get_param("~Kp", 1.8)
        self.max_vel = rospy.get_param("~max_vel", 3.0)
        self.max_xy_vel = rospy.get_param("~max_xy_vel", self.max_vel)
        self.max_z_vel = rospy.get_param("~max_z_vel", 1.5)
        self.max_yaw_rate_deg = rospy.get_param("~max_yaw_rate_deg", 60.0)
        self.cmd_timeout_s = rospy.get_param("~cmd_timeout_s", 0.5)
        self.goal_deadband_m = rospy.get_param("~goal_deadband_m", 0.25)
        self.brake_radius_m = rospy.get_param("~brake_radius_m", 1.5)
        self.Kd_vel = rospy.get_param("~Kd_vel", 0.35)
        self.control_mode = rospy.get_param("~control_mode", "velocity").strip().lower()
        self.move_to_pos_speed = rospy.get_param("~move_to_pos_speed", 2.5)
        self.move_to_pos_interval_s = rospy.get_param("~move_to_pos_interval_s", 0.2)
        odom_topic = rospy.get_param(
            "~odom_topic",
            "/airsim_node/{}/odom_local_enu".format(self.vehicle_name),
        )

        self.current_pos = None
        self.current_vel = np.zeros(3)
        self.receiving_cmd = False
        self.alive = True
        self.last_cmd_time = rospy.Time(0)
        self._hover_sent_after_timeout = False
        self._last_move_to_pos_cmd_t = rospy.Time(0)

        # --- Trajectory plotting ---
        self.traj_points_ned = []
        self.last_plot_pos_ned = None
        self.plot_min_dist = 0.3
        self.plot_lock = threading.Lock()

        # --- Connect to AirSim ---
        rospy.loginfo("Connecting to AirSim at %s ...", host)
        self.client = airsim.MultirotorClient(ip=host)
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        rospy.loginfo("Taking off to %.1f m ...", self.takeoff_height)
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        self.client.moveToZAsync(
            -self.takeoff_height, 1.0, vehicle_name=self.vehicle_name
        ).join()
        rospy.loginfo("Takeoff complete. Hovering.")

        # --- ROS subscribers ---
        rospy.Subscriber(odom_topic, Odometry, self._odom_cb, queue_size=5)
        rospy.Subscriber(
            "/planning/pos_cmd", PositionCommand, self._cmd_cb, queue_size=5
        )
        rospy.Subscriber(
            "/traj_start_trigger", Empty, self._trigger_cb, queue_size=1
        )

        # Timer: draw trajectory in AirSim every 1s
        rospy.Timer(rospy.Duration(1.0), self._plot_traj_cb)
        # Watchdog: if planner command stream stalls, stop pushing stale velocity.
        rospy.Timer(rospy.Duration(0.1), self._cmd_watchdog_cb)

        rospy.loginfo(
            "Bridge ready.\n"
            "  Odom topic : %s\n"
            "  Cmd topic  : /planning/pos_cmd\n"
            "  Vehicle    : %s\n"
            "  mode=%s  Kp=%.2f  Kd=%.2f  max_vel=%.1f  max_xy=%.1f  max_z=%.1f\n"
            "  AirSim trajectory plot: enabled (red line)\n"
            "  Press Ctrl+C to stop (twice to force quit).",
            odom_topic, self.vehicle_name, self.control_mode, self.Kp, self.Kd_vel,
            self.max_vel, self.max_xy_vel, self.max_z_vel,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _odom_cb(self, msg):
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])
        self.current_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ])

        # Record for AirSim trajectory plotting (ENU -> NED)
        p_ned = airsim.Vector3r(
            msg.pose.pose.position.y,
            msg.pose.pose.position.x,
            -msg.pose.pose.position.z,
        )
        with self.plot_lock:
            if self.last_plot_pos_ned is None:
                self.traj_points_ned.append(p_ned)
                self.last_plot_pos_ned = p_ned
            else:
                dx = p_ned.x_val - self.last_plot_pos_ned.x_val
                dy = p_ned.y_val - self.last_plot_pos_ned.y_val
                dz = p_ned.z_val - self.last_plot_pos_ned.z_val
                if math.sqrt(dx*dx + dy*dy + dz*dz) >= self.plot_min_dist:
                    self.traj_points_ned.append(p_ned)
                    self.last_plot_pos_ned = p_ned

    def _trigger_cb(self, msg):
        rospy.loginfo("Trajectory triggered.")

    def _cmd_cb(self, msg):
        if self.current_pos is None or not self.alive:
            return
        if msg.trajectory_flag != PositionCommand.TRAJECTORY_STATUS_READY:
            return

        self.last_cmd_time = rospy.Time.now()
        self._hover_sent_after_timeout = False
        if not self.receiving_cmd:
            self.receiving_cmd = True
            rospy.loginfo("Receiving trajectory commands from EGO-Planner.")

        target = np.array([msg.position.x, msg.position.y, msg.position.z])
        pos_err = target - self.current_pos

        if self.control_mode == "position":
            dist = np.linalg.norm(pos_err)
            if dist < self.goal_deadband_m:
                try:
                    self.client.hoverAsync(vehicle_name=self.vehicle_name)
                except Exception:
                    pass
                return

            now = rospy.Time.now()
            if self._last_move_to_pos_cmd_t.to_sec() > 0.0:
                if (now - self._last_move_to_pos_cmd_t).to_sec() < self.move_to_pos_interval_s:
                    return
            self._last_move_to_pos_cmd_t = now

            # ENU -> NED target
            x_ned = target[1]
            y_ned = target[0]
            z_ned = -target[2]
            try:
                self.client.moveToPositionAsync(
                    x_ned,
                    y_ned,
                    z_ned,
                    self.move_to_pos_speed,
                    yaw_mode=airsim.YawMode(
                        is_rate=True,
                        yaw_or_rate=np.clip(
                            -msg.yaw_dot * 180.0 / math.pi,
                            -self.max_yaw_rate_deg,
                            self.max_yaw_rate_deg,
                        ),
                    ),
                    vehicle_name=self.vehicle_name,
                )
            except Exception:
                pass
            return

        # velocity mode: PD control in ENU frame
        vel_ff = np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z])
        # Position + velocity damping controller.
        vel_enu = vel_ff + self.Kp * pos_err - self.Kd_vel * self.current_vel

        dist = np.linalg.norm(pos_err)
        # Always hard-stop in deadband; do not rely on vel_ff becoming small.
        if dist < self.goal_deadband_m:
            vel_enu = np.zeros(3)
        # Smoothly reduce feed-forward near target to avoid carry-over drift
        # when switching to the next waypoint.
        elif dist < self.brake_radius_m and self.brake_radius_m > self.goal_deadband_m:
            scale = (dist - self.goal_deadband_m) / (self.brake_radius_m - self.goal_deadband_m)
            scale = float(np.clip(scale, 0.0, 1.0))
            vel_enu = scale * vel_ff + self.Kp * pos_err - self.Kd_vel * self.current_vel

        # Clamp horizontal and vertical components separately for safer behavior.
        xy_speed = np.linalg.norm(vel_enu[:2])
        if xy_speed > self.max_xy_vel > 0.0:
            vel_enu[:2] = vel_enu[:2] / xy_speed * self.max_xy_vel
        vel_enu[2] = np.clip(vel_enu[2], -self.max_z_vel, self.max_z_vel)
        speed = np.linalg.norm(vel_enu)
        if speed > self.max_vel > 0.0:
            vel_enu = vel_enu / speed * self.max_vel

        # ENU -> NED
        vx_ned = vel_enu[1]
        vy_ned = vel_enu[0]
        vz_ned = -vel_enu[2]

        yaw_rate_deg = np.clip(
            -msg.yaw_dot * 180.0 / math.pi,
            -self.max_yaw_rate_deg,
            self.max_yaw_rate_deg,
        )

        try:
            self.client.moveByVelocityAsync(
                vx_ned, vy_ned, vz_ned,
                duration=0.1,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_deg),
                vehicle_name=self.vehicle_name,
            )
        except Exception:
            pass

    def _cmd_watchdog_cb(self, _event):
        if not self.alive or not self.receiving_cmd:
            return
        if self.last_cmd_time.to_sec() <= 0.0:
            return
        dt = (rospy.Time.now() - self.last_cmd_time).to_sec()
        if dt <= self.cmd_timeout_s:
            return
        if self._hover_sent_after_timeout:
            return
        self._hover_sent_after_timeout = True
        rospy.logwarn(
            "No /planning/pos_cmd for %.2fs (>%.2fs); hover for safety.",
            dt,
            self.cmd_timeout_s,
        )
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # AirSim trajectory plotting
    # ------------------------------------------------------------------

    def _plot_traj_cb(self, event):
        with self.plot_lock:
            pts = list(self.traj_points_ned)
        if len(pts) < 2:
            return
        try:
            self.client.simPlotLineStrip(
                pts,
                color_rgba=[1.0, 0.0, 0.0, 1.0],
                thickness=5.0,
                duration=60.0,
                is_persistent=False,
            )
        except Exception as e:
            rospy.logwarn_throttle(10.0, "simPlotLineStrip: %s" % e)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        self.alive = False
        rospy.loginfo("Shutting down bridge...")
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name)
            rospy.sleep(0.5)
        except Exception:
            pass
        rospy.loginfo("Bridge stopped.")


def main():
    # First Ctrl+C: graceful shutdown. Second Ctrl+C: force exit.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    bridge = EgoAirSimBridge()
    rospy.on_shutdown(bridge.shutdown)

    # Override SIGINT after init so first Ctrl+C triggers rospy shutdown,
    # second Ctrl+C calls force_exit
    signal.signal(signal.SIGINT, force_exit)

    try:
        rospy.spin()
    except Exception:
        pass


if __name__ == "__main__":
    main()
