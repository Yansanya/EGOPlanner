#!/usr/bin/env python3
"""
EGO-Planner <-> AirSim Bridge Node

Converts EGO-Planner's PositionCommand output to AirSim velocity commands.

Data flow:
  traj_server (/planning/pos_cmd) --> this node --> AirSim Python API

This node:
  1. Connects to AirSim, arms and takes off the drone
  2. Subscribes to EGO-Planner's PositionCommand (position + velocity + acceleration)
  3. Applies PD control to convert to velocity commands
  4. Sends velocity commands to AirSim via Python API

Coordinate convention:
  EGO-Planner works in ENU (East-North-Up) when airsim_ros_pkgs uses world_enu.
  AirSim Python API always uses NED (North-East-Down).
  This node handles the ENU <-> NED conversion automatically.

Usage:
  rosrun ego_planner ego_to_airsim_bridge.py [_param:=value]

Parameters:
  ~vehicle_name    (string, default: "Drone1")  - AirSim vehicle name
  ~airsim_host     (string, default: "localhost") - AirSim host IP
  ~odom_topic      (string, default: "/airsim_node/Drone1/odom_local_enu")
  ~takeoff_height  (float,  default: 1.5) - Takeoff altitude in meters
  ~Kp              (float,  default: 1.5) - Position tracking P gain
"""

import math
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

try:
    from quadrotor_msgs.msg import PositionCommand
except ImportError:
    rospy.logfatal("Cannot import quadrotor_msgs. Make sure the workspace is built "
                   "and sourced: source devel/setup.bash")
    raise

try:
    import airsim
except ImportError:
    rospy.logfatal("Cannot import airsim. Install it with: pip install airsim")
    raise


class EgoAirSimBridge:
    def __init__(self):
        rospy.init_node("ego_airsim_bridge")

        self.vehicle_name = rospy.get_param("~vehicle_name", "Drone1")
        host = rospy.get_param("~airsim_host", "localhost")
        self.takeoff_height = rospy.get_param("~takeoff_height", 1.5)
        self.Kp = rospy.get_param("~Kp", 1.5)
        odom_topic = rospy.get_param(
            "~odom_topic",
            "/airsim_node/{}/odom_local_enu".format(self.vehicle_name),
        )

        self.current_pos = None
        self.current_yaw = None
        self.receiving_cmd = False

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

        rospy.loginfo(
            "Bridge ready.\n"
            "  Odom topic : %s\n"
            "  Cmd topic  : /planning/pos_cmd\n"
            "  Vehicle    : %s\n"
            "  Kp         : %.2f\n"
            "Now open RViz and send a 2D Nav Goal to start planning.",
            odom_topic,
            self.vehicle_name,
            self.Kp,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _odom_cb(self, msg):
        """Cache current position from airsim_ros_pkgs (ENU frame)."""
        self.current_pos = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
        )

    def _trigger_cb(self, msg):
        """Trajectory start trigger from waypoint_generator."""
        rospy.loginfo("Trajectory triggered.")

    def _cmd_cb(self, msg):
        """
        Receive PositionCommand from traj_server and send velocity to AirSim.

        PositionCommand contains desired position, velocity, acceleration,
        yaw and yaw_dot at 100 Hz. We use a simple P controller on the
        position error plus feedforward velocity to produce a velocity
        command for AirSim.
        """
        if self.current_pos is None:
            return

        if msg.trajectory_flag != PositionCommand.TRAJECTORY_STATUS_READY:
            return

        if not self.receiving_cmd:
            self.receiving_cmd = True
            rospy.loginfo("Receiving trajectory commands from EGO-Planner.")

        # --- PD control in ENU frame ---
        target = np.array([msg.position.x, msg.position.y, msg.position.z])
        vel_ff = np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z])
        pos_err = target - self.current_pos

        vel_enu = vel_ff + self.Kp * pos_err

        # --- ENU -> NED conversion for AirSim Python API ---
        #   ENU: X=East,  Y=North, Z=Up
        #   NED: X=North, Y=East,  Z=Down
        vx_ned = vel_enu[1]    # North = ENU.Y
        vy_ned = vel_enu[0]    # East  = ENU.X
        vz_ned = -vel_enu[2]   # Down  = -ENU.Z

        # Yaw rate: ENU positive = CCW, AirSim positive = CW when viewed from above
        yaw_rate_deg = -msg.yaw_dot * 180.0 / math.pi

        self.client.moveByVelocityAsync(
            vx_ned,
            vy_ned,
            vz_ned,
            duration=0.1,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_deg),
            vehicle_name=self.vehicle_name,
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        rospy.loginfo("Shutting down bridge, hovering drone...")
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            pass


def main():
    bridge = EgoAirSimBridge()
    rospy.on_shutdown(bridge.shutdown)
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
