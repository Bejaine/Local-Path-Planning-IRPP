#!/usr/bin/env python3
"""
Dynamic Window Approach (DWA) Local Motion Planner for TurtleBot3
=================================================================
Algorithm Reference:
  Fox, D., Burgard, W., & Thrun, S. (1997).
  "The dynamic window approach to collision avoidance."
  IEEE Robotics & Automation Magazine, 4(1), 23-33.

How it works:
  1. Sample (v, ω) pairs inside the dynamic window — the reachable
     velocities given the robot's current speed and its acceleration limits.
  2. Simulate each (v, ω) pair forward for a short horizon.
  3. Score each trajectory using a cost function:
       cost = α·heading + β·clearance + γ·velocity  (lower = better)
  4. Command the best (lowest-cost) trajectory.

Subscriptions:
  /scan          – LaserScan  (obstacle distances)
  /odom          – Odometry   (current pose & velocity)
  /goal_velocity – Twist      (desired velocity direction from operator)

Publications:
  /cmd_vel                    – Twist  (commanded velocity)
  /dwa_planner/trajectories   – MarkerArray (visualised candidates)
  /dwa_planner/best_trajectory– Marker      (selected trajectory)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs  # noqa: F401 – needed for transform registration


# ---------------------------------------------------------------------------
# Helper dataclass
# ---------------------------------------------------------------------------
class RobotState:
    """Holds the robot's current kinematic state."""
    __slots__ = ('x', 'y', 'yaw', 'v', 'omega')

    def __init__(self):
        self.x = self.y = self.yaw = self.v = self.omega = 0.0


# ---------------------------------------------------------------------------
# DWA Planner Node
# ---------------------------------------------------------------------------
class DWAPlannerNode(Node):

    def __init__(self):
        super().__init__('dwa_planner')

        # ------------------------------------------------------------------ #
        # Declare & read parameters (all tunable via config/dwa_params.yaml) #
        # ------------------------------------------------------------------ #
        self._declare_parameters()

        p = self._p  # convenience alias

        # ------------------------------------------------------------------ #
        # Internal state                                                       #
        # ------------------------------------------------------------------ #
        self.state = RobotState()
        self.scan_ranges: list[float] = []
        self.scan_angle_min = 0.0
        self.scan_angle_increment = 0.0
        self.goal_velocity = Twist()          # desired vx / omega from operator
        self.odom_received = False
        self.scan_received = False

        # ------------------------------------------------------------------ #
        # TF2                                                                  #
        # ------------------------------------------------------------------ #
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------------ #
        # Publishers                                                           #
        # ------------------------------------------------------------------ #
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.traj_pub = self.create_publisher(
            MarkerArray, '/dwa_planner/trajectories', 10)
        self.best_traj_pub = self.create_publisher(
            Marker, '/dwa_planner/best_trajectory', 10)

        # ------------------------------------------------------------------ #
        # Subscribers                                                          #
        # ------------------------------------------------------------------ #
        self.create_subscription(LaserScan, '/scan',
                                 self._scan_callback, 10)
        self.create_subscription(Odometry, '/odom',
                                 self._odom_callback, 10)
        self.create_subscription(Twist, '/goal_velocity',
                                 self._goal_callback, 10)

        # ------------------------------------------------------------------ #
        # Control loop timer                                                   #
        # ------------------------------------------------------------------ #
        dt = p('control_period')
        self.create_timer(dt, self._control_loop)
        self.get_logger().info(
            f'DWA Planner ready  (dt={dt:.3f}s, '
            f'v_max={p("max_speed"):.2f} m/s, '
            f'omega_max={p("max_yaw_rate"):.2f} rad/s)')

    # ======================================================================= #
    # Parameter helpers                                                        #
    # ======================================================================= #
    def _declare_parameters(self):
        defaults = {
            # Robot kinematics
            'max_speed':           0.22,   # [m/s]
            'min_speed':          -0.05,   # [m/s]  allow slight reversing
            'max_yaw_rate':        2.84,   # [rad/s]
            'max_accel':           0.5,    # [m/s²]
            'max_dyaw_rate':       3.2,    # [rad/s²]
            # Sampling resolution
            'v_resolution':        0.02,   # [m/s]
            'yaw_rate_resolution': 0.1,    # [rad/s]
            # Simulation horizon
            'predict_time':        3.0,    # [s]
            'control_period':      0.1,    # [s]  10 Hz
            # Cost weights  (all non-negative; tune to taste)
            'heading_cost_gain':   0.15,
            'dist_cost_gain':      1.0,
            'velocity_cost_gain':  1.0,
            # Safety
            'robot_radius':        0.22,   # [m]  burger radius ≈ 0.105 m; add margin
            'obstacle_cost_radius':0.5,    # [m]  start penalising within this range
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _p(self, name: str):
        return self.get_parameter(name).value

    # ======================================================================= #
    # Callbacks                                                                #
    # ======================================================================= #
    def _scan_callback(self, msg: LaserScan):
        self.scan_ranges = list(msg.ranges)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_increment = msg.angle_increment
        self.scan_received = True

    def _odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.state.x = pos.x
        self.state.y = pos.y
        # quaternion → yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.state.yaw = math.atan2(siny_cosp, cosy_cosp)
        self.state.v = msg.twist.twist.linear.x
        self.state.omega = msg.twist.twist.angular.z
        self.odom_received = True

    def _goal_callback(self, msg: Twist):
        self.goal_velocity = msg

    # ======================================================================= #
    # Main control loop                                                        #
    # ======================================================================= #
    def _control_loop(self):
        if not (self.odom_received and self.scan_received):
            return

        p = self._p
        dt = p('control_period')

        # Convert LaserScan → list of (x, y) obstacle points in robot frame
        obstacles = self._scan_to_obstacles()

        # Compute the dynamic window
        dw = self._dynamic_window(dt)

        # Evaluate all (v, omega) pairs and pick the best
        best_v, best_omega, all_trajs, best_traj = \
            self._evaluate_window(dw, obstacles, dt)

        # Publish command
        cmd = Twist()
        cmd.linear.x = best_v
        cmd.angular.z = best_omega
        self.cmd_pub.publish(cmd)

        # Publish visualisation
        self._publish_viz(all_trajs, best_traj)

    # ======================================================================= #
    # DWA Core                                                                 #
    # ======================================================================= #
    def _dynamic_window(self, dt: float) -> tuple:
        """
        Compute the Dynamic Window as the intersection of:
          Vs  – velocity space allowed by the robot's hard limits
          Vd  – velocities reachable within one control step
        Returns (v_min, v_max, omega_min, omega_max).
        """
        p = self._p
        # Reachable in one time step
        v_min = max(p('min_speed'),
                    self.state.v - p('max_accel') * dt)
        v_max = min(p('max_speed'),
                    self.state.v + p('max_accel') * dt)
        om_min = max(-p('max_yaw_rate'),
                     self.state.omega - p('max_dyaw_rate') * dt)
        om_max = min(p('max_yaw_rate'),
                     self.state.omega + p('max_dyaw_rate') * dt)
        return v_min, v_max, om_min, om_max

    def _evaluate_window(self, dw, obstacles, dt):
        """
        Sample (v, ω) pairs from the dynamic window, simulate each trajectory,
        score it, and return the best pair together with visualisation data.
        """
        p = self._p
        v_min, v_max, om_min, om_max = dw
        T = p('predict_time')

        best_score = float('inf')
        best_v, best_omega = 0.0, 0.0
        all_trajs: list = []
        best_traj: list = []

        v_res = p('v_resolution')
        om_res = p('yaw_rate_resolution')

        v_samples = np.arange(v_min, v_max + v_res * 0.5, v_res)
        om_samples = np.arange(om_min, om_max + om_res * 0.5, om_res)

        for v in v_samples:
            for omega in om_samples:
                traj = self._simulate_trajectory(v, omega, T, dt)
                score, valid = self._score_trajectory(traj, v, omega, obstacles)

                if valid:
                    all_trajs.append(traj)
                    if score < best_score:
                        best_score = score
                        best_v, best_omega = v, omega
                        best_traj = traj

        # If nothing is valid, rotate in place to find clearance
        if not best_traj:
            self.get_logger().warn('No valid trajectory found – rotating to clear.', throttle_duration_sec=2.0)
            best_v = 0.0
            best_omega = p('max_yaw_rate') * 0.5

        return best_v, best_omega, all_trajs, best_traj

    def _simulate_trajectory(self, v: float, omega: float,
                              T: float, dt: float) -> list:
        """
        Forward-simulate the unicycle model for time T starting from
        the robot's current pose.  Returns list of (x, y, yaw) poses.
        """
        x, y, yaw = self.state.x, self.state.y, self.state.yaw
        traj = [(x, y, yaw)]
        t = 0.0
        while t < T:
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += omega * dt
            traj.append((x, y, yaw))
            t += dt
        return traj

    def _score_trajectory(self, traj, v, omega, obstacles):
        """
        Score a trajectory.  Lower is better.

        Components
        ----------
        heading   – alignment of the robot's final heading with the
                    desired velocity direction (from /goal_velocity).
        clearance – inverse of distance to nearest obstacle along traj.
        velocity  – reward for moving fast (penalises slow/zero speed).

        Returns (score, is_valid).  is_valid=False when the trajectory
        collides with an obstacle within the robot's safety radius.
        """
        p = self._p
        r_robot = p('robot_radius')
        r_penalty = p('obstacle_cost_radius')

        # ── Collision check ────────────────────────────────────────────────
        min_dist = float('inf')
        for (px, py, _) in traj:
            for (ox, oy) in obstacles:
                d = math.hypot(px - ox, py - oy)
                if d < r_robot:
                    return float('inf'), False   # collision → invalid
                if d < min_dist:
                    min_dist = d

        # ── Clearance cost ─────────────────────────────────────────────────
        if min_dist > r_penalty:
            clearance_cost = 0.0
        else:
            # smooth penalty that grows as robot approaches obstacle
            clearance_cost = 1.0 - (min_dist / r_penalty)

        # ── Heading cost ───────────────────────────────────────────────────
        # desired direction is encoded in goal_velocity.linear.x (forward)
        # and goal_velocity.angular.z (turn)
        gv = self.goal_velocity
        desired_yaw = math.atan2(gv.angular.z, gv.linear.x) \
                      if (abs(gv.linear.x) > 1e-3 or abs(gv.angular.z) > 1e-3) \
                      else self.state.yaw           # keep current heading if no goal
        final_yaw = traj[-1][2]
        heading_diff = abs(math.atan2(
            math.sin(desired_yaw - final_yaw),
            math.cos(desired_yaw - final_yaw)))
        heading_cost = heading_diff / math.pi   # normalised [0, 1]

        # ── Velocity cost ──────────────────────────────────────────────────
        # Reward higher forward speeds (penalise being slow)
        velocity_cost = 1.0 - (v / p('max_speed')) \
            if p('max_speed') > 0.0 else 0.0
        velocity_cost = max(0.0, velocity_cost)

        total = (p('heading_cost_gain')  * heading_cost
                 + p('dist_cost_gain')   * clearance_cost
                 + p('velocity_cost_gain') * velocity_cost)

        return total, True

    # ======================================================================= #
    # Obstacle extraction from LaserScan                                       #
    # ======================================================================= #
    def _scan_to_obstacles(self) -> list:
        """
        Convert raw LaserScan into obstacle (x, y) points in the world frame.
        Invalid (inf / nan / 0) readings are discarded.
        """
        obstacles = []
        if not self.scan_ranges:
            return obstacles

        for i, r in enumerate(self.scan_ranges):
            if not math.isfinite(r) or r <= 0.02:
                continue
            angle = self.scan_angle_min + i * self.scan_angle_increment
            # Point in robot frame
            lx = r * math.cos(angle)
            ly = r * math.sin(angle)
            # Rotate to world frame using current robot pose
            cos_y = math.cos(self.state.yaw)
            sin_y = math.sin(self.state.yaw)
            wx = self.state.x + cos_y * lx - sin_y * ly
            wy = self.state.y + sin_y * lx + cos_y * ly
            obstacles.append((wx, wy))
        return obstacles

    # ======================================================================= #
    # Visualisation                                                            #
    # ======================================================================= #
    def _publish_viz(self, all_trajs, best_traj):
        now = self.get_clock().now().to_msg()

        # ── All candidate trajectories (light blue) ───────────────────────
        marker_array = MarkerArray()
        for idx, traj in enumerate(all_trajs):
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp = now
            m.ns = 'candidates'
            m.id = idx
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.01
            m.color.r = 0.3
            m.color.g = 0.6
            m.color.b = 1.0
            m.color.a = 0.4
            m.lifetime.sec = 0
            m.lifetime.nanosec = int(0.15e9)
            from geometry_msgs.msg import Point
            for (x, y, _) in traj:
                pt = Point()
                pt.x, pt.y, pt.z = x, y, 0.05
                m.points.append(pt)
            marker_array.markers.append(m)
        self.traj_pub.publish(marker_array)

        # ── Best trajectory (green) ───────────────────────────────────────
        if best_traj:
            bm = Marker()
            bm.header.frame_id = 'odom'
            bm.header.stamp = now
            bm.ns = 'best'
            bm.id = 0
            bm.type = Marker.LINE_STRIP
            bm.action = Marker.ADD
            bm.scale.x = 0.04
            bm.color.r = 0.0
            bm.color.g = 1.0
            bm.color.b = 0.2
            bm.color.a = 0.9
            bm.lifetime.sec = 0
            bm.lifetime.nanosec = int(0.15e9)
            from geometry_msgs.msg import Point
            for (x, y, _) in best_traj:
                pt = Point()
                pt.x, pt.y, pt.z = x, y, 0.05
                bm.points.append(pt)
            self.best_traj_pub.publish(bm)


def main(args=None):
    rclpy.init(args=args)
    node = DWAPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())   # stop robot on shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
