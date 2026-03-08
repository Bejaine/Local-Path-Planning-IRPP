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
  /goal_pose     – PoseStamped (desired destination pose)

Publications:
  /cmd_vel                    – Twist  (commanded velocity)
  /dwa_planner/trajectories   – MarkerArray (visualised candidates)
  /dwa_planner/best_trajectory– Marker      (selected trajectory)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped, Vector3, PoseStamped
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
# MPPI Planner Node
# ---------------------------------------------------------------------------
class MPPIPlannerNode(Node):

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
        self.goal_pose = None
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
        # Use TwistStamped for Jazzy / latest ros_gz_bridge compatibility
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.traj_pub = self.create_publisher(
            MarkerArray, '/mppi_planner/trajectories', 10)
        self.best_traj_pub = self.create_publisher(
            Marker, '/mppi_planner/best_trajectory', 10)

        # ------------------------------------------------------------------ #
        # Subscribers                                                          #
        # ------------------------------------------------------------------ #
        self.create_subscription(LaserScan, '/scan',
                                 self._scan_callback, 10)
        self.create_subscription(Odometry, '/odom',
                                 self._odom_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose',
                                 self._goal_callback, 10)

        # ------------------------------------------------------------------ #
        # Control loop timer                                                   #
        # ------------------------------------------------------------------ #
        dt = p('control_period')
        self.create_timer(dt, self._control_loop)
        self.get_logger().info(
            f'MPPI Planner ready  (dt={dt:.3f}s, '
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
            'heading_cost_gain':   1.0,
            'dist_cost_gain':      2.0,
            'velocity_cost_gain':  1.0,
            # Safety
            'robot_radius':        0.22,   # [m]  burger radius ≈ 0.105 m; add margin
            'obstacle_cost_radius':1.5,    # [m]  start penalising within this range
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

    def _goal_callback(self, msg: PoseStamped):
        self.goal_pose = msg

    # ======================================================================= #
    # Main control loop                                                        #
    # ======================================================================= #
    def _control_loop(self):
        if not (self.odom_received and self.scan_received):
            return

        # If zero goal, stay perfectly still and skip evaluation
        if self.goal_pose is None:
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_footprint'
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            # Reset internal control sequence history to avoid drifting on restart
            if hasattr(self, 'u_V'):
                self.u_V.fill(0.0)
                self.u_W.fill(0.0)
            return

        p = self._p
        dt = p('control_period')

        # Convert LaserScan → list of (x, y) obstacle points in robot frame
        obstacles = self._scan_to_obstacles()

        # Run the Model Predictive Path Integral logic
        best_v, best_omega, all_trajs, best_traj = \
            self._evaluate_mppi(obstacles, dt)

        # Publish command
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_footprint'
        cmd.twist.linear.x = best_v
        cmd.twist.angular.z = best_omega
        self.cmd_pub.publish(cmd)

        # Publish visualisation
        self._publish_viz(all_trajs, best_traj)

    # ======================================================================= #
    # Obstacle extraction from LaserScan                                       #
    # ======================================================================= #
    def _scan_to_obstacles(self) -> list:
        obstacles = []
        if not self.scan_ranges:
            return obstacles

        for i, r in enumerate(self.scan_ranges):
            if not math.isfinite(r) or r <= 0.02:
                continue
            angle = self.scan_angle_min + i * self.scan_angle_increment
            lx = r * math.cos(angle)
            ly = r * math.sin(angle)
            cos_y = math.cos(self.state.yaw)
            sin_y = math.sin(self.state.yaw)
            wx = self.state.x + cos_y * lx - sin_y * ly
            wy = self.state.y + sin_y * lx + cos_y * ly
            obstacles.append((wx, wy))
        return obstacles

    # ======================================================================= #
    # ======================================================================= #
    # MPPI Core                                                               #
    # ======================================================================= #
    def _evaluate_mppi(self, obstacles, dt):
        p = self._p
        K = 100   # Number of samples
        T = int(p('predict_time') / dt)
        
        # Initialize control sequences if not present
        if not hasattr(self, 'u_V'):
            self.u_V = np.zeros(T)
            self.u_W = np.zeros(T)
            
        # Sample noise
        std_v = 0.5  # Increased so it can sample higher speeds easier
        std_w = 1.0
        delta_V = np.random.normal(0.0, std_v, (K, T))
        delta_W = np.random.normal(0.0, std_w, (K, T))
        
        # Clip control inputs
        V_samples = np.clip(self.u_V + delta_V, p('min_speed'), p('max_speed'))
        W_samples = np.clip(self.u_W + delta_W, -p('max_yaw_rate'), p('max_yaw_rate'))
        
        # Setup costs and rollouts
        costs = np.zeros(K)
        all_trajs = []
        r_robot = p('robot_radius')
        r_penalty = p('obstacle_cost_radius')
        
        # Goal parameters
        if self.goal_pose is not None:
            desired_yaw = math.atan2(self.goal_pose.pose.position.y - self.state.y,
                                     self.goal_pose.pose.position.x - self.state.x)
        else:
            desired_yaw = self.state.yaw
        # Try to go at max speed in the requested direction
        target_v = p('max_speed')
        
        for k in range(K):
            traj = []
            x, y, yaw = self.state.x, self.state.y, self.state.yaw
            cost_k = 0.0
            min_dist = float('inf')
            
            for t in range(T):
                v = V_samples[k, t]
                w = W_samples[k, t]
                
                x += v * math.cos(yaw) * dt
                y += v * math.sin(yaw) * dt
                yaw += w * dt
                traj.append((x, y, yaw))
                
                # We can check collisions sparsely or just use final/min points
            
            # Collision check
            for (px, py, pyaw) in traj:
                for (ox, oy) in obstacles:
                    d = math.hypot(px - ox, py - oy)
                    if d < r_robot:
                        cost_k += 10000.0  # high collision penalty
                    if d < min_dist:
                        min_dist = d
            
            # Clearance cost
            if min_dist <= r_penalty and min_dist > r_robot:
                cost_k += p('dist_cost_gain') * (1.0 - (min_dist / r_penalty))
                
            # Heading cost
            final_yaw = traj[-1][2]
            heading_diff = abs(math.atan2(math.sin(desired_yaw - final_yaw), math.cos(desired_yaw - final_yaw)))
            cost_k += p('heading_cost_gain') * (heading_diff / math.pi)
            
            # Velocity cost (average speed diff)
            avg_v = np.mean(V_samples[k, :])
            if p('max_speed') > 0:
                cost_k += p('velocity_cost_gain') * (abs(target_v - avg_v) / p('max_speed'))
                
            # Stop condition strongly overrides
            if self.goal_pose is None:
                cost_k += abs(avg_v) / 0.1
                
            costs[k] = cost_k
            all_trajs.append(traj)
            
        # Compute weights
        lam = 0.5
        min_cost = np.min(costs)
        weights = np.exp(-(costs - min_cost) / lam)
        weights /= np.sum(weights)
        
        # Update control sequence
        self.u_V += np.sum(weights[:, None] * delta_V, axis=0)
        self.u_W += np.sum(weights[:, None] * delta_W, axis=0)
        
        # Clip again to hardware config margins
        self.u_V = np.clip(self.u_V, p('min_speed'), p('max_speed'))
        self.u_W = np.clip(self.u_W, -p('max_yaw_rate'), p('max_yaw_rate'))
        
        # Best command is the first element
        cmd_v = self.u_V[0]
        cmd_w = self.u_W[0]
        
        # Find best trajectory for visualization
        best_idx = np.argmin(costs)
        best_traj = all_trajs[best_idx]
        
        # Shift sequence forward for next iteration
        self.u_V = np.append(self.u_V[1:], 0.0)
        self.u_W = np.append(self.u_W[1:], 0.0)
        
        return cmd_v, cmd_w, all_trajs, best_traj

    # ======================================================================= #
    # Visualisation                                                            #
    # ======================================================================= #
    def _publish_viz(self, all_trajs, best_traj):
        now = self.get_clock().now().to_msg()
        # ── All trajectories ───────────────────────
        marker_array = MarkerArray()
        for idx, traj in enumerate(all_trajs):
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp = now
            m.ns = 'mppi_trajectories'
            m.id = idx
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.005
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 0.3
            for (px, py, _) in traj:
                from geometry_msgs.msg import Point
                pt = Point()
                pt.x, pt.y, pt.z = px, py, 0.01
                m.points.append(pt)
            marker_array.markers.append(m)
        self.traj_pub.publish(marker_array)

        # ── Best trajectory ──────────────────────────
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = now
        m.ns = 'mppi_best'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.02
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        if best_traj:
            for (px, py, _) in best_traj:
                from geometry_msgs.msg import Point
                pt = Point()
                pt.x, pt.y, pt.z = px, py, 0.02
                m.points.append(pt)
        self.best_traj_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = MPPIPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
