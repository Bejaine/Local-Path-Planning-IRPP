#!/usr/bin/env python3
"""
MPPI (Model Predictive Path Integral) Local Motion Planner for TurtleBot3
==========================================================================
Algorithm Reference:
  Williams, G., et al. (2016).
  "Aggressive driving with model predictive path integral control." ICRA.

  Williams, G., et al. (2017).
  "Information theoretic MPC for model-based reinforcement learning." ICRA.

How MPPI works vs DWB
---------------------
DWB samples individual (v, w) pairs and picks the single best one.
MPPI samples K full control SEQUENCES and computes a WEIGHTED AVERAGE:

  Step 1: Sample K sequences by perturbing nominal U with noise
            U_k = U + eps_k,   eps_k ~ N(0, sigma)

  Step 2: Roll out all K sequences through unicycle model (vectorised)

  Step 3: Score each sequence with cost S(tau_k)

  Step 4: Compute information-theoretic weights (softmin):
            w_k = exp(-(1/lambda) * S_k)
          Low  lambda -> winner-takes-all (aggressive)
          High lambda -> smooth averaging (conservative)

  Step 5: Update nominal sequence as weighted average:
            U* = sum(w_k * U_k) / sum(w_k)

  Step 6: Execute U*[0], shift U* forward (receding horizon), repeat.

Key advantage: MPPI optimises full multi-step sequences so it can plan
compound manoeuvres (e.g. slow + turn + accelerate) that DWB cannot,
since DWB commits to a constant (v,w) arc over the whole horizon.

Topics
------
  SUB  /goal_velocity  (Twist)      - desired direction from keyboard
  SUB  /scan           (LaserScan)  - LiDAR obstacles
  SUB  /odom           (Odometry)   - current pose & velocity
  PUB  /cmd_vel        (Twist)      - optimal velocity command
  PUB  /dwa_planner/trajectories    (MarkerArray) - sampled trajs (blue)
  PUB  /dwa_planner/best_trajectory (Marker)      - optimal traj (green)
  PUB  /dwa_planner/status          (String)      - IDLE / NAVIGATING
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class MPPIPlannerNode(Node):

    def __init__(self):
        super().__init__('mppi_planner')
        self._declare_parameters()

        # Robot state
        self.x = self.y = self.yaw = self.v = self.omega = 0.0
        self.odom_received  = False
        self.scan_received  = False
        self.scan_ranges    = []
        self.scan_angle_min = 0.0
        self.scan_angle_inc = 0.0
        self.obstacles      = []
        self.goal_velocity  = Twist()

        # Nominal control sequence  shape: (T, 2)  -> [v, omega] per step
        T       = self._p('horizon')
        self.U  = np.zeros((T, 2))

        # TF2
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers  (same topic names as DWB so RViz config works for both)
        self.cmd_pub    = self.create_publisher(Twist,       '/cmd_vel',                     10)
        self.traj_pub   = self.create_publisher(MarkerArray, '/dwa_planner/trajectories',    10)
        self.best_pub   = self.create_publisher(Marker,      '/dwa_planner/best_trajectory', 10)
        self.status_pub = self.create_publisher(String,      '/dwa_planner/status',          10)

        # Subscribers
        self.create_subscription(Twist,     '/goal_velocity', self._goal_cb, 10)
        self.create_subscription(LaserScan, '/scan',          self._scan_cb, 10)
        self.create_subscription(Odometry,  '/odom',          self._odom_cb, 10)

        # Control loop
        self.create_timer(self._p('dt'), self._control_loop)

        self.get_logger().info(
            f'MPPI Planner ready\n'
            f'  K={self._p("K")} samples | '
            f'T={self._p("horizon")} steps | '
            f'lambda={self._p("lambda_")} temperature\n'
            f'  Run keyboard: ros2 run dwa_planner goalpublisher')

    # ── Parameters ──────────────────────────────────────────────────────────
    def _declare_parameters(self):
        defaults = {
            # MPPI core
            'K':               300,    # number of sampled trajectories
            'horizon':          30,    # steps per rollout
            'dt':                0.1,  # seconds per step = control period
            'lambda_':           0.5,  # temperature (low=greedy, high=smooth)
            # Sampling noise
            'sigma_v':           0.15, # [m/s]  linear velocity noise std
            'sigma_omega':       0.5,  # [rad/s] angular velocity noise std
            # Robot limits (TurtleBot3 Burger)
            'max_v':             0.22,
            'min_v':            -0.05,
            'max_omega':         2.84,
            # Cost weights
            'w_obstacle':        8.0,  # increase = more cautious near walls
            'w_heading':         2.0,  # increase = tracks direction tighter
            'w_velocity':        0.5,  # increase = prefers higher speed
            'w_terminal':        3.0,  # heading cost at end of horizon
            # Safety
            'robot_radius':      0.22,
            'obstacle_cost_radius': 0.6,
        }
        for name, val in defaults.items():
            self.declare_parameter(name, val)

    def _p(self, name):
        return self.get_parameter(name).value

    # ── Callbacks ────────────────────────────────────────────────────────────
    def _goal_cb(self, msg):
        self.goal_velocity = msg

    def _scan_cb(self, msg):
        self.scan_ranges    = list(msg.ranges)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_inc = msg.angle_increment
        self.scan_received  = True
        self.obstacles      = self._scan_to_obstacles()

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        self.x   = pos.x
        self.y   = pos.y
        siny     = 2.0 * (q.w * q.z + q.x * q.y)
        cosy     = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)
        self.v   = msg.twist.twist.linear.x
        self.omega = msg.twist.twist.angular.z
        self.odom_received = True

    # ── Main control loop ────────────────────────────────────────────────────
    def _control_loop(self):
        if not (self.odom_received and self.scan_received):
            return

        gv     = self.goal_velocity
        moving = abs(gv.linear.x) > 0.01 or abs(gv.angular.z) > 0.01

        s      = String()
        s.data = 'NAVIGATING' if moving else 'IDLE'
        self.status_pub.publish(s)

        if not moving:
            self.cmd_pub.publish(Twist())
            self.U[:] = 0.0
            return

        desired_yaw = math.atan2(gv.angular.z, gv.linear.x)

        # Run one MPPI iteration
        viz_trajs, best_traj = self._mppi_step(desired_yaw)

        # Command first step of optimal sequence
        cmd           = Twist()
        cmd.linear.x  = float(np.clip(self.U[0, 0], self._p('min_v'),     self._p('max_v')))
        cmd.angular.z = float(np.clip(self.U[0, 1], -self._p('max_omega'), self._p('max_omega')))
        self.cmd_pub.publish(cmd)

        # Receding horizon shift
        self.U[:-1] = self.U[1:]
        self.U[-1]  = self.U[-2]

        self._publish_viz(viz_trajs, best_traj)

    # ── MPPI Core ────────────────────────────────────────────────────────────
    def _mppi_step(self, desired_yaw: float):
        K      = self._p('K')
        T      = self._p('horizon')
        dt     = self._p('dt')
        lam    = self._p('lambda_')

        # Step 1: Sample noise eps ~ N(0, sigma)  shape: (K, T, 2)
        eps = np.zeros((K, T, 2))
        eps[:, :, 0] = np.random.normal(0, self._p('sigma_v'),     (K, T))
        eps[:, :, 1] = np.random.normal(0, self._p('sigma_omega'), (K, T))

        # Step 2: Perturbed sequences  shape: (K, T, 2)
        U_k = self.U[np.newaxis] + eps
        U_k[:, :, 0] = np.clip(U_k[:, :, 0], self._p('min_v'),      self._p('max_v'))
        U_k[:, :, 1] = np.clip(U_k[:, :, 1], -self._p('max_omega'), self._p('max_omega'))

        # Steps 3+4: Vectorised rollout and scoring
        costs, trajs = self._rollout_and_score(U_k, desired_yaw, dt)

        # Step 5: Softmin weights  w_k = exp(-(S_k - min_S) / lambda)
        beta    = np.min(costs)
        weights = np.exp(-(costs - beta) / lam)
        weights /= np.sum(weights) + 1e-8

        # Step 6: Weighted update of nominal sequence
        self.U = np.einsum('k,ktj->tj', weights, U_k)

        # Visualisation: top 20 lowest-cost trajectories
        top_idx   = np.argsort(costs)[:20]
        viz_trajs = [trajs[i] for i in top_idx]
        best_traj = self._rollout_single(self.U, dt)

        return viz_trajs, best_traj

    # ── Vectorised rollout (all K trajectories at once) ──────────────────────
    def _rollout_and_score(self, U_k, desired_yaw, dt):
        K, T, _ = U_k.shape
        w_obs   = self._p('w_obstacle')
        w_head  = self._p('w_heading')
        w_vel   = self._p('w_velocity')
        r_rob   = self._p('robot_radius')
        r_pen   = self._p('obstacle_cost_radius')
        max_v   = self._p('max_v')

        # Initialise all K states from current robot pose
        states      = np.zeros((K, 3))  # [x, y, yaw]
        states[:, 0] = self.x
        states[:, 1] = self.y
        states[:, 2] = self.yaw

        costs = np.zeros(K)
        trajs = [[(self.x, self.y, self.yaw)] for _ in range(K)]

        obs_arr = np.array(self.obstacles) if self.obstacles else None

        for t in range(T):
            v_t  = U_k[:, t, 0]
            om_t = U_k[:, t, 1]

            # Unicycle model
            states[:, 0] += v_t * np.cos(states[:, 2]) * dt
            states[:, 1] += v_t * np.sin(states[:, 2]) * dt
            states[:, 2] += om_t * dt

            # Obstacle cost
            if obs_arr is not None and len(obs_arr) > 0:
                dx       = states[:, 0:1] - obs_arr[:, 0]  # (K, N)
                dy       = states[:, 1:2] - obs_arr[:, 1]  # (K, N)
                min_dist = np.min(np.sqrt(dx**2 + dy**2), axis=1)  # (K,)

                # Hard collision penalty
                costs += (min_dist < r_rob) * 1e6 * w_obs

                # Soft proximity penalty
                in_zone = (min_dist >= r_rob) & (min_dist < r_pen)
                costs[in_zone] += w_obs * (1.0 - min_dist[in_zone] / r_pen)

            # Heading cost — evaluated at every step (key MPPI advantage)
            yaw_err = np.abs(np.arctan2(
                np.sin(desired_yaw - states[:, 2]),
                np.cos(desired_yaw - states[:, 2])))
            costs += w_head * (yaw_err / math.pi)

            # Velocity cost
            costs += w_vel * (1.0 - np.clip(v_t / max_v, 0, 1))

            for k in range(K):
                trajs[k].append((float(states[k, 0]),
                                 float(states[k, 1]),
                                 float(states[k, 2])))

        # Terminal heading cost
        final_err = np.abs(np.arctan2(
            np.sin(desired_yaw - states[:, 2]),
            np.cos(desired_yaw - states[:, 2])))
        costs += self._p('w_terminal') * (final_err / math.pi)

        return costs, trajs

    def _rollout_single(self, U, dt):
        x, y, yaw = self.x, self.y, self.yaw
        traj = [(x, y, yaw)]
        for t in range(len(U)):
            x   += U[t, 0] * math.cos(yaw) * dt
            y   += U[t, 0] * math.sin(yaw) * dt
            yaw += U[t, 1] * dt
            traj.append((x, y, yaw))
        return traj

    # ── Obstacle extraction ───────────────────────────────────────────────────
    def _scan_to_obstacles(self):
        obs = []
        for i, r in enumerate(self.scan_ranges):
            if not math.isfinite(r) or r <= 0.02:
                continue
            angle = self.scan_angle_min + i * self.scan_angle_inc
            lx    = r * math.cos(angle)
            ly    = r * math.sin(angle)
            cy    = math.cos(self.yaw)
            sy    = math.sin(self.yaw)
            obs.append((self.x + cy * lx - sy * ly,
                        self.y + sy * lx + cy * ly))
        return obs

    # ── Visualisation ─────────────────────────────────────────────────────────
    def _publish_viz(self, viz_trajs, best_traj):
        now  = self.get_clock().now().to_msg()
        lt   = int(0.15e9)

        # Sampled trajectories — light blue
        ma = MarkerArray()
        for idx, traj in enumerate(viz_trajs):
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp    = now
            m.ns, m.id        = 'mppi_samples', idx
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 0.01
            m.color.r = 0.2; m.color.g = 0.5
            m.color.b = 1.0; m.color.a = 0.3
            m.lifetime.nanosec = lt
            for (x, y, _) in traj:
                pt = Point(); pt.x = x; pt.y = y; pt.z = 0.05
                m.points.append(pt)
            ma.markers.append(m)
        self.traj_pub.publish(ma)

        # Optimal trajectory — bright green
        if best_traj:
            bm = Marker()
            bm.header.frame_id = 'odom'
            bm.header.stamp    = now
            bm.ns, bm.id       = 'mppi_best', 0
            bm.type            = Marker.LINE_STRIP
            bm.action          = Marker.ADD
            bm.scale.x         = 0.05
            bm.color.r = 0.0; bm.color.g = 1.0
            bm.color.b = 0.2; bm.color.a = 1.0
            bm.lifetime.nanosec = lt
            for (x, y, _) in best_traj:
                pt = Point(); pt.x = x; pt.y = y; pt.z = 0.05
                bm.points.append(pt)
            self.best_pub.publish(bm)


def main(args=None):
    rclpy.init(args=args)
    node = MPPIPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()_
