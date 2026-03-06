#!/usr/bin/env python3
"""
dwa_node.py  -  DWB planner with VFH obstacle steering
=======================================================
Requirement: must use DWB (Dynamic Window Approach) trajectory scoring.

The problem with pure DWB + goal vector:
  GoalAlignCritic points the robot straight at the goal vector.
  If a wall is in that direction, DWB scores all forward trajectories
  as obstacle-colliding and picks a random bad one -> oscillation.

Fix: Vector Field Histogram (VFH) pre-filter.
  Before DWB runs each tick, compute a "steered vector" that:
    1. Starts as the locked goal vector.
    2. Scans LiDAR 360deg, builds a histogram of obstacle density per 5deg bin.
    3. Finds the nearest open sector to the goal vector direction.
    4. Blends goal vector toward that open sector proportionally to how blocked
       the goal direction is.
  DWB then scores trajectories against this steered vector instead of the raw
  goal vector -- so it naturally curves around walls while still pulling back
  toward the original direction once clear.

No state machines. No escape loops. DWB does all the work; VFH just gives it
a sane desired direction.
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, Vector3
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class RobotState:
    __slots__ = ('x', 'y', 'yaw', 'v', 'omega')
    def __init__(self):
        self.x = self.y = self.yaw = self.v = self.omega = 0.0


class GoalAlignCritic:
    """Score trajectory by displacement alignment with a world-frame unit vector."""
    def score(self, traj, gvx, gvy):
        if not traj or len(traj) < 2:
            return 1.0
        dx = traj[-1][0] - traj[0][0]
        dy = traj[-1][1] - traj[0][1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 1.0
        dot = (dx/dist)*gvx + (dy/dist)*gvy
        return (1.0 - dot) / 2.0   # 0=perfect alignment, 1=opposite


class ObstacleClearanceCritic:
    def score(self, traj, obstacles, robot_radius, penalty_radius):
        min_dist = float('inf')
        for (px, py, _) in traj:
            for (ox, oy) in obstacles:
                d = math.hypot(px - ox, py - oy)
                if d < robot_radius:
                    return float('inf'), False, 0.0
                if d < min_dist:
                    min_dist = d
        if min_dist > penalty_radius:
            return 0.0, True, min_dist
        return 1.0 - (min_dist / penalty_radius), True, min_dist


class VelocityCritic:
    def score(self, v, max_speed):
        if max_speed <= 0:
            return 0.0
        return max(0.0, 1.0 - (v / max_speed))


class VFHSteering:
    """
    Vector Field Histogram pre-filter for DWB.

    Builds a polar obstacle density histogram from LiDAR, finds the
    nearest open sector to the goal direction, and returns a steered
    vector that blends goal direction with the clearest nearby path.

    The robot will naturally curve around obstacles and snap back to
    the goal vector once clear -- without any state machine.
    """
    BIN_DEG   = 5       # histogram bin width in degrees
    BINS      = 360 // BIN_DEG

    # A bin is "open" if its obstacle density is below this threshold.
    # Density is computed as: sum(1/r^2) for all readings in that bin.
    # Low value = far obstacles or no obstacles = safe to go there.
    OPEN_THRESHOLD = 8.0

    # Smoothing window: average N adjacent bins to avoid sharp steering
    SMOOTH_WINDOW = 3

    def __init__(self, robot_radius=0.115):
        self.robot_radius = robot_radius

    def steer(self, scan_ranges, scan_angle_min, scan_angle_inc,
              goal_vx, goal_vy, robot_yaw):
        """
        Returns (steered_vx, steered_vy, blockage_ratio).
        blockage_ratio: 0.0 = goal direction fully open, 1.0 = fully blocked.
        """
        # Build obstacle density histogram in robot frame
        hist = [0.0] * self.BINS
        for i, r in enumerate(scan_ranges):
            if not math.isfinite(r) or r < 0.05 or r > 3.5:
                continue
            angle_rad = scan_angle_min + i * scan_angle_inc
            angle_deg = math.degrees(angle_rad)
            while angle_deg >= 180:  angle_deg -= 360
            while angle_deg < -180:  angle_deg += 360
            bin_idx = int((angle_deg + 180) / self.BIN_DEG) % self.BINS
            # Weight by inverse square of distance -- close obstacles count more
            hist[bin_idx] += 1.0 / (r * r + 0.01)

        # Smooth histogram to avoid jitter
        smoothed = [0.0] * self.BINS
        hw = self.SMOOTH_WINDOW // 2
        for i in range(self.BINS):
            total = 0.0
            for j in range(-hw, hw+1):
                total += hist[(i+j) % self.BINS]
            smoothed[i] = total / self.SMOOTH_WINDOW

        # Goal direction in robot frame -> bin index
        goal_world_angle = math.atan2(goal_vy, goal_vx)
        goal_robot_angle = goal_world_angle - robot_yaw
        while goal_robot_angle >= math.pi:  goal_robot_angle -= 2*math.pi
        while goal_robot_angle < -math.pi:  goal_robot_angle += 2*math.pi
        goal_deg = math.degrees(goal_robot_angle)
        goal_bin = int((goal_deg + 180) / self.BIN_DEG) % self.BINS

        # How blocked is the goal direction?
        goal_density = smoothed[goal_bin]
        blockage = min(1.0, goal_density / self.OPEN_THRESHOLD)

        if blockage < 0.15:
            # Goal direction is open -- use it directly
            return goal_vx, goal_vy, blockage

        # Find the nearest open bin to the goal bin
        best_bin   = goal_bin
        best_delta = float('inf')
        for delta in range(1, self.BINS // 2):
            for sign in (1, -1):
                candidate = (goal_bin + sign * delta) % self.BINS
                if smoothed[candidate] < self.OPEN_THRESHOLD:
                    dist = delta  # angular distance from goal
                    if dist < best_delta:
                        best_delta = dist
                        best_bin = candidate
                    break   # found nearest on this side, stop searching this side

        # Convert best_bin back to world-frame vector
        best_deg_robot = (best_bin * self.BIN_DEG) - 180 + self.BIN_DEG / 2.0
        best_angle_world = math.radians(best_deg_robot) + robot_yaw
        open_vx = math.cos(best_angle_world)
        open_vy = math.sin(best_angle_world)

        # Blend: more blocked -> steer more toward open direction
        # Use a smooth sigmoid-like blend so steering is gradual
        blend = min(1.0, blockage * 1.5)   # ramps up as blockage increases
        svx = goal_vx * (1-blend) + open_vx * blend
        svy = goal_vy * (1-blend) + open_vy * blend
        mag = math.hypot(svx, svy)
        if mag < 1e-6:
            return open_vx, open_vy, blockage
        return svx/mag, svy/mag, blockage


# Zone thresholds
ZONE_CLEAR   = 0.50
ZONE_SLOW    = 0.35
ZONE_CAUTION = 0.25
ZONE_DANGER  = 0.22


class DWBPlannerNode(Node):
    def __init__(self):
        super().__init__('dwb_planner')
        self._declare_parameters()

        self.state        = RobotState()
        self.scan_ranges  = []
        self._scan_buffer = []
        self.scan_angle_min = 0.0
        self.scan_angle_inc = 0.0
        self.odom_received  = False
        self.scan_received  = False

        self.goal_vx = 0.0
        self.goal_vy = 0.0

        self.vfh = VFHSteering(robot_radius=0.115)

        self.goal_align_critic = GoalAlignCritic()
        self.obstacle_critic   = ObstacleClearanceCritic()
        self.velocity_critic   = VelocityCritic()

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub    = self.create_publisher(Twist,       '/cmd_vel',                     10)
        self.traj_pub   = self.create_publisher(MarkerArray, '/dwa_planner/trajectories',    10)
        self.best_pub   = self.create_publisher(Marker,      '/dwa_planner/best_trajectory', 10)
        self.status_pub = self.create_publisher(String,      '/dwa_planner/status',          10)

        self.create_subscription(Vector3,   '/goal_vector', self._vector_cb, 10)
        self.create_subscription(LaserScan, '/scan',        self._scan_cb,   10)
        self.create_subscription(Odometry,  '/odom',        self._odom_cb,   10)
        self.create_timer(self._p('control_period'), self._control_loop)

        self.get_logger().info('DWB + VFH steering ready')

    def _declare_parameters(self):
        defaults = {
            'max_speed':            0.22,
            'min_speed':           -0.05,
            'max_yaw_rate':         2.84,
            'max_accel':            0.5,
            'max_dyaw_rate':        3.2,
            'v_resolution':         0.02,   # coarser = faster evaluation
            'yaw_rate_resolution':  0.1,
            'predict_time':         1.0,
            'control_period':       0.1,
            'goal_align_weight':    1.2,
            'obstacle_weight':      3.5,
            'velocity_weight':      0.2,
            'robot_radius':         0.115,
            'obstacle_cost_radius': 0.28,
        }
        for name, val in defaults.items():
            self.declare_parameter(name, val)

    def _p(self, name):
        return self.get_parameter(name).value

    def _vector_cb(self, msg):
        mag = math.hypot(msg.x, msg.y)
        if mag < 0.01:
            if self.goal_vx != 0.0 or self.goal_vy != 0.0:
                self.get_logger().info('Vector cleared -> IDLE')
            self.goal_vx = self.goal_vy = 0.0
        else:
            nvx, nvy = msg.x/mag, msg.y/mag
            if abs(nvx - self.goal_vx) > 0.02 or abs(nvy - self.goal_vy) > 0.02:
                self.get_logger().info(
                    f'Vector locked -> {math.degrees(math.atan2(nvy,nvx)):.0f}deg')
            self.goal_vx, self.goal_vy = nvx, nvy

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        self.state.x = pos.x; self.state.y = pos.y
        siny = 2.0*(q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        self.state.yaw   = math.atan2(siny, cosy)
        self.state.v     = msg.twist.twist.linear.x
        self.state.omega = msg.twist.twist.angular.z
        self.odom_received = True

    def _scan_cb(self, msg):
        new = list(msg.ranges)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_inc = msg.angle_increment
        self._scan_buffer.append(new)
        if len(self._scan_buffer) > 3:
            self._scan_buffer.pop(0)
        if len(self._scan_buffer) >= 2:
            self.scan_ranges = [min(a,b)
                for a,b in zip(self._scan_buffer[-2], self._scan_buffer[-1])]
        else:
            self.scan_ranges = new
        self.scan_received = True

    def _nearest(self):
        return min((r for r in self.scan_ranges
                    if math.isfinite(r) and 0.08 < r < 3.5), default=float('inf'))

    def _control_loop(self):
        if not (self.odom_received and self.scan_received):
            return

        s = String()

        if self.goal_vx == 0.0 and self.goal_vy == 0.0:
            self.cmd_pub.publish(Twist())
            s.data = 'IDLE'; self.status_pub.publish(s); return

        near = self._nearest()

        # ── VFH: compute steered vector ───────────────────────────────────────
        svx, svy, blockage = self.vfh.steer(
            self.scan_ranges, self.scan_angle_min, self.scan_angle_inc,
            self.goal_vx, self.goal_vy, self.state.yaw)

        # ── Zone-based speed limit ────────────────────────────────────────────
        if near < ZONE_DANGER:
            eff_max_speed = 0.0
            eff_align_w   = 0.3
            s.data = 'DANGER'
        elif near < ZONE_CAUTION:
            t = (near - ZONE_DANGER) / (ZONE_CAUTION - ZONE_DANGER)
            eff_max_speed = 0.07 * t
            eff_align_w   = self._p('goal_align_weight') * 0.4
            s.data = 'CAUTION'
        elif near < ZONE_SLOW:
            t = (near - ZONE_CAUTION) / (ZONE_SLOW - ZONE_CAUTION)
            eff_max_speed = 0.07 + 0.10 * t
            eff_align_w   = self._p('goal_align_weight') * (0.4 + 0.4*t)
            s.data = 'SLOW'
        else:
            eff_max_speed = self._p('max_speed')
            eff_align_w   = self._p('goal_align_weight')
            s.data = 'NAVIGATING'

        if blockage > 0.1:
            s.data += f'|VFH_STEER({blockage:.0%})'

        self.status_pub.publish(s)

        obstacles = self._scan_to_obstacles()
        dw        = self._dynamic_window(self._p('control_period'))

        best_v, best_omega, all_trajs, best_traj, no_valid, n_col = \
            self._evaluate_window(dw, obstacles, eff_align_w, eff_max_speed, svx, svy)

        if no_valid:
            # Last resort: rotate toward the steered vector direction
            err = math.atan2(svy, svx) - self.state.yaw
            err = math.atan2(math.sin(err), math.cos(err))
            cmd = Twist()
            cmd.angular.z = max(-2.0, min(2.0, 3.0 * err))
            self.get_logger().warn(
                f'NO VALID TRAJ near={near:.2f}m col={n_col} -> rotating to clear')
            self.cmd_pub.publish(cmd)
        else:
            cmd = Twist()
            cmd.linear.x  = best_v
            cmd.angular.z = best_omega
            self.cmd_pub.publish(cmd)

        self._publish_viz(all_trajs, best_traj)

    def _dynamic_window(self, dt):
        v_min  = max(self._p('min_speed'),     self.state.v     - self._p('max_accel')     * dt)
        v_max  = min(self._p('max_speed'),     self.state.v     + self._p('max_accel')     * dt)
        om_min = max(-self._p('max_yaw_rate'), self.state.omega - self._p('max_dyaw_rate') * dt)
        om_max = min( self._p('max_yaw_rate'), self.state.omega + self._p('max_dyaw_rate') * dt)
        return v_min, v_max, om_min, om_max

    def _evaluate_window(self, dw, obstacles, eff_align, eff_max_speed, svx, svy):
        v_min, v_max, om_min, om_max = dw
        v_max = min(v_max, eff_max_speed)
        T, v_res, om_res, dt = (self._p(k) for k in
            ('predict_time','v_resolution','yaw_rate_resolution','control_period'))
        robot_r  = self._p('robot_radius')
        cost_r   = self._p('obstacle_cost_radius')
        obs_w    = self._p('obstacle_weight')
        vel_w    = self._p('velocity_weight')
        max_spd  = self._p('max_speed')

        best_score = float('inf')
        best_v = best_omega = 0.0
        all_trajs, best_traj = [], []
        n_col = 0

        for v in np.unique(np.append(np.arange(v_min, v_max+v_res*0.5, v_res), 0.0)):
            for omega in np.arange(om_min, om_max+om_res*0.5, om_res):
                traj = self._simulate(v, omega, T, dt)
                obs_cost, valid, _ = self.obstacle_critic.score(
                    traj, obstacles, robot_r, cost_r)
                if not valid:
                    n_col += 1; continue
                align_cost = self.goal_align_critic.score(traj, svx, svy)
                vel_cost   = self.velocity_critic.score(v, max_spd)
                score = eff_align*align_cost + obs_w*obs_cost + vel_w*vel_cost
                all_trajs.append(traj)
                if score < best_score:
                    best_score = score
                    best_v, best_omega, best_traj = v, omega, traj

        return best_v, best_omega, all_trajs, best_traj, len(best_traj)==0, n_col

    def _simulate(self, v, omega, T, dt):
        x, y, yaw = self.state.x, self.state.y, self.state.yaw
        traj, t = [(x,y,yaw)], 0.0
        while t < T:
            x += v*math.cos(yaw)*dt; y += v*math.sin(yaw)*dt; yaw += omega*dt
            traj.append((x,y,yaw)); t += dt
        return traj

    def _scan_to_obstacles(self):
        obs = []
        cy, sy = math.cos(self.state.yaw), math.sin(self.state.yaw)
        for i, r in enumerate(self.scan_ranges):
            if not math.isfinite(r) or r <= 0.08 or r > 3.5: continue
            angle = self.scan_angle_min + i*self.scan_angle_inc
            lx, ly = r*math.cos(angle), r*math.sin(angle)
            obs.append((self.state.x+cy*lx-sy*ly, self.state.y+sy*lx+cy*ly))
        return obs

    def _publish_viz(self, all_trajs, best_traj):
        now, lt = self.get_clock().now().to_msg(), int(0.15e9)
        ma = MarkerArray()
        for idx, traj in enumerate(all_trajs):
            m = Marker()
            m.header.frame_id='odom'; m.header.stamp=now
            m.ns='candidates'; m.id=idx; m.type=Marker.LINE_STRIP; m.action=Marker.ADD
            m.scale.x=0.01
            m.color.r=0.3; m.color.g=0.6; m.color.b=1.0; m.color.a=0.4
            m.lifetime.nanosec=lt
            for (x,y,_) in traj:
                pt=Point(); pt.x=x; pt.y=y; pt.z=0.05; m.points.append(pt)
            ma.markers.append(m)
        self.traj_pub.publish(ma)
        if best_traj:
            bm = Marker()
            bm.header.frame_id='odom'; bm.header.stamp=now
            bm.ns='best'; bm.id=0; bm.type=Marker.LINE_STRIP; bm.action=Marker.ADD
            bm.scale.x=0.04
            bm.color.r=0.0; bm.color.g=1.0; bm.color.b=0.2; bm.color.a=0.9
            bm.lifetime.nanosec=lt
            for (x,y,_) in best_traj:
                pt=Point(); pt.x=x; pt.y=y; pt.z=0.05; bm.points.append(pt)
            self.best_pub.publish(bm)


def main(args=None):
    rclpy.init(args=args)
    node = DWBPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try: node.cmd_pub.publish(Twist())
        except: pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
