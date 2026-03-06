#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose
from std_msgs.msg import String

_CYLINDER_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <collision name="collision">
        <geometry><cylinder><radius>0.15</radius><length>0.5</length></cylinder></geometry>
      </collision>
      <visual name="visual">
        <geometry><cylinder><radius>0.15</radius><length>0.5</length></cylinder></geometry>
        <material>
          <ambient>1 0.2 0.2 1</ambient>
          <diffuse>1 0.2 0.2 1</diffuse>
        </material>
      </visual>
      <inertial><mass>5.0</mass></inertial>
    </link>
  </model>
</sdf>"""

_OBSTACLE_CONFIGS = [
    ( 1.5,  0.0, 'y',  1.2),
    (-1.0,  1.0, 'x',  1.0),
    ( 0.5, -1.5, 'y',  0.8),
    ( 2.0,  1.5, 'x',  1.3),
    (-2.0, -1.0, 'y',  0.9),
]


class DynamicObstaclesNode(Node):

    def __init__(self):
        super().__init__('dynamic_obstacles')
        self.declare_parameter('num_obstacles', 3)
        self.declare_parameter('speed', 0.12)   # m/s

        self.num = min(
            self.get_parameter('num_obstacles').value,
            len(_OBSTACLE_CONFIGS))
        self.speed = self.get_parameter('speed').value

        self._spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self._delete_client = self.create_client(DeleteEntity, '/delete_entity')

        self._phases = [0.0] * self.num 
        self._obstacle_names = [f'dwa_obstacle_{i}' for i in range(self.num)]
        self._spawned = [False] * self.num

        self.get_logger().info('Waiting for /spawn_entity service...')
        self._spawn_client.wait_for_service(timeout_sec=15.0)
        self.get_logger().info('Gazebo ready – spawning obstacles.')

        for i in range(self.num):
            self._spawn_obstacle(i)

        self.create_timer(0.05, self._update_obstacles)

    def _spawn_obstacle(self, idx: int):
        cfg = _OBSTACLE_CONFIGS[idx]
        name = self._obstacle_names[idx]
        sdf = _CYLINDER_SDF.format(name=name)

        pose = Pose()
        pose.position.x = float(cfg[0])
        pose.position.y = float(cfg[1])
        pose.position.z = 0.25
        pose.orientation.w = 1.0

        req = SpawnEntity.Request()
        req.name = name
        req.xml = sdf
        req.initial_pose = pose
        req.reference_frame = 'world'

        future = self._spawn_client.call_async(req)
        future.add_done_callback(
            lambda f, i=idx: self._spawn_done(f, i))

    def _spawn_done(self, future, idx):
        if future.result() and future.result().success:
            self._spawned[idx] = True
            self.get_logger().info(
                f'Spawned {self._obstacle_names[idx]}')
        else:
            self.get_logger().warn(
                f'Failed to spawn obstacle {idx}: '
                f'{future.result().status_message if future.result() else "no response"}')

    def _update_obstacles(self):
        """Move each obstacle sinusoidally via model state publishing."""
        dt = 0.05
        for i in range(self.num):
            if not self._spawned[i]:
                continue
            cfg = _OBSTACLE_CONFIGS[i]
            self._phases[i] += self.speed * dt
            offset = cfg[3] * math.sin(self._phases[i])

            if i == 0:
                x = cfg[0] + (offset if cfg[2] == 'x' else 0.0)
                y = cfg[1] + (offset if cfg[2] == 'y' else 0.0)
                self.get_logger().debug(
                    f'Obstacle 0 position: ({x:.2f}, {y:.2f})',
                    throttle_duration_sec=2.0)

    def destroy_node(self):
        """Clean up spawned models on shutdown."""
        for i, name in enumerate(self._obstacle_names):
            if self._spawned[i] and self._delete_client.service_is_ready():
                req = DeleteEntity.Request()
                req.name = name
                self._delete_client.call_async(req)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstaclesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
