#!/usr/bin/env python3
"""
goal_publisher.py - simple WASD controller
Publishes /goal_vector (Vector3) as a world-frame unit direction.
w=forward(N), s=back(S), a=left(W), d=right(E), SPACE/x=stop
"""
import os, tty, termios, threading, math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3

BANNER = """
+==============================+
|  DWB Vector Controller       |
+------------------------------+
|  w = North  (forward)        |
|  s = South  (backward)       |
|  a = West   (left)           |
|  d = East   (right)          |
|  q = NW     e = NE           |
|  z = SW     c = SE           |
|  SPACE / x = stop            |
|  Ctrl+C    = quit            |
+==============================+
"""

DIRECTIONS = {
    'w': ( 0.0,    1.0),    # North
    's': ( 0.0,   -1.0),    # South
    'a': (-1.0,    0.0),    # West
    'd': ( 1.0,    0.0),    # East
    'q': (-0.707,  0.707),  # NW
    'e': ( 0.707,  0.707),  # NE
    'z': (-0.707, -0.707),  # SW
    'c': ( 0.707, -0.707),  # SE
}

def get_key(fd, old_settings):
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1).decode('utf-8', errors='ignore')
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')
        self.pub = self.create_publisher(Vector3, '/goal_vector', 10)
        self.vx  = 0.0
        self.vy  = 0.0
        self.create_timer(0.1, self._publish)

    def _publish(self):
        msg = Vector3()
        msg.x, msg.y = self.vx, self.vy
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisher()
    print(BANNER)

    try:
        fd = os.open('/dev/tty', os.O_RDWR | os.O_NOCTTY)
        old_settings = termios.tcgetattr(fd)
    except Exception as e:
        node.get_logger().error(f'Cannot open terminal: {e}')
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
        return

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    try:
        while rclpy.ok():
            key = get_key(fd, old_settings)
            if key in DIRECTIONS:
                vx, vy = DIRECTIONS[key]
                node.vx, node.vy = vx, vy
                deg = math.degrees(math.atan2(vy, vx))
                print(f'\r  {key} -> ({vx:+.2f},{vy:+.2f}) = {deg:+.0f}deg   ',
                      end='', flush=True)
            elif key in (' ', 'x'):
                node.vx = node.vy = 0.0
                print(f'\r  STOPPED                        ', end='', flush=True)
            elif key in ('c', '\x03', '\x1b'):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        os.close(fd)
        node.vx = node.vy = 0.0
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
