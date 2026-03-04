#!/usr/bin/env python3
import sys
import os
import tty
import termios
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

BANNER = """
╔══════════════════════════════════════╗
║   DWA Goal Velocity Publisher        ║
╠══════════════════════════════════════╣
║  w   forward                         ║
║  s   backward / slow                 ║
║  a   turn left                       ║
║  d   turn right                      ║
║  SPACE  stop                         ║
║  q   quit                            ║
╚══════════════════════════════════════╝
"""

V_STEP = 0.02
O_STEP = 0.1
V_MAX  = 0.22
O_MAX  = 2.84

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
        self.pub = self.create_publisher(Twist, '/goal_velocity', 10)
        self.vx = 0.0
        self.oz = 0.0
        self.create_timer(0.1, self._publish)

    def _publish(self):
        msg = Twist()
        msg.linear.x  = self.vx
        msg.angular.z = self.oz
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

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            key = get_key(fd, old_settings)
            if key == 'w':
                node.vx = min(node.vx + V_STEP, V_MAX)
            elif key == 's':
                node.vx = max(node.vx - V_STEP, -0.05)
            elif key == 'a':
                node.oz = min(node.oz + O_STEP, O_MAX)
            elif key == 'd':
                node.oz = max(node.oz - O_STEP, -O_MAX)
            elif key == ' ':
                node.vx = 0.0
                node.oz = 0.0
            elif key in ('q', '\x03', '\x1b'):
                break
            print(f'\r  goal  vx={node.vx:+.2f} m/s   oz={node.oz:+.2f} rad/s   ', end='', flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        os.close(fd)
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
PYEOF
