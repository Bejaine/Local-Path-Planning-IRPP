#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
import time
import csv
import psutil
import os

class PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('performance_monitor')
        
        # Subscriptions
        self.create_subscription(Vector3, '/goal_vector', self._goal_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        
        # State variables
        self.is_recording = False
        self.start_time = 0.0
        
        # Metrics
        self.total_distance = 0.0
        self.min_clearance = float('inf')
        self.angular_velocities = []
        self.cpu_usages = []
        self.ram_usages = []
        
        # Tracking previous pose for distance calculation
        self.prev_x = None
        self.prev_y = None
        
        # Resource tracking timer (1 Hz)
        self.create_timer(1.0, self._track_resources)
        
        # CSV Setup
        self.csv_filename = os.path.expanduser('~/ws/src/dwa_planner/planner_performance_metrics.csv')
        self._init_csv()
        
        self.get_logger().info('Performance Monitor ready. Waiting for movement commands...')

    def _init_csv(self):
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Run ID', 'Total Time (s)', 'Distance (m)', 'Avg Speed (m/s)', 
                                 'Min Clearance (m)', 'Smoothness (Angular Var)', 'Avg CPU (%)', 'Avg RAM (%)'])

    def _goal_cb(self, msg):
        mag = math.hypot(msg.x, msg.y)
        
        # Start recording if vector is non-zero and we aren't already recording
        if mag > 0.01 and not self.is_recording:
            self.get_logger().info('--- RUN STARTED: Recording Metrics ---')
            self.is_recording = True
            self.start_time = time.time()
            self._reset_metrics()
            
        # Stop recording and save if vector is zero and we are recording
        elif mag < 0.01 and self.is_recording:
            self.is_recording = False
            self._save_run_metrics()

    def _odom_cb(self, msg):
        if not self.is_recording:
            return
            
        # Calculate distance traveled
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        
        if self.prev_x is not None and self.prev_y is not None:
            dx = curr_x - self.prev_x
            dy = curr_y - self.prev_y
            self.total_distance += math.hypot(dx, dy)
            
        self.prev_x = curr_x
        self.prev_y = curr_y
        
        # Track angular velocity for smoothness
        self.angular_velocities.append(msg.twist.twist.angular.z)

    def _scan_cb(self, msg):
        if not self.is_recording:
            return
            
        # Find closest obstacle in this scan
        valid_ranges = [r for r in msg.ranges if math.isfinite(r) and r > 0.02]
        if valid_ranges:
            current_min = min(valid_ranges)
            if current_min < self.min_clearance:
                self.min_clearance = current_min

    def _track_resources(self):
        if self.is_recording:
            # Note: This tracks overall system usage. For exact node usage, PID tracking is required.
            self.cpu_usages.append(psutil.cpu_percent())
            self.ram_usages.append(psutil.virtual_memory().percent)

    def _reset_metrics(self):
        self.total_distance = 0.0
        self.min_clearance = float('inf')
        self.angular_velocities.clear()
        self.cpu_usages.clear()
        self.ram_usages.clear()
        self.prev_x = None
        self.prev_y = None

    def _calculate_variance(self, data):
        if not data or len(data) < 2:
            return 0.0
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return variance

    def _save_run_metrics(self):
        end_time = time.time()
        total_time = end_time - self.start_time
        avg_speed = self.total_distance / total_time if total_time > 0 else 0.0
        smoothness = self._calculate_variance(self.angular_velocities)
        avg_cpu = sum(self.cpu_usages) / len(self.cpu_usages) if self.cpu_usages else 0.0
        avg_ram = sum(self.ram_usages) / len(self.ram_usages) if self.ram_usages else 0.0
        
        run_id = time.strftime("%H:%M:%S")
        
        # Log to terminal
        self.get_logger().info('--- RUN FINISHED: Metrics Saved ---')
        self.get_logger().info(f'Time: {total_time:.2f}s | Dist: {self.total_distance:.2f}m | Speed: {avg_speed:.2f}m/s')
        self.get_logger().info(f'Clearance: {self.min_clearance:.2f}m | Smoothness: {smoothness:.4f} | CPU: {avg_cpu:.1f}%')

        # Write to CSV
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run_id, round(total_time, 2), round(self.total_distance, 2), 
                             round(avg_speed, 2), round(self.min_clearance, 2), 
                             round(smoothness, 4), round(avg_cpu, 1), round(avg_ram, 1)])

def main(args=None):
    rclpy.init(args=args)
    node = PerformanceMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
