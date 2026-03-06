"""
planner_launch.py
=================
Unified launch file — switch between DWB and MPPI with one argument.

Usage
-----
  # Run DWB (default)
  ros2 launch dwa_planner planner_launch.py planner:=dwb

  # Run MPPI
  ros2 launch dwa_planner planner_launch.py planner:=mppi

  # With dynamic obstacles (bonus)
  ros2 launch dwa_planner planner_launch.py planner:=mppi dynamic:=true

What this launches
------------------
  - Gazebo with turtlebot3_world
  - TurtleBot3 burger
  - robot_state_publisher
  - Either dwb_node or mppi_node depending on planner argument
  - goal_publisher (keyboard control)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Launch arguments ──────────────────────────────────────────────────
    planner_arg = DeclareLaunchArgument(
        'planner',
        default_value='dwb',
        description='Which planner to use: dwb or mppi')

    dynamic_arg = DeclareLaunchArgument(
        'dynamic',
        default_value='false',
        description='Spawn moving obstacles for bonus demo')

    # ── DWB node (launched when planner:=dwb) ─────────────────────────────
    dwb_node = Node(
        package='dwa_planner',
        executable='dwa_node',
        name='dwb_planner',
        output='screen',
        emulate_tty=True,
        condition=LaunchConfigurationEquals('planner', 'dwb'))

    # ── MPPI node (launched when planner:=mppi) ───────────────────────────
    mppi_node = Node(
        package='dwa_planner',
        executable='mppi_node',
        name='mppi_planner',
        output='screen',
        emulate_tty=True,
        condition=LaunchConfigurationEquals('planner', 'mppi'))

    # ── Dynamic obstacles (bonus, optional) ───────────────────────────────
    dynamic_obstacles = Node(
        package='dwa_planner',
        executable='dynamic_obstacles',
        name='dynamic_obstacles',
        output='screen',
        condition=IfCondition(LaunchConfiguration('dynamic')))

    # Delay planner start slightly to let Gazebo settle
    delayed_dwb  = TimerAction(period=3.0, actions=[dwb_node])
    delayed_mppi = TimerAction(period=3.0, actions=[mppi_node])
    delayed_dyn  = TimerAction(period=5.0, actions=[dynamic_obstacles])

    return LaunchDescription([
        planner_arg,
        dynamic_arg,
        delayed_dwb,
        delayed_mppi,
        delayed_dyn,
    ])
