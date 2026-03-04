"""
dwa_demo.launch.py
==================
Launches:
  1. Gazebo with a custom obstacle world
  2. TurtleBot3 (burger) spawned at the origin
  3. robot_state_publisher  (URDF → TF)
  4. DWA planner node
  5. RViz2  (optional, set launch_rviz:=false to skip)

Usage
-----
  # Static obstacles
  ros2 launch dwa_planner dwa_demo.launch.py

  # Dynamic obstacles (bonus marks)
  ros2 launch dwa_planner dwa_demo.launch.py dynamic:=true

  # Without RViz (headless)
  ros2 launch dwa_planner dwa_demo.launch.py launch_rviz:=false
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             ExecuteProcess, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_dwa = get_package_share_directory('dwa_planner')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_tb3_desc   = get_package_share_directory('turtlebot3_description')

    # ── Launch arguments ──────────────────────────────────────────────────
    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz', default_value='true',
        description='Launch RViz2 for visualisation')

    dynamic_arg = DeclareLaunchArgument(
        'dynamic', default_value='false',
        description='Spawn moving obstacles (bonus marks)')

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(pkg_dwa, 'worlds', 'dwa_world.world'),
        description='Gazebo world file')

    # ── Gazebo ────────────────────────────────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'),
                         'launch', 'gazebo.launch.py')),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'verbose': 'false',
        }.items())

    # ── TurtleBot3 spawn ──────────────────────────────────────────────────
    tb3_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, 'launch',
                         'spawn_turtlebot3.launch.py')),
        launch_arguments={
            'x_pose': '0.0',
            'y_pose': '0.0',
        }.items())

    # ── robot_state_publisher ─────────────────────────────────────────────
    urdf_file = os.path.join(pkg_tb3_desc, 'urdf', 'turtlebot3_burger.urdf')
    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_desc,
                     'use_sim_time': True}])

    # ── DWA planner ───────────────────────────────────────────────────────
    dwa_node = Node(
        package='dwa_planner',
        executable='dwa_node',
        name='dwa_planner',
        parameters=[
            os.path.join(pkg_dwa, 'config', 'dwa_params.yaml'),
            {'use_sim_time': True}
        ],
        output='screen',
        emulate_tty=True)

    # ── Dynamic obstacles (bonus) ─────────────────────────────────────────
    dynamic_obstacles = Node(
        package='dwa_planner',
        executable='dynamic_obstacles',
        name='dynamic_obstacles',
        parameters=[{'use_sim_time': True,
                     'num_obstacles': 3,
                     'speed': 0.12}],
        condition=IfCondition(LaunchConfiguration('dynamic')),
        output='screen')

    # ── RViz2 ─────────────────────────────────────────────────────────────
    rviz_cfg = os.path.join(pkg_dwa, 'config', 'dwa_rviz.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg] if os.path.exists(rviz_cfg) else [],
        condition=IfCondition(LaunchConfiguration('launch_rviz')),
        parameters=[{'use_sim_time': True}])

    # Delay DWA planner start to let Gazebo/robot finish loading
    delayed_dwa = TimerAction(period=5.0, actions=[dwa_node])
    delayed_dyn = TimerAction(period=7.0, actions=[dynamic_obstacles])

    return LaunchDescription([
        launch_rviz_arg,
        dynamic_arg,
        world_arg,
        gazebo,
        tb3_spawn,
        rsp,
        delayed_dwa,
        delayed_dyn,
        rviz,
    ])
