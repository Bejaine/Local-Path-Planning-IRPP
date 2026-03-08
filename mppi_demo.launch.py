"""
mppi_demo.launch.py
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
  ros2 launch mppi_planner mppi_demo.launch.py

  # Dynamic obstacles (bonus marks)
  ros2 launch mppi_planner mppi_demo.launch.py dynamic:=true

  # Without RViz (headless)
  ros2 launch mppi_planner mppi_demo.launch.py launch_rviz:=false
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             ExecuteProcess, TimerAction, AppendEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, PythonExpression
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


# ── Ensure GZ_SIM_RESOURCE_PATH is in os.environ BEFORE generate_launch_description
#    so that gz_sim.launch.py's launch_gz() → environ.get() picks it up when
#    building its own additional_env dict for the Gazebo subprocess.
_pkg_tb3_gazebo_path = get_package_share_directory('turtlebot3_gazebo')
_pkg_dwa = get_package_share_directory('mppi_planner') # We'll replace this below based on file
_models_path_global = os.path.join(_pkg_tb3_gazebo_path, 'models')
_models_path_local = os.path.join(_pkg_dwa, 'models')

_cur = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
paths_to_add = [_models_path_global, _models_path_local]
for p in paths_to_add:
    if p not in _cur:
        _cur = (p + os.pathsep + _cur).strip(os.pathsep)
os.environ['GZ_SIM_RESOURCE_PATH'] = _cur


def generate_launch_description():
    pkg_dwa = get_package_share_directory('mppi_planner')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_tb3_desc   = get_package_share_directory('turtlebot3_description')

    models_path_global = os.path.join(pkg_tb3_gazebo, 'models')
    models_path_local = os.path.join(pkg_dwa, 'models')

    # Also set as launch action (belt-and-suspenders)
    set_gz_resource_path = AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', models_path_local + os.pathsep + models_path_global)

    # ── Launch arguments ──────────────────────────────────────────────────
    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz', default_value='true',
        description='Launch RViz2 for visualisation')

    dynamic_arg = DeclareLaunchArgument(
        'dynamic', default_value='false',
        description='Spawn moving obstacles (bonus marks)')

    city_arg = DeclareLaunchArgument(
        'city', default_value='true',
        description='Launch the massive City Junction environment')

    # Resolve world path directly — avoids substitution-list concatenation bugs
    _city_world  = os.path.join(pkg_dwa, 'worlds', 'city_junction.world')
    _basic_world = os.path.join(pkg_dwa, 'worlds', 'mppi_world.world')

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=PythonExpression(["'", _city_world, "' if '", LaunchConfiguration('city'), "' == 'true' else '", _basic_world, "'"]),
        description='Gazebo world file (.world / .sdf)')

    # ── Gazebo ────────────────────────────────────────────────────────────
    ros_gz_sim_pkg = get_package_share_directory('ros_gz_sim')

    gz_sim_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -v2 ', LaunchConfiguration('world')]}.items()
    )

    # ── ROS-GZ Bridge for moving cars ─────────────────────────────────────
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/model/car_1/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/model/car_2/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '--ros-args', '-p', 'use_sim_time:=true'
        ],
        output='screen'
    )

    # ── TurtleBot3 Waffle spawn — use local SDF with absolute file:// mesh URIs
    #    to guarantee Gazebo's GUI renderer finds the mesh files.
    _tb3_sdf   = os.path.join(pkg_dwa, 'models', 'turtlebot3_waffle.sdf')
    _tb3_bridge_yaml = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'params', 'turtlebot3_waffle_bridge.yaml')

    tb3_spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'turtlebot3_waffle',
                   '-file', _tb3_sdf,
                   '-x', '0.0', '-y', '-4.0', '-z', '0.01'],
        output='screen')

    tb3_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '--ros-args', 
            '-p', 'use_sim_time:=true', 
            '-p', f'config_file:={_tb3_bridge_yaml}'
        ],
        output='screen')

    # ── robot_state_publisher ─────────────────────────────────────────────
    urdf_file = os.path.join(pkg_tb3_desc, 'urdf', 'turtlebot3_waffle.urdf')
    robot_desc = Command(['xacro ', urdf_file])

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_desc,
                     'use_sim_time': True}])

    # ── DWA planner ───────────────────────────────────────────────────────
    mppi_node = Node(
        package='mppi_planner',
        executable='mppi_node',
        name='mppi_planner',
        parameters=[
            os.path.join(pkg_dwa, 'config', 'mppi_params.yaml'),
            {'use_sim_time': True}
        ],
        output='screen',
        emulate_tty=True)

    # ── Dynamic obstacles (bonus) ─────────────────────────────────────────
    dynamic_obstacles = Node(
        package='mppi_planner',
        executable='dynamic_obstacles',
        name='dynamic_obstacles',
        parameters=[{'use_sim_time': True,
                     'num_obstacles': 3,
                     'speed': 0.12}],
        condition=IfCondition(LaunchConfiguration('dynamic')),
        output='screen')

    city_traffic = Node(
        package='mppi_planner',
        executable='city_traffic',
        name='city_traffic',
        condition=IfCondition(LaunchConfiguration('city')),
        output='screen'
    )

    # ── RViz2 ─────────────────────────────────────────────────────────────
    rviz_cfg = os.path.join(pkg_dwa, 'config', 'mppi_rviz.rviz')
    city_map_viz = Node(
        package="mppi_planner",
        executable="city_map_viz",
        name="city_map_viz",
        condition=IfCondition(LaunchConfiguration("city")),
        output="screen"
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg] if os.path.exists(rviz_cfg) else [],
        condition=IfCondition(LaunchConfiguration('launch_rviz')),
        parameters=[{'use_sim_time': True}])

    # Delay spawns to let Gazebo fully initialise the world first
    delayed_spawn      = TimerAction(period=8.0, actions=[tb3_spawn, tb3_bridge])
    delayed_dwa        = TimerAction(period=10.0, actions=[mppi_node])
    delayed_dyn        = TimerAction(period=12.0, actions=[dynamic_obstacles])
    delayed_city       = TimerAction(period=12.0, actions=[city_traffic])

    return LaunchDescription([
        set_gz_resource_path,
        launch_rviz_arg,
        dynamic_arg,
        city_arg,
        world_arg,
        gz_sim_cmd,
        gz_bridge,
        rsp,
        delayed_spawn,
        delayed_dwa,
        delayed_dyn,
        delayed_city,
        city_map_viz,
        rviz,
    ])
