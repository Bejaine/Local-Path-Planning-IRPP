from setuptools import setup
import os
from glob import glob

package_name = 'dwa_planner'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@university.edu',
    description='DWA local motion planner for TurtleBot3',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dwa_node = dwa_planner.dwa_node:main',
            'goal_publisher = dwa_planner.goal_publisher:main',
            'dynamic_obstacles = dwa_planner.dynamic_obstacles:main',
        ],
    },
)
