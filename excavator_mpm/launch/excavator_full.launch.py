"""
excavator_full.launch.py
========================
Launches both the MPM simulation and the trajectory sender together.

The trajectory sender subscribes to /excavator/sim_time published by the sim,
so it automatically waits for the sim to be ready before moving any joints.

Arguments
---------
  viewer   : gl (default) | none
  usd      : path to arm.usd  (default: ~/ecorobotic/excavator/arm.usd)
  cycles   : number of dig cycles to run  (default: 0 = infinite)

Usage
-----
    ros2 launch excavator_mpm excavator_full.launch.py
    ros2 launch excavator_mpm excavator_full.launch.py cycles:=3
    ros2 launch excavator_mpm excavator_full.launch.py viewer:=none cycles:=1
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ── Python environment with warp + newton + rclpy ────────────────────────────
_PYENV_PYTHON = os.path.expanduser(
    "~/.pyenv/versions/ecorobotic-newton-ros2/bin/python"
)
_PYENV_SITE = os.path.expanduser(
    "~/.pyenv/versions/ecorobotic-newton-ros2/lib/python3.10/site-packages"
)
_ROS2_DIST = "/opt/ros/humble/local/lib/python3.10/dist-packages"
_ROS2_SITE = "/opt/ros/humble/lib/python3.10/site-packages"

_sim_env = {
    "PYTHONPATH": ":".join([_PYENV_SITE, _ROS2_DIST, _ROS2_SITE,
                             os.environ.get("PYTHONPATH", "")]),
}

# trajectory_sender only needs rclpy — system Python is fine
_sender_env = {
    "PYTHONPATH": ":".join([_ROS2_DIST, _ROS2_SITE,
                             os.environ.get("PYTHONPATH", "")]),
}

# arm.usd must stay at original path so embedded relative mesh refs resolve
_ARM_USD = os.path.expanduser("~/ecorobotic/excavator/arm.usd")


def generate_launch_description():
    return LaunchDescription([

        # ── Launch arguments ──────────────────────────────────────────────────
        DeclareLaunchArgument(
            "viewer", default_value="gl",
            description="Viewer type: gl or none"),

        DeclareLaunchArgument(
            "usd", default_value=_ARM_USD,
            description="Path to arm.usd"),

        DeclareLaunchArgument(
            "cycles", default_value="0",
            description="Dig cycles to run (0 = infinite)"),

        # ── Node 1: MPM simulation ────────────────────────────────────────────
        Node(
            package="excavator_mpm",
            executable="excavator_ros2",
            name="excavator_sim",
            output="screen",
            additional_env=_sim_env,
            prefix=_PYENV_PYTHON,
            arguments=[
                "--viewer", LaunchConfiguration("viewer"),
                "--usd",    LaunchConfiguration("usd"),
            ],
        ),

        # ── Node 2: trajectory sender (delayed 3 s to let sim initialise) ────
        # The sender waits for /excavator/sim_time anyway, but the delay avoids
        # a flood of "waiting for sim_time" warnings during CUDA kernel loading.
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package="excavator_mpm",
                    executable="trajectory_sender",
                    name="excavator_trajectory",
                    output="screen",
                    additional_env=_sender_env,
                    parameters=[{
                        "cycles": LaunchConfiguration("cycles"),
                    }],
                ),
            ],
        ),
    ])
