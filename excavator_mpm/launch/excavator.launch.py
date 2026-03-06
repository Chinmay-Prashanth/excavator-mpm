import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Python 3.10 pyenv env — compatible with ROS2 Humble's rclpy (.so compiled for 3.10)
_PYENV_PYTHON = os.path.expanduser(
    "~/.pyenv/versions/ecorobotic-newton-ros2/bin/python"
)
_PYENV_SITE = os.path.expanduser(
    "~/.pyenv/versions/ecorobotic-newton-ros2/lib/python3.10/site-packages"
)
# ROS2 Humble Python packages live in dist-packages AND site-packages
_ROS2_DIST  = "/opt/ros/humble/local/lib/python3.10/dist-packages"
_ROS2_SITE  = "/opt/ros/humble/lib/python3.10/site-packages"

_env = {
    "PYTHONPATH": ":".join([_PYENV_SITE, _ROS2_DIST, _ROS2_SITE, os.environ.get("PYTHONPATH", "")]),
}


# arm.usd has relative mesh references — must load from its original location
_ARM_USD = os.path.expanduser("~/ecorobotic/excavator/arm.usd")


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "viewer", default_value="gl",
            description="Viewer type: gl or none",
        ),
        DeclareLaunchArgument(
            "usd", default_value=_ARM_USD,
            description="Path to arm.usd (must stay in its original folder so relative mesh paths resolve)",
        ),
        Node(
            package="excavator_mpm",
            executable="excavator_ros2",
            name="excavator_mpm",
            output="screen",
            additional_env=_env,
            prefix=_PYENV_PYTHON,
            arguments=[
                "--viewer", LaunchConfiguration("viewer"),
                "--usd",    LaunchConfiguration("usd"),
            ],
        ),
    ])
