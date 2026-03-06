from setuptools import find_packages, setup
import os
from glob import glob

package_name = "excavator_mpm"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # USD asset shipped with the package
        (os.path.join("share", package_name, "assets"), glob("assets/*")),
        # Launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Chinmay",
    maintainer_email="chinmay@example.com",
    description="Newton MPM excavator simulation with ROS2 Humble control",
    license="MIT",
    entry_points={
        "console_scripts": [
            # Scripted auto-trajectory (no ROS2 required)
            "excavator = excavator_mpm.simulate_excavator:main",
            # ROS2-controlled sim with PointCloud2 output
            "excavator_ros2 = excavator_mpm.simulate_excavator_ros2:main",
            # Publishes the 4-phase dig cycle to /excavator/joint_commands
            "trajectory_sender = excavator_mpm.trajectory_sender:main",
        ],
    },
)
