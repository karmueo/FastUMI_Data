"""定义 fastumi_rm75 ROS 2 Python 包的安装规则。"""

from glob import glob

from setuptools import find_packages, setup


# ROS 2 包名，用于安装入口和共享资源。
PACKAGE_NAME = "fastumi_rm75"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{PACKAGE_NAME}"],
        ),
        (f"share/{PACKAGE_NAME}", ["package.xml", "README.md"]),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE_NAME}/assets", glob("assets/*")),
    ],
    install_requires=["placo>=0.9.23", "setuptools"],
    zip_safe=True,
    maintainer="FastUMI Maintainer",
    maintainer_email="maintainer@example.com",
    description="FastUMI RM75 Placo 关节控制与夹爪适配。",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "rm75_policy_bridge = "
            "fastumi_rm75.rm75_policy_bridge:main",
            "rm75_placo_controller = "
            "fastumi_rm75.rm75_placo_controller:main",
            "gripper_bridge = fastumi_rm75.gripper_bridge_node:main",
        ],
    },
)
