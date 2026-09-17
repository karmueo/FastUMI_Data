"""Unitree Dex1-1 ROS 2 包的安装规则。"""

from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = "unitree_gripper"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml", "README.md", "THIRD_PARTY_NOTICES.md", "UNITREE_SDK_LICENSE"]),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE_NAME}/vendor", ["vendor/dex1_1_gripper_server"]),
        (f"share/{PACKAGE_NAME}/vendor/lib", glob("vendor/lib/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="FastUMI Maintainer",
    maintainer_email="maintainer@example.com",
    description="Unitree Dex1-1 夹爪 ROS 2 控制与状态桥接。",
    license="Apache-2.0 AND BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "gripper_node = unitree_gripper.gripper_node:main",
        ],
    },
)
