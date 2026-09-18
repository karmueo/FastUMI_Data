"""定义 gripper_openness ROS 2 Python 包的安装规则。"""

from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = "gripper_openness"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{PACKAGE_NAME}"],
        ),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FastUMI Maintainer",
    maintainer_email="maintainer@example.com",
    description="ROS 2 夹爪范围标定和双 ArUco 开合度预测。",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "gripper_calibration_node = gripper_openness.gripper_calibration_node:main",
            "gripper_openness_node = gripper_openness.gripper_openness_node:main",
        ],
    },
)
