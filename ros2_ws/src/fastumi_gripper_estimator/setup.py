"""定义 fastumi_gripper_estimator ROS 2 Python 包的安装规则。"""

from glob import glob

from setuptools import find_packages, setup


# ROS 2 包名，用于安装路径和入口点注册。
PACKAGE_NAME = "fastumi_gripper_estimator"


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
    tests_require=["pytest"],
    maintainer_email="maintainer@example.com",
    description="基于双 ArUco 鱼眼三维位姿发布无量纲夹爪归一化距离。",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "gripper_openness_node = "
            "fastumi_gripper_estimator.gripper_openness_node:main",
        ],
    },
)
