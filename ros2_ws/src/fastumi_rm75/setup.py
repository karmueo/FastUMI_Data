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
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FastUMI Maintainer",
    maintainer_email="maintainer@example.com",
    description="FastUMI RM75 笛卡尔透传与平行夹爪安全适配。",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rm75_policy_bridge = "
            "fastumi_rm75.rm75_policy_bridge:main",
            "gripper_bridge = fastumi_rm75.gripper_bridge_node:main",
        ],
    },
)
