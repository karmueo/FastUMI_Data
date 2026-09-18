"""定义 tracker_teleoperated ROS 2 Python 包的安装规则。"""

from glob import glob
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.install_scripts import install_scripts


# ROS 2 包名，用于安装节点入口和共享资源。
PACKAGE_NAME = "tracker_teleoperated"


class RosInstallScripts(install_scripts):
    """将控制节点入口安装到 ROS 2 约定的包级 lib 目录。"""

    def finalize_options(self):
        """解析安装前缀，并覆盖虚拟环境默认使用的 bin 目录。"""

        super().finalize_options()
        # colcon 已在 install 命令中解析出当前包的独立安装根目录。
        install_command = self.get_finalized_command("install")
        self.install_dir = str(
            Path(install_command.install_base) / "lib" / PACKAGE_NAME
        )


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
    install_requires=[
        "setuptools",
        "numpy>=1.26",
        "PyYAML>=6.0",
        "scipy>=1.14",
        "placo==0.9.23",
    ],
    zip_safe=True,
    maintainer="FastUMI Maintainer",
    maintainer_email="maintainer@example.com",
    description="基于 VIVE Tracker 和夹爪视觉预测的 RM75 与 Unitree 遥操节点。",
    license="Apache-2.0",
    tests_require=["pytest"],
    cmdclass={"install_scripts": RosInstallScripts},
    entry_points={
        "console_scripts": [
            "tracker_teleop_node = tracker_teleoperated.node:main",
            "tracker_teleop_keyboard = tracker_teleoperated.keyboard:main",
            "tracker_teleop_wait_ready = tracker_teleoperated.bringup_checks:main",
        ],
    },
)
