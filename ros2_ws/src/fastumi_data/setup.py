"""定义 fastumi_data ROS 2 Python 包的安装规则。"""

from glob import glob

from setuptools import find_packages, setup


# ROS 2 包名，用于共享目录和入口点安装。
PACKAGE_NAME = "fastumi_data"


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
    description="FastUMI MCAP 会话管理、标定、同步和 HDF5 数据生成工具。",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "episode_manager = fastumi_data.episode_manager:main",
            "episode_command = fastumi_data.episode_command:main",
            "record_session = fastumi_data.session_recorder:main",
            "convert_mcap = fastumi_data.mcap_converter:main",
            "calibrate_tracker_tcp = fastumi_data.calibration_cli:main",
            "calibrate_tracker_camera = fastumi_data.tracker_camera_cli:main",
            "calibrate_aruco_tcp = fastumi_data.aruco_tcp_cli:main",
        ],
    },
)
