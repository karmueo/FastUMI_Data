"""定义 FastUMI 相机内参标定 ROS 2 Python 包的安装规则。"""

from setuptools import find_packages, setup


# ROS 2 包名，用于共享目录和入口点安装。
PACKAGE_NAME = "fastumi_camera_calibration"


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
        (
            f"share/{PACKAGE_NAME}/vendor",
            ["vendor/kalibr_ros2.repos"],
        ),
        (
            f"share/{PACKAGE_NAME}/vendor/patches",
            ["vendor/patches/kalibr_ros2-jazzy.patch"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FastUMI Maintainer",
    maintainer_email="maintainer@example.com",
    description="从 FastUMI MCAP 数据生成单鱼眼相机 Kalibr 内参。",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            (
                "calibrate_camera_intrinsics = "
                "fastumi_camera_calibration.cli:main"
            ),
        ],
    },
)
