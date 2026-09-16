"""定义 USB 单目相机 ROS 2 Python 包的安装规则。"""

from glob import glob

from setuptools import find_packages, setup


# ROS 2 包名及共享资源安装目录名称。
PACKAGE_NAME = "fastumi_usb_camera"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml", "README.md"]),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy<2", "pupil-labs-uvc==1.0.4"],
    zip_safe=True,
    maintainer="FastUMI Maintainer",
    maintainer_email="maintainer@example.com",
    description="采集 UVC MJPEG 并发布 ROS 2 单目相机图像。",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "usb_camera_node = fastumi_usb_camera.usb_camera_node:main",
        ],
    },
)
