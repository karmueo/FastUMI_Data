"""安装 ROS 入口；模型/GPU 依赖由独立 Jetson 环境提供。"""

from glob import glob
from setuptools import find_packages, setup

setup(
    name="dexgraspvla_infer", version="0.1.0", packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/dexgraspvla_infer"]),
        ("share/dexgraspvla_infer", ["package.xml", "README.md", "requirements-orin.txt"]),
        ("share/dexgraspvla_infer/config", glob("config/*.yaml")),
        ("share/dexgraspvla_infer/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"], zip_safe=False,
    maintainer="FastUMI Maintainer", maintainer_email="maintainer@example.com",
    description="RM75 DexGraspVLA inference", license="Apache-2.0",
    entry_points={"console_scripts": [
        "dexgraspvla_infer_node = dexgraspvla_infer.cli:policy_main",
        "perception_node = dexgraspvla_infer.cli:perception_main",
        "combined_node = dexgraspvla_infer.cli:combined_main",
        "prepare_assets = dexgraspvla_infer.prepare_assets:main",
        "offline_check = dexgraspvla_infer.offline:main",
        "benchmark = dexgraspvla_infer.benchmark:main",
    ]},
)
