"""安装 Jetson 本地硬件统一启动包。"""

from glob import glob

from setuptools import setup


PACKAGE_NAME = "fastumi_bringup"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml", "README.md"]),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.example")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    maintainer="FastUMI Maintainer",
    maintainer_email="maintainer@example.com",
    description="FastUMI Jetson local hardware bringup.",
    license="Apache-2.0",
)
