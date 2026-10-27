"""安装 UMI 相机到 TCP 离线标定包及命令入口。"""

from setuptools import find_packages, setup


setup(
    name="fastumi_umi_tcp_calibration",
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/fastumi_umi_tcp_calibration"]),
        ("share/fastumi_umi_tcp_calibration", ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FastUMI Maintainer",
    maintainer_email="maintainer@example.com",
    description="从 MCAP 双 ArUco 图像标定 UMI 相机到 TCP 的变换。",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "calibrate_umi_tcp = fastumi_umi_tcp_calibration.cli:main",
        ]
    },
)
