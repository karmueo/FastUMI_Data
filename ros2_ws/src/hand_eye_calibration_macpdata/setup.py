"""安装 RM75 眼在手上手眼标定 MCAP 采集与导出求解工具。"""

from glob import glob

from setuptools import setup


PACKAGE_NAME = "hand_eye_calibration_macpdata"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=[PACKAGE_NAME],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml", "README.md"]),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    maintainer="FastUMI Maintainer",
    maintainer_email="maintainer@example.com",
    description="FastUMI RM75 eye-in-hand calibration MCAP capture and solver.",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "keyboard_hand_eye_capture = hand_eye_calibration_macpdata.capture:main",
        "export_hand_eye_dataset = hand_eye_calibration_macpdata.export_dataset:main",
    ]},
)
