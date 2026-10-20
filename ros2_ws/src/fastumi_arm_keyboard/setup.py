"""安装 RM75 键盘急停与回位客户端。"""

from setuptools import setup


PACKAGE_NAME = "fastumi_arm_keyboard"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=[PACKAGE_NAME],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    maintainer="FastUMI Maintainer",
    maintainer_email="maintainer@example.com",
    description="FastUMI RM75 keyboard emergency stop and return-home client.",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "keyboard_control = fastumi_arm_keyboard.keyboard_control:main",
    ]},
)
