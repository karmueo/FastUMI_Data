"""安装独立 DP 推理 ROS 2 包、模型运行时及 launch 资源。"""

from glob import glob
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.install_scripts import install_scripts


PACKAGE_NAME = "dp_infer"  # ROS 包及 ament 资源索引名。


class RosInstallScripts(install_scripts):
    """将可执行入口安装到 ROS 2 约定的包级 lib 目录。"""

    def finalize_options(self):
        """在 colcon 安装前缀内固定入口位置与构建解释器。"""
        super().finalize_options()
        install_command = self.get_finalized_command("install")  # 当前 colcon 安装命令。
        self.install_dir = str(Path(install_command.install_base) / "lib" / PACKAGE_NAME)


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml", "README.md"]),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE_NAME}/runtime", ["runtime/pyproject.toml", "runtime/uv.lock"]),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="FastUMI Maintainer",
    maintainer_email="maintainer@example.com",
    description="订阅 FastUMI 观测并发布 RM75 Link7 扩散策略推荐序列。",
    license="Apache-2.0",
    tests_require=["pytest"],
    cmdclass={"install_scripts": RosInstallScripts},
    entry_points={"console_scripts": ["dp_infer_node = dp_infer.node:main"]},
)
