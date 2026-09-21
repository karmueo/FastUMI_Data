"""安装独立录制服务与启动入口。"""
from glob import glob
from setuptools import setup

setup(
    name='fastumi_recorder', version='0.1.0', packages=['fastumi_recorder'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/fastumi_recorder']),
        ('share/fastumi_recorder', ['package.xml', 'README.md']),
        ('share/fastumi_recorder/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'], extras_require={'test': ['pytest']},
    zip_safe=False,
    maintainer='FastUMI Maintainer', maintainer_email='maintainer@example.com',
    description='Jetson hardware recording services', license='Apache-2.0',
    entry_points={'console_scripts': ['recorder = fastumi_recorder.node:main']},
)
