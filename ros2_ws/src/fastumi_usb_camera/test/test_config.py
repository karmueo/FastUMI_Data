"""验证默认参数文件能应用于带命名空间的相机节点。"""

from pathlib import Path

import rclpy
from rclpy.node import Node
import yaml


def test_default_config_matches_namespaced_node():
    """通过 ROS 参数解析器加载配置，防止命名空间使 YAML 失效。"""
    # 从测试目录定位随包提供的默认参数文件。
    config = Path(__file__).resolve().parents[1] / "config" / "usb_camera.yaml"
    # 使用当前工作区设备配置，测试聚焦命名空间下的参数装载。
    expected = yaml.safe_load(config.read_text())["/**"]["ros__parameters"]
    rclpy.init(args=[
        "--ros-args", "-r", "__ns:=/usb_camera", "--params-file", str(config)
    ])
    node = Node(
        "usb_camera_node", automatically_declare_parameters_from_overrides=True
    )
    try:
        assert node.get_namespace() == "/usb_camera"
        assert node.get_parameter("video_device").value == expected["video_device"]
        assert not node.has_parameter("vendor_id")
        assert not node.has_parameter("product_id")
        assert not node.has_parameter("serial_number")
        assert node.get_parameter("width").value == 1920
        assert node.get_parameter("publish_compressed").value is False
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_ffmpeg_config_uses_automatic_video_device_selection():
    """FFmpeg 默认配置使用空路径，且不包含已移除的选择参数。"""
    # 从测试目录定位 FFmpeg 默认参数文件。
    config = Path(__file__).resolve().parents[1] / "config" / "ffmpeg.yaml"
    # 发送节点的默认参数字典。
    parameters = yaml.safe_load(config.read_text())["usb_camera_ffmpeg"][
        "ros__parameters"
    ]
    assert parameters["video_device"] == ""
    assert {"vendor_id", "product_id", "serial_number"}.isdisjoint(parameters)
