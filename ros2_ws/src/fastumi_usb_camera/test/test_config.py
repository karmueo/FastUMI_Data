"""验证默认参数文件能应用于带命名空间的相机节点。"""

from pathlib import Path

import rclpy
from rclpy.node import Node


def test_default_config_matches_namespaced_node():
    """通过 ROS 参数解析器加载配置，防止命名空间使 YAML 失效。"""
    # 从测试目录定位随包提供的默认参数文件。
    config = Path(__file__).resolve().parents[1] / "config" / "usb_camera.yaml"
    rclpy.init(args=[
        "--ros-args", "-r", "__ns:=/usb_camera", "--params-file", str(config)
    ])
    node = Node(
        "usb_camera_node", automatically_declare_parameters_from_overrides=True
    )
    try:
        assert node.get_namespace() == "/usb_camera"
        assert node.get_parameter("vendor_id").value == 0x1BCF
        assert node.get_parameter("product_id").value == 0x28C4
        assert node.get_parameter("width").value == 1280
        assert node.get_parameter("publish_compressed").value is False
    finally:
        node.destroy_node()
        rclpy.shutdown()
