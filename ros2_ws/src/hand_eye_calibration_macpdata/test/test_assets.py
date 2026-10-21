"""验证采集节点的默认 URDF 只依赖 fastumi_bringup 的 share 目录。"""

from pathlib import Path

import rclpy
from rclpy.parameter import Parameter

from hand_eye_calibration_macpdata import capture


SRC = Path(__file__).resolve().parents[2]
REPO = SRC.parents[1]
BRINGUP = SRC / "fastumi_bringup"


def test_default_urdf_only_needs_bringup_share(monkeypatch, tmp_path):
    """模拟没有 fastumi_rm75 的安装环境，采集节点仍能加载 bringup 自带的 URDF。"""
    def package_share(name):
        if name != "fastumi_bringup":
            raise LookupError(f"未安装 {name}")
        return str(BRINGUP)

    monkeypatch.setattr(capture, "get_package_share_directory", package_share)
    rclpy.init()
    node = None
    try:
        node = capture.HandEyeCapture(parameter_overrides=[
            Parameter("board_config", value=str(REPO / "config/calibration/april_6x6.yaml")),
            Parameter("camera_config", value=str(
                SRC / "fastumi_usb_camera/config/EndEffectorCalib.yaml")),
            Parameter("dataset_root", value=str(tmp_path)),
        ])
        assert len(node.kinematics.joints) == 7
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()
