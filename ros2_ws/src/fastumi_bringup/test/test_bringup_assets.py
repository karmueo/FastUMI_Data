"""验证基础 bringup 独立提供默认运动学资源，不需要安装策略部署包。"""

from pathlib import Path
import xml.etree.ElementTree as ElementTree

import pytest
import rclpy
from rclpy.parameter import Parameter

from fastumi_bringup import arm_control_service, keyboard_hand_eye_capture


PACKAGE = Path(__file__).resolve().parents[1]
SRC = PACKAGE.parent
REPO = SRC.parents[1]


def test_bundled_urdf_matches_deployment_chain():
    """两份模型的链、关节轴、限位及几何必须一致，防止控制与标定漂移。"""
    bundled = ElementTree.parse(PACKAGE / "assets/rm_75_kinematic.urdf").getroot()
    deployment = ElementTree.parse(SRC / "fastumi_rm75/assets/rm_75_kinematic.urdf").getroot()
    assert ElementTree.tostring(bundled) == ElementTree.tostring(deployment)


@pytest.mark.parametrize("node_kind", ["arm", "capture"])
def test_default_urdf_only_needs_bringup_share(monkeypatch, tmp_path, node_kind):
    """模拟没有 fastumi_rm75 的安装环境，两个节点仍能加载默认 URDF。"""
    def package_share(name):
        if name != "fastumi_bringup":
            raise LookupError(f"未安装 {name}")
        return str(PACKAGE)

    monkeypatch.setattr(arm_control_service, "get_package_share_directory", package_share)
    monkeypatch.setattr(keyboard_hand_eye_capture, "get_package_share_directory", package_share)
    rclpy.init()
    node = None
    try:
        if node_kind == "arm":
            node = arm_control_service.ArmControlService()
        else:
            node = keyboard_hand_eye_capture.HandEyeCapture(parameter_overrides=[
                Parameter("board_config", value=str(REPO / "config/calibration/april_6x6.yaml")),
                Parameter("camera_config", value=str(
                    SRC / "fastumi_usb_camera/config/EndEffectorCalib.yaml")),
                Parameter("dataset_root", value=str(tmp_path)),
            ])
        assert len(node.kinematics.joints) == 7
    finally:
        if node is not None:
            if node_kind == "capture":
                node.close()
            node.destroy_node()
        rclpy.shutdown()
