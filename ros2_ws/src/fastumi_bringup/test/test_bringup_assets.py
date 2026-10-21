"""验证基础 bringup 独立提供默认运动学资源，不需要安装策略部署包。

手眼采集节点的同类检查在 hand_eye_calibration_macpdata/test/test_assets.py。
"""

from pathlib import Path
import xml.etree.ElementTree as ElementTree

import rclpy

from fastumi_bringup import arm_control_service


PACKAGE = Path(__file__).resolve().parents[1]
SRC = PACKAGE.parent


def test_bundled_urdf_matches_deployment_chain():
    """两份模型的链、关节轴、限位及几何必须一致，防止控制与标定漂移。"""
    bundled = ElementTree.parse(PACKAGE / "assets/rm_75_kinematic.urdf").getroot()
    deployment = ElementTree.parse(SRC / "fastumi_rm75/assets/rm_75_kinematic.urdf").getroot()
    assert ElementTree.tostring(bundled) == ElementTree.tostring(deployment)


def test_default_urdf_only_needs_bringup_share(monkeypatch):
    """模拟没有 fastumi_rm75 的安装环境，关节服务仍能加载默认 URDF。"""
    def package_share(name):
        if name != "fastumi_bringup":
            raise LookupError(f"未安装 {name}")
        return str(PACKAGE)

    monkeypatch.setattr(arm_control_service, "get_package_share_directory", package_share)
    rclpy.init()
    node = None
    try:
        node = arm_control_service.ArmControlService()
        assert len(node.kinematics.joints) == 7
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
