"""验证实机控制入口拒绝缺少起始状态参数的启动。"""

import pytest
import rclpy

from fastumi_rm75.rm75_placo_controller import Rm75PlacoController


def test_real_controller_requires_explicit_start_parameters():
    """不加载现场配置时，节点不得进入可发布 CANFD 指令的实机模式。"""
    rclpy.init(args=["--ros-args", "-p", "dry_run:=false"])
    try:
        with pytest.raises(ValueError, match="explicit start-state parameters"):
            Rm75PlacoController()
    finally:
        rclpy.shutdown()
