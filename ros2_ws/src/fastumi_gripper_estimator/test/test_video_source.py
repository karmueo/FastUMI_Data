"""验证预测节点动态订阅、旧回调隔离、状态重置和失败保留输入。"""

import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image

from fastumi_gripper_estimator.gripper_openness_node import GripperOpennessNode
from test_shared_gripper_calibration import _write_calibration_files


def test_switch_resets_prediction_and_rejects_old_callbacks(tmp_path, monkeypatch):
    """新源应用后清除平滑值，旧订阅代次和失败切换不污染预测。"""
    camera, calibration = _write_calibration_files(tmp_path)
    rclpy.init(args=["--ros-args", "-p", f"camera_calibration_path:={camera}",
                    "-p", f"gripper_calibration_path:={calibration}"])
    node = GripperOpennessNode()
    try:
        received = []
        invalid_states = []
        monkeypatch.setattr(node, "_image_callback", received.append)
        monkeypatch.setattr(node._state_publisher, "publish", invalid_states.append)
        node._estimator._smoothed_openness = 0.9
        old = node._image_generation
        assert node.set_parameters_atomically([Parameter("image_topic", value="/new/image")]).successful
        assert node._estimator._smoothed_openness is None
        assert len(invalid_states) == 1 and not invalid_states[0].valid
        node._source_image(Image(), old)
        assert not received
        node._source_image(Image(), old + 1)
        assert len(received) == 1
        subscription = node._image_subscription

        def fail(*_args, **_kwargs):
            """模拟 ROS 无法创建新订阅。"""
            raise RuntimeError("模拟失败")

        monkeypatch.setattr(node, "create_subscription", fail)
        assert not node.set_parameters_atomically([Parameter("image_topic", value="/failed")]).successful
        assert node.get_parameter("image_topic").value == "/new/image"
        assert node._image_subscription is subscription
    finally:
        node.destroy_node()
        rclpy.shutdown()
