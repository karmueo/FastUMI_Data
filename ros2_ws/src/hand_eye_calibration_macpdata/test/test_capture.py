"""手眼采集节点：按键安全逻辑与无硬件仿真全流程。"""

import json
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import rclpy
from rclpy.serialization import deserialize_message
import rosbag2_py
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String

from hand_eye_calibration_macpdata import capture as capture_module
from hand_eye_calibration_macpdata.capture import (
    REFERENCE_POSE_NAME, load_reference_pose, main, reference_pose_yaml,
)
from fastumi_bringup.initial_pose import INITIAL_JOINT_POSITIONS
from hand_eye_sim import (
    CALIBRATION_JOINTS, feed, make_node, run_session, run_until, warm_up,
)


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node_rig(ros_context, tmp_path):
    node, rig = make_node(tmp_path)
    yield node, rig
    node.close()
    node.destroy_node()


def test_keys_require_reference_and_feedback(node_rig):
    node, rig = node_rig
    node.handle_key("a")
    node.handle_key("h")
    node.handle_key("c")
    assert node.phase == "idle"
    node.movej_publisher.publish.assert_not_called()
    node.handle_key("r")  # 无反馈：不能设参考位姿
    assert node.reference is None


def test_space_stops_motion_and_pauses(node_rig):
    node, rig = node_rig
    warm_up(node, rig)
    node.handle_key("r")
    node.handle_key("a")
    node.tick()
    assert node.motion is not None
    rig.pending_result = False  # 驱动尚未回报
    node.handle_key(" ")
    node.stop_publisher.publish.assert_called_once()
    assert node.phase == "paused" and node.motion.purpose == "stopped"
    node._result_callback(Bool(data=False))
    assert node.phase == "paused"
    published = node.movej_publisher.publish.call_count
    node.tick()
    assert node.movej_publisher.publish.call_count == published
    node.handle_key("a")
    assert node.phase == "bootstrap"


def test_movej_failure_locks(node_rig):
    node, rig = node_rig
    warm_up(node, rig)
    node.handle_key("r")
    node.handle_key("a")
    node.tick()
    rig.pending_result = False
    node._result_callback(Bool(data=False))
    assert node.phase == "locked"
    node.handle_key("a")
    node.handle_key("h")
    assert node.phase == "locked"


def test_motion_timeout_locks(node_rig):
    node, rig = node_rig
    warm_up(node, rig)
    node.handle_key("r")
    node.handle_key("a")
    node.tick()
    node.motion.sent_at -= node.motion_timeout + 1.0
    node.tick()
    assert node.phase == "locked" and node.motion is None
    node.stop_publisher.publish.assert_called_once()
    published = node.movej_publisher.publish.call_count
    node.tick()
    node.handle_key("a")
    node.handle_key("h")
    node.handle_key("q")
    assert node.finished
    assert node.movej_publisher.publish.call_count == published
    node.stop_publisher.publish.assert_called_once()


def test_stale_feedback_blocks_motion(node_rig):
    node, rig = node_rig
    warm_up(node, rig)
    node.handle_key("r")
    node.handle_key("a")
    node.last_valid_feedback_at = None
    node.tick()
    node.movej_publisher.publish.assert_not_called()


@pytest.mark.parametrize("board, target_type, pattern", [
    ("aprilgrid", "aprilgrid", None), ("aprilgrid_24mm", "aprilgrid", None),
    ("checkerboard", "checkerboard", "2 8 11 0.03")])
def test_full_simulated_session_writes_samples(tmp_path, board, target_type, pattern):
    session_dir, rig, samples, mask = run_session(tmp_path, board, num_samples=8)
    assert len(samples) == 8 and mask is not None
    assert (session_dir / "gripper_mask.png").is_file()
    assert json.loads((session_dir / "summary.json").read_text())["samples"]

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(session_dir / "bag"), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    counts, stamps = {}, {}
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        counts[topic] = counts.get(topic, 0) + 1
        if topic == "/hand_eye/sample_meta":
            stamps.setdefault("meta", set()).add(stamp)
        elif topic == "/hand_eye/sample_image":
            stamps.setdefault("image", set()).add(stamp)
            assert deserialize_message(data, CompressedImage).format == "jpeg"
        elif topic == "/hand_eye/session":
            session = json.loads(deserialize_message(data, String).data)
    assert counts["/hand_eye/sample_meta"] == counts["/hand_eye/sample_flange_pose"] == 8
    assert stamps["meta"] == stamps["image"]
    assert counts["/joint_states"] > 0 and counts["/hand_eye/gripper_mask"] == 1
    assert session["hand_eye_pattern"] == pattern
    assert session["board"]["target_type"] == target_type


def test_undo_writes_rejection(node_rig):
    node, rig = node_rig
    warm_up(node, rig)
    node.handle_key("r")
    node.capture_purpose = "manual"
    assert run_until(node, rig, lambda: node.samples, timeout=5.0)
    node.writer.write = Mock()
    node.handle_key("u")
    assert not node.samples
    topic, payload, _ = node.writer.write.call_args.args
    assert topic == "/hand_eye/sample_rejected"
    assert json.loads(deserialize_message(payload, String).data) == {"index": 0}


def test_main_rejects_noninteractive_input(monkeypatch, capsys):
    stream = Mock()
    stream.isatty.return_value = False
    monkeypatch.setattr(sys, "stdin", stream)
    assert main() == 2
    assert "交互终端" in capsys.readouterr().err


def test_feed_helper_reports_results(node_rig):
    node, rig = node_rig
    rig.pending_result = True
    feed(node, rig)
    assert not rig.pending_result


def test_reference_pose_file_round_trip(node_rig):
    """按 r 后写出 reference_pose.yaml，读回与参考关节一致（度保留 3 位小数）。"""
    node, rig = node_rig
    warm_up(node, rig)
    node.handle_key("r")
    path = node.session_dir / REFERENCE_POSE_NAME
    assert path.is_file()
    assert load_reference_pose(path) == pytest.approx(node.reference["joints"], abs=1e-4)


def test_load_reference_pose_rejects_bad_files(tmp_path):
    path = tmp_path / "pose.yaml"
    for text in ("reference_joints_deg: [1, 2, 3]\n", "other: 1\n",
                 "reference_joints_deg: [1, 2, 3, 4, 5, 6, .nan]\n"):
        path.write_text(text)
        with pytest.raises(ValueError):
            load_reference_pose(path)
    path.write_text(reference_pose_yaml(np.radians([1, 2, 3, 4, 5, 6, 7]), "测试"))
    assert np.degrees(load_reference_pose(path)) == pytest.approx([1, 2, 3, 4, 5, 6, 7])


def test_reference_pose_outside_limits_is_rejected(ros_context, tmp_path):
    path = tmp_path / "pose.yaml"
    path.write_text("reference_joints_deg: [0, 0, 0, 0, 0, 0, 400]\n")
    with pytest.raises(ValueError, match="限位"):
        make_node(tmp_path, reference_pose_path=str(path))


def test_goto_reference_requires_path(node_rig):
    node, rig = node_rig
    warm_up(node, rig)
    node.handle_key("g")
    node.handle_key("g")
    node.movej_publisher.publish.assert_not_called()


@pytest.fixture
def goto_rig(ros_context, tmp_path):
    """参考位姿文件为标定位姿；机械臂从 bringup 数采初始位姿出发（看不到板）。"""
    path = tmp_path / "pose.yaml"
    path.write_text(reference_pose_yaml(CALIBRATION_JOINTS, "测试"))
    node, rig = make_node(tmp_path, reference_pose_path=str(path))
    rig.joints = np.array(INITIAL_JOINT_POSITIONS, dtype=float)
    yield node, rig, load_reference_pose(path)
    node.close()
    node.destroy_node()


def test_goto_reference_needs_confirmation_then_sets_reference(goto_rig):
    node, rig, target = goto_rig
    warm_up(node, rig)
    node.handle_key("g")
    node.movej_publisher.publish.assert_not_called()
    node.handle_key("g")
    node.movej_publisher.publish.assert_called_once()
    command = node.movej_publisher.publish.call_args.args[0]
    assert list(command.joint) == pytest.approx(target, abs=1e-6)  # Movej 为 float32
    assert node.motion.purpose == "goto_reference"
    assert run_until(node, rig, lambda: node.reference is not None, timeout=10.0)
    assert node.reference["joints"] == pytest.approx(target, abs=1e-6)
    assert node.phase == "idle" and node.movej_publisher.publish.call_count == 1


def test_goto_reference_confirmation_expires(goto_rig):
    node, rig, _ = goto_rig
    warm_up(node, rig)
    node.handle_key("g")
    node.goto_confirm_at -= capture_module.GOTO_CONFIRM_SECONDS + 0.1
    node.handle_key("g")  # 超时：重新进入确认
    node.movej_publisher.publish.assert_not_called()
    node.handle_key("g")
    node.movej_publisher.publish.assert_called_once()


@pytest.mark.parametrize("dry_run", [False, True])
def test_space_cancels_goto_reference_confirmation(goto_rig, dry_run):
    """g → 空格取消确认；之后必须重新按两次 g 才能下发参考位姿运动。"""
    node, rig, _ = goto_rig
    warm_up(node, rig)
    node.dry_run = dry_run
    node.handle_key("g")
    assert node.goto_confirm_at is not None
    node.handle_key(" ")
    assert node.goto_confirm_at is None and node.phase == "paused"

    node.handle_key("g")
    node.movej_publisher.publish.assert_not_called()
    assert node.goto_confirm_at is not None and node.motion is None
    node.handle_key("g")
    assert node.goto_confirm_at is None
    if dry_run:
        node.movej_publisher.publish.assert_not_called()
        node.stop_publisher.publish.assert_not_called()
    else:
        node.movej_publisher.publish.assert_called_once()
        node.stop_publisher.publish.assert_called_once()
        assert node.motion.purpose == "goto_reference"


def test_space_stops_goto_reference(goto_rig):
    node, rig, _ = goto_rig
    warm_up(node, rig)
    node.handle_key("g")
    node.handle_key("g")
    rig.pending_result = False  # 驱动尚未回报
    node.handle_key(" ")
    node.stop_publisher.publish.assert_called_once()
    assert node.phase == "paused" and node.motion.purpose == "stopped"
    node._result_callback(Bool(data=False))
    assert node.phase == "paused" and not node.pending_reference
