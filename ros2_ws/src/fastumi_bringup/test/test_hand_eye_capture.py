"""手眼采集节点：按键安全逻辑与无硬件仿真全流程。"""

import json
import sys
from unittest.mock import Mock

import pytest
import rclpy
from rclpy.serialization import deserialize_message
import rosbag2_py
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String

from fastumi_bringup.keyboard_hand_eye_capture import main
from hand_eye_sim import feed, make_node, run_session, run_until, warm_up


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


@pytest.mark.parametrize("board, pattern", [
    ("aprilgrid", None), ("checkerboard", "2 8 11 0.03")])
def test_full_simulated_session_writes_samples(tmp_path, board, pattern):
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
    assert session["board"]["target_type"] == board


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
