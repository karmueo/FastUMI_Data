"""双 ArUco 相机→TCP 标定节点：按键安全逻辑与无硬件仿真全流程。"""

import csv
import math
from pathlib import Path
import time
from unittest.mock import Mock

import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Float32
import cv2
import yaml

from fastumi_bringup.kinematics import JOINT_NAMES, ArmKinematics
from fastumi_bringup.transforms import invert, rotation_angle
from hand_eye_calibration_macpdata.board import load_fisheye_camera
from hand_eye_calibration_macpdata.dual_tag import load_matrix_yaml, transform_difference
from hand_eye_calibration_macpdata.sampler import SamplerSettings, check_reachable
from hand_eye_calibration_macpdata.synthetic import render_dual_tags
from hand_eye_calibration_macpdata.tag_tcp_capture import TagTcpCapture, random_flange_target
from test_dual_tag import gripper_dtag_pose


SRC = Path(__file__).resolve().parents[2]
REPO = SRC.parents[1]
URDF = SRC / "fastumi_rm75/assets/rm_75_kinematic.urdf"
CAMERA = SRC / "fastumi_usb_camera/config/EndEffectorCalib.yaml"
CALIBRATION = REPO / "config/calibration"
START_JOINTS = tuple(np.radians([0.0, 20.0, 0.0, 70.0, 0.0, 90.0, 90.0]))


class TagRig:
    """假机械臂：MoveJ 立即到位；相机与 tag 固连，图像恒为同一张合成图。"""

    def __init__(self):
        self.joints = np.array(START_JOINTS, dtype=float)
        self.truth = gripper_dtag_pose()
        image = render_dual_tags(load_fisheye_camera(CAMERA), self.truth)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        self.message = CompressedImage(format="jpeg", data=encoded.tobytes())
        self.pending_result = None
        self.moves = 0

    def on_movej(self, command):
        self.joints = np.asarray(command.joint, dtype=float)
        self.pending_result = True
        self.moves += 1


def make_node(directory, **overrides):
    parameters = {
        "T_d_tag_tcp_path": str(CALIBRATION / "T_d_tag_tcp.yaml"),
        "T_cam_end_path": str(CALIBRATION / "T_cam_end.yaml"),
        "T_end_tcp_path": str(CALIBRATION / "T_end_tcp.yaml"),
        "camera_config": str(CAMERA), "urdf_path": str(URDF),
        "dataset_root": str(Path(directory) / "dataset"), "num_poses": 3, "frames_per_pose": 3,
        "settle_seconds": 0.05, "random_seed": 2,
    }
    parameters.update(overrides)
    node = TagTcpCapture(parameter_overrides=[
        Parameter(name, value=value) for name, value in parameters.items()])
    rig = TagRig()
    node.movej_publisher = Mock()
    node.movej_publisher.get_subscription_count.return_value = 1
    node.movej_publisher.publish.side_effect = rig.on_movej
    node.stop_publisher = Mock()
    node.gripper_publisher = Mock()
    return node, rig


def feed(node, rig, result=True):
    node._joint_state_callback(JointState(name=list(JOINT_NAMES),
                                          position=[float(v) for v in rig.joints]))
    node._feedback_valid_callback(Bool(data=True))
    node._gripper_callback(Float32(data=1.0))
    node._image_callback(rig.message)
    if rig.pending_result:
        rig.pending_result = False
        node._result_callback(Bool(data=result))


def run_until(node, rig, predicate, timeout=60.0, result=True):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        feed(node, rig, result)
        node.tick()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def warm_up(node, rig, seconds=0.4):
    started = time.monotonic()
    assert run_until(node, rig, lambda: time.monotonic() - started > seconds)


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node_rig(ros_context, tmp_path):
    node, rig = make_node(tmp_path)
    yield node, rig
    node.destroy_node()


def test_requires_dtag_tcp_path(ros_context, tmp_path):
    with pytest.raises(ValueError, match="T_d_tag_tcp_path"):
        make_node(tmp_path, T_d_tag_tcp_path="")


def test_start_requires_feedback(node_rig):
    node, _ = node_rig
    node.handle_key("a")
    node.handle_key("h")
    assert node.phase == "idle" and node.home is None
    node.movej_publisher.publish.assert_not_called()


def test_full_simulated_session_writes_result(node_rig):
    node, rig = node_rig
    warm_up(node, rig)
    node.handle_key("a")
    assert node.phase == "running" and node.pose_capture is not None
    assert run_until(node, rig, lambda: node.phase in ("done", "locked", "paused"))
    assert node.phase == "done", node.phase
    # 起始位姿采样不运动；其余 num_poses-1 个随机位姿 + 1 次回起始位姿。
    assert rig.moves == 3
    assert np.allclose(rig.joints, START_JOINTS)
    result = yaml.safe_load((node.session_dir / "result.yaml").read_text(encoding="utf-8"))
    assert result["samples"] == 3
    dtag_from_tcp = load_matrix_yaml(CALIBRATION / "T_d_tag_tcp.yaml", "T_d_tag_tcp")
    expected = rig.truth @ dtag_from_tcp
    translation_mm, angle_deg = transform_difference(expected, np.array(result["A_d"]))
    assert np.linalg.norm(translation_mm) < 3.0 and angle_deg < 2.0
    camera_from_flange = load_matrix_yaml(CALIBRATION / "T_cam_end.yaml", "cam0.T_cam_hand")
    assert np.allclose(result["T_link7_tcp_measured"],
                       invert(camera_from_flange) @ np.array(result["A_d"]))
    assert "vs_nominal_T_end_tcp" in result
    with open(node.session_dir / "samples.csv", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 3
    assert len(list((node.session_dir / "images").glob("*.jpg"))) == 3
    assert len(list((node.session_dir / "debug").glob("*.jpg"))) == 3


def test_space_stops_motion_and_pauses(node_rig):
    node, rig = node_rig
    warm_up(node, rig)
    node.handle_key("a")
    assert run_until(node, rig, lambda: node.motion is not None, timeout=30.0)
    rig.pending_result = None  # 模拟运动尚未完成
    node.handle_key(" ")
    node.stop_publisher.publish.assert_called()
    assert node.phase == "paused" and node.pose_capture is None


@pytest.mark.parametrize("num_poses", [1, 3])
def test_quit_between_poses_does_not_start_another_move(node_rig, num_poses):
    """采样完成后按 q，即使主循环再调用 tick，也不再随机走位或回起始位姿。"""
    node, rig = node_rig
    node.num_poses = num_poses
    warm_up(node, rig)
    node.handle_key("a")
    assert run_until(node, rig, lambda: len(node.samples) == 1, timeout=5.0)
    assert node.motion is None and node.pose_capture is None

    node.handle_key("q")
    node.tick()  # 与终端主循环相同：处理按键后仍调用一次 tick。

    assert node.finished and node.motion is None
    node.movej_publisher.publish.assert_not_called()


def test_quit_during_motion_sends_stop(node_rig):
    """运动中退出仍发送停止命令，后续 tick 不再发布运动。"""
    node, rig = node_rig
    warm_up(node, rig)
    node.handle_key("a")
    assert run_until(node, rig, lambda: node.motion is not None, timeout=30.0)
    moves = rig.moves

    node.handle_key("q")
    node.tick()

    assert node.finished and rig.moves == moves
    node.stop_publisher.publish.assert_called_once()


def test_movej_failure_locks(node_rig):
    node, rig = node_rig
    warm_up(node, rig)
    node.handle_key("a")
    assert run_until(node, rig, lambda: node.phase == "locked", result=False)
    moves = rig.moves
    node.handle_key("a")
    run_until(node, rig, lambda: False, timeout=0.2)
    assert rig.moves == moves


def test_motion_timeout_locks(ros_context, tmp_path):
    node, rig = make_node(tmp_path, motion_timeout=0.2)
    try:
        warm_up(node, rig)
        node.handle_key("a")
        assert run_until(node, rig, lambda: node.motion is not None, timeout=30.0)
        rig.pending_result = None
        node._result_callback = lambda message: None  # 驱动不回报结果
        assert run_until(node, rig, lambda: node.phase == "locked", timeout=5.0)
        node.stop_publisher.publish.assert_called()
    finally:
        node.destroy_node()


def test_dry_run_does_not_move(ros_context, tmp_path):
    node, rig = make_node(tmp_path, dry_run=True)
    try:
        warm_up(node, rig)
        node.handle_key("a")
        assert run_until(node, rig, lambda: node.phase == "done", timeout=30.0)
        node.movej_publisher.publish.assert_not_called()
        node.close()
        assert not (node.session_dir / "result.yaml").exists()
    finally:
        node.destroy_node()


def test_random_targets_stay_within_limits():
    kinematics = ArmKinematics.from_urdf(URDF)
    home = kinematics.forward(START_JOINTS)
    settings = SamplerSettings(max_joint_step_rad=0.5)
    rng = np.random.default_rng(0)
    reachable = 0
    for _ in range(40):
        target = random_flange_target(home, rng, 0.04, 25.0)
        assert np.all(np.abs(target[:3, 3] - home[:3, 3]) <= 0.04 + 1e-12)
        assert math.degrees(rotation_angle(home[:3, :3], target[:3, :3])) <= 25.0 + 1e-9
        joints, _ = check_reachable(kinematics, target, np.array(START_JOINTS), settings)
        if joints is not None:
            reachable += 1
            assert np.max(np.abs(joints - np.array(START_JOINTS))) <= 0.5
            assert kinematics.within_limits(joints, 0.1)
    assert reachable > 10
