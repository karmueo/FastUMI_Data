"""手眼采集节点的无硬件仿真：假 RM75 + 鱼眼合成图 + 固定夹爪遮挡。

供 test_capture.py 和 test_export_dataset.py 共用。
真值：^Link7 T_cam 取仓库 T_cam_end.yaml，板放在 CALIBRATION_JOINTS 位姿相机光轴前方 0.33 m。
板型取仓库 config/calibration 下的 Kalibr YAML，通过 run_session(board=...) 选择。
"""

from pathlib import Path
import time
from unittest.mock import Mock

import cv2
import numpy as np
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Float32
import yaml

from fastumi_bringup.kinematics import JOINT_NAMES, ArmKinematics
from fastumi_bringup.transforms import invert, make_transform
from hand_eye_calibration_macpdata.board import AprilGridSpec, load_board_spec, load_fisheye_camera
from hand_eye_calibration_macpdata.capture import HandEyeCapture
from hand_eye_calibration_macpdata.estimate import load_prior
from hand_eye_calibration_macpdata.synthetic import gripper_occluder, render_board


SRC = Path(__file__).resolve().parents[2]
REPO = SRC.parents[1]
URDF = SRC / "fastumi_rm75/assets/rm_75_kinematic.urdf"
CAMERA = SRC / "fastumi_usb_camera/config/EndEffectorCalib.yaml"
PRIOR = REPO / "config/calibration/T_cam_end.yaml"
BOARDS = {
    "aprilgrid": REPO / "config/calibration/april_6x6.yaml",
    "aprilgrid_24mm": REPO / "config/calibration/april_6x6_24mm.yaml",
    "checkerboard": REPO / "config/calibration/checkerboard_11x8.yaml",
}
# 标定参考位姿（弧度，joint1..joint7）：实机在该位姿下能看到整块板。与 fastumi_bringup 的
# 数采初始位姿（initial_pose.yaml）无关，后者变化不影响仿真。
CALIBRATION_JOINTS = tuple(np.radians([0.0, 20.0, 0.0, 70.0, 0.0, 90.0, 90.0]))
# 与实机腕部相机一致的全分辨率：AprilGrid 斜视时 960 宽下小标签解码不稳。
WIDTH, HEIGHT = 1920, 1080


class SimulatedRig:
    """假机械臂：MoveJ 立即到位；按当前关节渲染鱼眼图。"""

    def __init__(self, spec):
        self.spec = spec
        self.kinematics = ArmKinematics.from_urdf(URDF)
        self.camera = load_fisheye_camera(CAMERA).scaled(WIDTH, HEIGHT)
        self.flange_from_camera = load_prior(PRIOR)
        self.joints = np.array(CALIBRATION_JOINTS, dtype=float)
        base_from_camera = self.kinematics.forward(self.joints) @ self.flange_from_camera
        # 板正面朝向参考相机，中心在光轴前方 0.33 m，略偏上避开夹爪。棋盘格（OpenCV 约定）
        # 相机在板 -z 侧，板系与相机系同向；AprilGrid（Kalibr 约定）相机在 +z 侧，绕 x 翻转。
        center = base_from_camera[:3, 3] + 0.33 * base_from_camera[:3, 2] - \
            0.06 * base_from_camera[:3, 1]
        rotation = base_from_camera[:3, :3]
        if isinstance(spec, AprilGridSpec):
            rotation = rotation @ np.diag([1.0, -1.0, -1.0])
        self.base_from_board = make_transform(rotation, center - rotation @ spec.center_point())
        self.occluder = gripper_occluder(WIDTH, HEIGHT)
        self.pending_result = False
        self.moves = 0
        self._cache = None

    def on_movej(self, command):
        """记录 MoveJ 并瞬时到位。"""
        self.joints = np.asarray(command.joint, dtype=float)
        self.pending_result = True
        self.moves += 1

    def image_message(self):
        """当前关节下的 JPEG 图像消息（按关节缓存渲染结果）。"""
        key = tuple(np.round(self.joints, 9))
        if self._cache is None or self._cache[0] != key:
            base_from_camera = self.kinematics.forward(self.joints) @ self.flange_from_camera
            image = render_board(self.spec, self.camera,
                                 invert(base_from_camera) @ self.base_from_board,
                                 occluder=self.occluder, supersample=1)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            assert ok
            self._cache = (key, CompressedImage(format="jpeg", data=encoded.tobytes()))
        return self._cache[1]


def write_configs(directory):
    """写入缩放到仿真分辨率的相机内参，返回其路径。"""
    directory = Path(directory)
    camera = load_fisheye_camera(CAMERA).scaled(WIDTH, HEIGHT)
    camera_path = directory / "camera.yaml"
    camera_path.write_text(yaml.safe_dump({"cam0": {
        "intrinsics": [float(camera.K[0, 0]), float(camera.K[1, 1]),
                       float(camera.K[0, 2]), float(camera.K[1, 2])],
        "distortion_model": "fisheye",
        "distortion_coeffs": [float(v) for v in camera.D.reshape(-1)],
        "resolution": [WIDTH, HEIGHT],
    }}), encoding="utf-8")
    return camera_path


def make_node(directory, board="aprilgrid", **overrides):
    """创建真实节点（需已 rclpy.init），发布器替换为接到 SimulatedRig 的 Mock。"""
    camera = write_configs(directory)
    parameters = {
        "board_config": str(BOARDS[board]), "camera_config": str(camera), "urdf_path": str(URDF),
        "dataset_root": str(Path(directory) / "dataset"), "num_samples": 8,
        "settle_seconds": 0.05, "random_seed": 4, "max_reprojection_px": 1.5,
    }
    parameters.update(overrides)
    node = HandEyeCapture(parameter_overrides=[
        Parameter(name, value=value) for name, value in parameters.items()])
    rig = SimulatedRig(load_board_spec(BOARDS[board]))
    node.movej_publisher = Mock()
    node.movej_publisher.get_subscription_count.return_value = 1
    node.movej_publisher.publish.side_effect = rig.on_movej
    node.stop_publisher = Mock()
    node.gripper_publisher = Mock()
    return node, rig


def feed(node, rig):
    """模拟一次传感器回调：关节、驱动有效、夹爪状态、图像，以及待回报的 MoveJ 结果。"""
    node._joint_state_callback(JointState(name=list(JOINT_NAMES),
                                          position=[float(v) for v in rig.joints]))
    node._feedback_valid_callback(Bool(data=True))
    node._gripper_callback(Float32(data=1.0))
    node._image_callback(rig.image_message())
    if rig.pending_result:
        rig.pending_result = False
        node._result_callback(Bool(data=True))


def run_until(node, rig, predicate, timeout=180.0):
    """以约 100 Hz 驱动节点直到 predicate 成立；超时返回 False。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        feed(node, rig)
        node.tick()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def warm_up(node, rig, seconds=0.5):
    """先完成首帧渲染（全分辨率首帧较慢），再连续喂入 seconds 秒使关节静止判定成立。"""
    assert run_until(node, rig, lambda: node.latest_image is not None, timeout=30.0)
    started = time.monotonic()
    assert run_until(node, rig, lambda: time.monotonic() - started > seconds, timeout=10.0)


def run_session(directory, board="aprilgrid", **overrides):
    """完整跑一轮：r → a → 采满 → q，返回会话目录和仿真器。"""
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    node, rig = make_node(directory, board, **overrides)
    try:
        warm_up(node, rig)
        node.handle_key("r")
        assert node.reference is not None
        node.handle_key("a")
        assert node.phase == "bootstrap"
        assert run_until(node, rig, lambda: node.phase in ("done", "locked", "paused"))
        assert node.phase == "done", node.phase
        node.handle_key("q")
        session_dir = node.session_dir
        samples = list(node.samples)
        mask = node.mask
    finally:
        node.close()
        node.destroy_node()
        if owns_context:
            rclpy.shutdown()
    return session_dir, rig, samples, mask
