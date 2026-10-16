"""使用合成 MCAP 验证图像读取、范围标定、失败保护及现有输入模式。"""

from pathlib import Path
import subprocess
import sys

import cv2
from cv_bridge import CvBridge
from gripper_openness.calibration import load_gripper_calibration
from gripper_openness.gripper_calibration_node import GripperCalibrationNode, main
from gripper_openness.mcap_source import McapImageSource, resolve_camera_path, resolve_mcap_path
import numpy as np
import pytest
import rclpy
from rclpy.serialization import serialize_message
import rosbag2_py
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import Trigger
import yaml


def make_frame(gap: int = 180) -> np.ndarray:
    """构造双 ArUco 图像，标签中心间距由 gap 像素指定。"""
    image = np.full((480, 640), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, center_x in ((0, 320 - gap // 2), (1, 320 + gap // 2)):
        image[200:280, center_x - 40:center_x + 40] = cv2.aruco.generateImageMarker(
            dictionary, marker_id, 80
        )
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def stroke(frames_per_end: int = 5) -> list:
    """构造在闭合（180 px）和全开（300 px）两端各停留若干帧的完整行程。"""
    return [make_frame(180)] * frames_per_end + [make_frame(300)] * frames_per_end


def image_message(image: np.ndarray, encoding: str = "raw"):
    """将测试图像封装为原始、JPEG 或 PNG 消息，并设置源时间戳。"""
    if encoding == "raw":
        message = CvBridge().cv2_to_imgmsg(image, encoding="bgr8")
    else:
        message = CompressedImage()
        message.format = encoding
        success, encoded = cv2.imencode("." + encoding, image)
        assert success
        message.data = encoded.tobytes()
    message.header.stamp.sec = 17
    return message


def write_bag(path: Path, messages: list) -> Path:
    """写入合成 MCAP，消息列表元素为话题与 ROS 消息。"""
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topics = {}
    for topic, message in messages:
        if topic not in topics:
            message_type = type(message)
            kind = message_type.__module__.split(".")[0] + "/msg/" + message_type.__name__
            topics[topic] = kind
            writer.create_topic(rosbag2_py.TopicMetadata(
                id=len(topics), name=topic, type=kind, serialization_format="cdr"
            ))
    for index, (topic, message) in enumerate(messages):
        writer.write(topic, serialize_message(message), 1_000_000_000 + index)
    writer.close()
    return path


@pytest.fixture
def camera_path(tmp_path: Path) -> Path:
    """创建使用零畸变针孔模型的测试内参。"""
    path = tmp_path / "camera.yaml"
    path.write_text(yaml.safe_dump({"cam0": {
        "intrinsics": [400.0, 400.0, 320.0, 240.0],
        "distortion_model": "plumb_bob",
        "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
        "resolution": [640, 480],
    }}), encoding="utf-8")
    return path


@pytest.fixture
def ros_context():
    """为节点测试建立并清理独立 ROS 上下文。"""
    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    yield
    if owned and rclpy.ok():
        rclpy.shutdown()


@pytest.mark.parametrize("encoding", ["raw", "jpg", "png"])
@pytest.mark.parametrize("single_file", [False, True])
def test_reads_image_formats_and_paths(tmp_path, encoding, single_file):
    """文件和 bag 目录均能读取标准原始或压缩图像，并保留 Header。"""
    bag = write_bag(tmp_path / "bag", [("/camera", image_message(make_frame(), encoding))])
    source = McapImageSource(str(next(bag.glob("*.mcap"))) if single_file else str(bag))
    try:
        frames = list(source.frames())
        assert len(frames) == 1
        assert frames[0][0].shape == (480, 640, 3)
        assert frames[0][1].stamp.sec == 17
        assert source.message_count == 1
        assert source.decode_failures == 0
    finally:
        source.close()


def test_session_topic_snapshot_and_explicit_precedence(tmp_path, camera_path):
    """录制目录、bag 目录和单文件均找到快照；显式话题和内参覆盖快照。"""
    session = tmp_path / "session"
    (session / "raw").mkdir(parents=True)
    snapshot = session / "calibration_snapshot/calib.yaml"
    snapshot.parent.mkdir()
    snapshot.write_bytes(camera_path.read_bytes())
    (session / "session.yaml").write_text(
        "bag_uri: raw/bag\nimage:\n  topic: /recorded\n", encoding="utf-8"
    )
    bag = write_bag(session / "raw/bag", [
        ("/recorded", image_message(make_frame())),
        ("/override", image_message(make_frame())),
    ])
    for path in (session, bag, next(bag.glob("*.mcap"))):
        source = McapImageSource(str(path))
        try:
            assert source.image_topic == "/recorded"
            assert source.session_dir == session
            assert len(list(source.frames())) == 1
        finally:
            source.close()
    source = McapImageSource(str(session), "/override")
    assert source.image_topic == "/override"
    source.close()
    assert resolve_camera_path("", session, "default") == str(snapshot)
    assert resolve_camera_path(str(camera_path), session, "default") == str(camera_path)
    assert resolve_camera_path("", None, "default") == "default"


def test_session_node_uses_camera_snapshot(tmp_path, camera_path, ros_context):
    """不传相机参数时，节点使用录制快照完成端到端标定。"""
    session = tmp_path / "session"
    (session / "raw").mkdir(parents=True)
    snapshot = session / "calibration_snapshot/calib.yaml"
    snapshot.parent.mkdir()
    snapshot.write_bytes(camera_path.read_bytes())
    (session / "session.yaml").write_text("bag_uri: raw/bag\n", encoding="utf-8")
    write_bag(session / "raw/bag", [("/camera", image_message(frame)) for frame in stroke()])
    output = tmp_path / "result.yaml"
    node = make_node(session, camera_path, output, camera_calibration_path="")
    try:
        assert node._camera_calibration.resolution == (640, 480)
        node.run_mcap()
        assert load_gripper_calibration(str(output)).total_valid_frames == 10
    finally:
        node.destroy_node()


def test_missing_ambiguous_and_unsupported_topics(tmp_path):
    """缺失、多个候选和不支持的消息类型应在读取前明确报错。"""
    bag = write_bag(tmp_path / "bag", [
        ("/a", image_message(make_frame())),
        ("/b", image_message(make_frame())),
        ("/text", String(data="test")),
    ])
    with pytest.raises(ValueError, match="/a, /b"):
        McapImageSource(str(bag))
    with pytest.raises(ValueError, match="缺少图像话题"):
        McapImageSource(str(bag), "/missing")
    with pytest.raises(ValueError, match="不支持"):
        McapImageSource(str(bag), "/text")
    with pytest.raises(ValueError, match="必须设置 mcap_path"):
        resolve_mcap_path("")
    with pytest.raises(ValueError, match="不存在"):
        resolve_mcap_path(str(tmp_path / "missing.mcap"))


def make_node(bag: Path, camera: Path, output: Path, **overrides) -> GripperCalibrationNode:
    """创建向测试目录输出的 MCAP 标定节点。"""
    parameters = {
        "source_mode": "mcap", "mcap_path": str(bag),
        "camera_calibration_path": str(camera), "output_path": str(output),
    }
    parameters.update(overrides)
    return GripperCalibrationNode(parameter_overrides=list(parameters.items()))


def test_calibrates_and_skips_corrupt_image(tmp_path, camera_path, ros_context):
    """坏 JPEG 不影响后续有效帧，生成的标定能回读且保存正确距离范围和统计。"""
    frames = [("/camera", image_message(frame, "jpg")) for frame in stroke()]
    frames.insert(5, ("/camera", CompressedImage(format="jpeg", data=b"bad jpeg")))
    bag = write_bag(tmp_path / "bag", frames)
    source = McapImageSource(str(bag))
    try:
        assert len(list(source.frames())) == 10
        assert source.message_count == 11
        assert source.decode_failures == 1
    finally:
        source.close()
    output = tmp_path / "result.yaml"
    node = make_node(bag, camera_path, output)
    try:
        node.run_mcap()
        result = load_gripper_calibration(str(output))
        assert result.total_valid_frames == 10
        assert 30 < result.min_marker_dist_mm < 40
        assert 55 < result.max_marker_dist_mm < 65
        statistics = result.statistics
        assert statistics["method"] == "plateau_median"
        assert statistics["min_endpoint"]["frames"] == 5
        assert statistics["max_endpoint"]["frames"] == 5
        assert statistics["thresholds"]["max_reprojection_error_px"] == 2.0
    finally:
        node.destroy_node()


@pytest.mark.parametrize(
    "scenario", ["resolution", "no_markers", "degenerate", "no_dwell", "overwrite"]
)
def test_failed_calibration_preserves_existing_output(
    tmp_path, camera_path, ros_context, scenario
):
    """分辨率错误、无双码、退化范围、端点未停留及禁止覆盖时保留已有文件。"""
    frames = stroke()
    if scenario == "resolution":
        frames[-1] = np.zeros((240, 320, 3), dtype=np.uint8)
    elif scenario == "no_markers":
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
    elif scenario == "degenerate":
        frames = [make_frame(180)] * 10
    elif scenario == "no_dwell":
        frames = [make_frame(180), make_frame(300)]
    bag = write_bag(tmp_path / "bag", [("/camera", image_message(frame)) for frame in frames])
    output = tmp_path / "existing.yaml"
    output.write_text("keep existing calibration", encoding="utf-8")
    node = make_node(bag, camera_path, output, overwrite=scenario != "overwrite")
    try:
        with pytest.raises((ValueError, RuntimeError)):
            node.run_mcap()
        assert output.read_text(encoding="utf-8") == "keep existing calibration"
    finally:
        node.destroy_node()


def test_main_returns_nonzero_on_missing_input(tmp_path, camera_path):
    """入口在 MCAP 启动失败时返回非零，便于命令行和自动化判断失败。"""
    with pytest.raises(SystemExit) as error:
        main(args=["--ros-args", "-p", "source_mode:=mcap", "-p",
                   f"mcap_path:={tmp_path / 'missing.mcap'}", "-p",
                   f"camera_calibration_path:={camera_path}"])
    assert error.value.code == 1


def test_storage_read_failure_does_not_save(tmp_path, camera_path, ros_context, monkeypatch):
    """扫描中途发生存储读取错误时，已有标定不会被部分采样结果覆盖。"""
    bag = write_bag(tmp_path / "bag", [("/camera", image_message(make_frame()))])
    output = tmp_path / "existing.yaml"
    output.write_text("keep existing calibration", encoding="utf-8")

    def broken_frames(source):
        """模拟读取一个有效图像后发生存储损坏。"""
        yield make_frame(180), image_message(make_frame()).header
        raise RuntimeError("storage read failed")

    monkeypatch.setattr(McapImageSource, "frames", broken_frames)
    node = make_node(bag, camera_path, output, overwrite=True)
    try:
        with pytest.raises(RuntimeError, match="storage read failed"):
            node.run_mcap()
        assert output.read_text(encoding="utf-8") == "keep existing calibration"
    finally:
        node.destroy_node()


def test_topic_mode_still_requires_start(tmp_path, camera_path, ros_context):
    """实时模式仍等待 start，随后可采集图像并通过 save 保存。"""
    output = tmp_path / "topic.yaml"
    node = GripperCalibrationNode(parameter_overrides=[
        ("camera_calibration_path", str(camera_path)), ("output_path", str(output)),
    ])
    try:
        node._image_callback(image_message(make_frame(180)))
        assert node._accumulator.count == 0
        assert node._start(Trigger.Request(), Trigger.Response()).success
        for frame in stroke():
            node._image_callback(image_message(frame))
        assert node._save(Trigger.Request(), Trigger.Response()).success
        assert load_gripper_calibration(str(output)).total_valid_frames == 10
    finally:
        node.destroy_node()


def test_video_mode_still_saves_and_exits(tmp_path, camera_path):
    """本地视频模式通过原有入口自动保存并正常退出。"""
    video_path = tmp_path / "gripper.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    assert writer.isOpened()
    for frame in stroke():
        writer.write(frame)
    writer.release()
    output = tmp_path / "video.yaml"
    result = subprocess.run([
        sys.executable, "-m", "gripper_openness.gripper_calibration_node", "--ros-args",
        "-p", "source_mode:=video", "-p", f"video_path:={video_path}",
        "-p", f"camera_calibration_path:={camera_path}", "-p", f"output_path:={output}",
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert load_gripper_calibration(str(output)).total_valid_frames == 10
