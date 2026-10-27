"""使用真实序列化的合成 MCAP 验证离线全链路、异常处理及结果保护。"""

import csv
from dataclasses import replace
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import numpy as np
import pytest
from rclpy.serialization import serialize_message
import rosbag2_py
from sensor_msgs.msg import CompressedImage
import yaml

from fastumi_umi_tcp_calibration.calibration import solve
from fastumi_umi_tcp_calibration.cli import main
from fastumi_umi_tcp_calibration.config import QualitySettings, load_fisheye_camera
from fastumi_umi_tcp_calibration.dual_tag import DualTagEstimator
from fastumi_umi_tcp_calibration.mcap_source import McapImageSource
from fastumi_umi_tcp_calibration.output import publish_result
from fastumi_umi_tcp_calibration.transforms import axis_angle_matrix, make_transform
from synthetic import render_dual_tags
from test_dual_tag import gripper_utag_pose


REPO = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def scene():
    """使用仓库 UMI 内参渲染一组已知双 TAG 位姿。"""
    camera = load_fisheye_camera(REPO / "config/calibration/UmiCalib.yaml")
    truth = gripper_utag_pose()
    return camera, truth, render_dual_tags(camera, truth)


def image_message(image, encoding="raw"):
    """把 BGR 合成图转换为带源时间戳的 raw/JPEG/PNG 消息。"""
    if encoding == "raw":
        message = CvBridge().cv2_to_imgmsg(image, encoding="bgr8")
    else:
        message = CompressedImage()
        message.format = encoding
        success, data = cv2.imencode("." + encoding, image)
        assert success
        message.data = data.tobytes()
    message.header.stamp.sec = 17
    return message


def write_bag(path, messages, split=False):
    """写入真实 MCAP；可显式分卷以检验目录读取覆盖所有文件。"""
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topics = {}
    for topic, message in messages:
        if topic not in topics:
            kind = type(message).__module__.split(".")[0] + "/msg/" + type(message).__name__
            topics[topic] = kind
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    id=len(topics), name=topic, type=kind, serialization_format="cdr"
                )
            )
    for index, (topic, message) in enumerate(messages):
        if split and index == len(messages) // 2:
            writer.split_bagfile()
        writer.write(topic, serialize_message(message), 1_000_000_000 + index)
    writer.close()
    return path


def arguments(tmp_path, bag, minimum=3):
    """生成完整 CLI 参数和包含非单位旋转的 TAG→TCP 真值配置。"""
    tag_tcp = make_transform(axis_angle_matrix([1, 2, 3], 0.4), [0.01, 0.12, -0.018])
    config = tmp_path / "tag.yaml"
    config.write_text(yaml.safe_dump({"T_u_tag_tcp": tag_tcp.tolist()}))
    return [
        "--bag",
        str(bag),
        "--camera-config",
        str(REPO / "config/calibration/UmiCalib.yaml"),
        "--tag-tcp-config",
        str(config),
        "--output",
        str(tmp_path / "result.yaml"),
        "--diagnostics-dir",
        str(tmp_path / "diagnostics"),
        "--min-inliers",
        str(minimum),
    ], tag_tcp


@pytest.mark.parametrize("encoding", ["raw", "jpg", "png"])
@pytest.mark.parametrize("single_file", [False, True])
def test_end_to_end(tmp_path, scene, encoding, single_file):
    """真实录包经检测及聚合恢复 TCP，保留所有时间戳和诊断。"""
    camera, truth, image = scene
    bag = write_bag(tmp_path / "bag", [("/umi/image", image_message(image, encoding))] * 4)
    args, tag_tcp = arguments(tmp_path, next(bag.glob("*.mcap")) if single_file else bag)
    assert main(args) == 0
    result = np.array(yaml.safe_load((tmp_path / "result.yaml").read_text())["T_u_cam_tcp"])
    expected = truth @ tag_tcp
    assert np.linalg.norm(result[:3, 3] - expected[:3, 3]) < 0.004
    report = yaml.safe_load((tmp_path / "diagnostics/report.yaml").read_text())
    assert report["success"] and report["inliers"] == 4
    estimate = DualTagEstimator(camera).estimate(image).camera_from_utag
    if encoding == "raw":
        assert np.allclose(result, estimate @ tag_tcp)
    with (tmp_path / "diagnostics/samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and rows[0]["timestamp_ns"] == "17000000000"
    assert len(list((tmp_path / "diagnostics/debug").glob("*.jpg"))) == 4
    assert (tmp_path / "diagnostics/config/T_u_tag_tcp.yaml").read_bytes() == (
        tmp_path / "tag.yaml"
    ).read_bytes()


def test_session_split_and_topic_selection(tmp_path, scene):
    """会话选择 UMI 话题且遍历两个分卷，显式话题覆盖会话选择。"""
    image = scene[2]
    session = tmp_path / "session"
    session.mkdir()
    write_bag(
        session / "raw/bag",
        [("/umi", image_message(image)), ("/other", image_message(image))] * 3,
        split=True,
    )
    (session / "session.yaml").write_text(
        yaml.safe_dump({"bag_uri": "raw/bag", "image": {"topic": "/umi"}})
    )
    assert len(list((session / "raw/bag").glob("*.mcap"))) == 2
    source = McapImageSource(session)
    try:
        assert source.image_topic == "/umi" and len(list(source.frames())) == 3
    finally:
        source.close()
    source = McapImageSource(session, "/other")
    source.close()
    (session / "session.yaml").unlink()
    with pytest.raises(ValueError, match="候选"):
        McapImageSource(session / "raw/bag")
    with pytest.raises(ValueError, match="缺少"):
        McapImageSource(session / "raw/bag", "/missing")


def test_corrupt_frames_and_minimum(tmp_path, scene):
    """损坏图片记入拒绝计数；有效内点不足时保留旧输出。"""
    message = image_message(scene[2], "png")
    bad = CompressedImage()
    bad.format = "png"
    bad.data = b"not an image"
    bag = write_bag(tmp_path / "bag", [("/image", message), ("/image", bad), ("/image", message)])
    args, _ = arguments(tmp_path, bag, minimum=3)
    output = tmp_path / "result.yaml"
    output.write_text("existing calibration")
    assert main(args) == 1
    assert not (tmp_path / "diagnostics").exists()
    assert main([*args, "--overwrite"]) == 1
    assert output.read_text() == "existing calibration"
    report = yaml.safe_load((tmp_path / "diagnostics/report.yaml").read_text())
    assert not report["success"] and report["decode_failures"] == 1 and report["inliers"] == 2


def test_dimension_mismatch_is_fatal(tmp_path):
    """图像尺寸不符时终止运行，不发布结果。"""
    bag = write_bag(tmp_path / "bag", [("/image", image_message(np.zeros((10, 10, 3), np.uint8)))])
    args, _ = arguments(tmp_path, bag)
    assert main(args) == 1
    assert not (tmp_path / "result.yaml").exists()
    report = yaml.safe_load((tmp_path / "diagnostics/report.yaml").read_text())
    assert "图像尺寸" in report["error"]


def test_no_tags_and_empty_bag(tmp_path, scene):
    """全包缺码时非零退出且不输出标定矩阵。"""
    blank = np.full_like(scene[2], 100)
    bag = write_bag(tmp_path / "bag", [("/image", image_message(blank))])
    args, _ = arguments(tmp_path, bag)
    assert main(args) == 1
    assert not (tmp_path / "result.yaml").exists()


def test_composition_and_quality(scene):
    """验证非单位旋转的矩阵乘法顺序和全部单帧质量门限。"""
    observation = DualTagEstimator(scene[0]).estimate(scene[2])
    tag_tcp = make_transform(axis_angle_matrix([1, 1, 0], 0.7), [0.02, 0.12, -0.018])
    matrix, _, report = solve([observation] * 3, tag_tcp, QualitySettings(min_inliers=3))
    assert np.allclose(matrix, observation.camera_from_utag @ tag_tcp)
    assert not np.allclose(matrix, tag_tcp @ observation.camera_from_utag)
    settings = QualitySettings()
    assert settings.rejection_reason(replace(observation, normal_angle_deg=11))
    assert settings.rejection_reason(replace(observation, coplanarity_deg=11))
    assert settings.rejection_reason(
        replace(observation, tag0=replace(observation.tag0, reprojection_rms_px=2))
    )
    assert report["tcp_spread"]["translation_rms_mm"] < 1e-9


@pytest.mark.parametrize(
    "field,value", [("max_reprojection_px", float("nan")), ("mad_k", -1), ("min_inliers", 0)]
)
def test_invalid_settings(field, value):
    """拒绝无法解释的质量门限。"""
    with pytest.raises(ValueError):
        QualitySettings(**{field: value})


def test_atomic_publication(tmp_path):
    """验证无覆盖模式不会截断旧文件，显式覆盖正常且无暂存文件残留。"""
    target = tmp_path / "result.yaml"
    target.write_text("old")
    with pytest.raises(FileExistsError):
        publish_result(target, np.eye(4))
    assert target.read_text() == "old"
    publish_result(target, np.eye(4), overwrite=True)
    assert "T_u_cam_tcp" in target.read_text()
    assert not list(tmp_path.glob(".umi_tcp_*"))


def test_default_minimum_and_debug_cap(tmp_path, scene):
    """默认最小 30 内点可成功处理全包，调试图始终不超过 20 张。"""
    bag = write_bag(tmp_path / "bag", [("/image", image_message(scene[2], "png"))] * 31)
    args, _ = arguments(tmp_path, bag)
    args = args[:-2]
    (tmp_path / "result.yaml").write_text("old")
    assert main([*args, "--overwrite"]) == 0
    report = yaml.safe_load((tmp_path / "diagnostics/report.yaml").read_text())
    assert report["inliers"] == 31
    assert len(list((tmp_path / "diagnostics/debug").glob("*.jpg"))) == 20


@pytest.mark.parametrize(
    "change",
    [
        {"intrinsics": [0, 1, 0, 0]},
        {"intrinsics": [float("nan"), 1, 0, 0]},
        {"resolution": [1920.5, 1080]},
        {"distortion_model": "plumb_bob"},
        {"distortion_coeffs": [0, 0, 0]},
        {"camera_model": "omni"},
    ],
)
def test_invalid_camera(tmp_path, change):
    """内参模型或数值无效时拒绝处理，防止静默缩放或使用错误模型。"""
    data = yaml.safe_load((REPO / "config/calibration/UmiCalib.yaml").read_text())
    data["cam0"].update(change)
    path = tmp_path / "camera.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_fisheye_camera(path)


def test_input_and_diagnostics_collision(tmp_path, scene):
    """输出不得覆盖输入配置或占用诊断文件路径。"""
    bag = write_bag(tmp_path / "bag", [("/image", image_message(scene[2]))])
    args, _ = arguments(tmp_path, bag)
    original = (tmp_path / "tag.yaml").read_bytes()
    assert main([*args, "--output", str(tmp_path / "tag.yaml"), "--overwrite"]) == 1
    assert (tmp_path / "tag.yaml").read_bytes() == original
    assert main([*args, "--output", str(tmp_path / "diagnostics/report.yaml")]) == 1


def test_reader_failure_does_not_publish(tmp_path, scene, monkeypatch):
    """读取过程中发生存储故障时保留失败报告并阻止部分数据生成正式标定。"""
    bag = write_bag(tmp_path / "bag", [("/image", image_message(scene[2]))])
    args, _ = arguments(tmp_path, bag, minimum=1)
    original = McapImageSource.frames

    def broken_frames(self):
        """先提供有效帧，再模拟存储层读取失败。"""
        yield from original(self)
        raise RuntimeError("模拟存储读取失败")

    monkeypatch.setattr(McapImageSource, "frames", broken_frames)
    assert main(args) == 1
    assert not (tmp_path / "result.yaml").exists()
    report = yaml.safe_load((tmp_path / "diagnostics/report.yaml").read_text())
    assert not report["success"] and "存储读取失败" in report["error"]
