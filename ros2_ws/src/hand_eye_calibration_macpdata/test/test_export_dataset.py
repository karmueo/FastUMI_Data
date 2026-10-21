"""手眼标定 MCAP 导出与求解：仿真采集 → 导出 → 对照真值。

需要已 source ros2_ws/install/setup.bash（本包、fastumi_bringup、fastumi_recorder 可导入）。
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("hand_eye_calibration_macpdata.capture")
pytest.importorskip("fastumi_recorder.bag_writer")

from fastumi_bringup.transforms import rotation_angle  # noqa: E402
from hand_eye_calibration_macpdata import export_dataset as exporter  # noqa: E402
from hand_eye_calibration_macpdata.board import make_detector, FisheyeCamera  # noqa: E402
from hand_eye_calibration_macpdata.estimate import load_prior  # noqa: E402
from hand_eye_sim import PRIOR, run_session  # noqa: E402


@pytest.fixture(scope="module", params=["aprilgrid", "checkerboard"])
def session(request, tmp_path_factory):
    return request.param, run_session(
        tmp_path_factory.mktemp(request.param), request.param, num_samples=15)


def test_solves_ground_truth_and_writes_prior_format(session, tmp_path):
    board, (session_dir, rig, samples, _) = session
    output = tmp_path / "dataset"
    summary = exporter.export(session_dir, output, prior_path=PRIOR)
    assert summary["exported"] == len(samples) == 15 and not summary["undetected"]
    result = yaml.safe_load((output / "hand_eye_result.yaml").read_text())
    estimate = np.array(result["T_flange_cam"])
    truth = rig.flange_from_camera
    assert np.linalg.norm(estimate[:3, 3] - truth[:3, 3]) < 0.002
    assert rotation_angle(estimate[:3, :3], truth[:3, :3]) < np.radians(0.2)
    assert np.array(result["T_base_board"]) == pytest.approx(rig.base_from_board, abs=2e-3)
    assert result["reprojection_rms_px"] < 1.0 and not result["outliers"]
    # 仿真真值即先验，cam0.T_cam_hand 写法与 T_cam_end.yaml 一致，可被 load_prior 读回。
    assert result["delta_to_prior"]["translation_mm"] < 2.0
    assert load_prior(output / "hand_eye_result.yaml") == pytest.approx(estimate, abs=1e-9)
    assert result["board"]["target_type"] == board


def test_export_layout_and_formats(session, tmp_path):
    board, (session_dir, _, _, _) = session
    output = tmp_path / "dataset"
    summary = exporter.export(session_dir, output, solve=False)
    assert not (output / "hand_eye_result.yaml").exists()
    images = sorted(path.name for path in (output / "images").iterdir())
    assert images == sorted(f"{index}.png" for index in range(15))
    rows = (output / "poses.csv").read_text().strip().split("\n")
    assert len(rows) == 15
    values = [float(value) for value in rows[0].split("\t")]
    assert len(values) == 7 and np.linalg.norm(values[3:]) == pytest.approx(1.0, abs=1e-6)
    parameters = yaml.safe_load((output / "intrinsics.yaml").read_text())
    parameters = parameters["wrist_camera"]["ros__parameters"]
    assert len(parameters["K"]) == 9 and parameters["D"] == [0.0] * 5
    assert (parameters["width"], parameters["height"]) == (1920, 1080)
    readme = (output / "README.txt").read_text()
    if board == "aprilgrid":
        assert "command" not in summary and "不支持 AprilGrid" in readme
    else:
        assert '--calibration-pattern "2 8 11 0.03"' in readme
    with pytest.raises(FileExistsError):
        exporter.export(session_dir, output)


def test_rectified_images_match_pinhole_model(session, tmp_path):
    """矫正图在针孔模型（新 K、D=0）下检测到的板位姿与鱼眼检测一致。"""
    board, (session_dir, _, _, _) = session
    output = tmp_path / "dataset"
    exporter.export(session_dir, output, solve=False)
    parameters = yaml.safe_load((output / "intrinsics.yaml").read_text())
    K = np.array(parameters["wrist_camera"]["ros__parameters"]["K"]).reshape(3, 3)
    session_info, _, samples = exporter.read_session(session_dir)
    spec = exporter.board_spec_from_dict(session_info["board"])
    fisheye = exporter.session_camera(session_info["camera"])
    # 检测器只用于取特征点像素（其内置位姿按鱼眼模型计算，此处不用），位姿用针孔 PnP。
    pinhole = FisheyeCamera(K=K, D=np.zeros((4, 1)), width=fisheye.width, height=fisheye.height)
    for number, sample in enumerate(samples[:3]):
        image = cv2.imread(str(output / "images" / f"{number}.png"))
        observation = make_detector(spec, pinhole).detect(image)
        assert observation.corner_count >= 0.8 * spec.corner_count
        ok, rvec, tvec = cv2.solvePnP(spec.object_points(observation.ids),
                                      observation.pixels, K, None)
        assert ok
        rotation, _ = cv2.Rodrigues(rvec)
        reference = make_detector(spec, fisheye).detect(cv2.imdecode(
            np.frombuffer(bytes(sample.image.data), np.uint8), cv2.IMREAD_COLOR))
        assert np.linalg.norm(tvec.reshape(3) - reference.camera_from_board[:3, 3]) < 0.004
        assert rotation_angle(rotation, reference.camera_from_board[:3, :3]) < np.radians(1.0)


def test_rejected_samples_are_dropped(session, tmp_path):
    """撤销记录对应的样本不导出（直接在 bag 副本中追加撤销消息）。"""
    import json
    import shutil
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from std_msgs.msg import String

    _, (session_dir, _, _, _) = session
    copy = tmp_path / "session"
    shutil.copytree(session_dir / "bag", copy / "bag_source")
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(copy / "bag_source"), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(copy / "bag"), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    for metadata in reader.get_all_topics_and_types():
        writer.create_topic(metadata)
    last = 0
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        writer.write(topic, data, stamp)
        last = max(last, stamp)
    writer.write("/hand_eye/sample_rejected",
                 serialize_message(String(data=json.dumps({"index": 3}))), last + 1)
    del writer
    _, _, samples = exporter.read_session(copy)
    assert 3 not in [sample.index for sample in samples] and len(samples) == 14
