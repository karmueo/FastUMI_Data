"""验证示教/部署外参加载、Tracker 绑定身份核对、时间偏移符号和 Link7 组合方向。"""

from pathlib import Path
import shutil

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from fastumi_data.link7_calibration import (
    CalibrationError,
    compose_tracker_to_link7,
    load_link7_calibration,
    tracker_query_offset_ms,
)


REPOSITORY_CALIBRATION = Path(__file__).resolve().parents[4] / "config" / "calibration"
DEFAULT_TRACKER_SHA = "cce6807d8266935d6d5a9429773527707766ae007d7f66f167de4e662947c166"


def transform(rotvec, translation) -> np.ndarray:
    """构造齐次变换。"""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def write_extrinsic(path: Path, matrix: np.ndarray, **extra) -> Path:
    """写入 cam0.T_cam_hand 格式外参。"""
    document = {"cam0": {"T_cam_hand": matrix.tolist()}}
    document.update(extra)
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def write_binding(extrinsic: Path, serial: str, **overrides) -> Path:
    """写入与外参字节一致的绑定清单。"""
    import hashlib

    document = {
        "calibration_file": extrinsic.name,
        "calibration_sha256": hashlib.sha256(extrinsic.read_bytes()).hexdigest(),
        "tracker_serial": serial,
        "source_collection_uuid": "uuid",
        "source_status_topic": "/vive_tracker/status",
        "time_offset_convention": "t_camera_eq_t_tracker_plus_offset",
        "time_calibration_status": "provisional",
    }
    document.update(overrides)
    path = extrinsic.with_name(extrinsic.stem + ".binding.yaml")
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_default_extrinsic_binding_matches_source_tracker() -> None:
    """默认外参与绑定摘要一致，绑定设备为 LHR-B77A06A7，偏移按旧标定换算为 −32.364 ms。"""
    calibration = load_link7_calibration(
        str(REPOSITORY_CALIBRATION / "T_cam_tracker.yaml"),
        str(REPOSITORY_CALIBRATION / "T_cam_end.yaml"),
    )
    assert calibration.tracker_camera.sha256 == DEFAULT_TRACKER_SHA
    assert calibration.binding is not None
    assert calibration.binding.calibration_sha256 == DEFAULT_TRACKER_SHA
    assert calibration.tracker_serial == "LHR-B77A06A7"
    assert calibration.tracker_serial_source == "binding"
    assert calibration.tracker_time_offset_ms == pytest.approx(-32.364095977515536)
    assert calibration.time_calibration_status == "provisional"
    metadata = calibration.metadata()
    assert metadata["expected_tracker_serial"] == "LHR-B77A06A7"
    assert metadata["tracker_camera_extrinsic_sha256"] == DEFAULT_TRACKER_SHA
    assert metadata["time_calibration_timestamp_source"] == "callback_now"
    assert "tag_jpeg_1" in metadata["time_calibration_conditions"]


def test_composition_direction_and_camera_round_trip() -> None:
    """非平凡外参下 T_W_Link7 = T_W_Tracker × inv(T_Cu_Tracker) × T_Cd_Link7，且两相机光轴重合。"""
    camera_from_tracker = transform([0.3, -0.2, 0.5], [0.02, -0.05, 0.11])
    camera_from_link7 = transform([-0.4, 0.1, 0.2], [-0.01, 0.09, -0.02])
    tracker_from_link7 = compose_tracker_to_link7(camera_from_tracker, camera_from_link7)
    world_from_tracker = transform([0.1, 0.7, -0.3], [1.0, 2.0, 0.5])
    world_from_link7 = world_from_tracker @ tracker_from_link7
    # 示教相机与部署相机坐标重合：两条链得到同一个世界系相机位姿。
    world_from_camera_via_tracker = world_from_tracker @ np.linalg.inv(camera_from_tracker)
    world_from_camera_via_link7 = world_from_link7 @ np.linalg.inv(camera_from_link7)
    np.testing.assert_allclose(world_from_camera_via_tracker, world_from_camera_via_link7, atol=1e-12)
    # 错误的求逆顺序会得到不同结果。
    wrong = world_from_tracker @ camera_from_tracker @ np.linalg.inv(camera_from_link7)
    assert not np.allclose(wrong, world_from_link7)


def test_time_offset_sign_follows_declared_convention() -> None:
    """t_camera = t_tracker + offset 时查询时间为图像时间减 offset。"""
    assert tracker_query_offset_ms(0.032364, "t_camera_eq_t_tracker_plus_offset") == pytest.approx(-32.364)
    assert tracker_query_offset_ms(0.032364, "t_tracker_eq_t_camera_plus_offset") == pytest.approx(32.364)
    with pytest.raises(CalibrationError):
        tracker_query_offset_ms(0.03, "time_offset")


def make_pair(tmp_path: Path, **tracker_extra):
    """写一对自定义外参，返回路径。"""
    tracker = write_extrinsic(
        tmp_path / "custom_tracker.yaml", transform([0.1, 0, 0], [0, 0, 0.1]), **tracker_extra
    )
    end = write_extrinsic(tmp_path / "custom_end.yaml", transform([0, 0.2, 0], [0.05, 0, 0]))
    return tracker, end


def test_custom_extrinsic_without_identity_requires_declared_serial(tmp_path: Path) -> None:
    """缺少文件内身份与绑定时必须用 --tracker-serial 声明，并记为用户声明。"""
    tracker, end = make_pair(tmp_path)
    with pytest.raises(CalibrationError, match="--tracker-serial"):
        load_link7_calibration(str(tracker), str(end))
    calibration = load_link7_calibration(str(tracker), str(end), "LHR-CUSTOM")
    assert calibration.tracker_serial == "LHR-CUSTOM"
    assert calibration.tracker_serial_source == "user_declared"
    assert calibration.time_calibration_status == "provisional"


def test_declared_serial_cannot_override_existing_binding(tmp_path: Path) -> None:
    """--tracker-serial 不能覆盖既有绑定；一致时只作核对。"""
    tracker, end = make_pair(tmp_path)
    write_binding(tracker, "LHR-BOUND")
    with pytest.raises(CalibrationError, match="冲突"):
        load_link7_calibration(str(tracker), str(end), "LHR-OTHER")
    calibration = load_link7_calibration(str(tracker), str(end), "LHR-BOUND")
    assert calibration.tracker_serial_source == "binding"


def test_inline_serial_must_match_binding(tmp_path: Path) -> None:
    """文件内身份与绑定清单同时存在时序列号必须一致。"""
    tracker, end = make_pair(tmp_path, tracker_serial="LHR-INLINE")
    write_binding(tracker, "LHR-BOUND")
    with pytest.raises(CalibrationError, match="冲突"):
        load_link7_calibration(str(tracker), str(end))
    write_binding(tracker, "LHR-INLINE")
    assert load_link7_calibration(str(tracker), str(end)).tracker_serial_source == "inline+binding"


def test_binding_digest_change_is_hard_error(tmp_path: Path) -> None:
    """外参字节改变后绑定摘要不匹配，必须中止。"""
    tracker, end = make_pair(tmp_path)
    write_binding(tracker, "LHR-BOUND")
    tracker.write_text(tracker.read_text() + "# edited\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match="摘要"):
        load_link7_calibration(str(tracker), str(end))


def test_nonzero_offset_without_convention_is_rejected(tmp_path: Path) -> None:
    """非零 time_offset 未声明符号语义时报错，显式提供语义后可用。"""
    tracker, end = make_pair(tmp_path, time_offset=0.02)
    with pytest.raises(CalibrationError, match="语义"):
        load_link7_calibration(str(tracker), str(end), "LHR-X")
    calibration = load_link7_calibration(
        str(tracker), str(end), "LHR-X", "t_tracker_eq_t_camera_plus_offset"
    )
    assert calibration.tracker_time_offset_ms == pytest.approx(20.0)


def test_non_rigid_matrix_is_rejected(tmp_path: Path) -> None:
    """外参矩阵必须是刚体变换。"""
    tracker, end = make_pair(tmp_path)
    matrix = np.eye(4)
    matrix[0, 0] = 2.0
    write_extrinsic(end, matrix)
    with pytest.raises(CalibrationError, match="正交"):
        load_link7_calibration(str(tracker), str(end), "LHR-X")


def test_binding_status_values_are_validated(tmp_path: Path) -> None:
    """绑定清单的时间标定状态只能为 provisional 或 verified。"""
    tracker, end = make_pair(tmp_path)
    write_binding(tracker, "LHR-BOUND", time_calibration_status="done")
    with pytest.raises(CalibrationError):
        load_link7_calibration(str(tracker), str(end))
    write_binding(tracker, "LHR-BOUND", time_calibration_status="verified")
    assert load_link7_calibration(str(tracker), str(end)).time_calibration_status == "verified"


def test_default_files_are_consumed_byte_for_byte(tmp_path: Path) -> None:
    """复制默认外参与绑定到新目录后仍通过摘要预检（不依赖当前工作目录）。"""
    for name in ("T_cam_tracker.yaml", "T_cam_tracker.binding.yaml", "T_cam_end.yaml"):
        shutil.copy2(REPOSITORY_CALIBRATION / name, tmp_path / name)
    calibration = load_link7_calibration(
        str(tmp_path / "T_cam_tracker.yaml"), str(tmp_path / "T_cam_end.yaml")
    )
    assert calibration.tracker_serial == "LHR-B77A06A7"


def test_timestamp_source_must_match_calibration_chain(tmp_path: Path) -> None:
    """默认标定来自回调 now() 链路：旧记录（未记录来源）可用，V4L2 映射链路的记录被拒绝。"""
    from fastumi_data.link7_calibration import check_timestamp_source

    calibration = load_link7_calibration(
        str(REPOSITORY_CALIBRATION / "T_cam_tracker.yaml"),
        str(REPOSITORY_CALIBRATION / "T_cam_end.yaml"),
    )
    assert calibration.binding.time_calibration_timestamp_source == "callback_now"
    assert check_timestamp_source(calibration, None) == ""
    assert check_timestamp_source(calibration, "unrecorded") == ""
    assert check_timestamp_source(calibration, "callback_now") == ""
    with pytest.raises(CalibrationError, match="重新标定"):
        check_timestamp_source(calibration, "v4l2_buffer_monotonic_mapped_to_ros")
    tracker, end = make_pair(tmp_path)
    write_binding(tracker, "LHR-BOUND")
    custom = load_link7_calibration(str(tracker), str(end))
    assert "无法核对" in check_timestamp_source(custom, "v4l2_buffer_monotonic_mapped_to_ros")
