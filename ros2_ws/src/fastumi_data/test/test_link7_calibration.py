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
DEFAULT_TRACKER_SHA = "1e5f2758fd102b577946124d93e769c8298b60bbcc32648314226b4649118f50"


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
    }
    document.update(overrides)
    path = extrinsic.with_name(extrinsic.stem + ".binding.yaml")
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_default_extrinsic_binding_matches_source_tracker() -> None:
    """默认外参与绑定摘要一致，绑定设备为 LHR-B77A06A7，偏移按 V4L2 链路重新标定换算为 −1.841 ms。"""
    calibration = load_link7_calibration(
        str(REPOSITORY_CALIBRATION / "T_cam_tracker.yaml"),
        str(REPOSITORY_CALIBRATION / "T_cam_end.yaml"),
    )
    assert calibration.tracker_camera.sha256 == DEFAULT_TRACKER_SHA
    assert calibration.binding is not None
    assert calibration.binding.calibration_sha256 == DEFAULT_TRACKER_SHA
    assert calibration.tracker_serial == "LHR-B77A06A7"
    assert calibration.tracker_serial_source == "binding"
    assert calibration.tracker_time_offset_ms == pytest.approx(-1.8407007545651248)
    metadata = calibration.metadata()
    assert metadata["expected_tracker_serial"] == "LHR-B77A06A7"
    assert metadata["tracker_camera_extrinsic_sha256"] == DEFAULT_TRACKER_SHA
    assert metadata["time_calibration_timestamp_source"] == "v4l2_buffer_monotonic_mapped_to_ros"
    assert "tag_compressed_2" in metadata["time_calibration_conditions"]
    assert calibration.binding.source_collection_uuid == "4c5ce199-66f0-47a0-a9e4-0323373acf2a"


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


def test_time_offset_means_camera_later_than_tracker() -> None:
    """固定按 t_camera = t_tracker + offset：查询时间为图像时间减 offset。"""
    assert tracker_query_offset_ms(0.032364) == pytest.approx(-32.364)
    assert tracker_query_offset_ms(-0.002) == pytest.approx(2.0)


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


def test_inline_time_offset_needs_no_declared_convention(tmp_path: Path) -> None:
    """外参内非零 time_offset 无需声明语义，直接按 t_camera = t_tracker + offset 换算。"""
    tracker, end = make_pair(tmp_path, time_offset=0.02)
    calibration = load_link7_calibration(str(tracker), str(end), "LHR-X")
    assert calibration.tracker_time_offset_ms == pytest.approx(-20.0)
    assert "time_offset_convention" not in calibration.metadata()


@pytest.mark.parametrize("location", ["inline", "binding"])
@pytest.mark.parametrize("convention", [
    "t_camera_eq_t_tracker_plus_offset",
    "t_tracker_eq_t_camera_plus_offset",
    "unknown_convention",
])
def test_legacy_time_offset_declaration_is_checked(
    tmp_path: Path, location: str, convention: str
) -> None:
    """旧外参与绑定的同向声明保持兼容，相反或未知声明在同步前报错。"""
    # 按声明位置生成外参或配套绑定，确保摘要检查正常通过。
    extra = {"time_offset_convention": convention} if location == "inline" else {}
    tracker, end = make_pair(tmp_path, time_offset=0.02, **extra)
    if location == "binding":
        write_binding(tracker, "LHR-X", time_offset_convention=convention)

    if convention == "t_camera_eq_t_tracker_plus_offset":
        calibration = load_link7_calibration(str(tracker), str(end), "LHR-X")
        assert calibration.tracker_time_offset_ms == pytest.approx(-20.0)
    else:
        message = "取反" if convention == "t_tracker_eq_t_camera_plus_offset" else "未知"
        with pytest.raises(CalibrationError, match=message):
            load_link7_calibration(str(tracker), str(end), "LHR-X")


def test_missing_time_offset_defaults_to_zero(tmp_path: Path) -> None:
    """外参没有 time_offset 时按 0 处理并记录提示。"""
    tracker, end = make_pair(tmp_path)
    calibration = load_link7_calibration(str(tracker), str(end), "LHR-X")
    assert calibration.tracker_time_offset_ms == 0.0
    assert any("按 0 处理" in note for note in calibration.notes)


def test_non_rigid_matrix_is_rejected(tmp_path: Path) -> None:
    """外参矩阵必须是刚体变换。"""
    tracker, end = make_pair(tmp_path)
    matrix = np.eye(4)
    matrix[0, 0] = 2.0
    write_extrinsic(end, matrix)
    with pytest.raises(CalibrationError, match="正交"):
        load_link7_calibration(str(tracker), str(end), "LHR-X")


def test_default_files_are_consumed_byte_for_byte(tmp_path: Path) -> None:
    """复制默认外参与绑定到新目录后仍通过摘要预检（不依赖当前工作目录）。"""
    for name in ("T_cam_tracker.yaml", "T_cam_tracker.binding.yaml", "T_cam_end.yaml"):
        shutil.copy2(REPOSITORY_CALIBRATION / name, tmp_path / name)
    calibration = load_link7_calibration(
        str(tmp_path / "T_cam_tracker.yaml"), str(tmp_path / "T_cam_end.yaml")
    )
    assert calibration.tracker_serial == "LHR-B77A06A7"


def test_timestamp_source_must_match_calibration_chain(tmp_path: Path) -> None:
    """默认标定来自 V4L2 映射链路：同链路记录可用，旧回调 now() 记录（含未记录来源）被拒绝。"""
    from fastumi_data.link7_calibration import check_timestamp_source

    calibration = load_link7_calibration(
        str(REPOSITORY_CALIBRATION / "T_cam_tracker.yaml"),
        str(REPOSITORY_CALIBRATION / "T_cam_end.yaml"),
    )
    assert calibration.binding.time_calibration_timestamp_source == "v4l2_buffer_monotonic_mapped_to_ros"
    assert check_timestamp_source(calibration, "v4l2_buffer_monotonic_mapped_to_ros") == ""
    for legacy in (None, "unrecorded", "callback_now"):
        with pytest.raises(CalibrationError, match="重新标定"):
            check_timestamp_source(calibration, legacy)
    tracker, end = make_pair(tmp_path)
    write_binding(tracker, "LHR-BOUND")
    custom = load_link7_calibration(str(tracker), str(end))
    assert "无法核对" in check_timestamp_source(custom, "v4l2_buffer_monotonic_mapped_to_ros")
