"""验证 Tracker–相机标定配置解析和 AprilGrid 米制几何。"""

from pathlib import Path

import numpy as np
import pytest

from fastumi_data.tracker_camera_config import (
    AprilGridSpec,
    CalibrationSettings,
    FisheyeCameraModel,
    default_camera_config_path,
    load_aprilgrid,
    load_kalibr_camera,
    tag_object_corners,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_aprilgrid_tag_corners_use_kalibr_spacing() -> None:
    """标签间距应按 tagSpacing 与 tagSize 的乘积解释。"""
    spec = AprilGridSpec(6, 6, 0.055, 0.3, "tag36h11")
    np.testing.assert_allclose(
        tag_object_corners(spec, 0),
        [
            [0.0, 0.0, 0.0],
            [0.055, 0.0, 0.0],
            [0.055, 0.055, 0.0],
            [0.0, 0.055, 0.0],
        ],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        tag_object_corners(spec, 1),
        [
            [0.0715, 0.0, 0.0],
            [0.1265, 0.0, 0.0],
            [0.1265, 0.055, 0.0],
            [0.0715, 0.055, 0.0],
        ],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        tag_object_corners(spec, 6),
        [
            [0.0, 0.0715, 0.0],
            [0.055, 0.0715, 0.0],
            [0.055, 0.1265, 0.0],
            [0.0, 0.1265, 0.0],
        ],
        atol=1.0e-12,
    )
    assert spec.board_extent_m == pytest.approx((0.4125, 0.4125))


def test_load_default_tof_camera_and_target_files() -> None:
    """默认 ToF 相机和目标配置应解析为已确认参数。"""
    camera = load_kalibr_camera(default_camera_config_path())
    target = load_aprilgrid(str(PROJECT_ROOT / "docs/april_6x6.yaml"))
    assert camera.resolution == (2048, 1536)
    assert camera.distortion_model == "equidistant"
    assert camera.k[0, 0] == pytest.approx(405.67036610341245)
    assert camera.d[0] == pytest.approx(0.08165390616646588)
    assert target.tag_size_m == pytest.approx(0.052)
    assert target.tag_spacing == pytest.approx(0.3725)
    assert target.tag_family == "tag36h11"


def test_load_legacy_kalibr_camera_file() -> None:
    """显式旧 Kalibr 相机配置应继续作为兼容覆盖加载。"""
    camera = load_kalibr_camera(
        str(PROJECT_ROOT / "config/calibration/kalibr_data-camchain-imucam.yaml")
    )
    assert camera.resolution == (1280, 1280)
    assert camera.distortion_model == "equidistant"
    assert camera.k[0, 0] == pytest.approx(397.07575683833136)


def test_rejects_out_of_range_tag_id() -> None:
    """超出 0 到 35 的 ID 应被拒绝。"""
    with pytest.raises(ValueError, match="Tag ID"):
        tag_object_corners(
            AprilGridSpec(6, 6, 0.055, 0.3, "tag36h11"), 36
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("k", np.eye(2), "K"),
        ("d", np.zeros(5), "D"),
        ("resolution", (0, 1280), "分辨率"),
        ("distortion_model", "radtan", "equidistant"),
    ],
)
def test_camera_model_rejects_invalid_parameters(
    field: str, value: object, message: str
) -> None:
    """鱼眼模型应拒绝形状、分辨率和畸变类型错误。"""
    arguments = {
        "k": np.diag([400.0, 400.0, 1.0]),
        "d": np.zeros(4),
        "resolution": (1280, 1280),
        "camera_model": "pinhole",
        "distortion_model": "equidistant",
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        FisheyeCameraModel(**arguments)


def test_calibration_settings_validate_search_and_quality_limits() -> None:
    """运行设置应拒绝无效步长、筛选门限和反向时间范围。"""
    assert CalibrationSettings().min_tags == 6
    assert CalibrationSettings().intrinsics_frequency_hz == pytest.approx(4.0)
    with pytest.raises(ValueError, match="时间偏移"):
        CalibrationSettings(
            time_offset_min_ms=10.0, time_offset_max_ms=-10.0
        )
    with pytest.raises(ValueError, match="frame_stride"):
        CalibrationSettings(frame_stride=0)


def test_loaders_reject_wrong_kalibr_models(tmp_path: Path) -> None:
    """加载器应明确拒绝非 pinhole/equidistant 配置。"""
    camera_path = tmp_path / "camera.yaml"
    camera_path.write_text(
        """cam0:
  camera_model: omni
  distortion_model: radtan
  intrinsics: [400.0, 400.0, 640.0, 640.0]
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  resolution: [1280, 1280]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pinhole"):
        load_kalibr_camera(str(camera_path))


def test_loader_rejects_wrong_tof_distortion_model(tmp_path: Path) -> None:
    """ToF RGB 配置应拒绝非 fisheye 畸变模型。"""
    camera_path = tmp_path / "tof_camera.yaml"
    camera_path.write_text(
        """rgb:
  distortion_model: radtan
  intrinsics: [400.0, 400.0, 640.0, 480.0]
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  resolution: [1280, 960]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fisheye"):
        load_kalibr_camera(str(camera_path))


def test_loader_rejects_ambiguous_camera_blocks(tmp_path: Path) -> None:
    """ToF 与 Kalibr 相机块同时存在时应拒绝猜测标定来源。"""
    camera_path = tmp_path / "ambiguous_camera.yaml"
    camera_path.write_text(
        """rgb:
  distortion_model: fisheye
  intrinsics: [400.0, 400.0, 640.0, 480.0]
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  resolution: [1280, 960]
cam0:
  camera_model: pinhole
  distortion_model: equidistant
  intrinsics: [400.0, 400.0, 640.0, 480.0]
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  resolution: [1280, 960]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="同时包含"):
        load_kalibr_camera(str(camera_path))
