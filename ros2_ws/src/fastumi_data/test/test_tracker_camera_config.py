"""验证 Tracker–相机标定配置解析和 AprilGrid 米制几何。"""

from pathlib import Path

import numpy as np
import pytest

from fastumi_data.tracker_camera_config import (
    AprilGridSpec,
    CheckerboardSpec,
    CalibrationSettings,
    FisheyeCameraModel,
    checkerboard_object_points,
    default_camera_config_path,
    load_aprilgrid,
    load_calibration_target,
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


def test_checkerboard_spec_and_object_points(tmp_path: Path) -> None:
    """棋盘格规格应按内部角点数和行优先坐标生成。"""
    path = tmp_path / "checkerboard.yaml"
    path.write_text(
        "target_type: checkerboard\n"
        "targetCols: 11\n"
        "targetRows: 8\n"
        "rowSpacingMeters: 0.03\n"
        "colSpacingMeters: 0.025\n",
        encoding="utf-8",
    )
    spec = load_calibration_target(str(path))
    assert isinstance(spec, CheckerboardSpec)
    assert spec.corner_count == 88
    assert spec.board_extent_m == pytest.approx((0.25, 0.21))
    np.testing.assert_allclose(
        checkerboard_object_points(spec)[[0, 1, 11]],
        [[0.0, 0.0, 0.0], [0.025, 0.0, 0.0], [0.0, 0.03, 0.0]],
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 8, 0.03, 0.03),
        (1, 8, 0.03, 0.03),
        (11, 2, 0.03, 0.03),
        (11, 0, 0.03, 0.03),
        (11.5, 8, 0.03, 0.03),
        (11, 8, 0.0, 0.03),
    ],
)
def test_checkerboard_spec_rejects_invalid_geometry(arguments) -> None:
    """棋盘格尺寸必须受 OpenCV 支持，间距必须是有限正数。"""
    with pytest.raises(ValueError):
        CheckerboardSpec(*arguments)


def _write_checkerboard_yaml(path: Path, fields: dict) -> None:
    """写入测试使用的棋盘格 YAML 字段。"""
    content = "\n".join(
        f"{key}: {value}" for key, value in fields.items()
    )
    path.write_text(content + "\n", encoding="utf-8")


def _checkerboard_yaml_fields() -> dict[str, str]:
    """返回测试使用的完整棋盘格 YAML 字段。"""
    return {
        "target_type": "checkerboard",
        "targetCols": "11",
        "targetRows": "8",
        "rowSpacingMeters": "0.03",
        "colSpacingMeters": "0.03",
    }


@pytest.mark.parametrize(
    "missing_field",
    [
        "target_type",
        "targetCols",
        "targetRows",
        "rowSpacingMeters",
        "colSpacingMeters",
    ],
)
def test_checkerboard_loader_requires_fields(
    tmp_path: Path, missing_field: str
) -> None:
    """棋盘格 YAML 缺少任一必需字段时应拒绝加载。"""
    fields = _checkerboard_yaml_fields()
    fields.pop(missing_field)
    path = tmp_path / "checkerboard-missing.yaml"
    _write_checkerboard_yaml(path, fields)
    with pytest.raises(ValueError):
        load_calibration_target(str(path))


@pytest.mark.parametrize(
    "field,value",
    [
        ("targetCols", "1"),
        ("targetRows", "2"),
        ("targetCols", "11.5"),
        ("targetRows", "8.5"),
        ("targetCols", "true"),
        ("targetRows", "false"),
    ],
)
def test_checkerboard_loader_rejects_unsupported_dimensions(
    tmp_path: Path, field: str, value: str
) -> None:
    """棋盘格内部角点行列必须是 OpenCV 支持的非布尔正整数。"""
    fields = _checkerboard_yaml_fields()
    fields[field] = value
    path = tmp_path / "checkerboard-dimensions.yaml"
    _write_checkerboard_yaml(path, fields)
    with pytest.raises(ValueError, match="正整数"):
        load_calibration_target(str(path))


@pytest.mark.parametrize(
    "field,value",
    [
        ("targetCols", "columns"),
        ("targetRows", "rows"),
        ("rowSpacingMeters", "spacing"),
        ("colSpacingMeters", "spacing"),
    ],
)
def test_checkerboard_loader_rejects_nonnumeric_values(
    tmp_path: Path, field: str, value: str
) -> None:
    """棋盘格尺寸和间距字段必须可解析为数值。"""
    fields = _checkerboard_yaml_fields()
    fields[field] = value
    path = tmp_path / "checkerboard-nonnumeric.yaml"
    _write_checkerboard_yaml(path, fields)
    with pytest.raises(ValueError):
        load_calibration_target(str(path))


@pytest.mark.parametrize(
    "field,value",
    [
        ("rowSpacingMeters", ".nan"),
        ("rowSpacingMeters", ".inf"),
        ("colSpacingMeters", ".nan"),
        ("colSpacingMeters", ".inf"),
    ],
)
def test_checkerboard_loader_rejects_nonfinite_spacing(
    tmp_path: Path, field: str, value: str
) -> None:
    """棋盘格行列间距必须是有限正数。"""
    fields = _checkerboard_yaml_fields()
    fields[field] = value
    path = tmp_path / "checkerboard-spacing.yaml"
    _write_checkerboard_yaml(path, fields)
    with pytest.raises(ValueError, match="有限正数"):
        load_calibration_target(str(path))


def test_checkerboard_loader_rejects_unsupported_target_type(
    tmp_path: Path,
) -> None:
    """目标类型不属于支持集合时应拒绝加载。"""
    fields = _checkerboard_yaml_fields()
    fields["target_type"] = "circle"
    path = tmp_path / "unsupported-target.yaml"
    _write_checkerboard_yaml(path, fields)
    with pytest.raises(ValueError, match="aprilgrid 或 checkerboard"):
        load_calibration_target(str(path))
