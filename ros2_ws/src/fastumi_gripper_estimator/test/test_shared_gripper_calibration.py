"""验证旧估计节点读取夹爪标定包生成的相机和范围配置。"""

from pathlib import Path

from fastumi_gripper_estimator.gripper_openness_node import GripperOpennessNode
import pytest
import rclpy
import yaml


def _write_calibration_files(tmp_path: Path) -> tuple[Path, Path]:
    """创建尺寸、标签和距离端点都明确的测试标定文件。"""
    camera_path = tmp_path / "calib.yaml"
    camera_path.write_text(
        yaml.safe_dump({"cam0": {
            "distortion_model": "fisheye",
            "intrinsics": [600.0, 600.0, 960.0, 540.0],
            "distortion_coeffs": [0.1, 0.2, -0.1, 0.01],
            "resolution": [1920, 1080],
        }}), encoding="utf-8"
    )
    range_path = tmp_path / "calibration.yaml"
    range_path.write_text(
        yaml.safe_dump({"gripper_calibration": {
            "marker_size_mm": 18.0,
            "dictionary_name": "DICT_4X4_50",
            "left_finger_tag_id": 3,
            "right_finger_tag_id": 4,
            "min_marker_dist_mm": 41.0,
            "max_marker_dist_mm": 122.0,
            "resolution": [1920, 1080],
            "crop_reference": {
                "left_x": 300, "top_y": 400,
                "right_x": 1500, "bottom_y": 800,
            },
            "total_valid_frames": 30,
        }}), encoding="utf-8"
    )
    return camera_path, range_path


def test_estimator_uses_shared_calibration_values(tmp_path: Path) -> None:
    """确认生成的夹爪标定覆盖旧配置中的范围、标签和检测区域。"""
    camera_path, range_path = _write_calibration_files(tmp_path)
    rclpy.init(args=[
        "--ros-args", "-p", f"camera_calibration_path:={camera_path}",
        "-p", f"gripper_calibration_path:={range_path}",
    ])
    node = None
    try:
        node = GripperOpennessNode()
        estimator = node._estimator
        assert estimator.closed_distance_mm == pytest.approx(41.0)
        assert estimator.open_distance_mm == pytest.approx(122.0)
        assert estimator.marker_size_mm == pytest.approx(18.0)
        assert (estimator.left_marker_id, estimator.right_marker_id) == (3, 4)
        assert estimator.roi_ratios == pytest.approx((
            250 / 1920, 350 / 1080, 1550 / 1920, 850 / 1080,
        ))
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
