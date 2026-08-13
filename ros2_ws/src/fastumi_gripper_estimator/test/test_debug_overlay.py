"""测试夹爪标记鱼眼姿态调试叠加的投影、颜色和异常处理。"""

from types import SimpleNamespace

import cv2
from fastumi_gripper_estimator.estimator import MarkerPose, OpennessEstimate
from fastumi_gripper_estimator.gripper_openness_node import GripperOpennessNode
import numpy as np


# 合成调试图使用的鱼眼相机内参。
CAMERA_MATRIX = np.asarray(
    [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
# 合成调试图使用的四个 equidistant 畸变系数。
FISHEYE_DISTORTION = np.asarray([0.08, -0.017, 0.0046, -0.0031])


class DebugDrawer:
    """不初始化 ROS 节点而复用其纯 OpenCV 调试绘制方法。"""

    _draw_debug_image = GripperOpennessNode._draw_debug_image
    _draw_marker_pose = GripperOpennessNode._draw_marker_pose
    _project_marker_axes = GripperOpennessNode._project_marker_axes
    _rotation_vector_to_rpy_degrees = staticmethod(
        GripperOpennessNode._rotation_vector_to_rpy_degrees
    )

    def __init__(self) -> None:
        """提供调试绘制所需的最小估计器相机和标签配置。"""
        self._estimator = SimpleNamespace(
            camera_matrix=CAMERA_MATRIX,
            distortion_coefficients=FISHEYE_DISTORTION,
            marker_size_mm=16.0,
            left_marker_id=7,
            right_marker_id=42,
        )


def make_estimate() -> OpennessEstimate:
    """创建包含两枚有效相机相对位姿的合成开度估计。"""
    left_pose = MarkerPose((0.1, -0.03, 0.02), (-25.0, 6.0, 185.0))
    right_pose = MarkerPose((-0.05, 0.04, -0.02), (28.0, 5.0, 188.0))
    return OpennessEstimate(
        openness=0.42,
        raw_openness=0.44,
        distance_mm=53.2,
        left_center=(250.0, 330.0),
        right_center=(390.0, 332.0),
        left_pose=left_pose,
        right_pose=right_pose,
        roi=(100, 280, 540, 400),
    )


def test_debug_overlay_projects_both_configured_tags_and_axes(monkeypatch) -> None:
    """验证两枚标签使用实际 ID、标记边长和规定的 BGR 轴颜色。"""
    drawer = DebugDrawer()
    projected_object_points = []
    drawn_lines = []
    labels = []

    def fake_project_points(object_points, rvec, tvec, camera_matrix, distortion):
        """记录鱼眼投影输入并返回可绘制的四个像素点。"""
        projected_object_points.append(
            (object_points.copy(), rvec.copy(), tvec.copy(), camera_matrix, distortion)
        )
        x_offset = 20.0 * len(projected_object_points)
        return np.asarray(
            [[200.0 + x_offset, 300.0], [230.0 + x_offset, 300.0],
             [200.0 + x_offset, 270.0], [190.0 + x_offset, 315.0]],
            dtype=np.float64,
        ).reshape(4, 1, 2), None

    def fake_line(image, start, end, color, thickness):
        """记录坐标轴和已有中心连接线的绘制参数。"""
        drawn_lines.append((start, end, color, thickness))
        return image

    def fake_put_text(image, label, *args):
        """记录叠加文本，避免依赖字体渲染的像素细节。"""
        labels.append(label)
        return image

    monkeypatch.setattr(cv2.fisheye, "projectPoints", fake_project_points)
    monkeypatch.setattr(cv2, "line", fake_line)
    monkeypatch.setattr(cv2, "putText", fake_put_text)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    debug_image = drawer._draw_debug_image(image, make_estimate())

    assert not np.array_equal(debug_image, image)
    assert len(projected_object_points) == 2
    for object_points, _, _, camera_matrix, distortion in projected_object_points:
        axes = object_points.reshape(4, 3)
        assert np.allclose(axes[1], (16.0, 0.0, 0.0))
        assert np.allclose(axes[2], (0.0, 16.0, 0.0))
        assert np.allclose(axes[3], (0.0, 0.0, 16.0))
        assert camera_matrix is CAMERA_MATRIX
        assert distortion is FISHEYE_DISTORTION
    axis_colors = [line[2] for line in drawn_lines[1:]]
    assert axis_colors == [(0, 0, 255), (0, 255, 0), (255, 0, 0)] * 2
    assert "tag=7" in labels
    assert "tag=42" in labels
    assert any(label.startswith("t=") for label in labels)
    assert any(label.startswith("rpy=") for label in labels)


def test_invalid_fisheye_projection_skips_axes_but_keeps_pose_text(monkeypatch) -> None:
    """验证非有限鱼眼投影不会中断调试绘制或伪造坐标轴。"""
    drawer = DebugDrawer()
    drawn_lines = []
    labels = []

    def fake_project_points(*args):
        """模拟 PnP 有效但投影结果不可用的异常图像情况。"""
        return np.full((4, 1, 2), np.nan, dtype=np.float64), None

    def fake_line(image, start, end, color, thickness):
        """记录坐标轴绘制调用。"""
        drawn_lines.append((start, end, color, thickness))
        return image

    def fake_put_text(image, label, *args):
        """记录仍应显示的标签姿态文本。"""
        labels.append(label)
        return image

    monkeypatch.setattr(cv2.fisheye, "projectPoints", fake_project_points)
    monkeypatch.setattr(cv2, "line", fake_line)
    monkeypatch.setattr(cv2, "putText", fake_put_text)

    debug_image = drawer._draw_debug_image(
        np.zeros((480, 640, 3), dtype=np.uint8), make_estimate()
    )

    assert debug_image.shape == (480, 640, 3)
    assert len(drawn_lines) == 1
    assert "tag=7" in labels
    assert "tag=42" in labels
