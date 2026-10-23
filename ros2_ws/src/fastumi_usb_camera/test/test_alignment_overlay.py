"""验证安装对齐叠加的参考图读取、混合与码角点误差。"""

import cv2
import numpy as np
import pytest

from fastumi_usb_camera.alignment_overlay import (
    GRAY_MASK_COLOR, blend, load_reference, render, tag_errors)


def _write_reference(tmp_path, size=(40, 30)):
    """写入左半为 mask 的 BGRA 参考图及码角点旁注。"""
    width, height = size
    bgra = np.zeros((height, width, 4), np.uint8)
    bgra[..., :3] = (10, 20, 30)
    bgra[:, : width // 2, 3] = 255
    path = tmp_path / "mask.png"
    cv2.imwrite(str(path), bgra)
    path.with_suffix(".yaml").write_text(
        "umi_gripper_mask:\n  tag_corners_px:\n    0: [[1, 1], [5, 1], [5, 5], [1, 5]]\n")
    return path


def test_blend_keeps_outside_and_mixes_inside_mask(tmp_path):
    reference = load_reference(_write_reference(tmp_path))
    frame = np.full((30, 40, 3), 200, np.uint8)
    np.testing.assert_array_equal(blend(frame, reference, 0.0), frame)
    mixed = blend(frame, reference, 0.5)
    np.testing.assert_array_equal(mixed[:, 20:], frame[:, 20:])
    np.testing.assert_array_equal(mixed[0, 0], [105, 110, 115])
    np.testing.assert_array_equal(blend(frame, reference, 1.0)[0, 0], [10, 20, 30])
    with pytest.raises(ValueError, match="尺寸"):
        blend(np.zeros((10, 10, 3), np.uint8), reference, 0.5)
    with pytest.raises(ValueError, match="alpha"):
        blend(frame, reference, 1.5)


def test_gray_mask_is_tinted_and_sidecar_corners_are_loaded(tmp_path):
    gray = np.zeros((30, 40), np.uint8)
    gray[10:20, 10:20] = 255
    path = tmp_path / "gray.png"
    cv2.imwrite(str(path), gray)
    reference = load_reference(path)
    np.testing.assert_array_equal(reference.image[15, 15], GRAY_MASK_COLOR)
    assert reference.mask[15, 15] == 1.0 and reference.mask[0, 0] == 0.0
    assert reference.tag_corners == {}
    assert len(reference.contours) == 1
    tagged = load_reference(_write_reference(tmp_path))
    np.testing.assert_array_equal(tagged.tag_corners[0][2], [5, 5])
    cv2.imwrite(str(tmp_path / "empty.png"), np.zeros((5, 5), np.uint8))
    with pytest.raises(ValueError, match="为空"):
        load_reference(tmp_path / "empty.png")


def test_tag_errors_and_render_report_corner_offsets(tmp_path):
    reference = load_reference(_write_reference(tmp_path))
    live = {0: reference.tag_corners[0] + [3.0, 4.0], 1: np.zeros((4, 2))}
    assert tag_errors(live, reference.tag_corners) == {0: pytest.approx(5.0)}
    output, errors = render(np.full((30, 40, 3), 200, np.uint8), reference, 0.5, live)
    assert output.shape == (30, 40, 3) and errors == {0: pytest.approx(5.0)}
