"""验证彩色抓球目标定位与夹爪闭合区域判断。"""

import cv2
import numpy as np

from dp_infer.visual_guard import ball_center_bgr, ball_near_jaws


def _scene(ball_xy):
    image = np.full((960, 1280, 3), (100, 130, 145), dtype=np.uint8)
    cv2.circle(image, (500, 480), 170, (140, 140, 180), -1)  # 粉色托盘。
    cv2.circle(image, (110, 170), 40, (130, 40, 10), -1)  # 蓝色干扰物。
    if ball_xy is not None:
        cv2.circle(image, ball_xy, 55, (245, 245, 245), -1)
        cv2.circle(image, (ball_xy[0] - 18, ball_xy[1]), 19, (255, 0, 0), -1)
        cv2.circle(image, (ball_xy[0] + 23, ball_xy[1] + 8), 13, (0, 105, 250), -1)
    return image


def test_detector_rejects_colored_background_and_tracks_ball_at_gripper():
    assert ball_center_bgr(_scene(None)) is None
    for location, close in (((900, 350), False), ((650, 650), True)):
        image = _scene(location)
        center = ball_center_bgr(image)
        np.testing.assert_allclose(center, location, atol=5)
        assert ball_near_jaws(center, image.shape[1], image.shape[0]) is close


def test_close_guard_rejects_missing_or_invalid_ball_position():
    assert not ball_near_jaws(None, 1280, 960)
    assert not ball_near_jaws((np.nan, 640), 1280, 960)


def test_small_blue_orange_reflection_near_jaws_is_not_the_ball():
    image = _scene((900, 350))
    cv2.circle(image, (555, 680), 18, (245, 245, 245), -1)
    cv2.circle(image, (548, 677), 6, (255, 0, 0), -1)
    cv2.circle(image, (563, 684), 5, (0, 105, 250), -1)
    center = ball_center_bgr(image)
    np.testing.assert_allclose(center, (900, 350), atol=5)
    assert not ball_near_jaws(center, image.shape[1], image.shape[0])


def test_dark_frame_still_requires_white_ball_and_both_colored_markers():
    image = np.full((960, 1280, 3), (25, 30, 35), dtype=np.uint8)
    assert ball_center_bgr(image) is None
    cv2.circle(image, (900, 380), 55, (140, 140, 140), -1)
    assert ball_center_bgr(image) is None
    cv2.circle(image, (880, 380), 19, (140, 20, 10), -1)
    assert ball_center_bgr(image) is None
    cv2.circle(image, (925, 390), 14, (0, 100, 220), -1)
    np.testing.assert_allclose(ball_center_bgr(image), [900, 380], atol=5)


def test_bright_background_uses_bounded_marker_pair_only():
    image = np.full((960, 1280, 3), (190, 190, 190), dtype=np.uint8)
    cv2.rectangle(image, (400, 650), (600, 950), (0, 100, 220), -1)
    cv2.circle(image, (550, 700), 20, (180, 20, 10), -1)
    assert ball_center_bgr(image) is None
    cv2.circle(image, (620, 340), 55, (245, 245, 245), -1)
    cv2.circle(image, (600, 340), 19, (180, 20, 10), -1)
    cv2.circle(image, (645, 350), 14, (0, 100, 220), -1)
    np.testing.assert_allclose(ball_center_bgr(image), [620, 340], atol=7)


def test_wide_ball_markers_remain_distinct_from_gripper_jaws():
    image = np.full((960, 1280, 3), (190, 190, 190), dtype=np.uint8)
    cv2.ellipse(image, (400, 420), (63, 20), 0, 0, 360, (180, 20, 10), -1)
    cv2.ellipse(image, (450, 380), (62, 40), 0, 0, 360, (0, 100, 220), -1)
    np.testing.assert_allclose(ball_center_bgr(image), [425, 400], atol=8)
