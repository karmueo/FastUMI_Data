"""验证限时实机试验的腕部抓球目标检测。"""

from types import SimpleNamespace

import cv2
import numpy as np
from sensor_msgs.msg import Image

from live_guarded_trial import TrialMonitor, ball_center, outside_trial_region, target_lost_during_control


def _image(encoding, *, with_ball=True):
    pixels = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(pixels, (80, 100), 30, (180, 180, 210), -1)  # 粉色托盘。
    cv2.circle(pixels, (30, 45), 18, (120, 30, 0), -1)  # 蓝色背景干扰。
    if with_ball:
        cv2.circle(pixels, (220, 70), 20, (240, 240, 240), -1)
        cv2.circle(pixels, (213, 68), 7, (255, 0, 0), -1)
        cv2.circle(pixels, (227, 76), 5, (0, 100, 255), -1)
    if encoding == "rgb8":
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    message = Image()
    message.height, message.width, message.step = 240, 320, 960
    message.encoding = encoding
    message.data = pixels.tobytes()
    return message


def test_ball_center_tracks_blue_patch_in_bgr_and_rgb():
    for encoding in ("bgr8", "rgb8"):
        np.testing.assert_allclose(ball_center(_image(encoding)), [220, 70], atol=1.0)
    assert ball_center(_image("bgr8", with_ball=False)) is None


def test_one_missed_detection_keeps_last_valid_ball_until_timeout():
    monitor = SimpleNamespace(image_gaps=[], active_image_gaps=[], active_ball_misses=0,
                              first_command_at=None, control_end_at=None,
                              last_image_at=float("-inf"),
                              image_count=0, ball=None, ball_at=float("-inf"), image_size=None)
    TrialMonitor.on_image(monitor, _image("bgr8"))
    last_valid_at = monitor.ball_at
    monitor.first_command_at = last_valid_at
    TrialMonitor.on_image(monitor, _image("bgr8", with_ball=False))
    np.testing.assert_allclose(monitor.ball, [220, 70], atol=1.0)
    assert monitor.ball_at == last_valid_at
    assert monitor.active_ball_misses == 1
    assert not target_lost_during_control(True, last_valid_at, monitor.ball,
                                          monitor.ball_at, last_valid_at + 0.29)
    assert target_lost_during_control(True, last_valid_at, monitor.ball,
                                      monitor.ball_at, last_valid_at + 0.31)
    active_gaps = len(monitor.active_image_gaps)
    monitor.control_end_at = last_valid_at + 0.31
    TrialMonitor.on_image(monitor, _image("bgr8", with_ball=False))
    assert len(monitor.active_image_gaps) == active_gaps
    assert monitor.active_ball_misses == 1


def test_preflight_waits_for_all_fresh_inputs_together():
    monitor = SimpleNamespace(joints=np.zeros(7), joint_at=2.0, gripper=1.0,
                              gripper_at=2.0, valid=True, valid_at=2.0,
                              ball=(900, 400), ball_at=2.0)
    assert TrialMonitor.inputs_ready(monitor, 2.29)
    monitor.ball_at = 1.69
    assert not TrialMonitor.inputs_ready(monitor, 2.0)
    monitor.ball_at = 2.0
    monitor.valid = False
    assert not TrialMonitor.inputs_ready(monitor, 2.0)


def test_real_trial_region_limits_distance_and_vertical_rise():
    home = np.array([0.30, 0.0, 0.34])
    assert not outside_trial_region(home + [0.01, 0.01, -0.01], home)
    assert not outside_trial_region(home + [-0.021, 0.0, 0.0], home)
    assert outside_trial_region(home + [-0.041, 0.0, 0.0], home)
    assert outside_trial_region(home + [0.0, 0.0, 0.026], home)
    assert outside_trial_region(home + [0.0, 0.041, 0.0], home)
    assert outside_trial_region(home + [np.nan, 0.0, 0.0], home)
    assert not outside_trial_region(home + [0.0, 0.06, 0.0], home, 0.08)
    assert outside_trial_region(home + [0.0, 0.081, 0.0], home, 0.08)


def test_startup_video_gap_cannot_stop_a_trial_before_the_first_command():
    assert not target_lost_during_control(True, None, None, 1.0, 3.0)
    assert target_lost_during_control(True, 2.0, None, 2.0, 2.1)
    assert target_lost_during_control(True, 2.0, (900, 400), 2.0, 2.31)
    assert not target_lost_during_control(True, 2.0, (900, 400), 2.0, 2.29)
    assert not target_lost_during_control(False, 2.0, None, 2.0, 3.0)
