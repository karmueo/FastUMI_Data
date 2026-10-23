"""验证 UMI 夹爪 mask 的静态区域分割与鱼眼射线重投影。"""

import numpy as np

from extract_umi_gripper_mask import ray_remap, remap_points, segment_gripper

CAMERA = np.array([[600.0, 0, 480], [0, 600.0, 270], [0, 0, 1]])
DISTORTION = np.array([0.22, 0.5, -0.48, 0.17]).reshape(4, 1)


def test_segment_keeps_static_dark_block_touching_bottom():
    """场景随机变化、底部黑色夹爪静止：mask 覆盖夹爪，不纳入背景与悬空暗块。"""
    rng = np.random.default_rng(0)
    frames = rng.integers(90, 255, size=(40, 120, 160, 3), dtype=np.uint8)
    frames[:, 80:, 40:120] = 30                     # 静止夹爪，与底边相连
    frames[:, 90:100, 70:90] = 240                  # 夹爪上的白色码，应被填洞纳入
    frames[:, 10:30, 10:30] = 30                    # 悬空暗块，不与底边相连
    median = np.median(frames, axis=0).astype(np.uint8)
    mad = np.median(np.abs(frames.astype(np.int16) - median), axis=0).max(axis=2)
    mask = segment_gripper(median, mad)
    assert mask[85:, 45:115].all() and mask[90:100, 70:90].all()
    assert not mask[10:30, 10:30].any() and not mask[:70, :].any()


def test_same_intrinsics_remap_is_identity():
    map_x, map_y = ray_remap(CAMERA, DISTORTION, CAMERA, DISTORTION, (960, 540))
    grid_x, grid_y = np.meshgrid(np.arange(960), np.arange(540))
    np.testing.assert_allclose(map_x[100:440, 100:860], grid_x[100:440, 100:860], atol=0.05)
    np.testing.assert_allclose(map_y[100:440, 100:860], grid_y[100:440, 100:860], atol=0.05)


def test_principal_point_shift_moves_points_along_rays():
    """只有主点不同时，近光轴点按主点差平移，与重投影图一致。"""
    target = CAMERA.copy()
    target[:2, 2] += [75, 64]
    points = np.array([[480.0, 270.0], [500.0, 300.0]])
    moved = remap_points(points, CAMERA, DISTORTION, target, DISTORTION)
    np.testing.assert_allclose(moved, points + [75, 64], atol=1e-6)
    map_x, map_y = ray_remap(CAMERA, DISTORTION, target, DISTORTION, (960, 540))
    np.testing.assert_allclose([map_x[300 + 64, 500 + 75], map_y[300 + 64, 500 + 75]],
                               [500, 300], atol=0.05)
