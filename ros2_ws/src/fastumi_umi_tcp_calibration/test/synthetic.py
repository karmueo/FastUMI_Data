"""渲染用于离线标定回归测试的双 ArUco 鱼眼图像，来源于 jetson_dev。"""

import cv2
import numpy as np

from fastumi_umi_tcp_calibration.transforms import invert, make_transform


def render_dual_tags(
    camera,
    camera_from_dtag,
    *,
    dictionary_name="DICT_4X4_50",
    tag_ids=(0, 1),
    marker_size_m=0.016,
    tag_distance_m=0.06,
    background=100,
    pixels_per_meter=20000
):
    """渲染夹爪双 ArUco 的鱼眼合成图（BGR uint8，camera 分辨率）。

    d_tag 系定义见 dual_tag.py：ID 0 中心在 (-d/2, 0, 0)，ID 1 在 (+d/2, 0, 0)，tag 局部轴与
    d_tag 同向（+Z 指向相机）。每枚 tag 外围有宽为一个 tag 边长的白色静区，其余为灰色背景。
    只对静区投影包围盒内的像素反投影求交，避免整幅鱼眼反投影。
    """
    gray = np.full((camera.height, camera.width), background, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    side = int(round(marker_size_m * pixels_per_meter))
    half_zone = marker_size_m  # 静区外边界半宽（米）
    zone = int(round(2 * half_zone * pixels_per_meter))
    for tag_id, offset in zip(tag_ids, (-tag_distance_m / 2.0, tag_distance_m / 2.0)):
        texture = np.full((zone, zone), 255, dtype=np.uint8)
        start = (zone - side) // 2
        texture[start : start + side, start : start + side] = cv2.aruco.generateImageMarker(
            dictionary, tag_id, side, borderBits=1
        )
        camera_from_tag = camera_from_dtag @ make_transform(np.eye(3), (offset, 0.0, 0.0))
        outline = np.array(
            [
                [-half_zone, -half_zone, 0],
                [half_zone, -half_zone, 0],
                [half_zone, half_zone, 0],
                [-half_zone, half_zone, 0],
            ],
            np.float64,
        )
        projected = camera.project(outline @ camera_from_tag[:3, :3].T + camera_from_tag[:3, 3])
        if not np.all(np.isfinite(projected)):
            continue
        left, top = np.floor(projected.min(axis=0)).astype(int) - 2
        right, bottom = np.ceil(projected.max(axis=0)).astype(int) + 2
        left, top = max(left, 0), max(top, 0)
        right, bottom = min(right, camera.width - 1), min(bottom, camera.height - 1)
        if right <= left or bottom <= top:
            continue
        u, v = np.meshgrid(
            np.arange(left, right + 1, dtype=np.float64),
            np.arange(top, bottom + 1, dtype=np.float64),
        )
        pixels = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
        rays = cv2.fisheye.undistortPoints(pixels, camera.K, camera.D).reshape(-1, 2)
        rays = np.concatenate([rays, np.ones((len(rays), 1))], axis=1)
        tag_from_camera = invert(camera_from_tag)
        origin = tag_from_camera[:3, 3]
        directions = rays @ tag_from_camera[:3, :3].T
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = -origin[2] / directions[:, 2]
        points = origin + directions * scale[:, None]
        shape = u.shape
        map_x = ((points[:, 0] + half_zone) * pixels_per_meter).astype(np.float32).reshape(shape)
        map_y = ((half_zone - points[:, 1]) * pixels_per_meter).astype(np.float32).reshape(shape)
        patch = cv2.remap(
            texture, map_x, map_y, cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        inside = (
            (scale > 0) & (np.abs(points[:, 0]) <= half_zone) & (np.abs(points[:, 1]) <= half_zone)
        ).reshape(shape)
        region = gray[top : bottom + 1, left : right + 1]
        region[inside] = patch[inside]
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
