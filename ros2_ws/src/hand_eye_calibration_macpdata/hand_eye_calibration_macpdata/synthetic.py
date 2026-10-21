"""按鱼眼模型渲染 AprilGrid / 棋盘格的合成图，供单测和离线自检使用（不用于实机）。"""

import cv2
import numpy as np

from fastumi_bringup.transforms import invert
from hand_eye_calibration_macpdata.board import TAG_FAMILIES, AprilGridSpec, CheckerboardSpec


def board_texture(spec, pixels_per_meter):
    """板正面看到的打印图（mono8，行 0 在上）及其在 board 系中的范围。

    返回 (image, x_min, y_min, width_m, height_m, y_up)。y_up 为 True 时图像行 0
    对应 y 最大处（AprilGrid，相机在 +z 侧看到 y 向上）；否则行 0 对应 y_min
    （棋盘格，相机在 -z 侧看到 y 向下）。两者都使相机看到不镜像的正面。
    """
    if isinstance(spec, AprilGridSpec):
        return _aprilgrid_texture(spec, pixels_per_meter)
    if isinstance(spec, CheckerboardSpec):
        return _checkerboard_texture(spec, pixels_per_meter)
    raise TypeError(f"不支持的标定板类型 {type(spec).__name__}")


def _aprilgrid_texture(spec, ppm):
    """Kalibr AprilGrid 样式：标签间隙交点处有边长为间隙的小黑方块，外留白一个间隙。

    Kalibr 标签相对 OpenCV generateImageMarker 的图像旋转 180°（与检测端角点重排
    [1, 0, 3, 2] 对应）。
    """
    gap = spec.tag_size_m * spec.tag_spacing
    margin = max(gap, 0.01) * 2.0
    width_m, height_m = spec.size_m
    x_min, y_min = -margin, -margin
    total_w, total_h = width_m + 2 * margin, height_m + 2 * margin
    image = np.full((int(round(total_h * ppm)), int(round(total_w * ppm))), 255, np.uint8)

    def to_pixel(x, y):
        return int(round((x - x_min) * ppm)), int(round((y_min + total_h - y) * ppm))

    dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILIES[spec.tag_family])
    side = int(round(spec.tag_size_m * ppm))
    for tag_id in range(spec.tag_count):
        row, column = divmod(tag_id, spec.tag_cols)
        x0, y0 = column * spec.pitch_m, row * spec.pitch_m
        left, top = to_pixel(x0, y0 + spec.tag_size_m)
        marker = np.rot90(cv2.aruco.generateImageMarker(dictionary, tag_id, side, borderBits=1), 2)
        image[top:top + side, left:left + side] = marker
    if gap > 0.0:
        for row in range(spec.tag_rows + 1):
            for column in range(spec.tag_cols + 1):
                x0, y0 = column * spec.pitch_m - gap, row * spec.pitch_m - gap
                left, top = to_pixel(x0, y0 + gap)
                right, bottom = to_pixel(x0 + gap, y0)
                image[top:bottom, left:right] = 0
    return image, x_min, y_min, total_w, total_h, True


def _checkerboard_texture(spec, ppm):
    """OpenCV 棋盘格：左上外圈方格为黑，外留白一格。"""
    dx, dy = spec.col_spacing_m, spec.row_spacing_m
    columns, rows = spec.target_cols + 1, spec.target_rows + 1
    x_min, y_min = -2 * dx, -2 * dy
    total_w, total_h = (columns + 2) * dx, (rows + 2) * dy
    image = np.full((int(round(total_h * ppm)), int(round(total_w * ppm))), 255, np.uint8)
    for row in range(rows):
        for column in range(columns):
            if (row + column) % 2:
                continue
            x0, y0 = (column - 1) * dx, (row - 1) * dy
            left, top = int(round((x0 - x_min) * ppm)), int(round((y0 - y_min) * ppm))
            right, bottom = int(round((x0 + dx - x_min) * ppm)), int(round((y0 + dy - y_min) * ppm))
            image[top:bottom, left:right] = 0
    return image, x_min, y_min, total_w, total_h, False


def _surround_texture(size_m, pixels_per_meter, seed=7):
    """板周围桌面的固定随机纹理（平滑噪声），使场景随视角变化。"""
    pixels = int(round(size_m * pixels_per_meter))
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (pixels, pixels), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 2.0)


def render_board(spec, camera, camera_from_board, *, pixels_per_meter=3000,
                 surround_m=2.0, occluder=None, supersample=2):
    """渲染 ^cam T_board 位姿下的板图像（BGR uint8，camera 分辨率）。

    板位于边长 surround_m 的纹理平面中央；射线不与平面相交处为黑色（类似鱼眼暗角）。
    occluder 为可选 mono8 图像（非零像素以该灰度覆盖），模拟相对相机固定的夹爪。
    先按 supersample 倍分辨率渲染再面积下采样，近似镜头积分，避免远处小标签混叠。
    """
    if supersample > 1:
        # 像素中心对齐：高分辨率像素 j 的中心对应低分辨率 (j + 0.5) / s - 0.5。
        K = camera.K.copy()
        K[0, 0] *= supersample
        K[1, 1] *= supersample
        K[0, 2] = supersample * (K[0, 2] + 0.5) - 0.5
        K[1, 2] = supersample * (K[1, 2] + 0.5) - 0.5
        fine = type(camera)(K=K, D=camera.D, width=camera.width * supersample,
                            height=camera.height * supersample)
        gray = render_board(spec, fine, camera_from_board, pixels_per_meter=pixels_per_meter,
                            surround_m=surround_m, supersample=1)[:, :, 0]
        gray = cv2.resize(gray, (camera.width, camera.height), interpolation=cv2.INTER_AREA)
        if occluder is not None:
            gray = np.where(occluder > 0, occluder, gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    texture, x_min, y_min, width_m, height_m, y_up = board_texture(spec, pixels_per_meter)
    surround_ppm = 400.0
    surround = _surround_texture(surround_m, surround_ppm)

    u, v = np.meshgrid(np.arange(camera.width, dtype=np.float64),
                       np.arange(camera.height, dtype=np.float64))
    pixels = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
    rays = cv2.fisheye.undistortPoints(pixels, camera.K, camera.D).reshape(-1, 2)
    rays = np.concatenate([rays, np.ones((len(rays), 1))], axis=1)
    # undistortPoints 只对入射角 < 90° 有意义；按畸变后角度 θd(θ=85°) 截断视场。
    k1, k2, k3, k4 = camera.D.reshape(-1)
    theta_max = np.radians(85.0)
    theta_d_max = theta_max * (1 + k1 * theta_max ** 2 + k2 * theta_max ** 4 +
                               k3 * theta_max ** 6 + k4 * theta_max ** 8)
    theta_d = np.hypot((u.ravel() - camera.K[0, 2]) / camera.K[0, 0],
                       (v.ravel() - camera.K[1, 2]) / camera.K[1, 1])
    forward = theta_d < theta_d_max

    board_from_camera = invert(camera_from_board)
    origin = board_from_camera[:3, 3]
    directions = rays @ board_from_camera[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = -origin[2] / directions[:, 2]
    points = origin + directions * scale[:, None]
    valid = forward & (scale > 0)
    shape = (camera.height, camera.width)

    board_x = ((points[:, 0] - x_min) * pixels_per_meter).astype(np.float32).reshape(shape)
    rows = (y_min + height_m - points[:, 1]) if y_up else (points[:, 1] - y_min)
    board_y = (rows * pixels_per_meter).astype(np.float32).reshape(shape)
    board_gray = cv2.remap(texture, board_x, board_y, cv2.INTER_AREA,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    offset = surround_m / 2.0
    center_x, center_y = x_min + width_m / 2, y_min + height_m / 2
    surround_x = ((points[:, 0] - center_x + offset) * surround_ppm).astype(np.float32)
    surround_y = ((points[:, 1] - center_y + offset) * surround_ppm).astype(np.float32)
    surround_gray = cv2.remap(surround, surround_x.reshape(shape), surround_y.reshape(shape),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside_board = ((points[:, 0] >= x_min) & (points[:, 0] <= x_min + width_m) &
                    (points[:, 1] >= y_min) & (points[:, 1] <= y_min + height_m)).reshape(shape)
    gray = np.where(inside_board, board_gray, surround_gray)
    gray[~valid.reshape(shape)] = 0
    if occluder is not None:
        gray = np.where(occluder > 0, occluder, gray)
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def gripper_occluder(width, height):
    """相对相机固定的两指夹爪示意：从画面底边伸入的两块带纹理灰色区域（mono8）。"""
    occluder = np.zeros((height, width), dtype=np.uint8)
    finger_width, finger_height = int(width * 0.1), int(height * 0.28)
    rng = np.random.default_rng(3)
    for left in (int(width * 0.3), int(width * 0.6)):
        patch = rng.integers(40, 90, (finger_height, finger_width), dtype=np.uint8)
        occluder[height - finger_height:, left:left + finger_width] = patch
    return occluder


def look_at(camera_position, target, up_hint=(0.0, -1.0, 0.0)):
    """返回 ^world T_cam：光轴 z 指向 target，x 尽量与 up_hint 叉乘方向一致。"""
    z = np.asarray(target, dtype=float) - np.asarray(camera_position, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up_hint, dtype=float), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross((1.0, 0.0, 0.0), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    transform = np.eye(4)
    transform[:3, :3] = np.stack([x, y, z], axis=1)
    transform[:3, 3] = camera_position
    return transform
