"""眼在手上手眼标定求解：OpenCV calibrateHandEye 初值 + 鱼眼重投影联合优化。

记号：A_i = ^base T_flange（第 i 个样本的 FK），X = ^flange T_cam（待求），
Y = ^base T_board（板固定不动，同时求解）。第 i 个样本的板位姿预测为
^cam T_board = (A_i X)⁻¹ Y，残差为板特征点按鱼眼模型投影与检测像素之差（像素）。
长度单位米，角度单位弧度（报告中为度）。
"""

from dataclasses import dataclass, field
import math

import cv2
import numpy as np
from scipy.optimize import least_squares

from fastumi_bringup.hand_eye.estimate import diversity_stats, estimate_board_in_base
from fastumi_bringup.hand_eye.transforms import invert, make_transform, rotation_angle


HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class View:
    """一个样本的输入：法兰位姿、PnP 板位姿、特征点 (N,3) 米与鱼眼像素 (N,2)。"""

    base_from_flange: np.ndarray
    camera_from_board: np.ndarray
    object_points: np.ndarray
    pixels: np.ndarray


@dataclass
class HandEyeResult:
    """求解结果；per_view_rms_px 与输入 views 一一对应（含被剔除的样本）。"""

    flange_from_camera: np.ndarray
    base_from_board: np.ndarray
    rms_px: float
    initial_rms_px: float
    per_view_rms_px: list
    inliers: list
    methods: dict
    diversity: dict
    warnings: list = field(default_factory=list)


def _pack(transform):
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    return np.concatenate([rvec.reshape(3), transform[:3, 3]])


def _unpack(vector):
    rotation, _ = cv2.Rodrigues(np.asarray(vector[:3], dtype=np.float64))
    return make_transform(rotation, vector[3:6])


def _view_residuals(camera, view, flange_from_camera, base_from_board):
    camera_from_board = invert(view.base_from_flange @ flange_from_camera) @ base_from_board
    points = view.object_points @ camera_from_board[:3, :3].T + camera_from_board[:3, 3]
    projected = camera.project(points)
    # 板转到相机后方时投影为 NaN，用大残差代替以免优化器发散。
    return np.nan_to_num(projected - view.pixels, nan=1e3).reshape(-1)


def _per_view_rms(camera, views, flange_from_camera, base_from_board):
    return [float(np.sqrt(np.mean(_view_residuals(
        camera, view, flange_from_camera, base_from_board) ** 2) * 2.0)) for view in views]


def _refine(camera, views, flange_from_camera, base_from_board, loss_scale_px):
    def residuals(parameters):
        X, Y = _unpack(parameters[:6]), _unpack(parameters[6:])
        return np.concatenate([_view_residuals(camera, view, X, Y) for view in views])

    start = np.concatenate([_pack(flange_from_camera), _pack(base_from_board)])
    solution = least_squares(residuals, start, loss="huber", f_scale=loss_scale_px,
                             x_scale="jac", max_nfev=200)
    return _unpack(solution.x[:6]), _unpack(solution.x[6:])


def initial_estimates(views):
    """OpenCV 五种方法的 ^flange T_cam 及其板位姿一致性（板在基座中的离散度）。"""
    flanges = [view.base_from_flange for view in views]
    boards = [view.camera_from_board for view in views]
    results = {}
    for name, method in HAND_EYE_METHODS.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                [t[:3, :3] for t in flanges], [t[:3, 3] for t in flanges],
                [t[:3, :3] for t in boards], [t[:3, 3] for t in boards], method=method)
        except cv2.error:
            # 个别方法在近退化运动下数值失败（如 Tsai 的旋转归一化），其余方法照常使用。
            continue
        transform = make_transform(rotation, np.asarray(translation).reshape(3))
        if not np.all(np.isfinite(transform)):
            continue
        board = estimate_board_in_base(flanges, boards, transform)
        results[name] = {"flange_from_camera": transform, "base_from_board": board.base_from_board,
                         "board_spread_mm": board.spread_mm, "board_spread_deg": board.spread_deg}
    if not results:
        raise ValueError("calibrateHandEye 全部方法失败")
    return results


def solve_hand_eye(camera, views, *, min_views=15, min_rotation_deg=35.0,
                   min_axis_separation_deg=10.0, outlier_mad=3.0, min_outlier_px=1.0,
                   loss_scale_px=1.0):
    """求 ^flange T_cam 与 ^base T_board。

    1. 五种 calibrateHandEye 结果中取板离散度最小者为初值；
    2. 全部样本鱼眼重投影 Huber 最小二乘；
    3. 单样本 RMS 超过 median + outlier_mad·1.4826·MAD（且 > min_outlier_px）的剔除后再优化一次。
    样本数与旋转激励不足时只给警告，不中止。
    """
    if len(views) < 3:
        raise ValueError(f"至少需要 3 个样本，实际 {len(views)}")
    methods = initial_estimates(views)
    best = min(methods.values(), key=lambda item: item["board_spread_mm"])
    X, Y = best["flange_from_camera"], best["base_from_board"]
    initial_rms = _per_view_rms(camera, views, X, Y)

    X, Y = _refine(camera, views, X, Y, loss_scale_px)
    per_view = np.array(_per_view_rms(camera, views, X, Y))
    median = float(np.median(per_view))
    mad = float(np.median(np.abs(per_view - median)))
    threshold = max(median + outlier_mad * 1.4826 * mad, min_outlier_px)
    inliers = [i for i, value in enumerate(per_view) if value <= threshold]
    if len(inliers) < len(views) and len(inliers) >= 3:
        X, Y = _refine(camera, [views[i] for i in inliers], X, Y, loss_scale_px)
        per_view = np.array(_per_view_rms(camera, views, X, Y))
    else:
        inliers = list(range(len(views)))

    residuals = np.concatenate([_view_residuals(camera, views[i], X, Y) for i in inliers])
    stats = diversity_stats([views[i].base_from_flange for i in inliers])
    warnings = []
    if len(inliers) < min_views:
        warnings.append(f"有效样本 {len(inliers)} 个，少于建议的 {min_views} 个")
    if stats.rotation_span_deg < min_rotation_deg:
        warnings.append(f"旋转跨度 {stats.rotation_span_deg:.1f}° 小于 {min_rotation_deg}°")
    if stats.axis_separation_deg < min_axis_separation_deg:
        warnings.append(f"旋转轴间隔 {stats.axis_separation_deg:.1f}° 小于 "
                        f"{min_axis_separation_deg}°")
    report_methods = {
        name: {
            "flange_from_camera": item["flange_from_camera"].tolist(),
            "board_spread_mm": item["board_spread_mm"],
            "delta_to_refined_mm": float(np.linalg.norm(
                item["flange_from_camera"][:3, 3] - X[:3, 3]) * 1e3),
            "delta_to_refined_deg": math.degrees(rotation_angle(
                item["flange_from_camera"][:3, :3], X[:3, :3])),
        } for name, item in methods.items()}
    return HandEyeResult(
        flange_from_camera=X, base_from_board=Y,
        rms_px=float(np.sqrt(np.mean(residuals ** 2) * 2.0)),
        initial_rms_px=float(np.sqrt(np.mean(np.square(initial_rms)))),
        per_view_rms_px=per_view.tolist(), inliers=inliers, methods=report_methods,
        diversity={"sample_count": stats.sample_count,
                   "rotation_span_deg": stats.rotation_span_deg,
                   "axis_separation_deg": stats.axis_separation_deg},
        warnings=warnings)
