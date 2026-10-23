#!/usr/bin/env python3
"""从 UMI 训练图像提取夹爪（含两枚手指 ArUco 码）mask，并重投影到 RM75 末端相机像素系。

用途：重装末端相机时，把该 mask 叠加到实时画面（fastumi_usb_camera 的
``alignment_overlay``），调整相机直到实拍夹爪与 mask 重合，使末端相机相对夹爪的位姿
接近训练时 UMI 相机相对夹爪的位姿。

做法：
1. 从训练 HDF5 抽取夹爪张开（``qpos[:,7] >= --min-gripper``）的帧；夹爪相对相机静止而场景
   变化，逐像素中位数即"干净"的夹爪图像，中位绝对偏差（MAD）小的暗色区域即夹爪本体。
2. 本体（暗且 MAD 小）并上橙色指面，开闭运算后只保留与图像底边相连的连通域并填洞。
3. 两侧都是 KB4 鱼眼：末端像素经末端内参去畸变得到射线，再经 UMI 内参投影回 UMI 像素采样。
   相机相对夹爪位姿一致时，同一射线指向同一夹爪点，所以重投影后的 mask 与实拍应完全重合。
4. 手指码角点在每帧全分辨率原图上检测后取中位数，同样重投影，写入旁注 YAML 供精细对齐。

输出（默认 ros2_ws/src/fastumi_usb_camera/config/umi_gripper_mask.png）：
- ``*.png``：BGRA，BGR 为重投影后的 UMI 中位图，alpha 为 mask（0/255），尺寸为末端相机分辨率；
- ``*.yaml``：手指码参考角点（末端像素系，id → 4×2，左上起顺时针）与来源信息；
- ``*_preview.jpg``：人工检查用预览。

用法（ros2_ws/.venv-numpy1）::

    python extract_umi_gripper_mask.py --training-dataset dataset/processed/jingbao_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from compute_link7_correction import DICTIONARY, LEFT_ID, RIGHT_ID, load_fisheye_camera

# 统计在半分辨率上进行，控制内存（300 帧 × 960×540×3 约 470 MB）。
ANALYSIS_SCALE = 0.5
# 本体判据：中位图亮度上限与 MAD 上限（0–255 灰度）。
BODY_MAX_VALUE = 75
BODY_MAX_MAD = 12.0
# 橙色指面判据（OpenCV HSV，H∈[0,180)）。
ORANGE_MAX_HUE, ORANGE_MIN_SATURATION, ORANGE_MIN_VALUE = 14, 150, 100
# 与底边相连的连通域至少占分析图像的比例。
MIN_COMPONENT_FRACTION = 0.004


def open_gripper_frames(dataset: Path, min_gripper: float, max_frames: int, seed: int):
    """均匀跨 episode 随机抽取夹爪张开帧，逐帧产出 (源文件, RGB[H,W,3] uint8)。"""
    import h5py

    files = sorted(dataset.glob("records/**/episodes/*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no episode HDF5 under {dataset}/records")
    per_file = max(1, int(np.ceil(max_frames / len(files))))
    rng = np.random.default_rng(seed)
    produced = 0
    for path in files:
        with h5py.File(path, "r") as episode:
            encoding = episode.attrs["image_encoding"]
            if (encoding.decode() if isinstance(encoding, bytes) else encoding) != "rgb8":
                raise ValueError(f"{path} images must be rgb8")
            candidates = np.flatnonzero(episode["observations/qpos"][:, 7] >= min_gripper)
            if candidates.size == 0:
                continue
            chosen = rng.choice(candidates, size=min(per_file, candidates.size), replace=False)
            images = episode["observations/images/front"]
            for index in np.sort(chosen):
                yield path, np.asarray(images[index])
                produced += 1
                if produced >= max_frames:
                    return


def segment_gripper(median_rgb: np.ndarray, mad: np.ndarray) -> np.ndarray:
    """由中位图与逐像素 MAD（取三通道最大）分割夹爪，返回 uint8 mask（1 为夹爪）。"""
    hue, saturation, value = cv2.split(cv2.cvtColor(median_rgb, cv2.COLOR_RGB2HSV))
    body = (value < BODY_MAX_VALUE) & (mad < BODY_MAX_MAD)
    orange = ((hue < ORANGE_MAX_HUE) & (saturation > ORANGE_MIN_SATURATION)
              & (value > ORANGE_MIN_VALUE))
    mask = (body | orange).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    height = mask.shape[0]
    keep = np.zeros_like(mask)
    for label in range(1, count):
        touches_bottom = stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT] >= height
        if touches_bottom and stats[label, cv2.CC_STAT_AREA] >= MIN_COMPONENT_FRACTION * mask.size:
            keep[labels == label] = 1
    # 从图像外缘灌水，未被灌到的背景即夹爪内部的孔（白色码区等）。
    flooded = np.pad(keep, 1) * 255
    cv2.floodFill(flooded, np.zeros((flooded.shape[0] + 2, flooded.shape[1] + 2), np.uint8),
                  (0, 0), 255)
    return keep | (flooded[1:-1, 1:-1] == 0).astype(np.uint8)


def ray_remap(source_camera, source_distortion, target_camera, target_distortion,
              size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """返回 cv2.remap 用的 map_x/map_y[H,W]：目标像素 → 同一射线在源鱼眼图像中的像素。

    ``size`` 为目标图像 (宽, 高)。
    """
    width, height = size
    grid = np.stack(np.meshgrid(np.arange(width, dtype=np.float64),
                                np.arange(height, dtype=np.float64)), axis=-1).reshape(-1, 1, 2)
    rays = cv2.fisheye.undistortPoints(grid, target_camera, target_distortion).reshape(-1, 2)
    points = np.concatenate([rays, np.ones((rays.shape[0], 1))], axis=1).reshape(-1, 1, 3)
    source, _ = cv2.fisheye.projectPoints(points, np.zeros(3), np.zeros(3),
                                          source_camera, source_distortion)
    source = source.reshape(height, width, 2).astype(np.float32)
    return source[..., 0], source[..., 1]


def remap_points(points: np.ndarray, source_camera, source_distortion,
                 target_camera, target_distortion) -> np.ndarray:
    """把源鱼眼像素点 [N,2] 沿射线映射到目标鱼眼像素点 [N,2]。"""
    rays = cv2.fisheye.undistortPoints(points.reshape(-1, 1, 2).astype(np.float64),
                                       source_camera, source_distortion).reshape(-1, 2)
    rays3 = np.concatenate([rays, np.ones((rays.shape[0], 1))], axis=1).reshape(-1, 1, 3)
    target, _ = cv2.fisheye.projectPoints(rays3, np.zeros(3), np.zeros(3),
                                          target_camera, target_distortion)
    return target.reshape(-1, 2)


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--umi-camera", type=Path,
                        default=root / "config/calibration/UmiCalib.yaml")
    parser.add_argument("--robot-camera", type=Path, default=root /
                        "ros2_ws/src/fastumi_usb_camera/config/EndEffectorCalib.yaml")
    parser.add_argument("--min-gripper", type=float, default=0.9,
                        help="只用开度不低于该值的帧（对齐时末端夹爪命令到 1.0 全开）")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=root /
                        "ros2_ws/src/fastumi_usb_camera/config/umi_gripper_mask.png")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    umi_camera, umi_distortion = load_fisheye_camera(args.umi_camera)
    robot_camera, robot_distortion = load_fisheye_camera(args.robot_camera)
    robot_size = tuple(int(value) for value in yaml.safe_load(
        args.robot_camera.read_text(encoding="utf-8"))["cam0"]["resolution"])

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY),
                                       cv2.aruco.DetectorParameters())
    small, corners, sources, full_size = [], {LEFT_ID: [], RIGHT_ID: []}, set(), None
    for path, rgb in open_gripper_frames(args.training_dataset, args.min_gripper,
                                         args.max_frames, args.seed):
        full_size = (rgb.shape[1], rgb.shape[0])
        sources.add(path.relative_to(args.training_dataset).as_posix())
        small.append(cv2.resize(rgb, None, fx=ANALYSIS_SCALE, fy=ANALYSIS_SCALE,
                                interpolation=cv2.INTER_AREA))
        found, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if ids is not None:
            for corner, marker_id in zip(found, ids.ravel()):
                if int(marker_id) in corners:
                    corners[int(marker_id)].append(corner.reshape(4, 2))
    if len(small) < 20:
        raise RuntimeError(f"only {len(small)} open-gripper frames found; lower --min-gripper")
    frame_count = len(small)
    stack = np.stack(small)
    median = np.median(stack, axis=0).astype(np.uint8)
    mad = np.median(np.abs(stack.astype(np.int16) - median), axis=0).max(axis=2)
    del stack, small
    mask = segment_gripper(median, mad)

    # 回到 UMI 全分辨率后沿射线重投影到末端相机像素系。
    median_full = cv2.resize(median, full_size, interpolation=cv2.INTER_LINEAR)
    mask_full = cv2.resize(mask * 255, full_size, interpolation=cv2.INTER_LINEAR)
    map_x, map_y = ray_remap(umi_camera, umi_distortion, robot_camera, robot_distortion,
                             robot_size)
    reference = cv2.remap(median_full, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    alpha = cv2.remap(mask_full, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    alpha = np.where(alpha >= 128, 255, 0).astype(np.uint8)

    tags = {}
    for marker_id, samples in corners.items():
        if len(samples) < 10:
            raise RuntimeError(f"finger tag {marker_id} detected in only {len(samples)} frames")
        tags[marker_id] = remap_points(np.median(np.stack(samples), axis=0), umi_camera,
                                       umi_distortion, robot_camera, robot_distortion)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # mask 外的颜色不参与叠加，清零以减小 PNG 体积。
    bgra = np.dstack([cv2.cvtColor(reference, cv2.COLOR_RGB2BGR) * (alpha > 0)[..., None], alpha])
    if not cv2.imwrite(str(args.output), bgra):
        raise RuntimeError(f"failed to write {args.output}")
    document = {"umi_gripper_mask": {
        "image": args.output.name,
        "size": list(robot_size),
        "tag_corners_px": {int(key): value.round(2).tolist() for key, value in tags.items()},
        "tag_detections": {int(key): len(value) for key, value in corners.items()},
        "frames": frame_count,
        "min_gripper": args.min_gripper,
        "umi_camera": str(args.umi_camera),
        "robot_camera": str(args.robot_camera),
        "training_dataset": str(args.training_dataset),
        "episodes": len(sources),
    }}
    sidecar = args.output.with_suffix(".yaml")
    sidecar.write_text(
        "# 由 extract_umi_gripper_mask.py 生成：UMI 夹爪 mask 重投影到末端相机像素系，\n"
        "# tag_corners_px 为手指码角点（id → 左上起顺时针 4 点，像素）。\n"
        + yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    preview = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)
    preview[alpha == 0] //= 3
    for quad in tags.values():
        cv2.polylines(preview, [np.round(quad).astype(np.int32)], True, (0, 0, 255), 2)
    cv2.imwrite(str(args.output.with_name(args.output.stem + "_preview.jpg")), preview)
    print(f"frames: {frame_count} from {len(sources)} episodes, "
          f"tag detections: {document['umi_gripper_mask']['tag_detections']}, "
          f"mask coverage: {np.count_nonzero(alpha) / alpha.size:.1%}")
    print(f"wrote {args.output}, {sidecar}")


if __name__ == "__main__":
    main()
