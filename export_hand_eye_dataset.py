#!/usr/bin/env python3
"""导出 keyboard_hand_eye_capture 录制的手眼标定 MCAP，并求解眼在手上外参 ^Link7 T_cam。

标定板为 Kalibr AprilGrid 或棋盘格（会话信息 /hand_eye/session 中的 board）。导出时在
原始鱼眼 JPEG 上重新检测板特征点，用 OpenCV calibrateHandEye 五种方法取初值，再对全部
样本的鱼眼像素重投影误差联合优化 ^Link7 T_cam 与 ^base T_board（见
fastumi_bringup.hand_eye.solve）。同时输出鱼眼矫正后的针孔数据集，供其它工具使用。

需先 source ros2_ws/install/setup.bash（rosbag2_py、fastumi_bringup）。

启动：python export_hand_eye_dataset.py --input SESSION_DIR --output OUT_DIR
      [--prior config/calibration/T_cam_end.yaml] [--no-solve]
      [--balance 0.0] [--fov-scale 1.0] [--drop-outside] [--overwrite]
启动参数：
    --input：会话目录（含 bag/）或 bag 目录本身。
    --output：新的输出目录；已存在时需 --overwrite。
    --prior：可选外参先验（cam0.T_cam_hand），结果中报告与其差值。
    --no-solve：只导出数据集，不求解。
    --balance / --fov-scale：fisheye 矫正新内参的视场取舍，见
        cv2.fisheye.estimateNewCameraMatrixForUndistortRectify。
    --drop-outside：丢弃矫正后整块板不在画面内的样本（否则只警告）。
输入：/hand_eye/sample_{image,flange_pose,meta}、sample_rejected、session、gripper_mask。
      同一样本的消息共享同一 bag 时间戳（图像接收时间）。
输出：
    hand_eye_result.yaml：T_flange_cam（^Link7 T_cam，米）、与先验同格式的
        cam0.T_cam_hand（^cam T_Link7）、T_base_board、重投影 RMS、各方法初值、
        每样本残差、剔除样本与多样性统计。
    images/<i>.png（鱼眼矫正为针孔，i 从 0 连续）、poses.csv（制表符分隔，无表头，
    x y z qx qy qz qw，^base T_Link7，米）、intrinsics.yaml（ros__parameters 下 K/D=0/
    width/height）、samples.tsv、gripper_mask.png（若有）、README.txt。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
import yaml

from fastumi_bringup.hand_eye.board import FisheyeCamera, board_spec_from_dict, make_detector
from fastumi_bringup.hand_eye.estimate import load_prior
from fastumi_bringup.hand_eye.solve import View, solve_hand_eye
from fastumi_bringup.hand_eye.transforms import invert, rotation_angle


TOPICS = {
    "/hand_eye/sample_image": CompressedImage,
    "/hand_eye/sample_flange_pose": PoseStamped,
    "/hand_eye/sample_meta": String,
    "/hand_eye/sample_rejected": String,
    "/hand_eye/session": String,
    "/hand_eye/gripper_mask": Image,
}


@dataclass
class Sample:
    """一个已配对样本；pose 为 (x, y, z, qx, qy, qz, qw)。"""

    index: int
    stamp_ns: int
    image: CompressedImage
    pose: tuple
    meta: dict


def _bag_path(source: Path) -> Path:
    """接受会话目录或 bag 目录。"""
    if (source / "metadata.yaml").is_file():
        return source
    if (source / "bag" / "metadata.yaml").is_file():
        return source / "bag"
    raise FileNotFoundError(f"{source} 中没有 rosbag2 metadata.yaml")


def read_session(source: Path):
    """读取会话信息、掩膜与按 bag 时间戳配对后的有效样本（按序号排序）。"""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(_bag_path(source)), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    session = None
    mask = None
    rejected = set()
    groups: dict[int, dict] = {}
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        message_type = TOPICS.get(topic)
        if message_type is None:
            continue
        message = deserialize_message(data, message_type)
        if topic == "/hand_eye/session":
            session = json.loads(message.data)
        elif topic == "/hand_eye/gripper_mask":
            # 多次重估时以最后一次为准。
            mask = np.frombuffer(bytes(message.data), np.uint8).reshape(
                message.height, message.step)[:, :message.width].copy()
        elif topic == "/hand_eye/sample_rejected":
            rejected.add(int(json.loads(message.data)["index"]))
        else:
            groups.setdefault(stamp_ns, {})[topic] = message
    if session is None:
        raise ValueError("MCAP 中缺少 /hand_eye/session")

    samples = []
    for stamp_ns, group in groups.items():
        if len(group) != 3:
            raise ValueError(f"时间戳 {stamp_ns} 的样本消息不完整：{sorted(group)}")
        meta = json.loads(group["/hand_eye/sample_meta"].data)
        pose = group["/hand_eye/sample_flange_pose"].pose
        samples.append(Sample(
            index=int(meta["index"]), stamp_ns=stamp_ns,
            image=group["/hand_eye/sample_image"],
            pose=(pose.position.x, pose.position.y, pose.position.z, pose.orientation.x,
                  pose.orientation.y, pose.orientation.z, pose.orientation.w),
            meta=meta))
    samples = sorted((s for s in samples if s.index not in rejected), key=lambda s: s.index)
    return session, mask, samples


def rectification(camera: dict, balance: float, fov_scale: float):
    """返回 (新针孔 K, map1, map2)；输出分辨率与原图相同。"""
    K = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(camera["D"], dtype=np.float64).reshape(4, 1)
    size = (int(camera["width"]), int(camera["height"]))
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, size, np.eye(3), balance=balance, new_size=size, fov_scale=fov_scale)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, size, cv2.CV_16SC2)
    return new_K, map1, map2


def board_inside(spec, camera_from_board, new_K, size, margin_px=5) -> bool:
    """矫正后的针孔图中整块板是否在画面内。"""
    transform = np.asarray(camera_from_board, dtype=float)
    points = spec.outline_points() @ transform[:3, :3].T + transform[:3, 3]
    if np.any(points[:, 2] <= 0):
        return False
    pixels = (new_K @ (points / points[:, 2:3]).T).T[:, :2]
    return bool(np.all(pixels >= margin_px) and np.all(pixels[:, 0] < size[0] - margin_px) and
                np.all(pixels[:, 1] < size[1] - margin_px))


def session_camera(camera: dict) -> FisheyeCamera:
    """会话信息中的鱼眼内参（采集时实际使用的分辨率）。"""
    return FisheyeCamera(K=np.asarray(camera["K"], dtype=np.float64).reshape(3, 3),
                         D=np.asarray(camera["D"], dtype=np.float64).reshape(4, 1),
                         width=int(camera["width"]), height=int(camera["height"]))


def _pose_matrix(pose) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = pose
    rotation = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = (x, y, z)
    return transform


def _matrix(transform) -> list:
    return [[float(v) for v in row] for row in np.asarray(transform)]


def solve_report(camera, spec, samples, observations, prior=None) -> dict:
    """求解并组装 hand_eye_result.yaml 内容；observations 与 samples 一一对应。"""
    views = [View(base_from_flange=_pose_matrix(sample.pose),
                  camera_from_board=observation.camera_from_board,
                  object_points=spec.object_points(observation.ids),
                  pixels=observation.pixels)
             for sample, observation in zip(samples, observations)]
    result = solve_hand_eye(camera, views)
    X = result.flange_from_camera
    report = {
        "description": "眼在手上手眼标定：T_flange_cam = ^Link7 T_cam（相机光学系点 → Link7），"
                       "cam0.T_cam_hand = ^cam T_Link7（与 config/calibration/T_cam_end.yaml 同格式），"
                       "T_base_board = ^base_link T_board，平移单位米。",
        "board": spec.to_dict(),
        "T_flange_cam": _matrix(X),
        "cam0": {"T_cam_hand": _matrix(invert(X))},
        "T_base_board": _matrix(result.base_from_board),
        "reprojection_rms_px": result.rms_px,
        "initial_reprojection_rms_px": result.initial_rms_px,
        "samples_used": len(result.inliers),
        "outliers": [samples[i].index for i in range(len(samples)) if i not in result.inliers],
        "per_sample_rms_px": {samples[i].index: value
                              for i, value in enumerate(result.per_view_rms_px)},
        "diversity": result.diversity,
        "methods": result.methods,
        "warnings": result.warnings,
    }
    if prior is not None:
        report["delta_to_prior"] = {
            "translation_mm": float(np.linalg.norm(prior[:3, 3] - X[:3, 3]) * 1e3),
            "rotation_deg": math.degrees(rotation_angle(prior[:3, :3], X[:3, :3])),
        }
    return report


def export(source: Path, output: Path, *, balance=0.0, fov_scale=1.0, drop_outside=False,
           overwrite=False, solve=True, prior_path=None) -> dict:
    """执行导出（及求解）并返回摘要字典。"""
    session, mask, samples = read_session(source)
    if not samples:
        raise ValueError("没有有效样本")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} 已存在；使用 --overwrite 覆盖")
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True)

    spec = board_spec_from_dict(session["board"])
    camera = session_camera(session["camera"])
    detector = make_detector(spec, camera)
    size = (camera.width, camera.height)
    new_K, map1, map2 = rectification(session["camera"], balance, fov_scale)

    exported, observations, skipped, undetected = [], [], [], []
    pose_lines, sample_lines = [], ["index\tsource_index\timage_receive_ns\tcorners\t"
                                    "fisheye_pnp_rms_px\tboard_inside_rectified"]
    for sample in samples:
        image = cv2.imdecode(np.frombuffer(bytes(sample.image.data), np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[1::-1] != size:
            raise ValueError(f"样本 {sample.index} 图像解码失败或尺寸不符")
        observation = detector.detect(image)
        if observation.camera_from_board is None:
            undetected.append(sample.index)
            continue
        inside = board_inside(spec, observation.camera_from_board, new_K, size)
        if not inside and drop_outside:
            skipped.append(sample.index)
            continue
        rectified = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        number = len(exported)
        cv2.imwrite(str(output / "images" / f"{number}.png"), rectified)
        pose_lines.append("\t".join(f"{value:.9f}" for value in sample.pose))
        sample_lines.append(
            f"{number}\t{sample.index}\t{sample.stamp_ns}\t{observation.corner_count}\t"
            f"{observation.reprojection_rms_px:.4f}\t{int(inside)}")
        exported.append(sample)
        observations.append(observation)

    (output / "poses.csv").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
    (output / "samples.tsv").write_text("\n".join(sample_lines) + "\n", encoding="utf-8")
    intrinsics = {"wrist_camera": {"ros__parameters": {
        "K": [float(v) for v in new_K.reshape(-1)], "D": [0.0] * 5,
        "width": size[0], "height": size[1]}}}
    (output / "intrinsics.yaml").write_text(
        yaml.safe_dump(intrinsics, sort_keys=False), encoding="utf-8")
    if mask is not None:
        cv2.imwrite(str(output / "gripper_mask.png"), mask)

    summary = {"exported": len(exported), "skipped_outside": skipped,
               "undetected": undetected, "board": spec.kind}
    readme = [
        "眼在手上手眼标定导出（keyboard_hand_eye_capture 会话）。",
        "hand_eye_result.yaml：本脚本的求解结果（T_flange_cam = ^Link7 T_cam；"
        "cam0.T_cam_hand 可直接替换 config/calibration/T_cam_end.yaml）。",
        "images/ 已由鱼眼矫正为针孔（intrinsics.yaml 中 D=0）；"
        "poses.csv：^base T_Link7，x y z qx qy qz qw（米）。",
    ]
    pattern = spec.hand_eye_pattern()
    if pattern is None:
        readme.append("hand_eye_calibration 不支持 AprilGrid，以 hand_eye_result.yaml 为准。")
    else:
        command = (
            "ros2 run hand_eye_calibration compute_calibration \\\n"
            f"  --dataset-path {output} --eye-to-hand false --poses-format 1 \\\n"
            f"  --intrinsics-path {output / 'intrinsics.yaml'} \\\n"
            f"  --calibration-pattern \"{pattern}\" \\\n"
            "  --base-frame base_link --effector-frame Link7 "
            "--camera-frame wrist_camera_optical_frame \\\n"
            f"  --output {output / 'calibration.yaml'}")
        readme += ["", "可选交叉验证（hand_eye_calibration 不做棋盘格 180° 消歧，个别样本可能被其判为离群）：",
                   command]
        summary["command"] = command
    (output / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    if solve:
        if len(exported) < 3:
            raise ValueError(f"可求解样本仅 {len(exported)} 个，至少需要 3 个")
        prior = load_prior(prior_path) if prior_path else None
        report = solve_report(camera, spec, exported, observations, prior)
        (output / "hand_eye_result.yaml").write_text(
            yaml.safe_dump(report, sort_keys=False, allow_unicode=True), encoding="utf-8")
        summary["result"] = report
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--no-solve", action="store_true")
    parser.add_argument("--balance", type=float, default=0.0)
    parser.add_argument("--fov-scale", type=float, default=1.0)
    parser.add_argument("--drop-outside", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = export(args.input, args.output, balance=args.balance,
                         fov_scale=args.fov_scale, drop_outside=args.drop_outside,
                         overwrite=args.overwrite, solve=not args.no_solve,
                         prior_path=args.prior)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"导出失败：{error}", file=sys.stderr)
        return 1
    print(f"导出 {summary['exported']} 个样本（{summary['board']}）到 {args.output}")
    if summary["undetected"]:
        print(f"重新检测不到板而跳过的样本：{summary['undetected']}")
    if summary["skipped_outside"]:
        print(f"矫正后板出画面而丢弃的样本：{summary['skipped_outside']}")
    result = summary.get("result")
    if result is not None:
        translation = np.asarray(result["T_flange_cam"])[:3, 3] * 1e3
        print(f"^Link7 T_cam 平移 [{translation[0]:.2f}, {translation[1]:.2f}, "
              f"{translation[2]:.2f}] mm，重投影 RMS {result['reprojection_rms_px']:.3f} px"
              f"（初值 {result['initial_reprojection_rms_px']:.3f} px），"
              f"使用 {result['samples_used']} 个样本")
        diversity = result["diversity"]
        print(f"旋转跨度 {diversity['rotation_span_deg']:.1f}°，"
              f"轴间隔 {diversity['axis_separation_deg']:.1f}°")
        if result["outliers"]:
            print(f"剔除离群样本：{result['outliers']}")
        if "delta_to_prior" in result:
            delta = result["delta_to_prior"]
            print(f"与先验相差 {delta['translation_mm']:.2f} mm / {delta['rotation_deg']:.3f}°")
        for warning in result["warnings"]:
            print(f"警告：{warning}")
        print(f"结果：{args.output / 'hand_eye_result.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
