"""提供 calibrate_umi_tcp 命令，自动读取全包、生成诊断并保存合格的 TCP 外参。"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import csv
import shutil
import sys
import time

import cv2

from fastumi_umi_tcp_calibration.calibration import solve
from fastumi_umi_tcp_calibration.config import QualitySettings, load_fisheye_camera
from fastumi_umi_tcp_calibration.dual_tag import (
    DualTagError,
    DualTagEstimator,
    draw_observation,
    load_matrix_yaml,
)
from fastumi_umi_tcp_calibration.output import publish_result, write_yaml


# CSV 每行对应一条所选图像话题消息，包括无法解码的消息。
CSV_FIELDS = [
    "index",
    "timestamp_ns",
    "bag_timestamp_ns",
    "status",
    "reason",
    "reprojection_rms_px",
    "normal_angle_deg",
    "coplanarity_deg",
    "tag_distance_m",
    *[f"cam_tag_{row}{column}" for row in range(3) for column in range(4)],
]


def default_repository_root():
    """从当前目录或包安装位置向上定位仓库，找不到时使用当前目录。

    只用于未显式指定的默认路径；不改变用户提供的相对路径含义。
    依据目录结构定位，配置文件缺失时仍能给出正确的缺失路径。
    """
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if (candidate / "config/calibration").is_dir() and (candidate / "ros2_ws/src").is_dir():
                return candidate
    return Path.cwd()


def parser():
    """构造参数解析器；默认路径基于仓库，显式相对路径基于当前目录。"""
    root = default_repository_root()
    result = argparse.ArgumentParser(description="从 MCAP 双 ArUco 图像标定 UMI 相机到 TCP（米制）")
    result.add_argument("--bag", required=True, help="MCAP 文件、bag 目录或 FastUMI 会话目录")
    result.add_argument("--camera-config", default=str(root / "config/calibration/UmiCalib.yaml"))
    result.add_argument(
        "--tag-tcp-config", default=str(root / "config/calibration/T_u_tag_tcp.yaml")
    )
    result.add_argument("--output", default=str(root / "config/calibration/T_u_cam_tcp.yaml"))
    result.add_argument("--image-topic", default="", help="留空时从会话或唯一图像话题推断")
    result.add_argument("--dictionary", default="DICT_4X4_50")
    result.add_argument("--tag0-id", type=int, default=0)
    result.add_argument("--tag1-id", type=int, default=1)
    result.add_argument("--marker-size-m", type=float, default=0.016)
    result.add_argument("--max-reprojection-px", type=float, default=1.5)
    result.add_argument("--max-normal-angle-deg", type=float, default=10.0)
    result.add_argument("--max-coplanarity-deg", type=float, default=10.0)
    result.add_argument("--min-inliers", type=int, default=30)
    result.add_argument("--mad-k", type=float, default=3.0)
    result.add_argument("--translation-floor-m", type=float, default=0.001)
    result.add_argument("--rotation-floor-deg", type=float, default=0.5)
    result.add_argument(
        "--diagnostics-dir", help="新的诊断目录；默认 dataset/umi_tcp_calibration/<时间>"
    )
    result.add_argument("--overwrite", action="store_true", help="允许替换已有输出标定")
    return result


def run(args):
    """运行离线标定，失败时保留报告与 CSV，成功后原子发布矩阵。"""
    # ROS 读取依赖延迟导入，使 --help 和纯算法测试不需要 ROS 环境。
    from fastumi_umi_tcp_calibration.mcap_source import McapImageSource

    output = Path(args.output).expanduser().absolute()
    inputs = [
        Path(p).expanduser().resolve() for p in (args.bag, args.camera_config, args.tag_tcp_config)
    ]
    if output.resolve() in inputs:
        raise ValueError("输出不能覆盖录包或输入配置")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在: {output}；如需替换请使用 --overwrite")
    diagnostic = (
        Path(args.diagnostics_dir).expanduser()
        if args.diagnostics_dir
        else (
            default_repository_root()
            / "dataset/umi_tcp_calibration"
            / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    )
    if output.resolve().is_relative_to(diagnostic.resolve()):
        raise ValueError("标定输出应位于诊断目录外，避免与报告或配置快照冲突")
    # 单次运行独占目录，避免诊断覆盖历史结果。
    diagnostic.mkdir(parents=True, exist_ok=False)
    report = {"success": False, "input_messages": 0, "inliers": 0}
    rows, observations, candidate_rows = [], [], []
    rejected = Counter()
    source = None
    try:
        settings = QualitySettings(
            **{name: getattr(args, name) for name in QualitySettings.__dataclass_fields__}
        )
        camera = load_fisheye_camera(inputs[1])
        tag_from_tcp = load_matrix_yaml(inputs[2], "T_u_tag_tcp")
        estimator = DualTagEstimator(
            camera, args.dictionary, args.marker_size_m, (args.tag0_id, args.tag1_id)
        )
        snapshot = diagnostic / "config"
        snapshot.mkdir()
        shutil.copyfile(inputs[1], snapshot / "camera.yaml")
        shutil.copyfile(inputs[2], snapshot / "T_u_tag_tcp.yaml")
        write_yaml(
            snapshot / "parameters.yaml",
            {
                **vars(args),
                "quality": asdict(settings),
                "resolved_inputs": [str(p) for p in inputs],
                "geometry_note": "T_u_tag_tcp 作为已知输入使用，其测量误差会传递到结果；请核实输入中的几何说明。",
            },
        )
        source = McapImageSource(args.bag, args.image_topic)
        report.update(
            {
                "bag": str(source.path),
                "image_topic": source.image_topic,
                "message_type": source.message_type,
                "T_u_tag_tcp": tag_from_tcp.tolist(),
            }
        )
        debug_dir = diagnostic / "debug"
        debug_dir.mkdir()
        debug_count = 0
        last_progress = time.monotonic()
        print(
            f"读取 {source.path}，图像话题 {source.image_topic}；诊断目录 {diagnostic}", flush=True
        )
        for frame in source.frames():
            if time.monotonic() - last_progress >= 10:
                print(
                    f"已读取 {source.message_count} 帧，单帧合格 {len(observations)} 帧", flush=True
                )
                last_progress = time.monotonic()
            row = {
                "index": frame.index,
                "timestamp_ns": frame.timestamp_ns,
                "bag_timestamp_ns": frame.bag_timestamp_ns,
                "status": "rejected",
                "reason": "",
            }
            rows.append(row)
            if frame.error:
                row["reason"] = "图像解码失败"
                rejected[row["reason"]] += 1
                row["reason"] += ": " + frame.error
                continue
            try:
                observation = estimator.estimate(frame.image)
            except DualTagError as error:
                row["reason"] = str(error)
                rejected[row["reason"]] += 1
                continue
            except ValueError as error:
                row["reason"] = str(error)
                rejected[row["reason"]] += 1
                raise
            row.update(
                {
                    "reprojection_rms_px": observation.reprojection_rms_px,
                    "normal_angle_deg": observation.normal_angle_deg,
                    "coplanarity_deg": observation.coplanarity_deg,
                    "tag_distance_m": observation.tag_distance_m,
                    **{
                        f"cam_tag_{r}{c}": float(observation.camera_from_utag[r, c])
                        for r in range(3)
                        for c in range(4)
                    },
                }
            )
            reason = settings.rejection_reason(observation)
            row["reason"] = reason
            if reason:
                rejected[reason] += 1
                continue
            row["status"] = "candidate"
            observations.append(observation)
            candidate_rows.append(row)
            if debug_count < 20:
                canvas = draw_observation(frame.image, camera, observation, tag_from_tcp)
                if not cv2.imwrite(str(debug_dir / f"frame_{frame.index:08d}.jpg"), canvas):
                    raise OSError("保存调试图片失败")
                debug_count += 1
        matrix, inliers, metrics = solve(observations, tag_from_tcp, settings)
        report.update(metrics)
        for row, keep in zip(candidate_rows, inliers):
            row["status"] = "inlier" if keep else "outlier"
            if not keep:
                row["reason"] = "MAD 离群"
                rejected[row["reason"]] += 1
        if report["inliers"] < settings.min_inliers:
            raise ValueError(f"有效内点 {report['inliers']} 少于 {settings.min_inliers}")
        # 先保存全部诊断，再发布结果；诊断写入失败不会产生成功标定文件。
        report["success"] = True
        report["output"] = str(output)
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = str(error) or "用户中断"
        raise
    finally:
        if source is not None:
            report["input_messages"] = source.message_count
            report["decode_failures"] = source.decode_failures
            source.close()
        report["single_frame_valid"] = len(observations)
        report["rejections"] = dict(rejected)
        with (diagnostic / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        write_yaml(diagnostic / "report.yaml", report)
    try:
        publish_result(output, matrix, args.overwrite)
    except Exception as error:
        report.update(success=False, error=str(error))
        write_yaml(diagnostic / "report.yaml", report)
        raise
    print(
        f"标定完成：内点 {report['inliers']}/{report['input_messages']}；"
        f"TCP 平移 RMS {report['tcp_spread']['translation_rms_mm']:.3f} mm，"
        f"旋转 RMS {report['tcp_spread']['rotation_rms_deg']:.3f}°；输出 {output}"
    )
    return report


def main(argv=None):
    """解析命令参数；成功返回 0，配置、数据或保存失败返回 1。"""
    args = parser().parse_args(argv)
    try:
        run(args)
    except (Exception, KeyboardInterrupt) as error:
        print(f"标定失败：{str(error) or '用户中断'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
