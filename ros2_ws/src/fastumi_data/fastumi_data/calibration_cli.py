"""从配对姿态或 pivot 姿态生成版本化 Tracker 到 TCP 外参。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from fastumi_data.calibration_solver import (
    document_to_transform,
    solve_pivot_translation,
    solve_robot_world_hand_eye,
    transform_to_document,
)
from fastumi_data.pose_math import pose_to_matrix


def _read_yaml(path: str) -> dict:
    """读取 YAML 映射并提供稳定错误信息。"""
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取 {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{path} 顶层必须是 YAML 映射")
    return document


def _write_result(path: str, document: dict) -> None:
    """写入标定结果 YAML。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _paired(arguments: argparse.Namespace) -> None:
    """执行 robot-world/hand-eye 配对姿态标定。"""
    document = _read_yaml(arguments.input)
    samples = document.get("samples")
    if not isinstance(samples, list):
        raise ValueError("配对标定文件缺少 samples 数组")
    base_tcp_poses = [
        document_to_transform(sample["base_tcp"]) for sample in samples
    ]
    tracker_poses = [
        document_to_transform(sample["vive_tracker"]) for sample in samples
    ]
    initial = None
    if "initial_tracker_to_tcp" in document:
        initial = document_to_transform(document["initial_tracker_to_tcp"])
    result = solve_robot_world_hand_eye(
        base_tcp_poses, tracker_poses, initial_tracker_to_tcp=initial
    )
    accepted = (
        result.translation_rmse_mm <= 2.0
        and result.rotation_rmse_deg <= 1.0
    )
    if not arguments.allow_high_residual and not accepted:
        raise RuntimeError(
            "留出集误差超过门限: "
            f"{result.translation_rmse_mm:.3f} mm, "
            f"{result.rotation_rmse_deg:.3f} deg"
        )
    output = {
        "schema_version": 2,
        "accepted": accepted,
        "tracker_serial": arguments.tracker_serial,
        "fixture_version": arguments.fixture_version,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "training_sample_count": result.training_sample_count,
        "validation_sample_count": result.validation_sample_count,
        "translation_rmse_mm": result.translation_rmse_mm,
        "rotation_rmse_deg": result.rotation_rmse_deg,
        "tracker_to_tcp": transform_to_document(result.tracker_to_tcp),
        "base_to_vive": transform_to_document(result.base_to_vive),
    }
    _write_result(arguments.output, output)


def _pivot(arguments: argparse.Namespace) -> None:
    """执行固定点 pivot 平移标定并使用给定方向外参。"""
    document = _read_yaml(arguments.input)
    samples = document.get("samples")
    if not isinstance(samples, list):
        raise ValueError("pivot 标定文件缺少 samples 数组")
    tracker_poses = [
        document_to_transform(sample["vive_tracker"]) for sample in samples
    ]
    translation, rmse_mm = solve_pivot_translation(tracker_poses)
    quaternion = np.asarray(arguments.quaternion_xyzw, dtype=np.float64)
    tracker_to_tcp = pose_to_matrix(translation, quaternion)
    accepted = rmse_mm <= 2.0
    if not arguments.allow_high_residual and not accepted:
        raise RuntimeError(f"pivot 平移 RMSE {rmse_mm:.3f} mm 超过门限")
    output = {
        "schema_version": 2,
        "accepted": accepted,
        "tracker_serial": arguments.tracker_serial,
        "fixture_version": arguments.fixture_version,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "method": "pivot_translation_with_fixture_orientation",
        "sample_count": len(samples),
        "translation_rmse_mm": rmse_mm,
        "rotation_rmse_deg": None,
        "tracker_to_tcp": transform_to_document(tracker_to_tcp),
    }
    _write_result(arguments.output, output)


def main(argv: Optional[list[str]] = None) -> None:
    """解析标定方法和输入输出参数。"""
    parser = argparse.ArgumentParser(description="标定 Tracker 到 UMI TCP")
    subparsers = parser.add_subparsers(dest="method", required=True)
    paired = subparsers.add_parser("paired", help="RM75 配对姿态联合标定")
    pivot = subparsers.add_parser("pivot", help="固定点 pivot 降级标定")
    for command in (paired, pivot):
        command.add_argument("--input", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--tracker-serial", required=True)
        command.add_argument("--fixture-version", required=True)
        command.add_argument("--allow-high-residual", action="store_true")
    pivot.add_argument(
        "--quaternion-xyzw",
        required=True,
        nargs=4,
        type=float,
        help="定向槽或 CAD 给出的 Tracker 到 TCP 方向",
    )
    paired.set_defaults(handler=_paired)
    pivot.set_defaults(handler=_pivot)
    arguments = parser.parse_args(argv)
    arguments.handler(arguments)
    print(f"标定结果已写入 {arguments.output}")


if __name__ == "__main__":
    main()
