"""提供独立单鱼眼相机 Kalibr 内参标定命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from fastumi_camera_calibration.calibration import (
    DEFAULT_IMAGE_TOPIC,
    IntrinsicCalibrationError,
    calibrate_camera_intrinsics,
)


def _finite_positive_float(value: Any) -> float:
    """解析有限正浮点数。

    Args:
        value: 命令行输入值。

    Returns:
        有限正浮点数。

    Raises:
        argparse.ArgumentTypeError: 输入无法满足范围约束。
    """
    try:
        # 统一接受 argparse 字符串和测试传入的数值。
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("必须是浮点数") from error
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("必须是有限正数")
    return parsed


def _finite_nonnegative_float(value: Any) -> float:
    """解析有限非负浮点数。"""
    try:
        # 时间窗口允许零，用于只保留首个时间点。
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("必须是浮点数") from error
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("必须是有限非负数")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """创建独立内参标定参数解析器。"""
    # 首版接口固定单相机 pinhole-equi 模型。
    parser = argparse.ArgumentParser(
        description="从 FastUMI MCAP 生成单鱼眼相机 Kalibr 内参"
    )
    parser.add_argument("--bag", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument(
        "--frequency-hz", type=_finite_positive_float, default=4.0
    )
    parser.add_argument(
        "--sample-end-offset-s", type=_finite_nonnegative_float
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """解析参数、执行内参标定并输出机器可读摘要。

    Args:
        argv: 可选的命令行参数；省略时使用进程参数。

    Raises:
        SystemExit: 标定失败时以状态码 2 退出。
    """
    # 参数错误由 argparse 统一返回状态码 2。
    arguments = build_argument_parser().parse_args(argv)
    try:
        # 核心函数负责临时文件、Kalibr 调用和原子发布。
        result = calibrate_camera_intrinsics(
            arguments.bag,
            arguments.image_topic,
            arguments.target_config,
            arguments.output_dir,
            arguments.frequency_hz,
            arguments.sample_end_offset_s,
        )
    except IntrinsicCalibrationError as error:
        # 错误摘要只引用本次实际保留的日志。
        payload = {
            "accepted": False,
            "failure": str(error),
            "output_paths": (
                {"log": str(error.log_path)}
                if error.log_path is not None
                else {}
            ),
        }
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from error

    # 成功摘要公开稳定的四类产物路径与完整 provenance。
    artifacts = result.artifacts
    payload = {
        "accepted": True,
        "output_paths": {
            "yaml": str(artifacts.yaml_path),
            "results": str(artifacts.results_path),
            "report": str(artifacts.report_path),
            "log": str(artifacts.log_path),
        },
        "provenance": dict(result.provenance),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
