"""比较原模型入口与 DP 部署包在同一录制观测上的输出。"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def compare(left, right):
    """返回位姿和夹爪目标差异；坐标单位米、角度单位度。"""
    delta_position = left["positions"] - right["positions"]
    delta_rotation = (
        Rotation.from_quat(left["quaternions"]).inv()
        * Rotation.from_quat(right["quaternions"])
    )
    return {
        "position_max_abs_m": float(np.max(np.abs(delta_position))),
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(delta_position ** 2, axis=1)))),
        "rotation_max_deg": float(np.max(np.rad2deg(delta_rotation.magnitude()))),
        "gripper_max_abs": float(np.max(np.abs(
            left["gripper_openness"] - right["gripper_openness"]))),
        "time_equal": bool(np.array_equal(
            left["time_from_start"], right["time_from_start"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate-16", required=True, type=Path)
    parser.add_argument("--candidate-8", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with (np.load(args.reference) as reference,
          np.load(args.candidate_16) as candidate_16,
          np.load(args.candidate_8) as candidate_8):
        report = {
            "reference_16_vs_dp_16": compare(reference, candidate_16),
            "reference_16_vs_dp_8": compare(reference, candidate_8),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    exact = report["reference_16_vs_dp_16"]
    if (not exact["time_equal"] or exact["position_max_abs_m"] > 1e-6
            or exact["rotation_max_deg"] > 1e-4
            or exact["gripper_max_abs"] > 1e-6):
        raise AssertionError("deployment model differs from 16-step training reference")


if __name__ == "__main__":
    main()
