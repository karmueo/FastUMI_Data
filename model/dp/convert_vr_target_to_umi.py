"""将已对齐的 RM75 关节 Zarr 转换为以 Link7 为末端的 UMI 位姿数据。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation
import zarr
from numcodecs import Blosc
from tqdm import tqdm

from diffusion_policy.common.urdf_kinematics import UrdfKinematics

# HDF5 及关节 Zarr 中固定的七轴顺序。
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))


def matrix_to_pose(matrix):
    """将批量齐次变换转换为米制 xyz 和弧度旋转向量。"""
    return np.concatenate((matrix[:, :3, 3], Rotation.from_matrix(matrix[:, :3, :3]).as_rotvec()), axis=-1)


def convert(input_path, urdf_path, output_path):
    """保留图像、时间轴和来源字段，分别将状态和控制目标经 FK 转换；拒绝覆盖。"""
    source = zarr.open_group(str(input_path), mode="r")
    if source.attrs.get("format") != "rm75-joint-image-v1" or not source.attrs.get("complete"):
        raise ValueError("Expected complete joint dataset")
    fk = UrdfKinematics(urdf_path, JOINT_NAMES)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output_path), mode="w-")
    root.attrs.update(dict(source.attrs))
    urdf_hash = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    root.attrs.update({"format": "rm75-umi-pose-v1", "complete": False,
                       "action_layout": "pose10", "stored_action_layout": "xyz_rotvec_gripper",
                       "base_frame": "base_link", "end_frame": "Link7", "tool_offset": np.eye(4).tolist(),
                       "position_unit": "m", "rotation_unit": "rad", "gripper_representation": "normalized_0_1",
                       "joint_names": list(JOINT_NAMES), "urdf_sha256": urdf_hash,
                       "source_dataset": str(input_path.resolve())})
    zarr.copy(source["meta"], root, name="meta")
    data = root.create_group("data")
    original = source["data"]
    # 原样复制压缩数组，确保 RGB、时间轴及源视频索引保持一致。
    for key in ("camera0_rgb", "timestamp", "source_image_index"):
        zarr.copy(original[key], data, name=key)
    count = len(original["action"])
    codec = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    for key, dimension in (("robot0_eef_pos", 3), ("robot0_eef_rot_axis_angle", 3),
                           ("robot0_gripper_width", 1), ("action", 7), ("robot0_demo_start_pose", 6)):
        data.create_dataset(key, shape=(count, dimension), chunks=(1024, dimension), dtype="f4", compressor=codec)
    ends = source["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    reports = []
    for index, (start, end) in enumerate(tqdm(zip(starts, ends), total=len(ends), desc="FK episodes")):
        actions = original["action"][start:end]
        states = original["robot0_joint_pos"][start:end]
        observed = matrix_to_pose(fk.forward(states)).astype(np.float32)
        targets = matrix_to_pose(fk.forward(actions[:, :7])).astype(np.float32)
        data["robot0_eef_pos"][start:end] = observed[:, :3]
        data["robot0_eef_rot_axis_angle"][start:end] = observed[:, 3:]
        data["robot0_gripper_width"][start:end] = original["robot0_gripper_position"][start:end]
        data["action"][start:end] = np.concatenate((targets, actions[:, 7:]), axis=-1)
        data["robot0_demo_start_pose"][start:end] = np.repeat(observed[:1], end - start, axis=0)
        angular_step = (Rotation.from_rotvec(observed[:-1, 3:]).inv() * Rotation.from_rotvec(observed[1:, 3:])).magnitude()
        reports.append({"episode_index": index, "frames": int(end-start),
                        "max_position_step_m": float(np.linalg.norm(np.diff(observed[:, :3], axis=0), axis=-1).max()),
                        "max_rotation_step_rad": float(angular_step.max()),
                        "position_min_m": observed[:, :3].min(axis=0).tolist(),
                        "position_max_m": observed[:, :3].max(axis=0).tolist()})
    # 全量逐块验证复制字段，不依赖仅比较形状的弱校验。
    for start in tqdm(range(0, count, 256), desc="Verify arrays"):
        end = min(start+256, count)
        for key in ("camera0_rgb", "timestamp", "source_image_index"):
            if not np.array_equal(data[key][start:end], original[key][start:end]):
                raise ValueError(f"Copy mismatch: {key}")
        if not np.array_equal(data["action"][start:end, -1], original["action"][start:end, -1]):
            raise ValueError("Action gripper changed")
        if not np.array_equal(data["robot0_gripper_width"][start:end], original["robot0_gripper_position"][start:end]):
            raise ValueError("Observed gripper changed")
        for key in ("robot0_eef_pos", "robot0_eef_rot_axis_angle", "action"):
            if not np.isfinite(data[key][start:end]).all():
                raise ValueError(f"Nonfinite pose: {key}")
    shutil.copy2(urdf_path, output_path.parent / "rm_75.urdf")
    report = {"frames": count, "episodes": len(ends), "urdf_sha256": urdf_hash,
              "copy_verified": True, "joint_order": list(JOINT_NAMES), "episode_statistics": reports}
    (output_path.parent / "conversion_report.json").write_text(json.dumps(report, indent=2))
    root.attrs["complete"] = True
    return report


def main():
    """解析显式输入、URDF 与新输出路径，执行只读源数据转换。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = convert(args.input, args.urdf, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "episode_statistics"}, indent=2))


if __name__ == "__main__":
    main()
