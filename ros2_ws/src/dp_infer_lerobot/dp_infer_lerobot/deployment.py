"""绑定 checkpoint、处理器、部署 URDF 和安装标定；无需训练数据访问。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .core import rigid_matrix
from .processors import read_processor_spec

CONTRACT = {
    "frequency_hz": 30, "action_reference": "per_step_link7",
    "rotation_representation": "matrix_first_two_rows", "position_unit": "m",
    "base_frame": "base_link", "end_frame": "Link7", "gripper_time_index": "current",
    "gripper_representation": "normalized_0_1", "image_preprocessing": "rgb_pillow_box_letterbox_224",
}
REQUIRED_FILES = {"config.json", "model.safetensors", "normalization_stats.json",
                  "policy_preprocessor.json", "policy_postprocessor.json"}


def sha256_file(path):
    """分块读取摘要，避免把模型权重整体读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_sha256(matrix):
    """与现有修正生成器一致：round12、默认 JSON、矩阵指纹。"""
    values = np.round(rigid_matrix(matrix), 12).tolist()
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def correction_from_profile(profile):
    """由配套训练/部署手眼和相机安装测量重算 C，验证保存的来源摘要。"""
    cal = profile["calibration"]
    training = rigid_matrix(cal["T_Cd_Link7_training"])
    deploy = rigid_matrix(cal["T_Cd_Link7_deploy"])
    for name, value in (("training", training), ("deploy", deploy)):
        if matrix_sha256(value) != cal[f"T_Cd_Link7_{name}_sha256"]:
            raise ValueError(f"Calibration {name} matrix fingerprint mismatch")
    correction = (np.linalg.inv(training) @ rigid_matrix(cal["T_Cu_F"])
                  @ np.linalg.inv(rigid_matrix(cal["T_Cd_F"])) @ deploy)
    rigid_matrix(correction)
    # 测量保存为有限小数；按既有加载器约定重新正交化。
    correction[:3, :3] = Rotation.from_matrix(correction[:3, :3]).as_matrix()
    return correction


def load_profile(path):
    """校验部署配置格式与动作语义，模型形状另由 config 校验。"""
    profile = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(profile, dict) or profile.get("version") != 1
            or profile.get("contract") != CONTRACT):
        raise ValueError("Unsupported deployment contract")
    correction_from_profile(profile)
    return profile


def validate_checkpoint(checkpoint, profile):
    """只接受当前配置绑定的一套权重/处理器，不读取训练恢复状态。"""
    root = Path(checkpoint)
    if not root.is_dir():
        raise ValueError("checkpoint must name a local pretrained directory")
    pre, _ = read_processor_spec(root)
    post, _ = read_processor_spec(root, inverse=True)
    expected = REQUIRED_FILES | {pre.name, post.name}
    files = profile.get("checkpoint_files", {})
    if set(files) != expected:
        raise ValueError("Deployment profile must bind every required checkpoint file")
    for name, digest in files.items():
        if Path(name).name != name or sha256_file(root / name) != digest:
            raise ValueError(f"Checkpoint fingerprint mismatch: {name}")
    stats = json.loads((root / "normalization_stats.json").read_text(encoding="utf-8"))
    if stats.get("source") != "train":
        raise ValueError("Expected training-subset normalization statistics")


def validate_urdf(path, profile):
    """该模型没有训练 URDF；只验证冻结的部署运动链。"""
    if sha256_file(path) != profile["deployment_urdf_sha256"]:
        raise ValueError("Deployment URDF fingerprint mismatch")


def load_correction(path, profile):
    """空路径使用已绑定测量计算的 C；显式文件必须与该模型标定一致。"""
    expected = correction_from_profile(profile)
    if not path:
        return expected
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["link7_correction"]
    for name in ("training", "deploy"):
        key = f"T_Cd_Link7_{name}_sha256"
        if document.get(key) != profile["calibration"][key]:
            raise ValueError("Link7 correction calibration fingerprint mismatch")
    actual = rigid_matrix(document["T_virtual_link7"])
    if not np.allclose(actual, expected, atol=2e-8, rtol=0):
        raise ValueError("Link7 correction does not match deployment measurements")
    return expected
