"""验证正式夹爪模型的版本绑定、单向迁移及 Link7 米制运动学。"""

import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model/dp"))

from diffusion_policy.common.urdf_kinematics import UrdfKinematics
from fastumi_urdf_binding import RM75_URDF_MIGRATIONS, RM75_URDF_SHA256, validate_urdf_binding

# 原训练 URDF 的摘要，迁移后仍用于记录历史数据来源。
TRAINING_SHA256 = "c37545834f0d6a4719548e51337a258be0c3b95bb6b79467d835875a205bb056"
URDF = ROOT / "assets/rm_75.urdf"


def test_official_model_and_defaults_match():
    """默认配置绑定正式文件；旧模型只迁移到已核对的这一个版本。"""
    assert hashlib.sha256(URDF.read_bytes()).hexdigest() == RM75_URDF_SHA256
    assert RM75_URDF_MIGRATIONS == {TRAINING_SHA256: RM75_URDF_SHA256}
    for relative in ("ros2_ws/src/dp_infer/config/dp_infer.yaml", "model/dp/config/vr_umi_ros2.yaml"):
        config = yaml.safe_load((ROOT / relative).read_text())
        assert next(iter(config.values()))["ros__parameters"]["expected_urdf_sha256"] == RM75_URDF_SHA256
    assert validate_urdf_binding(URDF, RM75_URDF_SHA256) == RM75_URDF_SHA256
    assert validate_urdf_binding(URDF, TRAINING_SHA256) == RM75_URDF_SHA256
    assert validate_urdf_binding(URDF, TRAINING_SHA256, RM75_URDF_SHA256) == RM75_URDF_SHA256


def test_binding_rejects_unknown_or_modified_models(tmp_path):
    """不接受任意摘要覆盖、反向迁移、缺失文件或被改写的已登记文件。"""
    changed = tmp_path / "changed.urdf"
    changed.write_bytes(URDF.read_bytes() + b"\n")
    for path, training, target in (
        (changed, TRAINING_SHA256, None),
        (URDF, "a" * 64, None),
        (URDF, "a" * 64, RM75_URDF_SHA256),
        (URDF, RM75_URDF_SHA256, TRAINING_SHA256),
        (URDF, TRAINING_SHA256, TRAINING_SHA256),
        (URDF, "", None),
        (URDF, TRAINING_SHA256, ""),
        (tmp_path / "missing.urdf", TRAINING_SHA256, None),
    ):
        with pytest.raises(ValueError, match="URDF"):
            validate_urdf_binding(path, training, target)
    # 无迁移的其他模型仍可按原始完整文件摘要正常校验。
    changed_digest = hashlib.sha256(changed.read_bytes()).hexdigest()
    assert validate_urdf_binding(changed, changed_digest) == changed_digest


def test_link7_kinematics_matches_original_model():
    """对比原训练 URDF 的两个参考 FK；输入为弧度，输出位置单位米。"""
    fk = UrdfKinematics(URDF, [f"joint{i}" for i in range(1, 8)])
    angles = np.array([[0.] * 7, [.2, -.4, .6, -.8, 1., -1.2, 1.4]])
    expected = np.array([
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, .8505], [0, 0, 0, 1]],
        [[-.6406759048805354, .7456040046195956, .18332772076455991, -.22248000033300352],
         [-.022807715489883692, .22018049583219232, -.9751924719609055, -.2776684259050178],
         [-.7674726008266609, -.628963605902847, -.12405881439885298, .5447693548167676],
         [0, 0, 0, 1]],
    ])
    np.testing.assert_allclose(fk.forward(angles), expected, atol=1e-12, rtol=0)
