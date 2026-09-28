"""检查推理端与 RM75 控制端部署约定。"""

from pathlib import Path

import pytest

from dp_infer.urdf_contract import validate_kinematic_equivalence
from dp_infer.policy_loader import load_policy
from dp_infer import cli


REPOSITORY = Path(__file__).resolve().parents[4]
TRAINING_URDF = REPOSITORY / "dataset/vr_target_umi/rm_75.urdf"
CONTROL_URDF = REPOSITORY / "ros2_ws/src/fastumi_rm75/assets/rm_75_kinematic.urdf"


def test_training_and_controller_joint_chain_match(tmp_path):
    """当前训练和部署 URDF 的七轴几何与限制一致，改动须阻止控制。"""
    validate_kinematic_equivalence(TRAINING_URDF, CONTROL_URDF)
    modified = tmp_path / "wrong.urdf"
    modified.write_text(CONTROL_URDF.read_text().replace('xyz="0 0 0.2405"', 'xyz="0 0 0.2505"'))
    with pytest.raises(ValueError, match="differs"):
        validate_kinematic_equivalence(TRAINING_URDF, modified)


def test_checkpoint_and_inference_steps_fail_fast(tmp_path):
    """错误路径与采样次数应在反序列化模型前拒绝。"""
    with pytest.raises(ValueError, match="checkpoint"):
        load_policy(tmp_path / "missing.ckpt", "cuda:0", 8)
    present = tmp_path / "model.ckpt"
    present.write_bytes(b"not a checkpoint")
    for steps in (0, 51, 8.5, True):
        with pytest.raises(ValueError, match="num_inference_steps"):
            load_policy(present, "cuda:0", steps)


def test_ros_entry_uses_model_environment_without_resolving_python_link(
        tmp_path, monkeypatch):
    """ros2 run 保留虚拟环境解释器路径与传入的 ROS 参数。"""
    executable = tmp_path / "python"
    executable.symlink_to("/usr/bin/python3")
    monkeypatch.setenv("DP_INFER_PYTHON", str(executable))
    monkeypatch.setattr(cli.sys, "argv", ["dp_infer_node", "--ros-args", "-p", "device:=cpu"])
    calls = []
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: calls.append((path, argv)))
    cli.main()
    assert calls == [(
        str(executable), [str(executable), "-m", "dp_infer.node",
                          "--ros-args", "-p", "device:=cpu"])]
