"""验证 RM75 DP 数据审计器与训练数据契约的一致性。"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import zarr

from datatool.validate_dp_dataset import discover_datasets, main, validate_dataset


def make_dataset(path, kind="rm75_joint", version=2, lengths=(18, 18)):
    """生成两段可训练的 30 Hz、224×224 RGB Zarr。"""
    count = sum(lengths)
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(format=("rm75-joint-image" if kind == "rm75_joint" else
                              "rm75-umi-pose") + f"-v{version}",
                      complete=True, frequency=30, image_size=224,
                      action_layout="joint8" if kind == "rm75_joint" else "pose10")
    if kind == "rm75_umi":
        root.attrs.update(stored_action_layout="xyz_rotvec_gripper", base_frame="base_link",
                          end_frame="Link7", position_unit="m", rotation_unit="rad",
                          tool_offset=np.eye(4).tolist(), gripper_representation="normalized_0_1",
                          joint_names=[f"joint{i}" for i in range(1, 8)], urdf_sha256="a" * 64)
    if version == 2:
        root.attrs.update(normalization_contract="urdf-joint-limits-v1",
                          joint_names=[f"joint{i}" for i in range(1, 8)],
                          joint_lower=[-3] * 7, joint_upper=[3] * 7,
                          gripper_lower=0, gripper_upper=1, urdf_sha256="a" * 64)
    data = root.create_group("data")
    data.create_dataset("camera0_rgb", data=np.zeros((count, 224, 224, 3), dtype="u1"),
                        chunks=(2, 224, 224, 3))
    times = np.concatenate([np.arange(n) / 30 for n in lengths])
    data.create_dataset("timestamp", data=times, chunks=(5,))
    data.create_dataset("source_image_index", data=np.concatenate(
        [np.arange(n) for n in lengths]).astype("i8"), chunks=(5,))
    if kind == "rm75_joint":
        data.create_dataset("robot0_joint_pos", data=np.zeros((count, 7), dtype="f4"), chunks=(5, 7))
        data.create_dataset("robot0_gripper_position", data=np.full((count, 1), 0.5, dtype="f4"))
        data.create_dataset("action", data=np.c_[np.zeros((count, 7)),
                                                 np.full(count, 0.5)].astype("f4"))
    else:
        data.create_dataset("robot0_eef_pos", data=np.zeros((count, 3), dtype="f4"))
        data.create_dataset("robot0_eef_rot_axis_angle", data=np.zeros((count, 3), dtype="f4"))
        data.create_dataset("robot0_gripper_width", data=np.full((count, 1), 0.5, dtype="f4"))
        data.create_dataset("robot0_demo_start_pose", data=np.zeros((count, 6), dtype="f4"))
        data.create_dataset("action", data=np.c_[np.zeros((count, 6)),
                                                 np.full(count, 0.5)].astype("f4"))
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=np.cumsum(lengths, dtype="i8"))
    meta.create_dataset("episode_names", data=np.asarray([f"episode_{i}" for i in range(len(lengths))]))
    return root


def codes(result):
    """返回错误码集合，便于检查具体拒绝原因。"""
    return {issue["code"] for issue in result["errors"]}


@pytest.mark.parametrize("kind", ["rm75_joint", "rm75_umi"])
@pytest.mark.parametrize("version", [1, 2])
def test_valid_versions_and_windows(tmp_path, kind, version):
    """两类 v1/v2 数据都支持 16 步动作与确定性训练/验证划分。"""
    path = tmp_path / "target.zarr"
    make_dataset(path, kind, version)
    result = validate_dataset(path, kind)
    assert result["valid"], result["errors"]
    assert result["windows"] == {"train": 3, "validation": 3, "total": 6}
    assert result["frames"] == 36
    assert result["episodes"] == 2


def test_wrong_type_and_missing_field(tmp_path):
    """格式不匹配和缺失动作都不能通过。"""
    path = tmp_path / "target.zarr"
    root = make_dataset(path)
    assert "FORMAT_MISMATCH" in codes(validate_dataset(path, "rm75_umi"))
    del root["data/action"]
    assert "ARRAY_MISSING" in codes(validate_dataset(path, "rm75_joint"))


def test_shape_dtype_nonfinite_and_bounds(tmp_path):
    """同时定位维度、非有限值、关节和夹爪越界。"""
    path = tmp_path / "target.zarr"
    root = make_dataset(path)
    action = root["data/action"]
    action[2, 0] = np.nan
    action[3, 1] = 4
    action[4, 7] = 1.2
    del root["data/robot0_joint_pos"]
    root["data"].create_dataset("robot0_joint_pos", data=np.zeros((36, 6), dtype="f4"))
    result = validate_dataset(path, "rm75_joint")
    assert {"SHAPE_MISMATCH", "NONFINITE", "JOINT_OUT_OF_RANGE",
            "GRIPPER_OUT_OF_RANGE"} <= codes(result)
    assert next(issue for issue in result["errors"] if issue["code"] == "NONFINITE")[
        "examples"][0]["frame"] == 2


def test_metadata_and_start_pose(tmp_path):
    """Link7 坐标元数据和每段起始姿态必须匹配。"""
    path = tmp_path / "target.zarr"
    root = make_dataset(path, "rm75_umi")
    root.attrs["end_frame"] = "tool0"
    root["data/robot0_demo_start_pose"][20, 0] = 0.1
    assert {"ATTRIBUTE_MISMATCH", "START_POSE_MISMATCH"} <= codes(
        validate_dataset(path, "rm75_umi"))


def test_time_reset_allowed_but_reverse_rejected(tmp_path):
    """跨段时间重置合法；段内倒退要带上帧定位。"""
    path = tmp_path / "target.zarr"
    root = make_dataset(path)
    assert validate_dataset(path, "rm75_joint")["valid"]
    root["data/timestamp"][21] = -1
    result = validate_dataset(path, "rm75_joint")
    issue = next(issue for issue in result["errors"] if issue["code"] == "TIMESTAMP_INVALID")
    assert issue["examples"][0]["episode"] == 1
    assert issue["examples"][0]["frame"] == 21


def test_invalid_episode_boundary_and_empty_split(tmp_path):
    """拒绝重叠边界和无验证窗口的数据。"""
    path = tmp_path / "target.zarr"
    root = make_dataset(path, lengths=(18, 4))
    assert "NO_VALIDATION_WINDOWS" in codes(validate_dataset(path, "rm75_joint")) or \
        "NO_TRAIN_WINDOWS" in codes(validate_dataset(path, "rm75_joint"))
    assert "EPISODE_TOO_SHORT" in {issue["code"] for issue in
                                    validate_dataset(path, "rm75_joint")["warnings"]}
    root["meta/episode_ends"][:] = [18, 18]
    assert "EPISODE_ENDS_INVALID" in codes(validate_dataset(path, "rm75_joint"))


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_chunk_integrity(tmp_path, failure):
    """缺失 chunk 不能被填充值掩盖，损坏 chunk 应报告读取失败。"""
    path = tmp_path / "target.zarr"
    make_dataset(path)
    chunk = path / "data" / "camera0_rgb" / "0.0.0.0"
    if failure == "missing":
        chunk.unlink()
        expected = "CHUNK_MISSING"
    else:
        chunk.write_bytes(b"invalid blosc chunk")
        expected = "CHUNK_READ_FAILED"
    assert expected in codes(validate_dataset(path, "rm75_joint"))


def test_discovery_report_exit_and_readonly(tmp_path, capsys):
    """父目录批量扫描并报告失败，且不写入源数据。"""
    first = tmp_path / "one" / "target.zarr"
    second = tmp_path / "two" / "target.zarr"
    first.parent.mkdir()
    second.parent.mkdir()
    make_dataset(first)
    root = make_dataset(second)
    root.attrs["complete"] = False
    before = {str(path): path.stat().st_mtime_ns for path in tmp_path.rglob("*") if path.is_file()}
    assert discover_datasets(tmp_path) == [first, second]
    report = tmp_path.parent / "report.json"
    assert main(["--input", str(tmp_path), "--type", "rm75_joint",
                 "--report", str(report)]) == 1
    payload = json.loads(report.read_text())
    assert len(payload["datasets"]) == 2
    assert [item["valid"] for item in payload["datasets"]] == [True, False]
    assert "不通过" in capsys.readouterr().out
    assert before == {str(path): path.stat().st_mtime_ns for path in tmp_path.rglob("*") if path.is_file()}


def test_corrupt_metadata_does_not_stop_other_datasets(tmp_path):
    """一个数据集元数据损坏时仍可继续验证其他数据集。"""
    first = tmp_path / "one.zarr"
    second = tmp_path / "two.zarr"
    make_dataset(first)
    make_dataset(second)
    (first / ".zattrs").write_text("invalid JSON")
    report = tmp_path.parent / "metadata-report.json"
    assert main(["--input", str(tmp_path), "--type", "rm75_joint",
                 "--report", str(report)]) == 1
    datasets = json.loads(report.read_text())["datasets"]
    assert [item["valid"] for item in datasets] == [False, True]
    assert "ATTRIBUTE_READ_FAILED" in codes(datasets[0])


def test_no_dataset_and_invalid_parameters(tmp_path):
    """无数据时退出 1，参数错误退出 2。"""
    assert main(["--input", str(tmp_path), "--type", "rm75_joint"]) == 1
    with pytest.raises(SystemExit) as exc:
        main(["--input", str(tmp_path), "--type", "rm75_joint", "--action-horizon", "0"])
    assert exc.value.code == 2
    data = tmp_path / "target.zarr"
    make_dataset(data)
    assert main(["--input", str(data), "--type", "rm75_joint",
                 "--report", str(data / "result.json")]) == 2


def test_loader_contract_on_small_fixture(tmp_path, monkeypatch):
    """检查窗口计数与现有加载器一致，并验证 UMI 实际输出五键/10D。"""
    # 根目录离线环境不安装 DP 训练依赖；完整 DP 环境仍执行此契约检查。
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "model" / "dp"))
    from diffusion_policy.dataset.vr_joint_image_dataset import VrJointImageDataset
    from diffusion_policy.dataset.umi_dataset import UmiDataset

    joint_path = tmp_path / "joint.zarr"
    umi_path = tmp_path / "umi.zarr"
    make_dataset(joint_path)
    make_dataset(umi_path, "rm75_umi")
    joint_shape = {"obs": {"camera0_rgb": {"shape": [3, 224, 224], "horizon": 2},
                           "robot0_joint_pos": {"shape": [7], "horizon": 2},
                           "robot0_gripper_position": {"shape": [1], "horizon": 2}},
                   "action": {"shape": [8], "horizon": 16}}
    joint = VrJointImageDataset(str(joint_path), joint_shape)
    assert len(joint) == validate_dataset(joint_path, "rm75_joint")["windows"]["train"]

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = Path(__file__).resolve().parents[1] / "model" / "dp" / "diffusion_policy" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="train_diffusion_unet_timm_vr_umi_workspace")
    dataset = UmiDataset(config.task.shape_meta, str(umi_path), val_ratio=0.1,
                         normalizer_num_workers=0, start_pose_noise_std=0)
    sample = dataset[0]
    assert len(dataset) == validate_dataset(umi_path, "rm75_umi")["windows"]["train"]
    assert set(sample["obs"]) == set(config.task.shape_meta.obs)
    assert sample["action"].shape == (16, 10)
