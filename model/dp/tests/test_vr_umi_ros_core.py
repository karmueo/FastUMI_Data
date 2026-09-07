"""验证 ROS 推理核心与训练预处理一致，以及同步、解码、扩展和契约拒绝行为。"""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from hydra import compose, initialize_config_dir

from convert_vr_target import letterbox_rgb as training_letterbox
from vr_umi_ros.core import (ActionSequence, InferenceContext, JOINT_NAMES, Observation, PolicyEngine,
                             build_observations, decode_actions, image_to_rgb, letterbox_rgb,
                             load_processors, ordered_joints, validate_contract, validate_sequence,
                             validate_urdf)
from vr_umi_ros.synchronizer import ObservationBuffer
from umi.common.pose_util import mat_to_pose10d, pose_to_mat


def config():
    """读取真实训练配置，供契约测试使用。"""
    directory = Path(__file__).resolve().parents[1] / "diffusion_policy/config"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        cfg = compose(config_name="train_diffusion_unet_timm_vr_umi_workspace")
    cfg.task.contract.urdf_sha256 = "0" * 64
    return cfg


def test_joint_names_and_invalid_values():
    """任意顺序正确重排，缺关节、重名和 NaN 不能进入模型。"""
    np.testing.assert_array_equal(ordered_joints(list(JOINT_NAMES)[::-1], list(range(7))[::-1]), np.arange(7))
    for names, values in ((list(JOINT_NAMES)[:-1], [0] * 6),
                          (list(JOINT_NAMES) + ["joint1"], [0] * 8),
                          (list(JOINT_NAMES), [np.nan] * 7)):
        with pytest.raises(ValueError):
            ordered_joints(names, values)


def test_image_stride_color_and_training_letterbox():
    """非方形带填充 BGR 解码后，与训练 letterbox 像素完全一致。"""
    bgr = np.random.default_rng(7).integers(0, 256, (13, 27, 3), dtype=np.uint8)
    padded = np.full((13, 88), 77, dtype=np.uint8)
    padded[:, :81] = bgr.reshape(13, 81)
    rgb = image_to_rgb(padded.tobytes(), 13, 27, 88, "bgr8")
    np.testing.assert_array_equal(rgb, bgr[..., ::-1])
    expected = np.moveaxis(training_letterbox(bgr), -1, 0).astype(np.float32) / 255
    np.testing.assert_array_equal(letterbox_rgb(rgb), expected)
    np.testing.assert_array_equal(image_to_rgb(rgb.tobytes(), 13, 27, 81, "rgb8"), rgb)
    with pytest.raises(ValueError):
        image_to_rgb(b"", 1, 1, 3, "mono8")
    with pytest.raises(ValueError):
        image_to_rgb(b"", 1, 1, 3, "rgb8")


def test_actual_dataset_observation_parity(tmp_path):
    """通过实际 UmiDataset 样本对比全部五键，覆盖 episode 起点和旋转参考系。"""
    from test_vr_umi import minimal_urdf, synthetic_joint_zarr
    from convert_vr_target_to_umi import convert
    from diffusion_policy.dataset.umi_dataset import UmiDataset

    synthetic_joint_zarr(tmp_path / "joints.zarr")
    convert(tmp_path / "joints.zarr", minimal_urdf(tmp_path / "arm.urdf", 7), tmp_path / "poses.zarr")
    cfg = config()
    cfg.task.shape_meta.obs.camera0_rgb.shape = [3, 8, 8]
    dataset = UmiDataset(cfg.shape_meta, str(tmp_path / "poses.zarr"),
                         pose_repr={"obs_pose_repr": "relative", "action_pose_repr": "relative"},
                         train_episode_indices=[0], val_episode_indices=[2], start_pose_noise_std=0)
    raw = dataset.sampler.sample_sequence(2)
    poses = pose_to_mat(np.c_[raw["robot0_eef_pos"], raw["robot0_eef_rot_axis_angle"]])
    start = pose_to_mat(raw["robot0_demo_start_pose"][0])
    history = [Observation(index, np.moveaxis(raw["camera0_rgb"][index], -1, 0) / 255,
                           poses[index], float(raw["robot0_gripper_width"][index, 0])) for index in range(2)]
    actual = build_observations(history, start)
    for name, expected in dataset[2]["obs"].items():
        np.testing.assert_allclose(actual[name][0], expected.numpy(), atol=1e-6)


def test_absolute_pose_decode_and_validation():
    """非交换旋转和平移验证左乘还原，同时覆盖夹爪裁剪、四元数符号和退化输入。"""
    reference = pose_to_mat(np.array([0.3, -0.2, 0.1, 0.2, -0.4, 0.7]))
    relative = pose_to_mat(np.array([0.1, 0.2, -0.1, -0.5, 0.1, 0.2]))
    actions = np.tile(np.r_[mat_to_pose10d(relative), 1.5], (16, 1))
    context = InferenceContext(reference, 123, 2, 5)
    sequence = decode_actions(actions, context)
    expected = reference @ relative
    np.testing.assert_allclose(sequence.positions, np.tile(expected[:3, 3], (16, 1)))
    np.testing.assert_allclose(Rotation.from_quat(sequence.quaternions).as_matrix(),
                               np.tile(expected[:3, :3], (16, 1, 1)), atol=1e-12)
    assert np.all(sequence.gripper_openness == 1)
    sequence.quaternions[1::2] *= -2
    sequence.gripper_openness[:] = -0.5
    sequence = validate_sequence(sequence)
    assert np.all(sequence.gripper_openness == 0)
    assert np.all(np.sum(sequence.quaternions[1:] * sequence.quaternions[:-1], axis=-1) > 0)
    for invalid in (np.zeros((16, 10)), np.full((16, 10), np.nan), np.zeros((8, 10))):
        with pytest.raises(ValueError):
            decode_actions(invalid, context)
    actions[:, 6:9] = actions[:, 3:6]
    with pytest.raises(ValueError, match="second"):
        decode_actions(actions, context)
    sequence.time_from_start[1] = -1
    with pytest.raises(ValueError, match="times"):
        validate_sequence(sequence)


def test_custom_processor_engine(monkeypatch):
    """工厂扩展实际经过引擎调用；无效后处理不能返回到发布层。"""
    import torch

    class Processor:
        """将目标平移一米，测试上下文传递。"""

        def process(self, sequence, context):
            """修改目标并验证上下文。"""
            assert context.episode_id == 8
            sequence.positions[:, 0] += 1
            return sequence

    monkeypatch.setitem(sys.modules, "fake_processor", SimpleNamespace(create=Processor))
    processors = load_processors(["fake_processor:create"])
    actions = torch.tensor(np.tile(np.r_[np.zeros(3), [1, 0, 0, 0, 1, 0], 0.3], (1, 16, 1)))
    engine = PolicyEngine.__new__(PolicyEngine)
    engine.policy = SimpleNamespace(predict_action=lambda inputs: {"action_pred": actions})
    engine.device = "cpu"
    engine.processors = processors
    result = engine.predict({}, InferenceContext(np.eye(4), 0, 8, 1))
    np.testing.assert_array_equal(result.positions[:, 0], np.ones(16))
    engine.processors = [SimpleNamespace(process=lambda sequence, context: ActionSequence(
        np.full((16, 3), np.nan), sequence.quaternions, sequence.gripper_openness, sequence.time_from_start))]
    with pytest.raises(ValueError):
        engine.predict({}, InferenceContext(np.eye(4), 0, 8, 1))


def test_contract_rejects_incompatible_checkpoints():
    """真实配置通过，错误末端、参考系或观测历史被拒绝。"""
    cfg = config()
    validate_contract(cfg)
    cfg.task.contract.end_frame = "fastumi_tcp"
    with pytest.raises(ValueError):
        validate_contract(cfg)
    cfg = config()
    cfg.task.pose_repr.action_pose_repr = "rel"
    with pytest.raises(ValueError):
        validate_contract(cfg)
    cfg = config()
    cfg.task.low_dim_obs_horizon = 3
    with pytest.raises(ValueError):
        validate_contract(cfg)
    cfg = config()
    cfg.task.contract.urdf_sha256 = None
    with pytest.raises(ValueError, match="URDF SHA-256"):
        validate_contract(cfg)


def test_deployment_urdf_must_match_checkpoint(tmp_path):
    """部署文件只有原始字节摘要与训练契约一致时才能用于 FK。"""
    urdf = tmp_path / "arm.urdf"
    urdf.write_text("<robot name='expected'/>")
    cfg = config()
    cfg.task.contract.urdf_sha256 = hashlib.sha256(urdf.read_bytes()).hexdigest()
    validate_urdf(urdf, cfg)
    urdf.write_text("<robot name='changed'/>")
    with pytest.raises(ValueError, match="does not match checkpoint"):
        validate_urdf(urdf, cfg)


def add_frame(buffer, stamp, gripper=True):
    """向缓存添加一组同步的合成观测。"""
    buffer.add_image(stamp, np.zeros((3, 224, 224), dtype=np.float32))
    pose = np.eye(4)
    pose[0, 3] = stamp / 1e9
    buffer.add_pose(stamp, pose)
    if gripper:
        buffer.add_gripper(stamp, 0.4)


def test_sync_history_stale_missing_and_reset():
    """验证历史间隔、重复窗口、缺流、过期、重置与归一化范围。"""
    buffer = ObservationBuffer()
    start, delta = 1_000_000_000, 33_333_333
    add_frame(buffer, start)
    assert buffer.poll(start) is None
    add_frame(buffer, start + delta)
    window = buffer.poll(start + delta)
    assert [item.stamp_ns for item in window] == [start, start + delta]
    assert buffer.poll(start + delta) is None
    np.testing.assert_allclose(buffer.start_pose[0, 3], 1)
    buffer.reset()
    assert buffer.episode_id == 1 and buffer.start_pose is None
    add_frame(buffer, start + 2 * delta, gripper=False)
    add_frame(buffer, start + 3 * delta, gripper=False)
    assert buffer.poll(start + 3 * delta) is None
    buffer.add_gripper(start + 3 * delta, 0.5)
    assert buffer.poll(start + 3 * delta) is not None
    add_frame(buffer, start + 4 * delta)
    assert buffer.poll(start + 1_000_000_000) is None
    for value in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError):
            buffer.add_gripper(start, value)


def test_history_rejects_large_gaps_and_chooses_nearest():
    """历史必须靠近 1/30 秒；状态选择按时间距离，与到达顺序无关。"""
    buffer = ObservationBuffer()
    stamp = 1_000_000_000
    add_frame(buffer, stamp)
    assert buffer.poll(stamp) is None
    add_frame(buffer, stamp + 100_000_000)
    assert buffer.poll(stamp + 100_000_000) is None
    add_frame(buffer, stamp + 133_333_333)
    pose = np.eye(4)
    pose[0, 3] = 999
    buffer.add_pose(stamp + 120_000_000, pose)
    window = buffer.poll(stamp + 133_333_333)
    assert window[-1].pose[0, 3] != 999
