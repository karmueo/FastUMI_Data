"""验证 URDF 正运动学、UMI 末端转换、数据划分与物理动作误差。"""

from pathlib import Path
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch
import zarr
from hydra import compose, initialize_config_dir

from convert_vr_target_to_umi import convert
from diffusion_policy.common.urdf_kinematics import UrdfKinematics
from diffusion_policy.dataset.umi_dataset import UmiDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import _pose_physical_metrics, evaluate_policy
from umi.common.pose_util import mat_to_pose10d, pose10d_to_mat


def minimal_urdf(path, joint_count=2):
    """生成带负旋转轴和末端固定偏移的解析测试机械臂。"""
    text = '<robot name="test"><link name="base_link"/>'
    for index in range(1, joint_count + 1):
        parent = 'base_link' if index == 1 else f'Link{index-1}'
        text += f'''<link name="Link{index}"/><joint name="joint{index}" type="revolute">
        <parent link="{parent}"/><child link="Link{index}"/><origin xyz="1 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 -1"/></joint>'''
    text += f'''<link name="tip"/><joint name="fixed_tip" type="fixed">
    <parent link="Link{joint_count}"/><child link="tip"/><origin xyz="1 0 0"/></joint></robot>'''
    path.write_text(text)
    return path


def test_analytic_fk_axis_origin_and_fixed_tip(tmp_path):
    """负轴 90 度关节旋转后，后续关节原点与固定工具偏移都应随链旋转。"""
    fk = UrdfKinematics(minimal_urdf(tmp_path/'arm.urdf'), ['joint1', 'joint2'], end_link='tip')
    result = fk.forward([[0, 0], [np.pi/2, 0], [np.pi/2, -np.pi/2]])
    np.testing.assert_allclose(result[:, :3, 3], [[3,0,0], [1,-2,0], [2,-1,0]], atol=1e-12)
    np.testing.assert_allclose(result[1, :3, :3], Rotation.from_euler('z', -90, degrees=True).as_matrix(), atol=1e-12)
    with pytest.raises(ValueError):
        fk.forward([np.nan, 0])
    with pytest.raises(ValueError, match='order'):
        UrdfKinematics(tmp_path/'arm.urdf', ['joint2', 'joint1'], end_link='tip')


def test_rpy_is_fixed_axis_and_precedes_joint_motion(tmp_path):
    """URDF origin 采用固定轴 RPY，并先于局部关节转动应用。"""
    path = minimal_urdf(tmp_path/'arm.urdf', 1)
    path.write_text(path.read_text().replace('rpy="0 0 0"', 'rpy="0.3 -0.2 0.7"'))
    fk = UrdfKinematics(path, ['joint1'], end_link='Link1')
    rx = Rotation.from_rotvec([0.3,0,0]).as_matrix()
    ry = Rotation.from_rotvec([0,-0.2,0]).as_matrix()
    rz = Rotation.from_rotvec([0,0,0.7]).as_matrix()
    expected = rz @ ry @ rx @ Rotation.from_rotvec([0,0,-0.4]).as_matrix()
    np.testing.assert_allclose(fk.forward([0.4])[:3,:3], expected, atol=1e-12)


def test_pose6d_roundtrip_and_physical_error():
    """验证 rotation_6d 行编码往返，以及米制位置/角度/夹爪误差。"""
    matrix = np.eye(4)[None].repeat(3, axis=0)
    matrix[:, :3, :3] = Rotation.from_euler('xyz', [[0,0,0], [0.2,1,-2], [-2,0.5,3]]).as_matrix()
    np.testing.assert_allclose(pose10d_to_mat(mat_to_pose10d(matrix)), matrix, atol=1e-12)
    target = torch.tensor(np.r_[np.zeros(3), [1,0,0,0,1,0], 0], dtype=torch.float32)[None,None]
    predicted = target.clone()
    predicted[..., :3] = torch.tensor([0.03,0.04,0])
    predicted[..., 3:9] = torch.tensor([0,-1,0,1,0,0])
    predicted[..., 9] = 0.5
    metrics = _pose_physical_metrics(predicted, target)
    assert metrics['val_position_mse_m2'].item() == pytest.approx(0.0025)
    assert metrics['val_rotation_error_deg'].item() == pytest.approx(90)
    assert metrics['val_gripper_mse'].item() == pytest.approx(0.25)


def synthetic_joint_zarr(path):
    """构造三段关节 Zarr，每段动作目标与实际状态不同。"""
    root = zarr.open_group(str(path), mode='w')
    root.attrs.update(format='rm75-joint-image-v1', complete=True, frequency=30)
    data = root.create_group('data')
    count = 60
    data.create_dataset('camera0_rgb', data=np.zeros((count, 8, 8, 3), dtype='u1'))
    data.create_dataset('timestamp', data=np.arange(count)/30)
    data.create_dataset('source_image_index', data=np.arange(count))
    state = np.zeros((count, 7), dtype='f4')
    for index in range(count):
        state[index, 0] = index/100
    data.create_dataset('robot0_joint_pos', data=state)
    data.create_dataset('robot0_gripper_position', data=np.full((count,1), 0.4, dtype='f4'))
    action = np.c_[state + 0.1, np.full(count, 0.7)].astype('f4')
    data.create_dataset('action', data=action)
    meta = root.create_group('meta')
    meta.create_dataset('episode_ends', data=np.array([20,40,60]))
    meta.create_dataset('episode_names', data=np.array(['episode_0','episode_1','episode_2']))
    return root


def umi_shape():
    """组合原 UMI 五键 shape_meta，将测试图像尺寸缩小。"""
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/'diffusion_policy/config')):
        cfg = compose(config_name='train_diffusion_unet_timm_vr_umi_workspace')
    cfg.task.shape_meta.obs.camera0_rgb.shape = [3,8,8]
    return cfg.task.shape_meta


def test_conversion_and_explicit_split(tmp_path):
    """状态及目标独立转换，夹爪保持原值，显式划分不能跨集或跨 episode。"""
    joint = synthetic_joint_zarr(tmp_path/'joint.zarr')
    urdf = minimal_urdf(tmp_path/'arm.urdf', 7)
    convert(tmp_path/'joint.zarr', urdf, tmp_path/'umi.zarr')
    output = zarr.open_group(str(tmp_path/'umi.zarr'), mode='r')
    assert output.attrs['gripper_representation'] == 'normalized_0_1'
    assert output['data/action'].shape == (60,7)
    np.testing.assert_array_equal(output['data/action'][:,-1], joint['data/action'][:,-1])
    np.testing.assert_array_equal(output['data/robot0_gripper_width'][:], joint['data/robot0_gripper_position'][:])
    assert not np.allclose(output['data/action'][:,:3], output['data/robot0_eef_pos'][:])
    dataset = UmiDataset(umi_shape(), str(tmp_path/'umi.zarr'), pose_repr={'obs_pose_repr':'relative','action_pose_repr':'relative'},
                         train_episode_indices=[0], val_episode_indices=[2], start_pose_noise_std=0, normalizer_num_workers=0)
    assert dataset.dataset_attrs['urdf_sha256'] == output.attrs['urdf_sha256']
    assert len(dataset) == 5
    assert len(dataset.get_validation_dataset()) == 5
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample['action'].shape == (16,10)
        assert set(sample['obs']) == set(umi_shape()['obs'])
    np.testing.assert_array_equal(dataset[0]['action'], dataset[0]['action'])
    # 最新观测相对于自身的位姿为恒等；动作仍使用目标关节 FK。
    np.testing.assert_allclose(dataset[0]['obs']['robot0_eef_pos'][-1], 0, atol=1e-6)
    before = dataset.get_normalizer()['action'].params_dict['scale'].clone()
    # MemoryStore 副本中验证 episode 改动不能影响训练归一化。
    dataset.replay_buffer['action'][40:60, :3] = 1000
    after = dataset.get_normalizer()['action'].params_dict['scale']
    assert torch.equal(before, after)
    with pytest.raises(ValueError, match='disjoint'):
        UmiDataset(umi_shape(), str(tmp_path/'umi.zarr'), train_episode_indices=[0], val_episode_indices=[0])


def test_validation_sampling_does_not_change_loss_rng():
    """增加动作采样不能改变验证 loss 的随机噪声序列。"""
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    class Policy:
        """提供消耗随机数的最小验证策略。"""
        def __init__(self):
            self.normalizer = LinearNormalizer()
            self.normalizer.fit({'action': torch.zeros(2,16,8)})
        def __call__(self, batch):
            return torch.rand(())
        def predict_action(self, obs):
            return {'action_pred':torch.randn(2,16,8)}
    batch = {'obs':{}, 'action':torch.zeros(2,16,8)}
    policy = Policy()
    without = evaluate_policy(policy, [batch,batch], 'cpu', 'joint8', sample=False)
    with_sample = evaluate_policy(policy, [batch,batch], 'cpu', 'joint8', sample=True)
    assert without['val_loss'] == with_sample['val_loss']


def test_validation_metrics_reduce_weighted_totals_across_processes():
    """跨 rank 聚合样本加权总量与数量，返回完整验证集均值。"""
    class Policy:
        """返回本 rank 固定损失的最小策略。"""

        def __call__(self, batch):
            """用 batch 中的标量作为损失。"""
            return batch["loss"]

    class Accelerator:
        """模拟另一个 rank 含三个样本、损失总量为 12。"""

        def reduce(self, packed, reduction):
            """将远端加权总量和样本数合并到本 rank。"""
            assert reduction == "sum"
            return packed + torch.tensor([12.0, 3.0], dtype=packed.dtype)

    batch = {"obs": {}, "action": torch.zeros(2, 16, 8), "loss": torch.tensor(1.0)}
    metrics = evaluate_policy(Policy(), [batch], "cpu", accelerator=Accelerator())
    assert metrics["val_loss"] == pytest.approx((2 + 12) / 5)


def test_best_loss_roundtrips_through_checkpoint(tmp_path):
    """latest checkpoint 恢复 workspace 时保留中断前的最佳验证分数。"""
    cfg = umi_shape()
    workspace = TrainDiffusionUnetImageWorkspace.__new__(TrainDiffusionUnetImageWorkspace)
    BaseWorkspace.__init__(workspace, cfg, output_dir=str(tmp_path))
    workspace.global_step = 12
    workspace.epoch = 3
    workspace.best_loss = 0.125
    checkpoint = workspace.save_checkpoint(use_thread=False)

    restored = TrainDiffusionUnetImageWorkspace.__new__(TrainDiffusionUnetImageWorkspace)
    BaseWorkspace.__init__(restored, cfg, output_dir=str(tmp_path))
    restored.best_loss = float("inf")
    restored.load_checkpoint(checkpoint)
    assert restored.best_loss == pytest.approx(0.125)


@pytest.mark.skipif(not os.environ.get('RM75_URDF'), reason='Local RM75 URDF unavailable')
def test_real_rm75_zero_pose():
    """RM75 零位七轴原点沿 z 轴累加到 0.8505 m，姿态为恒等。"""
    fk = UrdfKinematics(os.environ['RM75_URDF'], [f'joint{i}' for i in range(1,8)])
    pose = fk.forward(np.zeros(7))
    np.testing.assert_allclose(pose[:3,3], [0,0,0.8505], atol=1e-10)
    np.testing.assert_allclose(pose[:3,:3], np.eye(3), atol=1e-12)


def test_decode_policy_batch_and_horizon():
    """评估器须正确解码 [B,16,10]，避免原工具仅支持二维数组的广播错误。"""
    from evaluate_vr_umi import decode_pose_actions
    rotations = Rotation.from_euler('xyz', [[0.2,0.1,-0.7],[-0.5,0.4,1.1]]).as_matrix()
    matrices = np.broadcast_to(np.eye(4), (2,16,4,4)).copy()
    matrices[:, :, :3, :3] = rotations[:,None]
    matrices[:, :, :3, 3] = np.array([0.1,0.2,0.3])
    actions = np.concatenate([mat_to_pose10d(matrices), np.ones((2,16,1))], axis=-1)
    np.testing.assert_allclose(decode_pose_actions(actions), matrices, atol=1e-12)
