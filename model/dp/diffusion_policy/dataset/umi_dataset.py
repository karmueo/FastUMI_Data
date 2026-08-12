"""提供 UMI ReplayBuffer 数据集及目录/ZIP Zarr 容器读取适配。"""

import copy
import contextlib
import os
import pathlib
import shutil
import zipfile
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import scipy.spatial.transform as st
import torch
import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from filelock import FileLock
from threadpoolctl import threadpool_limits
from tqdm import tqdm, trange
from umi.common.pose_util import mat_to_pose10d, pose_to_mat

register_codecs()


@contextlib.contextmanager
def _open_replay_store(dataset_path):
    """以只读方式打开目录或 ZIP 格式的 Zarr ReplayBuffer。

    Args:
        dataset_path: 目录 Zarr 或 ZIP Zarr 文件路径。

    Yields:
        zarr.storage.BaseStore: 已验证根组存在的只读 Store。

    Raises:
        FileNotFoundError: 目标路径不存在。
        ValueError: 目标路径不是有效的 Zarr 容器。
    """
    # 展开用户目录并规范化路径，便于错误信息和日志复现。
    path = pathlib.Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Zarr dataset does not exist: {path}")

    # 记录当前打开的 Store，确保异常路径也会释放 ZIP 文件句柄。
    store = None
    try:
        try:
            if path.is_dir():
                store = zarr.DirectoryStore(str(path))
            else:
                store = zarr.ZipStore(str(path), mode="r")
            zarr.open_group(store=store, mode="r")
        except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Invalid Zarr dataset container: {path}") from exc

        # 将消费者代码放在容器异常捕获范围之外，保留其原始异常类型和消息。
        yield store
    finally:
        # DirectoryStore 没有 close；ZipStore 需要显式关闭底层归档文件。
        close = getattr(store, "close", None)
        if close is not None:
            close()


def _extract_robot_action_pose(action: np.ndarray, robot_id: int, num_robot: int):
    """从原始 action 数组中提取单个机器人的 xyz + rotvec 位姿和夹爪宽度。

    Args:
        action: Zarr 数据集中的原始 action 序列，最后一维为动作维度。
        robot_id: 当前机器人索引。
        num_robot: 数据集中机器人数量。

    Returns:
        tuple[np.ndarray, np.ndarray]: 第一项为 `[x, y, z, rotvec_x, rotvec_y, rotvec_z]`
        位姿数组，第二项为 `[gripper_width]` 夹爪宽度数组。

    Raises:
        ValueError: action 维度不是每机器人 7 维 rotvec 或 8 维 wxyz 四元数时抛出。
    """
    # 每个机器人动作维度，用于兼容新版 rotvec 和旧版 wxyz 数据集。
    action_stride = action.shape[-1] // num_robot
    if action.shape[-1] % num_robot != 0 or action_stride not in (7, 8):
        raise ValueError(
            "UMI action must have 7 values (xyz + rotvec + gripper) or "
            "8 values (xyz + wxyz quaternion + gripper) per robot, got "
            f"{action.shape[-1]} values for {num_robot} robot(s)."
        )

    # 当前机器人的动作切片起点。
    start = action_stride * robot_id
    if action_stride == 7:
        return action[..., start : start + 6], action[..., start + 6 : start + 7]

    # 旧版数据使用 wxyz 四元数；scipy 需要 xyzw 顺序。
    action_pose = np.zeros(action.shape[:-1] + (6,), dtype=action.dtype)
    action_pose[..., :3] = action[..., start : start + 3]
    quat_wxyz = action[..., start + 3 : start + 7]
    action_pose[..., 3:6] = st.Rotation.from_quat(
        np.concatenate([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], axis=-1)
    ).as_rotvec()
    return action_pose, action[..., start + 7 : start + 8]


class UmiDataset(BaseDataset):
    """读取 FastUMI Zarr 数据并生成 canonical 训练样本。"""

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str] = None,
        pose_repr: dict = {},
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        repeat_frame_prob: float = 0.0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_duration: Optional[float] = None,
        normalizer_num_workers: int = 32,
    ):
        """初始化数据集及可配置的归一化统计 DataLoader。"""
        self.pose_repr = pose_repr
        # 使用相对还是绝对位姿
        self.obs_pose_repr = self.pose_repr.get("obs_pose_repr", "rel")
        self.action_pose_repr = self.pose_repr.get("action_pose_repr", "rel")

        if cache_dir is None:
            # load into memory store
            with _open_replay_store(dataset_path) as replay_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=replay_store, store=zarr.MemoryStore()
                )
        else:
            # TODO: refactor into a stand alone function?
            # determine path name
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split(".")[0]
            cache_name = "_".join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + ".zarr.mdb")
            lock_path = cache_dir.joinpath(cache_name + ".lock")

            # load cached file
            print("Acquiring lock on cache.")
            with FileLock(lock_path):
                # cache does not exist
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(
                            str(cache_path),
                            writemap=True,
                            metasync=False,
                            sync=False,
                            map_async=True,
                            lock=False,
                        ) as lmdb_store:
                            with _open_replay_store(dataset_path) as replay_store:
                                print(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=replay_store, store=lmdb_store
                                )
                        print("Cache written to disk!")
                    except Exception as e:
                        if cache_path.exists():
                            shutil.rmtree(cache_path)
                        raise e

            # open read-only lmdb store
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(group=zarr.group(store))

        self.num_robot = 0
        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

            if key.endswith("eef_pos"):
                self.num_robot += 1

            # solve obs_horizon
            horizon = shape_meta["obs"][key]["horizon"]
            key_horizon[key] = horizon

            # solve latency_steps
            latency_steps = shape_meta["obs"][key]["latency_steps"]
            key_latency_steps[key] = latency_steps

            # solve down_sample_steps
            down_sample_steps = shape_meta["obs"][key]["down_sample_steps"]
            key_down_sample_steps[key] = down_sample_steps

        # solve action
        key_horizon["action"] = shape_meta["action"]["horizon"]
        key_latency_steps["action"] = shape_meta["action"]["latency_steps"]
        key_down_sample_steps["action"] = shape_meta["action"]["down_sample_steps"]

        # 生成 mask，mask 的部分作为验证集
        val_mask = get_val_mask(n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not "wrt" in key:
                self.sampler_lowdim_keys.append(key)

        for key in replay_buffer.keys():
            if key.endswith("_demo_start_pose") or key.endswith("_demo_end_pose"):
                self.sampler_lowdim_keys.append(key)
                query_key = key.split("_")[0] + "_eef_pos"
                key_horizon[key] = shape_meta["obs"][query_key]["horizon"]
                key_latency_steps[key] = shape_meta["obs"][query_key]["latency_steps"]
                key_down_sample_steps[key] = shape_meta["obs"][query_key]["down_sample_steps"]

        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration,
        )
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        # 归一化统计阶段的 DataLoader worker 数量，smoke 可显式设为零。
        self.normalizer_num_workers = int(normalizer_num_workers)
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """遍历训练样本并计算观测与动作的归一化参数。"""
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + ["action"]}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=self.normalizer_num_workers,
        )
        for batch in tqdm(dataloader, desc="iterating dataset to get normalization"):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch["obs"][key]))
            data_cache["action"].append(copy.deepcopy(batch["action"]))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B * T, D)

        # action
        assert data_cache["action"].shape[-1] % self.num_robot == 0
        dim_a = data_cache["action"].shape[-1] // self.num_robot
        action_normalizers = list()
        for i in range(self.num_robot):
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(data_cache["action"][..., i * dim_a : i * dim_a + 3])
                )
            )  # pos
            action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(data_cache["action"][..., i * dim_a + 3 : (i + 1) * dim_a - 1])
                )
            )  # rot
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(data_cache["action"][..., (i + 1) * dim_a - 1 : (i + 1) * dim_a])
                )
            )  # gripper

        normalizer["action"] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])

            if key.endswith("pos") or "pos_wrt" in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("pos_abs"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("rot_axis_angle") or "rot_axis_angle_wrt" in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("gripper_width"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.0
            # T,C,H,W
            del data[key]
        for key in self.sampler_lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]

        # generate relative pose between two ees
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(
                np.concatenate(
                    [
                        obs_dict[f"robot{robot_id}_eef_pos"],
                        obs_dict[f"robot{robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )
            for other_robot_id in range(self.num_robot):
                if robot_id == other_robot_id:
                    continue
                if not f"robot{robot_id}_eef_pos_wrt{other_robot_id}" in self.lowdim_keys:
                    continue
                other_pose_mat = pose_to_mat(
                    np.concatenate(
                        [
                            obs_dict[f"robot{other_robot_id}_eef_pos"],
                            obs_dict[f"robot{other_robot_id}_eef_rot_axis_angle"],
                        ],
                        axis=-1,
                    )
                )
                rel_obs_pose_mat = convert_pose_mat_rep(
                    pose_mat, base_pose_mat=other_pose_mat[-1], pose_rep="relative", backward=False
                )
                rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
                obs_dict[f"robot{robot_id}_eef_pos_wrt{other_robot_id}"] = rel_obs_pose[:, :3]
                obs_dict[f"robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}"] = rel_obs_pose[
                    :, 3:
                ]

        # generate relative pose with respect to episode start
        for robot_id in range(self.num_robot):
            # HACK: add noise to episode start pose
            if (f"robot{other_robot_id}_eef_pos_wrt_start" not in self.shape_meta["obs"]) and (
                f"robot{other_robot_id}_eef_rot_axis_angle_wrt_start" not in self.shape_meta["obs"]
            ):
                continue

            # convert pose to mat
            pose_mat = pose_to_mat(
                np.concatenate(
                    [
                        obs_dict[f"robot{robot_id}_eef_pos"],
                        obs_dict[f"robot{robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )

            # get start pose
            start_pose = obs_dict[f"robot{robot_id}_demo_start_pose"][0]
            # HACK: add noise to episode start pose
            start_pose += np.random.normal(
                scale=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05], size=start_pose.shape
            )
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat, base_pose_mat=start_pose_mat, pose_rep="relative", backward=False
            )

            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # HACK: add noise to episode start pose
            # obs_dict[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict[f"robot{robot_id}_eef_rot_axis_angle_wrt_start"] = rel_obs_pose[:, 3:]

        del_keys = list()
        for key in obs_dict:
            if key.endswith("_demo_start_pose") or key.endswith("_demo_end_pose"):
                del_keys.append(key)
        for key in del_keys:
            del obs_dict[key]

        actions = list()
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(
                np.concatenate(
                    [
                        obs_dict[f"robot{robot_id}_eef_pos"],
                        obs_dict[f"robot{robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )
            action_pose, action_gripper = _extract_robot_action_pose(
                data["action"], robot_id=robot_id, num_robot=self.num_robot
            )
            action_mat = pose_to_mat(action_pose)

            # solve relative obs
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat, base_pose_mat=pose_mat[-1], pose_rep=self.obs_pose_repr, backward=False
            )
            action_pose_mat = convert_pose_mat_rep(
                action_mat,
                base_pose_mat=pose_mat[-1],
                pose_rep=self.action_pose_repr,
                backward=False,
            )

            # convert pose to pos + rot6d representation
            obs_pose = mat_to_pose10d(obs_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)

            actions.append(np.concatenate([action_pose, action_gripper], axis=-1))

            # generate data
            obs_dict[f"robot{robot_id}_eef_pos"] = obs_pose[:, :3]
            obs_dict[f"robot{robot_id}_eef_rot_axis_angle"] = obs_pose[:, 3:]

        data["action"] = np.concatenate(actions, axis=-1)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data
