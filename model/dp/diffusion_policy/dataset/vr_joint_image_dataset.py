"""读取遥操关节图像 Zarr，构造无跨 episode 泄漏的历史观测与未来目标。"""

import copy

import numpy as np
import torch
import zarr
from diffusion_policy.common.normalize_util import get_image_identity_normalizer
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class VrJointImageDataset(BaseImageDataset):
    """提供 2 帧历史观测与 16 步绝对关节/夹爪动作，归一化仅拟合训练集。"""

    def __init__(self, dataset_path, shape_meta, seed=42, val_ratio=0.1):
        """只读打开转换结果，固定 episode 划分并建立训练样本索引。"""
        self.root = zarr.open_group(str(dataset_path), mode="r")
        if self.root.attrs.get("format") != "rm75-joint-image-v1" or not self.root.attrs.get("complete"):
            raise ValueError("Expected complete rm75-joint-image-v1 dataset")
        self.data = self.root["data"]
        self.ends = self.root["meta/episode_ends"][:]
        self.starts = np.r_[0, self.ends[:-1]]
        self.episode_names = self.root["meta/episode_names"][:].tolist()
        self.shape_meta = shape_meta
        self.obs_horizon = int(shape_meta["obs"]["camera0_rgb"]["horizon"])
        self.action_horizon = int(shape_meta["action"]["horizon"])
        if list(shape_meta["action"]["shape"]) != [8]:
            raise ValueError("Joint actions must have shape [8]")
        for key, spec in shape_meta["obs"].items():
            if int(spec["horizon"]) != self.obs_horizon:
                raise ValueError("All observations must use the same history horizon")
            expected = tuple(spec["shape"])
            actual = self.data[key].shape[1:]
            if key == "camera0_rgb":
                actual = (actual[2], actual[0], actual[1])
            if actual != expected:
                raise ValueError(f"Unexpected observation shape for {key}: {actual}")
        self.val_mask = get_val_mask(len(self.ends), val_ratio, seed)
        self.train_mask = ~self.val_mask
        self.indices = self._make_indices(self.train_mask)
        if not len(self.indices):
            raise ValueError("No training windows")

    def _make_indices(self, mask):
        """返回 (当前帧, episode 起点) 索引，舍弃未来动作不足的尾部。"""
        return [(current, int(start)) for start, end, selected in zip(self.starts, self.ends, mask)
                if selected for current in range(int(start), int(end) - self.action_horizon + 1)]

    def get_validation_dataset(self):
        """返回共享只读数据的验证视图，保留相同的训练集归一化来源。"""
        result = copy.copy(self)
        result.indices = self._make_indices(self.val_mask)
        return result

    def get_split_manifest(self):
        """返回可复现实验的训练及验证 episode 名单。"""
        return {name: [episode for episode, selected in zip(self.episode_names, mask) if selected]
                for name, mask in (("train", self.train_mask), ("validation", self.val_mask))}

    def get_normalizer(self, **kwargs):
        """只用训练 episode 的低维状态和动作拟合 [-1,1]，图像保持 [0,1]。"""
        normalizer = LinearNormalizer()
        values = {}
        for key in ("robot0_joint_pos", "robot0_gripper_position", "action"):
            values[key] = np.concatenate([self.data[key][start:end] for start, end, selected
                                          in zip(self.starts, self.ends, self.train_mask) if selected])
        normalizer.fit(values, last_n_dims=1, mode="limits")
        normalizer["camera0_rgb"] = get_image_identity_normalizer()
        return normalizer

    def get_all_actions(self):
        """返回训练 episode 的原始 8D 动作张量。"""
        return torch.from_numpy(np.concatenate([self.data["action"][start:end] for start, end, selected
                                                in zip(self.starts, self.ends, self.train_mask) if selected]))

    def __len__(self):
        """返回可用时间窗口数量。"""
        return len(self.indices)

    def __getitem__(self, index):
        """输出 obs 字典与 [16,8] 动作；开头历史通过复制首帧补齐。"""
        current, start = self.indices[index]
        first = max(start, current - self.obs_horizon + 1)
        obs = {}
        for key in self.shape_meta["obs"]:
            value = self.data[key][first:current + 1]
            if len(value) < self.obs_horizon:
                value = np.concatenate([np.repeat(value[:1], self.obs_horizon - len(value), axis=0), value])
            if key == "camera0_rgb":
                value = np.moveaxis(value, -1, 1).astype(np.float32) / 255.0
            obs[key] = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
        return {"obs": obs, "action": torch.from_numpy(self.data["action"][current:current + self.action_horizon])}
