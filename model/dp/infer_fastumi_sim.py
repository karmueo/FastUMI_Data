"""提供 legacy FastUMI 图像 DP checkpoint 在 ROS 2 仿真闭环中的推理入口。

该模块负责把仿真侧 RGB 图像观测适配为 FastUMI 训练使用的
`camera0_rgb` 张量, 调用本地或显式指定的 Diffusion Policy checkpoint 推理, 再把
7 维绝对末端动作转换为当前 Isaac Sim 控制器使用的 8 维动作格式。
"""

import argparse
import inspect
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

DEFAULT_DIFFUSION_POLICY_ROOT = str(Path(__file__).resolve().parent)
"""默认的本地 Diffusion Policy 项目根目录。"""

FASTUMI_RGB_KEY = "camera0_rgb"
"""FastUMI 图像策略使用的 RGB 观测键。"""

EXPECTED_ACTION_SHAPE = [7]
"""FastUMI checkpoint 期望的 action shape。"""

FASTUMI_ACTION_DIM = EXPECTED_ACTION_SHAPE[0]
"""FastUMI action 的维度。"""

POLICY_ACTION_NDIM = 2
"""单次 policy 输出动作序列的数组维度。"""

def ensure_diffusion_policy_root_first(diffusion_policy_root):
    """把本地或显式指定的 Diffusion Policy 根目录插入到 `sys.path[0]`。

    Args:
        diffusion_policy_root: 本地或兼容项目根目录路径。

    Returns:
        str: 插入后的绝对或原样路径字符串。
    """
    root_path = str(Path(diffusion_policy_root).expanduser())
    sys.path = [path for path in sys.path if path != root_path]
    sys.path.insert(0, root_path)
    return root_path


def _torch_load_checkpoint(ckpt_path):
    """使用 dill 读取外部 Diffusion Policy checkpoint。

    Args:
        ckpt_path: checkpoint 文件路径。

    Returns:
        dict: `torch.load` 读取出的 checkpoint payload。
    """
    import dill  # noqa: PLC0415

    load_kwargs = {
        "pickle_module": dill,
        "map_location": "cpu",
    }
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    with open(ckpt_path, "rb") as ckpt_file:
        return torch.load(ckpt_file, **load_kwargs)


def _shape_to_list(shape_value):
    """把 OmegaConf/ListConfig 或 tuple shape 转成普通列表。

    Args:
        shape_value: 配置中的 shape 字段。

    Returns:
        list: 普通 Python 列表形式的 shape。
    """
    return list(shape_value)


def validate_fastumi_shape_meta(cfg):
    """校验 checkpoint 的 obs/action shape 是否符合 FastUMI 图像部署约定。

    Args:
        cfg: checkpoint payload 中的 Hydra 配置对象。

    Raises:
        ValueError: shape_meta 不符合当前适配器要求时抛出。
    """
    shape_meta = cfg.shape_meta
    obs_meta = shape_meta.obs
    obs_keys = set(obs_meta.keys())
    if obs_keys != {FASTUMI_RGB_KEY}:
        raise ValueError(
            f"FastUMI obs keys must be only {{{FASTUMI_RGB_KEY}}}, got {obs_keys}"
        )

    action_shape = _shape_to_list(shape_meta.action.shape)
    if action_shape != EXPECTED_ACTION_SHAPE:
        raise ValueError(
            f"FastUMI action shape must be {EXPECTED_ACTION_SHAPE}, got {action_shape}"
        )


def resolve_device(device_name):
    """根据 CUDA 可用性解析推理设备。

    Args:
        device_name: CLI 传入的 torch device 字符串。

    Returns:
        torch.device: 可用于推理的设备。
    """
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is not available, fallback from {device_name} to cpu.")
        return torch.device("cpu")
    return torch.device(device_name)


def load_fastumi_policy(ckpt_path, diffusion_policy_root, device):
    """加载 legacy FastUMI Diffusion Policy checkpoint 并返回可推理 policy。

    Args:
        ckpt_path: FastUMI checkpoint 文件路径。
        diffusion_policy_root: 本地或兼容项目根目录。
        device: 推理设备。

    Returns:
        tuple: `(policy, cfg)`, 其中 policy 已切换到 eval 并移动到目标设备。
    """
    ensure_diffusion_policy_root_first(diffusion_policy_root)

    import hydra  # noqa: PLC0415

    payload = _torch_load_checkpoint(ckpt_path)
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval().to(device)
    validate_fastumi_shape_meta(cfg)

    print(f"checkpoint: {ckpt_path}")
    print(f"diffusion_policy_root: {sys.path[0]}")
    print(f"workspace: {workspace}")
    print(f"obs keys: {list(cfg.shape_meta.obs.keys())}")
    print(f"action shape: {_shape_to_list(cfg.shape_meta.action.shape)}")
    print(f"n_obs_steps: {cfg.n_obs_steps}, n_action_steps: {cfg.n_action_steps}")
    return policy, cfg


def decode_rgb_image(rgb_msg):
    """把 ROS 2 `sensor_msgs/Image` 解码为 HWC uint8 RGB 数组。

    Args:
        rgb_msg: ROS 2 RGB 图像消息。

    Returns:
        np.ndarray: 形状为 `[height, width, 3]` 的 RGB uint8 图像。
    """
    height = int(rgb_msg.height)
    width = int(rgb_msg.width)
    step = int(getattr(rgb_msg, "step", width * 3))
    encoding = getattr(rgb_msg, "encoding", "rgb8").lower()

    raw_bytes = np.frombuffer(rgb_msg.data, dtype=np.uint8)
    if step > width * 3:
        row_major = raw_bytes.reshape(height, step)
        image = row_major[:, : width * 3].reshape(height, width, 3)
    else:
        image = raw_bytes[: height * width * 3].reshape(height, width, 3)

    if encoding == "bgr8":
        image = image[..., ::-1]
    return np.ascontiguousarray(image)


def center_crop_to_aspect_ratio(image, target_width, target_height):
    """按目标宽高比对图像进行中心裁剪。

    Args:
        image: 输入 RGB 图像。
        target_width: 目标宽度。
        target_height: 目标高度。

    Returns:
        np.ndarray: 中心裁剪后的 RGB 图像。
    """
    height, width = image.shape[:2]
    target_ratio = target_width / target_height
    current_ratio = width / height

    if current_ratio > target_ratio:
        crop_width = round(height * target_ratio)
        left = (width - crop_width) // 2
        return image[:, left : left + crop_width]

    crop_height = round(width / target_ratio)
    top = (height - crop_height) // 2
    return image[top : top + crop_height, :]


def preprocess_rgb_image(image, img_size):
    """把单帧 RGB 图像裁剪、缩放并归一化为 CHW float32 数组。

    Args:
        image: HWC uint8 RGB 图像。
        img_size: 输出正方形图像边长。

    Returns:
        np.ndarray: 形状为 `[3, img_size, img_size]`, 数值范围 `[0, 1]`。
    """
    cropped_image = center_crop_to_aspect_ratio(image, img_size, img_size)
    pil_image = Image.fromarray(cropped_image)
    resized_image = pil_image.resize((img_size, img_size), Image.Resampling.LANCZOS)
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    return np.moveaxis(image_array, -1, 0)


class FastUmiFrameBuffer:
    """维护 FastUMI 图像策略所需的历史 RGB 帧。"""

    def __init__(self, buffer_size):
        """初始化固定长度帧缓存。

        Args:
            buffer_size: 历史观测窗口长度。
        """
        self.buffer_size = int(buffer_size)
        """历史观测窗口长度。"""
        self.buffer = deque(maxlen=self.buffer_size)
        """按时间顺序保存的 RGB 帧缓存。"""

    def update(self, frame):
        """写入新帧, 首帧会填满整个历史窗口。

        Args:
            frame: HWC uint8 RGB 图像。
        """
        if not self.buffer:
            self.buffer.extend([frame] * self.buffer_size)
            return
        self.buffer.append(frame)

    def get_all_frames(self):
        """读取当前历史窗口内的全部帧。

        Returns:
            list: 按时间顺序排列的 RGB 帧列表。
        """
        return list(self.buffer)


def build_obs_dict(frame_buffer, img_size, device):
    """从图像历史缓存构造 FastUMI policy 输入字典。

    Args:
        frame_buffer: FastUMI 图像历史缓存。
        img_size: 输出图像边长。
        device: 推理设备。

    Returns:
        dict: 只包含 `camera0_rgb` 的 policy 输入字典。
    """
    image_history = []
    for frame in frame_buffer.get_all_frames():
        image_history.append(preprocess_rgb_image(frame, img_size=img_size))
    rgb_array = np.stack(image_history, axis=0).astype(np.float32)
    rgb_tensor = torch.from_numpy(rgb_array).unsqueeze(0).to(device)
    return {FASTUMI_RGB_KEY: rgb_tensor}


def fastumi_action_to_sim_action(action):
    """把 FastUMI 7 维绝对动作转换为仿真侧 8 维动作。

    Args:
        action: `[x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper_width]`。

    Returns:
        np.ndarray: `[x, y, z, w, qx, qy, qz, gripper_width]`。
    """
    action_array = np.asarray(action, dtype=np.float64)
    if action_array.shape[-1] != FASTUMI_ACTION_DIM:
        raise ValueError(
            f"FastUMI action must have {FASTUMI_ACTION_DIM} values, got shape "
            f"{action_array.shape}"
        )

    position = action_array[:3]
    rotvec = action_array[3:6]
    gripper_width = action_array[6:7]
    quat_xyzw = Rotation.from_rotvec(rotvec).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return np.concatenate([position, quat_wxyz, gripper_width])


def predict_fastumi_actions(policy, obs_dict):
    """执行 FastUMI policy 推理并取出 `[T, 7]` 动作数组。

    Args:
        policy: 外部 Diffusion Policy policy 对象。
        obs_dict: FastUMI 观测输入字典。

    Returns:
        np.ndarray: 形状为 `[T, 7]` 的动作序列。
    """
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    actions = result["action"][0].detach().to("cpu").numpy()
    if actions.ndim != POLICY_ACTION_NDIM or actions.shape[-1] != FASTUMI_ACTION_DIM:
        raise ValueError(f"Policy action must have shape [T, 7], got {actions.shape}")
    return actions


class FastUmiSimInferencer:
    """把 ROS 2 同步观测转换为 FastUMI policy 推理动作的可调用对象。"""

    def __init__(self, policy, device, n_obs_steps, n_action_steps, img_size):
        """初始化推理器和动作缓存。

        Args:
            policy: 已加载的 FastUMI policy。
            device: 推理设备。
            n_obs_steps: 图像历史窗口长度。
            n_action_steps: 每次 policy 推理后执行的动作步数。
            img_size: policy 输入图像边长。
        """
        self.policy = policy
        """FastUMI policy 对象。"""
        self.device = device
        """policy 推理设备。"""
        self.n_action_steps = int(n_action_steps)
        """每次推理后发布的动作步数。"""
        self.img_size = int(img_size)
        """policy 输入图像边长。"""
        self.frame_buffer = FastUmiFrameBuffer(buffer_size=n_obs_steps)
        """RGB 历史帧缓存。"""
        self.action_buffer = deque(maxlen=self.n_action_steps)
        """缓存 policy 一次输出中的剩余动作。"""

    def __call__(self, rgb_msg, joint_msg, ee_pose_msg):
        """处理 ROS 2 同步观测并返回当前要发布的仿真动作。

        Args:
            rgb_msg: RGB 图像消息。
            joint_msg: 关节状态消息, FastUMI 图像策略暂不使用。
            ee_pose_msg: 末端位姿消息, FastUMI 图像策略暂不使用。

        Returns:
            np.ndarray: 8 维仿真侧动作。
        """
        del joint_msg, ee_pose_msg

        rgb_image = decode_rgb_image(rgb_msg)
        self.frame_buffer.update(rgb_image)

        if self.action_buffer:
            return self.action_buffer.popleft()

        obs_dict = build_obs_dict(
            frame_buffer=self.frame_buffer,
            img_size=self.img_size,
            device=self.device,
        )
        predicted_actions = predict_fastumi_actions(self.policy, obs_dict)
        selected_actions = predicted_actions[: self.n_action_steps]
        if len(selected_actions) == 0:
            raise RuntimeError("FastUMI policy returned no action.")

        sim_actions = [fastumi_action_to_sim_action(action) for action in selected_actions]
        self.action_buffer.extend(sim_actions[1:])
        target_action = sim_actions[0]

        print("actions:", target_action)
        print("actions shape:", target_action.shape)
        return target_action


def parse_args():
    """解析 FastUMI 仿真推理 CLI 参数。

    Returns:
        argparse.Namespace: 命令行参数命名空间。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="FastUMI checkpoint path.")
    parser.add_argument(
        "--diffusion_policy_root",
        type=str,
        default=DEFAULT_DIFFUSION_POLICY_ROOT,
        help="本地项目根目录；仅在兼容旧 checkpoint 时显式覆盖。",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch inference device.")
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=4,
        help="How many predicted actions to execute before next inference.",
    )
    parser.add_argument("--img_size", type=int, default=224, help="FastUMI image size.")
    return parser.parse_args()


def main():
    """加载 FastUMI checkpoint 并启动 ROS 2 推理节点。"""
    args = parse_args()
    device = resolve_device(args.device)
    policy, cfg = load_fastumi_policy(
        ckpt_path=args.ckpt_path,
        diffusion_policy_root=args.diffusion_policy_root,
        device=device,
    )
    inferencer = FastUmiSimInferencer(
        policy=policy,
        device=device,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=args.n_action_steps,
        img_size=args.img_size,
    )

    from ros_receiver import RgbJointEePoseActionNode  # noqa: PLC0415

    RgbJointEePoseActionNode(process_fn=inferencer)


if __name__ == "__main__":
    main()
