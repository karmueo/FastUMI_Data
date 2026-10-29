"""LeRobot TCP 相对位姿后端；复用 ``dp_infer`` 共享运行层的 ROS 生命周期与操作协议。

共享运行层的观测位姿为 ``FK(q) @ inv(C)``，发布前对序列右乘 ``C``。本后端令
``C = inverse(T_Link7_TCP)``：观测即 ``base_link → TCP``（模型契约），发布即 ``base_link → Link7``
（RM75 控制器契约）。``T_Link7_TCP`` 由部署外参 ``T_cam_end.yaml`` 与 ``T_d_cam_tcp.yaml`` 计算，
并与部署档中的示教 ``umi_camera_from_tcp`` 比较轴向。
"""

from pathlib import Path

import numpy as np

from dp_infer.common import compressed_image_to_rgb
from dp_infer.core import letterbox_rgb, load_processors
from dp_infer.runtime import PolicyRuntimeNode, run_node

from . import core, deployment
from .config import validate_backend


class LerobotBackend:
    """独立模型契约：两帧 TCP 状态 + 图像输入，输出 8 个 30 Hz 绝对 TCP 目标。"""

    node_name = "dp_infer_lerobot"
    parameter_defaults = {
        "num_inference_steps": 16, "deployment_config": "",
        # pytorch 读取 checkpoint；tensorrt_fp32/tensorrt_fp16 读取 export_dir 本机模型包中的 engine。
        "backend": "pytorch", "export_dir": "",
        # 空字符串沿用训练 DDPM；DDIM 复用同一噪声表，可用较少步数降低延迟。
        "noise_scheduler": "DDIM",
        # 部署相机←Link7（T_cam_end.yaml）与部署相机←TCP（T_d_cam_tcp.yaml），均必填。
        "end_camera_extrinsic": "", "deploy_tcp_extrinsic": "",
        "max_tcp_axis_deg": deployment.DEFAULT_MAX_TCP_AXIS_DEG,
        "record_dir_name": "dp_infer_lerobot", "task_control_enabled": True,
    }
    point_count = core.ACTION_STEPS
    # 首点是观测时刻的预测动作，不含额外的起始锚点。
    sequence_starts_with_anchor = False
    visual_guard = False
    # 与训练数据转换一致：cv2 INTER_AREA 等比缩放 + 黑边 letterbox 到 224。
    preprocess_image = staticmethod(letterbox_rgb)
    decode_compressed = staticmethod(compressed_image_to_rgb)
    build_observations = staticmethod(core.build_observations)
    correct_sequence = staticmethod(core.correct_sequence)
    result_is_usable = staticmethod(core.result_is_usable)

    def __init__(self):
        """部署档在创建引擎或校验部署时加载。"""
        self.profile = None

    def _load_profile(self, parameter):
        """读取并缓存部署档；格式或契约不符时抛出 ValueError。"""
        if self.profile is None:
            self.profile = deployment.load_profile(parameter("deployment_config"))
        return self.profile

    def create_engine(self, parameter):
        """校验部署档与模型指纹后才创建真实模型；按 ``backend`` 要求 checkpoint 或 export_dir。"""
        backend, checkpoint, export_dir = (parameter(name) for name in ("backend", "checkpoint", "export_dir"))
        if validate_backend(backend) is None:
            if not checkpoint or not Path(checkpoint).is_dir():
                raise ValueError("checkpoint must name an existing pretrained_model directory")
        elif not export_dir or not Path(export_dir).is_dir():
            raise ValueError("TensorRT backends require export_dir naming a TensorRT model package")
        from .engine import PolicyEngine

        processors = load_processors([
            item.strip() for item in parameter("postprocessors").split(",") if item.strip()])
        return PolicyEngine(checkpoint, self._load_profile(parameter), parameter("device"),
                            parameter("noise_scheduler"), parameter("num_inference_steps"), processors,
                            backend, export_dir)

    def validate_deployment(self, urdf_path, engine, parameter):
        """注入假引擎时仍校验部署契约及训练 URDF 指纹（FK 必须与训练数据一致）。"""
        deployment.validate_urdf(urdf_path, self._load_profile(parameter))

    def load_correction(self, parameter):
        """返回 ``C = inverse(T_Link7_TCP)``；忽略 ``link7_correction_file``。"""
        self.link7_from_tcp, offset_mm, angle_deg = deployment.load_tool_offset(
            self._load_profile(parameter), parameter("end_camera_extrinsic"),
            parameter("deploy_tcp_extrinsic"), float(parameter("max_tcp_axis_deg")))
        self.tcp_log = (f"Deployment TCP: Link7->TCP {np.linalg.norm(self.link7_from_tcp[:3, 3]) * 1000:.1f} mm; "
                        f"camera->TCP differs from UMI by {offset_mm:.1f} mm / {angle_deg:.2f} deg")
        return np.linalg.inv(self.link7_from_tcp)


class LerobotInferenceNode(PolicyRuntimeNode):
    """同一命名空间中替换其他策略节点；不能与另一策略节点同时控制。"""

    def __init__(self, engine=None):
        """构建共享运行层；``engine`` 供确定性 ROS 测试注入。"""
        super().__init__(LerobotBackend(), engine)
        # 运行层日志中的 link7_correction 即 inverse(T_Link7_TCP)，此处补充明确的 TCP 说明。
        self.get_logger().info(
            f"{self.backend.tcp_log}; model={getattr(self.engine, 'description', 'injected')}, "
            "observations base_link->TCP, published targets base_link->Link7")


def main(args=None):
    """目标机入口；模型与 ROS 实际运行仅发生在显式启动后。"""
    run_node(LerobotInferenceNode, args)


if __name__ == "__main__":
    main()
