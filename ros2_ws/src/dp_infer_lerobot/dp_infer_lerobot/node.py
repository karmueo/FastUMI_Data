"""LeRobot 局部增量后端；复用旧 DP 的 ROS 生命周期与操作协议。"""

from dp_infer.core import load_processors
from dp_infer.runtime import PolicyRuntimeNode, run_node

from . import core, deployment


class LerobotBackend:
    """独立模型契约，不向旧后端伪造 Hydra 配置或训练 URDF。"""

    node_name = "dp_infer_lerobot"
    parameter_defaults = {"num_inference_steps": 16, "deployment_config": "",
                          "record_dir_name": "dp_infer_lerobot", "task_control_enabled": True}
    point_count = 9
    visual_guard = False
    preprocess_image = staticmethod(core.letterbox_rgb)
    decode_compressed = staticmethod(core.compressed_image_to_rgb)
    build_observations = staticmethod(core.build_observations)
    correct_sequence = staticmethod(core.correct_sequence)
    result_is_usable = staticmethod(core.result_is_usable)

    def create_engine(self, parameter):
        """加载绑定部署配置后才创建真实模型。"""
        from .engine import PolicyEngine

        self.profile = deployment.load_profile(parameter("deployment_config"))
        processors = load_processors([
            item.strip() for item in parameter("postprocessors").split(",") if item.strip()])
        return PolicyEngine(parameter("checkpoint"), self.profile, parameter("device"),
                            parameter("num_inference_steps"), processors)

    def validate_deployment(self, urdf_path, engine, parameter):
        """注入假引擎时仍校验部署契约及运动链。"""
        self.profile = deployment.load_profile(parameter("deployment_config"))
        deployment.validate_urdf(urdf_path, self.profile)

    def load_correction(self, parameter):
        """只接受绑定该训练数据的新标定修正。"""
        return deployment.load_correction(parameter("link7_correction_file"), self.profile)


class LerobotInferenceNode(PolicyRuntimeNode):
    """同一命名空间中替换旧节点；不能与另一策略节点同时控制。"""

    def __init__(self, engine=None):
        super().__init__(LerobotBackend(), engine)


def main(args=None):
    """目标机入口；模型与 ROS 实际运行仅发生在显式启动后。"""
    run_node(LerobotInferenceNode, args)


if __name__ == "__main__":
    main()
