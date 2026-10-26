"""旧 UMI 后端的兼容 ROS 入口；生命周期与调度由共享运行层维护。"""

from pathlib import Path

import rclpy  # 保留入口级 ROS 注入，便于退出流程回归测试。

from dp_infer import core
from dp_infer.runtime import (PolicyRuntimeNode, close_guard_allowed, run_node,
                              sequence_message, stamp_ns)


class LegacyBackend:
    """旧单文件 checkpoint、五键观测和 16 步共享参考位姿的模型契约。"""

    node_name = "dp_infer"
    parameter_defaults = {}
    point_count = 16
    visual_guard = True
    preprocess_image = staticmethod(core.letterbox_rgb)
    decode_compressed = staticmethod(core.compressed_image_to_rgb)
    build_observations = staticmethod(core.build_observations)
    correct_sequence = staticmethod(core.correct_sequence)

    @staticmethod
    def create_engine(parameter):
        """保留旧模型的文件校验、后处理器与默认去噪配置。"""
        checkpoint = Path(parameter("checkpoint"))
        if not checkpoint.is_file():
            raise ValueError("checkpoint must explicitly name an existing file")
        processors = core.load_processors([
            item.strip() for item in parameter("postprocessors").split(",") if item.strip()])
        return core.PolicyEngine(checkpoint, parameter("device"), processors,
                                 parameter("expected_urdf_sha256"), parameter("num_inference_steps"))

    @staticmethod
    def validate_deployment(urdf_path, engine, parameter):
        """保留旧训练 URDF 摘要约束。"""
        expected = getattr(engine, "urdf_sha256", parameter("expected_urdf_sha256"))
        core.validate_urdf(urdf_path, engine.cfg, expected)

    @staticmethod
    def load_correction(parameter):
        """保留旧修正文件语义。"""
        return core.load_link7_correction(parameter("link7_correction_file"))

    @staticmethod
    def result_is_usable(sequence, context, now_ns):
        """旧后端继续使用原有结果年龄上限。"""
        return True


class DpInferenceNode(PolicyRuntimeNode):
    """旧包入口保持节点名、参数和测试引擎注入方式兼容。"""

    def __init__(self, engine=None):
        super().__init__(LegacyBackend(), engine, guard_detector=lambda rgb: close_guard_allowed(rgb))


def main(args=None):
    """运行旧 UMI 后端。"""
    run_node(DpInferenceNode, args)


if __name__ == "__main__":
    main()
