"""订阅相机/七轴/夹爪，运行 LeRobot TCP 扩散策略并发布带观测时间的 Link7 目标序列。

ROS 生命周期沿用 ``dp_infer`` 节点：同步缓冲、单工作线程推理、最新窗口覆盖、结果过期丢弃与
episode 重置服务。差异在于模型契约：观测为 ``base_link → TCP``（``FK @ T_Link7_TCP``），
输出 8 个点（k/30 秒），checkpoint 为 LeRobot ``pretrained_model`` 目录并由部署档绑定。
节点只发布推荐结果，不会向机械臂或夹爪发送命令。
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from fastumi_interfaces.msg import PolicyActionSequence

from diffusion_policy.common.urdf_kinematics import UrdfKinematics
from dp_infer.core import (DEFAULT_MAX_TCP_AXIS_DEG, InferenceContext, JOINT_NAMES, image_to_rgb,
                           letterbox_rgb, load_processors, ordered_joints)
from dp_infer.node import sequence_message, stamp_ns
from dp_infer.synchronizer import ObservationBuffer

from . import core, deployment
from .config import validate_backend


class LerobotInferenceNode(Node):
    """ROS 状态管理层；只发布预测结果，推理在独立单工作线程中执行。"""

    def __init__(self, engine=None):
        """声明参数、校验部署档并加载模型；``engine`` 供确定性 ROS 测试注入。"""
        super().__init__("dp_infer_lerobot")
        defaults = {
            "checkpoint": "", "deployment_config": "", "device": "cuda:0", "urdf_path": "",
            # pytorch 读取 checkpoint；tensorrt_fp32/tensorrt_fp16 读取 export_dir 模型包中的 engine。
            "backend": "pytorch", "export_dir": "",
            # 空字符串沿用训练 DDPM；DDIM 可配合较少步数降低延迟。0 表示 checkpoint 默认步数。
            "noise_scheduler": "", "num_inference_steps": 0,
            "image_topic": "/usb_camera/image_raw", "joint_topic": "/joint_states",
            "gripper_topic": "/gripper/openness", "output_topic": "/fastumi/policy/action_sequence",
            "reset_service": "/fastumi/policy/reset_episode", "sync_slop_s": 0.05,
            "input_timeout_s": 0.2, "history_tolerance_s": 0.015,
            "max_inference_hz": 10.0, "result_timeout_s": 0.5, "postprocessors": "",
            # 部署相机←Link7（T_cam_end.yaml）与部署相机←TCP（T_d_cam_tcp.yaml），均必填。
            "end_camera_extrinsic": "", "deploy_tcp_extrinsic": "",
            "max_tcp_axis_deg": DEFAULT_MAX_TCP_AXIS_DEG,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.buffer = ObservationBuffer(self.parameter("sync_slop_s"), self.parameter("input_timeout_s"),
                                        self.parameter("history_tolerance_s"))
        rate, result_timeout = self.parameter("max_inference_hz"), self.parameter("result_timeout_s")
        if not all(np.isfinite(value) and value > 0 for value in (rate, result_timeout)):
            raise ValueError("Inference rate and result timeout must be positive and finite")
        self.period_s = 1 / rate  # 按单调时钟限制启动推理的最高频率。
        self.result_timeout_ns = round(result_timeout * 1e9)  # 相对采集时刻的最大结果年龄。
        profile = deployment.load_profile(self.parameter("deployment_config"))
        urdf_value = self.parameter("urdf_path")
        if not urdf_value:
            raise ValueError("urdf_path must explicitly name the training URDF")
        deployment.validate_urdf(urdf_value, profile)
        link7_from_tcp, offset_mm, angle_deg = deployment.load_tool_offset(
            profile, self.parameter("end_camera_extrinsic"), self.parameter("deploy_tcp_extrinsic"),
            self.parameter("max_tcp_axis_deg"))
        self.get_logger().info(
            f"Deployment TCP: Link7->TCP {np.linalg.norm(link7_from_tcp[:3, 3]) * 1000:.1f} mm; "
            f"camera->TCP differs from UMI by {offset_mm:.1f} mm / {angle_deg:.2f} deg")
        self.link7_from_tool = link7_from_tcp  # T_Link7_TCP，观测右乘、输出右乘其逆。
        self.link7_from_tool.setflags(write=False)
        if engine is None:
            backend, checkpoint, export_dir = (self.parameter(name) for name in ("backend", "checkpoint",
                                                                                  "export_dir"))
            if validate_backend(backend) is None:
                if not checkpoint or not Path(checkpoint).is_dir():
                    raise ValueError("checkpoint must name an existing pretrained_model directory")
            elif not export_dir or not Path(export_dir).is_dir():
                raise ValueError("TensorRT backends require export_dir naming a lerobot-export package")
            from .engine import PolicyEngine

            processors = load_processors([item.strip() for item in self.parameter("postprocessors").split(",")
                                          if item.strip()])
            engine = PolicyEngine(checkpoint, profile, self.parameter("device"),
                                  self.parameter("noise_scheduler"), self.parameter("num_inference_steps"),
                                  processors, backend, export_dir)
        self.fk = UrdfKinematics(Path(urdf_value), JOINT_NAMES)
        self.engine = engine  # 只由工作线程调用的策略实例。
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dp_infer_lerobot_policy")
        self.future = None  # 唯一在途推理任务。
        self.pending = None  # 最新待推理的冻结观测及上下文，覆盖旧候选。
        self.active_context = None  # 唯一在途任务的上下文。
        self.last_started_s = float("-inf")  # 上一次推理启动的单调时钟。
        self.last_clock_ns = self.get_clock().now().nanoseconds  # 检测 ROS 时钟回退。
        self.sequence_id = 0  # 生命周期内递增，不随 episode 重置归零。
        self.publisher = self.create_publisher(PolicyActionSequence, self.parameter("output_topic"), 10)
        self.image_sub = self.create_subscription(Image, self.parameter("image_topic"), self.on_image,
                                                  qos_profile_sensor_data)
        self.joint_sub = self.create_subscription(JointState, self.parameter("joint_topic"), self.on_joint,
                                                  qos_profile_sensor_data)
        self.gripper_sub = self.create_subscription(Float32, self.parameter("gripper_topic"), self.on_gripper,
                                                    qos_profile_sensor_data)
        self.reset_service = self.create_service(Trigger, self.parameter("reset_service"), self.on_reset)
        self.timer = self.create_timer(0.01, self.tick)
        self.get_logger().info(
            f"LeRobot DP inference ready ({getattr(engine, 'description', 'injected')}): "
            f"base_link -> tcp observations, Link7 targets, {core.ACTION_STEPS} steps at 30 Hz")

    def parameter(self, name):
        """读取启动时声明的 ROS 参数值。"""
        return self.get_parameter(name).value

    def warn(self, text):
        """节流报告无效输入或被丢弃的结果，避免高频传感器刷屏。"""
        self.get_logger().warning(text, throttle_duration_sec=2.0)

    def on_image(self, message):
        """解码相机消息并缓存图像采集时间；非法编码或尺寸跳过。"""
        try:
            rgb = image_to_rgb(message.data, message.height, message.width, message.step, message.encoding)
            self.buffer.add_image(stamp_ns(message.header.stamp), letterbox_rgb(rgb))
        except (ValueError, TypeError) as error:
            self.warn(f"Dropped image: {error}")

    def on_joint(self, message):
        """重排七轴角度并执行 FK，缓存 base_link 下的 TCP 位姿。"""
        try:
            angles = ordered_joints(list(message.name), message.position)
            self.buffer.add_pose(stamp_ns(message.header.stamp),
                                 self.fk.forward(angles) @ self.link7_from_tool)
        except (ValueError, TypeError) as error:
            self.warn(f"Dropped joint state: {error}")

    def on_gripper(self, message):
        """缓存 Float32 实测开度；无 header 的消息使用 ROS 接收时间。"""
        try:
            self.buffer.add_gripper(self.get_clock().now().nanoseconds, message.data)
        except ValueError as error:
            self.warn(f"Dropped gripper state: {error}")

    def reset_episode(self):
        """清理未处理状态；在途任务完成后按 episode 编号丢弃。"""
        self.buffer.reset()
        self.pending = None

    def on_reset(self, request, response):
        """处理 episode 重置服务。"""
        self.reset_episode()
        response.success = True
        response.message = f"Reset to episode {self.buffer.episode_id}"
        return response

    def tick(self):
        """接收工作线程结果、选择最新窗口和提交推理；不在 ROS 回调内等待模型。"""
        now_ns = self.get_clock().now().nanoseconds
        if now_ns < self.last_clock_ns:
            self.reset_episode()
            self.warn("ROS clock moved backwards; episode reset")
        self.last_clock_ns = now_ns
        if self.future is not None and self.future.done():
            context = self.active_context
            try:
                sequence = self.future.result()
                age_ns = now_ns - context.stamp_ns
                if context.episode_id == self.buffer.episode_id and 0 <= age_ns <= self.result_timeout_ns:
                    self.publisher.publish(sequence_message(sequence, context))
                    self.get_logger().info(
                        f"Published sequence={context.sequence_id} episode={context.episode_id} "
                        f"points={len(sequence.positions)} "
                        f"inference_ms={(time.monotonic() - self.last_started_s) * 1000:.1f} "
                        f"age_ms={age_ns / 1e6:.1f}")
                else:
                    self.warn("Dropped prediction: episode changed or observation/result expired")
            except Exception as error:
                self.warn(f"Inference/postprocessing failed: {error}")
            self.future = None
        history = self.buffer.poll(now_ns)
        if history is not None:
            reference = history[-1].pose.copy()
            reference.setflags(write=False)
            context = InferenceContext(reference, history[-1].stamp_ns, self.buffer.episode_id, 0,
                                       self.link7_from_tool)
            try:
                self.pending = (core.build_observations(history), context)
            except ValueError as error:
                self.warn(f"Dropped observation window: {error}")
        if self.future is not None or self.pending is None or time.monotonic() - self.last_started_s < self.period_s:
            return
        observations, context = self.pending
        self.pending = None
        if not 0 <= now_ns - context.stamp_ns <= self.buffer.timeout_ns:
            return
        self.sequence_id += 1
        context = InferenceContext(context.reference_pose, context.stamp_ns, context.episode_id,
                                   self.sequence_id, context.link7_from_tool)
        self.active_context = context
        self.last_started_s = time.monotonic()
        self.future = self.worker.submit(self.engine.predict, observations, context)

    def destroy_node(self):
        """停止工作线程后销毁 ROS 资源，防止退出过程中访问已释放的模型/消息。"""
        if hasattr(self, "worker"):
            self.worker.shutdown(wait=True, cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    """初始化 ROS2 并运行单线程回调执行器；Ctrl-C 后等待推理线程清理。"""
    import torch

    torch.set_num_threads(4)
    rclpy.init(args=args)
    node = None
    try:
        node = LerobotInferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
