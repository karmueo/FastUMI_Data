"""订阅 ROS2 相机/七轴/夹爪状态，异步推理并发布带观测时间的绝对目标序列。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from fastumi_interfaces.msg import PolicyActionSequence

from diffusion_policy.common.urdf_kinematics import UrdfKinematics
from vr_umi_ros.core import (InferenceContext, JOINT_NAMES, PolicyEngine, build_observations,
                             image_to_rgb, letterbox_rgb, load_processors, ordered_joints,
                             validate_urdf)
from vr_umi_ros.synchronizer import ObservationBuffer


def stamp_ns(stamp):
    """将 ROS sec/nanosec 时间精确转换为整数纳秒。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def sequence_message(sequence, context):
    """将已校验的核心结果转换为 ROS 消息；header 保留原始观测时间。"""
    message = PolicyActionSequence()
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(context.stamp_ns, 1_000_000_000)
    message.header.frame_id = "base_link"
    message.end_frame = "Link7"
    message.episode_id = context.episode_id
    message.sequence_id = context.sequence_id
    for position, quaternion, offset in zip(sequence.positions, sequence.quaternions,
                                            sequence.time_from_start):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, position)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, quaternion)
        message.poses.append(pose)
        seconds, nanoseconds = divmod(round(float(offset) * 1e9), 1_000_000_000)
        message.time_from_start.append(Duration(sec=seconds, nanosec=nanoseconds))
    message.gripper_openness = sequence.gripper_openness.astype(float).tolist()
    return message


class VrUmiInferenceNode(Node):
    """ROS 状态管理层；只发布预测结果，推理在独立单工作线程中执行。"""

    def __init__(self, engine=None):
        """声明参数、加载模型并注册订阅；engine 参数供确定性 ROS 测试注入。"""
        super().__init__("vr_umi_inference")
        default_urdf = Path(__file__).resolve().parents[3] / "dataset/vr_target_umi/rm_75.urdf"
        defaults = {
            "checkpoint": "", "device": "cuda:0", "urdf_path": str(default_urdf),
            "image_topic": "/camera/image_raw", "joint_topic": "/joint_states",
            "gripper_topic": "/fastumi/gripper/state", "output_topic": "/fastumi/policy/action_sequence",
            "reset_service": "/fastumi/policy/reset_episode", "sync_slop_s": 0.05,
            "input_timeout_s": 0.2, "history_tolerance_s": 0.015,
            "max_inference_hz": 10.0, "result_timeout_s": 0.5, "postprocessors": "",
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
        urdf_path = Path(self.parameter("urdf_path"))
        if engine is None:
            checkpoint = Path(self.parameter("checkpoint"))
            if not checkpoint.is_file():
                raise ValueError("checkpoint must explicitly name an existing file")
            processors = load_processors([item.strip() for item in self.parameter("postprocessors").split(",")
                                          if item.strip()])
            engine = PolicyEngine(checkpoint, self.parameter("device"), processors)
        validate_urdf(urdf_path, engine.cfg)
        self.fk = UrdfKinematics(urdf_path, JOINT_NAMES)
        self.engine = engine  # 只由工作线程调用的策略实例。
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vr_umi_policy")
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
        self.get_logger().info("VR UMI inference ready: base_link -> Link7, 16 steps at 30 Hz")

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
        """重排七轴角度并执行 FK，缓存 base_link 下的 Link7 位姿。"""
        try:
            angles = ordered_joints(list(message.name), message.position)
            self.buffer.add_pose(stamp_ns(message.header.stamp), self.fk.forward(angles))
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
        """处理 episode 重置服务，下一有效同步样本建立新的起始位姿。"""
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
                        f"steps=16 inference_ms={(time.monotonic() - self.last_started_s) * 1000:.1f} "
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
            context = InferenceContext(reference, history[-1].stamp_ns, self.buffer.episode_id, 0)
            self.pending = (build_observations(history, self.buffer.start_pose), context)
        if self.future is not None or self.pending is None or time.monotonic() - self.last_started_s < self.period_s:
            return
        observations, context = self.pending
        self.pending = None
        if not 0 <= now_ns - context.stamp_ns <= self.buffer.timeout_ns:
            return
        self.sequence_id += 1
        context = InferenceContext(context.reference_pose, context.stamp_ns, context.episode_id, self.sequence_id)
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
        node = VrUmiInferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
