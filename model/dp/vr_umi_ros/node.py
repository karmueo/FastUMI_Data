"""订阅 ROS2 相机/七轴/夹爪状态，异步推理并发布带观测时间的绝对目标序列。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from fastumi_interfaces.msg import PolicyActionSequence

from diffusion_policy.common.urdf_kinematics import UrdfKinematics
from vr_umi_ros.core import (InferenceContext, JOINT_NAMES, PolicyEngine, build_observations,
                             compressed_image_to_rgb, image_to_rgb, letterbox_rgb,
                             load_processors, ordered_joints,
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


def predict_with_timing(engine, observations, context):
    """执行策略并返回纯模型、解码与后处理耗时，不包含 ROS 轮询等待。"""
    started_s = time.perf_counter()
    sequence = engine.predict(observations, context)
    return sequence, time.perf_counter() - started_s


def format_prediction(message, inference_seconds, age_ns):
    """把已发布的完整动作序列格式化为适合终端查看的固定宽度表格。"""
    lines = [
        (
            "Prediction published | "
            f"sequence={message.sequence_id} | episode={message.episode_id} | "
            f"inference={inference_seconds * 1000:.2f} ms | "
            f"age={age_ns / 1e6:.2f} ms | steps={len(message.poses)}"
        ),
        "step  time_ms |       x_m       y_m       z_m |        qx        qy        qz        qw | gripper",
        "---- -------- | --------- --------- --------- | --------- --------- --------- --------- | -------",
    ]
    for index, (pose, offset, gripper) in enumerate(zip(
            message.poses, message.time_from_start, message.gripper_openness)):
        lines.append(
            f"{index:>4d} {stamp_ns(offset) / 1e6:>8.1f} | "
            f"{pose.position.x:>9.5f} {pose.position.y:>9.5f} {pose.position.z:>9.5f} | "
            f"{pose.orientation.x:>9.5f} {pose.orientation.y:>9.5f} "
            f"{pose.orientation.z:>9.5f} {pose.orientation.w:>9.5f} | "
            f"{gripper:>7.4f}"
        )
    return "\n".join(lines)


class VrUmiInferenceNode(Node):
    """ROS 状态管理层；只发布预测结果，推理在独立单工作线程中执行。"""

    def __init__(self, engine=None, parameter_defaults=None, result_observer=None,
                 inference_gate=None):
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
            "image_type": "raw", "synchronous_inference": False,
        }
        if parameter_defaults:
            defaults.update(parameter_defaults)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.buffer_lock = threading.RLock()
        self.input_group = MutuallyExclusiveCallbackGroup()
        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.buffer = ObservationBuffer(self.parameter("sync_slop_s"), self.parameter("input_timeout_s"),
                                        self.parameter("history_tolerance_s"))
        self.sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        self.last_joint_stamp_ns = -1
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
        if not getattr(engine, "legacy_missing_urdf_hash", False):
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
        image_type = self.parameter("image_type")
        if image_type == "raw":
            self.image_sub = self.create_subscription(
                Image, self.parameter("image_topic"), self.on_image,
                self.sensor_qos, callback_group=self.input_group)
        elif image_type == "compressed":
            self.image_sub = self.create_subscription(
                CompressedImage, self.parameter("image_topic"), self.on_compressed_image,
                self.sensor_qos, callback_group=self.input_group)
        else:
            raise ValueError("image_type must be 'raw' or 'compressed'")
        self.joint_sub = self.create_subscription(
            JointState, self.parameter("joint_topic"), self.on_joint,
            self.sensor_qos, callback_group=self.input_group)
        self.gripper_sub = self.create_subscription(
            Float32, self.parameter("gripper_topic"), self.on_gripper,
            self.sensor_qos, callback_group=self.input_group)
        self.reset_service = self.create_service(
            Trigger, self.parameter("reset_service"), self.on_reset,
            callback_group=self.input_group)
        self.timer = self.create_timer(0.01, self.tick, callback_group=self.timer_group)
        self.result_observer = result_observer
        self.inference_gate = inference_gate
        self.last_inference_seconds = None
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
            image = letterbox_rgb(rgb)
            with self.buffer_lock:
                self.buffer.add_image(stamp_ns(message.header.stamp), image)
        except (ValueError, TypeError) as error:
            self.warn(f"Dropped image: {error}")

    def on_compressed_image(self, message):
        """解码压缩腕部相机消息，并保留其原始采集时间。"""
        try:
            rgb = compressed_image_to_rgb(message.data)
            image = letterbox_rgb(rgb)
            with self.buffer_lock:
                self.buffer.add_image(stamp_ns(message.header.stamp), image)
        except (ValueError, TypeError) as error:
            self.warn(f"Dropped compressed image: {error}")

    def on_joint(self, message):
        """重排七轴角度并执行 FK，缓存 base_link 下的 Link7 位姿。"""
        try:
            timestamp = stamp_ns(message.header.stamp)
            with self.buffer_lock:
                if timestamp <= self.last_joint_stamp_ns:
                    return
                if (self.last_joint_stamp_ns >= 0
                        and timestamp - self.last_joint_stamp_ns < 20_000_000):
                    return
                self.last_joint_stamp_ns = timestamp
            angles = ordered_joints(list(message.name), message.position)
            pose = self.fk.forward(angles)
            with self.buffer_lock:
                self.buffer.add_pose(timestamp, pose)
        except (ValueError, TypeError) as error:
            self.warn(f"Dropped joint state: {error}")

    def on_gripper(self, message):
        """缓存 Float32 实测开度；无 header 的消息使用 ROS 接收时间。"""
        try:
            with self.buffer_lock:
                self.buffer.add_gripper(self.get_clock().now().nanoseconds, message.data)
        except ValueError as error:
            self.warn(f"Dropped gripper state: {error}")

    def reset_episode(self):
        """清理未处理状态；在途任务完成后按 episode 编号丢弃。"""
        with self.buffer_lock:
            self.buffer.reset()
            self.pending = None
            self.last_joint_stamp_ns = -1

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
                sequence, inference_seconds = self.future.result()
                self._handle_result(sequence, context, inference_seconds)
            except Exception as error:
                self.warn(f"Inference/postprocessing failed: {error}")
            self.future = None
        if self.inference_gate is not None and not self.inference_gate():
            return
        with self.buffer_lock:
            history = self.buffer.poll(now_ns)
            start_pose = None if self.buffer.start_pose is None else self.buffer.start_pose.copy()
            episode_id = self.buffer.episode_id
        if history is not None:
            reference = history[-1].pose.copy()
            reference.setflags(write=False)
            context = InferenceContext(reference, history[-1].stamp_ns, episode_id, 0)
            self.pending = (build_observations(history, start_pose), context)
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
        if self.parameter("synchronous_inference"):
            try:
                sequence, inference_seconds = predict_with_timing(
                    self.engine, observations, context)
                self._handle_result(sequence, context, inference_seconds)
            except Exception as error:
                self.warn(f"Inference/postprocessing failed: {error}")
        else:
            self.future = self.worker.submit(
                predict_with_timing, self.engine, observations, context)

    def _handle_result(self, sequence, context, inference_seconds):
        """仅发布仍在当前 episode 且未超过动作时域的预测结果。"""
        now_ns = self.get_clock().now().nanoseconds
        self.last_inference_seconds = inference_seconds
        age_ns = now_ns - context.stamp_ns
        with self.buffer_lock:
            current_episode_id = self.buffer.episode_id
        if context.episode_id != current_episode_id or not 0 <= age_ns <= self.result_timeout_ns:
            self.warn(
                "Dropped prediction: episode changed or observation/result expired "
                f"(inference_ms={inference_seconds * 1000:.1f}, age_ms={age_ns / 1e6:.1f})")
            return
        message = sequence_message(sequence, context)
        if self.result_observer is not None:
            self.result_observer(message, inference_seconds)
        self.publisher.publish(message)
        self.get_logger().info(format_prediction(message, inference_seconds, age_ns))

    def destroy_node(self):
        """停止工作线程后销毁 ROS 资源，防止退出过程中访问已释放的模型/消息。"""
        if hasattr(self, "worker"):
            self.worker.shutdown(wait=True, cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    """初始化 ROS2 并运行多线程回调执行器；Ctrl-C 后等待推理线程清理。"""
    import torch

    torch.set_num_threads(4)
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        node = VrUmiInferenceNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
