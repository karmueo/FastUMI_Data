"""订阅 ROS2 相机/七轴/夹爪状态，异步推理并发布带观测时间的绝对目标序列。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import signal
import time
import uuid

import cv2
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool, Float32
from std_srvs.srv import Trigger
from fastumi_interfaces.msg import PolicyActionSequence
from fastumi_interfaces.srv import (CancelRecording, CancelRecordingRequest, ResetPolicyController,
                                    SetNumInferenceSteps, StartRecording, StopRecording)

from diffusion_policy.common.urdf_kinematics import UrdfKinematics
from dp_infer.common import (InferenceContext, JOINT_NAMES, image_to_rgb,
                             ordered_joints)
from dp_infer.synchronizer import ObservationBuffer
from dp_infer.visual_guard import ball_center_bgr, ball_near_jaws


def stamp_ns(stamp):
    """将 ROS sec/nanosec 时间精确转换为整数纳秒。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def close_guard_allowed(rgb):
    """仅计算图像中的闭合许可；工作线程不访问 ROS 或任务状态。"""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    center = ball_center_bgr(bgr)
    return ball_near_jaws(center, bgr.shape[1], bgr.shape[0])


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


class PolicyRuntimeNode(Node):
    """ROS 状态管理层；只发布预测结果，推理在独立单工作线程中执行。"""

    def __init__(self, backend, engine=None, guard_detector=None):
        """声明参数、加载模型并注册订阅；engine 参数供确定性 ROS 测试注入。"""
        super().__init__(backend.node_name)
        self.backend = backend
        self.guard_detector = guard_detector or close_guard_allowed
        defaults = {
            "checkpoint": "", "device": "cuda:0", "urdf_path": "",
            "expected_urdf_sha256": "", "num_inference_steps": 8,
            "set_inference_steps_service": "/fastumi/policy/set_inference_steps",
            "image_topic": "/wrist_camera/image_raw/compressed", "image_type": "compressed",
            "joint_topic": "/joint_states", "gripper_topic": "/motion_control/gripper_state",
            "output_topic": "/fastumi/policy/action_sequence",
            "close_guard_topic": "/fastumi/policy/gripper_close_allowed",
            "reset_service": "/fastumi/policy/reset_episode", "sync_slop_s": 0.05,
            "start_service": "/fastumi/policy/start_task",
            "stop_service": "/fastumi/policy/stop_task",
            "return_service": "/fastumi/policy/return_to_start",
            "controller_start_service": "/fastumi/rm75/placo/start_task",
            "controller_return_service": "/fastumi/rm75/placo/return_to_start",
            "task_control_enabled": False,
            "controller_reset_service": "/fastumi/rm75/placo/reset_episode",
            "require_controller_reset": True, "controller_reset_timeout_s": 1.0,
            # 开始/停止任务时联动 fastumi_recorder 录制；录制失败只告警，不阻止任务。
            "record_task": False, "record_dir_name": "dp_infer", "record_name": "rollout",
            "recording_start_service": "/fastumi/recording/start",
            "recording_stop_service": "/fastumi/recording/stop",
            "recording_cancel_service": "/fastumi/recording/cancel",
            "recording_cancel_request_service": "/fastumi/recording/cancel_request",
            "input_timeout_s": 0.2, "history_tolerance_s": 0.015,
            "max_inference_hz": 10.0, "result_timeout_s": 0.5, "postprocessors": "",
            # Link7 部署修正文件；空值语义由模型后端决定。
            "link7_correction_file": "",
        }
        defaults.update(backend.parameter_defaults)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.buffer = ObservationBuffer(self.parameter("sync_slop_s"), self.parameter("input_timeout_s"),
                                        self.parameter("history_tolerance_s"))
        rate, result_timeout = self.parameter("max_inference_hz"), self.parameter("result_timeout_s")
        if not all(np.isfinite(value) and value > 0 for value in (rate, result_timeout)):
            raise ValueError("Inference rate and result timeout must be positive and finite")
        reset_timeout = float(self.parameter("controller_reset_timeout_s"))
        if not np.isfinite(reset_timeout) or reset_timeout <= 0:
            raise ValueError("controller_reset_timeout_s must be positive and finite")
        self.period_s = 1 / rate  # 按单调时钟限制启动推理的最高频率。
        self.result_timeout_ns = round(result_timeout * 1e9)  # 相对采集时刻的最大结果年龄。
        urdf_value = self.parameter("urdf_path")
        if not urdf_value:
            raise ValueError("urdf_path must explicitly name the deployment URDF")
        urdf_path = Path(urdf_value)
        if engine is None:
            engine = backend.create_engine(self.parameter)
        backend.validate_deployment(urdf_path, engine, self.parameter)
        self.fk = UrdfKinematics(urdf_path, JOINT_NAMES)
        # FK 转到模型虚拟 Link7；后端输出在发布前转回真实 Link7。
        self.link7_correction = backend.load_correction(self.parameter)
        self.link7_correction_inverse = np.linalg.inv(self.link7_correction)
        self.engine = engine  # 只由工作线程调用的策略实例。
        if hasattr(engine, "warmup"):
            engine.warmup()
        self._requested_steps = self.parameter("num_inference_steps")
        self.add_on_set_parameters_callback(self._on_parameters_changed)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dp_infer_policy")
        # 1080p JPEG 解码与视觉检测串行会超过 30 Hz 帧预算，饿死同步历史。
        # 检测只保留一个在途任务，ROS 回调不等待它，也不累积旧图像。
        self.guard_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dp_infer_guard")
        self._guard_future = None
        self._guard_context = None
        self._guard_generation = 0
        self.future = None  # 唯一在途推理任务。
        self.pending = None  # 最新待推理的冻结观测及上下文，覆盖旧候选。
        self.active_context = None  # 唯一在途任务的上下文。
        self.last_started_s = float("-inf")  # 上一次推理启动的单调时钟。
        self.last_clock_ns = self.get_clock().now().nanoseconds  # 检测 ROS 时钟回退。
        self.sequence_id = 0  # 生命周期内递增，不随 episode 重置归零。
        self.active_num_inference_steps = self._requested_steps  # 在途推理使用的去噪步数。
        self._require_controller_reset = bool(self.parameter("require_controller_reset"))
        self._task_control_enabled = bool(self.parameter("task_control_enabled"))
        self.task_state = "idle" if self._task_control_enabled else "running"
        self.operation_id = 0
        self._reset_acknowledged = True
        self._controller_reset_future = None
        self._controller_reset_timer = None
        self._reset_io_group = ReentrantCallbackGroup()
        self.controller_reset_client = self.create_client(
            ResetPolicyController, self.parameter("controller_reset_service"),
            callback_group=self._reset_io_group)
        self.controller_start_client = self.create_client(
            ResetPolicyController, self.parameter("controller_start_service"),
            callback_group=self._reset_io_group)
        self.controller_return_client = self.create_client(
            ResetPolicyController, self.parameter("controller_return_service"),
            callback_group=self._reset_io_group)
        self._record_task = bool(self.parameter("record_task"))
        self._recording_id = ""
        self._recording_request_id = ""  # 开始应答丢失时仍可按持久化请求身份撤销。
        self.recording_clients = {
            operation: self.create_client(service, self.parameter(f"recording_{operation}_service"),
                                          callback_group=self._reset_io_group)
            for operation, service in (("start", StartRecording), ("stop", StopRecording),
                                       ("cancel", CancelRecording),
                                       ("cancel_request", CancelRecordingRequest))
        }
        self.publisher = self.create_publisher(PolicyActionSequence, self.parameter("output_topic"), 10)
        self.close_guard_publisher = self.create_publisher(
            Bool, self.parameter("close_guard_topic"), 10)
        image_type = self.parameter("image_type")
        if image_type == "compressed":
            self.image_sub = self.create_subscription(
                CompressedImage, self.parameter("image_topic"),
                self.on_compressed_image, qos_profile_sensor_data)
        elif image_type == "raw":
            self.image_sub = self.create_subscription(
                Image, self.parameter("image_topic"), self.on_image,
                qos_profile_sensor_data)
        else:
            raise ValueError("image_type must be 'compressed' or 'raw'")
        self.joint_sub = self.create_subscription(JointState, self.parameter("joint_topic"), self.on_joint,
                                                  qos_profile_sensor_data)
        self.gripper_sub = self.create_subscription(Float32, self.parameter("gripper_topic"), self.on_gripper,
                                                    qos_profile_sensor_data)
        self.reset_service = self.create_service(Trigger, self.parameter("reset_service"), self.on_reset)
        self.set_inference_steps_service = self.create_service(
            SetNumInferenceSteps, self.parameter("set_inference_steps_service"),
            self.on_set_inference_steps)
        self.start_service = self.create_service(
            Trigger, self.parameter("start_service"), self.on_start_task,
            callback_group=self._reset_io_group)
        self.stop_service = self.create_service(
            Trigger, self.parameter("stop_service"), self.on_stop_task,
            callback_group=self._reset_io_group)
        self.return_service = self.create_service(
            Trigger, self.parameter("return_service"), self.on_return_to_start,
            callback_group=self._reset_io_group)
        self.timer = self.create_timer(0.01, self.tick)
        correction_deg = np.degrees(Rotation.from_matrix(self.link7_correction[:3, :3]).magnitude())
        self.get_logger().info(
            f"DP inference ready: base_link -> Link7, {backend.point_count} points at 30 Hz, "
            f"denoise_steps={self._requested_steps}, "
            f"link7_correction={correction_deg:.2f} deg / "
            f"{np.linalg.norm(self.link7_correction[:3, 3]) * 1000:.1f} mm")

    def parameter(self, name):
        """读取已声明的 ROS 参数当前值。"""
        return self.get_parameter(name).value

    def _on_parameters_changed(self, parameters):
        """校验并保存下一次推理使用的去噪步数。"""
        for parameter in parameters:
            if parameter.name != "num_inference_steps":
                continue
            value = parameter.value
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 50:
                return SetParametersResult(
                    successful=False, reason="num_inference_steps must be an integer in [1, 50]")
            self._requested_steps = value
        return SetParametersResult(successful=True)

    def on_set_inference_steps(self, request, response):
        """接收新步数；在途推理不变，下一次任务提交时读取新值。"""
        result = self.set_parameters_atomically([
            Parameter("num_inference_steps", value=request.num_inference_steps)
        ])
        response.success = result.successful
        response.message = (f"Configured {request.num_inference_steps} denoising steps for the "
                            "next inference" if result.successful else result.reason)
        return response

    def warn(self, text):
        """节流报告无效输入或被丢弃的结果，避免高频传感器刷屏。"""
        self.get_logger().warning(text, throttle_duration_sec=2.0)

    def publish_close_guard(self, rgb, image_stamp_ns):
        """提交新鲜图像的异步检测，避免阻塞传感器接收和推理调度。"""
        if not self.backend.visual_guard:
            return
        if self._task_control_enabled and self.task_state != "running":
            self._invalidate_close_guard()
            return
        age_ns = self.get_clock().now().nanoseconds - image_stamp_ns
        if not 0 <= age_ns <= self.buffer.timeout_ns:
            self._invalidate_close_guard()
            return
        self._finish_close_guard()
        if self._guard_future is None:
            self._guard_context = (image_stamp_ns, self.buffer.episode_id, self._guard_generation)
            self._guard_future = self.guard_worker.submit(self.guard_detector, rgb)

    def _invalidate_close_guard(self):
        """撤销许可，并使重置前或异常图像之前的在途检测失效。"""
        self._guard_generation += 1
        if self._guard_future is not None:
            self._guard_future.cancel()
        self.close_guard_publisher.publish(Bool(data=False))

    def _finish_close_guard(self):
        """在 ROS 回调中收取检测结果，重新校验 episode、任务及采集时间。"""
        future = self._guard_future
        if future is None or not future.done():
            return
        image_stamp_ns, episode_id, generation = self._guard_context
        self._guard_future = None
        self._guard_context = None
        allowed = False
        if not future.cancelled():
            try:
                allowed = bool(future.result())
            except Exception as error:
                self.warn(f"Close guard failed: {error}")
        age_ns = self.get_clock().now().nanoseconds - image_stamp_ns
        allowed = (allowed and episode_id == self.buffer.episode_id
                   and generation == self._guard_generation
                   and (not self._task_control_enabled or self.task_state == "running")
                   and 0 <= age_ns <= self.buffer.timeout_ns)
        self.close_guard_publisher.publish(Bool(data=allowed))

    def on_image(self, message):
        """解码相机消息并缓存图像采集时间；非法编码或尺寸跳过。"""
        try:
            rgb = image_to_rgb(message.data, message.height, message.width, message.step, message.encoding)
            self.publish_close_guard(rgb, stamp_ns(message.header.stamp))
            self.buffer.add_image(stamp_ns(message.header.stamp), self.backend.preprocess_image(rgb))
        except (ValueError, TypeError) as error:
            self._invalidate_close_guard()
            self.warn(f"Dropped image: {error}")

    def on_compressed_image(self, message):
        """解码压缩图像，保留 JPEG 消息中的原始采集时间。"""
        try:
            rgb = self.backend.decode_compressed(message.data)
            self.publish_close_guard(rgb, stamp_ns(message.header.stamp))
            self.buffer.add_image(stamp_ns(message.header.stamp), self.backend.preprocess_image(rgb))
        except (ValueError, TypeError) as error:
            self._invalidate_close_guard()
            self.warn(f"Dropped compressed image: {error}")

    def on_joint(self, message):
        """重排七轴角度并执行 FK，缓存 base_link 下的 Link7 位姿。"""
        try:
            angles = ordered_joints(list(message.name), message.position)
            self.buffer.add_pose(stamp_ns(message.header.stamp),
                                 self.fk.forward(angles) @ self.link7_correction_inverse)
        except (ValueError, TypeError) as error:
            self.warn(f"Dropped joint state: {error}")

    def on_gripper(self, message):
        """缓存 Float32 实测开度；无 header 的消息使用 ROS 接收时间。"""
        try:
            self.buffer.add_gripper(self.get_clock().now().nanoseconds, message.data)
        except ValueError as error:
            self.warn(f"Dropped gripper state: {error}")

    def reset_episode(self):
        """先使旧预测失效，再请求控制器停止并锁定新 episode。"""
        self.buffer.reset()
        self.pending = None
        self._invalidate_close_guard()
        self._reset_acknowledged = not self._require_controller_reset
        self._controller_reset_future = None
        if self._controller_reset_timer is not None:
            self.destroy_timer(self._controller_reset_timer)
            self._controller_reset_timer = None
        if not self._require_controller_reset or not self.controller_reset_client.service_is_ready():
            return
        request = ResetPolicyController.Request()
        request.episode_id = self.buffer.episode_id
        future = self.controller_reset_client.call_async(request)
        self._controller_reset_future = future

        def expire_reset():
            if not future.done() and not future.cancelled():
                future.cancel()

        self._controller_reset_timer = self.create_timer(
            float(self.parameter("controller_reset_timeout_s")), expire_reset,
            callback_group=self._reset_io_group)

    def _finish_controller_reset(self):
        """只在控制器确认已停机后允许下一轮推理发布。"""
        future = self._controller_reset_future
        if future is None or not (future.done() or future.cancelled()):
            return False
        if self._controller_reset_timer is not None:
            self.destroy_timer(self._controller_reset_timer)
            self._controller_reset_timer = None
        self._controller_reset_future = None
        try:
            result = None if future.cancelled() else future.result()
            self._reset_acknowledged = bool(result is not None and result.success)
        except Exception as error:
            self.warn(f"Controller reset failed: {error}")
            self._reset_acknowledged = False
        return self._reset_acknowledged

    async def on_reset(self, request, response):
        """等待控制器确认停止后才向调用方报告重置成功。"""
        if getattr(self, "_task_control_enabled", False):
            return await self.on_stop_task(request, response)
        self.reset_episode()
        future = self._controller_reset_future
        if future is not None:
            await future
            self._finish_controller_reset()
        response.success = self._reset_acknowledged
        response.message = (f"Reset to episode {self.buffer.episode_id}" if response.success
                            else "Controller did not acknowledge episode reset")
        return response

    def _fresh_inputs(self):
        """开始前确认三路观测均有新鲜数据；同步仍由推理循环验证。"""
        now_ns = self.get_clock().now().nanoseconds
        sources = (self.buffer.images, self.buffer.joints, self.buffer.grippers)
        return all(entries and 0 <= now_ns - entries[-1][0] <= self.buffer.timeout_ns
                   for entries in sources)

    def _clear_task(self):
        """递增 episode，立即丢弃旧观测和待发布预测。"""
        self.operation_id += 1
        self.buffer.reset()
        self.pending = None
        self._invalidate_close_guard()
        return self.operation_id

    async def _service_call(self, client, request, timeout_s, label):
        """异步调用服务并在超时后取消；返回 (结果或 None, 失败说明)。"""
        if not client.service_is_ready():
            return None, f"{label} is unavailable"
        future = client.call_async(request)
        timer = self.create_timer(timeout_s, lambda: future.cancel(),
                                  callback_group=self._reset_io_group)
        try:
            await future
            if future.cancelled():
                return None, f"{label} timed out"
            return future.result(), ""
        except Exception as error:
            return None, str(error)
        finally:
            self.destroy_timer(timer)

    async def _controller_call(self, client, timeout_s):
        """异步等待控制器服务；超时后不允许旧答复改变任务状态。"""
        request = ResetPolicyController.Request()
        request.episode_id = self.buffer.episode_id
        result, error = await self._service_call(client, request, timeout_s, "Controller service")
        if result is None:
            return False, error
        return bool(result.success), result.message

    def _forget_recording(self, request_id, recording_id):
        """只清除已确认结束的同一会话，避免迟到应答覆盖新会话身份。"""
        if request_id:
            if self._recording_request_id != request_id:
                return
        elif self._recording_id != recording_id:
            return
        self._recording_request_id = ""
        self._recording_id = ""

    async def _recording_call(self, operation, *, request_id=None, recording_id=None):
        """调用录制事务服务；返回 (成功, 说明, recording_id)。"""
        recording_id = self._recording_id if recording_id is None else recording_id
        if operation == "start":
            request_id = str(uuid.uuid4())
            self._recording_request_id = request_id
            request = StartRecording.Request(
                request_id=request_id, dir_name=self.parameter("record_dir_name"),
                name=self.parameter("record_name"))
        elif operation == "cancel_request":
            request_id = self._recording_request_id if request_id is None else request_id
            request = CancelRecordingRequest.Request(request_id=request_id)
        else:
            request = (StopRecording if operation == "stop" else CancelRecording).Request(
                recording_id=recording_id)
        result, error = await self._service_call(
            self.recording_clients[operation], request, 5.0, "Recording service")
        if result is None:
            if operation == "start":
                # 取消本地 future 不会撤销服务端的开始操作；为同一请求写入撤销标记。
                cancelled, cancel_message, _ = await self._recording_call(
                    "cancel_request", request_id=request_id)
                if cancelled:
                    self._forget_recording(request_id, "")
                else:
                    error += f"; cancellation unconfirmed: {cancel_message}"
            return False, error, ""
        if operation == "cancel_request":
            # completed/failed 也是终态，无活动录制需要继续追踪。
            success = bool(result.success or result.state in ("cancelled", "completed", "failed"))
            return success, result.message, result.recording_id
        if operation == "start" and not result.success:
            self._forget_recording(request_id, "")
        return bool(result.success), result.message, result.recording_id

    async def _finish_recording(self, operation):
        """停止（保存）或取消当前录制，返回附加到任务响应的说明；无录制时为空。"""
        recording_id = self._recording_id
        request_id = self._recording_request_id
        if not recording_id and not request_id:
            return ""
        # 未收到开始应答时没有 recording_id，按请求身份撤销，包含尚未到达的开始请求。
        actual_operation = operation if recording_id else "cancel_request"
        success, message, _ = await self._recording_call(
            actual_operation, request_id=request_id, recording_id=recording_id)
        identity = recording_id or request_id
        if not success:
            self.get_logger().warning(f"Recording {identity} {actual_operation} failed: {message}")
            return f"; recording {identity} {actual_operation} failed: {message}"
        self._forget_recording(request_id, recording_id)
        state = "saving" if actual_operation == "stop" else "cancelled"
        return f"; recording {identity} {state}"

    async def on_start_task(self, request, response):
        """从新 episode 开始策略；控制器确认起点后才开放发布。"""
        if self.task_state == "running":
            response.success, response.message = True, "Task is already running"
            return response
        if self.task_state != "idle":
            response.success, response.message = False, f"Cannot start while {self.task_state}"
            return response
        if not self._fresh_inputs():
            response.success, response.message = False, "Image, joints or gripper feedback is stale"
            return response
        self.task_state = "starting"
        operation = self._clear_task()
        note = ""
        if self._record_task:
            # 上次收尾失败时先重试；不能把新任务悄悄接入上一轮录制。
            note = await self._finish_recording("stop")
            if operation != self.operation_id:
                response.success, response.message = False, "Start was interrupted" + note
                return response
        if self._record_task and not (self._recording_id or self._recording_request_id):
            # 先开录再启动控制器，记录从起点开始的完整过程；失败只告警，任务照常开始。
            recorded, record_message, recording_id = await self._recording_call("start")
            if operation != self.operation_id:
                # 停止请求已按 request_id 撤销此开始，不接纳迟到的 recording_id。
                response.success, response.message = False, "Start was interrupted"
                return response
            if recorded:
                self._recording_id = recording_id
                note += f"; recording {recording_id} started"
            else:
                self.get_logger().warning(f"Recording not started: {record_message}")
                note += f"; recording not started: {record_message}"
        elif self._record_task:
            note += "; recording not started: previous recording cleanup is unconfirmed"
        if self._require_controller_reset:
            success, message = await self._controller_call(self.controller_start_client, 3.0)
        else:
            success, message = True, "Inference-only task started"
        if operation != self.operation_id:
            response.success, response.message = False, "Start was interrupted"
            return response
        if not success:
            note += await self._finish_recording("cancel")
            if operation != self.operation_id:
                response.success, response.message = False, "Start was interrupted" + note
                return response
        self.task_state = "running" if success else "idle"
        response.success, response.message = success, message + note
        return response

    async def on_stop_task(self, _request, response):
        """先撤销推理，再等待控制器和驱动确认停机。"""
        # 开始请求超时只取消本端等待；控制器仍可能稍后接受该请求。
        # 实机即使处于 idle，也要用更新的 episode 确认控制器已停机。
        if (self.task_state == "idle" and not self._require_controller_reset
                and not (self._recording_id or self._recording_request_id)):
            response.success, response.message = True, "Task is already stopped"
            return response
        if self.task_state == "stopping":
            response.success, response.message = False, "Task stop is in progress"
            return response
        self.task_state = "stopping"
        operation = self._clear_task()
        if self._require_controller_reset:
            success, message = await self._controller_call(self.controller_reset_client, 3.0)
        else:
            success, message = True, "Inference stopped"
        if operation != self.operation_id:
            response.success, response.message = False, "Stop was superseded"
            return response
        # 停机确认后再停止录制，保留停机过程；录制结果不影响停机结果。
        note = await self._finish_recording("stop")
        if operation != self.operation_id:
            response.success, response.message = False, "Stop was superseded" + note
            return response
        self.task_state = "idle" if success else "fault"
        response.success, response.message = success, message + note
        return response

    async def on_return_to_start(self, request, response):
        """先停任务，再请求控制器安全回位，成功后保持待命。"""
        if self.task_state in ("homing", "stopping", "starting"):
            response.success, response.message = False, f"Cannot return while {self.task_state}"
            return response
        if self.task_state != "idle" or self._recording_id or self._recording_request_id:
            stop_response = await self.on_stop_task(request, Trigger.Response())
            if not stop_response.success:
                response.success, response.message = False, stop_response.message
                return response
        if not self._require_controller_reset:
            response.success, response.message = False, "Controller is unavailable"
            return response
        else:
            self.task_state = "homing"
            stop_operation = self._clear_task()
            stopped, stop_message = await self._controller_call(
                self.controller_reset_client, 3.0)
            if stop_operation != self.operation_id:
                response.success, response.message = False, "Return was interrupted"
                return response
            if not stopped:
                self.task_state = "fault"
                response.success, response.message = False, stop_message
                return response
        self.task_state = "homing"
        operation = self._clear_task()
        success, message = await self._controller_call(self.controller_return_client, 155.0)
        if operation != self.operation_id:
            response.success, response.message = False, "Return was interrupted"
            return response
        self.task_state = "idle" if success else "fault"
        response.success, response.message = success, message
        return response

    def tick(self):
        """接收工作线程结果、选择最新窗口和提交推理；不在 ROS 回调内等待模型。"""
        now_ns = self.get_clock().now().nanoseconds
        if now_ns < self.last_clock_ns:
            self.reset_episode()
            if self._task_control_enabled:
                self.task_state = "fault"
                self.operation_id += 1
            self.warn("ROS clock moved backwards; episode reset")
        self.last_clock_ns = now_ns
        self._finish_close_guard()
        self._finish_controller_reset()
        if self._task_control_enabled and self.task_state != "running":
            if self.future is not None and self.future.done():
                self.future = None
            return
        if not self._reset_acknowledged:
            return
        if self.future is not None and self.future.done():
            context = self.active_context
            try:
                sequence = self.future.result()
                age_ns = now_ns - context.stamp_ns
                inference_ms = (time.monotonic() - self.last_started_s) * 1000
                if context.episode_id != self.buffer.episode_id:
                    self.warn(
                        f"Dropped prediction: episode changed "
                        f"sequence={context.sequence_id} predicted_episode={context.episode_id} "
                        f"current_episode={self.buffer.episode_id} "
                        f"denoise_steps={self.active_num_inference_steps} "
                        f"inference_ms={inference_ms:.1f}")
                elif not 0 <= age_ns <= self.result_timeout_ns:
                    self.warn(
                        f"Dropped prediction: result expired "
                        f"sequence={context.sequence_id} age_ms={age_ns / 1e6:.1f} "
                        f"limit_ms={self.result_timeout_ns / 1e6:.1f} "
                        f"denoise_steps={self.active_num_inference_steps} "
                        f"inference_ms={inference_ms:.1f}")
                elif not self.backend.result_is_usable(
                        sequence, context, self.get_clock().now().nanoseconds):
                    self.warn(f"Dropped prediction: trajectory fully expired sequence={context.sequence_id}")
                else:
                    self.publisher.publish(sequence_message(
                        self.backend.correct_sequence(sequence, self.link7_correction), context))
                    self.get_logger().info(
                        f"Published sequence={context.sequence_id} episode={context.episode_id} "
                        f"actions={len(sequence.positions)} denoise_steps={self.active_num_inference_steps} "
                        f"inference_ms={inference_ms:.1f} "
                        f"age_ms={age_ns / 1e6:.1f}")
            except Exception as error:
                self.warn(f"Inference/postprocessing failed: {error}")
            self.future = None
        history = self.buffer.poll(now_ns)
        if history is not None:
            reference = history[-1].pose.copy()
            reference.setflags(write=False)
            context = InferenceContext(reference, history[-1].stamp_ns, self.buffer.episode_id, 0)
            self.pending = (self.backend.build_observations(history, self.buffer.start_pose), context)
        if self.future is not None or self.pending is None or time.monotonic() - self.last_started_s < self.period_s:
            return
        observations, context = self.pending
        self.pending = None
        if not 0 <= now_ns - context.stamp_ns <= self.buffer.timeout_ns:
            return
        self.sequence_id += 1
        context = InferenceContext(context.reference_pose, context.stamp_ns, context.episode_id, self.sequence_id)
        self.active_context = context
        self.active_num_inference_steps = self._requested_steps
        self.last_started_s = time.monotonic()
        self.future = self.worker.submit(
            self.engine.predict, observations, context, self.active_num_inference_steps)

    def destroy_node(self):
        """停止工作线程后销毁 ROS 资源，防止退出过程中访问已释放的模型/消息。"""
        if hasattr(self, "worker"):
            self.worker.shutdown(wait=True, cancel_futures=True)
        if hasattr(self, "guard_worker"):
            self.guard_worker.shutdown(wait=True, cancel_futures=True)
        if getattr(self, "_controller_reset_timer", None) is not None:
            self.destroy_timer(self._controller_reset_timer)
        if hasattr(self, "close_guard_publisher"):
            self.close_guard_publisher.publish(Bool(data=False))
        return super().destroy_node()


def run_node(node_factory, args=None):
    """初始化 ROS2 并运行单线程回调执行器；Ctrl-C 后等待推理线程清理。"""
    import cv2
    import torch
    from rclpy.executors import ExternalShutdownException
    from rclpy.signals import SignalHandlerOptions

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    previous_int_handler = signal.getsignal(signal.SIGINT)
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    shutdown_requested = False

    def handle_shutdown(_signum, _frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    node = None
    try:
        node = node_factory()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        if not shutdown_requested or "Unable to convert call argument to Python object" not in str(error):
            raise
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
            signal.signal(signal.SIGINT, previous_int_handler)
            signal.signal(signal.SIGTERM, previous_term_handler)

