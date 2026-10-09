"""同帧观测、滚动策略及兼容 DP 键盘的非阻塞任务服务。"""

import json
import time

from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from fastumi_interfaces.msg import PolicyJointActionSequence, TargetMask
from fastumi_interfaces.srv import ResetPolicyController, SetNumInferenceSteps
import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from dexgraspvla_infer.core import JOINT_NAMES, ObservationBuffer
from dexgraspvla_infer.model_runtime import ModelRuntime
from dexgraspvla_infer.images import serialized_image


def stamp_ns(stamp):
    """ROS 时间转整数纳秒。"""
    return int(stamp.sec) * 10**9 + int(stamp.nanosec)


class PolicyNode(Node):
    """停止先撤销 episode；异步推理和服务结果不能重启已停止的任务。"""

    def __init__(self, worker, external_images=True, external_masks=True, compute=None):
        super().__init__("dexgraspvla_infer")
        defaults = {
            "checkpoint": "", "model_root": "", "asset_root": "", "urdf_path": "",
            "device": "cuda:0", "num_inference_steps": 16,
            "image_topic": "/wrist_camera/image_decoded", "image_type": "raw",
            "joint_topic": "/joint_states", "gripper_topic": "/motion_control/gripper_state",
            "mask_topic": "/fastumi/perception/target_mask",
            "output_topic": "/fastumi/policy/joint_action_sequence",
            "controller_prefix": "/fastumi/rm75/joint", "sync_slop_s": 0.05,
            "input_timeout_s": 1.0, "feedback_timeout_s": 0.25,
            "result_timeout_s": 1.0, "camera_timeout_s": 0.5,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        get = lambda name: self.get_parameter(name).value
        for name in ("input_timeout_s", "feedback_timeout_s", "result_timeout_s", "camera_timeout_s", "sync_slop_s"):
            if not np.isfinite(get(name)) or get(name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self.worker, self.bridge = worker, CvBridge()
        parameters = dict(checkpoint=get("checkpoint"), model_root=get("model_root"), asset_root=get("asset_root"),
                          urdf=get("urdf_path"), device=get("device"))
        if compute is None:
            self.runtime = ModelRuntime(**parameters)
        else:
            from dexgraspvla_infer.compute import ModelProxy
            self.runtime = ModelProxy(compute, **parameters)
        if not 1 <= get("num_inference_steps") <= self.runtime.max_steps:
            raise ValueError("invalid num_inference_steps")
        self.buffer = ObservationBuffer(get("sync_slop_s"))
        self.state, self.reason = "idle", "waiting for warmup and observations"
        self.episode, self.sequence = time.time_ns(), 0
        self.identity = None
        self.latest_identity = None
        self.inflight, self.warmed = False, False
        self.last_submitted = 0
        self.last_clock = 0
        self.last_image, self.last_joint, self.last_gripper = -np.inf, -np.inf, -np.inf
        self.last_tracking = -np.inf
        self.status_received = -np.inf
        self.status_episode = None
        self.status_enabled = False
        self.last_publish = None
        self.prediction_timing = {}
        self.image_type = get("image_type")
        if self.image_type not in ("raw", "compressed"):
            raise ValueError("image_type must be raw or compressed (JPEG/PNG)")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(PolicyJointActionSequence, get("output_topic"), qos)
        self.status = self.create_publisher(String, "/fastumi/policy/status", qos)
        self.metrics = self.create_publisher(String, "/fastumi/policy/metrics", qos)
        if external_images:
            self.create_subscription(Image if self.image_type == "raw" else CompressedImage,
                                     get("image_topic"), self._on_image, sensor_qos, raw=self.image_type == "raw")
        self.create_subscription(JointState, get("joint_topic"), self._on_joint, sensor_qos)
        self.create_subscription(Float32, get("gripper_topic"), self._on_gripper, sensor_qos)
        if external_masks:
            self.create_subscription(TargetMask, get("mask_topic"), self._on_mask, qos)
        self.create_subscription(String, get("controller_prefix") + "/status", self._on_controller_status, qos)
        self.io_group = ReentrantCallbackGroup()
        self.controller_clients = {key: self.create_client(ResetPolicyController, get("controller_prefix") + "/" + name,
                                               callback_group=self.io_group)
                        for key, name in (("start", "start_task"), ("stop", "reset_episode"), ("home", "return_to_start"))}
        for name, callback in (("start_task", self._start), ("stop_task", self._stop), ("return_to_start", self._home)):
            self.create_service(Trigger, "/fastumi/policy/" + name, callback, callback_group=self.io_group)
        self.create_service(SetNumInferenceSteps, "/fastumi/policy/set_inference_steps", self._steps,
                            callback_group=self.io_group)
        self.create_timer(0.02, self._tick)
        self.create_timer(0.01, worker.poll)
        self.create_timer(0.2, self._publish_status)

    def _on_image(self, message):
        try:
            if self.image_type == "raw":
                header, bgr = serialized_image(message)
            else:
                header = message.header
                bgr = self.bridge.compressed_imgmsg_to_cv2(message, "bgr8")
            self.buffer.image(stamp_ns(header.stamp), header.frame_id, bgr.copy())
            self.last_image = time.monotonic()
        except Exception as error:
            self._fail(f"invalid image: {error}")

    def _on_joint(self, message):
        try:
            if len(message.name) != len(message.position) or len(set(message.name)) != len(message.name):
                raise ValueError("ambiguous joint feedback")
            mapping = dict(zip(message.name, message.position))
            values = np.array([mapping[name] for name in JOINT_NAMES])
            if not np.isfinite(values).all():
                raise ValueError("nonfinite joint feedback")
            self.buffer.joints.append((stamp_ns(message.header.stamp), values))
            self.last_joint = time.monotonic()
        except (ValueError, KeyError) as error:
            self._fail(str(error))

    def _on_gripper(self, message):
        if not np.isfinite(message.data) or not 0 <= message.data <= 1:
            self._fail("invalid gripper feedback")
            return
        self.buffer.grippers.append((self.get_clock().now().nanoseconds, float(message.data)))
        self.last_gripper = time.monotonic()

    def _on_mask(self, message):
        try:
            stamp = stamp_ns(message.header.stamp)
            if (stamp != stamp_ns(message.mask.header.stamp)
                    or message.header.frame_id != message.mask.header.frame_id
                    or message.mask.encoding != "mono8"):
                raise ValueError("mask header/encoding disagrees with source frame")
            if message.state != TargetMask.TRACKING or message.tracking_id == 0:
                raise ValueError(message.diagnostic or "target is not tracking")
            if self.state in ("starting", "running") and message.tracking_id != self.identity:
                raise ValueError("tracking target changed")
            mask = self.bridge.imgmsg_to_cv2(message.mask, "mono8").copy()
            if not mask.any() or not np.isin(mask, [0, 255]).all():
                raise ValueError("empty or nonbinary mask")
            self.buffer.mask(stamp, message.header.frame_id, message.tracking_id, mask)
            self.latest_identity = int(message.tracking_id)
            self.last_tracking = time.monotonic()
        except Exception as error:
            self.latest_identity = None
            self.buffer.masks.clear()
            self._fail(f"target lost: {error}")

    def _on_controller_status(self, message):
        try:
            status = json.loads(message.data)
            self.status_episode = int(status["episode_id"])
            self.status_enabled = bool(status["enabled"])
            self.status_received = time.monotonic()
            if (self.state == "running" and self.status_episode == self.episode
                    and not self.status_enabled):
                self._fail("controller stopped: " + status["reason"])
        except (ValueError, KeyError):
            self._fail("invalid controller status")

    def _fresh(self):
        now = time.monotonic()
        get = lambda key: self.get_parameter(key).value
        return (now - self.last_image <= get("camera_timeout_s")
                and now - self.last_joint <= get("feedback_timeout_s")
                and now - self.last_gripper <= get("feedback_timeout_s")
                and now - self.last_tracking <= get("input_timeout_s"))

    def _observation(self):
        if not self._fresh() or self.latest_identity is None:
            return None
        return self.buffer.latest(self.get_clock().now().nanoseconds,
                                  self.get_parameter("input_timeout_s").value,
                                  self.identity if self.state in ("starting", "running") else self.latest_identity)

    async def _call(self, key, episode, timeout):
        client = self.controller_clients[key]
        if not client.service_is_ready():
            return False, f"controller {key} service is unavailable"
        request = ResetPolicyController.Request()
        request.episode_id = episode
        future = client.call_async(request)
        timer = self.create_timer(timeout, lambda: future.cancel() if not future.done() else None,
                                  callback_group=self.io_group)
        try:
            await future
            if future.cancelled():
                return False, f"controller {key} timed out"
            result = future.result()
            return bool(result.success), result.message
        except Exception as error:
            return False, str(error)
        finally:
            self.destroy_timer(timer)

    def _invalidate(self, state, reason):
        self.episode += 1
        self.state, self.reason = state, reason
        self.identity = None
        if self.worker.cancel("policy"):
            self.inflight = False
        self.last_submitted = 0
        self.last_publish = None
        # Retain current observations for restarting; episode IDs invalidate old work.
        return self.episode

    async def _start(self, request, response):
        observation = self._observation()
        if self.state != "idle" or not self.warmed or observation is None:
            response.success, response.message = False, "Need idle state, warm model and fresh synchronized tracking"
            return response
        episode = self._invalidate("starting", "checking controller start state")
        self.identity = observation.tracking_id
        ok, detail = await self._call("start", episode, 2.0)
        if episode != self.episode or self.state != "starting":
            response.success, response.message = False, "Start superseded by stop"
            return response
        if ok and self._observation() is not None:
            # Do not reuse the pre-warmup/pre-start mask. Wait for a source
            # frame acquired after the controller has confirmed this episode.
            self.last_submitted = self.get_clock().now().nanoseconds
            self.state, self.reason = "running", "task started"
            response.success, response.message = True, detail
        else:
            self._fail(detail if not ok else "observations lost during start")
            response.success, response.message = False, self.reason
        return response

    async def _stop(self, request, response):
        episode = self._invalidate("stopping", "operator stop")
        ok, detail = await self._call("stop", episode, 2.0)
        if episode == self.episode:
            self.state, self.reason = ("idle" if ok else "fault"), detail
        response.success, response.message = ok, detail
        return response

    async def _home(self, request, response):
        if self.state in ("starting", "stopping", "returning"):
            response.success, response.message = False, "Task transition in progress"
            return response
        episode = self._invalidate("returning", "return to start")
        ok, detail = await self._call("stop", episode, 2.0)
        if ok and episode == self.episode:
            ok, detail = await self._call("home", episode, 160.0)
        if episode != self.episode:
            ok, detail = False, "Return interrupted by stop"
        else:
            self.state, self.reason = ("idle" if ok else "fault"), detail
        response.success, response.message = ok, detail
        return response

    def _steps(self, request, response):
        steps = int(request.num_inference_steps)
        if not 1 <= steps <= self.runtime.max_steps:
            response.success, response.message = False, f"steps must be in [1,{self.runtime.max_steps}]"
        else:
            self.set_parameters([Parameter("num_inference_steps", value=steps)])
            response.success, response.message = True, f"Next prediction uses {steps} steps"
        return response

    def _fail(self, reason):
        if self.state not in ("starting", "running"):
            self.reason = reason
            return
        episode = self._invalidate("stopping", reason)
        self.get_logger().error(reason)
        self.get_logger().error("Policy timing at stop: " + json.dumps({
            "prediction": getattr(self, "prediction_timing", {}),
            "inflight": self.inflight, "gpu_worker": self.worker.snapshot(),
            "model_timing": self.runtime.last_diagnostic,
        }))
        client = self.controller_clients["stop"]
        if not client.service_is_ready():
            self.state = "fault"
            return
        request = ResetPolicyController.Request()
        request.episode_id = episode
        future = client.call_async(request)
        deadline = time.monotonic() + 2.0

        def check():
            if episode != self.episode:
                self.destroy_timer(timer)
                return
            if future.done() or time.monotonic() >= deadline:
                try:
                    success = future.done() and not future.cancelled() and future.result().success
                except Exception:
                    success = False
                self.state = "idle" if success else "fault"
                self.destroy_timer(timer)
        timer = self.create_timer(0.02, check, callback_group=self.io_group)

    def _tick(self):
        now = self.get_clock().now().nanoseconds
        if now < self.last_clock:
            self.buffer.clear()
            self._fail("ROS clock moved backwards")
        self.last_clock = now
        if self.state == "running":
            if not self._fresh():
                current = time.monotonic()
                self._fail("observation timed out: " + ", ".join(
                    f"{key}={current - stamp:.3f}s" for key, stamp in (
                        ("camera", self.last_image), ("joint", self.last_joint),
                        ("gripper", self.last_gripper), ("tracking", self.last_tracking))))
                return
            if time.monotonic() - self.status_received > 0.5:
                self._fail("controller heartbeat timed out")
                return
        if self.state not in ("idle", "running") or self.inflight:
            return
        if self.state == "idle" and self.warmed:
            return
        observation = self._observation()
        if observation is None or observation.stamp_ns <= self.last_submitted:
            return
        self.inflight = True
        # Watermark acquisition at dispatch, not the older mask's stamp. A
        # second mask completed just before this prediction must not become
        # the next input merely because its source stamp is a little newer.
        # Wait for an image acquired after this inference was submitted.
        self.last_submitted = now
        episode, warm = self.episode, self.state == "idle"
        steps = int(self.get_parameter("num_inference_steps").value)
        self.prediction_timing["submitted"] = {
            "steps": steps, "source_ns": observation.stamp_ns,
            "dispatch_age_s": (now - observation.stamp_ns) / 1e9,
        }
        started = time.monotonic()
        function = self.runtime.warmup if warm else self.runtime.predict
        self.worker.submit("policy", lambda: function(observation, steps),
                           lambda result, error: self._result(observation, episode, warm, steps, started, result, error))

    def _result(self, observation, episode, warm, steps, started, actions, error):
        self.inflight = False
        if episode != self.episode and not warm:
            return
        if error:
            self.get_logger().error(f"Policy inference failed: {error}")
            self._fail(f"policy inference failed: {error}")
            return
        if warm:
            self.warmed = True
            self.reason = "model warmed; ready for start"
            return
        if self.state != "running" or episode != self.episode or observation.tracking_id != self.identity:
            return
        now = self.get_clock().now().nanoseconds
        age = (now - observation.stamp_ns) / 1e9
        if not 0 <= age <= self.get_parameter("result_timeout_s").value:
            self._fail("policy result expired")
            return
        message = PolicyJointActionSequence()
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(observation.stamp_ns, 10**9)
        message.header.frame_id = "base_link"
        self.sequence += 1
        message.episode_id, message.sequence_id = episode, self.sequence
        message.joint_names = list(JOINT_NAMES)
        for index, row in enumerate(actions):
            point = JointTrajectoryPoint()
            point.positions = row[:7].tolist()
            sec, nanosec = divmod(index * 20_000_000, 10**9)
            point.time_from_start = Duration(sec=sec, nanosec=nanosec)
            message.points.append(point)
        message.gripper_openness = actions[:, 7].astype(float).tolist()
        self.publisher.publish(message)
        interval = None if self.last_publish is None else (now - self.last_publish) / 1e9
        self.prediction_timing["published"] = {
            "sequence_id": self.sequence, "steps": steps, "age_s": age,
            "interval_s": interval, "inference_s": time.monotonic() - started,
        }
        self.last_publish = now
        self.metrics.publish(String(data=json.dumps({"episode_id": episode, "sequence_id": self.sequence,
                             "steps": steps, "age_s": age, "interval_s": interval,
                             "inference_s": time.monotonic() - started})))

    def _publish_status(self):
        observation = self.buffer.latest(self.get_clock().now().nanoseconds,
                      self.get_parameter("input_timeout_s").value, self.latest_identity, copy_images=False)
        now = time.monotonic()
        ages = {name: (now - timestamp if np.isfinite(timestamp) else None)
                for name, timestamp in (("camera", self.last_image), ("joint", self.last_joint),
                                        ("gripper", self.last_gripper), ("tracking", self.last_tracking))}
        self.status.publish(String(data=json.dumps({"state": self.state, "episode_id": self.episode,
                            "tracking_id": self.identity, "latest_tracking_id": self.latest_identity,
                            "warmed": self.warmed, "reason": self.reason, "ages_s": ages,
                            "gpu_worker": self.worker.snapshot(),
                            "prediction_timing": self.prediction_timing,
                            "model_timing": self.runtime.last_diagnostic,
                            "ready": self.state == "idle" and self.warmed and self._fresh() and observation is not None})))

    def request_shutdown_stop(self):
        """在 ROS 上下文关闭前撤销预测并请求驱动确认停车。"""
        if hasattr(self, "shutdown_future"):
            return self.shutdown_future
        self._invalidate("stopping", "node shutdown")
        client = self.controller_clients["stop"]
        self.shutdown_future = None
        if client.service_is_ready():
            request = ResetPolicyController.Request()
            request.episode_id = self.episode
            self.shutdown_future = client.call_async(request)
        return self.shutdown_future

    def destroy_node(self):
        self.request_shutdown_stop()
        return super().destroy_node()
