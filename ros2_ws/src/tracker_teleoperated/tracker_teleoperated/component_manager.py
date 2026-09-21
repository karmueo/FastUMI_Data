"""管理遥操会话的本机进程、输入健康和异步保存退出流程。"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import threading
import time

from ament_index_python.packages import get_package_share_directory, get_package_prefix
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from fastumi_interfaces.msg import GripperState, TrackerStatus
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool, Trigger
import yaml

from tracker_teleoperated.recording_paths import output_directory
from tracker_teleoperated.camera_devices import CameraDevices
from tracker_teleoperated.component_runtime import (
    Health, ManagedProcess, process_table, read_configuration, workspace_root,
)


# 所有控制接口共享的命名空间。
PREFIX = "/tracker_teleoperated/"
# 显式列出可管理组件，配置不能注入任意 shell 命令。
COMPONENT_IDS = (
    "arm", "tracker", "gripper", "umi_camera", "estimator",
    "wrist_camera", "teleop", "recorder",
)


@dataclass
class Component:
    """保存一个组件的管理归属、进程和健康信息。"""

    # 稳定组件标识。
    key: str
    # 启动与模式配置。
    config: dict
    # 数据探针及最后接收时间。
    health: Health
    # 受管理进程；外部组件始终为空。
    process: ManagedProcess | None = None
    # local、external 或 none，外部归属一旦发现即保持。
    ownership: str = "none"
    # 进程状态与业务状态分开表示。
    state: str = "stopped"
    # 最近启动或退出错误。
    error: str = ""
    # 启动时刻，用于设备初始化宽限。
    started: float = 0.0
    # 相机启动的 ROS 时间，用于丢弃上一采集实例的排队图像。
    started_stamp: int = 0
    # 最近进程退出码。
    exit_code: str = ""
    # 本次组件日志文件。
    log_path: str = ""


class ComponentManager(Node):
    """用单线程 ROS 定时状态机串行化启停，避免 GUI 或服务线程等待写盘。"""

    def __init__(self, parameter_overrides=None, process_factory=ManagedProcess):
        """加载配置、创建探针及控制服务，不直接启动任何硬件。"""
        super().__init__("tracker_component_manager", parameter_overrides=parameter_overrides)
        share = Path(get_package_share_directory("tracker_teleoperated"))
        for name, default in {
            "manager_config": str(share / "config/component_manager.yaml"),
            "config_file": str(share / "config/tracker_teleoperated.yaml"),
            "autostart": True, "use_recorder": True,
        }.items():
            self.declare_parameter(name, default)
        self.config = read_configuration(self.get_parameter("manager_config").value)
        self.root = (Path(self.config["workspace_root"]).expanduser().resolve()
                     if self.config.get("workspace_root") else workspace_root())
        self.config_file = str(Path(self.get_parameter("config_file").value).resolve())
        raw = yaml.safe_load(Path(self.config_file).read_text())
        self.control = raw["tracker_teleop"]["ros__parameters"]
        self.record = raw["tracker_teleop_recorder"]["ros__parameters"]
        # 本机当前任务目录独立于记录节点是否启动，供面板浏览历史记录。
        self.output_directory = str(output_directory(
            self.record.get("dataset_root", "dataset/h5dy_data"),
            self.record.get("dir_name", "test"), self.record.get("name", "default_test"),
        ))
        # flock 只约束同主机同 ROS domain 的本应用管理器。
        ros_home = Path(os.environ.get("ROS_HOME", str(Path.home() / ".ros")))
        session_root = ros_home / "tracker_teleoperated"
        session_root.mkdir(parents=True, exist_ok=True)
        self._lock_file = (session_root / f"manager-{os.environ.get('ROS_DOMAIN_ID', '0')}.lock").open("a")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._lock_file.close()
            raise RuntimeError("同一 ROS domain 已有本机遥操管理器") from error
        self.log_root = session_root / "logs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        self.process_factory = process_factory
        self.components = {}
        self.probes = {}
        self.operation = None
        self.shutdown_complete = False
        self.last_operation = "就绪"
        self._next_graph = 0.0
        self._next_publish = 0.0
        self._nodes = set()
        self._autostart_at = time.monotonic() + float(self.config.get("discovery_wait_s", 2.0))
        self._autostart = bool(self.get_parameter("autostart").value)
        state_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_publisher = self.create_publisher(DiagnosticArray, PREFIX + "components/status", state_qos)
        self.pause_client = self.create_client(SetBool, PREFIX + "set_enabled")
        self.save_client = self.create_client(Trigger, PREFIX + "stop_recording")
        for key in COMPONENT_IDS:
            item = dict(self.config["components"].get(key, {"mode": "disabled"}))
            if key == "recorder" and not self.get_parameter("use_recorder").value:
                item["mode"] = "disabled"
            self.components[key] = Component(key, item, Health({}))
            self.create_service(
                SetBool, PREFIX + f"components/{key}/set_running",
                lambda request, response, key=key: self._set_running(key, request, response),
            )
        self._make_probes(state_qos)
        self.video_sources = CameraDevices(self, state_qos)
        self.create_service(Trigger, PREFIX + "start_all", self._start_all_service)
        self.create_service(Trigger, PREFIX + "stop_all", self._stop_all_service)
        self.create_service(Trigger, PREFIX + "shutdown_session", self._shutdown_service)
        self.create_timer(0.1, self._tick)

    def _probe(self, key, topic, message_type, timeout, validate, qos):
        """绑定 ROS 话题与纯健康模型；无效消息也更新接收时间。"""
        component = self.components[key]
        component.health.deadlines[topic] = float(component.config.get("health_timeout_s", timeout))
        self.probes.setdefault(key, []).append(topic)
        def receive(message):
            """按接收时间判断新鲜度，重启前拍摄的图像不参与启动确认。"""
            if key.endswith('_camera') and component.ownership == 'local':
                stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
                if stamp < component.started_stamp:
                    return
            component.health.receive(topic, validate(message))

        self.create_subscription(message_type, topic, receive, qos)

    def _make_probes(self, state_qos):
        """为八个组件配置业务有效性探针，复用控制端超时配置。"""
        c = self.control
        sensor = qos_profile_sensor_data
        self._probe("arm", c["joint_state_topic"], JointState, c["feedback_timeout_s"],
                    lambda m: len(m.name) == len(m.position) and
                    all(f"joint{i}" in m.name for i in range(1, 8)) and
                    all(math.isfinite(v) for v in m.position), sensor)
        self._probe("tracker", c["tracker_odom_topic"], Odometry, c["pose_timeout_s"],
                    valid_odometry, sensor)
        self._probe("tracker", c["tracker_status_topic"], TrackerStatus, c["pose_timeout_s"],
                    lambda m: m.device_connected and m.pose_valid and
                    m.tracking_state == TrackerStatus.TRACKING_RUNNING_OK, sensor)
        self._probe("gripper", c["gripper_feedback_topic"], Float32,
                    c["gripper_feedback_timeout_s"], lambda m: math.isfinite(m.data) and 0 <= m.data <= 1, sensor)
        self._probe("estimator", c["gripper_estimate_topic"], GripperState,
                    c["gripper_estimate_timeout_s"],
                    lambda m: m.valid and math.isfinite(m.filtered_openness) and 0 <= m.filtered_openness <= 1, sensor)
        # 相机进程探针固定为其输出，消费节点的可选输入另行监测。
        for key in ("umi_camera", "wrist_camera"):
            namespace = self.components[key].config.get("parameters", {}).get("namespace", key)
            topic = f"/{namespace.strip('/')}/image_raw"
            self._probe(key, topic, Image, 1.0,
                        lambda m: m.width > 0 and m.height > 0 and m.step > 0 and len(m.data) >= m.step * m.height, sensor)
        self._probe("teleop", PREFIX + "enabled", Bool, 2.0, lambda m: True, state_qos)
        self._probe("recorder", PREFIX + "record_state", String, 2.0,
                    lambda m: m.data in ("idle", "recording", "saving"), state_qos)

    def _command(self, key):
        """按组件创建参数数组和独立环境，保持 NumPy 1/2 隔离。"""
        item = self.components[key].config
        environment = dict(os.environ)
        venv = self.root / (".venv-numpy2" if key in ("teleop", "recorder") else ".venv-numpy1")
        python = venv / "bin/python"
        if not python.is_file():
            raise RuntimeError(f"缺少 Python 环境: {python}")
        environment["VIRTUAL_ENV"] = str(venv)
        environment["TRACKER_COMPONENT_STOP_GRACE"] = "300" if key == "recorder" else "5"
        environment["PATH"] = str(venv / "bin") + os.pathsep + os.environ.get("PATH", "")
        # 安装入口的解释器已固定；ros2 CLI 显式使用对应环境解释器。
        ros2 = shutil.which("ros2")
        if not ros2:
            raise RuntimeError("未找到 ros2，请先加载工作区环境")
        base = [str(python), ros2]
        parameters = dict(item.get("parameters", {}))
        if key in ("teleop", "recorder"):
            executable = "tracker_teleop_node" if key == "teleop" else "tracker_teleop_recorder"
            entry = Path(get_package_prefix("tracker_teleoperated")) / "lib/tracker_teleoperated" / executable
            command = [str(python), str(entry), "--ros-args", "--params-file", self.config_file]
            if key == "recorder":
                command += ["-p", f"image_topic:={self.video_sources.launch_topic(key)}"]
            return command, environment
        if key == "gripper":
            script = self.root / "src/unitree_gripper/run_gripper.sh"
            command = ["bash", str(script)]
            if item.get("config_file"):
                command += ["-c", item["config_file"]]
            if item.get("network_interface"):
                command += ["-n", item["network_interface"]]
            return command, environment
        if key == "arm":
            package, launch = "rm_driver", "rm_75_driver.launch.py"
        elif key == "tracker":
            package, launch = "vive_tracker", "vive_tracker.launch.py"
            parameters["use_rviz"] = "false"
        elif key == "estimator":
            package, launch = "fastumi_gripper_estimator", "gripper_openness.launch.py"
            parameters["image_topic"] = self.video_sources.launch_topic(key)
        else:
            package, launch = "fastumi_usb_camera", "usb_camera.launch.py"
            camera_config = "usb_camera.yaml" if key == "umi_camera" else "usb_camera_1280_960.yaml"
            parameters.setdefault("config", str(Path(get_package_share_directory(package)) / "config" / camera_config))
        return base + ["launch", "--noninteractive", package, launch] + [
            f"{name}:={str(value).lower() if isinstance(value, bool) else value}"
            for name, value in parameters.items()
        ], environment

    def _external_present(self, component):
        """依据配置节点名和数据发布端识别外部组件，不用订阅者误判。"""
        return bool(set(component.config.get("nodes", [])) & self._nodes) or any(
            self.count_publishers(topic) for topic in self.probes.get(component.key, [])
        )

    def _start(self, key):
        """幂等启动组件；发现外部实例时只监测，启动失败保留面板。"""
        component = self.components[key]
        mode = component.config.get("mode", "auto")
        if mode != "auto":
            return False, "该组件为仅监测或禁用模式"
        if component.process is not None:
            return True, "组件已启动"
        if component.ownership == "local" and self._external_present(component):
            return False, "旧节点尚未从 ROS 图移除，请稍后重试"
        if component.ownership == "external" or self._external_present(component):
            component.ownership = "external"
            component.state = "external"
            return True, "检测到外部组件，仅监测"
        try:
            command, environment = self._command(key)
            log = self.log_root / f"{key}.log"
            component.process = self.process_factory(command, environment, log, self.root)
            component.ownership = "local"
            component.state = "starting"
            component.started = time.monotonic()
            component.started_stamp = self.get_clock().now().nanoseconds
            component.error = component.exit_code = ""
            component.log_path = str(log)
            component.health.samples.clear()
            return True, "启动请求已接受"
        except Exception as error:
            component.error = str(error)
            component.state = "failed"
            return False, component.error

    def _set_running(self, key, request, response):
        """组件服务只接受一个会话操作，最终进度经诊断话题发布。"""
        if self.operation or (request.data and self.video_sources.operation):
            response.success, response.message = False, "正在保存或停止，请等待"
        elif request.data:
            response.success, response.message = self._start(key)
        elif self.components[key].ownership != "local":
            response.success, response.message = False, "无法停止外部或未启动组件"
        else:
            self._begin_stop([key])
            response.success, response.message = True, "正在暂停并保存，随后停止组件"
        return response

    def _start_all_service(self, _request, response):
        """按配置顺序启动所有自动组件，并汇总启动错误。"""
        if self.operation or self.video_sources.operation or self.video_sources.unlock:
            response.success, response.message = False, "正在执行停止操作"
            return response
        errors = []
        for key, component in self.components.items():
            if component.config.get("mode", "auto") == "auto":
                ok, message = self._start(key)
                if not ok:
                    errors.append(f"{key}: {message}")
        response.success = not errors
        response.message = "；".join(errors) or "启动请求已接受，等待设备数据"
        return response

    def _stop_all_service(self, _request, response):
        """停止本管理器拥有的全部组件，保留 RViz 与管理器。"""
        response.success = self._begin_stop(list(reversed(COMPONENT_IDS)))
        response.message = "正在暂停并保存" if response.success else "已有停止操作"
        return response

    def _shutdown_service(self, _request, response):
        """保存并结束本次会话，由顶层 launch 关闭 RViz。"""
        response.success = self.request_shutdown()
        response.message = "正在保存并退出" if response.success else "正在退出"
        return response

    def _begin_stop(self, targets, shutdown=False):
        """创建异步收尾状态机，只选择本机受管理组件。"""
        self.video_sources.cancel()
        if self.operation:
            if shutdown:
                self.operation["shutdown"] = True
                self.operation["targets"] = list(reversed(COMPONENT_IDS))
            return False
        self._autostart = False
        self.operation = {
            "phase": "pause", "targets": list(targets), "shutdown": shutdown,
            "future": None, "deadline": time.monotonic() + 3.0,
            "next_call": 0.0,
        }
        self.last_operation = "正在暂停遥操"
        return True

    def request_shutdown(self):
        """信号与服务共用退出入口，不依赖 RViz 析构发消息。"""
        return self._begin_stop(list(reversed(COMPONENT_IDS)), shutdown=True)

    def _owned_alive(self, key):
        """只允许会话收尾流程调用本会话拥有的控制、记录服务。"""
        c = self.components[key]
        return c.ownership == "local" and c.process is not None

    def _advance_stop(self, now):
        """非阻塞推进暂停、保存和进程回收，超时错误保留在状态中。"""
        op = self.operation
        if not op:
            return
        if op["phase"] == "pause":
            if self._owned_alive("teleop"):
                if op["future"] is None and self.pause_client.service_is_ready():
                    op["future"] = self.pause_client.call_async(SetBool.Request(data=False))
                if op["future"] is not None and op["future"].done():
                    result = op["future"].result()
                    if not result or not result.success:
                        self.components["teleop"].state = "stopping"
                        self.components["teleop"].process.stop()
                elif now < op["deadline"]:
                    return
                else:
                    if op["future"] is not None:
                        self.pause_client.remove_pending_request(op["future"])
                    self.get_logger().error("暂停服务超时，终止受管理控制进程")
                    self.components["teleop"].state = "stopping"
                    self.components["teleop"].process.stop()
            op.update(phase="save", future=None,
                      deadline=now + float(self.config.get("save_timeout_s", 300.0)))
            self.last_operation = "正在停止录制并等待保存"
        if op["phase"] == "save":
            if self._owned_alive("recorder"):
                if op["future"] is None and self.save_client.service_is_ready() and now >= op["next_call"]:
                    op["future"] = self.save_client.call_async(Trigger.Request())
                if op["future"] is not None and op["future"].done():
                    result = op["future"].result()
                    op["future"] = None
                    op["next_call"] = now + 0.25
                    if result and not result.success:
                        self.last_operation = f"保存失败: {result.message}"
                        self.get_logger().error(self.last_operation)
                        op["error"] = self.last_operation
                        op["phase"] = "stop"
                    elif result and result.message == "idle":
                        op["phase"] = "stop"
                if op["phase"] == "save" and now < op["deadline"]:
                    return
                if op["phase"] == "save":
                    if op["future"] is not None:
                        self.save_client.remove_pending_request(op["future"])
                    op["error"] = "保存超时，数据可能未完整写入"
                    self.get_logger().error(op["error"])
            op["phase"] = "stop"
        if op["phase"] == "stop":
            pending = False
            for key in op["targets"]:
                component = self.components[key]
                if self._owned_alive(key):
                    component.state = "stopping"
                    component.process.stop()
                    pending = True
            if pending:
                return
            self.last_operation = op.get("error", "组件已停止")
            self.shutdown_complete = op["shutdown"]
            self.operation = None

    def _tick(self):
        """轮询进程与图发现，独立以 2 Hz 发布管理状态。"""
        now = time.monotonic()
        if now >= self._next_graph:
            self._nodes = {f"{namespace.rstrip('/')}/{name}" for name, namespace in self.get_node_names_and_namespaces()}
            self._next_graph = now + 0.5
        # 所有组件共享一次 /proc 快照，避免按组件重复扫描。
        table = process_table() if any(c.process for c in self.components.values()) else {}
        for key, component in self.components.items():
            if component.process is not None:
                code = component.process.poll(table=table)
                if code is not None:
                    intended = component.state == "stopping"
                    component.process = None
                    component.exit_code = str(code)
                    component.state = "stopped" if intended else "exited"
                    if not intended:
                        component.error = f"进程退出，退出码 {code}；查看日志"
                        if key == "recorder" and self._owned_alive("teleop"):
                            # 使用同一暂停状态机，服务无响应时仍能终止控制进程。
                            self._begin_stop([])
                            self.operation["error"] = "记录节点异常退出，已请求暂停遥操"
                elif component.state == "starting" and (
                    set(component.config.get("nodes", [])) & self._nodes
                    or component.health.samples
                ):
                    # ROS 节点已出现或已发布数据即可确认启动，数据有效性单独报告。
                    component.state = "running"
            elif component.config.get("mode") != "disabled" and component.ownership != "local":
                if component.config.get("mode") == "observe" or self._external_present(component):
                    component.ownership, component.state = "external", "external"
        if (self._autostart and now >= self._autostart_at and not self.operation
                and not self.video_sources.operation and not self.video_sources.unlock):
            self._autostart = False
            response = self._start_all_service(Trigger.Request(), Trigger.Response())
            self.last_operation = response.message
        self._advance_stop(now)
        self.video_sources.tick(now)
        if now >= self._next_publish:
            self._publish_status(now)
            self._next_publish = now + 0.5

    def _publish_status(self, now):
        """发布进程与数据健康的独立字段，面板不解析中文提示。"""
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        for key, component in self.components.items():
            healthy, detail = component.health.check()
            if key == "estimator" and "数据无效" in detail:
                detail += "；预测结果无效，请检查 UMI 图像中的夹爪标记、检测区域及相机/夹爪标定"
            mode = component.config.get("mode", "auto")
            level = DiagnosticStatus.OK if healthy else DiagnosticStatus.ERROR
            if component.state in ("stopped", "starting", "stopping") or mode == "disabled":
                level = DiagnosticStatus.WARN
            if component.state == "starting" and now - component.started > float(self.config.get("startup_grace_s", 30)):
                level = DiagnosticStatus.ERROR
            if component.state in ("failed", "exited"):
                level = DiagnosticStatus.ERROR
            item = DiagnosticStatus(name=key, level=level, hardware_id=component.ownership,
                                    message=component.error or detail)
            values = {
                "label": component.config.get("label", key), "mode": mode,
                "process_state": component.state, "healthy": str(healthy).lower(),
                "ownership": component.ownership, "nodes": ", ".join(component.config.get("nodes", [])),
                "topics": ", ".join(self.probes.get(key, [])), "ages": component.health.ages(),
                "exit_code": component.exit_code, "log_path": component.log_path,
            }
            if key == "recorder":
                values["output_directory"] = self.output_directory
            if key in ("umi_camera", "wrist_camera"):
                values.update(self.video_sources.fields(key))
            item.values = [KeyValue(key=k, value=v) for k, v in values.items()]
            message.status.append(item)
        message.status.append(DiagnosticStatus(
            name="session", message=self.last_operation,
            values=[KeyValue(key="busy", value=str(bool(self.operation or self.video_sources.operation or self.video_sources.unlock)).lower())],
        ))
        self.status_publisher.publish(message)

    def destroy_node(self):
        """释放本机会话锁；进程必须先经过收尾状态机。"""
        if hasattr(self, "video_sources"):
            self.video_sources.close()
        if hasattr(self, "_lock_file"):
            self._lock_file.close()
        return super().destroy_node()


def valid_odometry(message):
    """检查里程计的有限位置与非零四元数。"""
    p, q = message.pose.pose.position, message.pose.pose.orientation
    return all(math.isfinite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w)) and sum(
        v * v for v in (q.x, q.y, q.z, q.w)
    ) > 1e-12


def main(args=None):
    """信号只设置退出意图，继续 spin 到保存和子进程回收完成。"""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    node = None
    try:
        node = ComponentManager()
        while rclpy.ok() and not node.shutdown_complete:
            if stop.is_set():
                node.request_shutdown()
                stop.clear()
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
