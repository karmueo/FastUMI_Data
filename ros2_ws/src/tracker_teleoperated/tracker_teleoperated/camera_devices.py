"""管理本机 UMI 相机设备选择，并与遥操和远端录制状态互锁。"""

from concurrent.futures import ThreadPoolExecutor
import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import SetBool, Trigger

from fastumi_usb_camera.capture import is_physical_video_device_path
from tracker_teleoperated.camera_catalog import scan_devices


class CameraDevices:
    """只管理本机 UMI 相机；末端相机由远端入口管理。"""

    def __init__(self, manager, qos):
        """创建 UMI 设备目录、参数接口和非阻塞切换状态机。"""
        self.manager = manager
        self.devices = []
        self.operation = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.scan = None
        self.next_scan = 0.0
        config = manager.components["umi_camera"].config.setdefault("parameters", {})
        default = "/dev/v4l/by-path/pci-0000:06:00.4-usb-0:2.4:1.0-video-index0"
        device = manager.declare_parameter(
            "umi_video_device", config.get("video_device", default)).value
        if not is_physical_video_device_path(device):
            raise ValueError(
                "UMI 相机必须使用 /dev/v4l/by-path/*-video-index0: " + device)
        config["video_device"] = device
        self.source = {
            "actual": device, "requested": device, "state": "deferred", "error": ""}
        self.publisher = manager.create_publisher(
            DiagnosticArray, "/tracker_teleoperated/camera_devices", qos)
        manager.create_service(
            Trigger, "/tracker_teleoperated/refresh_camera_devices", self.refresh)
        manager.add_on_set_parameters_callback(self._request)

    def close(self):
        """停止后台设备扫描。"""
        self.pool.shutdown(wait=True, cancel_futures=True)

    def refresh(self, _request, response):
        """请求下一轮立即扫描设备。"""
        self.next_scan = 0.0
        response.success, response.message = True, "已请求刷新 UMI 设备"
        return response

    def reason(self, _key="umi_camera"):
        """返回当前禁止 UMI 相机切换的原因。"""
        if self.manager.operation or self.operation:
            return "正在切换或停止组件"
        if self.manager.recording_enabled:
            age = time.monotonic() - self.manager.recording_seen
            if self.manager.recording_seen == 0 or age > 2.0:
                return "远端录制状态尚未确认"
            if self.manager.recording_state != "idle":
                return "远端正在录制或保存"
        teleop = self.manager.components["teleop"]
        if teleop.ownership == "external":
            return "遥操节点是外部节点，无法确认安全暂停"
        if teleop.ownership != "local" and self.manager._external_present(teleop):
            return "检测到外部遥操节点，无法确认安全暂停"
        return ""

    def _request(self, parameters):
        """将 UMI 设备参数变化转换为暂停、重启和验证事务。"""
        selected = [p for p in parameters if p.name == "umi_video_device"]
        if not selected:
            return SetParametersResult(successful=True)
        target = str(selected[-1].value)
        try:
            if not is_physical_video_device_path(target):
                raise ValueError("请选择 /dev/v4l/by-path/*-video-index0 物理端口")
            reason = self.reason()
            if reason:
                raise ValueError(reason)
            if target not in {item["device"] for item in self.devices if not item["error"]}:
                raise ValueError("设备不存在、不可访问或不是有效采集入口")
            if target == self.source["actual"]:
                return SetParametersResult(successful=True)
            running = [key for key in ("estimator", "umi_camera")
                       if self.manager._owned_alive(key)]
            self.operation = {
                "phase": "pause", "old": self.source["actual"], "new": target,
                "running": running, "future": None, "deadline": time.monotonic() + 3,
                "rollback": False, "failure": ""}
            self.source.update(requested=target, state="pending", error="")
            return SetParametersResult(successful=True)
        except ValueError as error:
            return SetParametersResult(successful=False, reason=str(error))

    def launch_topic(self, key):
        """返回夹爪估计器使用的固定 UMI 图像话题。"""
        if key != "estimator":
            raise ValueError("只有夹爪估计器使用 UMI 设备话题")
        return self.manager.probes["umi_camera"][0]

    def cancel(self):
        """取消尚未完成的切换并停止事务创建的进程。"""
        if not self.operation:
            return
        self._stop_running()
        self.source.update(state="error", error="停止操作已中止 UMI 设备切换")
        self.operation = None

    def _stop_running(self):
        """停止切换前运行的 UMI 相机及估计器。"""
        for key in self.operation["running"]:
            if self.manager._owned_alive(key):
                self.manager.components[key].state = "stopping"
                self.manager.components[key].process.stop()

    def _finish(self, error=""):
        """结束切换并保留可展示结果。"""
        self.source.update(
            state="error" if error else "ready", error=error,
            actual=self.manager.components["umi_camera"].config["parameters"]["video_device"])
        self.manager.last_operation = error or "UMI 设备切换完成，遥操保持暂停"
        self.operation = None

    def tick(self, now):
        """推进设备扫描和 UMI 相机切换状态机。"""
        if self.scan is not None and self.scan.done():
            message = DiagnosticArray()
            try:
                self.devices = self.scan.result()
                for device in self.devices:
                    message.status.append(DiagnosticStatus(
                        name=device["device"], message=device["error"],
                        level=DiagnosticStatus.ERROR if device["error"] else DiagnosticStatus.OK,
                        values=[KeyValue(key=k, value=v) for k, v in device.items()]))
            except Exception as error:
                message.status.append(DiagnosticStatus(
                    name="scan", level=DiagnosticStatus.ERROR, message=str(error)))
            self.publisher.publish(message)
            self.scan = None
        if now >= self.next_scan and self.scan is None:
            self.scan = self.pool.submit(scan_devices)
            self.next_scan = now + 2
        op = self.operation
        if not op or self.manager.operation:
            return
        if op["phase"] == "pause":
            if self.manager._owned_alive("teleop"):
                if op["future"] is None and self.manager.pause_client.service_is_ready():
                    op["future"] = self.manager.pause_client.call_async(
                        SetBool.Request(data=False))
                if op["future"] is None or not op["future"].done():
                    if now > op["deadline"]:
                        self._finish("暂停遥操超时，未切换 UMI 设备")
                    return
                if not op["future"].result().success:
                    self._finish("暂停遥操失败，未切换 UMI 设备")
                    return
            self._stop_running()
            op.update(phase="release", deadline=now + 15)
        if op["phase"] == "release":
            if any(self.manager.components[key].process for key in op["running"]):
                if now > op["deadline"]:
                    self._finish("释放 UMI 相机超时")
                return
            target = op["old"] if op["rollback"] else op["new"]
            self.manager.components["umi_camera"].config["parameters"]["video_device"] = target
            op.update(phase="start", deadline=now + float(
                self.manager.config.get("startup_grace_s", 30)))
        if op["phase"] == "start":
            for key in reversed(op["running"]):
                if self.manager.components[key].process:
                    continue
                ok, error = self.manager._start(key)
                if not ok:
                    if not op["rollback"]:
                        op.update(rollback=True, failure=error, phase="release")
                        return
                    self._finish(op["failure"] + "；恢复失败：" + error)
                    return
            op["phase"] = "verify"
        if op["phase"] == "verify":
            camera = self.manager.components["umi_camera"]
            if "umi_camera" in op["running"] and not (
                    camera.process and camera.health.samples and camera.health.check()[0]):
                if now > op["deadline"]:
                    if not op["rollback"]:
                        self._stop_running()
                        op.update(
                            rollback=True, failure="UMI 相机未收到有效图像", phase="release")
                    else:
                        self._finish(op["failure"] + "；恢复后仍无有效图像")
                return
            self._finish(op["failure"] + "；已恢复原设备" if op["rollback"] else "")

    def fields(self, key):
        """提供 UMI 设备或远端视频的只读字段。"""
        if key == "umi_camera":
            state = self.source["state"]
            if state == "deferred" and self.manager.components[key].process:
                state = "ready"
            return {
                "requested_video_device": self.source["requested"],
                "video_device": self.source["actual"], "source_state": state,
                "source_error": self.source["error"],
                "source_editable": str(not self.reason()).lower(),
                "source_reason": self.reason(),
                "image_topic": self.manager.probes[key][0]}
        return {
            "source_editable": "false", "source_reason": "末端相机由远端入口管理",
            "encoded_topic": self.manager.probes["wrist_encoder"][0],
            "image_topic": self.manager.probes["wrist_decoder"][0]}
