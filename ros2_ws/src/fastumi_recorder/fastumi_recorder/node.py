"""将本地硬件话题接入录制引擎，提供强类型 ROS 服务。"""

import math
from pathlib import Path
import signal
import sqlite3
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from nav_msgs.msg import Odometry
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float32
from rm_ros_interfaces.msg import Jointpos
from fastumi_interfaces.msg import RecordingInfo, RecordingStatus
from fastumi_interfaces.srv import (
    CancelRecording, CancelRecordingRequest, DeleteRecording, GetRecordingRequest,
    GetRecordingStatus, ListRecordings, StartRecording, StopRecording,
)

from fastumi_recorder.catalog import RecordingError, default_dataset_root
from fastumi_recorder.core import JOINT_NAMES
from fastumi_recorder.engine import RecorderEngine, ffmpeg_packet_to_h264


PREFIX = '/fastumi/recording'


def info_message(info):
    """将持久化摘要转换为共享接口。"""
    return RecordingInfo(**info)


class RecorderNode(Node):
    """单 executor 接收 ROS 请求，后台线程只负责编码和文件写入。"""

    def __init__(self, parameter_overrides=None):
        """初始化配置和本实例持有的资源。"""
        super().__init__('fastumi_recorder', parameter_overrides=parameter_overrides)
        defaults = dict(
            dataset_root='', dir_name='test', name='default_test', camera_fps=30,
            record_camera=True, input_freshness=2.0, shutdown_save_timeout=120.0,
            joint_state_topic='/joint_states', joint_action_topic='/rm_driver/movej_canfd_cmd',
            gripper_state_topic='/motion_control/gripper_state',
            gripper_action_topic='/motion_control/gripper_command',
            tracker_odom_topic='/vive_tracker/odom',
            image_topic='/wrist_camera/image_raw/ffmpeg', image_transport='ffmpeg',
        )
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        values = {name: self.get_parameter(name).value for name in defaults}
        root = values['dataset_root']
        if root and not Path(root).expanduser().is_absolute():
            raise ValueError('dataset_root 覆盖值必须为绝对路径')
        image_transport = str(values['image_transport']).strip().lower()
        if image_transport not in ('raw', 'jpeg', 'ffmpeg'):
            raise ValueError('image_transport 必须是 raw、jpeg 或 ffmpeg')
        self.engine = RecorderEngine(
            root or default_dataset_root(), dir_name=values['dir_name'], name=values['name'],
            fps=values['camera_fps'], record_camera=values['record_camera'],
            image_transport=image_transport,
            freshness=values['input_freshness'], shutdown_timeout=values['shutdown_save_timeout'],
        )
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(RecordingStatus, PREFIX + '/status', qos)
        self._published_state = None
        for service, name, handler in (
            (StartRecording, 'start', self._start), (StopRecording, 'stop', self._stop),
            (CancelRecording, 'cancel', self._cancel), (DeleteRecording, 'delete', self._delete),
            (CancelRecordingRequest, 'cancel_request', self._cancel_request),
            (GetRecordingRequest, 'get_request', self._get_request),
            (GetRecordingStatus, 'get_status', self._get_status), (ListRecordings, 'list', self._list),
        ):
            self.create_service(service, PREFIX + '/' + name, handler)
        for message_type, key, handler, qos_value in (
            (JointState, 'joint_state_topic', self._joint_state, 10),
            (Jointpos, 'joint_action_topic', self._joint_action, 10),
            (Float32, 'gripper_state_topic', self._gripper_state, 10),
            (Float32, 'gripper_action_topic', self._gripper_action, 10),
            (Odometry, 'tracker_odom_topic', self._tracker, qos_profile_sensor_data),
        ):
            self.create_subscription(message_type, values[key], handler, qos_value)
        if values['record_camera']:
            image_type, image_handler = {
                'raw': (Image, self._image),
                'jpeg': (CompressedImage, self._compressed_image),
                'ffmpeg': (FFMPEGPacket, self._ffmpeg_image),
            }[image_transport]
            self.create_subscription(
                image_type, values['image_topic'], image_handler, qos_profile_sensor_data)
        self.create_timer(0.5, self.publish_status)
        self.create_timer(0.05, self._publish_change)
        self.publish_status()
        self.get_logger().info(f'录制服务就绪，输出: {self.engine.catalog.root}')

    def _now(self):
        """返回当前 ROS 时钟的浮点秒。"""
        return self.get_clock().now().nanoseconds * 1e-9

    def _stamp(self, message):
        """优先使用有效消息时间，否则使用本机 ROS 时间。"""
        value = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        return value if value > 0 else self._now()

    def _status_message(self):
        """将引擎快照转换为强类型 ROS 状态。"""
        status = self.engine.status(self._now())
        for key in ('current', 'last_completed'):
            status[key] = info_message(status[key])
        return RecordingStatus(**status)

    def publish_status(self):
        """发布状态并记录最后一次发布的状态标识。"""
        message = self._status_message()
        self.publisher.publish(message)
        self._published_state = (message.state, message.recording_id, message.last_error)

    def _publish_change(self):
        """及时发布后台保存引起的状态变化。"""
        status = self.engine.status(self._now())
        if (status['state'], status['recording_id'], status['last_error']) != self._published_state:
            self.publish_status()

    def _operation(self, response, function, *args):
        """将事务结果或异常转换为统一服务响应。"""
        try:
            response.recording_id, response.code = function(*args)
            response.success = True
            response.message = response.code
        except RecordingError as error:
            response.success, response.code, response.message = False, error.code, str(error)
        except (OSError, ValueError, sqlite3.Error) as error:
            response.success, response.code, response.message = False, 'IO_ERROR', str(error)
        self.publish_status()
        return response

    def _start(self, request, response):
        """处理开始请求，任务空值使用配置默认值。"""
        if not request.request_id:
            response.success, response.code, response.message = False, 'INVALID_ARGUMENT', 'request_id 必填'
            return response
        return self._operation(response, self.engine.start, request.dir_name,
                               request.name, self._now(), request.request_id)

    def _stop(self, request, response):
        """处理带录制 ID 的幂等停止请求。"""
        return self._operation(response, self.engine.stop, request.recording_id, self._now())

    def _cancel(self, request, response):
        """处理尚未提交保存的录制取消请求。"""
        return self._operation(response, self.engine.cancel, request.recording_id)

    def _cancel_request(self, request, response):
        """撤销请求身份，即使启动服务尚未收到它。"""
        try:
            response.recording_id, response.code, response.state = self.engine.cancel_request(request.request_id)
            response.success = response.state == 'cancelled' or response.code == 'CANCEL_ACCEPTED'
            response.message = response.code
        except (RecordingError, OSError, sqlite3.Error) as error:
            response.success = False
            response.code = getattr(error, 'code', 'IO_ERROR')
            response.message = str(error)
        self.publish_status()
        return response

    def _get_request(self, request, response):
        """读取持久化的请求状态。"""
        try:
            row = self.engine.get_request(request.request_id)
            response.found = row is not None
            if row:
                for key in ('state', 'recording_id', 'code', 'message'):
                    setattr(response, key, row[key])
        except (RecordingError, OSError, sqlite3.Error) as error:
            response.code = getattr(error, 'code', 'IO_ERROR')
            response.message = str(error)
        return response

    def _delete(self, request, response):
        """处理已保存条目的回收请求。"""
        return self._operation(response, self.engine.delete, request.recording_id)

    def _get_status(self, request, response):
        """返回当前权威状态。"""
        response.status = self._status_message()
        return response

    def _list(self, request, response):
        """返回过滤、分页后的持久化摘要。"""
        try:
            entries, total = self.engine.list(dir_name=request.dir_name, name=request.name,
                                              offset=request.offset, limit=request.limit)
            response.recordings = [info_message(info) for info in entries]
            response.total, response.success, response.code = total, True, 'OK'
        except (RecordingError, OSError, ValueError) as error:
            response.success = False
            response.code = getattr(error, 'code', 'IO_ERROR')
            response.message = str(error)
        return response

    def _joint_state(self, message):
        """按 joint1 至 joint7 顺序记录有限弧度值。"""
        if len(message.name) != len(message.position):
            return
        positions = dict(zip(message.name, message.position))
        if any(name not in positions for name in JOINT_NAMES):
            return
        values = [float(positions[name]) for name in JOINT_NAMES]
        if all(math.isfinite(value) for value in values):
            self.engine.add('joint_state', self._stamp(message), values)

    def _joint_action(self, message):
        """只记录合法七轴 CANFD 目标。"""
        if message.dof == 7 and len(message.joint) == 7 and all(math.isfinite(v) for v in message.joint):
            self.engine.add('joint_action', self._now(), list(message.joint))

    def _gripper_state(self, message):
        """记录零至一范围内的真实夹爪开度。"""
        if math.isfinite(message.data) and 0 <= message.data <= 1:
            self.engine.add('gripper_state', self._now(), float(message.data))

    def _gripper_action(self, message):
        """记录合法的归一化夹爪目标。"""
        if math.isfinite(message.data) and 0 <= message.data <= 1:
            self.engine.add('gripper_action', self._now(), float(message.data))

    def _tracker(self, message):
        """记录可选远端位姿，保留米和 xyzw 四元数语义。"""
        pose = message.pose.pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        if all(math.isfinite(v) for v in position + orientation) and any(orientation):
            self.engine.add('tracker_pose', self._stamp(message), position,
                            orientation=orientation, frame_id=message.header.frame_id)

    def _image(self, message):
        """验证图像布局后入队，避免无效输入刷新就绪时间。"""
        channels = {'rgb8': 3, 'bgr8': 3, 'mono8': 1}.get(message.encoding.lower())
        if (channels and message.width > 0 and message.height > 0
                and message.step >= message.width * channels
                and len(message.data) >= message.step * message.height):
            self.engine.image(self._stamp(message), message)

    def _compressed_image(self, message):
        """验证相机原生 JPEG 后直接入队，不执行解码或重复编码。"""
        image_format = message.format.lower()
        data = bytes(message.data)
        if ('jpeg' in image_format or 'jpg' in image_format) and len(data) >= 4 \
                and data[:2] == b'\xff\xd8' and data[-2:] == b'\xff\xd9':
            self.engine.image(self._stamp(message), message)

    def _ffmpeg_image(self, message):
        """验证 H.264 transport 包后入队，并保留采集时间戳和关键帧标志。"""
        try:
            ffmpeg_packet_to_h264(message)
        except (TypeError, ValueError):
            return
        self.engine.image(self._stamp(message), message)

    def close(self):
        """等待保存并报告退出时未完成的数据。"""
        complete = self.engine.close(self._now())
        if not complete or self.engine.state == 'error':
            self.get_logger().error(self.engine.last_error)
        return complete


def main(args=None):
    """信号先触发保存屏障，之后才销毁 ROS 上下文。"""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = RecorderNode()
    stopping = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    try:
        while rclpy.ok() and not stopping.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
