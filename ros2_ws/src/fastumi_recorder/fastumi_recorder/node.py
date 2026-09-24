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
from fastumi_recorder.engine import DEFAULT_TOPICS, IMAGE_TYPES, RecorderEngine


PREFIX = '/fastumi/recording'


def info_message(info):
    """将持久化摘要转换为共享接口。"""
    return RecordingInfo(**info)


class RecorderNode(Node):
    """单 executor 接收 ROS 请求，后台线程顺序写入 MCAP。"""

    def __init__(self, parameter_overrides=None):
        """初始化配置和本实例持有的资源。"""
        super().__init__('fastumi_recorder', parameter_overrides=parameter_overrides)
        defaults = dict(
            dataset_root='', dir_name='test', name='default_test',
            record_camera=True, input_freshness=2.0, shutdown_save_timeout=120.0,
            joint_state_topic='/joint_states', joint_action_topic='/rm_driver/movej_canfd_cmd',
            gripper_state_topic='/motion_control/gripper_state',
            gripper_action_topic='/motion_control/gripper_command',
            tracker_odom_topic='/vive_tracker/odom',
            image_topic='/wrist_camera/image_raw/ffmpeg', image_transport='ffmpeg',
            image_reliability='reliable', image_qos_depth=30,
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
        image_reliability = str(values['image_reliability']).strip().lower()
        if image_reliability not in ('reliable', 'best_effort'):
            raise ValueError('image_reliability 只能是 reliable 或 best_effort')
        image_qos_depth = values['image_qos_depth']
        if type(image_qos_depth) is not int or image_qos_depth <= 0:
            raise ValueError('image_qos_depth 必须是正整数')
        topics = {
            stream: (values[parameter], message_type)
            for stream, parameter, message_type in (
                ('joint_state', 'joint_state_topic', DEFAULT_TOPICS['joint_state'][1]),
                ('joint_action', 'joint_action_topic', DEFAULT_TOPICS['joint_action'][1]),
                ('gripper_state', 'gripper_state_topic', DEFAULT_TOPICS['gripper_state'][1]),
                ('gripper_action', 'gripper_action_topic', DEFAULT_TOPICS['gripper_action'][1]),
                ('tracker_pose', 'tracker_odom_topic', DEFAULT_TOPICS['tracker_pose'][1]),
            )
        }
        if values['record_camera']:
            topics['image'] = (values['image_topic'], IMAGE_TYPES[image_transport])
        self.engine = RecorderEngine(
            root or default_dataset_root(), dir_name=values['dir_name'], name=values['name'],
            record_camera=values['record_camera'], image_transport=image_transport, topics=topics,
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
            self.image_subscription = self.create_subscription(
                image_type, values['image_topic'], image_handler,
                QoSProfile(
                    depth=image_qos_depth,
                    reliability=(ReliabilityPolicy.RELIABLE
                                 if image_reliability == 'reliable'
                                 else ReliabilityPolicy.BEST_EFFORT),
                    durability=DurabilityPolicy.VOLATILE))
        self.create_timer(0.5, self.publish_status)
        self.create_timer(0.05, self._publish_change)
        self.publish_status()
        self.get_logger().info(f'录制服务就绪，输出: {self.engine.catalog.root}')

    def _now_ns(self):
        """返回当前 ROS 接收时间，单位为纳秒。"""
        return self.get_clock().now().nanoseconds

    def _now(self):
        """返回当前 ROS 时钟的浮点秒。"""
        return self.get_clock().now().nanoseconds * 1e-9

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
        """保存原始关节消息，七轴有效性仅用于启动就绪门控。"""
        positions = dict(zip(message.name, message.position))
        valid = (len(message.name) == len(message.position)
                 and all(f'joint{i}' in positions for i in range(1, 8))
                 and all(math.isfinite(positions[f'joint{i}']) for i in range(1, 8)))
        self.engine.record('joint_state', message, self._now_ns(), valid_for_readiness=valid)

    def _joint_action(self, message):
        """保存未经筛选的关节指令。"""
        self.engine.record('joint_action', message, self._now_ns())

    def _gripper_state(self, message):
        """保存原始夹爪状态；有效范围仅用于就绪门控。"""
        valid = math.isfinite(message.data) and 0 <= message.data <= 1
        self.engine.record('gripper_state', message, self._now_ns(), valid_for_readiness=valid)

    def _gripper_action(self, message):
        """保存原始夹爪指令。"""
        self.engine.record('gripper_action', message, self._now_ns())

    def _tracker(self, message):
        """保存原始 Tracker odom；有效位姿用于输入 age。"""
        pose = message.pose.pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        valid = all(math.isfinite(v) for v in position + orientation) and any(orientation)
        self.engine.record('tracker_pose', message, self._now_ns(), valid_for_readiness=valid)

    def _image(self, message):
        """原样保存图像，布局校验仅影响启动就绪。"""
        channels = {'rgb8': 3, 'bgr8': 3, 'mono8': 1}.get(message.encoding.lower())
        valid = bool(channels and message.width > 0 and message.height > 0
                     and message.step >= message.width * channels
                     and len(message.data) >= message.step * message.height)
        self.engine.record('image', message, self._now_ns(), valid_for_readiness=valid)

    def _compressed_image(self, message):
        """原样保存 JPEG 消息，检查仅用于启动就绪。"""
        image_format = message.format.lower()
        data = bytes(message.data)
        valid = (('jpeg' in image_format or 'jpg' in image_format) and len(data) >= 4
                 and data[:2] == b'\xff\xd8' and data[-2:] == b'\xff\xd9')
        self.engine.record('image', message, self._now_ns(), valid_for_readiness=valid)

    def _ffmpeg_image(self, message):
        """原样保存 H.264 transport 包。"""
        valid = (str(message.encoding).split(';', 1)[0].strip().lower() == 'h264'
                 and message.width > 0 and message.height > 0 and bool(message.data))
        self.engine.record('image', message, self._now_ns(), valid_for_readiness=valid)

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
