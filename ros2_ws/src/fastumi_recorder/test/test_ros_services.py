"""经真实 ROS 服务和隔离输入话题验证录制契约，不向驱动发送命令。"""

import time
import uuid

import pytest
import rclpy
import rosbag2_py
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from rm_ros_interfaces.msg import Jointpos
from fastumi_interfaces.srv import (
    StartRecording, StopRecording, CancelRecording, CancelRecordingRequest,
    GetRecordingRequest, GetRecordingStatus,
    ListRecordings, DeleteRecording,
)
from fastumi_recorder.node import RecorderNode


pytestmark = pytest.mark.skipif(
    'mcap' not in rosbag2_py.get_registered_writers(),
    reason='rosbag2_storage_mcap is not installed')


def test_best_effort_image_subscription_override(tmp_path, monkeypatch):
    """独立录制入口可选 Best Effort 和自定义图像队列深度。"""
    monkeypatch.setenv('ROS_DOMAIN_ID', '224')
    rclpy.init()
    recorder = RecorderNode(parameter_overrides=[
        Parameter('dataset_root', value=str(tmp_path)),
        Parameter('image_reliability', value='best_effort'),
        Parameter('image_qos_depth', value=12),
    ])
    try:
        assert recorder.image_subscription.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
        assert recorder.image_subscription.qos_profile.depth == 12
    finally:
        recorder.close()
        recorder.destroy_node()
        rclpy.shutdown()


def test_ros_service_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', '223')
    rclpy.init()
    prefix = '/test_recorder_' + uuid.uuid4().hex
    params = {'dataset_root': str(tmp_path), 'joint_state_topic': prefix+'/joint',
              'joint_action_topic': prefix+'/action', 'gripper_state_topic': prefix+'/gripper',
              'gripper_action_topic': prefix+'/gripper_action', 'image_topic': prefix+'/image',
              'image_transport': 'raw',
              'image_qos_depth': 45,
              'tracker_odom_topic': prefix+'/tracker'}
    recorder = RecorderNode(parameter_overrides=[Parameter(key, value=value) for key, value in params.items()])
    assert recorder.image_subscription.qos_profile.reliability == ReliabilityPolicy.RELIABLE
    assert recorder.image_subscription.qos_profile.depth == 45
    client = Node('recorder_test_client')
    executor = SingleThreadedExecutor()
    executor.add_node(recorder)
    executor.add_node(client)
    publishers = {key: client.create_publisher(kind, prefix+'/'+key, 10)
                  for key, kind in [('joint', JointState), ('gripper', Float32),
                                    ('image', Image), ('action', Jointpos),
                                    ('gripper_action', Float32), ('tracker', Odometry)]}
    clients = {name: client.create_client(kind, '/fastumi/recording/'+name)
               for name, kind in [('start', StartRecording), ('stop', StopRecording),
                                  ('cancel', CancelRecording), ('get_status', GetRecordingStatus),
                                  ('cancel_request', CancelRecordingRequest),
                                  ('get_request', GetRecordingRequest),
                                  ('list', ListRecordings), ('delete', DeleteRecording)]}

    def call(name, request):
        assert clients[name].wait_for_service(timeout_sec=2)
        future = clients[name].call_async(request)
        executor.spin_until_future_complete(future, timeout_sec=3)
        assert future.done()
        return future.result()

    def pump(seconds=.2, actions=False):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            stamp = client.get_clock().now().to_msg()
            joints = JointState(name=[f'joint{i}' for i in range(1, 8)], position=[0.]*7)
            joints.header.stamp = stamp
            publishers['joint'].publish(joints)
            publishers['gripper'].publish(Float32(data=.8))
            publishers['gripper_action'].publish(Float32(data=.4))
            image = Image(width=16, height=16, step=48, encoding='rgb8', data=bytes(768))
            image.header.stamp = stamp
            publishers['image'].publish(image)
            tracker = Odometry()
            tracker.header.stamp = stamp
            tracker.pose.pose.orientation.w = 1.
            publishers['tracker'].publish(tracker)
            if actions:
                publishers['action'].publish(Jointpos(dof=7, joint=[0.]*7))
            for _ in range(4):
                executor.spin_once(timeout_sec=.005)

    try:
        not_ready = call('start', StartRecording.Request(request_id=str(uuid.uuid4())))
        assert not_ready.code == 'NOT_READY', not_ready.message
        pump(.4)
        request_id = str(uuid.uuid4())
        start = call('start', StartRecording.Request(request_id=request_id, dir_name='test', name='services'))
        assert start.success
        assert call('start', StartRecording.Request(request_id=request_id)).recording_id == start.recording_id
        assert call('get_request', GetRecordingRequest.Request(request_id=request_id)).state == 'recording'
        assert call('start', StartRecording.Request(request_id=str(uuid.uuid4()))).code == 'BUSY'
        pump(.4, actions=True)
        stopped = call('stop', StopRecording.Request(recording_id=start.recording_id))
        assert stopped.success and stopped.code == 'STOP_ACCEPTED'
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            status = call('get_status', GetRecordingStatus.Request()).status
            if status.state != 'saving':
                break
        assert status.state == 'idle', status.last_error
        assert status.last_completed.recording_id == start.recording_id
        assert call('get_request', GetRecordingRequest.Request(request_id=request_id)).state == 'completed'
        entries = call('list', ListRecordings.Request())
        assert entries.success and entries.total == 1
        info = entries.recordings[0]
        bag = tmp_path / info.relative_path / 'bag'
        assert (bag / 'metadata.yaml').is_file()
        assert list(bag.glob('*.mcap'))
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'),
                    rosbag2_py.ConverterOptions('', ''))
        topics = set()
        while reader.has_next():
            topic, _, timestamp_ns = reader.read_next()
            topics.add(topic)
            assert timestamp_ns > 0
        assert topics == {prefix + '/' + name for name in
                          ('joint', 'action', 'gripper', 'gripper_action', 'tracker', 'image')}
        del reader
        assert info.has_joint_action and info.image_frames > 0
        assert call('delete', DeleteRecording.Request(recording_id=start.recording_id)).success
        assert call('list', ListRecordings.Request()).total == 0
        assert call('delete', DeleteRecording.Request(recording_id='../../etc')).code == 'INVALID_ARGUMENT'
        pump()
        start = call('start', StartRecording.Request(request_id=str(uuid.uuid4())))
        assert call('cancel', CancelRecording.Request(recording_id=start.recording_id)).success
        blocked_id = str(uuid.uuid4())
        assert call('cancel_request', CancelRecordingRequest.Request(request_id=blocked_id)).state == 'cancelled'
        assert call('start', StartRecording.Request(request_id=blocked_id)).code == 'CANCELLED'
    finally:
        recorder.close()
        executor.shutdown()
        recorder.destroy_node()
        client.destroy_node()
        rclpy.shutdown()
