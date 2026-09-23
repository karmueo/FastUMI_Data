"""Recording services and an ordered, bounded MCAP writer queue."""

from copy import deepcopy
import queue
import shutil
import sqlite3
import threading
import time
import uuid

from rclpy.serialization import serialize_message

from fastumi_recorder.bag_writer import BagWriter
from fastumi_recorder.catalog import Catalog, RecordingError, atomic_json
from fastumi_recorder.request_ledger import RequestLedger, canonical_request_id


MAX_QUEUED_BYTES = 256 * 1024 * 1024
IMAGE_TYPES = {
    'raw': 'sensor_msgs/msg/Image',
    'jpeg': 'sensor_msgs/msg/CompressedImage',
    'ffmpeg': 'ffmpeg_image_transport_msgs/msg/FFMPEGPacket',
}
DEFAULT_TOPICS = {
    'joint_state': ('/joint_states', 'sensor_msgs/msg/JointState'),
    'joint_action': ('/rm_driver/movej_canfd_cmd', 'rm_ros_interfaces/msg/Jointpos'),
    'gripper_state': ('/motion_control/gripper_state', 'std_msgs/msg/Float32'),
    'gripper_action': ('/motion_control/gripper_command', 'std_msgs/msg/Float32'),
    'tracker_pose': ('/vive_tracker/odom', 'nav_msgs/msg/Odometry'),
}
SAMPLE_FIELDS = {
    'joint_state': 'joint_samples', 'joint_action': 'action_samples',
    'gripper_state': 'gripper_samples', 'tracker_pose': 'tracker_samples',
}


class RecorderEngine:
    """Serialize service transitions while a worker writes original ROS messages."""

    def __init__(self, root, *, dir_name='test', name='default_test',
                 record_camera=True, image_transport='ffmpeg', topics=None,
                 freshness=2.0, shutdown_timeout=120.0,
                 max_queued_bytes=MAX_QUEUED_BYTES, writer_factory=BagWriter):
        if freshness <= 0 or shutdown_timeout <= 0 or max_queued_bytes <= 0:
            raise ValueError('超时和队列容量必须大于零')
        if image_transport not in IMAGE_TYPES:
            raise ValueError('image_transport 必须是 raw、jpeg 或 ffmpeg')
        self.catalog = Catalog(root)
        self.requests = RequestLedger(self.catalog.root)
        self.requests.recover(self.catalog)
        self.dir_name, self.name = dir_name, name
        self.record_camera = record_camera
        self.topics = dict(topics or DEFAULT_TOPICS)
        if record_camera:
            self.topics.setdefault('image',
                                   ('/wrist_camera/image_raw/ffmpeg', IMAGE_TYPES[image_transport]))
        else:
            self.topics.pop('image', None)
        self.freshness, self.shutdown_timeout = freshness, shutdown_timeout
        self.max_queued_bytes, self.writer_factory = max_queued_bytes, writer_factory
        self.lock = threading.RLock()
        self.state = 'idle'
        self.current = {}
        saved = self.catalog.list(limit=1)[0]
        self.last_completed = dict(saved[0]) if saved else {}
        self.last_error = ''
        self._temporary = None
        self._current_request_id = None
        self._save_cancel_requested = False
        self._writer = None
        self._writer_active = False
        self._writer_error = ''
        self._finalize_queued = False
        self._generation = 0
        self._seen = {}
        self._cancelled = set()
        self._counts = {}
        self._reset_counts()
        self._queue = queue.Queue()
        self._queued_bytes = 0
        self._closing = False
        self._closed = False
        self._worker = threading.Thread(target=self._work, name='fastumi-mcap-writer', daemon=True)
        self._worker.start()

    def _reset_counts(self):
        self._counts = dict(joint_samples=0, action_samples=0, gripper_samples=0,
                            tracker_samples=0, received_frames=0, encoded_frames=0,
                            saved_frames=0, dropped_frames=0)

    def _check_id(self, recording_id):
        if not recording_id or recording_id != self.current.get('recording_id'):
            raise RecordingError('NOT_FOUND', '录制 ID 不是当前条目')

    def _fail(self, error):
        """Mark this episode failed and enqueue one writer-close barrier."""
        self._writer_error = str(error)
        self.last_error = f'{error}；未完成目录: {self._temporary}'
        self.state = 'error'
        if self._current_request_id:
            try:
                self.requests.put(self._current_request_id, state='failed',
                                  code='SAVE_FAILED', message=self.last_error)
            except sqlite3.Error as ledger_error:
                self.last_error += f'；台账更新失败: {ledger_error}'
        if not self._finalize_queued:
            self._finalize_queued = True
            self._queue.put(('finalize', self._generation, 'error', None))

    def start(self, dir_name='', name='', timestamp=None, request_id=None):
        """Open and register all MCAP topics before acknowledging the start."""
        with self.lock:
            try:
                request_id = canonical_request_id(request_id or str(uuid.uuid4()))
            except ValueError as error:
                raise RecordingError('INVALID_ARGUMENT', str(error)) from error
            row = self.requests.get(request_id)
            if row is not None:
                if row['start_success'] is not None:
                    if row['start_success']:
                        return row['recording_id'], row['start_code']
                    raise RecordingError(row['start_code'], row['start_message'])
                if row['state'] == 'cancelled':
                    self.requests.put(request_id, start_success=0, start_code='CANCELLED',
                                      start_message='启动请求已撤销')
                    raise RecordingError('CANCELLED', '启动请求已撤销')
            else:
                self.requests.put(request_id, state='pending')
            temporary = None
            writer = None
            try:
                if self._closing or self._writer_active or self.state in ('recording', 'saving'):
                    raise RecordingError('BUSY', '录制器正在录制、保存或退出')
                required = ['joint', 'gripper'] + (['image'] if self.record_camera else [])
                now = time.monotonic()
                missing = [stream for stream in required
                           if stream not in self._seen or now - self._seen[stream] > self.freshness]
                if missing:
                    raise RecordingError('NOT_READY', '缺少最近有效输入: ' + ', '.join(missing))
                stamp = time.time() if timestamp is None else timestamp
                info, temporary = self.catalog.allocate(dir_name or self.dir_name,
                                                        name or self.name, stamp)
                writer = self.writer_factory(temporary / 'bag', self.topics)
                self.requests.put(request_id, state='recording', recording_id=info['recording_id'],
                                  start_success=1, start_code='STARTED', start_message='STARTED',
                                  code='STARTED', message='录制已启动')
                self._writer = writer
                self._writer_active = True
                self._generation += 1
                self.current, self._temporary = info, temporary
                self._current_request_id = request_id
                self._save_cancel_requested = False
                self._writer_error = ''
                self._finalize_queued = False
                self._reset_counts()
                self.last_error = ''
                self.state = 'recording'
                return info['recording_id'], 'STARTED'
            except Exception as error:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                if temporary is not None:
                    shutil.rmtree(temporary, ignore_errors=True)
                code = error.code if isinstance(error, RecordingError) else 'IO_ERROR'
                self.requests.put(request_id, state='failed', start_success=0,
                                  start_code=code, start_message=str(error),
                                  code=code, message=str(error))
                if isinstance(error, RecordingError):
                    raise
                raise RecordingError(code, str(error)) from error

    def stop(self, recording_id, timestamp=None):
        """Queue a barrier after every message accepted before this call."""
        with self.lock:
            if recording_id in self.catalog.entries:
                return recording_id, 'ALREADY_SAVED'
            self._check_id(recording_id)
            if self.state == 'saving':
                return recording_id, 'ALREADY_STOPPING'
            if self.state == 'error':
                raise RecordingError('SAVE_FAILED', self.last_error)
            if self.state != 'recording':
                raise RecordingError('NOT_RECORDING', '当前条目没有在录制')
            stamp = time.time() if timestamp is None else timestamp
            if stamp < self.current['started_at']:
                raise RecordingError('INVALID_ARGUMENT', '停止时间早于开始时间')
            if self._current_request_id:
                self.requests.put(self._current_request_id, state='saving',
                                  code='SAVING', message='正在保存')
            self.current.update(stopped_at=stamp, duration=stamp - self.current['started_at'])
            self.state = 'saving'
            self._finalize_queued = True
            self._queue.put(('finalize', self._generation, 'save', None))
            return recording_id, 'STOP_ACCEPTED'

    def cancel(self, recording_id):
        """Close and remove an active bag before acknowledging cancellation."""
        with self.lock:
            if recording_id in self._cancelled:
                return recording_id, 'ALREADY_CANCELLED'
            self._check_id(recording_id)
            if self.state == 'saving':
                raise RecordingError('BUSY', '保存阶段不能取消')
            if self.state != 'recording':
                raise RecordingError('NOT_RECORDING', '当前条目没有在录制')
            done = threading.Event()
            self.state = 'saving'
            self._save_cancel_requested = True
            self._finalize_queued = True
            if self._current_request_id:
                self.requests.put(self._current_request_id, cancel_requested=1,
                                  code='CANCEL_ACCEPTED', message='正在撤销录制')
            self._queue.put(('finalize', self._generation, 'cancel', done))
        if not done.wait(self.shutdown_timeout):
            raise RecordingError('IO_ERROR', '等待 MCAP 关闭超时')
        with self.lock:
            if self.state == 'error':
                raise RecordingError('IO_ERROR', self.last_error)
            return recording_id, 'CANCELLED'

    def cancel_request(self, request_id):
        """Tombstone unknown starts or cancel by durable request identity."""
        try:
            request_id = canonical_request_id(request_id)
        except ValueError as error:
            raise RecordingError('INVALID_ARGUMENT', str(error)) from error
        with self.lock:
            row = self.requests.get(request_id)
            if row and row['recording_id'] in self.catalog.entries:
                return row['recording_id'], 'COMPLETED', 'completed'
            if row is None:
                self.requests.put(request_id, state='cancelled', code='CANCELLED',
                                  message='启动前已撤销', cancel_requested=1)
                return '', 'CANCELLED', 'cancelled'
            if row['state'] in ('cancelled', 'failed', 'completed'):
                return row['recording_id'], row['code'], row['state']
            if request_id != self._current_request_id:
                raise RecordingError('NOT_FOUND', '请求不是当前录制')
            if self.state == 'saving':
                self.requests.put(request_id, code='CANCEL_ACCEPTED',
                                  message='保存中止已请求', cancel_requested=1)
                self._save_cancel_requested = True
                return row['recording_id'], 'CANCEL_ACCEPTED', 'saving'
            if self.state != 'recording':
                raise RecordingError('NOT_RECORDING', '请求没有活动录制')
            recording_id = row['recording_id']
        self.cancel(recording_id)
        return recording_id, 'CANCELLED', 'cancelled'

    def get_request(self, request_id):
        try:
            request_id = canonical_request_id(request_id)
        except ValueError as error:
            raise RecordingError('INVALID_ARGUMENT', str(error)) from error
        with self.lock:
            row = self.requests.get(request_id)
            if row and row['recording_id'] in self.catalog.entries:
                row.update(state='completed', code='COMPLETED', message='录制已保存')
            return row

    def delete(self, recording_id):
        with self.lock:
            if recording_id == self.current.get('recording_id') and self._writer_active:
                raise RecordingError('BUSY', '不能删除正在录制或保存的条目')
            return recording_id, self.catalog.delete(recording_id)

    def list(self, **kwargs):
        with self.lock:
            return self.catalog.list(**kwargs)

    def record(self, stream, message, timestamp_ns, *, valid_for_readiness=False):
        """Queue original CDR; validity affects readiness only, never bag content."""
        with self.lock:
            readiness = {'joint_state': 'joint', 'gripper_state': 'gripper',
                         'tracker_pose': 'tracker', 'image': 'image'}.get(stream)
            if valid_for_readiness and readiness:
                self._seen[readiness] = time.monotonic()
            if self.state != 'recording' or stream not in self.topics:
                return
            if stream == 'image':
                self._counts['received_frames'] += 1
            try:
                serialized = serialize_message(message)
            except Exception as error:
                if stream == 'image':
                    self._counts['dropped_frames'] += 1
                self._fail(f'ROS 消息序列化失败: {error}')
                return
            if self._queued_bytes + len(serialized) > self.max_queued_bytes:
                if stream == 'image':
                    self._counts['dropped_frames'] += 1
                self._fail('MCAP 写入队列已满')
                return
            self._queued_bytes += len(serialized)
            self._queue.put(('write', self._generation, stream,
                             self.topics[stream][0], serialized, int(timestamp_ns)))

    def status(self, timestamp=None):
        with self.lock:
            now = time.monotonic()
            duration = self.current.get('duration', 0.0)
            if self.state == 'recording':
                duration = max(0.0, (time.time() if timestamp is None else timestamp)
                               - self.current['started_at'])
            return dict(state=self.state, recording_id=self.current.get('recording_id', ''),
                        current=deepcopy(self.current), last_completed=deepcopy(self.last_completed),
                        duration=duration, last_error=self.last_error, **self._counts,
                        **{f'{s}_age': now - self._seen[s] if s in self._seen else -1.0
                           for s in ('joint', 'gripper', 'image', 'tracker')})

    def _work(self):
        """Write messages and process ordered close/publish barriers."""
        while True:
            task = self._queue.get()
            if task[0] == 'shutdown':
                return
            if task[0] == 'write':
                _, generation, stream, topic, serialized, timestamp_ns = task
                try:
                    with self.lock:
                        should_write = generation == self._generation and not self._writer_error
                    if should_write:
                        self._writer.write(topic, serialized, timestamp_ns)
                        with self.lock:
                            field = SAMPLE_FIELDS.get(stream)
                            if field:
                                self._counts[field] += 1
                            elif stream == 'image':
                                self._counts['encoded_frames'] += 1
                except Exception as error:
                    with self.lock:
                        self._fail(f'MCAP 写入失败: {error}')
                finally:
                    with self.lock:
                        self._queued_bytes -= len(serialized)
                continue
            _, generation, mode, done = task
            try:
                writer, self._writer = self._writer, None
                close_error = None
                if writer is not None:
                    try:
                        writer.close()
                    except Exception as error:
                        close_error = error
                with self.lock:
                    cancel = mode == 'cancel' or self._save_cancel_requested
                    failed = bool(self._writer_error) or mode == 'error'
                    if cancel:
                        shutil.rmtree(self._temporary)
                        if self._current_request_id:
                            self.requests.put(self._current_request_id, state='cancelled',
                                              code='CANCELLED', message='录制已撤销')
                        self._cancelled.add(self.current['recording_id'])
                        self.state = 'idle'
                    elif close_error is not None:
                        raise close_error
                    elif failed:
                        atomic_json(self._temporary / 'error.json',
                                    {'error': self._writer_error,
                                     'recording_id': self.current['recording_id']})
                        self.state = 'error'
                    else:
                        info = dict(self.current)
                        info.update(image_frames=self._counts['encoded_frames'],
                                    has_joint_action=self._counts['action_samples'] > 0,
                                    joint_samples=self._counts['joint_samples'],
                                    action_samples=self._counts['action_samples'],
                                    gripper_samples=self._counts['gripper_samples'],
                                    tracker_samples=self._counts['tracker_samples'])
                        self.catalog.publish(info, self._temporary)
                        if self._current_request_id:
                            try:
                                self.requests.put(self._current_request_id, state='completed',
                                                  code='COMPLETED', message='录制已保存')
                            except sqlite3.Error as error:
                                self.last_error = f'录制已保存，但请求台账更新失败: {error}'
                        self.current = self.last_completed = dict(info)
                        self._counts['saved_frames'] = info['image_frames']
                        self.state = 'idle'
            except Exception as error:
                with self.lock:
                    self.last_error = f'{error}；未完成目录: {self._temporary}'
                    self.state = 'error'
                    if self._current_request_id:
                        try:
                            self.requests.put(self._current_request_id, state='failed',
                                              code='SAVE_FAILED', message=self.last_error)
                        except sqlite3.Error:
                            pass
                    try:
                        atomic_json(self._temporary / 'error.json',
                                    {'error': str(error), 'recording_id': self.current['recording_id']})
                    except OSError:
                        pass
            finally:
                with self.lock:
                    self._writer_active = False
                    self._save_cancel_requested = False
                if done is not None:
                    done.set()

    def close(self, timestamp=None):
        """Auto-stop an active episode and wait for its MCAP writer."""
        with self.lock:
            if self._closed:
                return not self._worker.is_alive()
            self._closed = self._closing = True
            if self.state == 'recording':
                self.stop(self.current['recording_id'], timestamp)
            self._queue.put(('shutdown',))
        self._worker.join(self.shutdown_timeout)
        completed = not self._worker.is_alive()
        if not completed:
            with self.lock:
                self.last_error = '退出保存超时；未完成 MCAP 保留在隐藏临时目录'
                self.state = 'error'
        else:
            self.requests.close()
            self.catalog.close()
        return completed
