"""与 ROS 无关的录制状态机、后台编码和服务事务。"""

from copy import deepcopy
import queue
import shutil
import sqlite3
import threading
import time
import uuid

import numpy as np

from fastumi_recorder.catalog import Catalog, RecordingError, atomic_json
from fastumi_recorder.core import RecordingSession
from fastumi_recorder.request_ledger import RequestLedger, canonical_request_id
from fastumi_recorder.storage import write_episode


class Encoder:
    """持有 FFmpeg 子进程，退出超时时主动中断管道写入。"""

    def __init__(self):
        """初始化配置和本实例持有的资源。"""
        self.executable = shutil.which('ffmpeg')
        if not self.executable:
            raise RecordingError('NOT_READY', '找不到系统 ffmpeg')
        self._lock = threading.Lock()
        self._process = None
        self._aborted = False
        self._cancel_current = False

    def register(self, process):
        """登记编码进程，已中止时立即终止新进程。"""
        with self._lock:
            self._process = process
            if self._aborted or self._cancel_current:
                process.kill()

    def unregister(self, process):
        """清除已结束编码进程的引用。"""
        with self._lock:
            if self._process is process:
                self._process = None

    def abort(self):
        """中止当前及后续编码，解除阻塞的输入管道。"""
        with self._lock:
            self._aborted = True
            if self._process is not None and self._process.poll() is None:
                self._process.kill()

    def abort_current(self):
        """Interrupt this save without disabling future encoding."""
        with self._lock:
            self._cancel_current = True
            if self._process is not None and self._process.poll() is None:
                self._process.kill()

    def reset_cancel(self):
        """Allow encoding for a newly accepted recording."""
        with self._lock:
            self._cancel_current = False


def image_to_jpeg(message):
    """返回已校验的相机原生 JPEG 字节。"""
    if hasattr(message, 'format') and not hasattr(message, 'encoding'):
        image_format = str(message.format).lower()
        data = bytes(message.data)
        if 'jpeg' not in image_format and 'jpg' not in image_format:
            raise ValueError(f'不支持的压缩图像格式: {message.format}')
        if len(data) < 4 or data[:2] != b'\xff\xd8' or data[-2:] != b'\xff\xd9':
            raise ValueError('JPEG 数据不完整')
        return data
    raise ValueError('JPEG 模式需要 sensor_msgs/msg/CompressedImage')


def image_to_bgr24(message):
    """按 ROS 行跨度将 bgr8/rgb8/mono8 转成连续 BGR24 字节及宽高。"""
    channels = {'bgr8': 3, 'rgb8': 3, 'mono8': 1}.get(
        str(message.encoding).lower())
    if channels is None:
        raise ValueError(f'不支持的图像编码: {message.encoding}')
    width, height, step = int(message.width), int(message.height), int(message.step)
    if width <= 0 or height <= 0 or step < width * channels:
        raise ValueError('图像尺寸或行跨度无效')
    data = bytes(message.data)
    if len(data) < step * height:
        raise ValueError('图像数据不足')
    if channels == 3 and step == width * 3 and message.encoding.lower() == 'bgr8':
        return data[:step * height], (width, height)
    rows = np.frombuffer(data, dtype=np.uint8, count=step * height).reshape(height, step)
    pixels = rows[:, :width * channels].reshape(height, width, channels)
    if channels == 1:
        bgr = np.repeat(pixels, 3, axis=2)
    elif message.encoding.lower() == 'rgb8':
        bgr = pixels[:, :, ::-1]
    else:
        bgr = pixels
    return bgr.tobytes(), (width, height)


def ffmpeg_packet_to_h264(message):
    """校验 FFmpeg transport 的 H.264 包并返回内容及关键帧标志。"""
    codec = str(message.encoding).split(';', 1)[0].strip().lower()
    if codec != 'h264':
        raise ValueError(f'不支持的 FFmpeg 编码: {message.encoding}')
    if int(message.width) <= 0 or int(message.height) <= 0:
        raise ValueError('FFmpeg 视频尺寸无效')
    data = bytes(message.data)
    if not data:
        raise ValueError('FFmpeg 视频包为空')
    return data, bool(int(message.flags) & 0x01)


class RecorderEngine:
    """通过单一事务锁串行化服务；耗时保存不占用事务锁。"""

    def __init__(self, root, *, dir_name='test', name='default_test', fps=30,
                 record_camera=True, image_transport='ffmpeg', freshness=2.0,
                 shutdown_timeout=120.0):
        """初始化配置和本实例持有的资源。"""
        if fps <= 0 or freshness <= 0 or shutdown_timeout <= 0:
            raise ValueError('帧率和超时时间必须大于零')
        if image_transport not in ('raw', 'jpeg', 'ffmpeg'):
            raise ValueError('image_transport 必须是 raw、jpeg 或 ffmpeg')
        self.encoder = Encoder()
        self.catalog = Catalog(root)
        self.requests = RequestLedger(self.catalog.root)
        self.requests.recover(self.catalog)
        self.dir_name, self.name = dir_name, name
        self.fps, self.record_camera = fps, record_camera
        self.image_transport = image_transport
        self.freshness, self.shutdown_timeout = freshness, shutdown_timeout
        self.lock = threading.RLock()
        self.session = RecordingSession()
        self.state = 'idle'
        self.current = {}
        saved = self.catalog.list(limit=1)[0]
        self.last_completed = dict(saved[0]) if saved else {}
        self.last_error = ''
        self._temporary = None
        self._current_request_id = None
        self._save_cancel_requested = False
        self._image_spool_error = ''
        self._seen = {}
        self._cancelled = set()
        self._counts = {}
        self._reset_counts()
        self._queue = queue.Queue()
        self._queued_images = 0
        self._closing = False
        self._closed = False
        self._worker = threading.Thread(target=self._work, name='fastumi-recorder-writer', daemon=True)
        self._worker.start()

    def _reset_counts(self):
        """重置本轮采样与图像计数。"""
        self._counts = dict(joint_samples=0, action_samples=0, gripper_samples=0,
                            tracker_samples=0, received_frames=0, encoded_frames=0,
                            saved_frames=0, dropped_frames=0)

    def _check_id(self, recording_id):
        """验证请求指向当前录制条目。"""
        if not recording_id or recording_id != self.current.get('recording_id'):
            raise RecordingError('NOT_FOUND', '录制 ID 不是当前条目')

    def start(self, dir_name='', name='', timestamp=None, request_id=None):
        """就绪检查及编号分配完成后才进入 recording。"""
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
            try:
                if self._closing or self.state in ('recording', 'saving'):
                    raise RecordingError('BUSY', '录制器正在录制、保存或退出')
                required = ['joint', 'gripper'] + (['image'] if self.record_camera else [])
                now = time.monotonic()
                missing = [stream for stream in required
                           if stream not in self._seen or now - self._seen[stream] > self.freshness]
                if missing:
                    raise RecordingError('NOT_READY', '缺少最近有效输入: ' + ', '.join(missing))
                stamp = time.time() if timestamp is None else timestamp
                info, temporary = self.catalog.allocate(dir_name or self.dir_name, name or self.name, stamp)
                self.requests.put(request_id, state='recording', recording_id=info['recording_id'],
                                  start_success=1, start_code='STARTED', start_message='STARTED',
                                  code='STARTED', message='录制已启动')
                self.session.command('a', stamp, temporary)
                self.current, self._temporary = info, temporary
                self._current_request_id = request_id
                self.encoder.reset_cancel()
                self._reset_counts()
                self._image_spool_error = ''
                self.last_error = ''
                self.state = 'recording'
                return info['recording_id'], 'STARTED'
            except RecordingError as error:
                self.requests.put(request_id, state='failed', start_success=0,
                                  start_code=error.code, start_message=str(error),
                                  code=error.code, message=str(error))
                raise
            except (OSError, ValueError) as error:
                self.requests.put(request_id, state='failed', start_success=0,
                                  start_code='IO_ERROR', start_message=str(error),
                                  code='IO_ERROR', message=str(error))
                raise

    def stop(self, recording_id, timestamp=None):
        """停止只排入保存屏障；不等待视频编码结束。"""
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
            _, episode, generation = self.session.command('stop', stamp)
            self.current.update(stopped_at=stamp, duration=stamp - self.current['started_at'])
            self.state = 'saving'
            self._queue.put(('save', generation, episode, dict(self.current), self._temporary))
            return recording_id, 'STOP_ACCEPTED'

    def cancel(self, recording_id):
        """取消仅适用于尚未保存的本轮，旧图像由代次门控丢弃。"""
        with self.lock:
            if recording_id in self._cancelled:
                return recording_id, 'ALREADY_CANCELLED'
            self._check_id(recording_id)
            if self.state == 'saving':
                raise RecordingError('BUSY', '保存阶段不能取消')
            if self.state != 'recording':
                raise RecordingError('NOT_RECORDING', '当前条目没有在录制')
            if self._current_request_id:
                self.requests.put(self._current_request_id, cancel_requested=1,
                                  code='CANCEL_ACCEPTED', message='正在撤销录制')
            self.session.command('b', time.time())
            self.state = 'idle'
            try:
                shutil.rmtree(self._temporary)
            except OSError as error:
                if self._current_request_id:
                    self.requests.put(self._current_request_id, state='failed',
                                      code='IO_ERROR', message=str(error))
                raise
            if self._current_request_id:
                self.requests.put(self._current_request_id, state='cancelled',
                                  code='CANCELLED', message='录制已撤销')
            self._cancelled.add(recording_id)
            return recording_id, 'CANCELLED'

    def cancel_request(self, request_id):
        """Tombstone unknown starts or cancel an active recording by request identity."""
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
            if self.state == 'recording':
                self.requests.put(request_id, state='cancelled', code='CANCELLED',
                                  message='录制已撤销', cancel_requested=1)
                self.session.command('b', time.time())
                self._cancelled.add(row['recording_id'])
                self.state = 'idle'
                try:
                    shutil.rmtree(self._temporary)
                except OSError as error:
                    self.requests.put(request_id, state='failed', code='IO_ERROR', message=str(error))
                    raise
                return row['recording_id'], 'CANCELLED', 'cancelled'
            if self.state == 'saving':
                self.requests.put(request_id, code='CANCEL_ACCEPTED',
                                  message='保存中止已请求', cancel_requested=1)
                self._save_cancel_requested = True
                self.encoder.abort_current()
                return row['recording_id'], 'CANCEL_ACCEPTED', 'saving'
            raise RecordingError('NOT_RECORDING', '请求没有活动录制')

    def get_request(self, request_id):
        """Return the durable request result and current state."""
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
        """在事务锁内回收已完成条目。"""
        with self.lock:
            if recording_id == self.current.get('recording_id') and self.state in ('recording', 'saving'):
                raise RecordingError('BUSY', '不能删除正在录制或保存的条目')
            return recording_id, self.catalog.delete(recording_id)

    def list(self, **kwargs):
        """返回任务过滤后的录制摘要分页。"""
        with self.lock:
            return self.catalog.list(**kwargs)

    def add(self, stream, timestamp, value, *, orientation=None, frame_id=''):
        """有效样本刷新本机接收新鲜度，消息时间用于数据时间轴。"""
        with self.lock:
            aliases = {'joint_state': 'joint', 'gripper_state': 'gripper', 'tracker_pose': 'tracker'}
            if stream in aliases:
                self._seen[aliases[stream]] = time.monotonic()
            if stream == 'gripper_action':
                self.session.add_gripper_action(value)
            elif stream == 'gripper_state':
                self.session.add_gripper_state(timestamp, value)
            elif stream == 'tracker_pose':
                self.session.add_tracker(timestamp, value, orientation, frame_id)
            else:
                self.session.add(stream, timestamp, value)
            if self.state == 'recording':
                key = {'joint_state': 'joint_samples', 'joint_action': 'action_samples',
                       'gripper_state': 'gripper_samples', 'tracker_pose': 'tracker_samples'}.get(stream)
                if key:
                    self._counts[key] += 1

    def image(self, timestamp, message):
        """最多排队八帧；由节点预先校验消息结构。"""
        with self.lock:
            self._seen['image'] = time.monotonic()
            accepting, generation = self.session.accepts_image()
            if not accepting or not self.record_camera:
                return
            self._counts['received_frames'] += 1
            if self._queued_images >= 8:
                self._counts['dropped_frames'] += 1
                return
            self._queued_images += 1
            self._queue.put(('image', generation, timestamp, message))

    def status(self, timestamp=None):
        """返回线程安全状态快照，时间和输入 age 单位为秒。"""
        with self.lock:
            now = time.monotonic()
            duration = self.current.get('duration', 0.0)
            if self.state == 'recording':
                duration = max(0.0, (time.time() if timestamp is None else timestamp) - self.current['started_at'])
            return dict(state=self.state, recording_id=self.current.get('recording_id', ''),
                        current=deepcopy(self.current), last_completed=deepcopy(self.last_completed),
                        duration=duration, last_error=self.last_error, **self._counts,
                        **{f'{s}_age': now - self._seen[s] if s in self._seen else -1.0
                           for s in ('joint', 'gripper', 'image', 'tracker')})

    def _work(self):
        """按队列顺序编码图像并执行保存屏障。"""
        while True:
            task = self._queue.get()
            if task[0] == 'shutdown':
                return
            if task[0] == 'image':
                _, generation, timestamp, message = task
                try:
                    if self.image_transport == 'ffmpeg':
                        data, keyframe = ffmpeg_packet_to_h264(message)
                        codec = 'h264'
                        dimensions = None
                    elif self.image_transport == 'raw':
                        data, dimensions = image_to_bgr24(message)
                        codec, keyframe = 'bgr24', True
                    else:
                        data = image_to_jpeg(message)
                        codec, keyframe = 'mjpeg', True
                        dimensions = None
                    with self.lock:
                        if (generation == self.session.generation and
                                self.state in ('recording', 'saving') and
                                not self._image_spool_error):
                            try:
                                self.session.append_image(
                                    generation, timestamp, data, codec=codec,
                                    keyframe=keyframe, dimensions=dimensions)
                            except OSError as error:
                                if codec == 'bgr24':
                                    self._image_spool_error = f'raw 临时缓存写入失败: {error}'
                                    self.last_error = self._image_spool_error
                                raise
                            self._counts['encoded_frames'] += 1
                except Exception:
                    with self.lock:
                        if generation == self.session.generation:
                            self._counts['dropped_frames'] += 1
                finally:
                    with self.lock:
                        self._queued_images -= 1
                continue
            _, generation, episode, info, temporary = task
            try:
                if self._image_spool_error:
                    raise RuntimeError(self._image_spool_error)
                stats = write_episode(temporary, episode, self.fps, self.encoder)
                info.update(stats)
                info['size_bytes'] = sum(p.stat().st_size for p in temporary.iterdir() if p.is_file())
                with self.lock:
                    if self.encoder._aborted:
                        raise RuntimeError('退出等待超时，编码已中止')
                    if self._save_cancel_requested:
                        raise RuntimeError('保存已由请求撤销')
                    self.catalog.publish(info, temporary)
                    if self._current_request_id:
                        try:
                            self.requests.put(self._current_request_id, state='completed',
                                              code='COMPLETED', message='录制已保存')
                        except sqlite3.Error as error:
                            # The catalog commit is authoritative; query and restart reconcile it.
                            self.last_error = f'录制已保存，但请求台账更新失败: {error}'
                    self.last_completed = dict(info)
                    self.current = dict(info)
                    self._counts['saved_frames'] = stats['image_frames']
                    self.session.finish_save()
                    self.state = 'idle'
            except Exception as error:
                with self.lock:
                    if self._save_cancel_requested:
                        try:
                            shutil.rmtree(temporary)
                            self.requests.put(self._current_request_id, state='cancelled',
                                              code='CANCELLED', message='保存已中止并丢弃')
                            self.state = 'idle'
                        except Exception as cleanup_error:
                            self.last_error = str(cleanup_error)
                            try:
                                self.requests.put(self._current_request_id, state='failed',
                                                  code='IO_ERROR', message=self.last_error)
                            except sqlite3.Error as ledger_error:
                                self.last_error += f'；台账更新失败: {ledger_error}'
                            self.state = 'error'
                    else:
                        self.last_error = f'{error}；未完成目录: {temporary}'
                        self.state = 'error'
                        if self._current_request_id:
                            try:
                                self.requests.put(self._current_request_id, state='failed',
                                                  code='SAVE_FAILED', message=self.last_error)
                            except sqlite3.Error as ledger_error:
                                self.last_error += f'；台账更新失败: {ledger_error}'
                        try:
                            atomic_json(temporary / 'error.json', {'error': str(error), 'recording_id': info['recording_id']})
                        except OSError:
                            pass
                    self.session.finish_save()
                    self._save_cancel_requested = False
            finally:
                episode.close()

    def close(self, timestamp=None):
        """正常退出自动保存；超时中止 FFmpeg 并保留临时现场。"""
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
            self.encoder.abort()
            self._worker.join(5.0)
            with self.lock:
                self.last_error = '退出保存超时；未完成数据保留在隐藏临时目录'
                self.state = 'error'
        if not self._worker.is_alive():
            self.session.close()
            self.requests.close()
            self.catalog.close()
        return completed
