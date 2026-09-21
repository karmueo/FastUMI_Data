"""与 ROS 无关的录制状态机、后台编码和服务事务。"""

from copy import deepcopy
from io import BytesIO
import queue
import shutil
import threading
import time

from PIL import Image as PillowImage

from fastumi_recorder.catalog import Catalog, RecordingError, atomic_json
from fastumi_recorder.core import RecordingSession
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

    def register(self, process):
        """登记编码进程，已中止时立即终止新进程。"""
        with self._lock:
            self._process = process
            if self._aborted:
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


def image_to_jpeg(message):
    """把 ROS 原始图像转换为 JPEG，或直接返回已校验的压缩 JPEG。"""
    if hasattr(message, 'format') and not hasattr(message, 'encoding'):
        image_format = str(message.format).lower()
        data = bytes(message.data)
        if 'jpeg' not in image_format and 'jpg' not in image_format:
            raise ValueError(f'不支持的压缩图像格式: {message.format}')
        if len(data) < 4 or data[:2] != b'\xff\xd8' or data[-2:] != b'\xff\xd9':
            raise ValueError('JPEG 数据不完整')
        return data
    formats = {'bgr8': ('RGB', 'BGR', 3), 'rgb8': ('RGB', 'RGB', 3),
               'mono8': ('L', 'L', 1)}
    if message.encoding.lower() not in formats:
        raise ValueError(f'不支持的图像编码: {message.encoding}')
    mode, raw_mode, channels = formats[message.encoding.lower()]
    width, height, step = int(message.width), int(message.height), int(message.step)
    if width <= 0 or height <= 0 or step < width * channels:
        raise ValueError('图像尺寸或行跨度无效')
    data = bytes(message.data)
    if len(data) < step * height:
        raise ValueError('图像数据不足')
    image = PillowImage.frombytes(mode, (width, height), data[:step * height], 'raw', raw_mode, step)
    with BytesIO() as output:
        image.save(output, format='JPEG', quality=90)
        return output.getvalue()


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

    def start(self, dir_name='', name='', timestamp=None):
        """就绪检查及编号分配完成后才进入 recording。"""
        with self.lock:
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
            self.session.command('a', stamp)
            self.current, self._temporary = info, temporary
            self._reset_counts()
            self.last_error = ''
            self.state = 'recording'
            return info['recording_id'], 'STARTED'

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
            self.session.command('b', time.time())
            self._cancelled.add(recording_id)
            self.state = 'idle'
            shutil.rmtree(self._temporary)
            return recording_id, 'CANCELLED'

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
                    else:
                        data = image_to_jpeg(message)
                        codec, keyframe = 'mjpeg', True
                    with self.lock:
                        if generation == self.session.generation and self.state in ('recording', 'saving'):
                            self.session.append_image(
                                generation, timestamp, data, codec=codec,
                                keyframe=keyframe)
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
                stats = write_episode(temporary, episode, self.fps, self.encoder)
                info.update(stats)
                info['size_bytes'] = sum(p.stat().st_size for p in temporary.iterdir() if p.is_file())
                with self.lock:
                    if self.encoder._aborted:
                        raise RuntimeError('退出等待超时，编码已中止')
                    self.catalog.publish(info, temporary)
                    self.last_completed = dict(info)
                    self.current = dict(info)
                    self._counts['saved_frames'] = stats['image_frames']
                    self.session.finish_save()
                    self.state = 'idle'
            except Exception as error:
                with self.lock:
                    self.last_error = f'{error}；未完成目录: {temporary}'
                    self.session.finish_save()
                    self.state = 'error'
                    try:
                        atomic_json(temporary / 'error.json', {'error': str(error), 'recording_id': info['recording_id']})
                    except OSError:
                        pass
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
            self.catalog.close()
        return completed
