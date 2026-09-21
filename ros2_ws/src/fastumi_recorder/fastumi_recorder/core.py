"""管理遥操录制状态、各路时序数据和有界内存的相机缓存。"""

from __future__ import annotations

from dataclasses import dataclass, field
import struct
import tempfile
import threading
from typing import Iterator


JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
CAMERA_NAME = "gripper"
CAMERA_SPOOL_MEMORY_LIMIT = 8 * 1024 * 1024
_FRAME_HEADER = struct.Struct("<dQBB")
_CODEC_IDS = {"mjpeg": 1, "h264": 2}
_CODECS = {value: key for key, value in _CODEC_IDS.items()}


class CameraFrameSpool:
    """按时间顺序缓存压缩视频帧，超过阈值后自动转入临时文件。"""

    def __init__(self, memory_limit: int = CAMERA_SPOOL_MEMORY_LIMIT) -> None:
        """创建可回放的临时帧缓存。"""
        self._file = tempfile.SpooledTemporaryFile(max_size=memory_limit, mode="w+b")
        self._count = 0

    def append(
        self, timestamp: float, data: bytes, *, codec: str = "mjpeg",
        keyframe: bool = True,
    ) -> None:
        """写入一帧时间戳、编解码器、关键帧标志和压缩内容。"""
        if codec not in _CODEC_IDS:
            raise ValueError(f"不支持的视频编码: {codec}")
        self._file.write(
            _FRAME_HEADER.pack(timestamp, len(data), _CODEC_IDS[codec], keyframe)
        )
        self._file.write(data)
        self._count += 1

    def __len__(self) -> int:
        """返回缓存的帧数。"""
        return self._count

    def __iter__(self) -> Iterator[tuple[float, bytes, str, bool]]:
        """从开头回放缓存帧，并恢复原写入位置。"""
        self._file.flush()
        position = self._file.tell()
        self._file.seek(0)
        try:
            for _ in range(self._count):
                header = self._file.read(_FRAME_HEADER.size)
                if len(header) != _FRAME_HEADER.size:
                    raise IOError("相机缓存帧头不完整")
                timestamp, size, codec_id, keyframe = _FRAME_HEADER.unpack(header)
                data = self._file.read(size)
                if len(data) != size:
                    raise IOError("相机缓存视频帧不完整")
                codec = _CODECS.get(codec_id)
                if codec is None:
                    raise IOError(f"相机缓存包含未知编码编号: {codec_id}")
                yield timestamp, data, codec, bool(keyframe)
        finally:
            self._file.seek(position)

    def close(self) -> None:
        """释放临时文件。"""
        self._file.close()


@dataclass
class EpisodeBuffer:
    """保存一轮录制的原始时序数据。"""

    started_at: float = 0.0
    stopped_at: float = 0.0
    joint_state: list[tuple[float, list[float]]] = field(default_factory=list)
    joint_action: list[tuple[float, list[float]]] = field(default_factory=list)
    gripper_state: list[tuple[float, float]] = field(default_factory=list)
    gripper_action: list[tuple[float, float]] = field(default_factory=list)
    tracker_pose: list[tuple[float, list[float], list[float]]] = field(
        default_factory=list
    )
    tracker_frame_id: str = ""
    images: CameraFrameSpool = field(default_factory=CameraFrameSpool)

    def close(self) -> None:
        """释放该轮录制的相机缓存。"""
        self.images.close()


class RecordingSession:
    """在线程间协调开始、取消、停止和保存后的状态。"""

    def __init__(self) -> None:
        """初始化空闲会话及夹爪指令保持值。"""
        self._lock = threading.Lock()
        self.state = "idle"
        self.generation = 0
        self.started_at = 0.0
        self.stopped_at = 0.0
        self._episode: EpisodeBuffer | None = None
        self._saving_episode: EpisodeBuffer | None = None
        self._latest_gripper_action: float | None = None

    def command(
        self, key: str, timestamp: float
    ) -> tuple[str, EpisodeBuffer | None, int]:
        """处理单键命令并返回结果、待保存数据和录制代数。"""
        with self._lock:
            if self.state == "saving":
                return "busy", None, self.generation
            if key == "a" and self.state == "idle":
                self.generation += 1
                self._episode = EpisodeBuffer(started_at=timestamp)
                self.started_at = timestamp
                self.state = "recording"
                return "started", None, self.generation
            if key == "b" and self.state == "recording":
                assert self._episode is not None
                self._episode.close()
                self._episode = None
                self.state = "idle"
                return "discarded", None, self.generation
            if key in ("a", "stop") and self.state == "recording":
                assert self._episode is not None
                if self._episode.joint_action:
                    last_time, last_values = self._episode.joint_action[-1]
                    if timestamp > last_time:
                        self._episode.joint_action.append(
                            (timestamp, last_values.copy())
                        )
                self.stopped_at = timestamp
                self._episode.stopped_at = timestamp
                self._saving_episode = self._episode
                self._episode = None
                self.state = "saving"
                return "stopped", self._saving_episode, self.generation
            return "ignored", None, self.generation

    def snapshot(self) -> str:
        """在线程锁保护下取得空闲、录制或保存状态。"""
        with self._lock:
            return self.state

    def add(self, stream: str, timestamp: float, value) -> None:
        """仅在当前录制窗口追加数值消息。"""
        with self._lock:
            if self.state == "recording" and self._episode is not None:
                getattr(self._episode, stream).append((timestamp, value))

    def add_gripper_action(self, value: float) -> None:
        """更新最新夹爪指令，供反馈消息采样。"""
        with self._lock:
            self._latest_gripper_action = value

    def add_gripper_state(self, timestamp: float, value: float) -> None:
        """按反馈时间戳同步保存状态和保持的最新指令。"""
        with self._lock:
            if self._latest_gripper_action is None:
                self._latest_gripper_action = value
            if self.state == "recording" and self._episode is not None:
                self._episode.gripper_state.append((timestamp, value))
                self._episode.gripper_action.append(
                    (timestamp, self._latest_gripper_action)
                )

    def add_tracker(
        self, timestamp: float, position: list[float],
        orientation: list[float], frame_id: str,
    ) -> None:
        """保存 Tracker 原始位姿及首次收到的坐标系。"""
        with self._lock:
            if self.state == "recording" and self._episode is not None:
                if not self._episode.tracker_frame_id:
                    self._episode.tracker_frame_id = frame_id
                self._episode.tracker_pose.append(
                    (timestamp, position, orientation)
                )

    def accepts_image(self) -> tuple[bool, int]:
        """返回图像回调能否排队以及当前代数。"""
        with self._lock:
            return self.state == "recording", self.generation

    def append_image(
        self, generation: int, timestamp: float, data: bytes, *,
        codec: str = "mjpeg", keyframe: bool = True,
    ) -> None:
        """接受属于本轮录制且位于按键时间范围内的图像。"""
        with self._lock:
            if generation != self.generation:
                return
            episode = (
                self._episode if self.state == "recording"
                else self._saving_episode
            )
            if episode is None or not self.started_at <= timestamp <= (
                self.stopped_at if self.state == "saving" else float("inf")
            ):
                return
            episode.images.append(
                timestamp, data, codec=codec, keyframe=keyframe
            )

    def finish_save(self) -> None:
        """由后台线程完成落盘后恢复空闲状态。"""
        with self._lock:
            self._saving_episode = None
            self.state = "idle"

    def close(self) -> None:
        """退出时取消尚未停止的录制。"""
        with self._lock:
            if self._episode is not None:
                self._episode.close()
                self._episode = None
            if self.state == "recording":
                self.state = "idle"
