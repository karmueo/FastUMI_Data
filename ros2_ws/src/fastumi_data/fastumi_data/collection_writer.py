"""由单一写入线程消费有界队列，把序列化消息写入 rosbag2 MCAP。

订阅回调只调用 :meth:`QueuedBagWriter.offer` 入队；MCAP 的打开、写入和关闭
都发生在同一个后台线程。队列溢出、写入异常和关闭异常都会记录为明确的
错误码，调用方据此把整条采集标记为失败，禁止保存为成功记录。
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import threading
from typing import Callable, Deque, Dict, Optional, Protocol, Sequence, Tuple


# 队列溢出错误码。
ERROR_QUEUE_OVERFLOW = "QUEUE_OVERFLOW"
# MCAP 打开、写入或关闭失败错误码。
ERROR_WRITE_FAILED = "WRITE_FAILED"
# 排空队列超时错误码。
ERROR_FINISH_TIMEOUT = "FINISH_TIMEOUT"


@dataclass(frozen=True)
class TopicSpec:
    """描述写入 MCAP 的一个话题。"""

    name: str
    type_name: str
    """ROS 2 消息类型，例如 ``sensor_msgs/msg/Image``。"""


class BagBackend(Protocol):
    """MCAP 写入后端；生产实现使用 rosbag2_py，测试可注入替身。"""

    def open(self, uri: str, topics: Sequence[TopicSpec]) -> None:
        """创建 bag 并注册全部话题。"""

    def write(self, topic: str, data: bytes, timestamp_ns: int) -> None:
        """写入一条已序列化消息。"""

    def close(self) -> None:
        """写出索引、元数据并关闭 bag。"""


class RosbagMcapBackend:
    """使用 ``rosbag2_py.SequentialWriter`` 写入 zstd_fast 压缩的 MCAP。"""

    def __init__(self) -> None:
        """延迟导入 rosbag2_py，使纯逻辑测试无需 ROS 运行时。"""
        self._writer = None

    def open(self, uri: str, topics: Sequence[TopicSpec]) -> None:
        """打开 MCAP 存储并按顺序注册话题。"""
        import rosbag2_py

        self._writer = rosbag2_py.SequentialWriter()
        self._writer.open(
            rosbag2_py.StorageOptions(
                uri=uri,
                storage_id="mcap",
                storage_preset_profile="zstd_fast",
            ),
            rosbag2_py.ConverterOptions("", ""),
        )
        for index, topic in enumerate(topics):
            self._writer.create_topic(
                rosbag2_py.TopicMetadata(
                    id=index,
                    name=topic.name,
                    type=topic.type_name,
                    serialization_format="cdr",
                )
            )

    def write(self, topic: str, data: bytes, timestamp_ns: int) -> None:
        """写入一条消息，时间戳为 MCAP 日志时间。"""
        self._writer.write(topic, data, timestamp_ns)

    def close(self) -> None:
        """关闭写入器，使 MCAP 摘要和 ``metadata.yaml`` 落盘。"""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


@dataclass
class WriterResult:
    """写入线程结束后的汇总。"""

    message_counts: Dict[str, int] = field(default_factory=dict)
    """各话题实际写入的消息数量。"""
    error_code: str = ""
    """空字符串表示全部成功。"""
    error_message: str = ""
    thread_stopped: bool = True
    """写入线程已退出时为真；只有此时才可清理 bag 目录。"""

    @property
    def ok(self) -> bool:
        """写入、排空和关闭均成功时为 ``True``。"""
        return self.thread_stopped and not self.error_code


_Item = Tuple[str, bytes, int]
"""队列元素：话题、序列化字节、MCAP 日志时间纳秒。"""


class QueuedBagWriter:
    """线程安全的有界队列加单写入线程。"""

    def __init__(
        self,
        backend: BagBackend,
        max_messages: int,
        max_bytes: int,
        on_error: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """配置队列上限和故障回调。

        Args:
            backend: MCAP 写入后端。
            max_messages: 队列最多容纳的消息数。
            max_bytes: 队列最多容纳的字节数；空队列时允许单条超限消息。
            on_error: 首次故障时调用 ``(code, message)``，在锁外执行。
        """
        if max_messages <= 0 or max_bytes <= 0:
            raise ValueError("队列容量必须为正数")
        self._backend = backend
        self._max_messages = max_messages
        self._max_bytes = max_bytes
        self._on_error = on_error
        self._condition = threading.Condition()
        self._queue: Deque[_Item] = deque()
        self._queued_bytes = 0
        self._accepting = False
        self._closing = False
        self._thread: Optional[threading.Thread] = None
        self._error_code = ""
        self._error_message = ""
        self._counts: Counter = Counter()

    @property
    def capacity(self) -> int:
        """队列消息数量上限。"""
        return self._max_messages

    @property
    def depth(self) -> int:
        """当前排队的消息数量。"""
        with self._condition:
            return len(self._queue)

    @property
    def error(self) -> Tuple[str, str]:
        """已记录的首个故障 ``(code, message)``，无故障为两个空串。"""
        with self._condition:
            return self._error_code, self._error_message

    def start(self, uri: str, topics: Sequence[TopicSpec]) -> None:
        """在后台线程中打开 bag，并开始接受消息。

        打开失败时在调用线程抛出，不留下线程。
        """
        if self._thread is not None:
            raise RuntimeError("写入器已经启动")
        self._backend.open(uri, topics)
        with self._condition:
            self._accepting = True
        self._thread = threading.Thread(
            target=self._run, name="fastumi-mcap-writer", daemon=True
        )
        self._thread.start()

    def offer(self, topic: str, data: bytes, timestamp_ns: int) -> bool:
        """尝试入队一条消息。

        Returns:
            已入队返回 ``True``；写入器停止接受或发生故障返回 ``False``。
            队列溢出会同时记录故障并通过 ``on_error`` 通知。
        """
        overflow = False
        with self._condition:
            if not self._accepting or self._error_code:
                return False
            size = len(data)
            full = len(self._queue) >= self._max_messages or (
                self._queue and self._queued_bytes + size > self._max_bytes
            )
            if full:
                overflow = True
            else:
                self._queue.append((topic, data, timestamp_ns))
                self._queued_bytes += size
                self._condition.notify()
        if overflow:
            self._fail(
                ERROR_QUEUE_OVERFLOW,
                f"写入队列已满（{self._max_messages} 条或 "
                f"{self._max_bytes} 字节），话题 {topic} 的消息被丢弃",
            )
            return False
        return True

    def stop_accepting(self) -> None:
        """不再接受新消息；已入队消息仍会写出。"""
        with self._condition:
            self._accepting = False

    def finish(self, timeout_s: float) -> WriterResult:
        """停止接受、排空队列并关闭 bag。

        Args:
            timeout_s: 等待写入线程排空并关闭的最长秒数。

        Returns:
            写入汇总；超时、写入或关闭故障体现在 ``error_code``。
        """
        with self._condition:
            self._accepting = False
            self._closing = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                self._record_error(
                    ERROR_FINISH_TIMEOUT,
                    f"写入线程在 {timeout_s:g} 秒内未能排空队列",
                )
        return self._result()

    def abort(self, timeout_s: float = 10.0) -> WriterResult:
        """丢弃排队消息并等待关闭；超时后需保留目录并稍后重试。

        Args:
            timeout_s: 等待写入线程退出的最长秒数。

        Returns:
            写入汇总；``thread_stopped`` 为假时禁止清理目录。
            已记录的写入故障保留，取消超时记为 ``FINISH_TIMEOUT``。
        """
        with self._condition:
            self._accepting = False
            self._closing = True
            self._queue.clear()
            self._queued_bytes = 0
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                self._record_error(
                    ERROR_FINISH_TIMEOUT,
                    f"写入线程在 {timeout_s:g} 秒内未能取消并退出",
                )
        return self._result()

    def _result(self) -> WriterResult:
        """复制当前计数和故障状态。"""
        with self._condition:
            return WriterResult(
                message_counts=dict(self._counts),
                error_code=self._error_code,
                error_message=self._error_message,
                thread_stopped=(
                    self._thread is None or not self._thread.is_alive()
                ),
            )

    def _record_error(self, code: str, message: str) -> bool:
        """记录首个故障，返回是否为首次。"""
        with self._condition:
            if self._error_code:
                return False
            self._error_code = code
            self._error_message = message
            self._accepting = False
            return True

    def _fail(self, code: str, message: str) -> None:
        """记录故障并在锁外触发回调。"""
        if self._record_error(code, message) and self._on_error is not None:
            self._on_error(code, message)

    def _run(self) -> None:
        """写入线程主循环：逐条写出，直到收到关闭信号并排空。"""
        while True:
            with self._condition:
                while not self._queue and not self._closing:
                    self._condition.wait()
                if not self._queue:
                    break
                topic, data, timestamp_ns = self._queue.popleft()
                self._queued_bytes -= len(data)
            if self._error_code:
                continue  # 已失败：继续排空以释放内存，但不再写入。
            try:
                self._backend.write(topic, data, timestamp_ns)
            except Exception as error:  # 后端异常类型不固定。
                self._fail(ERROR_WRITE_FAILED, f"写入 {topic} 失败: {error}")
                continue
            with self._condition:
                self._counts[topic] += 1
        try:
            self._backend.close()
        except Exception as error:
            self._fail(ERROR_WRITE_FAILED, f"关闭 MCAP 失败: {error}")
