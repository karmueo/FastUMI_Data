"""提供线程安全的 episode 边界服务，并发布可录制事件。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Callable, Optional, Sequence, TypeVar

from fastumi_interfaces.msg import EpisodeEvent
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger


SnapshotPayload = TypeVar("SnapshotPayload")
"""状态锁内读取回调返回的泛型 payload 类型。"""


@dataclass(frozen=True)
class EpisodeSnapshot:
    """保存冻结时稳定的 episode 状态快照。"""

    active: bool
    """冻结时是否存在尚未结束的 episode。"""
    episode_index: int
    """下一条或当前 episode 的从零开始索引。"""
    revision: int = 0
    """每次成功变更标注状态后递增的单调版本号。"""


class EpisodeManager(Node):
    """管理连续 MCAP 会话内唯一活动的 episode。"""

    def __init__(
        self,
        event_observer: Optional[Callable[[EpisodeEvent], None]] = None,
    ) -> None:
        """声明服务和发布器。

        Args:
            event_observer: 可选同步观察器。在事件发布、状态提交前调用，
                用于回放标注进程缓存序列化字节。
        """
        super().__init__("episode_manager")
        self.declare_parameter("task_name", "test")
        self.declare_parameter("session_id", "")
        self._task_name = str(self.get_parameter("task_name").value).strip()
        configured_session = str(self.get_parameter("session_id").value).strip()
        self._session_id = configured_session or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        if not self._task_name:
            raise ValueError("task_name 不能为空")

        self._event_qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._event_publisher = self.create_publisher(
            EpisodeEvent, "/fastumi/episode/events", self._event_qos
        )
        self._start_service = self.create_service(
            Trigger, "/fastumi/episode/start", self._start
        )
        self._stop_service = self.create_service(
            Trigger, "/fastumi/episode/stop", self._stop
        )
        self._abort_service = self.create_service(
            Trigger, "/fastumi/episode/abort", self._abort
        )
        self._event_observer = event_observer
        self._lock = RLock()
        self._episode_index = 0
        self._active = False
        self._frozen = False
        self._revision = 0
        self.get_logger().info(
            f"会话 {self._session_id} 已就绪，任务: {self._task_name}"
        )

    def freeze(self) -> EpisodeSnapshot:
        """冻结状态转换并返回可供离线合并校验的稳定快照。"""
        with self._lock:
            self._frozen = True
            return EpisodeSnapshot(
                active=self._active,
                episode_index=self._episode_index,
                revision=self._revision,
            )

    def state_snapshot(self) -> EpisodeSnapshot:
        """返回当前 episode 状态，不改变后续转换能力。"""
        with self._lock:
            return EpisodeSnapshot(
                active=self._active,
                episode_index=self._episode_index,
                revision=self._revision,
            )

    def read_consistent_state(
        self,
        on_read: Callable[[], SnapshotPayload],
    ) -> tuple[EpisodeSnapshot, SnapshotPayload]:
        """在状态锁内读取管理器快照和关联权威 payload。

        Args:
            on_read: 在状态锁内执行的只读回调。回调可以取得与当前
                active、index 和 revision 对应的事件缓存摘要。

        Returns:
            同一线性化时刻的管理器快照和回调结果。
        """
        with self._lock:
            snapshot = EpisodeSnapshot(
                active=self._active,
                episode_index=self._episode_index,
                revision=self._revision,
            )
            return snapshot, on_read()

    def try_freeze_idle(self) -> Optional[EpisodeSnapshot]:
        """仅在没有活动 episode 时原子冻结状态转换。

        Returns:
            冻结成功时返回稳定快照；存在活动 episode 时返回 ``None``，
            并保持管理器可继续接收 STOP 或 ABORT。
        """
        with self._lock:
            if self._active:
                return None
            self._frozen = True
            return EpisodeSnapshot(
                active=False,
                episode_index=self._episode_index,
                revision=self._revision,
            )

    def try_clear_annotations(self, on_clear: Callable[[], None]) -> bool:
        """原子清空全部边界、活动状态和索引，并刷新瞬态发布器。

        Args:
            on_clear: 在状态锁内调用的事件缓存清空回调。

        Returns:
            清空成功时返回 ``True``；管理器已冻结或回调失败时返回
            ``False``。
        """
        with self._lock:
            if self._frozen:
                return False
            replacement_publisher = self.create_publisher(
                EpisodeEvent, "/fastumi/episode/events", self._event_qos
            )
            try:
                on_clear()
            except Exception as error:
                self.destroy_publisher(replacement_publisher)
                self.get_logger().error(f"清空 episode 事件失败: {error}")
                return False
            previous_publisher = self._event_publisher
            self._event_publisher = replacement_publisher
            self._episode_index = 0
            self._active = False
            self._revision += 1
            self.destroy_publisher(previous_publisher)
            self.get_logger().info("全部 episode 标记已删除，编号已重置为 0")
            return True

    def try_replace_annotations(
        self,
        on_replace: Callable[
            [], Optional[tuple[Sequence[EpisodeEvent], int]]
        ],
    ) -> Optional[EpisodeSnapshot]:
        """原子替换空闲会话的事件集、下一索引和瞬态发布器。

        Args:
            on_replace: 在状态锁内执行的事件缓存替换回调。成功时返回
                完整事件消息快照和下一 episode 索引；消息快照用于确认
                替换完成，瞬态发布器保持空历史。找不到删除目标时返回
                ``None``。

        Returns:
            替换后的稳定状态；活动 episode、冻结状态、目标不存在或
            回调失败时返回 ``None``。
        """
        with self._lock:
            if self._frozen or self._active:
                return None
            replacement_publisher = self.create_publisher(
                EpisodeEvent, "/fastumi/episode/events", self._event_qos
            )
            try:
                replacement = on_replace()
                if replacement is None:
                    self.destroy_publisher(replacement_publisher)
                    return None
                replacement_events, next_episode_index = replacement
                if next_episode_index < 0:
                    raise ValueError("下一 episode 索引不能为负数")
            except Exception as error:
                self.destroy_publisher(replacement_publisher)
                self.get_logger().error(f"替换 episode 事件失败: {error}")
                return None
            previous_publisher = self._event_publisher
            self._event_publisher = replacement_publisher
            self._episode_index = int(next_episode_index)
            self._revision += 1
            self.destroy_publisher(previous_publisher)
            # 新发布器必须保持空历史，避免 retained 事件被 Panel 当成新标记。
            # 完整历史和当前状态由权威列表服务恢复。
            del replacement_events
            self.get_logger().info(
                f"episode 标记已更新，下一编号为 {self._episode_index}"
            )
            return EpisodeSnapshot(
                active=False,
                episode_index=self._episode_index,
                revision=self._revision,
            )

    def _make_event(self, event_type: int) -> EpisodeEvent:
        """构造包含当前模拟时钟、任务及索引的边界事件。"""
        message = EpisodeEvent()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "system_time"
        message.session_id = self._session_id
        message.task_name = self._task_name
        message.episode_index = self._episode_index
        message.event_type = event_type
        return message

    def _emit_before_commit(self, event_type: int) -> None:
        """在持锁状态下同步观察、发布事件；调用方随后提交状态。"""
        event = self._make_event(event_type)
        if self._event_observer is not None:
            self._event_observer(event)
        self._event_publisher.publish(event)

    def _transition(
        self,
        event_type: int,
        should_be_active: bool,
        accepted_message: str,
        rejected_message: str,
    ) -> tuple[bool, str]:
        """按 observer、发布、提交顺序执行单次合法状态转换。"""
        with self._lock:
            if self._frozen:
                return False, "episode 管理器已冻结，回放标注不再接受事件"
            if self._active != should_be_active:
                return False, rejected_message
            try:
                self._emit_before_commit(event_type)
            except Exception as error:  # 服务失败必须不改变内存状态。
                self.get_logger().error(f"episode 事件缓存或发布失败: {error}")
                return False, f"episode 事件缓存或发布失败: {error}"
            completed_index = self._episode_index
            if event_type == EpisodeEvent.START:
                self._active = True
            else:
                self._episode_index += 1
                self._active = False
            self._revision += 1
            return True, accepted_message.format(index=completed_index)

    def _start(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """开始一条 episode，拒绝嵌套开始或冻结后的请求。"""
        del request
        response.success, response.message = self._transition(
            EpisodeEvent.START,
            False,
            "episode {index} 已开始",
            "已有活动 episode，请先 stop 或 abort",
        )
        if response.success:
            self.get_logger().info(response.message)
        return response

    def _stop(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """正常结束活动 episode；成功时递增索引。"""
        del request
        response.success, response.message = self._transition(
            EpisodeEvent.STOP,
            True,
            "episode {index} 已结束",
            "当前没有活动 episode",
        )
        if response.success:
            self.get_logger().info(response.message)
        return response

    def _abort(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """放弃活动 episode；成功时递增索引并发布 ABORT。"""
        del request
        response.success, response.message = self._transition(
            EpisodeEvent.ABORT,
            True,
            "episode {index} 已放弃",
            "当前没有活动 episode",
        )
        if response.success:
            self.get_logger().warning(response.message)
        return response


def main(args: Optional[list[str]] = None) -> None:
    """运行 episode 管理节点。"""
    rclpy.init(args=args)
    node = EpisodeManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
