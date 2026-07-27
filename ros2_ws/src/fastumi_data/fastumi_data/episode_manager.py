"""提供手动 episode 启停服务并发布可录制的边界事件。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastumi_interfaces.msg import EpisodeEvent
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger


class EpisodeManager(Node):
    """管理连续 MCAP 会话内唯一活动的 episode。"""

    def __init__(self) -> None:
        """声明参数、事件发布器和手动控制服务。"""
        super().__init__("episode_manager")
        self.declare_parameter("task_name", "test")
        self.declare_parameter("session_id", "")
        # 任务和会话标识会写入每个边界事件。
        self._task_name = str(self.get_parameter("task_name").value).strip()
        configured_session = str(
            self.get_parameter("session_id").value
        ).strip()
        self._session_id = configured_session or datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
        if not self._task_name:
            raise ValueError("task_name 不能为空")

        event_qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._event_publisher = self.create_publisher(
            EpisodeEvent, "/fastumi/episode/events", event_qos
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
        # 当前 episode 索引只在 STOP 或 ABORT 后递增。
        self._episode_index = 0
        self._active = False
        self.get_logger().info(
            f"会话 {self._session_id} 已就绪，任务: {self._task_name}"
        )

    def _publish_event(self, event_type: int) -> None:
        """发布当前 episode 的边界事件。"""
        message = EpisodeEvent()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "system_time"
        message.session_id = self._session_id
        message.task_name = self._task_name
        message.episode_index = self._episode_index
        message.event_type = event_type
        self._event_publisher.publish(message)

    def _start(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """开始一条 episode，拒绝嵌套开始。"""
        del request
        if self._active:
            response.success = False
            response.message = "已有活动 episode，请先 stop 或 abort"
            return response
        self._active = True
        self._publish_event(EpisodeEvent.START)
        response.success = True
        response.message = f"episode {self._episode_index} 已开始"
        self.get_logger().info(response.message)
        return response

    def _stop(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """正常结束活动 episode。"""
        del request
        if not self._active:
            response.success = False
            response.message = "当前没有活动 episode"
            return response
        self._publish_event(EpisodeEvent.STOP)
        response.success = True
        response.message = f"episode {self._episode_index} 已结束"
        self.get_logger().info(response.message)
        self._episode_index += 1
        self._active = False
        return response

    def _abort(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """放弃活动 episode，并保留事件供离线报告说明。"""
        del request
        if not self._active:
            response.success = False
            response.message = "当前没有活动 episode"
            return response
        self._publish_event(EpisodeEvent.ABORT)
        response.success = True
        response.message = f"episode {self._episode_index} 已放弃"
        self.get_logger().warning(response.message)
        self._episode_index += 1
        self._active = False
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
