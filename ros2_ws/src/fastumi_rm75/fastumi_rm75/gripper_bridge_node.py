"""把归一化夹爪命令映射为标准 control_msgs/GripperCommand Action。"""

from __future__ import annotations

from typing import Optional

from control_msgs.action import GripperCommand
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Float32
from std_srvs.srv import Trigger

from fastumi_rm75.gripper import openness_to_position, position_to_openness


class GripperBridgeNode(Node):
    """提供 dry-run 和标准 GripperCommand 两种平行夹爪适配。"""

    def __init__(self) -> None:
        """加载端点标定并创建命令、状态和停止接口。"""
        super().__init__("gripper_bridge")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("action_name", "/gripper_controller/gripper_cmd")
        self.declare_parameter("closed_position_m", 0.0)
        self.declare_parameter("open_position_m", 0.07)
        self.declare_parameter("max_effort", 20.0)
        self._dry_run = bool(self.get_parameter("dry_run").value)
        self._closed_position_m = float(
            self.get_parameter("closed_position_m").value
        )
        self._open_position_m = float(
            self.get_parameter("open_position_m").value
        )
        self._max_effort = float(self.get_parameter("max_effort").value)
        # 启动时立即验证标定端点。
        openness_to_position(
            0.0, self._closed_position_m, self._open_position_m
        )
        self._action_client = ActionClient(
            self,
            GripperCommand,
            str(self.get_parameter("action_name").value),
        )
        self._command_subscription = self.create_subscription(
            Float32,
            "/fastumi/gripper/command",
            self._command_callback,
            10,
        )
        self._state_publisher = self.create_publisher(
            Float32, "/fastumi/gripper/commanded_state", 10
        )
        self._feedback_publisher = self.create_publisher(
            Float32, "/fastumi/gripper/state", 10
        )
        self._stop_service = self.create_service(
            Trigger, "/fastumi/gripper/stop", self._stop
        )
        self._goal_handle = None
        self.get_logger().info(
            f"平行夹爪桥已启动，dry_run={self._dry_run}，"
            f"位置范围 {self._closed_position_m:.4f}~"
            f"{self._open_position_m:.4f} m"
        )

    def _command_callback(self, message: Float32) -> None:
        """裁剪并发送归一化夹爪目标。"""
        openness = min(max(float(message.data), 0.0), 1.0)
        commanded = Float32()
        commanded.data = openness
        self._state_publisher.publish(commanded)
        if self._dry_run:
            return
        if not self._action_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().error(
                "GripperCommand action server 不可用",
                throttle_duration_sec=1.0,
            )
            return
        goal = GripperCommand.Goal()
        goal.command.position = openness_to_position(
            openness, self._closed_position_m, self._open_position_m
        )
        goal.command.max_effort = self._max_effort
        future = self._action_client.send_goal_async(
            goal, feedback_callback=self._feedback
        )
        future.add_done_callback(self._goal_response)

    def _feedback(self, feedback_message) -> None:
        """把标准 Action 的实际位置反馈转换为归一化开度。"""
        try:
            openness = position_to_openness(
                float(feedback_message.feedback.position),
                self._closed_position_m,
                self._open_position_m,
            )
        except ValueError as error:
            self.get_logger().error(
                f"忽略无效夹爪位置反馈: {error}",
                throttle_duration_sec=1.0,
            )
            return
        message = Float32()
        message.data = openness
        self._feedback_publisher.publish(message)

    def _goal_response(self, future) -> None:
        """保存最新被 action server 接受的目标句柄。"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("夹爪目标被 action server 拒绝")
            return
        self._goal_handle = goal_handle

    def _stop(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """取消当前标准夹爪目标。"""
        del request
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None
        response.success = True
        response.message = "夹爪动作已取消"
        return response


def main(args: Optional[list[str]] = None) -> None:
    """运行归一化平行夹爪桥节点。"""
    rclpy.init(args=args)
    node = GripperBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            # ros2 launch 与外层进程管理器可能同时转发 SIGINT。
            pass


if __name__ == "__main__":
    main()
