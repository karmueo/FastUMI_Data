"""提供 DP 仿真推理侧的 ROS 2 观测接收与动作发布能力。

该文件订阅仿真侧发布的 RGB 图像、关节状态和末端位姿 topic, 调用 DP
推理函数生成动作, 再通过 `/sensor_processing_dp` 发布给仿真控制器。
"""

_ROS_IMPORT_ERROR = None
"""ROS 2 Python 依赖导入失败时的原始异常。"""

try:
    import rclpy
except (ModuleNotFoundError, ImportError) as exc:
    rclpy = None
    _ROS_IMPORT_ERROR = exc

try:
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float64MultiArray, String
except (ModuleNotFoundError, ImportError) as exc:
    PoseStamped = None
    Image = None
    JointState = None
    Float64MultiArray = None
    String = None
    if _ROS_IMPORT_ERROR is None:
        _ROS_IMPORT_ERROR = exc

JOINT_ROS_TOPIC_NAME = "/franka_joint_pos"
"""关节状态订阅 topic 名称。"""

EE_POSE_ROS_TOPIC_NAME = "/franka_ee_pose"
"""末端位姿订阅 topic 名称。"""

RGB_ROS_TOPIC_NAME = "/camera_rgb"
"""腕部 RGB 图像订阅 topic 名称。"""

ACTION_PUBLISH_NAME = "/sensor_processing_dp"
"""DP 动作发布 topic 名称。"""

COMMAND_TOPIC_NAME = "/umi_task_command"
"""任务启停命令订阅 topic 名称。"""


def _ensure_ros2_available():
    """检查 ROS 2 Python 依赖是否可用。

    Raises:
        RuntimeError: 未 source ROS 2 环境或缺少消息包时抛出。
    """
    if rclpy is None or any(
        msg_type is None
        for msg_type in [PoseStamped, Image, JointState, Float64MultiArray, String]
    ):
        error_detail = f" Original import error: {_ROS_IMPORT_ERROR}" if _ROS_IMPORT_ERROR else ""
        raise RuntimeError(
            "ROS 2 Python packages are not available. Please run "
            "`source /opt/ros/jazzy/setup.bash` before launching DP inference."
            f"{error_detail}"
        )


def _stamp_to_seconds(msg):
    """读取 ROS 消息时间戳并转换为秒。

    Args:
        msg: 带 `header.stamp` 字段的 ROS 消息。

    Returns:
        float | None: 时间戳秒数; 消息没有时间戳时返回 None。
    """
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class RgbJointEePoseActionNode:
    """订阅 RGB、关节和末端位姿, 并发布 DP 推理动作的 ROS 2 节点。"""

    def __init__(
        self,
        process_fn,
        queue_size=10,
        sync_slop=0.02,
        command_topic_name=COMMAND_TOPIC_NAME,
        start_paused=True,
        on_command=None,
        spin=True,
    ):
        """初始化订阅器、发布器和同步缓存。

        Args:
            process_fn: 接收 `(rgb_msg, joint_msg, ee_pose_msg)` 并返回动作数组的函数。
            queue_size: ROS 2 publisher/subscription 队列深度。
            sync_slop: 三路消息允许的最大时间戳差值, 单位秒。
            command_topic_name: 接收 `c`/`s` 任务命令的 topic 名称。
            start_paused: 是否在收到 `c` 前暂停推理。
            on_command: 收到有效任务命令后的可选回调。
            spin: 是否在初始化后进入 `rclpy.spin` 阻塞循环。
        """
        _ensure_ros2_available()

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node("rgb_joint_ee_pose_action_node")
        self.process_fn = process_fn
        self.sync_slop = sync_slop
        self.latest_rgb_msg = None
        self.latest_joint_msg = None
        self.latest_ee_pose_msg = None
        self.command_topic_name = command_topic_name
        self.paused = start_paused
        self.on_command = on_command

        self.rgb_sub = self.node.create_subscription(
            Image,
            RGB_ROS_TOPIC_NAME,
            self.rgb_callback,
            queue_size,
        )
        self.joint_sub = self.node.create_subscription(
            JointState,
            JOINT_ROS_TOPIC_NAME,
            self.joint_callback,
            queue_size,
        )
        self.ee_pose_sub = self.node.create_subscription(
            PoseStamped,
            EE_POSE_ROS_TOPIC_NAME,
            self.ee_pose_callback,
            queue_size,
        )
        self.command_sub = self.node.create_subscription(
            String,
            command_topic_name,
            self.command_callback,
            queue_size,
        )
        self.action_pub = self.node.create_publisher(
            Float64MultiArray,
            ACTION_PUBLISH_NAME,
            queue_size,
        )

        self.node.get_logger().info("RgbJointEePoseActionNode started.")
        if spin:
            rclpy.spin(self.node)

    def rgb_callback(self, rgb_msg):
        """缓存 RGB 图像消息并尝试触发同步推理。

        Args:
            rgb_msg: sensor_msgs/Image 消息。
        """
        if self.paused:
            return
        self.latest_rgb_msg = rgb_msg
        self._try_process_synced_messages()

    def joint_callback(self, joint_msg):
        """缓存关节状态消息并尝试触发同步推理。

        Args:
            joint_msg: sensor_msgs/JointState 消息。
        """
        if self.paused:
            return
        self.latest_joint_msg = joint_msg
        self._try_process_synced_messages()

    def ee_pose_callback(self, ee_pose_msg):
        """缓存末端位姿消息并尝试触发同步推理。

        Args:
            ee_pose_msg: geometry_msgs/PoseStamped 消息。
        """
        if self.paused:
            return
        self.latest_ee_pose_msg = ee_pose_msg
        self._try_process_synced_messages()

    def command_callback(self, command_msg):
        """处理任务启停命令。

        Args:
            command_msg: std_msgs/String 消息, data 为 `c` 时继续, 为 `s` 时暂停。
        """
        command = str(getattr(command_msg, "data", "")).strip().lower()
        if command not in {"c", "s"}:
            return

        if command == "c":
            self.paused = False
        elif command == "s":
            self.paused = True
            self._clear_cached_observations()

        if self.on_command is not None:
            self.on_command(command)

    def _clear_cached_observations(self):
        """清空尚未同步处理的观测缓存。"""
        self.latest_rgb_msg = None
        self.latest_joint_msg = None
        self.latest_ee_pose_msg = None

    def _try_process_synced_messages(self):
        """当三路观测都已就绪且时间戳接近时执行推理。"""
        if self.paused:
            return

        if (
            self.latest_rgb_msg is None
            or self.latest_joint_msg is None
            or self.latest_ee_pose_msg is None
        ):
            return

        stamps = [
            stamp
            for stamp in [
                _stamp_to_seconds(self.latest_rgb_msg),
                _stamp_to_seconds(self.latest_joint_msg),
                _stamp_to_seconds(self.latest_ee_pose_msg),
            ]
            if stamp is not None
        ]
        if stamps and max(stamps) - min(stamps) > self.sync_slop:
            return

        rgb_msg = self.latest_rgb_msg
        joint_msg = self.latest_joint_msg
        ee_pose_msg = self.latest_ee_pose_msg
        self.latest_rgb_msg = None
        self.latest_joint_msg = None
        self.latest_ee_pose_msg = None
        self.synced_callback(rgb_msg, joint_msg, ee_pose_msg)

    def synced_callback(self, rgb_msg, joint_msg, ee_pose_msg):
        """对同步观测执行 DP 推理并发布动作。

        Args:
            rgb_msg: 同步后的 RGB 图像消息。
            joint_msg: 同步后的关节状态消息。
            ee_pose_msg: 同步后的末端位姿消息。
        """
        self.node.get_logger().info("Received RGB, Joint, and EE Pose data.")
        action_data = self.process_fn(rgb_msg, joint_msg, ee_pose_msg)
        if action_data is None:
            return
        if hasattr(action_data, "tolist"):
            action_data = action_data.tolist()
        self.action_pub.publish(Float64MultiArray(data=action_data))
