"""启动 UMI 服务化采集后端和带采集面板的 RViz2，并可选启动设备节点。

默认只启动 ``collection_node`` 与 RViz2，设备按已有节点连接。相机、Tracker
和夹爪估计可分别用 ``start_camera``、``start_tracker``、``start_gripper``
按需启动。采集由 ``/fastumi/collection`` 服务或 RViz2 面板开始，不再随
launch 自动录包。
"""

import os
from pathlib import Path
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from fastumi_usb_camera.capture import is_physical_video_device_path


# UMI 相机默认原始图像话题；相机命名空间由它推导。
DEFAULT_IMAGE_TOPIC = "/umi_camera/image_raw"
# 图像话题必须以该后缀结尾，才能推导出 USB 相机节点的命名空间。
IMAGE_TOPIC_SUFFIX = "/image_raw"
# 采集 RViz2 默认固定坐标系，即 vive_tracker 发布位姿所在的 OpenVR 原点。
DEFAULT_FIXED_FRAME = "steamvr_tracking"
# Fast DDS 默认参与者配置的环境变量名。
FASTDDS_PROFILE_ENV = "FASTRTPS_DEFAULT_PROFILES_FILE"
# 旧 record_mcap 参数的迁移提示。
RECORD_MCAP_MIGRATION = (
    "record_mcap 已移除：launch 不再自动连续录包。请在 RViz2 的“数据采集”面板"
    "点击开始，或调用 /fastumi/collection/start 服务；每次采集会独立保存 MCAP。"
    "仍需连续录包时使用 `ros2 run fastumi_data record_session`。"
)


def _flag(context: LaunchContext, name: str) -> bool:
    """读取 true/false 启动参数，其他取值视为错误。"""
    value = LaunchConfiguration(name).perform(context).strip().lower()
    if value not in ("true", "false"):
        raise ValueError(f"{name} 只能是 true 或 false，当前为 {value!r}")
    return value == "true"


def _text(context: LaunchContext, name: str) -> str:
    """读取字符串启动参数并去除首尾空白。"""
    return LaunchConfiguration(name).perform(context).strip()


def _camera_namespace(image_topic: str) -> str:
    """从 ``/<namespace>/image_raw`` 推导 USB 相机命名空间。"""
    if not image_topic.startswith("/") or not image_topic.endswith(IMAGE_TOPIC_SUFFIX):
        raise ValueError(
            f"启动相机时 image_topic 必须形如 /<namespace>{IMAGE_TOPIC_SUFFIX}，"
            f"当前为 {image_topic!r}"
        )
    namespace = image_topic[1 : -len(IMAGE_TOPIC_SUFFIX)]
    if not namespace:
        raise ValueError("启动相机时 image_topic 必须带命名空间")
    return namespace


def _gripper_snapshot_files(
    camera_calibration: str, gripper_calibration: str
) -> List[str]:
    """解析夹爪估计实际使用的内参和夹爪标定文件，供采集快照。

    与 ``gripper_openness.launch.py`` 的默认规则一致：夹爪标定优先使用
    用户主目录下新生成的文件，否则使用包内示例。
    """
    share = Path(get_package_share_directory("gripper_openness")) / "config"
    camera = camera_calibration or str(share / "calib.yaml")
    user_calibration = Path.home() / "fastumi_gripper_calibration.yaml"
    gripper = gripper_calibration or (
        str(user_calibration)
        if user_calibration.is_file()
        else str(share / "calibration.yaml")
    )
    return [camera, gripper]


def _launch_setup(context: LaunchContext) -> list:
    """校验参数并按需展开采集节点、设备子 launch 和 RViz2。"""
    if _text(context, "record_mcap").lower() == "true":
        raise ValueError(RECORD_MCAP_MIGRATION)
    start_camera = _flag(context, "start_camera")
    start_tracker = _flag(context, "start_tracker")
    start_gripper = _flag(context, "start_gripper")
    image_topic = _text(context, "image_topic")
    video_device = _text(context, "video_device")
    camera_calibration = _text(context, "camera_calibration_path")
    gripper_calibration = _text(context, "gripper_calibration_path")

    actions: list = []
    profile = _text(context, "fastdds_profile")
    # 默认 SHM 段装不下 6 MB 的原始图像，会回退到 UDP 并在实机上丢约 7% 的帧。
    # 用户已自行设置该环境变量时不覆盖；本 launch 启动的全部子进程都继承它。
    if profile and FASTDDS_PROFILE_ENV not in os.environ:
        actions.append(SetEnvironmentVariable(FASTDDS_PROFILE_ENV, profile))
    if start_camera:
        if not video_device:
            raise ValueError("start_camera:=true 时必须显式提供 video_device")
        if not is_physical_video_device_path(video_device):
            raise ValueError(
                "video_device 必须使用 /dev/v4l/by-path/*-video-index0 物理端口路径"
            )
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("fastumi_usb_camera"),
                            "launch",
                            "usb_camera.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "namespace": _camera_namespace(image_topic),
                    "video_device": video_device,
                }.items(),
            )
        )
    if start_tracker:
        # 采集面板自带 RViz2，关闭 Tracker 子 launch 自带的那个。
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("vive_tracker"),
                            "launch",
                            "vive_tracker.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "serial": _text(context, "tracker_serial"),
                    "use_rviz": "false",
                }.items(),
            )
        )
    # 本 launch 声明了同名的标定参数（默认空），会遮蔽子 launch 的默认值使估计节点
    # 回退到 ToF 标定，因此始终把按子 launch 规则解析出的实际路径显式传入。
    resolved_camera, resolved_gripper = _gripper_snapshot_files(
        camera_calibration, gripper_calibration
    )
    gripper_arguments = {
        "image_topic": image_topic,
        "publish_debug_image": "true" if _flag(context, "publish_debug_image") else "false",
        "camera_calibration_path": resolved_camera,
        "gripper_calibration_path": resolved_gripper,
    }
    if start_gripper:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("fastumi_gripper_estimator"),
                            "launch",
                            "gripper_openness.launch.py",
                        ]
                    )
                ),
                launch_arguments=gripper_arguments.items(),
            )
        )

    parameters: dict = {
        "dataset_root": _text(context, "dataset_root"),
        "image_topic": image_topic,
    }
    for launch_name, parameter in (
        ("extrinsic_path", "extrinsic_path"),
        ("processing_config", "processing_config"),
    ):
        if _text(context, launch_name):
            parameters[parameter] = _text(context, launch_name)
    if start_gripper or camera_calibration or gripper_calibration:
        snapshots = [resolved_camera, resolved_gripper]
        if len({Path(path).name for path in snapshots}) != len(snapshots):
            raise ValueError(
                "相机内参与夹爪标定文件同名，无法同时写入采集快照: "
                + ", ".join(snapshots)
            )
        parameters["snapshot_files"] = snapshots
    actions.append(
        Node(
            package="fastumi_data",
            executable="collection_node",
            name="collection_node",
            output="screen",
            parameters=[
                str(
                    Path(get_package_share_directory("fastumi_data"))
                    / "config"
                    / "collection.yaml"
                ),
                parameters,
            ],
        )
    )
    # 在此解析而不是用运行时条件：Tracker 子 launch 的 use_rviz:=false 参数会泄漏到
    # 同一上下文，使稍后求值的 IfCondition 误判为关闭。
    if _flag(context, "use_rviz"):
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="fastumi_collection_rviz",
                output="screen",
                arguments=[
                    "-d",
                    _text(context, "rviz_config"),
                    "-f",
                    _text(context, "fixed_frame"),
                ],
            )
        )
    return actions


def generate_launch_description() -> LaunchDescription:
    """创建采集后端、RViz2 和可选设备节点的启动描述。"""
    data_share = FindPackageShare("fastumi_data")
    flag_choices = ["true", "false"]
    arguments = [
        DeclareLaunchArgument(
            "dataset_root",
            default_value="dataset",
            description="数据集根目录；内含 <task>/<UTC>_<UUID>、.staging 和 .trash。",
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value=DEFAULT_IMAGE_TOPIC,
            description="UMI 相机原始 sensor_msgs/Image 话题。",
        ),
        DeclareLaunchArgument(
            "extrinsic_path",
            default_value="",
            description="可选 Tracker 到 TCP 外参；留空时记录标记为未标定。",
        ),
        DeclareLaunchArgument(
            "processing_config",
            default_value="",
            description="可选处理配置模板；留空使用 fastumi_data 默认配置。",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            choices=flag_choices,
            description="是否启动带采集面板的 RViz2。",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution([data_share, "config", "collection.rviz"]),
            description="RViz2 配置文件，默认含采集面板、Image 和 Tracker Pose。",
        ),
        DeclareLaunchArgument(
            "fixed_frame",
            default_value=DEFAULT_FIXED_FRAME,
            description="RViz2 固定坐标系，覆盖配置文件中的值。",
        ),
        DeclareLaunchArgument(
            "start_camera",
            default_value="false",
            choices=flag_choices,
            description="是否启动 USB 相机；启用时必须提供 video_device。",
        ),
        DeclareLaunchArgument(
            "video_device",
            default_value="",
            description="相机 /dev/v4l/by-path/*-video-index0 物理端口路径。",
        ),
        DeclareLaunchArgument(
            "start_tracker",
            default_value="false",
            choices=flag_choices,
            description="是否启动 VIVE Tracker 位姿节点（不含其自带 RViz2）。",
        ),
        DeclareLaunchArgument(
            "tracker_serial",
            default_value="",
            description="可选 Tracker 序列号；留空使用 vive_tracker 配置。",
        ),
        DeclareLaunchArgument(
            "start_gripper",
            default_value="false",
            choices=flag_choices,
            description="是否启动夹爪开合度估计节点。",
        ),
        DeclareLaunchArgument(
            "publish_debug_image",
            default_value="false",
            choices=flag_choices,
            description="是否发布夹爪 ArUco 调试图像。",
        ),
        DeclareLaunchArgument(
            "camera_calibration_path",
            default_value="",
            description="可选相机内参 YAML；留空沿用夹爪估计默认内参。",
        ),
        DeclareLaunchArgument(
            "gripper_calibration_path",
            default_value="",
            description="可选夹爪范围标定 YAML；留空沿用夹爪估计默认标定。",
        ),
        DeclareLaunchArgument(
            "fastdds_profile",
            default_value=PathJoinSubstitution(
                [data_share, "config", "fastdds_large_images.xml"]
            ),
            description=(
                "Fast DDS 配置，放大共享内存段以无丢帧传输 1080p 图像；"
                "留空禁用。仅作用于本 launch 启动的进程，外部设备进程需自行设置 "
                "FASTRTPS_DEFAULT_PROFILES_FILE。"
            ),
        ),
        DeclareLaunchArgument(
            "record_mcap",
            default_value="false",
            description="已移除；为 true 时报迁移提示。",
        ),
    ]
    return LaunchDescription([*arguments, OpaqueFunction(function=_launch_setup)])
