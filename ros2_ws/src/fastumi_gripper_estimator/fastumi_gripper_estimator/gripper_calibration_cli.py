"""提供从 ROS 2 bag 或实时相机流标定夹爪开合距离的命令行工具。"""

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Callable, List, Optional, Sequence, Tuple

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from fastumi_gripper_estimator.range_calibration import \
    calculate_bag_statistics
from fastumi_gripper_estimator.range_calibration import \
    calculate_live_statistics
from fastumi_gripper_estimator.range_calibration import CalibrationQualityError
from fastumi_gripper_estimator.range_calibration import create_estimator
from fastumi_gripper_estimator.range_calibration import load_gripper_parameters
from fastumi_gripper_estimator.range_calibration import preflight_output_path
from fastumi_gripper_estimator.range_calibration import \
    RangeCalibrationStatistics
from fastumi_gripper_estimator.range_calibration import \
    validate_calibration_quality
from fastumi_gripper_estimator.range_calibration import write_calibrated_config
import yaml


def main(argv: Optional[Sequence[str]] = None) -> int:
    """解析命令行并在成功校验后写出新的完整 ROS 参数 YAML。"""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        preflight_output_path(
            arguments.config, arguments.output, arguments.force
        )
        _, parameters = load_gripper_parameters(arguments.config)
        image_topic = arguments.image_topic or str(parameters["image_topic"])
        camera_path = arguments.camera_calibration or _default_camera_path()
        estimator = create_estimator(parameters, camera_path)
        if arguments.command == "bag":
            statistics = collect_bag_statistics(
                arguments.bag_uri, image_topic, arguments.frame_stride,
                estimator
            )
            validate_calibration_quality(statistics, is_bag=True)
        else:
            statistics = collect_live_statistics(
                image_topic, arguments.countdown, arguments.sample_duration,
                estimator
            )
            validate_calibration_quality(statistics, is_bag=False)
        output_path = write_calibrated_config(
            arguments.config, arguments.output, statistics, arguments.force
        )
    except KeyboardInterrupt:
        print("标定已取消；未生成输出文件。", file=sys.stderr)
        return 130
    except (CalibrationQualityError, RuntimeError, ValueError, OSError,
            cv2.error) as error:
        print(f"标定失败：{error}", file=sys.stderr)
        return 2
    _print_statistics(statistics, output_path)
    return 0


def collect_bag_statistics(
    bag_uri: str,
    image_topic: str,
    frame_stride: int,
    estimator,
) -> RangeCalibrationStatistics:
    """流式读取 bag 的目标 Image 话题并计算离线端点统计。"""
    if frame_stride < 1:
        raise ValueError("--frame-stride 必须至少为 1")
    storage_identifier = _read_storage_identifier(bag_uri)
    if storage_identifier not in ("mcap", "sqlite3"):
        raise ValueError(
            "metadata.yaml storage_identifier 仅支持 mcap 或 sqlite3，"
            f"当前为 {storage_identifier}"
        )
    try:
        from rclpy.serialization import deserialize_message
        import rosbag2_py
        from sensor_msgs.msg import Image
    except ImportError as error:
        raise RuntimeError("缺少 ROS 2 bag 读取依赖") from error
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(Path(bag_uri)),
                                  storage_id=storage_identifier),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topic_types = {entry.name: entry.type
                   for entry in reader.get_all_topics_and_types()}
    if image_topic not in topic_types:
        raise ValueError(f"bag 中不存在目标图像话题: {image_topic}")
    if topic_types[image_topic] != "sensor_msgs/msg/Image":
        raise ValueError(
            f"目标话题类型必须为 sensor_msgs/msg/Image，当前为 "
            f"{topic_types[image_topic]}"
        )
    bridge = CvBridge()
    total_count = 0
    distances: List[Optional[float]] = []
    selected_count = 0
    while reader.has_next():
        topic, serialized_data, _ = reader.read_next()
        if topic != image_topic:
            continue
        if selected_count % frame_stride == 0:
            total_count += 1
            message = deserialize_message(serialized_data, Image)
            distances.append(_estimate_distance(bridge, estimator, message))
        selected_count += 1
    if total_count == 0:
        raise ValueError("目标图像话题没有可处理帧")
    return calculate_bag_statistics(distances, total_count)


def collect_live_statistics(
    image_topic: str,
    countdown: int,
    sample_duration: float,
    estimator,
    input_function: Callable[[str], str] = input,
) -> RangeCalibrationStatistics:
    """按闭合后张开的操作员流程订阅实时图像并计算端点统计。"""
    if countdown < 0:
        raise ValueError("--countdown 不能为负数")
    if not math.isfinite(sample_duration) or sample_duration <= 0.0:
        raise ValueError("--sample-duration 必须为正的有限数")
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy
        from rclpy.qos import QoSProfile
        from rclpy.qos import ReliabilityPolicy
        from sensor_msgs.msg import Image
    except ImportError as error:
        raise RuntimeError("缺少 ROS 2 实时订阅依赖") from error

    class LiveCollector(Node):
        """保存当前采样阶段的逐帧原始毫米距离。"""

        def __init__(self) -> None:
            """创建与估计节点相同 QoS 的图像订阅器。"""
            super().__init__("gripper_openness_calibrator")
            self.bridge = CvBridge()
            self.total_count = 0
            self.distances: List[Optional[float]] = []
            self._subscription = None
            self._qos = QoSProfile(
                depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE
            )

        def start_capture(self) -> None:
            """清空上一阶段样本，并创建当前端点窗口的临时订阅。"""
            self.total_count = 0
            self.distances = []
            self._subscription = self.create_subscription(
                Image, image_topic, self.callback, self._qos
            )

        def callback(self, message: Image) -> None:
            """记录当前阶段的一帧估计结果。"""
            self.total_count += 1
            self.distances.append(_estimate_distance(
                self.bridge, estimator, message
            ))

        def finish_capture(self) -> Tuple[List[Optional[float]], int]:
            """销毁当前窗口订阅并取出独立阶段样本。"""
            captured = (self.distances, self.total_count)
            if self._subscription is not None:
                self.destroy_subscription(self._subscription)
                self._subscription = None
            self.distances = []
            self.total_count = 0
            return captured

    initialized = False
    collector = None
    try:
        rclpy.init()
        initialized = True
        collector = LiveCollector()
        closed_distances, closed_total = _capture_endpoint(
            collector, rclpy, "请将夹爪完全闭合后按 Enter 开始采样", countdown,
            sample_duration, input_function
        )
        open_distances, open_total = _capture_endpoint(
            collector, rclpy, "请将夹爪完全张开后按 Enter 开始采样", countdown,
            sample_duration, input_function
        )
    finally:
        if collector is not None:
            collector.destroy_node()
        is_ok = getattr(rclpy, "ok", lambda: True)
        if initialized and is_ok():
            rclpy.shutdown()
    return calculate_live_statistics(
        closed_distances, closed_total, open_distances, open_total
    )


def _capture_endpoint(
    collector, rclpy_module, prompt: str, countdown: int,
    sample_duration: float, input_function: Callable[[str], str]
) -> Tuple[List[Optional[float]], int]:
    """执行一次 Enter、倒计时和固定时长的实时端点采样。"""
    input_function(f"{prompt}：")
    for seconds_left in range(countdown, 0, -1):
        print(f"{seconds_left}...", flush=True)
        time.sleep(1.0)
    print(f"开始采样 {sample_duration:.1f} 秒。", flush=True)
    collector.start_capture()
    try:
        end_time = time.monotonic() + sample_duration
        while time.monotonic() < end_time:
            rclpy_module.spin_once(collector, timeout_sec=0.1)
    finally:
        captured = collector.finish_capture()
    return captured


def _estimate_distance(bridge: CvBridge, estimator, message) -> Optional[float]:
    """转换 Image 并提取原始有限毫米距离，单帧失败记作无效。"""
    try:
        image = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        estimate = estimator.estimate(image)
    except (ValueError, RuntimeError, cv2.error):
        return None
    if estimate is None:
        return None
    distance = float(estimate.distance_mm)
    return distance if math.isfinite(distance) else None


def _read_storage_identifier(bag_uri: str) -> str:
    """读取并验证 bag 目录 metadata.yaml 中的存储插件标识。"""
    metadata_path = Path(bag_uri).expanduser() / "metadata.yaml"
    if not metadata_path.is_file():
        raise ValueError(f"bag 缺少 metadata.yaml: {metadata_path}")
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream)
        value = metadata["rosbag2_bagfile_information"]["storage_identifier"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        raise ValueError(f"读取 bag metadata.yaml 失败: {error}") from error
    if not isinstance(value, str) or not value:
        raise ValueError("metadata.yaml storage_identifier 必须是非空字符串")
    return value


def _default_config_path() -> str:
    """返回安装后包共享目录中的默认夹爪 ROS 参数 YAML。"""
    return str(Path(get_package_share_directory(
        "fastumi_gripper_estimator"
    )) / "config" / "gripper_openness.yaml")


def _default_camera_path() -> str:
    """返回安装后 ToF 包共享目录中的默认 RGB 相机标定 YAML。"""
    return str(Path(get_package_share_directory("tof_stereo_camera")) /
               "config" / "calibration.yaml")


def _build_parser() -> argparse.ArgumentParser:
    """构建带 bag 与 live 子命令的标定命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="gripper_openness_calibrate",
        description="从 ROS 2 bag 或实时图像流标定夹爪端点标记距离",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=_default_config_path(),
                        help="输入 ROS 参数 YAML")
    common.add_argument("--camera-calibration", default=None,
                        help="相机标定 YAML，默认 ToF 包共享标定")
    common.add_argument("--image-topic", default=None,
                        help="图像话题，默认采用配置中的 image_topic")
    common.add_argument("--output", required=True, help="新标定 YAML 输出路径")
    common.add_argument("--force", action="store_true", help="允许覆盖已有输出")
    bag_parser = subcommands.add_parser("bag", parents=[common],
                                        help="从录制 bag 标定")
    bag_parser.add_argument("bag_uri", help="ROS 2 bag 目录")
    bag_parser.add_argument("--frame-stride", type=int, default=1,
                            help="每隔多少图像帧处理一帧")
    live_parser = subcommands.add_parser("live", parents=[common],
                                         help="从实时相机流标定")
    live_parser.add_argument("--sample-duration", type=float, default=5.0,
                             help="每个端点采样时长，单位秒")
    live_parser.add_argument("--countdown", type=int, default=3,
                             help="每个端点开始前倒计时秒数")
    return parser


def _print_statistics(
    statistics: RangeCalibrationStatistics, output_path: Path
) -> None:
    """按运维可读格式输出质量统计与最终生成路径。"""
    print(f"总帧数: {statistics.total_count}")
    print(f"有效帧数: {statistics.valid_count} "
          f"({statistics.validity_rate:.1%})")
    for name, endpoint in (("闭合", statistics.closed),
                           ("张开", statistics.opened)):
        print(f"{name}: raw={endpoint.raw_count}, "
              f"retained={endpoint.retained_count}, "
              f"MAD={endpoint.mad_mm:.3f} mm, "
              f"robust_sigma={endpoint.robust_sigma_mm:.3f} mm")
    print("估计范围: "
          f"{statistics.closed.estimate_mm:.3f} ~ "
          f"{statistics.opened.estimate_mm:.3f} mm")
    print(f"输出配置: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
