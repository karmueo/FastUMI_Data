"""验证鱼眼相机与 Tracker 一键启动 launch 的结构和默认参数。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from xml.etree import ElementTree

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.utilities import perform_substitutions


# 当前测试文件对应的 fastumi_data 包根目录。
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# 待验证的外参采集一键启动文件。
LAUNCH_FILE = PACKAGE_ROOT / "launch" / "tracker_camera.launch.py"


def _load_launch_module():
    """加载文件名包含点号的 Python launch 模块。

    Returns:
        已执行并可生成启动描述的 Python 模块。
    """
    # 独立模块规格，避免将 launch 文件当作普通包导入。
    module_spec = spec_from_file_location("tracker_camera_launch", LAUNCH_FILE)
    assert module_spec is not None
    assert module_spec.loader is not None
    # 根据模块规格创建并执行待测 launch 模块。
    launch_module = module_from_spec(module_spec)
    module_spec.loader.exec_module(launch_module)
    return launch_module


def test_launch_includes_two_exclusive_cameras_and_tracker() -> None:
    """验证入口包含两路互斥相机和一路 Tracker 子 launch。"""
    # 生成待检查的启动描述。
    launch_description = _load_launch_module().generate_launch_description()
    # 顶层包含的三个子 launch。
    included_launches = [
        entity
        for entity in launch_description.entities
        if isinstance(entity, IncludeLaunchDescription)
    ]

    assert len(included_launches) == 3
    # ToF、XV 和 Tracker 子 launch 按固定顺序组装。
    tof_launch, xv_launch, tracker_launch = included_launches
    assert "tof_stereo_camera.launch.py" in str(
        tof_launch.launch_description_source.location
    )
    assert "xv_sdk_node_launch.py" in str(
        xv_launch.launch_description_source.location
    )
    assert "vive_tracker.launch.py" in str(
        tracker_launch.launch_description_source.location
    )
    assert tof_launch.condition is not None
    assert xv_launch.condition is not None
    assert tracker_launch.condition is None

    # ToF 模式只满足 ToF 子 launch 条件。
    launch_context = LaunchContext()
    launch_context.launch_configurations["camera_type"] = "tof"
    assert tof_launch.condition.evaluate(launch_context) is True
    assert xv_launch.condition.evaluate(launch_context) is False
    # XV 模式只满足 XV 子 launch 条件。
    launch_context.launch_configurations["camera_type"] = "xv"
    assert tof_launch.condition.evaluate(launch_context) is False
    assert xv_launch.condition.evaluate(launch_context) is True


def test_launch_defaults_to_tof_and_enables_rviz() -> None:
    """验证默认选择 ToF 相机并开启唯一的 Tracker RViz2。"""
    # 生成待检查的启动描述。
    launch_description = _load_launch_module().generate_launch_description()
    # 按参数名索引全部顶层参数声明。
    arguments = {
        entity.name: entity
        for entity in launch_description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    # 空上下文用于解析纯文本默认参数值。
    launch_context = LaunchContext()

    assert set(arguments) == {
        "camera_type",
        "device_path",
        "stream_profile",
        "sensor_qos_reliability",
        "rgb_fisheye_undistort_enable",
        "tracker_serial",
        "use_rviz",
    }
    assert arguments["camera_type"].choices == ["tof", "xv"]
    assert perform_substitutions(
        launch_context, arguments["camera_type"].default_value
    ) == "tof"
    assert perform_substitutions(
        launch_context, arguments["use_rviz"].default_value
    ) == "true"
    assert arguments["stream_profile"].choices == ["main", "sub"]
    assert perform_substitutions(
        launch_context, arguments["stream_profile"].default_value
    ) == "main"
    assert arguments["sensor_qos_reliability"].choices == [
        "best_effort",
        "reliable",
    ]
    assert perform_substitutions(
        launch_context,
        arguments["sensor_qos_reliability"].default_value,
    ) == "reliable"

    # 两个相机子 launch 均关闭自身附带的录制或 RViz 功能。
    included_launches = [
        entity
        for entity in launch_description.entities
        if isinstance(entity, IncludeLaunchDescription)
    ]
    tof_arguments = dict(included_launches[0].launch_arguments)
    xv_arguments = dict(included_launches[1].launch_arguments)
    tracker_arguments = dict(included_launches[2].launch_arguments)
    assert tof_arguments["enable_rviz"] == "false"
    assert xv_arguments["record_bag"] == "false"

    # ToF 子 launch 接收顶层选择的主码流或子码流档位。
    launch_context.launch_configurations["stream_profile"] = "sub"
    assert perform_substitutions(
        launch_context, [tof_arguments["stream_profile"]]
    ) == "sub"

    # ToF 子 launch 接收顶层选择的传感器话题可靠性。
    launch_context.launch_configurations[
        "sensor_qos_reliability"
    ] = "reliable"
    assert perform_substitutions(
        launch_context,
        [tof_arguments["sensor_qos_reliability"]],
    ) == "reliable"

    # Tracker 接收顶层默认开启的 RViz 开关。
    launch_context.launch_configurations["use_rviz"] = "true"
    assert perform_substitutions(
        launch_context, [tracker_arguments["use_rviz"]]
    ) == "true"


def test_package_declares_both_camera_runtime_dependencies() -> None:
    """验证一键入口所需的两种相机包均已声明运行时依赖。"""
    # fastumi_data 包清单中的全部运行时依赖名称。
    package_tree = ElementTree.parse(PACKAGE_ROOT / "package.xml")
    runtime_dependencies = {
        element.text for element in package_tree.findall("exec_depend")
    }

    assert "tof_stereo_camera" in runtime_dependencies
    assert "xv_sdk_ros2" in runtime_dependencies
