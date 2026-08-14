"""验证 FastUMI 设备数据节点统一 launch 的结构和参数。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from xml.etree import ElementTree

from launch import LaunchContext
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.utilities import perform_substitutions


# 当前测试文件对应的 fastumi_data 包根目录。
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# 待验证的统一采集 launch 文件。
LAUNCH_FILE = PACKAGE_ROOT / "launch" / "fastumi_collection.launch.py"


def _load_launch_module():
    """加载文件名包含点号的 Python launch 模块。

    Returns:
        已执行并可调用 `generate_launch_description` 的模块。
    """
    # 独立模块规格，避免将 launch 文件当作普通包导入。
    module_spec = spec_from_file_location(
        "fastumi_collection_launch", LAUNCH_FILE
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    # 根据规格创建待执行的 launch 模块。
    launch_module = module_from_spec(module_spec)
    module_spec.loader.exec_module(launch_module)
    return launch_module


def test_collection_launch_includes_devices_and_optional_recorder() -> None:
    """验证统一入口包含三个设备子 launch 和一个录制进程。"""
    # 生成待检查的统一启动描述。
    launch_description = _load_launch_module().generate_launch_description()
    # 全部顶层 launch action。
    entities = launch_description.entities
    # 三个被复用的子系统 launch。
    included_launches = [
        entity
        for entity in entities
        if isinstance(entity, IncludeLaunchDescription)
    ]
    # 受开关控制的 MCAP 录制进程。
    recorder_processes = [
        entity
        for entity in entities
        if isinstance(entity, ExecuteProcess)
    ]
    assert len(included_launches) == 3
    assert len(recorder_processes) == 1
    assert len(entities) == 10

    camera_launch, _tracker_launch, gripper_launch = included_launches
    assert "tof_stereo_camera.launch.py" in str(
        camera_launch.launch_description_source.location
    )
    camera_arguments = dict(camera_launch.launch_arguments)
    assert set(camera_arguments) == {"device_path", "enable_rviz"}
    assert camera_arguments["enable_rviz"] == "false"
    launch_context = LaunchContext()
    launch_context.launch_configurations["device_path"] = "/dev/video0"
    assert (
        perform_substitutions(
            launch_context, [camera_arguments["device_path"]]
        )
        == "/dev/video0"
    )
    gripper_arguments = dict(gripper_launch.launch_arguments)
    assert gripper_arguments["image_topic"] == (
        "/tof_stereo_camera/rgb/image_raw"
    )


def test_collection_launch_declares_recording_arguments_with_safe_defaults(
) -> None:
    """验证录制默认关闭且数据集根目录可移植。"""
    # 生成待检查的统一启动描述。
    launch_description = _load_launch_module().generate_launch_description()
    # 按参数名索引所有顶层参数声明。
    arguments = {
        entity.name: entity
        for entity in launch_description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }

    assert set(arguments) == {
        "device_path",
        "tracker_serial",
        "use_rviz",
        "publish_debug_image",
        "record_mcap",
        "dataset_root",
    }
    assert "camera_serial" not in arguments
    assert arguments["use_rviz"].choices == ["true", "false"]
    assert arguments["record_mcap"].choices == ["true", "false"]
    # 空启动上下文足以解析两个纯文本默认值。
    launch_context = LaunchContext()
    assert (
        perform_substitutions(
            launch_context, arguments["record_mcap"].default_value
        )
        == "false"
    )
    assert (
        perform_substitutions(
            launch_context, arguments["dataset_root"].default_value
        )
        == "dataset"
    )


def test_collection_launch_records_timestamped_zstd_mcap() -> None:
    """验证录制开关、MCAP 格式、压缩方式和输出路径。"""
    # 生成待检查的统一启动描述。
    launch_description = _load_launch_module().generate_launch_description()
    # 顶层唯一的可选录制进程。
    recorder = next(
        entity
        for entity in launch_description.entities
        if isinstance(entity, ExecuteProcess)
    )
    assert isinstance(recorder.condition, IfCondition)

    # 使用测试根目录解析动态 launch substitutions。
    launch_context = LaunchContext()
    launch_context.launch_configurations["record_mcap"] = "true"
    launch_context.launch_configurations["dataset_root"] = "/tmp/fastumi"
    assert recorder.condition.evaluate(launch_context) is True
    # 每个命令参数由一组 substitutions 组成，解析后恢复 argv。
    command = [
        perform_substitutions(launch_context, argument)
        for argument in recorder.cmd
    ]

    assert command[:9] == [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--storage-preset-profile",
        "zstd_fast",
        "--disable-keyboard-controls",
        "--output",
    ]
    assert command[10] == "--all-topics"
    # 时间戳目录遵循 FastUMI 已有 UTC 会话标识格式。
    output_path = Path(command[9])
    assert output_path.parent == Path("/tmp/fastumi")
    assert output_path.name.startswith("fastumi_20")
    assert output_path.name.endswith("Z")


def test_package_dependencies_include_collection_camera() -> None:
    """验证统一采集与夹爪估计均声明 ToF 运行时依赖。"""
    data_package = ElementTree.parse(PACKAGE_ROOT / "package.xml")
    estimator_package = ElementTree.parse(
        PACKAGE_ROOT.parent / "fastumi_gripper_estimator" / "package.xml"
    )
    data_dependencies = {
        element.text for element in data_package.findall("exec_depend")
    }
    estimator_dependencies = {
        element.text for element in estimator_package.findall("exec_depend")
    }

    assert "tof_stereo_camera" in data_dependencies
    assert "tof_stereo_camera" in estimator_dependencies
