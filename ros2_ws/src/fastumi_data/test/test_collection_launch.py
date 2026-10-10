"""验证 UMI 服务化采集 launch 的默认值、设备开关和迁移提示。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
from xml.etree import ElementTree

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.utilities import perform_substitutions
import pytest
import yaml


# 当前测试文件对应的 fastumi_data 包根目录。
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# 待验证的统一采集 launch 文件。
LAUNCH_FILE = PACKAGE_ROOT / "launch" / "fastumi_collection.launch.py"
# 符合物理端口规则的合法相机设备路径。
VIDEO_DEVICE = "/dev/v4l/by-path/pci-0000:06:00.4-usb-0:2.2:1.0-video-index0"


def _load_launch_module():
    """加载文件名包含点号的 Python launch 模块。"""
    module_spec = spec_from_file_location("fastumi_collection_launch", LAUNCH_FILE)
    assert module_spec is not None and module_spec.loader is not None
    launch_module = module_from_spec(module_spec)
    module_spec.loader.exec_module(launch_module)
    return launch_module


def _expand(**overrides):
    """用默认参数加覆盖值展开 launch，返回 (参数声明, 捕获的节点, 子 launch)。"""
    module = _load_launch_module()
    description = module.generate_launch_description()
    declarations = {
        entity.name: entity
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    context = LaunchContext()
    for name, declaration in declarations.items():
        context.launch_configurations[name] = perform_substitutions(
            context, declaration.default_value
        )
    context.launch_configurations.update(overrides)
    captured = []
    original_node = module.Node

    def capture(**kwargs):
        """保留真实节点对象，同时记录创建参数。"""
        captured.append(kwargs)
        return original_node(**kwargs)

    module.Node = capture
    opaque = next(e for e in description.entities if isinstance(e, OpaqueFunction))
    try:
        actions = opaque.execute(context)
    finally:
        module.Node = original_node
    includes = [a for a in actions if isinstance(a, IncludeLaunchDescription)]
    return declarations, captured, includes, context


def _include_name(include) -> str:
    """返回子 launch 的 ``<包名>/<文件名>``，从未解析的路径替换式中提取。"""
    text = str(include.launch_description_source.location)
    package = re.search(r"pkg='([^']+)'", text).group(1)
    filename = re.findall(r"'([^']+\.launch\.py)", text)[-1]
    return f"{package}/{filename}"


def test_default_launch_only_starts_collection_node_and_rviz() -> None:
    """验证默认不启动任何设备，只启动采集节点和 RViz2。"""
    declarations, nodes, includes, context = _expand()

    assert includes == []
    assert [(n["package"], n["executable"]) for n in nodes] == [
        ("fastumi_data", "collection_node"),
        ("rviz2", "rviz2"),
    ]
    for name in ("start_camera", "start_tracker", "start_gripper"):
        assert perform_substitutions(context, declarations[name].default_value) == "false"
        assert declarations[name].choices == ["true", "false"]
    assert perform_substitutions(context, declarations["use_rviz"].default_value) == "true"
    assert perform_substitutions(context, declarations["dataset_root"].default_value) == "dataset"
    assert (
        perform_substitutions(context, declarations["image_topic"].default_value)
        == "/umi_camera/image_raw"
    )
    assert "device_path" not in declarations


def test_rviz_loads_collection_layout_with_default_fixed_frame() -> None:
    """验证 RViz2 默认加载采集布局，固定坐标系为 steamvr_tracking 且可覆盖。"""
    _, nodes, _, _ = _expand()
    arguments = nodes[1]["arguments"]

    assert arguments[0] == "-d" and arguments[2] == "-f"
    assert arguments[1].endswith("/share/fastumi_data/config/collection.rviz")
    assert arguments[3] == "steamvr_tracking"
    _, nodes, _, _ = _expand(fixed_frame="vive_tracker_odom")
    assert nodes[1]["arguments"][3] == "vive_tracker_odom"


def test_rviz_stays_enabled_when_tracker_sublaunch_disables_its_own() -> None:
    """验证 Tracker 子 launch 的 use_rviz:=false 不会泄漏并关闭采集 RViz2。

    实机曾因运行时条件被子 launch 参数遮蔽而不启动 RViz2，因此在展开阶段解析开关。
    """
    _, nodes, includes, _ = _expand(start_tracker="true")

    assert [n["package"] for n in nodes] == ["fastumi_data", "rviz2"]
    assert dict(includes[0].launch_arguments)["use_rviz"] == "false"
    assert "condition" not in nodes[1]


def test_use_rviz_false_omits_rviz_node() -> None:
    """验证 use_rviz:=false 时不创建 RViz2 节点。"""
    _, nodes, _, _ = _expand(use_rviz="false")

    assert [n["package"] for n in nodes] == ["fastumi_data"]


def test_collection_rviz_config_registers_panel_and_displays() -> None:
    """验证 RViz 配置包含采集面板、原生 Image 和 Tracker Pose 显示。"""
    document = yaml.safe_load((PACKAGE_ROOT / "config" / "collection.rviz").read_text("utf-8"))

    assert document["Panels"][0]["Class"] == "fastumi_rviz_plugins/CollectionPanel"
    manager = document["Visualization Manager"]
    classes = {display["Class"]: display for display in manager["Displays"]}
    assert classes["rviz_default_plugins/Image"]["Topic"]["Value"] == "/umi_camera/image_raw"
    assert classes["rviz_default_plugins/Pose"]["Topic"]["Value"] == "/vive_tracker/pose"
    assert manager["Global Options"]["Fixed Frame"] == "steamvr_tracking"


def test_collection_node_receives_defaults_file_and_overrides() -> None:
    """验证节点先加载默认参数文件，再应用 launch 覆盖。"""
    _, nodes, _, _ = _expand(
        dataset_root="/data/umi", extrinsic_path="/cal/tracker_to_tcp.yaml"
    )
    parameters = nodes[0]["parameters"]

    assert parameters[0].endswith("/share/fastumi_data/config/collection.yaml")
    assert parameters[1] == {
        "dataset_root": "/data/umi",
        "image_topic": "/umi_camera/image_raw",
        "extrinsic_path": "/cal/tracker_to_tcp.yaml",
    }
    assert "processing_config" not in parameters[1]
    defaults = yaml.safe_load(Path(parameters[0]).read_text("utf-8"))
    assert set(defaults) == {"collection_node"}


def test_start_camera_requires_explicit_valid_video_device() -> None:
    """验证启用相机必须提供合法 by-path 设备，且不会猜测默认设备。"""
    with pytest.raises(ValueError, match="video_device"):
        _expand(start_camera="true")
    with pytest.raises(ValueError, match="by-path"):
        _expand(start_camera="true", video_device="/dev/video0")


def test_start_camera_includes_usb_camera_in_image_namespace() -> None:
    """验证相机子 launch 的命名空间由图像话题推导，发布 /umi_camera/image_raw。"""
    _, _, includes, _ = _expand(start_camera="true", video_device=VIDEO_DEVICE)

    assert [_include_name(i) for i in includes] == ["fastumi_usb_camera/usb_camera.launch.py"]
    assert dict(includes[0].launch_arguments) == {
        "namespace": "umi_camera",
        "video_device": VIDEO_DEVICE,
    }
    with pytest.raises(ValueError, match="image_raw"):
        _expand(start_camera="true", video_device=VIDEO_DEVICE, image_topic="/cam/raw")


def test_start_tracker_disables_its_own_rviz_and_forwards_serial() -> None:
    """验证 Tracker 子 launch 关闭自带 RViz2 并转发序列号。"""
    _, _, includes, _ = _expand(start_tracker="true", tracker_serial="LHR-TEST")

    assert [_include_name(i) for i in includes] == ["vive_tracker/vive_tracker.launch.py"]
    assert dict(includes[0].launch_arguments) == {
        "serial": "LHR-TEST",
        "use_rviz": "false",
    }


def test_start_gripper_reuses_default_calibration_resolution() -> None:
    """验证夹爪估计订阅采集图像话题，未覆盖标定时沿用其默认解析并写入快照。"""
    _, nodes, includes, _ = _expand(start_gripper="true", publish_debug_image="true")

    assert [_include_name(i) for i in includes] == ["fastumi_gripper_estimator/gripper_openness.launch.py"]
    arguments = dict(includes[0].launch_arguments)
    snapshots = nodes[0]["parameters"][1]["snapshot_files"]
    # 必须显式传入已解析的标定：同名的空参数会遮蔽子 launch 的默认值。
    assert arguments == {
        "image_topic": "/umi_camera/image_raw",
        "publish_debug_image": "true",
        "camera_calibration_path": snapshots[0],
        "gripper_calibration_path": snapshots[1],
    }
    assert [Path(p).name for p in snapshots][0] == "calib.yaml"
    assert Path(snapshots[1]).name in ("calibration.yaml", "fastumi_gripper_calibration.yaml")


def test_explicit_calibration_paths_are_forwarded_and_snapshotted() -> None:
    """验证显式相机内参和夹爪标定同时传给估计节点与采集快照。"""
    _, nodes, includes, _ = _expand(
        start_gripper="true",
        camera_calibration_path="/cal/camera.yaml",
        gripper_calibration_path="/cal/gripper.yaml",
    )

    arguments = dict(includes[0].launch_arguments)
    assert arguments["camera_calibration_path"] == "/cal/camera.yaml"
    assert arguments["gripper_calibration_path"] == "/cal/gripper.yaml"
    assert nodes[0]["parameters"][1]["snapshot_files"] == [
        "/cal/camera.yaml",
        "/cal/gripper.yaml",
    ]


def test_same_named_calibration_snapshots_are_rejected_at_launch() -> None:
    """验证同名标定文件在 launch 阶段就被拒绝，避免开始采集时才失败。"""
    with pytest.raises(ValueError, match="同名"):
        _expand(
            start_gripper="true",
            camera_calibration_path="/a/calibration.yaml",
            gripper_calibration_path="/b/calibration.yaml",
        )


def test_record_mcap_true_reports_migration_hint() -> None:
    """验证旧自动录包参数给出明确迁移提示，false 保持无害。"""
    with pytest.raises(ValueError) as raised:
        _expand(record_mcap="true")

    message = str(raised.value)
    assert "record_mcap 已移除" in message
    assert "/fastumi/collection/start" in message
    assert "record_session" in message
    _expand(record_mcap="false")


def test_invalid_boolean_flags_are_rejected() -> None:
    """验证布尔开关只接受 true/false。"""
    with pytest.raises(ValueError, match="start_tracker"):
        _expand(start_tracker="yes")


def test_launch_description_has_no_process_recorder() -> None:
    """验证 launch 不再包含 ros2 bag 自动录包进程。"""
    description = _load_launch_module().generate_launch_description()

    assert not any(type(e).__name__ == "ExecuteProcess" for e in description.entities)


def test_package_dependencies_cover_launched_packages() -> None:
    """验证包清单声明采集 launch 会启动的全部包。"""
    package = ElementTree.parse(PACKAGE_ROOT / "package.xml")
    dependencies = {element.text for element in package.findall("exec_depend")}

    assert {
        "fastumi_usb_camera",
        "vive_tracker",
        "fastumi_gripper_estimator",
        "gripper_openness",
        "rviz2",
        "fastumi_rviz_plugins",
        "tf2_msgs",
    } <= dependencies


def test_fastdds_profile_is_set_for_children_unless_user_already_chose(monkeypatch) -> None:
    """验证默认设置 Fast DDS 大图像配置；用户已设置环境变量或留空时不覆盖。"""
    from launch.actions import SetEnvironmentVariable

    monkeypatch.delenv("FASTRTPS_DEFAULT_PROFILES_FILE", raising=False)
    module = _load_launch_module()
    description = module.generate_launch_description()
    opaque = next(e for e in description.entities if isinstance(e, OpaqueFunction))

    def expand(**overrides):
        """展开 launch 并返回其中的环境变量设置动作。"""
        context = LaunchContext()
        for entity in description.entities:
            if isinstance(entity, DeclareLaunchArgument):
                context.launch_configurations[entity.name] = perform_substitutions(
                    context, entity.default_value
                )
        context.launch_configurations.update(overrides)
        return [a for a in opaque.execute(context) if isinstance(a, SetEnvironmentVariable)]

    default = expand()
    assert len(default) == 1
    profile = Path(PACKAGE_ROOT / "config" / "fastdds_large_images.xml")
    assert profile.is_file()
    assert perform_substitutions(LaunchContext(), default[0].value).endswith(
        "/share/fastumi_data/config/fastdds_large_images.xml"
    )
    assert expand(fastdds_profile="") == []
    monkeypatch.setenv("FASTRTPS_DEFAULT_PROFILES_FILE", "/custom.xml")
    assert expand() == []


def test_fastdds_profile_enlarges_shm_segment_beyond_a_1080p_frame() -> None:
    """验证配置文件的 SHM 段大于一帧 1080p bgr8 图像（约 6.2 MB）。"""
    root = ElementTree.parse(PACKAGE_ROOT / "config" / "fastdds_large_images.xml").getroot()
    namespace = {"p": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}
    shm = next(
        d for d in root.iterfind(".//p:transport_descriptor", namespace)
        if d.findtext("p:type", namespaces=namespace) == "SHM"
    )

    assert int(shm.findtext("p:segment_size", namespaces=namespace)) > 10 * 1920 * 1080 * 3
    assert root.find(".//p:participant", namespace).get("is_default_profile") == "true"
