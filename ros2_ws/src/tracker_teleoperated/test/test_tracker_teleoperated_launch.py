"""验证面板启动参数、RViz 话题同步与窗口退出后保存的事件编排。"""

import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.events.process import ProcessExited, SignalProcess
from launch_ros.actions import Node
import yaml

# 包内源码资源用于独立验证，不依赖旧的安装目录。
ROOT = Path(__file__).resolve().parents[1]
UMI_DEVICE = "/dev/v4l/by-path/pci-test-usb-0:2.4:1.0-video-index0"
WRIST_DEVICE = "/dev/v4l/by-path/pci-test-usb-0:2.3:1.0-video-index0"


def module():
    """加载启动模块供静态参数和事件测试使用。"""
    spec = importlib.util.spec_from_file_location("tracker_launch", ROOT / "launch/tracker_teleoperated.launch.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_launch_defaults():
    """默认自动启动管理器与 RViz，但不显示两路视频。"""
    arguments = {item.name: item for item in module().generate_launch_description().entities
                 if isinstance(item, DeclareLaunchArgument)}
    for name in ("autostart", "use_recorder", "use_rviz"):
        assert "".join(part.perform(LaunchContext()) for part in arguments[name].default_value) == "true"
    for name in ("show_wrist_video", "show_umi_video"):
        assert "".join(part.perform(LaunchContext()) for part in arguments[name].default_value) == "false"
    assert set(arguments) == {"config_file", "manager_config", "autostart", "use_recorder", "use_rviz",
                              "show_wrist_video", "show_umi_video", "rviz_config"}


def launch_context(rviz_config="", **overrides):
    """构造含全部显式参数的 launch 测试上下文。"""
    context = LaunchContext()
    context.launch_configurations.update({
        "config_file": str(ROOT / "config/tracker_teleoperated.yaml"),
        "manager_config": str(ROOT / "config/component_manager.yaml"),
        "autostart": "false", "use_recorder": "true", "use_rviz": "true",
        "show_wrist_video": "false", "show_umi_video": "false",
        "rviz_config": str(rviz_config), **overrides,
    })
    return context


def launch_display(loaded, context):
    """捕获本次启动的节点参数和临时 RViz 配置，并清理测试副本。"""
    captured = []
    original = loaded.Node

    def capture(**kwargs):
        """保留真实节点对象，同时记录创建时的公开参数。"""
        captured.append(kwargs)
        return original(**kwargs)

    loaded.Node = capture
    try:
        loaded._launch(context)
    finally:
        loaded.Node = original
    path = Path(captured[1]["arguments"][1])
    try:
        return captured, yaml.safe_load(path.read_text(encoding="utf-8"))
    finally:
        path.unlink()


def image_names(display):
    """读取 RViz 配置中实际保留的视频画面名称。"""
    return {item["Name"] for item in display["Visualization Manager"]["Displays"]
            if item["Class"] == "rviz_default_plugins/Image"}


def test_video_visibility_and_decoder_parameter():
    """两路显示开关独立控制画面，末端开关同时传给管理器。"""
    loaded = module()
    for wrist, umi, expected in (
        (False, False, set()),
        (True, False, {"末端视频"}),
        (False, True, {"UMI 视频"}),
        (True, True, {"末端视频", "UMI 视频"}),
    ):
        context = launch_context(show_wrist_video=str(wrist).lower(), show_umi_video=str(umi).lower())
        nodes, display = launch_display(loaded, context)
        assert image_names(display) == expected
        assert nodes[0]["parameters"][0]["show_wrist_video"] is wrist


def test_without_rviz_does_not_start_decoder():
    """RViz 未启动时，即使请求显示末端画面也不解码。"""
    captured = []
    loaded = module()
    loaded.Node = lambda **kwargs: captured.append(kwargs) or Node(**kwargs)
    loaded._launch(launch_context(use_rviz="false", show_wrist_video="true"))
    assert len(captured) == 1
    assert captured[0]["parameters"][0]["show_wrist_video"] is False


def test_custom_rviz_visibility_preserves_other_settings(tmp_path):
    """自定义配置的其他画面、话题与面板字段在过滤后保持原值。"""
    saved = yaml.safe_load((ROOT / "config/tracker_teleoperated.rviz").read_text(encoding="utf-8"))
    saved["Panels"][0]["UmiVideoDevice"] = UMI_DEVICE
    saved["Visualization Manager"]["Global Options"]["Fixed Frame"] = "saved_odom"
    for item in saved["Visualization Manager"]["Displays"]:
        if item["Class"] == "rviz_default_plugins/Image":
            item["Topic"]["Value"] = "/saved/umi" if item["Name"] == "UMI 视频" else "/saved/wrist"
    path = tmp_path / "custom.rviz"
    path.write_text(yaml.safe_dump(saved, allow_unicode=True), encoding="utf-8")
    original = path.read_text(encoding="utf-8")
    _, display = launch_display(module(), launch_context(path, show_umi_video="true"))
    assert image_names(display) == {"UMI 视频"}
    assert display["Panels"] == saved["Panels"]
    assert display["Visualization Manager"]["Global Options"] == saved["Visualization Manager"]["Global Options"]
    image = next(item for item in display["Visualization Manager"]["Displays"] if item.get("Name") == "UMI 视频")
    assert image["Topic"]["Value"] == "/saved/umi"
    assert path.read_text(encoding="utf-8") == original


def test_display_topics_follow_control_and_record_config(tmp_path):
    """更换配置后默认显示同步话题、固定坐标系和图像 QoS。"""
    parameters = yaml.safe_load((ROOT / "config/tracker_teleoperated.yaml").read_text())
    parameters["tracker_teleop"]["ros__parameters"].update(odom_frame="custom_odom", tracker_odom_topic="/custom/odom")
    custom = tmp_path / "config.yaml"
    custom.write_text(yaml.safe_dump(parameters))
    management = tmp_path / "manager.yaml"
    management.write_text(yaml.safe_dump({"components": {
        "umi_camera": {"parameters": {"namespace": "custom/umi"}},
        "wrist_decoder": {"parameters": {"output_topic": "/custom/wrist/decoded"}},
    }}))
    display = module().configured_rviz(ROOT / "config/tracker_teleoperated.rviz", custom, management)
    manager = display["Visualization Manager"]
    assert manager["Global Options"]["Fixed Frame"] == "custom_odom"
    items = {item["Class"]: item for item in manager["Displays"]}
    images = {item["Name"]: item for item in manager["Displays"]
              if item["Class"] == "rviz_default_plugins/Image"}
    assert set(images) == {"UMI 视频", "末端视频"}
    assert images["末端视频"]["Topic"]["Value"] == "/custom/wrist/decoded"
    assert images["UMI 视频"]["Topic"]["Value"] == "/custom/umi/image_raw"
    assert all(item["Enabled"] and item["Topic"]["Reliability Policy"] == "Best Effort"
               for item in images.values())
    assert items["rviz_default_plugins/Odometry"]["Topic"]["Value"] == "/custom/odom"
    assert items["rviz_default_plugins/Odometry"]["Enabled"]
    assert display["Panels"][0]["Class"] == "tracker_teleoperated/TeleopPanel"


def test_window_exit_signals_manager_instead_of_aborting_save(tmp_path, monkeypatch):
    """关闭窗口只通知管理器，避免整个 launch 提前终止保存。"""
    monkeypatch.setattr(module().tempfile, "tempdir", str(tmp_path))
    context = launch_context(ROOT / "config/tracker_teleoperated.rviz")
    actions = module()._launch(context)
    nodes = [item for item in actions if isinstance(item, Node)]
    assert len(nodes) == 2
    event = ProcessExited(action=nodes[1], returncode=0, name="rviz", cmd=["rviz2"], cwd=None, env=None, pid=123)
    handler = next(item.event_handler for item in actions
                   if isinstance(item, RegisterEventHandler) and item.event_handler.matches(event))
    assert isinstance(handler.handle(event, context)[0].event, SignalProcess)


def test_saved_rviz_sources_initialize_manager(tmp_path, monkeypatch):
    """加载保存的视频源时，管理器与 RViz 使用同一初始输入。"""
    loaded = module()
    monkeypatch.setattr(loaded.tempfile, "tempdir", str(tmp_path))
    saved = yaml.safe_load((ROOT / "config/tracker_teleoperated.rviz").read_text())
    saved['Panels'][0].update(
        UmiVideoDevice=UMI_DEVICE, WristVideoDevice=WRIST_DEVICE)
    for item in saved["Visualization Manager"]["Displays"]:
        if item["Class"] == "rviz_default_plugins/Image":
            item["Topic"]["Value"] = "/saved/umi" if item["Name"] == "UMI 视频" else "/saved/wrist"
    path = tmp_path / "saved.rviz"
    path.write_text(yaml.safe_dump(saved))
    captured = []

    def capture(**kwargs):
        """保留真实 launch 节点并捕获传入的参数。"""
        captured.append(kwargs)
        return Node(**kwargs)

    monkeypatch.setattr(loaded, "Node", capture)
    context = launch_context(path)
    loaded._launch(context)
    assert captured[0]["parameters"][0]["umi_video_device"] == UMI_DEVICE
    assert "wrist_video_device" not in captured[0]["parameters"][0]
    assert 'umi_image_topic' not in captured[0]['parameters'][0]


def test_legacy_dynamic_rviz_sources_are_ignored(tmp_path, monkeypatch):
    """旧 RViz 动态节点不覆盖管理配置中的稳定端口。"""
    loaded = module()
    monkeypatch.setattr(loaded.tempfile, "tempdir", str(tmp_path))
    saved = yaml.safe_load((ROOT / "config/tracker_teleoperated.rviz").read_text())
    saved['Panels'][0].update(UmiVideoDevice='/dev/video2', WristVideoDevice='/dev/video0')
    path = tmp_path / "legacy.rviz"
    path.write_text(yaml.safe_dump(saved))
    captured = []
    monkeypatch.setattr(loaded, "Node", lambda **kwargs: captured.append(kwargs) or Node(**kwargs))
    context = launch_context(path)
    loaded._launch(context)
    assert 'umi_video_device' not in captured[0]["parameters"][0]
    assert 'wrist_video_device' not in captured[0]["parameters"][0]
