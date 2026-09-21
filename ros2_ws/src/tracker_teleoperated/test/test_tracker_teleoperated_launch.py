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
    """默认自动启动管理器与 RViz，保留记录开关。"""
    arguments = {item.name: item for item in module().generate_launch_description().entities
                 if isinstance(item, DeclareLaunchArgument)}
    for name in ("autostart", "use_recorder", "use_rviz"):
        assert "".join(part.perform(LaunchContext()) for part in arguments[name].default_value) == "true"
    assert set(arguments) == {"config_file", "manager_config", "autostart", "use_recorder", "use_rviz", "rviz_config"}


def test_display_topics_follow_control_and_record_config(tmp_path):
    """更换配置后默认显示同步话题、固定坐标系和图像 QoS。"""
    parameters = yaml.safe_load((ROOT / "config/tracker_teleoperated.yaml").read_text())
    parameters["tracker_teleop"]["ros__parameters"].update(odom_frame="custom_odom", tracker_odom_topic="/custom/odom")
    parameters["tracker_teleop_recorder"]["ros__parameters"]["image_topic"] = "/custom/image"
    custom = tmp_path / "config.yaml"
    custom.write_text(yaml.safe_dump(parameters))
    management = tmp_path / "manager.yaml"
    management.write_text(yaml.safe_dump({"components": {
        "umi_camera": {"parameters": {"namespace": "custom/umi"}},
        "wrist_camera": {"parameters": {"namespace": "custom/wrist"}},
    }}))
    display = module().configured_rviz(ROOT / "config/tracker_teleoperated.rviz", custom, management)
    manager = display["Visualization Manager"]
    assert manager["Global Options"]["Fixed Frame"] == "custom_odom"
    items = {item["Class"]: item for item in manager["Displays"]}
    images = {item["Name"]: item for item in manager["Displays"]
              if item["Class"] == "rviz_default_plugins/Image"}
    assert set(images) == {"UMI 视频", "末端视频"}
    assert images["末端视频"]["Topic"]["Value"] == "/custom/wrist/image_raw"
    assert images["UMI 视频"]["Topic"]["Value"] == "/custom/umi/image_raw"
    assert all(item["Enabled"] and item["Topic"]["Reliability Policy"] == "Best Effort"
               for item in images.values())
    assert items["rviz_default_plugins/Odometry"]["Topic"]["Value"] == "/custom/odom"
    assert items["rviz_default_plugins/Odometry"]["Enabled"]
    assert display["Panels"][0]["Class"] == "tracker_teleoperated/TeleopPanel"


def test_window_exit_signals_manager_instead_of_aborting_save():
    """关闭窗口只通知管理器，避免整个 launch 提前终止保存。"""
    context = LaunchContext()
    context.launch_configurations.update({
        "config_file": str(ROOT / "config/tracker_teleoperated.yaml"),
        "manager_config": str(ROOT / "config/component_manager.yaml"),
        "autostart": "false", "use_recorder": "true", "use_rviz": "true",
        "rviz_config": str(ROOT / "config/tracker_teleoperated.rviz"),
    })
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
    context = LaunchContext()
    context.launch_configurations.update({
        "config_file": str(ROOT / "config/tracker_teleoperated.yaml"),
        "manager_config": str(ROOT / "config/component_manager.yaml"),
        "autostart": "false", "use_recorder": "true", "use_rviz": "true",
        "rviz_config": str(path),
    })
    loaded._launch(context)
    assert captured[0]["parameters"][0]["umi_video_device"] == UMI_DEVICE
    assert captured[0]["parameters"][0]["wrist_video_device"] == WRIST_DEVICE
    assert 'umi_image_topic' not in captured[0]['parameters'][0]


def test_legacy_dynamic_rviz_sources_are_ignored(tmp_path, monkeypatch):
    """旧 RViz 动态节点不覆盖管理配置中的稳定端口。"""
    loaded = module()
    saved = yaml.safe_load((ROOT / "config/tracker_teleoperated.rviz").read_text())
    saved['Panels'][0].update(UmiVideoDevice='/dev/video2', WristVideoDevice='/dev/video0')
    path = tmp_path / "legacy.rviz"
    path.write_text(yaml.safe_dump(saved))
    captured = []
    monkeypatch.setattr(loaded, "Node", lambda **kwargs: captured.append(kwargs) or Node(**kwargs))
    context = LaunchContext()
    context.launch_configurations.update({
        "config_file": str(ROOT / "config/tracker_teleoperated.yaml"),
        "manager_config": str(ROOT / "config/component_manager.yaml"),
        "autostart": "false", "use_recorder": "true", "use_rviz": "true",
        "rviz_config": str(path),
    })
    loaded._launch(context)
    assert 'umi_video_device' not in captured[0]["parameters"][0]
    assert 'wrist_video_device' not in captured[0]["parameters"][0]
