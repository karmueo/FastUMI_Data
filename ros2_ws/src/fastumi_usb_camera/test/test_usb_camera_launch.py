"""验证统一相机 launch 仅选择一种输出模式及参数优先级。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, EmitEvent
from launch.utilities import perform_substitutions
import pytest


LAUNCH_FILE = Path(__file__).resolve().parents[1] / "launch" / "usb_camera.launch.py"
PHYSICAL_DEVICE = "/dev/v4l/by-path/pci-test-usb-0:2.4:1.0-video-index0"


def _load_launch():
    """按路径加载带有 .launch.py 后缀的启动模块。"""
    spec = spec_from_file_location("fastumi_usb_camera_launch", LAUNCH_FILE)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context(module, **changes):
    """将声明的 launch 默认值放入上下文，然后应用测试覆盖值。"""
    context = LaunchContext()
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            context.launch_configurations[action.name] = perform_substitutions(
                context, action.default_value
            )
    context.launch_configurations.update(changes)
    return context


def _selected_node(monkeypatch, module, **changes):
    """截取 Node 声明，不启动相机或 ROS 进程。"""
    monkeypatch.setattr(module, "Node", lambda **kwargs: kwargs)
    nodes = module._camera_node(_context(module, **changes))
    assert len(nodes) == 1
    return nodes[0]


def test_default_mode_starts_only_python_raw_node(monkeypatch):
    """缺省启动入口及命名空间仍与原始图像模式一致。"""
    module = _load_launch()
    node = _selected_node(monkeypatch, module)
    assert node["executable"] == "usb_camera_node"
    assert node["parameters"][1] == {}
    assert node["namespace"].perform(_context(module)) == "usb_camera"


def test_jpeg_mode_starts_only_python_node(monkeypatch):
    """原生 JPEG 模式沿用现有 publish_compressed 参数。"""
    module = _load_launch()
    node = _selected_node(monkeypatch, module, publish_compressed="TRUE")
    assert node["executable"] == "usb_camera_node"
    assert node["parameters"][1] == {"publish_compressed": True}


def test_device_uid_is_passed_only_to_python_node(monkeypatch):
    """将设备 UID 传给 Python 节点，并拒绝 FFmpeg 模式中无效的 UID。"""
    module = _load_launch()
    node = _selected_node(monkeypatch, module, device_uid="1:15")
    assert node["parameters"][1] == {
        "device_uid": "1:15", "video_device": ""
    }
    with pytest.raises(ValueError, match="device_uid"):
        _selected_node(
            monkeypatch, module, device_uid="1:15", enable_ffmpeg="true"
        )


def test_video_device_is_passed_to_both_camera_modes(monkeypatch):
    """稳定物理路径传给两种节点，且不能与 UID 同时指定。"""
    module = _load_launch()
    node = _selected_node(monkeypatch, module, video_device=PHYSICAL_DEVICE)
    assert node["parameters"][1] == {"video_device": PHYSICAL_DEVICE}
    ffmpeg = _selected_node(
        monkeypatch, module, video_device=PHYSICAL_DEVICE,
        enable_ffmpeg="true",
    )
    assert ffmpeg["parameters"][2]["video_device"] == PHYSICAL_DEVICE
    with pytest.raises(ValueError, match="只能指定其中一个"):
        _selected_node(
            monkeypatch, module, video_device=PHYSICAL_DEVICE, device_uid="1:17"
        )
    with pytest.raises(ValueError, match="by-path"):
        _selected_node(monkeypatch, module, video_device="/dev/video0")


def test_explicit_usb_id_disables_default_video_device(monkeypatch):
    """显式 VID/PID 启动时清除 YAML 中的默认视频设备路径。"""
    module = _load_launch()
    node = _selected_node(
        monkeypatch, module, vendor_id="0x1bcf", product_id="0x28c4"
    )
    assert node["parameters"][1] == {
        "vendor_id": 0x1BCF,
        "product_id": 0x28C4,
        "video_device": "",
    }


def test_ffmpeg_takes_precedence_and_layers_camera_parameters(monkeypatch):
    """FFmpeg 模式只启一个 C++ 发送端，并依次叠加配置和显式相机参数。"""
    module = _load_launch()
    node = _selected_node(
        monkeypatch, module,
        enable_ffmpeg="TRUE", publish_compressed="true",
        ffmpeg_config="/tmp/ffmpeg.yaml", config="/tmp/camera.yaml",
        vendor_id="0x1bcf", width="640", frame_id="custom_frame",
        namespace="ignored_for_ffmpeg",
    )
    assert node["executable"] == "usb_camera_ffmpeg"
    assert node["parameters"][0].perform(
        _context(module, ffmpeg_config="/tmp/ffmpeg.yaml")
    ) == "/tmp/ffmpeg.yaml"
    assert node["parameters"][1] == "/tmp/camera.yaml"
    overrides = node["parameters"][2]
    assert overrides["vendor_id"] == 0x1BCF
    assert overrides["width"] == 640
    assert overrides["frame_id"] == "custom_frame"
    assert overrides["video_device"] == ""
    assert overrides["topic"] == "/usb_camera/image_raw"
    assert overrides["h264_encoder"] == "hardware"
    assert overrides["usb_camera.image_raw.ffmpeg.encoder"] == "libx264"
    assert callable(node["on_exit"])
    assert "namespace" not in node
    assert "publish_compressed" not in node["parameters"][2]


def test_ffmpeg_decoder_is_optional_and_uses_explicit_topics(monkeypatch):
    """FFmpeg 默认只启动发送端，显式开关才增加本地解码节点。"""
    module = _load_launch()
    monkeypatch.setattr(module, "Node", lambda **kwargs: kwargs)
    context = _context(
        module, enable_ffmpeg="true", enable_decoder="true",
        topic="/wrist_camera/image_raw",
        decoded_topic="/wrist_camera/image_decoded",
    )
    nodes = module._camera_node(context)
    assert [node["executable"] for node in nodes] == [
        "usb_camera_ffmpeg", "usb_camera_receiver"]
    assert nodes[0]["parameters"][2]["topic"] == "/wrist_camera/image_raw"
    assert nodes[1]["parameters"][1] == {
        "input_topic": "/wrist_camera/image_raw",
        "output_topic": "/wrist_camera/image_decoded",
    }


def test_ffmpeg_software_encoder_override(monkeypatch):
    """显式 software 选择保留原 libx264 发布路径。"""
    module = _load_launch()
    node = _selected_node(
        monkeypatch, module, enable_ffmpeg="true", h264_encoder="software")
    assert node["parameters"][2]["h264_encoder"] == "software"
    assert node["parameters"][2]["usb_camera.image_raw.ffmpeg.encoder"] == "libx264"


def test_invalid_h264_encoder_fails_before_node_start(monkeypatch):
    module = _load_launch()
    with pytest.raises(ValueError, match="h264_encoder"):
        _selected_node(
            monkeypatch, module, enable_ffmpeg="true", h264_encoder="automatic")


def test_h264_camera_exit_requests_global_shutdown():
    module = _load_launch()
    actions = module._shutdown_on_camera_exit(
        SimpleNamespace(returncode=1), LaunchContext())
    assert len(actions) == 1
    assert isinstance(actions[0], EmitEvent)


def test_decoder_requires_ffmpeg_mode(monkeypatch):
    module = _load_launch()
    with pytest.raises(ValueError, match="FFmpeg"):
        _selected_node(monkeypatch, module, enable_decoder="true")


def test_receive_launch_exposes_remote_topic_overrides(monkeypatch):
    """独立接收入口应把遥操端的基础输入和解码输出话题传给节点。"""
    receive_file = LAUNCH_FILE.with_name("receive.launch.py")
    spec = spec_from_file_location("fastumi_usb_camera_receive_launch", receive_file)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "Node", lambda **kwargs: kwargs)
    description = module.generate_launch_description()
    context = LaunchContext()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            context.launch_configurations[action.name] = perform_substitutions(
                context, action.default_value)
    context.launch_configurations.update({
        "input_topic": "/wrist_camera/image_raw",
        "output_topic": "/wrist_camera/image_decoded",
    })
    node = description.entities[-1]
    overrides = node["parameters"][1]
    assert overrides["input_topic"].perform(context) == "/wrist_camera/image_raw"
    assert overrides["output_topic"].perform(context) == "/wrist_camera/image_decoded"


def test_ffmpeg_defaults_to_installed_encoder_config(monkeypatch):
    """FFmpeg 开关默认使用包内编码配置，原相机配置排在其后。"""
    module = _load_launch()
    context = _context(module, enable_ffmpeg="true")
    monkeypatch.setattr(module, "Node", lambda **kwargs: kwargs)
    node = module._camera_node(context)[0]
    ffmpeg_config = node["parameters"][0].perform(context)
    assert ffmpeg_config.endswith("/config/ffmpeg.yaml")
    assert Path(ffmpeg_config).is_file()
    assert node["parameters"][1].endswith("/config/usb_camera.yaml")


@pytest.mark.parametrize(
    "name", ["enable_ffmpeg", "enable_decoder", "publish_compressed"])
def test_invalid_boolean_launch_value_fails(monkeypatch, name):
    """模式参数拼写错误时直接报错，避免意外发布错误的话题。"""
    module = _load_launch()
    with pytest.raises(ValueError, match=name):
        _selected_node(monkeypatch, module, **{name: "yes"})
