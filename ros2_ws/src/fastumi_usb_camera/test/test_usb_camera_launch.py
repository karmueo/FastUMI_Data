"""验证统一相机 launch 仅选择一种输出模式及参数优先级。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
import pytest


LAUNCH_FILE = Path(__file__).resolve().parents[1] / "launch" / "usb_camera.launch.py"


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
    assert node["parameters"][1:] == [
        "/tmp/camera.yaml",
        {"vendor_id": 0x1BCF, "width": 640, "frame_id": "custom_frame"},
    ]
    assert "namespace" not in node
    assert "publish_compressed" not in node["parameters"][2]


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


@pytest.mark.parametrize("name", ["enable_ffmpeg", "publish_compressed"])
def test_invalid_boolean_launch_value_fails(monkeypatch, name):
    """模式参数拼写错误时直接报错，避免意外发布错误的话题。"""
    module = _load_launch()
    with pytest.raises(ValueError, match=name):
        _selected_node(monkeypatch, module, **{name: "yes"})
