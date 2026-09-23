"""验证末端相机选择和回位后的录制门控。"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, EmitEvent
from launch.utilities import perform_substitutions
import pytest


def module():
    spec = spec_from_file_location('hardware_launch', Path(__file__).parents[1] / 'launch/hardware.launch.py')
    result = module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def context(mod, **changes):
    ctx = LaunchContext()
    for action in mod.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            ctx.launch_configurations[action.name] = perform_substitutions(ctx, action.default_value)
    ctx.launch_configurations.update(changes)
    return ctx


@pytest.mark.parametrize('mode, ffmpeg, compressed, topic, transport', [
    ('raw', 'false', 'false', '/wrist_camera/image_raw', 'raw'),
    ('jpeg', 'false', 'true', '/wrist_camera/image_raw/compressed', 'jpeg'),
    ('h264', 'true', 'false', '/wrist_camera/image_raw/ffmpeg', 'ffmpeg'),
])
def test_wrist_modes_match_recorder(monkeypatch, tmp_path, mode, ffmpeg, compressed,
                                   topic, transport):
    mod = module()
    device = tmp_path / 'camera-video-index0'
    device.touch()
    monkeypatch.setattr(mod, 'is_physical_video_device_path', lambda _: True)
    monkeypatch.setattr(mod, '_include', lambda package, launch, arguments=None: (package, arguments))
    actions = mod._launch_hardware(context(mod, start_arm='false', start_gripper='false',
                                          wrist_video_device=str(device),
                                          wrist_camera_mode=mode))
    assert [item[0] for item in actions] == ['fastumi_usb_camera', 'fastumi_recorder']
    assert actions[0][1]['namespace'] == 'wrist_camera'
    assert actions[0][1]['enable_ffmpeg'] == ffmpeg
    assert actions[0][1]['publish_compressed'] == compressed
    assert actions[0][1]['topic'] == '/wrist_camera/image_raw'
    assert actions[0][1]['enable_decoder'].perform(context(mod)) == 'false'
    assert ('h264_encoder' in actions[0][1]) == (mode == 'h264')
    assert actions[1][1]['image_topic'] == topic
    assert actions[1][1]['image_transport'] == transport
    assert not any('umi' in name for name in context(mod).launch_configurations)


def test_default_camera_mode_is_jpeg():
    mod = module()
    ctx = context(mod)
    assert ctx.launch_configurations['wrist_camera_mode'] == 'jpeg'


@pytest.mark.parametrize('changes, message', [
    ({'wrist_camera_mode': 'bad'}, 'wrist_camera_mode'),
    ({'enable_decoder': 'true'}, 'enable_decoder'),
    ({'wrist_camera_mode': 'raw', 'enable_decoder': 'true'}, 'enable_decoder'),
    ({'image_topic': '/other'}, 'image_topic/image_transport'),
    ({'image_transport': 'raw'}, 'image_topic/image_transport'),
])
def test_invalid_mode_or_conflicting_override_fails_before_device(changes, message):
    mod = module()
    with pytest.raises(ValueError, match=message):
        mod._launch_hardware(context(mod, **changes))


def test_matching_manual_override_is_allowed(monkeypatch, tmp_path):
    mod = module()
    device = tmp_path / 'camera-video-index0'
    device.touch()
    monkeypatch.setattr(mod, 'is_physical_video_device_path', lambda _: True)
    monkeypatch.setattr(mod, '_include', lambda package, launch, arguments=None: (package, arguments))
    actions = mod._launch_hardware(context(
        mod, start_arm='false', start_gripper='false', wrist_video_device=str(device),
        image_topic='/wrist_camera/image_raw/compressed', image_transport='jpeg'))
    assert actions[1][1]['image_transport'] == 'jpeg'


def test_missing_camera_fails_before_start():
    mod = module()
    with pytest.raises(ValueError, match='末端相机'):
        mod._launch_hardware(context(mod))


def test_wrist_decoder_override_is_forwarded(monkeypatch, tmp_path):
    mod = module()
    device = tmp_path / 'camera-video-index0'
    device.touch()
    monkeypatch.setattr(mod, 'is_physical_video_device_path', lambda _: True)
    monkeypatch.setattr(
        mod, '_include', lambda package, launch, arguments=None: (package, arguments))
    ctx = context(
        mod, start_arm='false', start_gripper='false', start_recorder='false',
        wrist_video_device=str(device), wrist_camera_mode='h264', enable_decoder='true')
    camera = mod._launch_hardware(ctx)[0]
    assert camera[1]['enable_decoder'].perform(ctx) == 'true'


def test_wrist_software_encoder_override_is_forwarded(monkeypatch, tmp_path):
    mod = module()
    device = tmp_path / 'camera-video-index0'
    device.touch()
    monkeypatch.setattr(mod, 'is_physical_video_device_path', lambda _: True)
    monkeypatch.setattr(
        mod, '_include', lambda package, launch, arguments=None: (package, arguments))
    ctx = context(
        mod, start_arm='false', start_gripper='false', start_recorder='false',
        wrist_video_device=str(device), wrist_camera_mode='h264', h264_encoder='software')
    camera = mod._launch_hardware(ctx)[0]
    assert camera[1]['h264_encoder'].perform(ctx) == 'software'


def test_disabled_devices_and_recorder_only(monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, '_include', lambda package, launch, arguments=None: package)
    assert mod._launch_hardware(context(mod, start_arm='false', start_gripper='false',
                                       start_wrist_camera='false')) == ['fastumi_recorder']


def test_external_image_source_when_wrist_camera_disabled(monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, '_include', lambda package, launch, arguments=None: (package, arguments))
    actions = mod._launch_hardware(context(
        mod, start_arm='false', start_gripper='false', start_wrist_camera='false',
        image_topic='/external/camera', image_transport='raw'))
    assert [item[0] for item in actions] == ['fastumi_recorder']
    assert actions[0][1]['image_topic'] == '/external/camera'
    assert actions[0][1]['image_transport'] == 'raw'


def test_default_gripper_config_is_installed_package_file():
    """默认值显式指向安装空间，不依赖空字符串回退。"""
    mod = module()
    configured = Path(context(mod).launch_configurations['gripper_config_file'])
    assert configured.name == 'gripper.yaml'
    assert configured.parent.name == 'config'
    assert configured.is_file()


@pytest.mark.parametrize('code', [1, -9])
def test_home_failure_and_timeout_gate(code):
    mod = module()
    recorder = object()
    actions = mod._after_home(SimpleNamespace(returncode=code), LaunchContext(), recorder)
    assert recorder not in actions
    assert any(isinstance(item, EmitEvent) for item in actions)


def test_home_success_and_shutdown():
    mod = module()
    recorder = object()
    assert mod._after_home(SimpleNamespace(returncode=0), LaunchContext(), recorder) == [recorder]
    assert mod._after_home(SimpleNamespace(returncode=0), SimpleNamespace(is_shutdown=True), recorder) == []


def test_maintenance_skips_homing(monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, '_include', lambda package, launch, arguments=None: package)
    actions = mod._launch_hardware(context(mod, start_wrist_camera='false', start_gripper='false',
                                          move_to_initial_pose='false'))
    assert actions == ['rm_driver', 'fastumi_recorder']
