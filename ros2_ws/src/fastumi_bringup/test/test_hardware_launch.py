"""验证末端相机选择和回位后的录制门控。"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction
from launch.utilities import perform_substitutions
import launch_ros.actions
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
    assert actions[0][1]['publish_reliability'] == 'reliable'
    assert actions[1][1]['image_topic'] == topic
    assert actions[1][1]['image_transport'] == transport
    assert actions[1][1]['image_reliability'] == 'reliable'
    assert actions[1][1]['image_qos_depth'] == '30'
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
    ({'camera_publish_reliability': 'invalid'}, 'camera_publish_reliability'),
    ({'camera_record_reliability': 'invalid'}, 'camera_record_reliability'),
    ({'camera_record_depth': '0'}, 'camera_record_depth'),
    ({'camera_publish_reliability': 'best_effort'}, '不兼容'),
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


def test_camera_qos_overrides_are_forwarded(monkeypatch, tmp_path):
    mod = module()
    device = tmp_path / 'camera-video-index0'
    device.touch()
    monkeypatch.setattr(mod, 'is_physical_video_device_path', lambda _: True)
    monkeypatch.setattr(mod, '_include', lambda package, launch, arguments=None: (package, arguments))
    actions = mod._launch_hardware(context(
        mod, start_arm='false', start_gripper='false', wrist_video_device=str(device),
        camera_publish_reliability='best_effort',
        camera_record_reliability='best_effort', camera_record_depth='45'))
    assert actions[0][1]['publish_reliability'] == 'best_effort'
    assert actions[1][1]['image_reliability'] == 'best_effort'
    assert actions[1][1]['image_qos_depth'] == '45'


def test_missing_camera_fails_before_start():
    mod = module()
    with pytest.raises(ValueError, match='末端相机'):
        mod._launch_hardware(context(mod))


@pytest.mark.parametrize('mode', ['raw', 'jpeg'])
def test_unbound_camera_can_start_by_physical_usb_port(monkeypatch, mode):
    """节点缺失但相同物理端口仍在线时，允许 libuvc 恢复采集。"""
    mod = module()
    calls = []
    device = '/dev/v4l/by-path/pci-missing-usb-0:4.2:1.0-video-index0'
    monkeypatch.setattr(mod, 'usb_location_from_physical_port',
                        lambda path: calls.append(path) or (1, 14))
    monkeypatch.setattr(mod, '_include', lambda package, launch, arguments=None: (package, arguments))
    actions = mod._launch_hardware(context(
        mod, start_arm='false', start_gripper='false', start_recorder='false',
        wrist_video_device=device, wrist_camera_mode=mode))
    assert calls == [device]
    assert actions[0][1]['video_device'] == device


def test_h264_still_requires_video_node(monkeypatch):
    """V4L2 后端不能使用缺失的视频节点。"""
    mod = module()
    with pytest.raises(ValueError, match='H.264'):
        mod._launch_hardware(context(
            mod, wrist_camera_mode='h264',
            wrist_video_device='/dev/v4l/by-path/pci-missing-video-index0'))


def test_disconnected_camera_fails_before_start(monkeypatch):
    """USB 端口本身离线时，仍在启动其他硬件前失败。"""
    mod = module()

    def missing(_device):
        raise ValueError('USB 离线')

    monkeypatch.setattr(mod, 'usb_location_from_physical_port', missing)
    with pytest.raises(ValueError, match='末端相机不可用.*USB 离线'):
        mod._launch_hardware(context(
            mod, wrist_video_device='/dev/v4l/by-path/pci-missing-video-index0'))


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
    monkeypatch.setattr(mod, '_rm_driver', lambda _context: 'rm_driver')
    actions = mod._launch_hardware(context(mod, start_wrist_camera='false', start_gripper='false',
                                          move_to_initial_pose='false'))
    assert actions[0] == 'rm_driver'
    assert actions[1].node_executable == 'arm_control_service'
    assert actions[2:] == ['fastumi_recorder']


def test_arm_control_service_can_be_disabled(monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, '_include', lambda package, launch, arguments=None: package)
    monkeypatch.setattr(mod, '_rm_driver', lambda _context: 'rm_driver')
    actions = mod._launch_hardware(context(mod, start_wrist_camera='false', start_gripper='false',
                                          move_to_initial_pose='false',
                                          start_arm_control='false'))
    assert actions == ['rm_driver', 'fastumi_recorder']


def _fixed_actions(mod, monkeypatch, tmp_path, **changes):
    wrist = tmp_path / 'wrist-video-index0'
    wrist.touch()
    monkeypatch.setattr(mod, 'is_physical_video_device_path', lambda _: True)
    monkeypatch.setattr(mod, '_include', lambda package, launch, arguments=None: (package, arguments))
    return mod._launch_hardware(context(
        mod, start_arm='false', start_gripper='false', wrist_video_device=str(wrist), **changes))


def test_fixed_camera_default_skipped(monkeypatch, tmp_path):
    mod = module()
    actions = _fixed_actions(mod, monkeypatch, tmp_path)
    assert [a[0] for a in actions] == ['fastumi_usb_camera', 'fastumi_recorder']
    assert actions[1][1]['fixed_image_topic'] == ''


def test_fixed_camera_online_starts_and_records(monkeypatch, tmp_path):
    mod = module()
    fixed = tmp_path / 'fixed-video-index0'
    fixed.touch()
    actions = _fixed_actions(mod, monkeypatch, tmp_path, fixed_video_device=str(fixed))
    assert [a[0] for a in actions] == ['fastumi_usb_camera', 'fastumi_usb_camera', 'fastumi_recorder']
    args = actions[1][1]
    assert args['namespace'] == 'fixed_camera' and args['video_device'] == str(fixed)
    assert args['publish_compressed'] == 'true' and args['enable_ffmpeg'] == 'false'
    assert args['frame_id'] == 'fixed_camera_optical_frame'
    assert args['width'].perform(context(mod)) == '1920'
    assert args['fps'].perform(context(mod)) == '30'
    assert actions[0][1]['namespace'] == 'wrist_camera'
    assert actions[2][1]['fixed_image_topic'] == '/fixed_camera/image_raw/compressed'


def test_fixed_camera_offline_is_skipped(monkeypatch, tmp_path):
    mod = module()

    def offline(_device):
        raise ValueError('USB 离线')

    monkeypatch.setattr(mod, 'usb_location_from_physical_port', offline)
    actions = _fixed_actions(mod, monkeypatch, tmp_path,
                             fixed_video_device='/dev/v4l/by-path/pci-missing-video-index0')
    assert [a[0] for a in actions] == ['fastumi_usb_camera', 'fastumi_recorder']
    assert actions[1][1]['fixed_image_topic'] == ''


def test_fixed_camera_missing_node_but_usb_online(monkeypatch, tmp_path):
    mod = module()
    monkeypatch.setattr(mod, 'usb_location_from_physical_port', lambda _: (1, 15))
    actions = _fixed_actions(mod, monkeypatch, tmp_path,
                             fixed_video_device='/dev/v4l/by-path/pci-missing-video-index0')
    assert actions[1][1]['namespace'] == 'fixed_camera'


def test_fixed_camera_same_device_fails(monkeypatch, tmp_path):
    mod = module()
    with pytest.raises(ValueError, match='同一设备'):
        _fixed_actions(mod, monkeypatch, tmp_path,
                       fixed_video_device=str(tmp_path / 'wrist-video-index0'))


def test_fixed_camera_invalid_path_fails(tmp_path):
    mod = module()
    with pytest.raises(ValueError, match='固定相机'):
        mod._launch_hardware(context(mod, start_arm='false', start_gripper='false',
                                     wrist_video_device='/dev/v4l/by-path/x-video-index0',
                                     fixed_video_device='/dev/video9'))


def test_fixed_camera_disabled_switch(monkeypatch, tmp_path):
    mod = module()
    actions = _fixed_actions(mod, monkeypatch, tmp_path, start_fixed_camera='false',
                             fixed_video_device='/dev/v4l/by-path/x-video-index0')
    assert actions[1][1]['fixed_image_topic'] == ''


def test_wrist_alignment_overlay_is_forwarded(monkeypatch, tmp_path):
    """安装对齐叠加默认关闭，开启与透明度透传给末端相机 launch。"""
    mod = module()
    device = tmp_path / 'camera-video-index0'
    device.touch()
    monkeypatch.setattr(mod, 'is_physical_video_device_path', lambda _: True)
    monkeypatch.setattr(
        mod, '_include', lambda package, launch, arguments=None: (package, arguments))
    ctx = context(
        mod, start_arm='false', start_gripper='false', start_recorder='false',
        wrist_video_device=str(device))
    camera = mod._launch_hardware(ctx)[0]
    assert camera[1]['alignment_overlay'].perform(ctx) == 'false'
    ctx.launch_configurations.update(wrist_alignment_overlay='true', wrist_alignment_alpha='0.3')
    assert camera[1]['alignment_overlay'].perform(ctx) == 'true'
    assert camera[1]['alignment_alpha'].perform(ctx) == '0.3'


@pytest.mark.parametrize('mode', ['raw', 'jpeg', 'h264'])
def test_wrist_alignment_does_not_enable_fixed_camera_overlay(monkeypatch, tmp_path, mode):
    """执行两次真实 include 的参数声明，防止腕部叠加开关泄漏到固定相机。"""
    nodes = []

    def capture_node(ctx, **settings):
        """只记录节点声明，不启动相机、解码器或其他硬件。"""
        settings['namespace'] = settings['namespace'].perform(ctx) \
            if 'namespace' in settings else ''
        nodes.append(settings)

    monkeypatch.setattr(
        launch_ros.actions, 'Node',
        lambda **settings: OpaqueFunction(function=capture_node, kwargs=settings))
    mod = module()
    camera_share = Path(__file__).resolve().parents[2] / 'fastumi_usb_camera'
    package_share = mod.get_package_share_directory
    monkeypatch.setattr(
        mod, 'get_package_share_directory',
        lambda name: str(camera_share) if name == 'fastumi_usb_camera' else package_share(name))
    monkeypatch.setattr(mod, 'is_physical_video_device_path', lambda _: True)
    wrist, fixed = tmp_path / 'wrist-video-index0', tmp_path / 'fixed-video-index0'
    wrist.touch()
    fixed.touch()
    # 相机 launch 自身仍验证物理端口格式；这里只替代路径验证，不打开设备。
    monkeypatch.setattr('fastumi_usb_camera.capture.is_physical_video_device_path', lambda _: True)
    ctx = context(
        mod, start_arm='false', start_gripper='false', start_recorder='false',
        wrist_video_device=str(wrist), fixed_video_device=str(fixed),
        wrist_camera_mode=mode, enable_decoder='true' if mode == 'h264' else 'false',
        wrist_alignment_overlay='true', wrist_alignment_alpha='0.3')

    def visit(entity):
        """按 launch 顺序执行声明和作用域动作，替代真正的进程启动。"""
        for child in entity.visit(ctx) or []:
            visit(child)

    for action in mod._launch_hardware(ctx):
        visit(action)
    overlays = [node for node in nodes if node['executable'] == 'alignment_overlay_node']
    assert len(overlays) == 1
    assert overlays[0]['namespace'] == 'wrist_camera'
    assert overlays[0]['parameters'][0]['alpha'] == 0.3
    assert overlays[0]['parameters'][0]['input_topic'] == {
        'raw': 'image_raw', 'jpeg': 'image_raw/compressed',
        'h264': '/wrist_camera/image_decoded',
    }[mode]
    fixed_nodes = [node for node in nodes if node['namespace'] == 'fixed_camera']
    assert [node['executable'] for node in fixed_nodes] == ['usb_camera_node']


def _driver_parameters(monkeypatch, tmp_path, **changes):
    mod = module()
    monkeypatch.setattr(mod, 'get_package_share_directory', lambda _package: str(tmp_path))
    return tmp_path, mod._rm_driver_parameters(context(mod, **changes))


def test_rm_driver_defaults_to_low_follow_config(monkeypatch, tmp_path):
    """默认低跟随只加载驱动配置文件，保持 trajectory_mode=0。"""
    share, parameters = _driver_parameters(monkeypatch, tmp_path)
    assert parameters == [str(share / 'config/rm_75_config.yaml')]


@pytest.mark.parametrize('changes, mode, radio', [
    ({}, 2, 200),
    ({'arm_trajectory_mode': '1'}, 1, 50),
    ({'arm_trajectory_mode': '2', 'arm_radio': '1000'}, 2, 1000),
    ({'arm_trajectory_mode': '1', 'arm_radio': '0'}, 1, 0),
])
def test_rm_driver_high_follow_overrides_smoothing(monkeypatch, tmp_path, changes, mode, radio):
    """高跟随在配置文件之后覆盖 trajectory_mode 与 radio，radio 留空按模式取默认值。"""
    share, parameters = _driver_parameters(
        monkeypatch, tmp_path, arm_follow_mode='high', **changes)
    assert parameters == [str(share / 'config/rm_75_config.yaml'),
                          {'trajectory_mode': mode, 'radio': radio}]


@pytest.mark.parametrize('changes, message', [
    ({'arm_follow_mode': 'medium'}, 'arm_follow_mode'),
    ({'arm_follow_mode': 'high', 'arm_trajectory_mode': '0'}, 'arm_trajectory_mode'),
    ({'arm_follow_mode': 'high', 'arm_trajectory_mode': '1', 'arm_radio': '101'}, '0～100'),
    ({'arm_follow_mode': 'high', 'arm_radio': '1001'}, '0～1000'),
    ({'arm_follow_mode': 'high', 'arm_radio': '-1'}, 'arm_radio'),
])
def test_rm_driver_rejects_invalid_follow_settings(monkeypatch, tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        _driver_parameters(monkeypatch, tmp_path, **changes)
