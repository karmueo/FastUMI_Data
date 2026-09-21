"""验证设备枚举、相机事务互锁、交换释放顺序及失败恢复。"""

from concurrent.futures import Future
from types import SimpleNamespace
import struct
import time

import pytest
from rclpy.parameter import Parameter
from std_msgs.msg import String
from std_srvs.srv import SetBool

from tracker_teleoperated import camera_catalog
from tracker_teleoperated.component_manager import ComponentManager
from test_component_manager import manager, FakeClient  # noqa: F401


def prepare(manager):
    """注入两个可用物理相机，不访问真实采集设备。"""
    source = manager.video_sources
    source.devices = [dict(device=f'/dev/video{i}', name=f'camera{i}', error='') for i in (0, 2, 10)]
    source.next_scan = float('inf')
    return source


def select(manager, key='umi', device='/dev/video10'):
    """走标准 ROS 原子参数接口发起选择。"""
    return manager.set_parameters_atomically([Parameter(key+'_video_device', value=device)])


def advance(manager, healthy=True):
    """回收模拟进程后按需提供当前采集实例的新鲜图像。"""
    manager._tick()
    if healthy:
        for key in ('umi_camera', 'wrist_camera'):
            component = manager.components[key]
            if component.process:
                component.health.receive(manager.probes[key][0], True)
    manager._tick()


def test_stopped_selection_and_stable_topics(manager):
    """未运行的相机只更新会话分配；消费话题保持角色绑定。"""
    source = prepare(manager)
    assert select(manager).successful
    source.tick(time.monotonic())
    assert source.operation is None
    assert source.sources['umi_camera']['actual'] == '/dev/video10'
    assert manager.components['umi_camera'].config['parameters']['video_device'] == '/dev/video10'
    command, _ = ComponentManager._command(manager, 'umi_camera')
    assert 'video_device:=/dev/video10' in command
    assert source.launch_topic('estimator') == '/umi_camera/image_raw'
    assert source.launch_topic('recorder') == '/wrist_camera/image_raw'


def test_swap_releases_both_before_restart(manager):
    """交换先释放两路及预测，所有启动完成后才确认新设备。"""
    source = prepare(manager)
    for key in ('umi_camera', 'wrist_camera', 'estimator'):
        manager._start(key)
    old = {k: manager.components[k].process for k in ('umi_camera', 'wrist_camera', 'estimator')}
    assert select(manager, device='/dev/video2').successful
    assert not select(manager).successful
    source.tick(time.monotonic())
    assert all(p.stop_started is not None for p in old.values())
    assert all(manager.components[k].process is p for k, p in old.items())
    advance(manager)
    assert source.operation is None
    assert source.sources['umi_camera']['actual'] == '/dev/video2'
    assert source.sources['wrist_camera']['actual'] == '/dev/video0'
    assert manager.get_parameter('wrist_video_device').value == '/dev/video0'
    assert all(manager.components[k].process is not p for k, p in old.items())


@pytest.mark.parametrize('state', ['recording', 'saving', 'unknown'])
@pytest.mark.parametrize('key', ['umi', 'wrist'])
def test_nonidle_blocks_both(manager, state, key):
    """录制、保存或状态失效时两路都拒绝切换。"""
    source = prepare(manager)
    manager._start('recorder')
    source._record(String(data=state))
    assert not select(manager, key).successful


@pytest.mark.parametrize('key', ['umi_camera', 'estimator', 'wrist_camera', 'recorder'])
def test_external_blocks_affected_role(manager, key):
    """仅监测或外部相机和消费节点不被重启。"""
    prepare(manager)
    manager.components[key].ownership = 'external'
    assert not select(manager, 'umi' if key in ('umi_camera', 'estimator') else 'wrist').successful


def test_failed_start_rolls_back(manager, monkeypatch):
    """新分配启动失败后恢复旧分配，保留失败原因。"""
    source = prepare(manager)
    manager._start('umi_camera')
    original = manager._command

    def command(key):
        """目标设备模拟不支持配置的采集模式。"""
        if key == 'umi_camera' and manager.components[key].config['parameters']['video_device'] == '/dev/video10':
            raise RuntimeError('不支持采集模式')
        return original(key)

    monkeypatch.setattr(manager, '_command', command)
    assert select(manager).successful
    for _ in range(4):
        advance(manager)
    assert source.operation is None
    assert source.sources['umi_camera']['actual'] == '/dev/video0'
    assert '已恢复' in source.sources['umi_camera']['error']
    assert manager.components['umi_camera'].process is not None


def test_shutdown_preempts_restart(manager):
    """关闭会话优先，已释放的相机不再自动恢复启动。"""
    source = prepare(manager)
    manager._start('umi_camera')
    assert select(manager).successful
    source.tick(time.monotonic())
    manager.request_shutdown()
    for _ in range(3):
        advance(manager)
    assert source.operation is None
    assert manager.components['umi_camera'].process is None
    assert manager.shutdown_complete


def test_pause_failure_keeps_original_camera(manager):
    """暂停未成功时不释放硬件，不自动恢复遥操。"""
    source = prepare(manager)
    manager._start('teleop')
    manager._start('umi_camera')
    process = manager.components['umi_camera'].process
    manager.pause_client = FakeClient(SetBool.Response(success=False))
    assert select(manager).successful
    source.tick(time.monotonic())
    assert source.operation is None
    assert process.stop_started is None


def test_catalog_filters_metadata_deduplicates_and_sorts(tmp_path, monkeypatch):
    """同物理相机只展示一个采集入口，元数据过滤，错误可见。"""
    for number in (10, 2, 3, 4, 5):
        (tmp_path / f'video{number}').touch()
    monkeypatch.setattr(camera_catalog, '_usb_location_from_video_device',
                        lambda path, *_: (1, 1 if path.endswith(('2', '3', '4')) else int(path.split('video')[-1])))
    real_open = camera_catalog.os.open
    descriptors = {}

    def open_device(path, flags):
        """模拟权限错误并保存描述符对应的节点名。"""
        if str(path).endswith('5'):
            raise PermissionError('权限不足')
        descriptor = real_open(path, flags)
        descriptors[descriptor] = str(path)
        return descriptor

    def query(descriptor, _request, buffer, _mutate):
        """模拟 QUERYCAP 的采集能力和元数据节点。"""
        buffer[16:19] = b'USB'
        struct.pack_into('II', buffer, 84, 0x80000000,
                         0x800000 if descriptors[descriptor].endswith('2') else 1)

    monkeypatch.setattr(camera_catalog.os, 'open', open_device)
    monkeypatch.setattr(camera_catalog.fcntl, 'ioctl', query)
    result = camera_catalog.scan_devices(tmp_path)
    assert [d['device'].split('/')[-1] for d in result] == ['video3', 'video5', 'video10']
    assert result[1]['error']


def test_empty_catalog_and_absent_selection(manager, tmp_path):
    """无设备可正常刷新，缺失设备请求不能被接受。"""
    assert camera_catalog.scan_devices(tmp_path) == []
    assert not select(manager).successful


def test_catalog_publishes_ros_byte_levels(manager):
    """通过真实 ROS 发布器序列化设备目录，覆盖 Jazzy 的 byte 类型等级。"""
    source = prepare(manager)
    source.scan = Future()
    source.scan.set_result(source.devices + [dict(device='/dev/video12', name='', error='权限不足')])
    source.tick(time.monotonic())
    assert source.scan is None
    assert source.devices[-1]['error'] == '权限不足'


class LockClient:
    """可控锁服务，用于验证先取得录制锁再释放相机。"""

    def __init__(self, failure='', delay=False):
        """保存锁响应策略及请求轨迹。"""
        self.failure, self.delay = failure, delay
        self.writes = []
        self.future = None

    def services_are_ready(self):
        """模拟已发现参数服务。"""
        return True

    def set_parameters_atomically(self, parameters):
        """根据测试要求延迟加锁，解锁始终返回完成。"""
        value = parameters[0].value
        self.writes.append(value)
        self.future = Future()
        if not value or not self.delay:
            self.future.set_result(SimpleNamespace(result=SimpleNamespace(
                successful=not self.failure, reason=self.failure)))
        return self.future


@pytest.mark.parametrize('failure', ['', '正在录制'])
def test_lock_acquired_before_release_and_always_released(manager, failure):
    """锁拒绝不释放设备，成功和失败路径均解除互锁。"""
    source = prepare(manager)
    manager._start('recorder')
    manager._start('umi_camera')
    source._record(String(data='idle'))
    client = LockClient(failure)
    source.client = client
    process = manager.components['umi_camera'].process
    assert select(manager).successful
    advance(manager)
    if failure:
        assert process.stop_started is None
    else:
        assert process.stop_started is not None
    client.failure = ''
    for _ in range(3):
        advance(manager)
    if source.unlock:
        source.tick(source.unlock['deadline'] + 0.1)
        source.tick(time.monotonic())
    assert client.writes[0] is True
    assert client.writes[-1] is False
    assert source.unlock is None


def test_lock_timeout_never_releases_camera(manager):
    """加锁响应超时仍请求解锁，防止已生效但迟到的响应留下锁。"""
    source = prepare(manager)
    manager._start('recorder')
    manager._start('umi_camera')
    source._record(String(data='idle'))
    source.client = LockClient(delay=True)
    process = manager.components['umi_camera'].process
    assert select(manager).successful
    now = time.monotonic()
    source.tick(now)
    source.tick(now+6)
    assert source.operation is None
    assert process.stop_started is None
    source.tick(now+6.1)
    source.tick(now+6.2)
    assert source.client.writes == [True, False]
    assert source.unlock is None


def test_verify_timeout_restores_old_assignment(manager):
    """进程存在但没有有效图像也判定切换失败并尝试恢复。"""
    source = prepare(manager)
    manager._start('umi_camera')
    assert select(manager).successful
    advance(manager, healthy=False)
    assert source.operation['phase'] == 'verify'
    source.tick(source.operation['deadline']+1)
    assert source.operation['rollback']
    advance(manager)
    assert source.operation is None
    assert source.sources['umi_camera']['actual'] == '/dev/video0'


def test_restore_failure_stops_new_processes(manager, monkeypatch):
    """原设备也不可用时明确报告恢复失败并留下停止状态。"""
    source = prepare(manager)
    manager._start('umi_camera')

    def fail(_key):
        """所有后续启动均失败。"""
        raise RuntimeError('设备已拔出')

    monkeypatch.setattr(manager, '_command', fail)
    assert select(manager).successful
    for _ in range(4):
        advance(manager)
    assert source.operation is None
    assert '恢复失败' in source.sources['umi_camera']['error']
    assert manager.components['umi_camera'].process is None


def test_external_device_readback_and_no_writes(manager):
    """仅监测相机读取实际设备参数，不向外部节点发出设置请求。"""
    source = prepare(manager)
    manager.components['umi_camera'].ownership = 'external'

    class Reader:
        """只提供参数查询接口的外部相机。"""

        def services_are_ready(self):
            """外部相机参数接口已就绪。"""
            return True

        def get_parameters(self, names):
            """返回实际使用的设备路径。"""
            assert names == ['video_device']
            future = Future()
            future.set_result(SimpleNamespace(values=[SimpleNamespace(string_value='/dev/video10')]))
            return future

    source.observers['umi_camera']['client'] = Reader()
    source.tick(time.monotonic())
    source.tick(time.monotonic())
    assert source.fields('umi_camera')['video_device'] == '/dev/video10'
    assert source.fields('umi_camera')['source_editable'] == 'false'
    assert not select(manager).successful


def test_swap_rejects_missing_return_device(manager):
    """当前设备拔出后保留原选择，但不能交换成无效分配。"""
    source = prepare(manager)
    source.devices = [d for d in source.devices if d['device'] != '/dev/video0']
    assert not select(manager, device='/dev/video2').successful
    assert source.sources['umi_camera']['actual'] == '/dev/video0'
    assert source.operation is None
