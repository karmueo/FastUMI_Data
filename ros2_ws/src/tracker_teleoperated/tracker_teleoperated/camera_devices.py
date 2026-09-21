"""协调设备目录、录制互锁和可恢复的双相机设备交换。"""

from concurrent.futures import ThreadPoolExecutor
import re
import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from tracker_teleoperated.camera_catalog import scan_devices


class CameraDevices:
    """单线程推进硬件事务，设备扫描单独后台执行，不阻塞 ROS 心跳。"""

    def __init__(self, manager, qos):
        """创建目录发布器及参数接口；构造时不启动硬件。"""
        self.manager = manager
        self.record_state = ''
        self.record_seen = 0.0
        self.devices = []
        self.operation = None
        self._syncing = False
        self.sources = {}
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.scan = None
        self.next_scan = 0.0
        self.unlock = None
        self.observers = {}
        self.client = AsyncParameterClient(manager, '/tracker_teleop_recorder')
        self.publisher = manager.create_publisher(DiagnosticArray, '/tracker_teleoperated/camera_devices', qos)
        manager.create_service(Trigger, '/tracker_teleoperated/refresh_camera_devices', self.refresh)
        manager.create_subscription(String, '/tracker_teleoperated/record_state', self._record, qos)
        for key, default in (('umi_camera', '/dev/video0'), ('wrist_camera', '/dev/video2')):
            parameter = key.replace('_camera', '_video_device')
            config = manager.components[key].config.setdefault('parameters', {})
            device = manager.declare_parameter(parameter, config.get('video_device', default)).value
            if not re.fullmatch(r'/dev/video\d+', device):
                raise ValueError(f'无效相机设备: {device}')
            config['video_device'] = device
            self.sources[key] = dict(parameter=parameter, actual=device, requested=device,
                                     state='deferred', error='')
            nodes = manager.components[key].config.get('nodes', [])
            self.observers[key] = dict(client=AsyncParameterClient(manager, nodes[0] if nodes else
                '/' + key + '/usb_camera_node'), future=None, next_query=0.0)
        manager.add_on_set_parameters_callback(self._request)

    def close(self):
        """回收后台扫描线程；不会留下访问 ROS 对象的后台回调。"""
        self.pool.shutdown(wait=True, cancel_futures=True)

    def refresh(self, _request, response):
        """合并手动刷新意图，下一次 tick 调度扫描。"""
        self.next_scan = 0.0
        response.success, response.message = True, '已请求刷新设备'
        return response

    def _record(self, message):
        """记录业务状态；过期状态不允许启动硬件切换。"""
        self.record_state, self.record_seen = message.data, time.monotonic()

    def reason(self, key):
        """检查事务、录制状态和相机及消费节点归属。"""
        manager = self.manager
        if manager.operation or self.operation or self.unlock:
            return '正在切换、停止或解除录制锁'
        for name in (key, 'estimator' if key == 'umi_camera' else 'recorder'):
            component = manager.components[name]
            if component.config.get('mode', 'auto') != 'auto' or component.ownership == 'external':
                return f'{name} 仅监测或禁用，不能切换设备'
            if component.ownership != 'local' and manager._external_present(component):
                return f'{name} 是外部节点，不能切换设备'
        recorder = manager.components['recorder']
        if recorder.ownership == 'external' or recorder.config.get('mode') == 'observe':
            return '记录节点仅监测，无法取得录制锁'
        if recorder.ownership != 'local' and manager._external_present(recorder):
            return '检测到外部记录节点，无法取得录制锁'
        teleop = manager.components['teleop']
        if teleop.ownership == 'external' or teleop.config.get('mode') == 'observe':
            return '遥操节点仅监测，无法确认安全暂停'
        if teleop.ownership != 'local' and manager._external_present(teleop):
            return '检测到外部遥操节点，无法确认安全暂停'
        if recorder.process and (self.record_state != 'idle' or time.monotonic() - self.record_seen > 2):
            return '仅录制空闲且状态有效时允许切换两路设备'
        return ''

    def _request(self, parameters):
        """接受单路设备请求，另一角色已占用时生成一次整体交换事务。"""
        if self._syncing:
            return SetParametersResult(successful=True)
        selected = [(key, p) for key, source in self.sources.items()
                    for p in parameters if p.name == source['parameter']]
        if not selected:
            return SetParametersResult(successful=True)
        try:
            if len(parameters) != 1:
                raise ValueError('请一次选择一个设备；设备交换由管理器协调')
            key, parameter = selected[0]
            target = parameter.value
            if not isinstance(target, str) or not re.fullmatch(r'/dev/video\d+', target):
                raise ValueError('请选择 /dev/videoN 设备')
            reason = self.reason(key)
            if reason:
                raise ValueError(reason)
            if target not in [d['device'] for d in self.devices if not d['error']]:
                raise ValueError('设备不存在、不可访问或不是有效采集入口')
            old = {k: s['actual'] for k, s in self.sources.items()}
            new = dict(old)
            other = 'wrist_camera' if key == 'umi_camera' else 'umi_camera'
            new[key] = target
            if target == old[other] and target != old[key]:
                reason = self.reason(other)
                if reason:
                    raise ValueError('无法交换：' + reason)
                new[other] = old[key]
            changed = [k for k in new if new[k] != old[k]]
            if not changed:
                return SetParametersResult(successful=True)
            available = {d['device'] for d in self.devices if not d['error']}
            if any(new[k] not in available for k in changed):
                raise ValueError('交换后的设备不存在或不可访问，未更改分配')
            targets = list(changed)
            if 'umi_camera' in changed:
                targets.insert(0, 'estimator')
            running = [k for k in targets if self.manager._owned_alive(k)]
            self.operation = dict(phase='lock', old=old, new=new, changed=changed,
                                  running=running, future=None, deadline=time.monotonic()+5,
                                  rollback=False, failure='', locked=False)
            for k in changed:
                self.sources[k].update(requested=new[k], state='pending', error='')
            return SetParametersResult(successful=True)
        except ValueError as error:
            return SetParametersResult(successful=False, reason=str(error))

    def launch_topic(self, key):
        """消费节点始终订阅对应角色的输出，设备切换不改变话题。"""
        camera = 'umi_camera' if key == 'estimator' else 'wrist_camera'
        return self.manager.probes[camera][0]

    def _release(self):
        """异步解除锁；失败时继续重试并通过诊断保持不可操作。"""
        if self.manager._owned_alive('recorder'):
            self.unlock = dict(future=None, deadline=0.0)

    def cancel(self):
        """停止优先：取消恢复和启动意图，解除已发出的录制锁。"""
        if self.operation:
            for key in self.operation['changed']:
                self.sources[key].update(state='error', error='停止操作已中止设备切换',
                    actual=self.manager.components[key].config['parameters']['video_device'])
            self._stop(self.operation['running'])
            self.operation = None
            self._release()

    def _stop(self, keys):
        """只停止本会话拥有的进程，由管理器轮询回收整棵进程树。"""
        for key in keys:
            component = self.manager.components[key]
            if self.manager._owned_alive(key):
                component.state = 'stopping'
                component.process.stop()

    def _finish(self, error=''):
        """发布最终结果并异步释放录制互锁。"""
        op = self.operation
        for key in op['changed']:
            self.sources[key].update(state='error' if error else (
                'ready' if self.manager.components[key].process else 'deferred'), error=error)
        self.manager.last_operation = error or '设备切换完成，遥操保持暂停'
        self.operation = None
        # 整体交换也同步另一角色的请求参数；不会触发第二次硬件事务。
        self._syncing = True
        try:
            self.manager.set_parameters_atomically([
                Parameter(source['parameter'], value=source['requested']) for source in self.sources.values()])
        finally:
            self._syncing = False
        self._release()

    def _fail(self, reason, now):
        """释放失败启动的所有进程；只在正常会话中尝试一次恢复。"""
        op = self.operation
        self._stop(op['running'])
        if op['rollback']:
            for key in op['changed']:
                self.sources[key]['actual'] = self.manager.components[key].config['parameters']['video_device']
            op.update(phase='abort', failure=op['failure'] + '；恢复失败：' + reason)
        else:
            op.update(rollback=True, failure=reason, phase='release', deadline=now+15)

    def tick(self, now):
        """推进扫描、锁确认、暂停、释放、启动与图像确认，无同步等待。"""
        for key, observer in self.observers.items():
            if self.manager.components[key].ownership != 'external':
                continue
            future = observer['future']
            if future and future.done():
                try:
                    device = future.result().values[0].string_value
                    if not re.fullmatch(r'/dev/video\d+', device):
                        raise ValueError('外部相机未提供有效 video_device 参数')
                    self.sources[key].update(actual=device, requested=device, state='ready')
                except Exception as error:
                    self.sources[key]['error'] = '读取外部相机参数失败：' + str(error)
                observer.update(future=None, next_query=now+2)
            elif now >= observer['next_query'] and observer['client'].services_are_ready():
                if future:
                    future.cancel()
                observer.update(future=observer['client'].get_parameters(['video_device']), next_query=now+3)
        if self.scan is not None and self.scan.done():
            message = DiagnosticArray()
            try:
                self.devices = self.scan.result()
                for device in self.devices:
                    message.status.append(DiagnosticStatus(name=device['device'], message=device['error'],
                        level=DiagnosticStatus.ERROR if device['error'] else DiagnosticStatus.OK,
                        values=[KeyValue(key=k, value=v) for k, v in device.items()]))
            except Exception as error:
                message.status.append(DiagnosticStatus(name='scan', level=DiagnosticStatus.ERROR, message=str(error)))
            self.publisher.publish(message)
            self.scan = None
        if now >= self.next_scan and self.scan is None:
            self.scan = self.pool.submit(scan_devices)
            self.next_scan = now + 2
        if self.unlock:
            future = self.unlock['future']
            if not self.manager._owned_alive('recorder'):
                self.unlock = None
            elif future and future.done():
                try:
                    if future.result().result.successful:
                        self.unlock = None
                    else:
                        self.unlock['future'] = None
                except Exception:
                    self.unlock['future'] = None
            elif now >= self.unlock['deadline'] and self.client.services_are_ready():
                if future:
                    future.cancel()
                self.unlock.update(future=self.client.set_parameters_atomically([
                    Parameter('camera_switch_locked', value=False)]), deadline=now+3)
        op = self.operation
        if not op or self.manager.operation:
            return
        try:
            if op['phase'] == 'abort':
                if not any(self.manager.components[k].process for k in op['running']):
                    self._finish(op['failure'])
                return
            if op['phase'] == 'lock':
                if self.manager._owned_alive('recorder'):
                    if op['future'] is None and self.client.services_are_ready():
                        op['future'] = self.client.set_parameters_atomically([Parameter('camera_switch_locked', value=True)])
                    if not op['future'] or not op['future'].done():
                        if now > op['deadline']:
                            self._finish('录制锁确认超时，未切换设备')
                        return
                    result = op['future'].result().result
                    if not result.successful:
                        self._finish('无法锁定录制：' + result.reason)
                        return
                op.update(phase='pause', future=None, deadline=now+3)
            if op['phase'] == 'pause':
                if self.manager._owned_alive('teleop'):
                    if op['future'] is None and self.manager.pause_client.service_is_ready():
                        op['future'] = self.manager.pause_client.call_async(SetBool.Request(data=False))
                    if not op['future'] or not op['future'].done():
                        if now > op['deadline']:
                            self._finish('暂停遥操超时，未切换设备')
                        return
                    if not op['future'].result().success:
                        self._finish('暂停遥操失败，未切换设备')
                        return
                self._stop(op['running'])
                op.update(phase='release', deadline=now+15)
            if op['phase'] == 'release':
                if any(self.manager.components[k].process for k in op['running']):
                    if now > op['deadline']:
                        self._fail('释放设备超时', now)
                    return
                allocation = op['old'] if op['rollback'] else op['new']
                for key in op['changed']:
                    self.manager.components[key].config['parameters']['video_device'] = allocation[key]
                op.update(phase='start', deadline=now+float(self.manager.config.get('startup_grace_s', 30)))
            if op['phase'] == 'start':
                for key in op['running']:
                    if self.manager.components[key].process:
                        continue
                    ok, error = self.manager._start(key)
                    if not ok:
                        if '旧节点' in error and now < op['deadline']:
                            return
                        self._fail(error, now)
                        return
                op['phase'] = 'verify'
            if op['phase'] == 'verify':
                for key in op['running']:
                    component = self.manager.components[key]
                    if component.process is None:
                        self._fail(key + ' 启动后退出', now)
                        return
                    if key.endswith('_camera') and not (component.health.samples and component.health.check()[0]):
                        if now > op['deadline']:
                            self._fail(key + ' 未收到有效图像，请检查采集模式和日志', now)
                        return
                allocation = op['old'] if op['rollback'] else op['new']
                for key in op['changed']:
                    self.sources[key]['actual'] = allocation[key]
                self._finish(op['failure'] + '；已恢复原设备分配' if op['rollback'] else '')
        except Exception as error:
            if op['phase'] in ('lock', 'pause'):
                self._finish('切换准备失败，未释放相机：' + str(error))
            else:
                self._fail(str(error), now)

    def fields(self, key):
        """输出设备、固定角色话题及只读原因，供所有面板同步。"""
        source = self.sources[key]
        state = source['state']
        if state == 'deferred' and self.manager.components[key].process:
            state = 'ready'
        return dict(requested_video_device=source['requested'], video_device=source['actual'],
                    source_state=state, source_error=source['error'],
                    switch_phase=self.operation['phase'] if self.operation else ('unlock' if self.unlock else ''),
                    source_editable=str(not self.reason(key)).lower(), source_reason=self.reason(key),
                    image_topic=self.manager.probes[key][0],
                    record_camera=str(bool(self.manager.record.get('record_camera', True))).lower())
