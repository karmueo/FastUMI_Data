"""验证独立录制事务、持久化、时间对齐和退出边界。"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
from types import SimpleNamespace

import h5py
import pytest

from fastumi_recorder.catalog import Catalog, RecordingError
from fastumi_recorder.engine import (
    RecorderEngine, ffmpeg_packet_to_h264, image_to_bgr24, image_to_jpeg,
)
from fastumi_recorder.request_ledger import RequestLedger
from fastumi_recorder.core import CameraFrameSpool, EpisodeBuffer
from fastumi_recorder.storage import select_video_samples
import fastumi_recorder.engine as engine_module


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def ready(engine):
    engine.add('joint_state', 10.0, [0.] * 7)
    engine.add('gripper_state', 10.0, 1.)
    engine.image(10.0, frame())


def frame():
    return SimpleNamespace(width=16, height=16, step=48, encoding='rgb8', data=bytes(16*16*3))


def test_compressed_jpeg_is_forwarded_without_reencoding():
    jpeg = b'\xff\xd8native-camera-jpeg\xff\xd9'
    message = SimpleNamespace(format='jpeg', data=jpeg)
    assert image_to_jpeg(message) is jpeg


@pytest.mark.parametrize('encoding, data, expected', [
    ('bgr8', bytes([1, 2, 3, 4, 5, 6, 99, 99]), bytes([1, 2, 3, 4, 5, 6])),
    ('rgb8', bytes([1, 2, 3, 4, 5, 6, 99, 99]), bytes([3, 2, 1, 6, 5, 4])),
    ('mono8', bytes([7, 8, 99, 99]), bytes([7, 7, 7, 8, 8, 8])),
])
def test_raw_image_colors_and_row_padding(encoding, data, expected):
    step = 8 if encoding != 'mono8' else 4
    message = SimpleNamespace(width=2, height=1, step=step,
                              encoding=encoding, data=data)
    assert image_to_bgr24(message) == (expected, (2, 1))


def test_raw_padding_is_skipped_between_rows():
    message = SimpleNamespace(width=1, height=2, step=5, encoding='bgr8',
                              data=bytes([1, 2, 3, 99, 99, 4, 5, 6, 88, 88]))
    assert image_to_bgr24(message) == (bytes([1, 2, 3, 4, 5, 6]), (1, 2))


def test_raw_frame_saves_decodable_mp4_without_jpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_module, 'image_to_jpeg', lambda _: pytest.fail('raw 调用了 JPEG'))
    instance = RecorderEngine(tmp_path, image_transport='raw')
    try:
        ready(instance)
        rid, _ = instance.start(timestamp=11.)
        # 16x16 纯红、纯蓝实帧；第二帧带行填充。
        red = bytes([0, 0, 255]) * 16
        blue = bytes([255, 0, 0]) * 16
        instance.image(11.2, SimpleNamespace(
            width=16, height=16, step=48, encoding='bgr8', data=red * 16))
        instance.image(11.3, SimpleNamespace(
            width=16, height=16, step=50, encoding='bgr8',
            data=(blue + b'xx') * 16))
        instance.stop(rid, 12.)
        wait_for(lambda: instance.state != 'saving')
        assert instance.state == 'idle', instance.last_error
        path = tmp_path / instance.list()[0][0]['relative_path']
        with h5py.File(path / 'proprio.hdf5') as data:
            assert data['observations/images/cam_gripper_timestamp'][:].tolist() == [11.2, 11.3]
        probe = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,nb_frames',
            '-of', 'default=noprint_wrappers=1', str(path / 'gripper.mp4'),
        ], check=True, capture_output=True, text=True).stdout
        assert 'codec_name=h264' in probe and 'nb_frames=2' in probe
        decoded = subprocess.run([
            'ffmpeg', '-v', 'error', '-i', str(path / 'gripper.mp4'),
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'
        ], check=True, capture_output=True).stdout
        assert len(decoded) == 2 * 16 * 16 * 3
        assert decoded[2] > 200 and decoded[0] < 60
        assert decoded[16 * 16 * 3] > 200 and decoded[16 * 16 * 3 + 2] < 60
    finally:
        instance.close()


def test_raw_size_change_drops_frame(tmp_path):
    instance = RecorderEngine(tmp_path, image_transport='raw')
    try:
        ready(instance)
        rid, _ = instance.start(timestamp=11.)
        instance.image(11.2, frame())
        instance.image(11.3, SimpleNamespace(
            width=32, height=16, step=96, encoding='rgb8', data=bytes(32*16*3)))
        instance.stop(rid, 12.)
        wait_for(lambda: instance.state != 'saving')
        assert instance.state == 'idle', instance.last_error
        assert instance.status()['dropped_frames'] == 1
        path = tmp_path / instance.list()[0][0]['relative_path']
        with h5py.File(path / 'proprio.hdf5') as data:
            assert len(data['observations/images/cam_gripper_timestamp']) == 1
    finally:
        instance.close()


def test_raw_spool_uses_dataset_disk_and_failure_blocks_publish(tmp_path, monkeypatch):
    instance = RecorderEngine(tmp_path, image_transport='raw')
    try:
        ready(instance)
        rid, _ = instance.start(timestamp=11.)
        spool = instance.session._episode.images
        spool._file.rollover()
        assert os.fstat(spool._file.fileno()).st_dev == os.stat(instance._temporary).st_dev
        real_append = CameraFrameSpool.append
        def fail_append(self, *args, **kwargs):
            if self is spool:
                raise OSError('spool full')
            return real_append(self, *args, **kwargs)
        monkeypatch.setattr(CameraFrameSpool, 'append', fail_append)
        instance.image(11.2, frame())
        wait_for(lambda: 'spool full' in instance.status()['last_error'])
        instance.stop(rid, 12.)
        wait_for(lambda: instance.state == 'error')
        assert instance.list()[1] == 0
        assert (instance._temporary / 'error.json').exists()
    finally:
        instance.close()


def test_ffmpeg_packet_validation_and_keyframe_flag():
    packet = SimpleNamespace(
        encoding='h264;yuv420p;bgr8;bgr8', width=1280, height=960,
        flags=1, data=b'h264-packet')
    assert ffmpeg_packet_to_h264(packet) == (b'h264-packet', True)
    packet.encoding = 'hevc;yuv420p;bgr8;bgr8'
    with pytest.raises(ValueError, match='编码'):
        ffmpeg_packet_to_h264(packet)


def test_h264_window_starts_at_first_keyframe():
    episode = EpisodeBuffer(started_at=1.0, stopped_at=3.0)
    try:
        episode.images.append(1.1, b'p0', codec='h264', keyframe=False)
        episode.images.append(1.2, b'i1', codec='h264', keyframe=True)
        episode.images.append(1.3, b'p2', codec='h264', keyframe=False)
        selected = list(select_video_samples(episode, (1.0, 2.0)))
        assert [item[1] for item in selected] == [b'i1', b'p2']
        assert list(select_video_samples(episode, (1.25, 2.0))) == []
    finally:
        episode.close()


def test_h264_packet_is_stream_copied_to_playable_mp4(tmp_path):
    """真实 H.264 包应无重编码封装，且时间戳数量与 MP4 帧数一致。"""
    elementary = tmp_path / 'one.h264'
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
        '-i', 'color=size=160x120:rate=30:color=black', '-frames:v', '1',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
        '-profile:v', 'baseline', '-pix_fmt', 'yuv420p', '-f', 'h264',
        str(elementary),
    ], check=True)
    packet = SimpleNamespace(
        encoding='h264;yuv420p;bgr8;bgr8', width=160, height=120,
        flags=1, data=elementary.read_bytes())
    root = tmp_path / 'dataset'
    instance = RecorderEngine(root, image_transport='ffmpeg')
    try:
        instance.add('joint_state', 10.0, [0.] * 7)
        instance.add('gripper_state', 10.0, 1.)
        instance.image(10.0, packet)
        recording_id, _ = instance.start(timestamp=11.0)
        instance.image(11.2, packet)
        instance.stop(recording_id, 12.0)
        wait_for(lambda: instance.state != 'saving')
        assert instance.state == 'idle', instance.last_error
        output = root / instance.list()[0][0]['relative_path']
        with h5py.File(output / 'proprio.hdf5') as data:
            assert data['observations/images/cam_gripper_timestamp'][:].tolist() == [11.2]
        probe = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,nb_frames',
            '-of', 'default=noprint_wrappers=1', str(output / 'gripper.mp4'),
        ], check=True, capture_output=True, text=True).stdout
        assert 'codec_name=h264' in probe
        assert 'nb_frames=1' in probe
    finally:
        instance.close(timestamp=13.0)


@pytest.fixture
def engine(tmp_path):
    instance = RecorderEngine(tmp_path, image_transport='raw')
    yield instance
    instance.close(timestamp=30.)


def record(engine, *, actions=False):
    ready(engine)
    rid, _ = engine.start(timestamp=11.)
    engine.add('joint_state', 11.2, [0.] * 7)
    engine.add('gripper_state', 11.2, .5)
    engine.image(11.2, frame())
    if actions:
        engine.add('joint_action', 11.5, [1.] * 7)
        engine.add('joint_state', 11.6, [1.] * 7)
        engine.image(11.6, frame())
    engine.stop(rid, 12.)
    wait_for(lambda: engine.state != 'saving')
    assert engine.state == 'idle', engine.last_error
    return rid


def test_static_recording_and_restart(engine, tmp_path):
    rid = record(engine)
    info = engine.list()[0][0]
    path = tmp_path / info['relative_path']
    with h5py.File(path / 'proprio.hdf5') as data:
        assert data['action/joint_action/position'].shape == (0, 7)
        assert data['observations/tracker_pose/position'].shape == (0, 3)
        assert data.attrs['alignment_reference'] == 'recording_window'
        assert not data.attrs['has_joint_action']
        assert len(data['observations/images/cam_gripper_timestamp']) == 1
    assert (path / 'gripper.mp4').stat().st_size > 0
    assert engine.stop(rid)[1] == 'ALREADY_SAVED'
    engine.close()
    other = RecorderEngine(tmp_path, image_transport='raw')
    try:
        assert other.list()[0][0]['recording_id'] == rid
        other.delete(rid)
        assert other.list()[1] == 0
        assert (tmp_path / '.trash' / rid / 'proprio.hdf5').exists()
        assert other.delete(rid)[1] == 'ALREADY_DELETED'
        next_id = record(other)
        assert other.list()[0][0]['relative_path'].endswith('episode_1')
        assert next_id != rid
    finally:
        other.close()


def test_action_alignment(engine, tmp_path):
    record(engine, actions=True)
    path = tmp_path / engine.list()[0][0]['relative_path']
    with h5py.File(path / 'proprio.hdf5') as data:
        assert data.attrs['alignment_reference'] == 'joint_action'
        assert data['action/joint_action/timestamp'][:].tolist() == [11.5, 12.]
        assert data['observations/joint_state/timestamp'][:].tolist() == [11.6]
        assert len(data['observations/images/cam_gripper_timestamp']) == 1


def test_readiness_and_stale_inputs(engine):
    with pytest.raises(RecordingError, match='joint, gripper, image'):
        engine.start()
    ready(engine)
    engine._seen['joint'] -= 3.
    with pytest.raises(RecordingError, match='joint'):
        engine.start()


def test_concurrent_start_cancel_and_old_id(engine):
    ready(engine)
    def start():
        try:
            return engine.start(timestamp=11.)[0]
        except RecordingError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=8) as pool:
        result = list(pool.map(lambda _: start(), range(8)))
    assert result.count('BUSY') == 7
    rid = next(item for item in result if item != 'BUSY')
    with pytest.raises(RecordingError):
        engine.delete(rid)
    engine.cancel(rid)
    assert engine.cancel(rid)[1] == 'ALREADY_CANCELLED'
    new_id, _ = engine.start(timestamp=12.)
    with pytest.raises(RecordingError):
        engine.stop(rid)
    assert engine.state == 'recording'
    engine.cancel(new_id)


def test_busy_during_save_and_duplicate_stop(engine, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real_write = engine_module.write_episode
    def blocked(*args):
        entered.set()
        assert release.wait(3.)
        return real_write(*args)
    monkeypatch.setattr(engine_module, 'write_episode', blocked)
    ready(engine)
    rid, _ = engine.start(timestamp=11.)
    engine.stop(rid, 12.)
    assert entered.wait(2.)
    try:
        assert engine.stop(rid)[1] == 'ALREADY_STOPPING'
        for operation in (lambda: engine.start(), lambda: engine.cancel(rid), lambda: engine.delete(rid)):
            with pytest.raises(RecordingError) as error:
                operation()
            assert error.value.code == 'BUSY'
        assert engine.status()['state'] == 'saving'
    finally:
        release.set()
    wait_for(lambda: engine.state == 'idle')


def test_request_identity_retry_cancel_first_and_restart(tmp_path):
    first = RecorderEngine(tmp_path, image_transport='raw')
    request_id, blocked_id = str(uuid.uuid4()), str(uuid.uuid4())
    try:
        assert first.cancel_request(blocked_id)[2] == 'cancelled'
        with pytest.raises(RecordingError) as error:
            first.start(request_id=blocked_id)
        assert error.value.code == 'CANCELLED'
        ready(first)
        recording_id, _ = first.start(timestamp=11., request_id=request_id)
        assert first.start(timestamp=12., request_id=request_id)[0] == recording_id
        assert first.get_request(request_id)['state'] == 'recording'
        assert first.cancel_request(request_id)[2] == 'cancelled'
        assert first.start(request_id=request_id)[0] == recording_id
    finally:
        first.close()
    second = RecorderEngine(tmp_path, image_transport='raw')
    try:
        assert second.get_request(blocked_id)['state'] == 'cancelled'
        assert second.start(request_id=request_id)[0] == recording_id
        assert second.get_request(request_id)['state'] == 'cancelled'
    finally:
        second.close()


def test_cancel_cleanup_failure_does_not_report_cancelled(engine, monkeypatch):
    request_id = str(uuid.uuid4())
    ready(engine)
    recording_id, _ = engine.start(timestamp=11., request_id=request_id)
    temporary = engine._temporary
    real_rmtree = engine_module.shutil.rmtree

    def fail_cleanup(path, *args, **kwargs):
        if path == temporary:
            raise OSError('cleanup denied')
        return real_rmtree(path, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(engine_module.shutil, 'rmtree', fail_cleanup)
        with pytest.raises(OSError, match='cleanup denied'):
            engine.cancel(recording_id)

    request = engine.get_request(request_id)
    assert request['state'] == 'failed'
    assert request['code'] == 'IO_ERROR'
    assert temporary.exists()
    assert engine.cancel_request(request_id)[2] == 'failed'


def test_ledger_restart_marks_interrupted_and_finishes_cancel(tmp_path):
    from fastumi_recorder.catalog import Catalog
    catalog = Catalog(tmp_path)
    pending_id, cancel_id = str(uuid.uuid4()), str(uuid.uuid4())
    recording_id = str(uuid.uuid4())
    temporary = tmp_path / 'task' / 'name' / f'.recording-{recording_id}'
    temporary.mkdir(parents=True)
    ledger = RequestLedger(tmp_path)
    try:
        ledger.put(pending_id, state='pending')
        ledger.put(cancel_id, state='saving', recording_id=recording_id,
                   cancel_requested=1, start_success=1, start_code='STARTED',
                   start_message='STARTED')
    finally:
        ledger.close()
    reopened = RequestLedger(tmp_path)
    try:
        reopened.recover(catalog)
        assert reopened.get(pending_id)['state'] == 'failed'
        assert reopened.get(pending_id)['start_code'] == 'INTERRUPTED'
        assert reopened.get(cancel_id)['state'] == 'cancelled'
        assert not temporary.exists()
    finally:
        reopened.close()
        catalog.close()
    restarted = RecorderEngine(tmp_path, image_transport='raw')
    try:
        with pytest.raises(RecordingError) as error:
            restarted.start(request_id=pending_id)
        assert error.value.code == 'INTERRUPTED'
    finally:
        restarted.close()


def test_request_cancel_during_save_discards_before_publish(engine, monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()
    real_write = engine_module.write_episode
    def blocked(*args):
        entered.set()
        assert release.wait(3.)
        return real_write(*args)
    monkeypatch.setattr(engine_module, 'write_episode', blocked)
    request_id = str(uuid.uuid4())
    ready(engine)
    rid, _ = engine.start(timestamp=11., request_id=request_id)
    engine.stop(rid, 12.)
    assert entered.wait(2.)
    assert engine.cancel_request(request_id)[1] == 'CANCEL_ACCEPTED'
    release.set()
    wait_for(lambda: engine.get_request(request_id)['state'] == 'cancelled')
    assert engine.list()[1] == 0
    assert not any(tmp_path.rglob('episode_*'))


def test_publish_commit_wins_late_cancel(engine, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real_publish = engine.catalog.publish
    def blocked_publish(*args):
        entered.set()
        assert release.wait(3.)
        return real_publish(*args)
    monkeypatch.setattr(engine.catalog, 'publish', blocked_publish)
    request_id = str(uuid.uuid4())
    ready(engine)
    rid, _ = engine.start(timestamp=11., request_id=request_id)
    engine.stop(rid, 12.)
    assert entered.wait(2.)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending_cancel = pool.submit(engine.cancel_request, request_id)
        release.set()
        assert pending_cancel.result(timeout=3.)[2] == 'completed'
    assert engine.list()[1] == 1


def test_cancel_kills_active_encoder(engine, monkeypatch, tmp_path):
    registered, release = threading.Event(), threading.Event()
    processes = []
    real_register = engine.encoder.register
    def blocked_register(process):
        real_register(process)
        processes.append(process)
        registered.set()
        assert release.wait(3.)
    monkeypatch.setattr(engine.encoder, 'register', blocked_register)
    request_id = str(uuid.uuid4())
    ready(engine)
    rid, _ = engine.start(timestamp=11., request_id=request_id)
    engine.image(11.2, frame())
    engine.stop(rid, 12.)
    assert registered.wait(2.)
    try:
        assert engine.cancel_request(request_id)[1] == 'CANCEL_ACCEPTED'
    finally:
        release.set()
    wait_for(lambda: engine.get_request(request_id)['state'] == 'cancelled')
    assert all(process.poll() is not None for process in processes)
    assert engine.list()[1] == 0
    assert not any(tmp_path.rglob('episode_*'))


def test_failed_save_preserves_temporary(engine, monkeypatch):
    monkeypatch.setattr(engine_module, 'write_episode', lambda *args: (_ for _ in ()).throw(OSError('disk full')))
    ready(engine)
    rid, _ = engine.start(timestamp=11.)
    engine.stop(rid, 12.)
    wait_for(lambda: engine.state == 'error')
    assert engine.list()[1] == 0
    assert 'disk full' in engine.status()['last_error']
    assert (engine._temporary / 'error.json').exists()
    with pytest.raises(RecordingError) as error:
        engine.stop(rid)
    assert error.value.code == 'SAVE_FAILED'
    ready(engine)
    rid, _ = engine.start(timestamp=13.)
    engine.cancel(rid)


@pytest.mark.parametrize('directory', ['..', '../escape', '/tmp/escape', 'a/b', 'a\\b', '.hidden'])
def test_bad_task_paths(engine, directory):
    ready(engine)
    with pytest.raises(RecordingError):
        engine.start(directory, 'task', timestamp=11.)


def test_symlink_and_root_lock(engine, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + '-outside')
    outside.mkdir()
    (tmp_path / 'linked').symlink_to(outside, target_is_directory=True)
    ready(engine)
    with pytest.raises(RecordingError):
        engine.start('linked', 'task', 11.)
    with pytest.raises(RecordingError, match='占用'):
        Catalog(tmp_path)
    with pytest.raises(RecordingError):
        engine.delete('../outside')


def test_close_saves_active_recording(engine):
    ready(engine)
    engine.start(timestamp=11.)
    engine.add('joint_state', 11.1, [0.] * 7)
    assert engine.close(timestamp=12.)
    assert engine.list()[1] == 1


def test_close_timeout_aborts_encoder(engine, monkeypatch):
    engine.shutdown_timeout = .02
    entered = threading.Event()
    def blocked(*args):
        entered.set()
        wait_for(lambda: engine.encoder._aborted)
        raise RuntimeError('aborted')
    monkeypatch.setattr(engine_module, 'write_episode', blocked)
    ready(engine)
    rid, _ = engine.start(timestamp=11.)
    engine.stop(rid, 12.)
    assert entered.wait(1.)
    assert not engine.close()
    assert not engine._worker.is_alive()
    assert engine.state == 'error'
    assert engine.list()[1] == 0


def test_cancelled_image_cannot_leak_into_next_episode(engine, monkeypatch, tmp_path):
    """取消时已排队的编码回调不能污染下一轮录制。"""
    entered, release = threading.Event(), threading.Event()
    real_encode = engine_module.image_to_bgr24
    def delayed(message):
        entered.set()
        assert release.wait(3.)
        return real_encode(message)
    monkeypatch.setattr(engine_module, 'image_to_bgr24', delayed)
    ready(engine)
    first, _ = engine.start(timestamp=11.)
    engine.image(11.1, frame())
    assert entered.wait(2.)
    try:
        engine.cancel(first)
        second, _ = engine.start(timestamp=12.)
        engine.image(12.1, frame())
        engine.stop(second, 13.)
    finally:
        release.set()
    wait_for(lambda: engine.state == 'idle')
    path = tmp_path / engine.list()[0][0]['relative_path']
    with h5py.File(path / 'proprio.hdf5') as data:
        assert data['observations/images/cam_gripper_timestamp'][:].tolist() == [12.1]


def test_delete_rejects_changed_symlink(engine, tmp_path):
    """回收时重新检查登记路径，拒绝随后替换的符号链接。"""
    rid = record(engine)
    path = tmp_path / engine.list()[0][0]['relative_path']
    (path / 'outside-link').symlink_to(tmp_path.parent)
    with pytest.raises(RecordingError, match='符号链接'):
        engine.delete(rid)
    assert path.exists()


def test_saved_size_includes_metadata(engine, tmp_path):
    """服务列出的大小应等于条目实际文件大小，包括 recording.json。"""
    record(engine)
    info = engine.list()[0][0]
    path = tmp_path / info['relative_path']
    assert info['size_bytes'] == sum(p.stat().st_size for p in path.iterdir())
