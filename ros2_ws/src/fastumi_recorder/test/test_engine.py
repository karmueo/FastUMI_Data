"""验证独立录制事务、持久化、时间对齐和退出边界。"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import h5py
import pytest

from fastumi_recorder.catalog import Catalog, RecordingError
from fastumi_recorder.engine import (
    RecorderEngine, ffmpeg_packet_to_h264, image_to_jpeg,
)
from fastumi_recorder.core import EpisodeBuffer
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
    real_encode = engine_module.image_to_jpeg
    def delayed(message):
        entered.set()
        assert release.wait(3.)
        return real_encode(message)
    monkeypatch.setattr(engine_module, 'image_to_jpeg', delayed)
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
