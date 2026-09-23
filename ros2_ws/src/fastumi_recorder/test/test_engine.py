"""Exercise MCAP recording transactions without hardware."""

import json
from pathlib import Path
import time
import uuid

import pytest
from rclpy.serialization import deserialize_message
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32

from fastumi_recorder.catalog import RecordingError
from fastumi_recorder.engine import RecorderEngine
from fastumi_recorder.request_ledger import RequestLedger


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


class FakeWriter:
    instances = []
    fail_write = False
    fail_close = False

    def __init__(self, uri, topics):
        self.uri = uri
        self.topics = dict(topics)
        self.messages = []
        self.uri.mkdir()
        self.instances.append(self)

    def write(self, topic, serialized, timestamp_ns):
        if self.fail_write:
            raise OSError('write failed')
        self.messages.append((topic, bytes(serialized), timestamp_ns))

    def close(self):
        if self.fail_close:
            raise OSError('close failed')
        (self.uri / 'metadata.yaml').write_text('storage_identifier: mcap')
        (self.uri / 'bag_0.mcap').write_bytes(b'fake')


@pytest.fixture
def engine(tmp_path):
    FakeWriter.instances.clear()
    FakeWriter.fail_write = False
    FakeWriter.fail_close = False
    instance = RecorderEngine(tmp_path, image_transport='raw', writer_factory=FakeWriter)
    yield instance
    instance.close(timestamp=30.)


def ready(instance):
    instance.record('joint_state', JointState(name=['joint1'], position=[0.]), 1,
                    valid_for_readiness=True)
    instance.record('gripper_state', Float32(data=0.5), 2,
                    valid_for_readiness=True)
    instance.record('image', Image(width=1, height=1, step=3, data=[1, 2, 3]), 3,
                    valid_for_readiness=True)


def test_start_waits_for_writer_and_preserves_original_messages(engine, tmp_path):
    with pytest.raises(RecordingError, match='joint, gripper, image'):
        engine.start(request_id=str(uuid.uuid4()))
    ready(engine)
    request_id = str(uuid.uuid4())
    recording_id, code = engine.start(timestamp=10., request_id=request_id)
    assert code == 'STARTED'
    assert engine.start(request_id=request_id) == (recording_id, 'STARTED')
    joint = JointState(name=['unusual'], position=[float('nan')])
    image = Image(width=1, height=1, step=3, data=[7, 8, 9])
    engine.record('joint_state', joint, 11_000_000_001)
    engine.record('image', image, 11_000_000_002)
    assert engine.stop(recording_id, 12.)[1] == 'STOP_ACCEPTED'
    wait_for(lambda: engine.state != 'saving')
    assert engine.state == 'idle', engine.last_error
    writer = FakeWriter.instances[-1]
    assert [(topic, stamp) for topic, _, stamp in writer.messages] == [
        ('/joint_states', 11_000_000_001),
        ('/wrist_camera/image_raw/ffmpeg', 11_000_000_002),
    ]
    saved_joint = deserialize_message(writer.messages[0][1], JointState)
    saved_image = deserialize_message(writer.messages[1][1], Image)
    assert saved_joint.name == ['unusual']
    assert len(saved_joint.position) == 1
    assert list(saved_image.data) == [7, 8, 9]
    info = engine.list()[0][0]
    path = tmp_path / info['relative_path']
    assert (path / 'bag/metadata.yaml').is_file()
    assert (path / 'bag/bag_0.mcap').is_file()
    assert not (path / 'proprio.hdf5').exists()
    assert not (path / 'gripper.mp4').exists()
    assert info['joint_samples'] == 1
    assert info['image_frames'] == 1
    assert info['size_bytes'] == sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
    assert engine.get_request(request_id)['state'] == 'completed'


def test_cancel_restart_and_legacy_entry_is_not_listed(engine, tmp_path):
    old = tmp_path / 'test/default_test/episode_4'
    old.mkdir(parents=True)
    (old / 'proprio.hdf5').write_bytes(b'old')
    (old / 'recording.json').write_text(json.dumps({
        'recording_id': str(uuid.uuid4()),
        'relative_path': 'test/default_test/episode_4',
    }))
    ready(engine)
    recording_id, _ = engine.start(timestamp=10.)
    assert engine.cancel(recording_id)[1] == 'CANCELLED'
    assert engine.list()[1] == 0
    assert engine.cancel(recording_id)[1] == 'ALREADY_CANCELLED'
    recording_id, _ = engine.start(timestamp=11.)
    engine.stop(recording_id, 12.)
    wait_for(lambda: engine.state != 'saving')
    assert engine.list()[0][0]['relative_path'].endswith('episode_6')
    engine.close()
    other = RecorderEngine(tmp_path, image_transport='raw', writer_factory=FakeWriter)
    try:
        assert other.list()[1] == 1
        assert other.delete(recording_id)[1] == 'DELETED'
        assert (tmp_path / '.trash' / recording_id / 'bag/bag_0.mcap').is_file()
        assert other.list()[1] == 0
        assert old.is_dir()
    finally:
        other.close()


def test_open_failure_never_acknowledges_start(tmp_path):
    def fail_open(uri, topics):
        raise OSError('MCAP plugin missing')

    instance = RecorderEngine(tmp_path, image_transport='raw', writer_factory=fail_open)
    try:
        ready(instance)
        request_id = str(uuid.uuid4())
        with pytest.raises(RecordingError, match='MCAP plugin missing'):
            instance.start(request_id=request_id)
        assert instance.get_request(request_id)['state'] == 'failed'
        assert instance.list()[1] == 0
    finally:
        instance.close()


@pytest.mark.parametrize('failure', ['write', 'close', 'overflow'])
def test_failed_bag_is_not_published(tmp_path, failure):
    FakeWriter.instances.clear()
    FakeWriter.fail_write = failure == 'write'
    FakeWriter.fail_close = failure == 'close'
    instance = RecorderEngine(tmp_path, image_transport='raw', writer_factory=FakeWriter,
                              max_queued_bytes=1 if failure == 'overflow' else 1024 * 1024)
    try:
        ready(instance)
        request_id = str(uuid.uuid4())
        recording_id, _ = instance.start(timestamp=10., request_id=request_id)
        instance.record('image', Image(data=[1, 2, 3]), 11)
        if instance.state == 'recording':
            instance.stop(recording_id, 12.)
        wait_for(lambda: not instance._writer_active)
        assert instance.state == 'error'
        assert instance.list()[1] == 0
        assert instance.get_request(request_id)['state'] == 'failed'
        assert list(tmp_path.glob('test/default_test/.recording-*'))
    finally:
        instance.close()


def test_cancel_request_during_save_discards_before_publish(engine, monkeypatch):
    ready(engine)
    request_id = str(uuid.uuid4())
    recording_id, _ = engine.start(timestamp=10., request_id=request_id)
    original_close = FakeWriter.close
    import threading
    entered, release = threading.Event(), threading.Event()

    def blocked_close(writer):
        entered.set()
        assert release.wait(3)
        original_close(writer)

    monkeypatch.setattr(FakeWriter, 'close', blocked_close)
    engine.stop(recording_id, 12.)
    assert entered.wait(3)
    assert engine.cancel_request(request_id)[1] == 'CANCEL_ACCEPTED'
    release.set()
    wait_for(lambda: not engine._writer_active)
    assert engine.get_request(request_id)['state'] == 'cancelled'
    assert engine.list()[1] == 0


def test_shutdown_auto_saves_without_camera(tmp_path):
    instance = RecorderEngine(tmp_path, record_camera=False, writer_factory=FakeWriter)
    instance.record('joint_state', JointState(), 1, valid_for_readiness=True)
    instance.record('gripper_state', Float32(), 2, valid_for_readiness=True)
    recording_id, _ = instance.start(timestamp=10.)
    instance.record('joint_action', Jointpos(dof=7, joint=[0.] * 7), 11)
    assert instance.close(timestamp=12.)
    info = instance.last_completed
    assert info['recording_id'] == recording_id
    assert info['image_frames'] == 0
    assert (tmp_path / info['relative_path'] / 'bag/bag_0.mcap').is_file()
