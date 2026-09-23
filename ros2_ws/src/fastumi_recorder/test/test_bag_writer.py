"""Read actual compressed MCAP bags back through rosbag2."""

import subprocess

from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
import pytest
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CompressedImage, Image

from fastumi_recorder.bag_writer import BagWriter
from fastumi_recorder.engine import IMAGE_TYPES


@pytest.mark.skipif('mcap' not in rosbag2_py.get_registered_writers(),
                    reason='rosbag2_storage_mcap is not installed')
@pytest.mark.parametrize('mode', ('raw', 'jpeg', 'ffmpeg'))
def test_mcap_round_trip_camera_modes(tmp_path, mode):
    topic = '/test/camera'
    message = {
        'raw': Image(width=1, height=1, step=3, encoding='rgb8', data=[1, 2, 3]),
        'jpeg': CompressedImage(format='jpeg', data=b'\xff\xd8image\xff\xd9'),
        'ffmpeg': FFMPEGPacket(encoding='h264', width=1, height=1, data=b'packet'),
    }[mode]
    message.header.stamp.sec = 123
    serialized = serialize_message(message)
    bag = tmp_path / 'bag'
    writer = BagWriter(bag, {'image': (topic, IMAGE_TYPES[mode])})
    writer.write(topic, serialized, 456_000_000_007)
    writer.close()

    assert (bag / 'metadata.yaml').is_file()
    mcap_files = list(bag.glob('*.mcap'))
    assert mcap_files and b'zstd' in mcap_files[0].read_bytes()
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    assert reader.has_next()
    actual_topic, actual_data, timestamp_ns = reader.read_next()
    assert (actual_topic, timestamp_ns) == (topic, 456_000_000_007)
    restored = deserialize_message(actual_data, type(message))
    assert restored.header.stamp.sec == 123
    assert bytes(restored.data) == bytes(message.data)
    assert not reader.has_next()
    del reader
    result = subprocess.run(['ros2', 'bag', 'info', str(bag)],
                            text=True, capture_output=True, check=True)
    assert 'mcap' in result.stdout and topic in result.stdout
