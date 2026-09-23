"""Write one ROS 2 MCAP bag with native Zstd chunk compression."""

from pathlib import Path

import rosbag2_py


class BagWriter:
    """Own a rosbag2 writer until its metadata and MCAP file are finalized."""

    def __init__(self, uri: Path, topics: dict[str, tuple[str, str]]):
        if 'mcap' not in rosbag2_py.get_registered_writers():
            raise RuntimeError('缺少 rosbag2_storage_mcap；请安装 ROS 2 MCAP 存储插件')
        self.uri = uri
        self._writer = rosbag2_py.SequentialWriter()
        try:
            self._writer.open(
                rosbag2_py.StorageOptions(
                    uri=str(uri), storage_id='mcap', storage_preset_profile='zstd_fast'),
                rosbag2_py.ConverterOptions('', ''),
            )
            for topic, message_type in topics.values():
                self._writer.create_topic(rosbag2_py.TopicMetadata(
                    name=topic, type=message_type, serialization_format='cdr'))
        except Exception:
            writer, self._writer = self._writer, None
            del writer
            raise

    def write(self, topic: str, serialized: bytes, timestamp_ns: int) -> None:
        """Append the unchanged CDR message with its receive time."""
        self._writer.write(topic, serialized, timestamp_ns)

    def close(self) -> None:
        """Release the writer so rosbag2 writes metadata.yaml and the MCAP footer."""
        writer, self._writer = self._writer, None
        del writer
        if not (self.uri / 'metadata.yaml').is_file() or not list(self.uri.glob('*.mcap')):
            raise RuntimeError('MCAP bag 未完整写入')
        metadata = rosbag2_py.Info().read_metadata(str(self.uri), 'mcap')
        if metadata.storage_identifier != 'mcap':
            raise RuntimeError('MCAP bag 存储类型不正确')
