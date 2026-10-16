"""Write one ROS 2 MCAP bag with native Zstd chunk compression."""

import fcntl
import json
from pathlib import Path
import struct
import subprocess
import sys

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
        verify_bag(self.uri)


def verify_bag(uri: Path) -> None:
    """Check that a closed bag has metadata, an MCAP file and the mcap storage id."""
    if not (uri / 'metadata.yaml').is_file() or not list(uri.glob('*.mcap')):
        raise RuntimeError('MCAP bag 未完整写入')
    metadata = rosbag2_py.Info().read_metadata(str(uri), 'mcap')
    if metadata.storage_identifier != 'mcap':
        raise RuntimeError('MCAP bag 存储类型不正确')


_FRAME = struct.Struct('<HqI')
_PIPE_BYTES = 1 << 20


class ProcessBagWriter:
    """Same interface as BagWriter, but rosbag2 runs in a child process.

    rosbag2_py.write() holds the GIL, so a disk stall (page-cache writeback) in the
    recorder process froze every subscription callback. Here write() only does a pipe
    write, which releases the GIL; a stalled disk merely backs up the pipe.
    """

    def __init__(self, uri: Path, topics: dict[str, tuple[str, str]]):
        self.uri = uri
        self._index = {topic: i for i, (topic, _) in enumerate(topics.values())}
        self._process = subprocess.Popen(
            [sys.executable, '-m', 'fastumi_recorder.bag_process', str(uri),
             json.dumps({k: list(v) for k, v in topics.items()})],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
            start_new_session=True)
        try:
            fcntl.fcntl(self._process.stdin.fileno(), 1031, _PIPE_BYTES)  # F_SETPIPE_SZ
        except OSError:
            pass
        line = self._process.stdout.readline().decode(errors='replace').strip()
        if line != 'READY':
            self._reap()
            raise RuntimeError(line or '写入子进程启动失败')

    def _reap(self) -> str:
        """Collect the child's last message and make sure it has exited."""
        try:
            self._process.stdin.close()
        except OSError:
            pass
        message = self._process.stdout.read().decode(errors='replace').strip()
        self._process.wait()
        return message

    def write(self, topic: str, serialized: bytes, timestamp_ns: int) -> None:
        """Send one unchanged CDR message to the writer process."""
        try:
            self._send(_FRAME.pack(self._index[topic], timestamp_ns, len(serialized))
                       + serialized)
        except (BrokenPipeError, ValueError, OSError) as error:
            raise RuntimeError(self._reap() or f'写入子进程已退出: {error}') from error

    def _send(self, data: bytes) -> None:
        view = memoryview(data)
        stdin = self._process.stdin
        while view:
            view = view[stdin.write(view):]

    def close(self) -> None:
        """Close stdin, wait for the child to finalize the MCAP, then verify it."""
        message = self._reap()
        if self._process.returncode != 0:
            raise RuntimeError(message or f'写入子进程异常退出: {self._process.returncode}')
        verify_bag(self.uri)
