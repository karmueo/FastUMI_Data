"""独立进程中写 MCAP：rosbag2_py 的 write() 持有 GIL，磁盘回写阻塞会拖停订阅回调。

协议：父进程经 stdin 发送帧 ``<HqI`` (话题序号, 接收时间 ns, 长度) + CDR 负载；
stdin EOF 表示正常结束，子进程关闭 bag 后退出。启动成功时向 stdout 输出 READY，
失败输出 ERROR <原因> 并以非零状态退出。
"""

import json
import signal
import struct
import sys
from pathlib import Path

from fastumi_recorder.bag_writer import BagWriter

FRAME = struct.Struct('<HqI')


def _read_exact(stream, size):
    """读取 size 字节；EOF 且无数据返回 b''，中途 EOF 抛错。"""
    data = stream.read(size)
    if len(data) not in (0, size):
        raise EOFError('写入帧被截断')
    return data


def main(argv=None):
    """按 argv: <bag_uri> <topics_json> 写入直到 stdin EOF。"""
    argv = sys.argv[1:] if argv is None else argv
    # 由父进程决定何时结束；Ctrl+C / SIGTERM 不能中断保存。
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    writer = None
    try:
        topics = json.loads(argv[1])
        writer = BagWriter(Path(argv[0]), {k: tuple(v) for k, v in topics.items()})
        names = [topic for topic, _ in topics.values()]
        print('READY', flush=True)
        stdin = sys.stdin.buffer
        while True:
            header = _read_exact(stdin, FRAME.size)
            if not header:
                break
            index, timestamp_ns, length = FRAME.unpack(header)
            payload = _read_exact(stdin, length) if length else b''
            if len(payload) != length:
                raise EOFError('写入帧被截断')
            writer.write(names[index], payload, timestamp_ns)
        writer.close()
        writer = None
    except BaseException as error:
        print(f'ERROR {type(error).__name__}: {error}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
