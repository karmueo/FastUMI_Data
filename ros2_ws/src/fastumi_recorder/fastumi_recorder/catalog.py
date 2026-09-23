"""管理录制身份、持久化编号和同文件系统内的回收操作。"""

import fcntl
import json
import os
from pathlib import Path
import uuid
import tempfile


class RecordingError(ValueError):
    """携带可供 ROS 客户端识别的结果码。"""

    def __init__(self, code, message):
        """初始化配置和本实例持有的资源。"""
        super().__init__(message)
        self.code = code


def task_component(value):
    """仅接受单层任务名称，禁止绝对路径和路径穿越。"""
    if (not isinstance(value, str) or not value.strip() or value in ('.', '..')
            or '/' in value or '\\' in value or '\x00' in value or value.startswith('.')):
        raise RecordingError('INVALID_ARGUMENT', '任务名必须为非隐藏的单层目录名')
    return value


def atomic_json(path, data):
    """先刷新临时文件，再原子替换元数据。"""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', dir=path.parent, prefix='.' + path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def default_dataset_root():
    """从源码或符号链接安装定位仓库；独立安装需显式配置根目录。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / 'ros2_ws/src/fastumi_recorder').is_dir():
            return parent / 'dataset/h5dy_data'
    raise RecordingError('INVALID_ARGUMENT', '独立安装请提供 dataset_root 绝对路径')


class Catalog:
    """单写入者录制目录；调用者负责线程互斥。"""

    def __init__(self, root):
        """初始化配置和本实例持有的资源。"""
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.safe_path('.recorder.lock').open('a')
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close()
            raise RecordingError('BUSY', '数据根目录已有录制服务占用')
        self.trash = self.safe_path('.trash')
        self.trash.mkdir(exist_ok=True)
        self.entries = {}
        for meta in self.root.glob('*/*/episode_*/recording.json'):
            try:
                self.safe_path(meta.relative_to(self.root))
                info = json.loads(meta.read_text())
                uuid.UUID(info['recording_id'])
                expected = meta.parent.relative_to(self.root).as_posix()
                bag = meta.parent / 'bag'
                if (info['relative_path'] != expected or bag.is_symlink()
                        or not (bag / 'metadata.yaml').is_file()
                        or not any(bag.glob('*.mcap')) or any(p.is_symlink() for p in bag.rglob('*'))):
                    continue
                self.entries[info['recording_id']] = info
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def safe_path(self, relative):
        """拒绝路径上的任意符号链接，包括指向根目录内部的链接。"""
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise RecordingError('INVALID_ARGUMENT', '非法数据相对路径')
        current = self.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise RecordingError('INVALID_ARGUMENT', '录制路径不能包含符号链接')
        if not current.resolve().is_relative_to(self.root):
            raise RecordingError('INVALID_ARGUMENT', '录制路径越过数据根目录')
        return current

    def allocate(self, dir_name, name, started_at):
        """预留 UUID 和持久化 episode 编号；取消也不复用编号。"""
        directory = self.safe_path(Path(task_component(dir_name)) / task_component(name))
        directory.mkdir(parents=True, exist_ok=True)
        sequence = self.safe_path(directory.relative_to(self.root) / '.next_episode.json')
        indices = [int(p.name[8:]) for p in directory.glob('episode_*') if p.name[8:].isdigit()]
        number = max(indices, default=-1) + 1
        if sequence.exists():
            number = max(number, int(json.loads(sequence.read_text())['next']))
        atomic_json(sequence, {'next': number + 1})
        recording_id = str(uuid.uuid4())
        relative = (directory / f'episode_{number}').relative_to(self.root).as_posix()
        temporary = self.safe_path(directory.relative_to(self.root) / f'.recording-{recording_id}')
        temporary.mkdir()
        info = dict(recording_id=recording_id, dir_name=dir_name, name=name,
                    relative_path=relative, started_at=started_at, stopped_at=0.0,
                    duration=0.0, joint_samples=0, action_samples=0,
                    gripper_samples=0, tracker_samples=0, image_frames=0,
                    size_bytes=0, has_joint_action=False)
        atomic_json(temporary / 'pending.json', info)
        return info, temporary

    def publish(self, info, temporary):
        """仅完整写入的条目进入列表，正式目录以原子 rename 公布。"""
        target = self.safe_path(info['relative_path'])
        if target.exists():
            raise RecordingError('IO_ERROR', '目标 episode 已存在')
        (temporary / 'pending.json').unlink(missing_ok=True)
        # 计入嵌套 bag 和元数据；字段位数稳定后得到实际目录文件总字节数。
        for _ in range(4):
            atomic_json(temporary / 'recording.json', info)
            size = sum(p.stat().st_size for p in temporary.rglob('*') if p.is_file())
            if info['size_bytes'] == size:
                break
            info['size_bytes'] = size
        os.rename(temporary, target)
        self.entries[info['recording_id']] = dict(info)

    def list(self, dir_name='', name='', offset=0, limit=100):
        """按开始时间和 UUID 排序，返回正常条目的一页。"""
        for value in (dir_name, name):
            if value:
                task_component(value)
        entries = [dict(e) for e in self.entries.values()
                   if (not dir_name or e['dir_name'] == dir_name)
                   and (not name or e['name'] == name)]
        entries.sort(key=lambda e: (e['started_at'], e['recording_id']), reverse=True)
        return entries[offset:offset + min(limit or 100, 1000)], len(entries)

    def delete(self, recording_id):
        """把已登记条目移入回收目录；重复回收幂等。"""
        try:
            if str(uuid.UUID(recording_id)) != recording_id:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise RecordingError('INVALID_ARGUMENT', 'recording_id 必须为标准 UUID')
        target = self.safe_path(Path('.trash') / recording_id)
        if recording_id not in self.entries:
            if target.is_dir():
                return 'ALREADY_DELETED'
            raise RecordingError('NOT_FOUND', '未找到录制条目')
        source = self.safe_path(self.entries[recording_id]['relative_path'])
        # 防止条目内的文件被替换为越界符号链接。
        if any(p.is_symlink() for p in source.rglob('*')):
            raise RecordingError('INVALID_ARGUMENT', '录制条目包含符号链接')
        os.rename(source, target)
        del self.entries[recording_id]
        return 'DELETED'

    def close(self):
        """释放跨进程数据目录锁。"""
        self._lock_file.close()
