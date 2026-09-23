"""Durable generation gate for RM75 enable and disable requests."""

import fcntl
import json
import os
from pathlib import Path
import tempfile


class TeleopGenerationStore:
    """Keep the highest operation generation across bridge restarts."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        if not self.path.is_absolute():
            raise ValueError('teleop_generation_file 必须为绝对路径')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.path.with_suffix(self.path.suffix + '.lock').open('a')
        fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if self.path.exists():
            self.record = json.loads(self.path.read_text())
            if self.record.get('state') == 'pending':
                self.finish(False, 'INTERRUPTED', '控制节点重启，操作未确认')
        else:
            self.record = dict(generation=0, operation='', state='done',
                               success=False, code='', message='')

    def _write(self):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent,
                                             prefix='.teleop-', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self.record, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def begin(self, generation, operation):
        if generation <= self.record['generation']:
            if generation == self.record['generation'] and operation == self.record['operation']:
                return 'duplicate'
            return 'stale'
        previous = dict(self.record)
        self.record = dict(generation=generation, operation=operation, state='pending',
                           success=False, code='PENDING', message='操作尚未完成')
        try:
            self._write()
        except Exception:
            self._reload_after_error(previous)
            raise
        return 'new'

    def finish(self, success, code, message):
        previous = dict(self.record)
        self.record.update(state='done', success=success, code=code, message=message)
        try:
            self._write()
        except Exception:
            self._reload_after_error(previous)
            raise

    def _reload_after_error(self, previous):
        """Preserve an already replaced generation after a later fsync error."""
        try:
            self.record = json.loads(self.path.read_text()) if self.path.exists() else previous
        except (OSError, ValueError):
            # Never lower the in-memory generation when disk state is uncertain.
            pass

    def close(self):
        self._lock_file.close()
