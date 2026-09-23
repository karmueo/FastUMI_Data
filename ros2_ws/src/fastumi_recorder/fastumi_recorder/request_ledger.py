"""Persist request identities before recording side effects."""

import shutil
import sqlite3
import uuid


def canonical_request_id(value):
    """Require the canonical UUID spelling used by ROS clients."""
    try:
        if str(uuid.UUID(value)) != value:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ValueError('request_id 必须为标准 UUID')
    return value


class RequestLedger:
    """A durable, unbounded request table owned by one recorder process."""

    def __init__(self, root):
        # RecorderEngine holds its transaction lock for every ledger call.
        self.connection = sqlite3.connect(str(root / '.recording_requests.sqlite3'),
                                          check_same_thread=False)
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('PRAGMA journal_mode=DELETE')
        self.connection.execute('''CREATE TABLE IF NOT EXISTS requests (
            request_id TEXT PRIMARY KEY, state TEXT NOT NULL,
            recording_id TEXT NOT NULL DEFAULT '',
            start_success INTEGER, start_code TEXT, start_message TEXT,
            code TEXT NOT NULL DEFAULT '', message TEXT NOT NULL DEFAULT '',
            cancel_requested INTEGER NOT NULL DEFAULT 0)''')
        self.connection.commit()

    def get(self, request_id):
        row = self.connection.execute(
            'SELECT state, recording_id, start_success, start_code, start_message, '
            'code, message, cancel_requested FROM requests WHERE request_id=?',
            (request_id,)).fetchone()
        if row is None:
            return None
        return dict(zip(('state', 'recording_id', 'start_success', 'start_code',
                         'start_message', 'code', 'message', 'cancel_requested'), row))

    def put(self, request_id, **fields):
        columns = ('state', 'recording_id', 'start_success', 'start_code',
                   'start_message', 'code', 'message', 'cancel_requested')
        values = {key: fields[key] for key in columns if key in fields}
        try:
            if self.get(request_id) is None:
                values.setdefault('state', 'pending')
                self.connection.execute(
                    'INSERT INTO requests (request_id, ' + ', '.join(values) + ') VALUES ('
                    + ', '.join('?' for _ in range(len(values) + 1)) + ')',
                    (request_id, *values.values()))
            else:
                self.connection.execute(
                    'UPDATE requests SET ' + ', '.join(key + '=?' for key in values)
                    + ' WHERE request_id=?', (*values.values(), request_id))
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def recover(self, catalog):
        """Resolve interrupted requests without re-running a start."""
        rows = self.connection.execute(
            "SELECT request_id, recording_id, state, cancel_requested, start_success FROM requests "
            "WHERE state IN ('pending', 'recording', 'saving', 'cancelled')"
        ).fetchall()
        for request_id, recording_id, state, cancel_requested, start_success in rows:
            if recording_id and recording_id in catalog.entries:
                self.put(request_id, state='completed', code='COMPLETED', message='录制已保存')
            elif state == 'cancelled' or cancel_requested:
                try:
                    if recording_id:
                        for candidate in catalog.root.glob(f'*/*/.recording-{recording_id}'):
                            safe = catalog.safe_path(candidate.relative_to(catalog.root))
                            shutil.rmtree(safe)
                    self.put(request_id, state='cancelled', code='CANCELLED',
                             message='撤销请求已恢复并完成')
                except OSError as error:
                    self.put(request_id, state='failed', code='IO_ERROR', message=str(error))
            else:
                fields = dict(state='failed', code='INTERRUPTED',
                              message='录制节点重启，未完成请求不会重新执行')
                if start_success is None:
                    fields.update(start_success=0, start_code='INTERRUPTED',
                                  start_message=fields['message'])
                self.put(request_id, **fields)

    def close(self):
        self.connection.close()
