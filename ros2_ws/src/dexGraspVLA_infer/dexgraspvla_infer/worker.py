"""单线程 GPU 调度：每类任务仅保留最新候选，交替服务感知与策略。"""

from collections import deque
from threading import Condition, Thread
import time


class GpuWorker:
    """重任务不在 ROS 回调执行；结果由 ROS timer 取回处理。"""

    def __init__(self):
        self.condition = Condition()
        self.pending = {}
        self.results = deque()
        self.last_kind = None
        self.closed = False
        self.active_kind, self.active_started = None, None
        self.durations = {}
        self.thread = Thread(target=self._run, name="dexgraspvla-gpu", daemon=True)
        self.thread.start()

    def submit(self, kind, function, done):
        with self.condition:
            if self.closed:
                return
            self.pending[kind] = (function, done)
            self.condition.notify()

    def cancel(self, kind):
        with self.condition:
            return self.pending.pop(kind, None) is not None

    def poll(self):
        """仅由 ROS executor 调用结果回调，保持节点状态串行更新。"""
        with self.condition:
            results = list(self.results)
            self.results.clear()
        for done, result, error in results:
            done(result, error)

    def snapshot(self):
        """就绪诊断显示正在执行的任务及最后一次实际运行时间。"""
        with self.condition:
            return {"active": self.active_kind, "active_age_s": None if self.active_started is None else time.monotonic() - self.active_started,
                    "last_duration_s": dict(self.durations), "pending": list(self.pending)}

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending)
                if self.closed:
                    return
                kind = next((key for key in self.pending if key != self.last_kind), next(iter(self.pending)))
                function, done = self.pending.pop(kind)
                self.last_kind = kind
                self.active_kind, self.active_started = kind, time.monotonic()
            try:
                result, error = function(), None
            except Exception as exception:
                result, error = None, exception
            with self.condition:
                self.durations[kind] = time.monotonic() - self.active_started
                self.active_kind, self.active_started = None, None
                if not self.closed:
                    self.results.append((done, result, error))

    def close(self):
        with self.condition:
            self.closed = True
            self.pending.clear()
            self.condition.notify_all()
        self.thread.join(timeout=2)
