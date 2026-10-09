"""检验最新候选覆盖及 GPU 串行调度，不依赖 GPU。"""

from threading import Event
import time

from dexgraspvla_infer.worker import GpuWorker


def test_latest_candidate_and_serial_gpu_work():
    worker, entered, release, finished = GpuWorker(), Event(), Event(), Event()
    results = []
    def blocked():
        entered.set()
        release.wait(1)
        return 1
    try:
        worker.submit("policy", blocked, lambda value, error: results.append(value))
        assert entered.wait(1)
        worker.submit("perception", lambda: 2, lambda value, error: results.append(value))
        worker.submit("perception", lambda: 3, lambda value, error: (results.append(value), finished.set()))
        release.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not finished.is_set():
            worker.poll()
            time.sleep(0.005)
        assert results == [1, 3]
    finally:
        release.set()
        worker.close()
