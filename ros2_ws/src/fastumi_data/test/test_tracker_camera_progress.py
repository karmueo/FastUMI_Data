"""验证 Tracker–鱼眼标定进度日志的双写、节流和 ETA。"""

from io import StringIO
from pathlib import Path

from fastumi_data.tracker_camera_progress import CalibrationProgressLogger


class ManualClock:
    """提供测试可手工推进的单调时钟。"""

    def __init__(self) -> None:
        """从 100 秒开始，避免测试依赖零值边界。"""
        self.now = 100.0

    def __call__(self) -> float:
        """返回当前单调时间。"""
        return self.now

    def advance(self, seconds: float) -> None:
        """把当前时间推进指定秒数。"""
        self.now += seconds


def test_progress_logger_dual_writes_throttles_and_estimates_eta(
    tmp_path: Path,
) -> None:
    """记录器应双写阶段进度，并按间隔抑制频繁更新。"""
    clock = ManualClock()
    stream = StringIO()
    logger = CalibrationProgressLogger(
        tmp_path, interval_seconds=30.0, clock=clock, stream=stream
    )

    logger.start_stage("time_offset_scan", "开始扫描")
    initial_text = stream.getvalue()
    assert "stage=time_offset_scan" in initial_text
    assert "开始扫描" in initial_text

    clock.advance(10.0)
    logger.update("已完成首点", completed=1, total=5)
    assert stream.getvalue() == initial_text

    clock.advance(20.0)
    logger.update("扫描推进", completed=2, total=5)
    output = stream.getvalue()
    assert "progress=40.0%" in output
    assert "ETA=00:00:45" in output
    assert output == logger.log_path.read_text(encoding="utf-8")

    logger.close()
    assert logger.log_path.open(encoding="utf-8").read() == output


def test_progress_logger_records_failure_and_closes_idempotently(
    tmp_path: Path,
) -> None:
    """失败原因必须持久化，重复关闭不能破坏已有日志。"""
    stream = StringIO()
    logger = CalibrationProgressLogger(tmp_path, stream=stream)
    logger.start_stage("joint_optimization", "开始优化")
    logger.fail("用户中断")

    logger.close()
    logger.close()

    output = logger.log_path.read_text(encoding="utf-8")
    assert "stage=joint_optimization" in output
    assert "status=FAILED" in output
    assert "用户中断" in output
