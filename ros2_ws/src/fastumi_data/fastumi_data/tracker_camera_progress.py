"""提供 Tracker–鱼眼相机标定的阶段进度、耗时、ETA 和双写日志。"""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import sys
import time
from typing import Callable, TextIO


def _format_duration(seconds: float) -> str:
    """把非负秒数格式化为 ``HH:MM:SS``。"""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}"


class CalibrationProgressLogger:
    """把节流后的标定阶段进度同时写入控制台和持久日志。"""

    def __init__(
        self,
        output_dir: Path | str,
        interval_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        stream: TextIO | None = None,
    ) -> None:
        """创建记录器并打开 ``calibration.log``。

        Args:
            output_dir: 标定输出目录。
            interval_seconds: 非强制更新之间的最短秒数。
            clock: 用于耗时计算的单调时钟。
            stream: 控制台输出流；默认使用标准输出。

        Raises:
            ValueError: 心跳间隔不是有限正数时抛出。
            OSError: 输出目录或日志文件无法创建时向上传播。
        """
        if not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
            raise ValueError("进度心跳间隔必须为有限正数")
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.log_path = destination / "calibration.log"
        self._interval_seconds = float(interval_seconds)
        self._clock = clock
        self._stream = stream if stream is not None else sys.stdout
        self._file = self.log_path.open("a", encoding="utf-8")
        self._run_start = self._clock()
        self._stage_start = self._run_start
        self._last_emit = -math.inf
        self._stage = "initialization"
        self._closed = False

    def __enter__(self) -> "CalibrationProgressLogger":
        """返回已打开的记录器。"""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """离开上下文时关闭日志文件。"""
        self.close()

    def _write(self, message: str, status: str, extra: str = "") -> None:
        """构造一行日志并立即双写和刷新。"""
        now = self._clock()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fields = [
            timestamp,
            f"stage={self._stage}",
            f"total={_format_duration(now - self._run_start)}",
            f"stage_elapsed={_format_duration(now - self._stage_start)}",
            f"status={status}",
        ]
        if extra:
            fields.append(extra)
        fields.append(message)
        line = " | ".join(fields) + "\n"
        self._stream.write(line)
        self._stream.flush()
        self._file.write(line)
        self._file.flush()
        self._last_emit = now

    def start_stage(self, stage: str, message: str) -> None:
        """开始新阶段并立即记录。"""
        self._stage = str(stage)
        self._stage_start = self._clock()
        self._last_emit = -math.inf
        self._write(message, "STARTED")

    def update(
        self,
        message: str,
        completed: int | None = None,
        total: int | None = None,
        force: bool = False,
        extra: str = "",
    ) -> bool:
        """达到心跳间隔时记录进度，并返回本次是否输出。"""
        now = self._clock()
        if not force and now - self._last_emit < self._interval_seconds:
            return False
        progress_fields = []
        if completed is not None and total is not None and total > 0:
            bounded_completed = min(max(int(completed), 0), int(total))
            progress_fields.append(
                f"progress={100.0 * bounded_completed / total:.1f}%"
            )
            elapsed = now - self._stage_start
            if bounded_completed > 0 and elapsed > 0.0:
                eta_seconds = (
                    elapsed / bounded_completed * (total - bounded_completed)
                )
                progress_fields.append(
                    f"ETA={_format_duration(eta_seconds)}"
                )
            else:
                progress_fields.append("ETA=未知")
        if extra:
            progress_fields.append(extra)
        self._write(message, "RUNNING", " | ".join(progress_fields))
        return True

    def finish_stage(self, message: str) -> None:
        """把当前阶段标记为完成并立即记录。"""
        self._write(message, "COMPLETED")

    def fail(self, message: str) -> None:
        """把当前阶段标记为失败并立即持久化原因。"""
        self._write(message, "FAILED")

    def close(self) -> None:
        """幂等关闭持久日志文件。"""
        if self._closed:
            return
        self._file.close()
        self._closed = True
