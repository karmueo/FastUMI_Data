"""为 30 Hz Link7 数据按标称源帧率选择图像帧，并检查源时间间隔质量。

两条路径按标称源帧率 ``f_source`` 判定，不用实测帧率推断：

* 同频逐帧（``|f_source/30 − 1| ≤ 1%``）：裁剪后的每张图像依次输出，源索引连续，
  不用固定 ``1/30`` 网格匹配。相邻间隔按 ``m = round(Δt·f_source)`` 归类：
  ``m = 1`` 正常；``m ≥ 2`` 且落在容差内为丢 ``m−1`` 帧（只提示 ``FRAME_DROP``）；
  其余为 ``FRAME_INTERVAL_ANOMALY`` 并拒绝。估计帧号 ``n`` 按 ``m`` 累加，最小二乘
  拟合 ``t ≈ a + n/f_fit`` 检查标称值声明。
* 高帧率抽帧（``f_source ≥ 60``）：以首帧时间为起点生成精确的 30 Hz 纳秒网格，
  每个网格点取最近源图像（等距取较早帧），零复用，匹配误差超门限拒绝。

时间均为整数纳秒；输出的名义时间以首个输出帧为零点。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np


# Link7 数据集固定的输出频率（Hz）。
TARGET_FREQUENCY_HZ = 30
SAME_RATE = "same_rate"
HIGH_RATE = "high_rate"
# 同频路径允许的标称帧率相对偏差。
SAME_RATE_TOLERANCE = 0.01
# 高帧率路径要求的最低标称帧率（Hz）。
HIGH_RATE_MINIMUM_HZ = 60.0

# 质量问题代码。
FRAME_DROP = "FRAME_DROP"
FRAME_INTERVAL_ANOMALY = "FRAME_INTERVAL_ANOMALY"
FRAME_RATE_DEVIATION = "FRAME_RATE_DEVIATION"
LOW_EFFECTIVE_RATE = "LOW_EFFECTIVE_RATE"
MATCH_ERROR_EXCEEDED = "MATCH_ERROR_EXCEEDED"
FRAME_REUSE = "FRAME_REUSE"
TIMESTAMP_NOT_INCREASING = "TIMESTAMP_NOT_INCREASING"
INSUFFICIENT_IMAGES = "INSUFFICIENT_IMAGES"


def issue(code: str, message: str, **details) -> Dict:
    """构造报告中的结构化问题条目。"""
    item = {"code": code, "message": message}
    item.update(details)
    return item


@dataclass(frozen=True)
class FrameSelectionConfig:
    """选帧和源时间质量门限。"""

    interval_tolerance_s: float = 0.008
    """同频路径相邻间隔相对整数倍周期的容差。"""
    max_rate_deviation_ratio: float = 0.005
    """同频路径 ``|f_fit/f_source − 1|`` 上限。"""
    min_effective_rate_ratio: float = 0.95
    """源图像有效帧率相对标称值的下限比例。"""
    high_rate_match_margin_s: float = 0.002
    """高帧率路径最大匹配误差 = ``0.5/f_source + 本值``。"""

    def validate(self) -> "FrameSelectionConfig":
        """检查门限均为有限正数，比例位于合理范围。"""
        values = {
            "interval_tolerance_s": self.interval_tolerance_s,
            "max_rate_deviation_ratio": self.max_rate_deviation_ratio,
            "min_effective_rate_ratio": self.min_effective_rate_ratio,
            "high_rate_match_margin_s": self.high_rate_match_margin_s,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须为有限正数，当前为 {value!r}")
        if self.min_effective_rate_ratio > 1.0:
            raise ValueError("min_effective_rate_ratio 不能大于 1")
        return self


def selection_path_for(nominal_source_fps: float) -> str:
    """按标称源帧率返回选帧路径；不属于两条路径时报配置错误。"""
    fps = float(nominal_source_fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"标称源帧率必须为有限正数，当前为 {nominal_source_fps!r}")
    if abs(fps / TARGET_FREQUENCY_HZ - 1.0) <= SAME_RATE_TOLERANCE + 1.0e-12:
        return SAME_RATE
    if fps >= HIGH_RATE_MINIMUM_HZ:
        return HIGH_RATE
    raise ValueError(
        f"标称源帧率 {fps:g} Hz 既不在 30 Hz ±1% 的同频路径内，也未达到高帧率路径的 "
        f"{HIGH_RATE_MINIMUM_HZ:g} Hz；不做非整数倍抽帧"
    )


def nominal_offset_ns(frame_number: int | np.ndarray) -> np.ndarray:
    """返回 30 Hz 第 ``n`` 帧的名义纳秒偏移 ``round(n·1e9/30)``（整数有理运算）。"""
    numbers = np.asarray(frame_number, dtype=np.int64)
    return (numbers * 1_000_000_000 + TARGET_FREQUENCY_HZ // 2) // TARGET_FREQUENCY_HZ


def effective_source_rate(timestamps_ns: Sequence[int]) -> Optional[float]:
    """按 ``(N−1)/(t_last − t_first)`` 计算时间戳唯一的源图像有效帧率。"""
    unique = np.unique(np.asarray(timestamps_ns, dtype=np.int64))
    if len(unique) < 2 or unique[-1] <= unique[0]:
        return None
    return (len(unique) - 1) * 1.0e9 / float(unique[-1] - unique[0])


def fit_frame_rate(
    timestamps_ns: Sequence[int], frame_numbers: Sequence[int]
) -> Optional[float]:
    """最小二乘拟合 ``t ≈ a + n/f``，返回 ``f``；数据不足时返回 None。"""
    times = np.asarray(timestamps_ns, dtype=np.int64)
    numbers = np.asarray(frame_numbers, dtype=np.float64)
    if len(times) < 2 or np.ptp(numbers) <= 0:
        return None
    seconds = (times - times[0]).astype(np.float64) / 1.0e9
    centered = numbers - numbers.mean()
    slope = float(np.dot(centered, seconds - seconds.mean()) / np.dot(centered, centered))
    if slope <= 0.0:
        return None
    return 1.0 / slope


def _percentiles(values: np.ndarray) -> Dict[str, Optional[float]]:
    """返回绝对值的 P95/P99/最大值（毫秒）。"""
    if values.size == 0:
        return {"p95": None, "p99": None, "max": None}
    absolute = np.abs(values)
    return {
        "p95": float(np.percentile(absolute, 95)),
        "p99": float(np.percentile(absolute, 99)),
        "max": float(np.max(absolute)),
    }


@dataclass
class FrameSelection:
    """一条 episode 的选帧结果；位置索引指向 episode 内按顺序排列的源图像。"""

    path: str
    nominal_source_fps: float
    positions: np.ndarray
    """选中源图像在 episode 图像序列中的位置（int64）。"""
    image_timestamps_ns: np.ndarray
    frame_numbers: np.ndarray
    """名义帧号（同频路径为估计帧号 ``n``，高帧率路径为网格序号 ``k``）。"""
    match_error_ns: Optional[np.ndarray] = None
    """高帧率路径的有符号匹配误差 ``t_image − t_grid``。"""
    rejections: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        """没有任何拒绝原因时为 True。"""
        return not self.rejections

    def __len__(self) -> int:
        """选中帧数。"""
        return int(len(self.positions))

    @property
    def nominal_timestamp_ns(self) -> np.ndarray:
        """以首个选中帧为零点的名义纳秒时间。"""
        if len(self.frame_numbers) == 0:
            return np.zeros(0, dtype=np.int64)
        return nominal_offset_ns(self.frame_numbers - self.frame_numbers[0])

    @property
    def dropped_frames_before(self) -> np.ndarray:
        """每帧之前检测到的丢帧数；首帧及高帧率路径恒为 0。"""
        dropped = np.zeros(len(self.frame_numbers), dtype=np.int32)
        if self.path == SAME_RATE and len(self.frame_numbers) > 1:
            dropped[1:] = (np.diff(self.frame_numbers) - 1).astype(np.int32)
        return dropped

    def subset(self, start: int, stop: int) -> "FrameSelection":
        """返回 ``[start, stop)`` 范围的选帧，名义时间与丢帧计数按新首帧重算。"""
        return replace(
            self,
            positions=self.positions[start:stop].copy(),
            image_timestamps_ns=self.image_timestamps_ns[start:stop].copy(),
            frame_numbers=self.frame_numbers[start:stop].copy(),
            match_error_ns=(
                None if self.match_error_ns is None
                else self.match_error_ns[start:stop].copy()
            ),
            rejections=list(self.rejections),
            warnings=list(self.warnings),
            stats=dict(self.stats),
        )


def _check_effective_rate(
    timestamps: np.ndarray, nominal: float, config: FrameSelectionConfig,
    stats: Dict, rejections: List[Dict],
) -> None:
    """计算有效源帧率并在低于门限时追加拒绝原因。"""
    rate = effective_source_rate(timestamps)
    threshold = config.min_effective_rate_ratio * nominal
    unique = np.unique(timestamps)
    stats.update(
        {
            "effective_source_rate_hz": rate,
            "effective_rate_threshold_hz": threshold,
            "effective_rate_frame_count": int(len(unique)),
            "effective_rate_duration_s": (
                float(unique[-1] - unique[0]) / 1.0e9 if len(unique) > 1 else 0.0
            ),
        }
    )
    if rate is None or rate < threshold:
        rejections.append(
            issue(
                LOW_EFFECTIVE_RATE,
                f"源图像有效帧率 {rate if rate is not None else 0.0:.3f} Hz 低于 "
                f"{threshold:.3f} Hz（标称 {nominal:g} Hz × {config.min_effective_rate_ratio:g}）",
                measured_hz=rate,
                threshold_hz=threshold,
            )
        )


def select_same_rate(
    timestamps_ns: Sequence[int],
    nominal_source_fps: float,
    config: FrameSelectionConfig = FrameSelectionConfig(),
) -> FrameSelection:
    """同频逐帧路径：依次输出每张图像，按相邻间隔检测丢帧与异常。"""
    config.validate()
    times = np.asarray(timestamps_ns, dtype=np.int64)
    rejections: List[Dict] = []
    warnings: List[Dict] = []
    stats: Dict = {"selection_path": SAME_RATE, "source_image_count": int(len(times))}
    count = len(times)
    frame_numbers = np.zeros(count, dtype=np.int64)
    if count < 2:
        rejections.append(issue(INSUFFICIENT_IMAGES, f"episode 内只有 {count} 张图像"))
        return FrameSelection(
            SAME_RATE, float(nominal_source_fps), np.arange(count, dtype=np.int64),
            times, frame_numbers, None, rejections, warnings, stats,
        )
    period_ns = 1.0e9 / float(nominal_source_fps)
    tolerance_ns = config.interval_tolerance_s * 1.0e9
    deltas = np.diff(times).astype(np.float64)
    multiples = np.rint(deltas / period_ns).astype(np.int64)
    residuals = deltas - multiples * period_ns
    anomalies = (deltas <= 0) | (multiples <= 0) | (np.abs(residuals) > tolerance_ns)
    for index in np.flatnonzero(anomalies)[:20]:
        rejections.append(
            issue(
                FRAME_INTERVAL_ANOMALY,
                f"第 {index} 与 {index + 1} 张图像间隔 {deltas[index] / 1e6:.3f} ms "
                f"无法归类为周期 {period_ns / 1e6:.3f} ms 的整数倍（容差 "
                f"{config.interval_tolerance_s * 1e3:g} ms）",
                position=int(index + 1),
                interval_ms=float(deltas[index] / 1.0e6),
            )
        )
    if np.count_nonzero(anomalies) > 20:
        rejections.append(
            issue(
                FRAME_INTERVAL_ANOMALY,
                f"另有 {int(np.count_nonzero(anomalies)) - 20} 处间隔异常",
            )
        )
    safe_multiples = np.where(anomalies, 1, multiples)
    frame_numbers[1:] = np.cumsum(safe_multiples)
    drops = []
    for index in np.flatnonzero(~anomalies & (multiples >= 2)):
        drops.append({"position": int(index + 1), "dropped": int(multiples[index] - 1)})
        warnings.append(
            issue(
                FRAME_DROP,
                f"第 {index + 1} 张图像之前丢 {int(multiples[index] - 1)} 帧"
                f"（间隔 {deltas[index] / 1e6:.3f} ms）",
                position=int(index + 1),
                dropped=int(multiples[index] - 1),
            )
        )
    normal = ~anomalies
    deviations_ms = residuals[normal] / 1.0e6
    stats.update(
        {
            "interval_ms": {
                "min": float(deltas.min() / 1e6),
                "max": float(deltas.max() / 1e6),
                "mean": float(deltas.mean() / 1e6),
            },
            "interval_deviation_ms": _percentiles(deviations_ms),
            "frame_drops": drops,
            "dropped_frame_total": int(sum(item["dropped"] for item in drops)),
            "interval_anomaly_count": int(np.count_nonzero(anomalies)),
        }
    )
    fitted = None if anomalies.any() else fit_frame_rate(times, frame_numbers)
    stats["fitted_frame_rate_hz"] = fitted
    if fitted is not None:
        deviation = fitted / float(nominal_source_fps) - 1.0
        stats["fitted_rate_deviation_ratio"] = deviation
        if abs(deviation) > config.max_rate_deviation_ratio:
            rejections.append(
                issue(
                    FRAME_RATE_DEVIATION,
                    f"拟合帧率 {fitted:.4f} Hz 相对标称 {nominal_source_fps:g} Hz 偏差 "
                    f"{deviation * 100:+.3f}%，超过 ±{config.max_rate_deviation_ratio * 100:g}%",
                    fitted_hz=fitted,
                )
            )
    _check_effective_rate(times, float(nominal_source_fps), config, stats, rejections)
    return FrameSelection(
        SAME_RATE, float(nominal_source_fps), np.arange(count, dtype=np.int64),
        times, frame_numbers, None, rejections, warnings, stats,
    )


def select_high_rate(
    timestamps_ns: Sequence[int],
    nominal_source_fps: float,
    config: FrameSelectionConfig = FrameSelectionConfig(),
) -> FrameSelection:
    """高帧率抽帧路径：在精确 30 Hz 网格上取最近源图像，零复用。"""
    config.validate()
    times = np.asarray(timestamps_ns, dtype=np.int64)
    rejections: List[Dict] = []
    stats: Dict = {"selection_path": HIGH_RATE, "source_image_count": int(len(times))}
    empty = np.zeros(0, dtype=np.int64)
    if len(times) < 2:
        rejections.append(issue(INSUFFICIENT_IMAGES, f"episode 内只有 {len(times)} 张图像"))
        return FrameSelection(
            HIGH_RATE, float(nominal_source_fps), empty, empty, empty, empty,
            rejections, [], stats,
        )
    if np.any(np.diff(times) <= 0):
        rejections.append(
            issue(TIMESTAMP_NOT_INCREASING, "源图像时间戳不是严格递增，无法可靠选帧")
        )
        return FrameSelection(
            HIGH_RATE, float(nominal_source_fps), empty, empty, empty, empty,
            rejections, [], stats,
        )
    span = int(times[-1] - times[0])
    # 额外生成候选末点，避免向下舍入的合法纳秒端点被网格数量计算漏掉。
    grid_count = int((span * TARGET_FREQUENCY_HZ) // 1_000_000_000) + 2
    grid = times[0] + nominal_offset_ns(np.arange(grid_count, dtype=np.int64))
    grid = grid[grid <= times[-1]]
    insertion = np.searchsorted(times, grid, side="left")
    before = np.clip(insertion - 1, 0, len(times) - 1)
    after = np.clip(insertion, 0, len(times) - 1)
    # 距离相同时选时间较早的一帧。
    choose_after = (times[after] - grid) < (grid - times[before])
    positions = np.where(choose_after, after, before).astype(np.int64)
    errors = times[positions] - grid
    maximum_ns = (0.5 / float(nominal_source_fps) + config.high_rate_match_margin_s) * 1.0e9
    over = np.flatnonzero(np.abs(errors) > maximum_ns)
    for index in over[:20]:
        rejections.append(
            issue(
                MATCH_ERROR_EXCEEDED,
                f"网格点 {index} 匹配误差 {errors[index] / 1e6:+.3f} ms 超过 "
                f"{maximum_ns / 1e6:.3f} ms",
                grid_index=int(index),
                match_error_ms=float(errors[index] / 1.0e6),
            )
        )
    reused = np.zeros(len(positions), dtype=bool)
    reused[1:] = np.diff(positions) <= 0
    for index in np.flatnonzero(reused)[:20]:
        rejections.append(
            issue(
                FRAME_REUSE,
                f"网格点 {index} 复用或回退到源图像 {int(positions[index])}",
                grid_index=int(index),
            )
        )
    stats.update(
        {
            "grid_count": int(len(grid)),
            "max_match_error_ms": maximum_ns / 1.0e6,
            "match_error_ms": _percentiles(errors.astype(np.float64) / 1.0e6),
            "reuse_count": int(np.count_nonzero(reused)),
            "frame_drops": [],
            "dropped_frame_total": 0,
        }
    )
    _check_effective_rate(times, float(nominal_source_fps), config, stats, rejections)
    return FrameSelection(
        HIGH_RATE, float(nominal_source_fps), positions, times[positions],
        np.arange(len(positions), dtype=np.int64), errors.astype(np.int64),
        rejections, [], stats,
    )


def select_frames(
    timestamps_ns: Sequence[int],
    nominal_source_fps: float,
    config: FrameSelectionConfig = FrameSelectionConfig(),
) -> FrameSelection:
    """按标称源帧率分派到同频或高帧率路径。"""
    if selection_path_for(nominal_source_fps) == SAME_RATE:
        return select_same_rate(timestamps_ns, nominal_source_fps, config)
    return select_high_rate(timestamps_ns, nominal_source_fps, config)


def windows_crossing_drops(
    dropped_frames_before: np.ndarray, window_frames: int
) -> int:
    """统计长度为 ``window_frames`` 的连续行窗口中跨越丢帧点的窗口数。"""
    count = len(dropped_frames_before)
    if window_frames < 2 or count < window_frames:
        return 0
    marks = (np.asarray(dropped_frames_before) > 0).astype(np.int64)
    # 窗口 [s, s+W) 内部相邻行的跳变点为 marks[s+1 : s+W]。
    cumulative = np.concatenate(([0], np.cumsum(marks)))
    starts = np.arange(0, count - window_frames + 1)
    inside = cumulative[starts + window_frames] - cumulative[starts + 1]
    return int(np.count_nonzero(inside))
