"""评估图像、Tracker 和夹爪三路输入的新鲜度、有效性、匹配与时间指标。

新鲜度只依赖单调接收时钟，不依赖位姿是否变化，因此静止但持续发布的
Tracker 仍然健康。图像和有效夹爪状态另按完整的 ``rate_window_s`` 窗口分别计算
源时间戳速率和接收速率，两者都须达到期望频率的 ``min_rate_ratio``：开始前窗口
不足或未达标时阻止开始；录制中连续低于门限 ``low_rate_persist_s`` 时报警
``IMAGE_LOW_RATE``/``GRIPPER_LOW_RATE``，连续达标同样时长后清除。图像与夹爪按完全相同的源时间戳配对，Tracker 按最近源
时间戳在窗口内配对。到达时间差反映消息先后到达采集节点的间隔，包含传输
与排队，不应被解释为纯算法耗时。夹爪偶发单帧无效（如漏检一枚标签）不立即判为无效：
只有即使下一帧恢复有效、相邻有效样本的源时间间隔也将超过 ``gripper_invalid_persist_s``
时才报 ``GRIPPER_INVALID``，与离线转换的 ``max_gripper_gap_s`` 插值门限一致。
显式标定模式仅要求图像与 Tracker，夹爪仍可观测和录制，但不阻止开始或产生报警。
"""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field, replace
import math
from typing import Callable, Deque, Dict, List, Mapping, Optional, Tuple

from fastumi_data.image_modes import MODE_H264, MODE_RAW


# 三路输入的固定名称。
IMAGE = "image"
TRACKER = "tracker"
GRIPPER = "gripper"
STREAMS = (IMAGE, TRACKER, GRIPPER)

# 与 fastumi_interfaces/TrackerStatus 的 TRACKING_RUNNING_OK 一致。
TRACKING_RUNNING_OK = 3
# 未收到 Tracker 状态消息时使用的跟踪状态值。
TRACKING_UNKNOWN = 255
# 乱序报警在最近一次乱序之后保持的秒数。
OUT_OF_ORDER_HOLD_S = 2.0
# 每路最少保留的样本数量，用于匹配；高帧率或长窗口额外保留完整频率窗口。
HISTORY_SAMPLES = 256
# 从最新图像向前尝试配对的图像数量。
PAIR_LOOKBACK_IMAGES = 8
# 相邻源时间戳间隔超过近期中位间隔该倍数时判为疑似丢帧（单帧丢失约为 2 倍）。
GAP_FACTOR = 1.6
# 估计中位间隔所需的最少样本，以及保留的最近间隔数量。
GAP_MIN_INTERVALS = 8
GAP_HISTORY = 16
# 丢帧报警在最近一次疑似丢帧之后保持的秒数。
GAP_HOLD_S = 2.0


@dataclass(frozen=True)
class HealthConfig:
    """健康检查门限。"""

    image_timeout_s: float = 1.0
    tracker_timeout_s: float = 0.25
    gripper_timeout_s: float = 0.25
    match_window_s: float = 0.030
    rate_window_s: float = 2.0
    gripper_lag_warn_s: float = 0.25
    """夹爪最新源时间戳落后最新图像超过该值时报警。"""
    image_mode: str = MODE_RAW
    """图像消息模式；h264 时跟踪关键帧。"""
    expected_image_rate_hz: float = 0.0
    """编码图像输入的期望频率；0 表示不检查绝对速率。"""
    expected_gripper_rate_hz: float = 0.0
    """有效夹爪状态的期望频率；0 表示不检查绝对速率。"""
    min_rate_ratio: float = 0.95
    """源时间戳速率与接收速率相对期望频率的下限比例。"""
    low_rate_persist_s: float = 2.0
    """低频报警的触发与清除都要求条件连续保持的秒数。"""
    gripper_invalid_persist_s: float = 0.2
    """相邻有效夹爪样本允许的最大源时间间隔，超过才判为无效；0 表示单帧即无效。"""

    calibration_mode: bool = False
    """标定采集模式：夹爪为可选输入，正式任务默认要求全部三路输入。"""

    @property
    def required_streams(self) -> Tuple[str, ...]:
        """返回当前模式必须通过预检和持续健康检查的输入。"""
        return (IMAGE, TRACKER) if self.calibration_mode else STREAMS

    def expected_rate_hz(self, stream: str) -> float:
        """返回需检查绝对速率的输入的期望频率，其余为 0。"""
        return {
            IMAGE: self.expected_image_rate_hz,
            GRIPPER: self.expected_gripper_rate_hz,
        }.get(stream, 0.0)

    def timeout_s(self, stream: str) -> float:
        """返回指定输入的新鲜度超时。"""
        return {
            IMAGE: self.image_timeout_s,
            TRACKER: self.tracker_timeout_s,
            GRIPPER: self.gripper_timeout_s,
        }[stream]


@dataclass(frozen=True)
class PreflightIssue:
    """开始采集前检查发现的单个阻塞原因。"""

    code: str
    message: str


@dataclass(frozen=True)
class StreamHealth:
    """单路输入的一次健康快照；浮点指标为 NaN 表示不可用。"""

    name: str
    discovered: bool
    fresh: bool
    valid: bool
    matched: bool
    age_s: float
    rate_hz: float
    source_delta_ms: float
    message_age_ms: float
    pair_arrival_delta_ms: float
    received: int
    out_of_order: int
    detail: str
    format: str = ""
    """图像流最近一帧的格式简述，例如 ``h264 1280x960``；其他流为空。"""
    awaiting_keyframe: bool = False
    """仅 h264 图像流：尚未收到含 SPS/PPS 的 IDR 关键帧，无法开始解码。"""
    keyframes: int = 0
    """仅 h264 图像流：已收到的关键帧数量。"""
    source_rate_hz: float = math.nan
    """完整窗口内有效样本的源时间戳速率 ``(N−1)/(t_last − t_first)``。"""
    receive_rate_hz: float = math.nan
    """完整窗口内有效样本的接收速率（个数 / 窗口秒数）。"""
    rate_window_complete: bool = False
    """是否已积累完整频率窗口。"""
    expected_rate_hz: float = 0.0
    """期望频率；0 表示该路不检查绝对速率。"""
    below_rate: bool = False
    """当前窗口的源或接收速率低于期望频率的门限。"""


@dataclass(frozen=True)
class HealthSnapshot:
    """三路输入加跨流指标的快照。"""

    image: StreamHealth
    tracker: StreamHealth
    gripper: StreamHealth
    gripper_percent: float
    gripper_valid: bool
    tracker_tracking_ok: bool
    tracker_tracking_state: int
    gripper_lag_ms: float
    alarms: Tuple[str, ...]

    def stream(self, name: str) -> StreamHealth:
        """按名称取得单路快照。"""
        return {IMAGE: self.image, TRACKER: self.tracker, GRIPPER: self.gripper}[name]


@dataclass
class _Sample:
    """一条已接收消息的源时间戳和单调接收时间。"""

    source_ns: int
    recv_ns: int
    valid: bool = True


@dataclass
class _StreamState:
    """单路输入的累积状态。"""

    samples: Deque[_Sample] = field(default_factory=deque)
    # 按接收时间保留的频率窗口长度，与样本数量下限共同限制历史淘汰。
    history_window_ns: int = 2_000_000_000
    # 首条消息的单调接收时间，独立于可淘汰的匹配历史；0 是合法时刻。
    first_receive_ns: Optional[int] = None
    received: int = 0
    out_of_order: int = 0
    last_out_of_order_ns: Optional[int] = None
    gaps: int = 0
    last_gap_ns: Optional[int] = None
    intervals: Deque[int] = field(default_factory=lambda: deque(maxlen=GAP_HISTORY))
    valid: bool = False
    detail: str = "尚未收到数据"
    # 最近一条有效样本的源时间戳，不随匹配历史淘汰。
    last_valid_source_ns: Optional[int] = None

    def observe(self, source_ns: int, recv_ns: int, valid: bool = True) -> None:
        """记录一条消息并检测源时间戳回退。"""
        if self.samples and source_ns < self.samples[-1].source_ns:
            self.out_of_order += 1
            self.last_out_of_order_ns = recv_ns
            # 回退前的有效时间戳不能作为新时间序列的插值基准。
            self.last_valid_source_ns = None
        elif self.samples:
            interval = source_ns - self.samples[-1].source_ns
            if interval > 0:
                if len(self.intervals) >= GAP_MIN_INTERVALS:
                    median = sorted(self.intervals)[len(self.intervals) // 2]
                    if interval > GAP_FACTOR * median:
                        # 间隔异常大：记为丢帧，且不计入基准间隔，避免抬高中位数。
                        self.gaps += 1
                        self.last_gap_ns = recv_ns
                        interval = 0
                if interval > 0:
                    self.intervals.append(interval)
        self.samples.append(_Sample(source_ns, recv_ns, bool(valid)))
        if valid:
            self.last_valid_source_ns = source_ns
        if self.first_receive_ns is None:
            self.first_receive_ns = recv_ns
        # 当前消息对应的频率窗口起点；匹配所需的最小历史仍保留。
        cutoff_ns = recv_ns - self.history_window_ns
        while (
            len(self.samples) > HISTORY_SAMPLES
            and self.samples[0].recv_ns <= cutoff_ns
        ):
            self.samples.popleft()
        self.received += 1


def preflight_issues(
    snapshot: HealthSnapshot, config: HealthConfig
) -> List[PreflightIssue]:
    """根据快照列出开始采集前的阻塞原因：未发现、数据过期或内容无效。"""
    labels = {IMAGE: "图像", TRACKER: "Tracker", GRIPPER: "夹爪"}
    issues: List[PreflightIssue] = []
    for name in config.required_streams:
        stream = snapshot.stream(name)
        label = labels[name]
        if not stream.discovered:
            issues.append(
                PreflightIssue(f"{name.upper()}_NOT_DISCOVERED", f"{label}话题没有发布者")
            )
        elif not stream.fresh:
            issues.append(
                PreflightIssue(
                    f"{name.upper()}_STALE",
                    f"{label}数据已超过 {config.timeout_s(name):g} s 未更新",
                )
            )
        elif not stream.valid:
            issues.append(
                PreflightIssue(f"{name.upper()}_INVALID", f"{label}数据无效: {stream.detail}")
            )
        elif stream.expected_rate_hz > 0.0 and not stream.rate_window_complete:
            issues.append(
                PreflightIssue(
                    f"{name.upper()}_RATE_WINDOW_INCOMPLETE",
                    f"{label}正在积累 {config.rate_window_s:g} s 频率窗口",
                )
            )
        elif stream.below_rate:
            threshold = config.min_rate_ratio * stream.expected_rate_hz
            issues.append(
                PreflightIssue(
                    f"{name.upper()}_LOW_RATE",
                    f"{label}源时间戳速率 {stream.source_rate_hz:.2f} Hz、接收速率 "
                    f"{stream.receive_rate_hz:.2f} Hz，低于 {threshold:.2f} Hz",
                )
            )
    return issues


class CollectionHealthMonitor:
    """线程不安全的健康状态机；调用方负责串行化访问。"""

    def __init__(
        self,
        config: HealthConfig,
        mono_clock_ns: Callable[[], int],
        wall_clock_ns: Callable[[], int],
    ) -> None:
        """保存门限和时钟。

        Args:
            config: 超时和匹配门限。
            mono_clock_ns: 单调时钟，用于新鲜度、频率和到达时间差。
            wall_clock_ns: 与消息源时间戳同一时钟域的墙钟，用于消息年龄。
        """
        self._config = config
        self._mono_ns = mono_clock_ns
        self._wall_ns = wall_clock_ns
        self._streams: Dict[str, _StreamState] = {
            name: _StreamState(history_window_ns=int(config.rate_window_s * 1e9))
            for name in STREAMS
        }
        self._tracker_status_ns: Optional[int] = None
        self._tracker_status_ok = False
        self._tracker_status_detail = "尚未收到 Tracker 状态"
        self._tracker_state = TRACKING_UNKNOWN
        self._gripper_percent = math.nan
        self._image_format = ""
        self._keyframes = 0
        # 低频报警状态机：条件开始成立/恢复的单调时间及当前是否报警。
        self._low_since: Dict[str, Optional[int]] = {name: None for name in STREAMS}
        self._ok_since: Dict[str, Optional[int]] = {name: None for name in STREAMS}
        self._low_alarm: Dict[str, bool] = {name: False for name in STREAMS}

    @property
    def config(self) -> HealthConfig:
        """当前健康门限。"""
        return self._config

    def observe_image(
        self,
        source_ns: int,
        valid: bool,
        detail: str = "",
        label: str = "",
        keyframe: bool = False,
    ) -> None:
        """记录一帧图像；无效帧也计入接收但标记内容异常。

        Args:
            label: 格式简述，供健康报告显示。
            keyframe: h264 包含 SPS/PPS 和 IDR，可作为解码起点。
        """
        state = self._streams[IMAGE]
        state.observe(source_ns, self._mono_ns(), valid)
        state.valid = valid
        state.detail = "" if valid else (detail or "图像尺寸或数据长度异常")
        if label:
            self._image_format = label
        if keyframe:
            self._keyframes += 1

    def observe_tracker_pose(
        self, source_ns: int, finite: bool, detail: str = ""
    ) -> None:
        """记录一条 Tracker 位姿；位姿含 NaN 或四元数异常时 ``finite`` 为假。"""
        state = self._streams[TRACKER]
        state.observe(source_ns, self._mono_ns())
        state.valid = finite
        state.detail = "" if finite else (detail or "Tracker 位姿含非有限值")

    def observe_tracker_status(
        self,
        device_connected: bool,
        pose_valid: bool,
        tracking_state: int,
    ) -> None:
        """记录 Tracker 状态，用于判断跟踪是否处于 RUNNING_OK。"""
        self._tracker_status_ns = self._mono_ns()
        self._tracker_state = tracking_state
        if not device_connected:
            self._tracker_status_ok, self._tracker_status_detail = (
                False,
                "Tracker 设备未连接",
            )
        elif not pose_valid:
            self._tracker_status_ok, self._tracker_status_detail = (
                False,
                "Tracker 位姿无效",
            )
        elif tracking_state != TRACKING_RUNNING_OK:
            self._tracker_status_ok, self._tracker_status_detail = (
                False,
                f"OpenVR 跟踪状态为 {tracking_state}，不是 RUNNING_OK",
            )
        else:
            self._tracker_status_ok, self._tracker_status_detail = True, ""

    def observe_gripper(
        self, source_ns: int, valid: bool, openness: float
    ) -> None:
        """记录一条夹爪状态；无效或 NaN 开合度视为内容异常。"""
        state = self._streams[GRIPPER]
        usable = bool(valid) and math.isfinite(openness)
        state.observe(source_ns, self._mono_ns(), usable)
        state.valid = usable
        state.detail = "" if usable else "夹爪估计无效或开合度为 NaN"
        self._gripper_percent = (
            max(0.0, min(100.0, openness * 100.0)) if usable else math.nan
        )

    def snapshot(self, discovered: Mapping[str, bool]) -> HealthSnapshot:
        """生成当前健康快照。

        Args:
            discovered: 每路输入当前是否存在发布者。
        """
        now_mono = self._mono_ns()
        now_wall = self._wall_ns()
        pairs = {
            GRIPPER: self._pair_gripper(),
            TRACKER: self._pair_tracker(),
        }
        streams: Dict[str, StreamHealth] = {}
        for name in STREAMS:
            streams[name] = self._stream_health(
                name, bool(discovered.get(name, False)), now_mono, now_wall,
                pairs.get(name),
            )
        image = streams[IMAGE]
        tracker = streams[TRACKER]
        gripper = streams[GRIPPER]
        # 图像是配对基准；标定模式只要求 Tracker 配对。
        streams[IMAGE] = replace(
            image,
            matched=tracker.matched and (self._config.calibration_mode or gripper.matched),
        )
        lag_ms = self._gripper_lag_ms()
        alarms = self._alarms(streams, lag_ms, now_mono)
        # Tracker 的 valid 已包含状态消息的 RUNNING_OK 判断。
        tracking_ok = tracker.fresh and tracker.valid
        return HealthSnapshot(
            image=streams[IMAGE],
            tracker=tracker,
            gripper=gripper,
            gripper_percent=self._gripper_percent if gripper.fresh else math.nan,
            # 开合度与有效标志只反映最新一帧，不受无效持续时间容忍影响。
            gripper_valid=gripper.fresh and self._streams[GRIPPER].valid,
            tracker_tracking_ok=tracking_ok,
            tracker_tracking_state=self._tracker_state,
            gripper_lag_ms=lag_ms,
            alarms=tuple(sorted(alarms)),
        )

    def _gripper_invalid_tolerated(self, state: _StreamState) -> bool:
        """最新夹爪样本无效，但下一帧若恢复有效，有效样本间隔仍不超过门限。

        预计间隔为最近有效样本到最新样本的源时间跨度，加上最近一个帧间隔。
        """
        samples = state.samples
        if (
            self._config.gripper_invalid_persist_s <= 0.0
            or state.last_valid_source_ns is None
            or len(samples) < 2
        ):
            return False
        span_ns = samples[-1].source_ns - state.last_valid_source_ns
        if span_ns < 0:
            return False
        frame_ns = max(0, samples[-1].source_ns - samples[-2].source_ns)
        return span_ns + frame_ns <= self._config.gripper_invalid_persist_s * 1e9

    def _fresh(self, name: str, now_mono: int) -> bool:
        """最近一次接收距现在不超过该路超时。"""
        state = self._streams[name]
        if not state.samples:
            return False
        age_s = (now_mono - state.samples[-1].recv_ns) / 1e9
        return age_s <= self._config.timeout_s(name)

    def _rate_hz(self, name: str, now_mono: int) -> float:
        """用滑动窗口内首尾接收时间估计频率。"""
        window_ns = int(self._config.rate_window_s * 1e9)
        recent = [
            sample.recv_ns
            for sample in self._streams[name].samples
            if now_mono - sample.recv_ns <= window_ns
        ]
        if len(recent) < 2 or recent[-1] == recent[0]:
            return math.nan
        return (len(recent) - 1) / ((recent[-1] - recent[0]) / 1e9)

    def _window_rates(self, name: str, now_mono: int) -> Tuple[bool, float, float]:
        """返回 ``(窗口是否完整, 接收速率, 源时间戳速率)``，只统计有效样本。"""
        window_ns = int(self._config.rate_window_s * 1e9)
        # 首次接收时刻独立保留，避免历史淘汰后窗口重新变为未满。
        state = self._streams[name]
        samples = state.samples
        complete = (
            state.first_receive_ns is not None
            and now_mono - state.first_receive_ns >= window_ns
        )
        recent = [
            sample for sample in samples
            if sample.valid and now_mono - sample.recv_ns < window_ns
        ]
        receive = len(recent) / self._config.rate_window_s if window_ns > 0 else math.nan
        sources = sorted({sample.source_ns for sample in recent})
        source = (
            (len(sources) - 1) / ((sources[-1] - sources[0]) / 1e9)
            if len(sources) >= 2 and sources[-1] > sources[0]
            else 0.0
        )
        return complete, receive, source

    def _update_low_rate(self, name: str, below: bool, now_mono: int) -> None:
        """按连续保持时长更新低频报警：触发与清除都需持续 ``low_rate_persist_s``。"""
        persist_ns = int(self._config.low_rate_persist_s * 1e9)
        if below:
            self._ok_since[name] = None
            if self._low_since[name] is None:
                self._low_since[name] = now_mono
            if now_mono - self._low_since[name] >= persist_ns:
                self._low_alarm[name] = True
        else:
            self._low_since[name] = None
            if self._low_alarm[name]:
                if self._ok_since[name] is None:
                    self._ok_since[name] = now_mono
                if now_mono - self._ok_since[name] >= persist_ns:
                    self._low_alarm[name] = False
                    self._ok_since[name] = None

    def _pair_gripper(self) -> Optional[Tuple[_Sample, _Sample]]:
        """从最新夹爪样本向前，在图像历史中查找源时间戳完全相同的帧。

        以夹爪为起点可在夹爪处理滞后时仍配对成功，滞后量体现在到达时间差
        和 ``GRIPPER_LAG`` 报警中，而不是被误判为匹配缺失。
        """
        image_by_stamp = {s.source_ns: s for s in self._streams[IMAGE].samples}
        grippers = list(self._streams[GRIPPER].samples)[-PAIR_LOOKBACK_IMAGES:]
        for gripper in reversed(grippers):
            image = image_by_stamp.get(gripper.source_ns)
            if image is not None:
                return image, gripper
        return None

    def _pair_tracker(self) -> Optional[Tuple[_Sample, _Sample]]:
        """在最近图像中查找窗口内最近的 Tracker 位姿。"""
        tracker_samples = sorted(
            self._streams[TRACKER].samples, key=lambda s: s.source_ns
        )
        if not tracker_samples:
            return None
        stamps = [sample.source_ns for sample in tracker_samples]
        window_ns = int(self._config.match_window_s * 1e9)
        images = list(self._streams[IMAGE].samples)[-PAIR_LOOKBACK_IMAGES:]
        for image in reversed(images):
            index = bisect_left(stamps, image.source_ns)
            candidates = [
                tracker_samples[i] for i in (index - 1, index)
                if 0 <= i < len(tracker_samples)
            ]
            best = min(
                candidates, key=lambda s: abs(s.source_ns - image.source_ns)
            )
            if abs(best.source_ns - image.source_ns) <= window_ns:
                return image, best
        return None

    def _gripper_lag_ms(self) -> float:
        """最新图像源时间戳减最新夹爪源时间戳，正值表示夹爪落后。"""
        images = self._streams[IMAGE].samples
        grippers = self._streams[GRIPPER].samples
        if not images or not grippers:
            return math.nan
        return (images[-1].source_ns - grippers[-1].source_ns) / 1e6

    def _stream_health(
        self,
        name: str,
        discovered: bool,
        now_mono: int,
        now_wall: int,
        pair: Optional[Tuple[_Sample, _Sample]],
    ) -> StreamHealth:
        """组装单路快照，配对指标仅在匹配成功时可用。"""
        state = self._streams[name]
        fresh = self._fresh(name, now_mono)
        valid, detail = state.valid, state.detail
        if name == TRACKER:
            status_fresh = (
                self._tracker_status_ns is not None
                and (now_mono - self._tracker_status_ns) / 1e9
                <= self._config.tracker_timeout_s
            )
            if valid and not status_fresh:
                valid, detail = False, "Tracker 状态消息已过期或尚未收到"
            elif valid and not self._tracker_status_ok:
                valid, detail = False, self._tracker_status_detail
        elif name == GRIPPER and not valid and self._gripper_invalid_tolerated(state):
            valid, detail = True, ""
        age_s = (
            (now_mono - state.samples[-1].recv_ns) / 1e9 if state.samples else -1.0
        )
        message_age_ms = (
            (now_wall - state.samples[-1].source_ns) / 1e6
            if state.samples
            else math.nan
        )
        source_delta = arrival_delta = math.nan
        matched = False
        if pair is not None:
            image_sample, other = pair
            matched = True
            source_delta = abs(other.source_ns - image_sample.source_ns) / 1e6
            arrival_delta = (other.recv_ns - image_sample.recv_ns) / 1e6
        image_h264 = name == IMAGE and self._config.image_mode == MODE_H264
        expected = self._config.expected_rate_hz(name)
        complete, receive_rate, source_rate = self._window_rates(name, now_mono)
        threshold = self._config.min_rate_ratio * expected
        below = (
            expected > 0.0 and complete
            and (receive_rate < threshold or source_rate < threshold)
        )
        if expected > 0.0:
            self._update_low_rate(name, below, now_mono)
        return StreamHealth(
            name=name,
            discovered=discovered,
            fresh=fresh,
            valid=valid,
            matched=matched,
            age_s=age_s,
            rate_hz=self._rate_hz(name, now_mono),
            source_delta_ms=source_delta,
            message_age_ms=message_age_ms,
            pair_arrival_delta_ms=arrival_delta,
            received=state.received,
            out_of_order=state.out_of_order,
            detail=detail,
            format=self._image_format if name == IMAGE else "",
            awaiting_keyframe=image_h264 and self._keyframes == 0,
            keyframes=self._keyframes if image_h264 else 0,
            source_rate_hz=source_rate if complete else math.nan,
            receive_rate_hz=receive_rate if complete else math.nan,
            rate_window_complete=complete,
            expected_rate_hz=expected,
            below_rate=below,
        )

    def _alarms(
        self,
        streams: Mapping[str, StreamHealth],
        gripper_lag_ms: float,
        now_mono: int,
    ) -> List[str]:
        """根据快照派生当前活动报警码。"""
        alarms: List[str] = []
        for name in self._config.required_streams:
            stream = streams[name]
            prefix = name.upper()
            if not stream.discovered:
                alarms.append(f"{prefix}_NOT_DISCOVERED")
            elif not stream.fresh:
                alarms.append(f"{prefix}_STALE")
            elif not stream.valid:
                alarms.append(f"{prefix}_INVALID")
            last_unordered = self._streams[name].last_out_of_order_ns
            if (
                last_unordered is not None
                and (now_mono - last_unordered) / 1e9 <= OUT_OF_ORDER_HOLD_S
            ):
                alarms.append(f"{prefix}_OUT_OF_ORDER")
            last_gap = self._streams[name].last_gap_ns
            if last_gap is not None and (now_mono - last_gap) / 1e9 <= GAP_HOLD_S:
                alarms.append(f"{prefix}_GAP")
            if self._low_alarm[name]:
                alarms.append(f"{prefix}_LOW_RATE")
        image = streams[IMAGE]
        for other in self._config.required_streams[1:]:
            stream = streams[other]
            if image.fresh and stream.fresh and not stream.matched:
                alarms.append(f"{other.upper()}_UNMATCHED")
        if (
            not self._config.calibration_mode
            and image.fresh
            and streams[GRIPPER].fresh
            and math.isfinite(gripper_lag_ms)
            and gripper_lag_ms > self._config.gripper_lag_warn_s * 1e3
        ):
            alarms.append("GRIPPER_LAG")
        return alarms
