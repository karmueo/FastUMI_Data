"""管理夹爪预测、实测反馈与遥操启停之间的跟随和保持状态。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class GripperDecision:
    """描述一次控制周期的夹爪命令和状态变化。"""

    # 本周期应发布的归一化目标；None 表示不发布。
    command: Optional[float]
    # 本周期判定的跟随或保持原因。
    mode: str
    # 模式是否较上一周期发生变化。
    changed: bool


class GripperFollower:
    """仅用单调时钟判断输入新鲜度，并生成跟随或一次性保持目标。"""

    def __init__(
        self, estimate_timeout_s: float, feedback_timeout_s: float
    ) -> None:
        """校验两个超时阈值并创建尚无输入的跟随状态。"""
        # 预测与真实反馈允许的最长接收间隔，单位为秒。
        self.estimate_timeout_s = self._positive_timeout(
            estimate_timeout_s, "gripper_estimate_timeout_s"
        )
        self.feedback_timeout_s = self._positive_timeout(
            feedback_timeout_s, "gripper_feedback_timeout_s"
        )
        # 最近一次有效预测及其接收时刻。
        self._estimate: Optional[float] = None
        self._estimate_time = 0.0
        # 预测消息是否已到达、最近一帧是否有效。
        self._estimate_seen = False
        self._estimate_valid = False
        # 最近一次有效真实开度及其接收时刻。
        self._feedback: Optional[float] = None
        self._feedback_time = 0.0
        # 最近一帧真实反馈是否有效。
        self._feedback_valid = False
        # 最近一次控制模式及当前保持目标是否已经发送。
        self._mode: Optional[str] = None
        self._hold_sent = False

    @staticmethod
    def _positive_timeout(value: float, name: str) -> float:
        """返回正有限超时值，不合法时抛出 ValueError。"""
        timeout = float(value)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError(f"{name} 必须为正有限数")
        return timeout

    @staticmethod
    def _valid_openness(value: float) -> bool:
        """判断归一化开度是否为 [0, 1] 内的有限值。"""
        return math.isfinite(value) and 0.0 <= value <= 1.0

    @staticmethod
    def _fresh(now: float, received_at: float, timeout_s: float) -> bool:
        """判断单调时钟上的最近输入是否尚未超时。"""
        age = now - received_at
        return 0.0 <= age <= timeout_s

    def update_estimate(self, valid: bool, openness: float, now: float) -> bool:
        """保存有效预测；无效帧立即使此前预测失效。"""
        self._estimate_seen = True
        self._estimate_valid = bool(valid and self._valid_openness(openness))
        if self._estimate_valid:
            self._estimate = float(openness)
            self._estimate_time = now
        return self._estimate_valid

    def update_feedback(self, openness: float, now: float) -> bool:
        """保存有效实测开度；非法反馈使跟随立即停止。"""
        self._feedback_valid = self._valid_openness(openness)
        if self._feedback_valid:
            self._feedback = float(openness)
            self._feedback_time = now
        return self._feedback_valid

    def step(self, enabled: bool, now: float) -> GripperDecision:
        """生成跟随命令，或在进入保持状态时发送最近实测值一次。"""
        if not enabled:
            mode = "paused"
        elif self._feedback is None:
            mode = "feedback_missing"
        elif not self._feedback_valid:
            mode = "feedback_invalid"
        elif not self._fresh(now, self._feedback_time, self.feedback_timeout_s):
            mode = "feedback_timeout"
        elif not self._estimate_seen:
            mode = "estimate_missing"
        elif not self._estimate_valid:
            mode = "estimate_invalid"
        elif not self._fresh(now, self._estimate_time, self.estimate_timeout_s):
            mode = "estimate_timeout"
        else:
            mode = "following"

        changed = mode != self._mode
        self._mode = mode
        if mode == "following":
            self._hold_sent = False
            return GripperDecision(self._estimate, mode, changed)
        if self._feedback is not None and not self._hold_sent:
            self._hold_sent = True
            return GripperDecision(self._feedback, mode, changed)
        return GripperDecision(None, mode, changed)
