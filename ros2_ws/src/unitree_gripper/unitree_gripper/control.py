"""归一化开度映射和双侧夹爪的限速控制。位置单位为弧度。"""

import math
from typing import Dict, Optional


def ratio_to_rad(ratio: float, pos_close: float, pos_open: float) -> float:
    """将 [0, 1] 开度映射到电机位置，0 为闭合。"""
    return pos_close + ratio * (pos_open - pos_close)


def rad_to_ratio(position: float, pos_close: float, pos_open: float) -> float:
    """将电机位置转换为限制在 [0, 1] 的实际开度。"""
    return max(0.0, min(1.0, (position - pos_close) / (pos_open - pos_close)))


class GripperControl:
    """逐侧追踪反馈并产生限速、防堵转的位置命令。"""

    def __init__(
        self,
        sides=("left", "right"),
        pos_close=0.02,
        pos_open=5.0,
        max_speed_rad_s=6.0,
        max_hold_error_rad=0.30,
        feedback_timeout_s=0.5,
        target_ratio=1.0,
    ):
        if not all(math.isfinite(value) for value in (
            pos_close, pos_open, max_speed_rad_s, max_hold_error_rad,
            feedback_timeout_s, target_ratio,
        )):
            raise ValueError("夹爪参数必须是有限数值")
        if pos_open <= pos_close or max_speed_rad_s <= 0:
            raise ValueError("开闭位置和最大速度参数无效")
        if max_hold_error_rad <= 0 or feedback_timeout_s <= 0:
            raise ValueError("误差限制和反馈超时时间必须大于零")
        if not 0.0 <= target_ratio <= 1.0:
            raise ValueError("初始开度必须在 [0, 1] 内")
        self.sides = tuple(sides)
        self.pos_close = pos_close
        self.pos_open = pos_open
        self.max_speed_rad_s = max_speed_rad_s
        self.max_hold_error_rad = max_hold_error_rad
        self.feedback_timeout_s = feedback_timeout_s
        self.target_ratio = target_ratio
        self.actual_q: Dict[str, float] = {}
        self.cmd_q: Dict[str, float] = {}
        self.last_feedback: Dict[str, float] = {}

    def set_target(self, ratio: float) -> bool:
        """接受有限且位于 [0, 1] 的 ROS 命令。"""
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            return False
        self.target_ratio = ratio
        return True

    def update_feedback(self, side: str, position: float, now: float) -> bool:
        """记录真实电机位置；首帧或失联恢复时同步命令位置。"""
        if side not in self.sides or not math.isfinite(position):
            return False
        if side not in self.last_feedback or not self._fresh(side, now):
            self.cmd_q[side] = position
        self.actual_q[side] = position
        self.last_feedback[side] = now
        return True

    def _fresh(self, side: str, now: float) -> bool:
        return (
            side in self.last_feedback
            and 0.0 <= now - self.last_feedback[side] <= self.feedback_timeout_s
        )

    def next_commands(self, now: float, dt: float) -> Dict[str, float]:
        """只向有新鲜反馈的侧生成命令；dt 为单调时钟秒数。"""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("控制周期必须大于零")
        target_rad = ratio_to_rad(
            self.target_ratio, self.pos_close, self.pos_open
        )
        max_step = self.max_speed_rad_s * dt
        commands = {}
        for side in self.sides:
            if not self._fresh(side, now):
                continue
            actual = self.actual_q[side]
            current = self.cmd_q[side]
            diff = target_rad - current
            current += max(-max_step, min(max_step, diff))
            error = current - actual
            current = actual + max(
                -self.max_hold_error_rad,
                min(self.max_hold_error_rad, error),
            )
            self.cmd_q[side] = current
            commands[side] = current
        return commands

    def current_ratio(self, now: float) -> Optional[float]:
        """返回在线电机的平均真实开度；全部失联时返回 None。"""
        ratios = [
            rad_to_ratio(self.actual_q[side], self.pos_close, self.pos_open)
            for side in self.sides if self._fresh(side, now)
        ]
        return sum(ratios) / len(ratios) if ratios else None
