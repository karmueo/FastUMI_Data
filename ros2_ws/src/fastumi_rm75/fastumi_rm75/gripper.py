"""定义归一化平行夹爪接口及标准 GripperCommand 映射。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def openness_to_position(
    openness: float, closed_position_m: float, open_position_m: float
) -> float:
    """把 `[0,1]` 归一化开度线性映射到夹爪位置米值。"""
    values = (openness, closed_position_m, open_position_m)
    if any(not np.isfinite(value) for value in values):
        raise ValueError("夹爪开度和位置端点必须为有限数值")
    if open_position_m <= closed_position_m:
        raise ValueError("张开位置必须大于闭合位置")
    normalized = float(np.clip(openness, 0.0, 1.0))
    return closed_position_m + normalized * (
        open_position_m - closed_position_m
    )


def position_to_openness(
    position_m: float, closed_position_m: float, open_position_m: float
) -> float:
    """把夹爪位置米值线性映射并裁剪到 `[0,1]` 归一化开度。"""
    # 复用正向映射对有限数值和端点顺序的校验。
    openness_to_position(0.0, closed_position_m, open_position_m)
    if not np.isfinite(position_m):
        raise ValueError("夹爪反馈位置必须为有限数值")
    openness = (position_m - closed_position_m) / (
        open_position_m - closed_position_m
    )
    return float(np.clip(openness, 0.0, 1.0))


class ParallelGripperAdapter(ABC):
    """定义与厂商协议无关的归一化平行夹爪控制契约。"""

    @abstractmethod
    def command(self, openness: float) -> None:
        """发送归一化开度目标。"""

    @abstractmethod
    def stop(self) -> None:
        """停止或取消当前夹爪动作。"""

    @abstractmethod
    def state_feedback(self) -> float | None:
        """返回最近一次归一化开度反馈；尚无反馈时返回 ``None``。"""

    @abstractmethod
    def calibrate_endpoints(
        self, closed_position_m: float, open_position_m: float
    ) -> None:
        """更新闭合和张开位置端点标定。"""


class MockParallelGripperAdapter(ParallelGripperAdapter):
    """仅记录命令、用于 dry-run 和单元测试的夹爪适配器。"""

    def __init__(self) -> None:
        """创建尚未接收命令的模拟适配器。"""
        self.last_openness: float | None = None
        self.stopped = False
        self.closed_position_m = 0.0
        self.open_position_m = 1.0

    def command(self, openness: float) -> None:
        """记录裁剪后的归一化开度。"""
        if not np.isfinite(openness):
            raise ValueError("夹爪开度必须为有限数值")
        self.last_openness = float(np.clip(openness, 0.0, 1.0))
        self.stopped = False

    def stop(self) -> None:
        """记录停止状态。"""
        self.stopped = True

    def state_feedback(self) -> float | None:
        """返回最近一次模拟开度反馈。"""
        return self.last_openness

    def calibrate_endpoints(
        self, closed_position_m: float, open_position_m: float
    ) -> None:
        """验证并保存模拟夹爪位置端点。"""
        openness_to_position(0.0, closed_position_m, open_position_m)
        self.closed_position_m = float(closed_position_m)
        self.open_position_m = float(open_position_m)
