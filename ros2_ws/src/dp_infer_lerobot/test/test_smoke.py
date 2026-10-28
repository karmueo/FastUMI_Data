"""验证回放冒烟测试按接收时间拒绝过期及未来时间戳的消息，无需真实模型。"""

import numpy as np
import pytest

pytest.importorskip("rclpy")

from dp_infer.core import ActionSequence, InferenceContext  # noqa: E402
from dp_infer.node import sequence_message  # noqa: E402
from smoke_dp_infer_lerobot import check_message  # noqa: E402


@pytest.mark.parametrize("age_ns", [0, 500_000_000])
def test_message_age_accepts_boundaries(age_ns):
    """零年龄和恰好达到 0.5 秒上限的合法消息通过。"""
    message, stamp = _message()
    check_message(message, {stamp}, stamp + age_ns, 0.5)


@pytest.mark.parametrize("age_ns", [-1, 500_000_001])
def test_message_age_rejects_outside_limits(age_ns):
    """未来时间戳和超过上限一纳秒的消息触发失败。"""
    message, stamp = _message()
    with pytest.raises(AssertionError, match="Result age"):
        check_message(message, {stamp}, stamp + age_ns, 0.5)


def test_message_age_uses_custom_limit():
    """调用者可以收紧接收端时效上限。"""
    message, stamp = _message()
    with pytest.raises(AssertionError, match="Result age"):
        check_message(message, {stamp}, stamp + 200_000_000, 0.1)


def _message():
    """构造 base_link→Link7 的合法八点序列，返回消息和图像采集时间（纳秒）。"""
    stamp = 1_000_000_000
    sequence = ActionSequence(
        np.zeros((8, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (8, 1)),
        np.full(8, 0.5), np.arange(8) / 30)
    return sequence_message(sequence, InferenceContext(np.eye(4), stamp, 0, 1)), stamp
