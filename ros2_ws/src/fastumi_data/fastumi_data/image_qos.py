"""由配置的可靠性与队列深度构造图像订阅 QoS。"""

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from fastumi_data.image_modes import (
    RELIABILITY_RELIABLE,
    validate_image_qos_depth,
    validate_image_reliability,
)


def image_qos_profile(reliability: str, depth: int) -> QoSProfile:
    """构造图像订阅 QoS：默认 BEST_EFFORT 兼容现有相机发布端。

    Args:
        reliability: ``best_effort`` 或 ``reliable``。
        depth: KEEP_LAST 队列深度，须为正整数。
    """
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=validate_image_qos_depth(depth),
        reliability=(
            ReliabilityPolicy.RELIABLE
            if validate_image_reliability(reliability) == RELIABILITY_RELIABLE
            else ReliabilityPolicy.BEST_EFFORT
        ),
        durability=DurabilityPolicy.VOLATILE,
    )
