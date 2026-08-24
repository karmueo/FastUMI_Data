"""验证相机帧序号 ROS 2 消息的公开字段。"""

from fastumi_interfaces.msg import FrameSequence


def test_frame_sequence_exposes_header_and_sdk_sequence() -> None:
    """消息应携带可同步的 Header 和完整 64 位 SDK 帧序号。"""
    message = FrameSequence()

    message.frame_seqidx = (1 << 64) - 1

    assert message.header.frame_id == ""
    assert message.frame_seqidx == (1 << 64) - 1
