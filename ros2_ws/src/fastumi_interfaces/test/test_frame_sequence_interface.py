"""验证相机帧序号 ROS 2 消息的公开字段。"""

from fastumi_interfaces.msg import FrameSequence, RecordingInfo, RecordingStatus
from fastumi_interfaces.srv import ListRecordings, StartRecording, StopRecording


def test_frame_sequence_exposes_header_and_sdk_sequence() -> None:
    """消息应携带可同步的 Header 和完整 64 位 SDK 帧序号。"""
    message = FrameSequence()

    message.frame_seqidx = (1 << 64) - 1

    assert message.header.frame_id == ""
    assert message.frame_seqidx == (1 << 64) - 1


def test_recording_interfaces_preserve_identity_and_pagination() -> None:
    """录制状态、操作响应和列表分页应共享稳定 UUID。"""
    info = RecordingInfo(
        recording_id="recording-1", dir_name="task", name="sample",
        relative_path="task/sample/episode_3", image_frames=42)
    status = RecordingStatus(
        state="saving", recording_id=info.recording_id, current=info)
    start = StartRecording.Request(dir_name="task", name="sample")
    stop = StopRecording.Request(recording_id=info.recording_id)
    page = ListRecordings.Request(offset=100, limit=100)

    assert status.current.relative_path == "task/sample/episode_3"
    assert status.current.image_frames == 42
    assert (start.dir_name, start.name) == ("task", "sample")
    assert stop.recording_id == status.recording_id
    assert (page.offset, page.limit) == (100, 100)
