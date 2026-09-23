"""验证相机帧序号 ROS 2 消息的公开字段。"""

from fastumi_interfaces.msg import FrameSequence, RecordingInfo, RecordingStatus
from fastumi_interfaces.srv import (
    CancelRecordingRequest,
    DeleteRecording,
    GetRecordingRequest,
    GetTeleopGeneration,
    ListRecordings,
    SetTeleopGeneration,
    StartRecording,
    StopRecording,
)


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
    start = StartRecording.Request(
        request_id="11111111-1111-4111-8111-111111111111", dir_name="task", name="sample")
    cancel_request = CancelRecordingRequest.Request(request_id=start.request_id)
    get_request = GetRecordingRequest.Request(request_id=start.request_id)
    stop = StopRecording.Request(recording_id=info.recording_id)
    delete = DeleteRecording.Request(recording_id=info.recording_id)
    page = ListRecordings.Request(offset=100, limit=100)

    assert status.current.relative_path == "task/sample/episode_3"
    assert status.current.image_frames == 42
    assert (start.dir_name, start.name) == ("task", "sample")
    assert cancel_request.request_id == get_request.request_id == start.request_id
    assert stop.recording_id == status.recording_id
    assert delete.recording_id == status.recording_id
    assert (page.offset, page.limit) == (100, 100)


def test_teleop_generation_interfaces_share_highest_generation() -> None:
    """遥操启停和查询接口均使用无符号 64 位代次。"""
    set_request = SetTeleopGeneration.Request(operation_generation=(1 << 64) - 1)
    get_response = GetTeleopGeneration.Response(
        operation_generation=set_request.operation_generation, enabled=False)

    assert get_response.operation_generation == (1 << 64) - 1
    assert not get_response.enabled
