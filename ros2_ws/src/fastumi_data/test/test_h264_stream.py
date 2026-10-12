"""验证 Annex B 访问单元分析：关键帧、非关键帧和 B 帧。"""

from fastumi_data.h264_stream import analyze_annex_b
from video_fixtures import h264_packets


def test_gop_structure_is_detected() -> None:
    """GOP 起点同时含 SPS/PPS/IDR，其余包只含非 IDR slice。"""
    packets = h264_packets(25, gop_size=10)
    infos = [analyze_annex_b(packet) for packet in packets]

    assert [info.keyframe for info in infos].count(True) == 3
    assert infos[0].keyframe and infos[10].keyframe and infos[20].keyframe
    assert not infos[1].keyframe
    assert not infos[1].has_sps and not infos[1].has_idr
    assert not any(info.has_b_slice for info in infos)


def test_b_slices_are_detected() -> None:
    """含 B 帧的码流至少有一个包被标记为 B slice。"""
    infos = [analyze_annex_b(packet) for packet in h264_packets(12, b_frames=2)]

    assert any(info.has_b_slice for info in infos)


def test_non_annex_b_data_is_flagged() -> None:
    """没有起始码的数据（例如 AVCC 长度前缀）不是 Annex B。"""
    info = analyze_annex_b(b"\x00\x00\x00\x05\x65\x88\x84\x00\x10")

    assert not info.annex_b
    assert not info.keyframe
    assert not analyze_annex_b(b"").annex_b


def test_missing_parameter_sets_are_not_keyframes() -> None:
    """只有 IDR 而没有 SPS/PPS 的包不能作为独立解码起点。"""
    packets = h264_packets(3)
    idr_start = packets[0].find(b"\x00\x00\x01\x65")
    assert idr_start > 0
    idr_only = analyze_annex_b(packets[0][idr_start:])

    assert idr_only.has_idr
    assert not idr_only.keyframe
