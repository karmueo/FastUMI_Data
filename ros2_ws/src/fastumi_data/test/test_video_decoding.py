"""验证 JPEG/PNG 与 H.264 共享解码状态。"""

import numpy as np
import pytest

from fastumi_data.image_messages import (
    packet_from_compressed_message,
    packet_from_ffmpeg_message,
    packet_from_message,
)
from fastumi_data.video_decoding import (
    DecodeReadiness,
    EncodedPacket,
    H264Decoder,
    StillImageDecoder,
    VideoDecodeError,
    create_decoder,
)
from video_fixtures import (
    band_column,
    compressed_message,
    ffmpeg_message,
    frame_pixels_bgr,
    h264_packets,
    stamp_of,
)


def _h264_packet(index: int, data: bytes, bag_ns: int = 0) -> EncodedPacket:
    """构造带源时间的 H.264 编码包。"""
    return packet_from_ffmpeg_message(ffmpeg_message(index, data), bag_ns)


def test_still_decoder_keeps_source_time_and_channel_order() -> None:
    """JPEG/PNG 解码保留 Header、bag 时间，RGB 与 BGR 互为通道翻转。"""
    for codec in ("jpeg", "png"):
        packet = packet_from_compressed_message(compressed_message(5, codec), 777)
        bgr = StillImageDecoder("bgr").feed(packet)[0]
        rgb = StillImageDecoder("rgb").feed(packet)[0]

        assert (bgr.stamp_ns, bgr.frame_id, bgr.bag_timestamp_ns) == (stamp_of(5), "cam", 777)
        assert bgr.color_order == "bgr" and rgb.color_order == "rgb"
        assert np.array_equal(bgr.image, rgb.image[:, :, ::-1])
        assert abs(band_column(bgr.image) - band_column(frame_pixels_bgr(5))) <= 1


def test_still_decoder_rejects_corrupt_data() -> None:
    """非图像数据和损坏的 JPEG 都明确失败。"""
    decoder = StillImageDecoder()
    garbage = packet_from_compressed_message(compressed_message(0))
    garbage = EncodedPacket(garbage.mode, b"nonsense", garbage.stamp_ns)
    with pytest.raises(VideoDecodeError, match="不是 JPEG 或 PNG"):
        decoder.feed(garbage)

    good = packet_from_compressed_message(compressed_message(0))
    truncated = EncodedPacket(good.mode, good.data[:20], good.stamp_ns)
    with pytest.raises(VideoDecodeError, match="解码失败"):
        decoder.feed(truncated)


def test_h264_decodes_from_the_first_packet_with_exact_time_mapping() -> None:
    """从关键帧开始的流每包输出一帧，源时间与帧内容一一对应。"""
    decoder = H264Decoder()
    frames = []
    for index, data in enumerate(h264_packets(25)):
        output = decoder.feed(_h264_packet(index, data, bag_ns=1000 + index))
        assert len(output) == 1
        frames.extend(output)
    frames.extend(decoder.drain())

    assert [f.stamp_ns for f in frames] == [stamp_of(i) for i in range(25)]
    assert [f.bag_timestamp_ns for f in frames] == [1000 + i for i in range(25)]
    assert all(f.frame_id == "cam" and f.image.shape == (48, 64, 3) for f in frames)
    for index, frame in enumerate(frames):
        assert abs(band_column(frame.image) - band_column(frame_pixels_bgr(index))) <= 2
    assert decoder.stats.frames == 25 and decoder.stats.keyframes == 3
    assert decoder.stats.skipped_before_keyframe == 0


def test_h264_starting_mid_gop_skips_prefix_until_keyframe() -> None:
    """从 GOP 中段开始时，关键帧之前的包被跳过并计数，之后正常输出。"""
    packets = h264_packets(25, gop_size=10)
    decoder = H264Decoder()
    frames = []
    for index in range(4, 25):
        frames.extend(decoder.feed(_h264_packet(index, packets[index])))

    assert decoder.stats.skipped_before_keyframe == 6
    assert [f.stamp_ns for f in frames] == [stamp_of(i) for i in range(10, 25)]
    assert abs(band_column(frames[0].image) - band_column(frame_pixels_bgr(10))) <= 2


def test_h264_rgb_and_bgr_outputs_are_consistent() -> None:
    """同一码流的 RGB 输出等于 BGR 输出翻转通道。"""
    packets = h264_packets(3)
    bgr, rgb = H264Decoder("bgr"), H264Decoder("rgb")
    for index, data in enumerate(packets):
        left = bgr.feed(_h264_packet(index, data))[0].image
        right = rgb.feed(_h264_packet(index, data))[0].image
        assert np.array_equal(left, right[:, :, ::-1])


def test_h264_rejects_b_frames() -> None:
    """含 B 帧的码流报告原因并失败。"""
    decoder = H264Decoder()
    with pytest.raises(VideoDecodeError, match="B 帧"):
        for index, data in enumerate(h264_packets(12, b_frames=2)):
            decoder.feed(_h264_packet(index, data))


def test_h264_rejects_non_annex_b_data() -> None:
    """没有起始码的数据明确失败，而不是被当作启动前缀跳过。"""
    with pytest.raises(VideoDecodeError, match="Annex B"):
        H264Decoder().feed(
            EncodedPacket("h264", b"\x00\x00\x00\x05\x65\x88\x84\x00\x10", 1)
        )


def test_h264_corrupt_or_missing_packet_fails_explicitly() -> None:
    """损坏的 slice 或丢失参考包不会静默产出错误数量的帧。"""
    packets = h264_packets(12)
    decoder = H264Decoder()
    decoder.feed(_h264_packet(0, packets[0]))
    corrupt = packets[1][:12] + bytes(len(packets[1]) - 12)

    with pytest.raises(VideoDecodeError):
        for index in range(1, 6):
            data = corrupt if index == 1 else packets[index]
            decoder.feed(_h264_packet(index, data))


def test_h264_reset_waits_for_next_keyframe_and_drain_is_idempotent() -> None:
    """重置丢弃参考帧，随后的非关键帧被跳过；排空后可重新开始。"""
    packets = h264_packets(25)
    decoder = H264Decoder()
    for index in range(3):
        decoder.feed(_h264_packet(index, packets[index]))
    decoder.reset()

    assert not decoder.started
    assert decoder.feed(_h264_packet(3, packets[3])) == []
    assert decoder.feed(_h264_packet(10, packets[10]))[0].stamp_ns == stamp_of(10)
    assert decoder.drain() == []
    assert decoder.drain() == []
    assert decoder.feed(_h264_packet(11, packets[11])) == []


def test_create_decoder_by_mode() -> None:
    """工厂按模式返回解码器，raw 不需要解码器。"""
    assert isinstance(create_decoder("compressed"), StillImageDecoder)
    assert isinstance(create_decoder("h264", "rgb"), H264Decoder)
    with pytest.raises(ValueError, match="raw"):
        create_decoder("raw")


def test_message_adapters_read_header_and_payload() -> None:
    """适配器取出 Header 时间、frame_id 和字节数据。"""
    message = compressed_message(2)
    packet = packet_from_message(message, "compressed", 99)
    assert (packet.stamp_ns, packet.frame_id, packet.bag_timestamp_ns) == (stamp_of(2), "cam", 99)
    assert packet.data == bytes(message.data)
    with pytest.raises(ValueError, match="没有编码包适配器"):
        packet_from_message(message, "raw")


def test_readiness_for_h264_requires_a_decodable_keyframe() -> None:
    """H.264 门控在关键帧成功解码前保持未就绪并说明原因，之后不再解码。"""
    packets = h264_packets(25)
    readiness = DecodeReadiness("h264")
    assert not readiness.ready and "关键帧" in readiness.status

    readiness.feed(_h264_packet(4, packets[4]))
    assert not readiness.ready
    readiness.feed(_h264_packet(10, packets[10]))
    assert readiness.ready and readiness.status == ""


def test_readiness_records_decode_errors_and_recovers() -> None:
    """门控遇到损坏数据时记录原因并重置，下一关键帧后恢复。"""
    packets = h264_packets(25)
    readiness = DecodeReadiness("h264")
    readiness.feed(_h264_packet(0, packets[0][:30]))
    assert not readiness.ready

    readiness.feed(_h264_packet(10, packets[10]))
    assert readiness.ready


def test_readiness_for_compressed_and_raw() -> None:
    """压缩图像在首次成功解码后就绪，raw 无需门控。"""
    assert DecodeReadiness("raw").ready
    readiness = DecodeReadiness("compressed")
    assert not readiness.ready
    readiness.feed(packet_from_compressed_message(compressed_message(0)))
    assert readiness.ready
