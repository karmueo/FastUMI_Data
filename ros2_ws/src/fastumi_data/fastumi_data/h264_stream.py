"""分析 Annex B 格式 H.264 访问单元，判断关键帧和 B 帧。

相机协议保证每个 ``FFMPEGPacket`` 恰好含一个访问单元。本模块只扫描 NAL
起始码并读取首个 slice 的 ``slice_type``，不解码像素，因此可在采集回调中
对每个包低成本调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union


# H.264 NAL 单元类型。
NAL_SLICE = 1
NAL_IDR = 5
NAL_SPS = 7
NAL_PPS = 8
# slice_type 取模 5 后的 B 类型。
_SLICE_TYPE_B = 1
# 读取 slice 头所需的最大字节数；first_mb 与 slice_type 的指数哥伦布码很短。
_SLICE_HEADER_BYTES = 16

_BytesLike = Union[bytes, bytearray, memoryview]


@dataclass(frozen=True)
class AccessUnitInfo:
    """一个 H.264 访问单元的结构摘要。"""

    annex_b: bool
    """数据以 Annex B 起始码开头并至少含一个 NAL。"""
    nal_types: Tuple[int, ...]
    """按出现顺序排列的 NAL 类型。"""
    has_sps: bool
    has_pps: bool
    has_idr: bool
    has_b_slice: bool
    """首个 slice 的 ``slice_type`` 为 B。"""

    @property
    def keyframe(self) -> bool:
        """同时含 SPS、PPS 和 IDR，可从该包独立开始解码。"""
        return self.has_sps and self.has_pps and self.has_idr


def _iter_nals(data: _BytesLike):
    """产出 ``(NAL 起点, NAL 终点)``，不含起始码和 4 字节起始码的前导零。"""
    raw = bytes(data) if not isinstance(data, bytes) else data
    starts = []
    position = raw.find(b"\x00\x00\x01")
    while position != -1:
        starts.append(position + 3)
        position = raw.find(b"\x00\x00\x01", position + 3)
    for index, start in enumerate(starts):
        if index + 1 < len(starts):
            end = starts[index + 1] - 3
            while end > start and raw[end - 1] == 0:
                end -= 1  # 下一个 4 字节起始码的前导零。
        else:
            end = len(raw)
        if end > start:
            yield start, end


def _unescape(payload: bytes) -> bytes:
    """去除 NAL 内的防竞争字节 ``00 00 03``。"""
    return payload.replace(b"\x00\x00\x03", b"\x00\x00")


def _read_ue(bits: bytes, position: int) -> Optional[Tuple[int, int]]:
    """从比特位置读取一个无符号指数哥伦布码，返回值和新位置。"""
    total = len(bits) * 8
    zeros = 0
    while True:
        if position + zeros >= total:
            return None
        if (bits[(position + zeros) >> 3] >> (7 - ((position + zeros) & 7))) & 1:
            break
        zeros += 1
    end = position + zeros + 1 + zeros
    if end > total:
        return None
    value = 0
    for offset in range(position + zeros, end):
        value = (value << 1) | (
            (bits[offset >> 3] >> (7 - (offset & 7))) & 1
        )
    return value - 1, end


def _slice_is_b(nal_payload: bytes) -> bool:
    """解析 slice 头中的 ``slice_type``，B 类型返回真；无法解析视为非 B。"""
    header = _unescape(nal_payload[1 : 1 + _SLICE_HEADER_BYTES])
    first_mb = _read_ue(header, 0)
    if first_mb is None:
        return False
    slice_type = _read_ue(header, first_mb[1])
    if slice_type is None:
        return False
    return slice_type[0] % 5 == _SLICE_TYPE_B


def analyze_annex_b(data: _BytesLike) -> AccessUnitInfo:
    """扫描一个访问单元并汇总 NAL 类型与首个 slice 的类型。"""
    raw = bytes(data) if not isinstance(data, bytes) else data
    nal_types = []
    has_b = False
    seen_slice = False
    for start, end in _iter_nals(raw):
        nal_type = raw[start] & 0x1F
        nal_types.append(nal_type)
        if nal_type in (NAL_SLICE, NAL_IDR) and not seen_slice:
            seen_slice = True
            has_b = _slice_is_b(raw[start:end])
    return AccessUnitInfo(
        annex_b=bool(nal_types),
        nal_types=tuple(nal_types),
        has_sps=NAL_SPS in nal_types,
        has_pps=NAL_PPS in nal_types,
        has_idr=NAL_IDR in nal_types,
        has_b_slice=has_b,
    )
