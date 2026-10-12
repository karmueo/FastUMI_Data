#ifndef FASTUMI_USB_CAMERA__H264_ACCESS_UNIT_HPP_
#define FASTUMI_USB_CAMERA__H264_ACCESS_UNIT_HPP_

#include <cstddef>
#include <cstdint>

namespace fastumi_usb_camera
{

// Structure of one Annex B H.264 access unit, found by scanning NAL start
// codes. The camera protocol sends exactly one access unit per packet.
struct AccessUnitInfo
{
  bool has_sps = false;
  bool has_pps = false;
  bool has_idr = false;

  // A decoder can start from this packet only if it carries both parameter
  // sets and an IDR slice; AV_PKT_FLAG_KEY alone is not enough.
  bool qualified_keyframe() const {return has_sps && has_pps && has_idr;}
};

// Scan `data` for NAL units and report which of SPS (7), PPS (8) and IDR (5)
// are present. Data without Annex B start codes yields an empty result.
inline AccessUnitInfo analyze_access_unit(const uint8_t * data, size_t size)
{
  AccessUnitInfo info;
  // A start code is 00 00 01; the NAL header byte follows it.
  for (size_t index = 0; index + 3 < size; ++index) {
    if (data[index] != 0 || data[index + 1] != 0 || data[index + 2] != 1) {
      continue;
    }
    const uint8_t nal_type = data[index + 3] & 0x1F;
    info.has_sps = info.has_sps || nal_type == 7;
    info.has_pps = info.has_pps || nal_type == 8;
    info.has_idr = info.has_idr || nal_type == 5;
    index += 2;
  }
  return info;
}

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__H264_ACCESS_UNIT_HPP_
