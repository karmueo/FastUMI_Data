#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "fastumi_usb_camera/h264_access_unit.hpp"

namespace
{

using fastumi_usb_camera::analyze_access_unit;

// Build an Annex B access unit from NAL header bytes, each behind a start code.
std::vector<uint8_t> access_unit(const std::vector<uint8_t> & nal_headers, bool long_start_code)
{
  std::vector<uint8_t> data;
  for (const auto header : nal_headers) {
    if (long_start_code) {
      data.push_back(0);
    }
    data.insert(data.end(), {0, 0, 1, header, 0x11, 0x22});
  }
  return data;
}

}  // namespace

TEST(H264AccessUnit, KeyframeNeedsSpsPpsAndIdr)
{
  // NAL headers: SPS 0x67, PPS 0x68, IDR slice 0x65, non-IDR slice 0x41.
  const auto key = access_unit({0x67, 0x68, 0x65}, false);
  const auto info = analyze_access_unit(key.data(), key.size());
  EXPECT_TRUE(info.has_sps);
  EXPECT_TRUE(info.has_pps);
  EXPECT_TRUE(info.has_idr);
  EXPECT_TRUE(info.qualified_keyframe());
}

TEST(H264AccessUnit, FourByteStartCodesAndSeiAreHandled)
{
  const auto key = access_unit({0x67, 0x68, 0x06, 0x65}, true);
  EXPECT_TRUE(analyze_access_unit(key.data(), key.size()).qualified_keyframe());
}

TEST(H264AccessUnit, IdrWithoutParameterSetsIsNotQualified)
{
  const auto idr_only = access_unit({0x65}, false);
  const auto info = analyze_access_unit(idr_only.data(), idr_only.size());
  EXPECT_TRUE(info.has_idr);
  EXPECT_FALSE(info.qualified_keyframe());

  const auto missing_pps = access_unit({0x67, 0x65}, false);
  EXPECT_FALSE(analyze_access_unit(missing_pps.data(), missing_pps.size()).qualified_keyframe());
}

TEST(H264AccessUnit, DeltaFrameAndNonAnnexBDataAreNotKeyframes)
{
  const auto delta = access_unit({0x41}, false);
  EXPECT_FALSE(analyze_access_unit(delta.data(), delta.size()).qualified_keyframe());

  // AVCC-style length prefix instead of a start code.
  const std::vector<uint8_t> avcc = {0, 0, 0, 5, 0x65, 0x88, 0x84, 0x00, 0x10};
  const auto info = analyze_access_unit(avcc.data(), avcc.size());
  EXPECT_FALSE(info.has_idr);
  EXPECT_FALSE(analyze_access_unit(nullptr, 0).qualified_keyframe());
}

TEST(H264AccessUnit, TruncatedStartCodeAtTheEndIsIgnored)
{
  const std::vector<uint8_t> truncated = {0x67, 0, 0, 1};
  EXPECT_FALSE(analyze_access_unit(truncated.data(), truncated.size()).has_sps);
}
