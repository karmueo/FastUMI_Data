#include <gtest/gtest.h>

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "fastumi_usb_camera/decoder_policy.hpp"

namespace
{

using fastumi_usb_camera::DecoderPolicy;
using fastumi_usb_camera::PacketStatus;

// A decoder whose per-name initialization and per-call decode results are scripted.
class FakeDecoder : public fastumi_usb_camera::PacketDecoder
{
public:
  bool initialized() const override {return open_;}

  bool initialize(const std::string & name) override
  {
    attempts.push_back(name);
    if (unavailable.count(name) != 0) {
      return false;
    }
    open_ = true;
    current = name;
    return true;
  }

  bool decode() override
  {
    decoded_with.push_back(current);
    if (!failures.empty() && failures.front() == current) {
      failures.erase(failures.begin());
      return false;
    }
    return true;
  }

  void reset() override
  {
    open_ = false;
    ++resets;
  }

  // Decoder names that cannot be opened.
  std::set<std::string> unavailable;
  // Names whose next decode() fails, consumed in order.
  std::vector<std::string> failures;
  std::vector<std::string> attempts;
  std::vector<std::string> decoded_with;
  std::string current;
  int resets = 0;

private:
  bool open_ = false;
};

}  // namespace

TEST(DecoderPolicy, WaitsForQualifiedKeyframeBeforeOpeningAnything)
{
  FakeDecoder decoder;
  DecoderPolicy policy(decoder, true);
  EXPECT_EQ(policy.process(false), PacketStatus::kWaitingForKeyframe);
  EXPECT_TRUE(decoder.attempts.empty());
  EXPECT_EQ(policy.process(true), PacketStatus::kDecoded);
  EXPECT_EQ(policy.active_decoder(), "h264_cuvid");
  EXPECT_EQ(policy.process(false), PacketStatus::kDecoded);
  EXPECT_FALSE(policy.software_locked());
  EXPECT_EQ(policy.fallbacks(), 0);
}

TEST(DecoderPolicy, AutoOffersTheSameKeyframeToSoftwareWhenCudaCannotOpen)
{
  FakeDecoder decoder;
  decoder.unavailable = {"h264_cuvid"};
  DecoderPolicy policy(decoder, true);

  EXPECT_EQ(policy.process(true), PacketStatus::kDecoded);

  EXPECT_EQ(decoder.attempts, (std::vector<std::string>{"h264_cuvid", "h264"}));
  EXPECT_EQ(decoder.decoded_with, (std::vector<std::string>{"h264"}));
  EXPECT_EQ(policy.active_decoder(), "h264");
  EXPECT_TRUE(policy.software_locked());
  EXPECT_EQ(policy.fallbacks(), 1);
}

TEST(DecoderPolicy, AutoSwitchesToSoftwareAtTheNextKeyframeAfterRuntimeFailure)
{
  FakeDecoder decoder;
  decoder.failures = {"h264_cuvid"};
  DecoderPolicy policy(decoder, true);
  EXPECT_EQ(policy.process(true), PacketStatus::kDecodeFailed);
  EXPECT_EQ(policy.active_decoder(), "");
  EXPECT_TRUE(policy.software_locked());

  // Delta frames cannot start software decoding; the next keyframe can.
  EXPECT_EQ(policy.process(false), PacketStatus::kWaitingForKeyframe);
  EXPECT_EQ(policy.process(true), PacketStatus::kDecoded);
  EXPECT_EQ(policy.active_decoder(), "h264");
  EXPECT_EQ(policy.fallbacks(), 1);
}

TEST(DecoderPolicy, SoftwareIsKeptUntilRestartEvenAfterSoftwareFailure)
{
  FakeDecoder decoder;
  decoder.failures = {"h264_cuvid", "h264"};
  DecoderPolicy policy(decoder, true);
  EXPECT_EQ(policy.process(true), PacketStatus::kDecodeFailed);
  EXPECT_EQ(policy.process(true), PacketStatus::kDecodeFailed);  // software, fails once
  EXPECT_EQ(policy.process(true), PacketStatus::kDecoded);

  // CUDA is never retried after the switch.
  for (const auto & name : decoder.attempts) {
    EXPECT_TRUE(name == "h264_cuvid" || name == "h264");
  }
  EXPECT_EQ(decoder.attempts, (std::vector<std::string>{"h264_cuvid", "h264", "h264"}));
  EXPECT_EQ(policy.fallbacks(), 1);
}

TEST(DecoderPolicy, CudaModeDoesNotFallBack)
{
  FakeDecoder unavailable;
  unavailable.unavailable = {"h264_cuvid"};
  DecoderPolicy cannot_open(unavailable, false);
  EXPECT_THROW(cannot_open.process(true), std::runtime_error);
  EXPECT_EQ(unavailable.attempts, (std::vector<std::string>{"h264_cuvid"}));

  FakeDecoder failing;
  failing.failures = {"h264_cuvid"};
  DecoderPolicy runtime_failure(failing, false);
  EXPECT_EQ(runtime_failure.process(true), PacketStatus::kDecodeFailed);
  EXPECT_FALSE(runtime_failure.software_locked());
  EXPECT_EQ(runtime_failure.process(true), PacketStatus::kDecoded);
  EXPECT_EQ(runtime_failure.active_decoder(), "h264_cuvid");
}

TEST(DecoderPolicy, FailsLoudlyWhenNoDecoderCanOpen)
{
  FakeDecoder decoder;
  decoder.unavailable = {"h264_cuvid", "h264"};
  DecoderPolicy policy(decoder, true);
  EXPECT_THROW(policy.process(true), std::runtime_error);
}
