#ifndef FASTUMI_USB_CAMERA__DECODER_POLICY_HPP_
#define FASTUMI_USB_CAMERA__DECODER_POLICY_HPP_

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fastumi_usb_camera
{

// H.264 decoder names understood by ffmpeg_encoder_decoder.
inline constexpr const char * kCudaDecoder = "h264_cuvid";
inline constexpr const char * kSoftwareDecoder = "h264";

// The decoder operations the policy needs; the receiver adapts
// ffmpeg_encoder_decoder::Decoder to it and tests use a fake.
class PacketDecoder
{
public:
  virtual ~PacketDecoder() = default;
  // Whether a decoder context is currently open.
  virtual bool initialized() const = 0;
  // Open the named decoder; false when it cannot be created.
  virtual bool initialize(const std::string & decoder_name) = 0;
  // Decode the current packet; false on a decode failure.
  virtual bool decode() = 0;
  // Drop the context and its reference frames.
  virtual void reset() = 0;
};

// What happened to one packet.
enum class PacketStatus
{
  // No decoder is open and the packet cannot start one; it was skipped.
  kWaitingForKeyframe,
  kDecoded,
  // The decoder failed on this packet and was reset; wait for a keyframe.
  kDecodeFailed,
};

// Chooses between the CUDA and software H.264 decoders.
//
// * `cuda`: only h264_cuvid; an initialization failure is fatal.
// * `auto`: h264_cuvid first. If it cannot be opened, the same keyframe is
//   offered to the software decoder. If it fails while running, the decoder
//   is reset and the software decoder starts at the next qualified keyframe.
//   Once software is chosen it is kept until restart.
//
// Both modes start only at a qualified keyframe (SPS, PPS and IDR).
class DecoderPolicy
{
public:
  // `hardware_decoder` is the preferred decoder name (h264_cuvid by default).
  DecoderPolicy(
    PacketDecoder & decoder, bool allow_software_fallback,
    std::string hardware_decoder = kCudaDecoder)
  : decoder_(decoder), allow_software_fallback_(allow_software_fallback),
    hardware_decoder_(std::move(hardware_decoder)) {}

  // Process one packet. `qualified_keyframe` says whether it can start
  // decoding. Throws std::runtime_error if no decoder can be opened.
  PacketStatus process(bool qualified_keyframe)
  {
    if (!decoder_.initialized()) {
      if (!qualified_keyframe) {
        return PacketStatus::kWaitingForKeyframe;
      }
      open_decoder();
    }
    if (decoder_.decode()) {
      return PacketStatus::kDecoded;
    }
    decoder_.reset();
    // A running GPU decoder that fails is replaced by software for good.
    if (allow_software_fallback_ && active_ == hardware_decoder_) {
      software_locked_ = true;
      ++fallbacks_;
    }
    active_.clear();
    return PacketStatus::kDecodeFailed;
  }

  // Name of the decoder currently open, empty when none is.
  const std::string & active_decoder() const {return active_;}
  // True once software decoding was chosen; it lasts until restart.
  bool software_locked() const {return software_locked_;}
  // Number of switches from CUDA to software (init or runtime failures).
  int fallbacks() const {return fallbacks_;}

private:
  void open_decoder()
  {
    std::vector<std::string> candidates;
    if (software_locked_) {
      candidates = {kSoftwareDecoder};
    } else if (allow_software_fallback_) {
      candidates = {hardware_decoder_, kSoftwareDecoder};
    } else {
      candidates = {hardware_decoder_};
    }
    for (const auto & name : candidates) {
      if (decoder_.initialize(name)) {
        active_ = name;
        if (name == kSoftwareDecoder && !software_locked_) {
          software_locked_ = true;
          ++fallbacks_;
        }
        return;
      }
    }
    throw std::runtime_error(
            allow_software_fallback_ ?
            "Failed to initialize " + hardware_decoder_ + " and the software h264 decoder" :
            "Failed to initialize " + hardware_decoder_ + " hardware decoder");
  }

  PacketDecoder & decoder_;
  bool allow_software_fallback_;
  std::string hardware_decoder_;
  bool software_locked_ = false;
  int fallbacks_ = 0;
  std::string active_;
};

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__DECODER_POLICY_HPP_
