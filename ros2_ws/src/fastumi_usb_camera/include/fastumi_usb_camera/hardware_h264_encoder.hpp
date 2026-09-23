/**
 * @file hardware_h264_encoder.hpp
 * @brief Jetson GStreamer MJPEG hardware decode and H.264 encode pipeline.
 */

#ifndef FASTUMI_USB_CAMERA__HARDWARE_H264_ENCODER_HPP_
#define FASTUMI_USB_CAMERA__HARDWARE_H264_ENCODER_HPP_

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace fastumi_usb_camera
{

/** @brief One H.264 access unit produced by the hardware pipeline. */
struct HardwareH264Packet
{
  uint64_t pts{0};
  bool keyframe{false};
  int64_t latency_nanoseconds{0};
  std::vector<uint8_t> data;
};

/** @brief Fixed hardware encoder settings shared with the software transport. */
struct HardwareH264Configuration
{
  int width{0};
  int height{0};
  int fps{0};
  int bitrate{4'000'000};
  int gop_size{10};
};

/**
 * @brief Owns a bounded NVIDIA GStreamer MJPEG-to-H.264 pipeline.
 *
 * Input and output timestamps use nanoseconds. The callback runs on a
 * GStreamer streaming thread and must not block.
 */
class HardwareH264Encoder
{
public:
  using PacketCallback = std::function<void(HardwareH264Packet &&)>;
  using SteadyClock = std::chrono::steady_clock;

  HardwareH264Encoder(
    const HardwareH264Configuration & configuration,
    PacketCallback callback);
  ~HardwareH264Encoder();

  HardwareH264Encoder(const HardwareH264Encoder &) = delete;
  HardwareH264Encoder & operator=(const HardwareH264Encoder &) = delete;

  /** @brief Push one complete JPEG frame, blocking only on the bounded input queue. */
  bool push(
    uint64_t pts, const std::vector<uint8_t> & jpeg,
    SteadyClock::time_point queued_at);

  /** @brief Stop the pipeline and unblock any input push before joining its caller. */
  void stop() noexcept;

  /** @brief Poll the pipeline bus and return true after a fatal error. */
  bool poll_error();

  /** @brief Return the most recent fatal pipeline error. */
  std::string error_message() const;

  /** @brief Check whether all required NVIDIA GStreamer factories exist. */
  static bool is_available(std::string * missing_factory = nullptr);

private:
  class Implementation;
  std::unique_ptr<Implementation> implementation_;
};

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__HARDWARE_H264_ENCODER_HPP_
