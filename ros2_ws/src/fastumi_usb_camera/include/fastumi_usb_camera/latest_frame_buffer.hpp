// Copyright 2026 karmueo

#ifndef FASTUMI_USB_CAMERA__LATEST_FRAME_BUFFER_HPP_
#define FASTUMI_USB_CAMERA__LATEST_FRAME_BUFFER_HPP_

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

namespace fastumi_usb_camera
{

template<typename FrameT>
class LatestFrameBuffer
{
public:
  bool push(FrameT frame)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return false;
    }
    if (frame_) {
      ++overwritten_count_;
    }
    frame_ = std::move(frame);
    condition_.notify_one();
    return true;
  }

  std::optional<FrameT> wait_and_take()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this]() {return closed_ || frame_.has_value();});
    if (!frame_) {
      return std::nullopt;
    }
    auto result = std::move(frame_);
    frame_.reset();
    return result;
  }

  void close()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    frame_.reset();
    condition_.notify_all();
  }

  uint64_t overwritten_count() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return overwritten_count_;
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<FrameT> frame_;
  uint64_t overwritten_count_{0};
  bool closed_{false};
};

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__LATEST_FRAME_BUFFER_HPP_
