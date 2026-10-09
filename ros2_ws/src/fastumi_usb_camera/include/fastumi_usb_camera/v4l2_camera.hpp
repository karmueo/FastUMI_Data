/**
 * @file v4l2_camera.hpp
 * @brief 提供可独立验证的 V4L2 MJPEG 采集及系统调用边界。
 */
#ifndef FASTUMI_USB_CAMERA__V4L2_CAMERA_HPP_
#define FASTUMI_USB_CAMERA__V4L2_CAMERA_HPP_

#include <fcntl.h>
#include <memory>
#include <functional>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "fastumi_usb_camera/configuration.hpp"

namespace fastumi_usb_camera
{

/** @brief Linux 系统调用边界；测试可模拟驱动故障而不打开真实相机。 */
class V4l2Io
{
public:
  virtual ~V4l2Io() = default;
  virtual int open_device(const std::string & path)
  {
    return open(path.c_str(), O_RDWR | O_NONBLOCK);
  }
  virtual int control(int descriptor, unsigned long request, void * argument)
  {
    return ioctl(descriptor, request, argument);
  }
  virtual int wait(pollfd * descriptor, int timeout_ms)
  {
    return poll(descriptor, 1, timeout_ms);
  }
  virtual void * map_buffer(int descriptor, size_t length, off_t offset)
  {
    return mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, offset);
  }
  virtual int unmap_buffer(void * address, size_t length) {return munmap(address, length);}
  virtual int close_device(int descriptor) {return close(descriptor);}
};

/** @brief 采集主机收到的完整 MJPEG；错误恢复只重启采集，保持编码器和时间戳连续。 */
class V4l2Camera
{
public:
  using FrameCallback = void (*)(const uint8_t *, size_t, void *);
  using RecoveryCallback = std::function<void(const std::string &)>;

  /** @brief 严格协商设备模式并启动线程；帧回调须在返回前复制数据。 */
  V4l2Camera(
    const CameraConfiguration & config, FrameCallback callback, void * user_data,
    RecoveryCallback recovery_callback = {},
    std::shared_ptr<V4l2Io> io = std::make_shared<V4l2Io>());
  V4l2Camera(const V4l2Camera &) = delete;
  V4l2Camera & operator=(const V4l2Camera &) = delete;
  ~V4l2Camera();

  /** @brief 停止并等待线程，释放视频设备；不能从帧回调内部调用。 */
  void stop();
  bool failed() const;
  std::string error_message() const;
  uint64_t invalid_count() const;
  uint64_t recovery_count() const;
  const std::string & device_path() const;

private:
  struct Impl;
  std::unique_ptr<Impl> implementation_;
};

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__V4L2_CAMERA_HPP_
