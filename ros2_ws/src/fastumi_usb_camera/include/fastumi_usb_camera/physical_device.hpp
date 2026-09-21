/**
 * @file physical_device.hpp
 * @brief 校验 V4L2 by-path 物理端口，并从 sysfs 解析对应 USB 地址。
 */
#ifndef FASTUMI_USB_CAMERA__PHYSICAL_DEVICE_HPP_
#define FASTUMI_USB_CAMERA__PHYSICAL_DEVICE_HPP_

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>

namespace fastumi_usb_camera
{

/** @brief USB 设备在当前枚举周期中的总线号和设备地址。 */
struct UsbLocation
{
  uint8_t bus_number;  ///< USB 总线号。
  uint8_t device_address;  ///< USB 设备地址。
};

/**
 * @brief 判断字符串是否为主视频节点的稳定 udev by-path 路径。
 * @param[in] video_device 待检查路径。
 * @param[in] by_path_root udev V4L2 物理路径目录。
 * @return 路径格式符合要求时为 true。
 */
inline bool is_physical_video_device_path(
  const std::string & video_device,
  const std::filesystem::path & by_path_root = "/dev/v4l/by-path")
{
  /** @brief 保持符号链接本身路径，用于验证配置来源。 */
  const std::filesystem::path path(video_device);
  /** @brief udev 主视频节点文件名格式。 */
  static const std::regex pattern(R"([^/]+-video-index0)");
  return path.is_absolute() && path.parent_path() == by_path_root &&
         std::regex_match(path.filename().string(), pattern);
}

/**
 * @brief 读取仅含十进制整数的 sysfs 属性。
 * @param[in] path 属性文件路径。
 * @return 属性整数值。
 * @throws std::runtime_error 文件不可读或内容超出 uint8_t 范围。
 */
inline uint8_t read_usb_number(const std::filesystem::path & path)
{
  /** @brief sysfs 属性输入流。 */
  std::ifstream stream(path);
  /** @brief 使用有符号整数检查读取结果和范围。 */
  int value = -1;
  if (!(stream >> value) || value < 0 || value > 255) {
    throw std::runtime_error("cannot read USB number from " + path.string());
  }
  return static_cast<uint8_t>(value);
}

/**
 * @brief 从稳定 V4L2 物理路径解析当前 USB 总线号和设备地址。
 * @param[in] video_device `/dev/v4l/by-path/` 下以 `-video-index0` 结尾的路径。
 * @param[in] sysfs_root video4linux sysfs 类目录。
 * @param[in] device_root V4L2 字符设备目录。
 * @param[in] by_path_root udev V4L2 物理路径目录。
 * @return 当前 USB 地址。
 * @throws std::invalid_argument 路径格式不合法。
 * @throws std::runtime_error 链接、sysfs 或 USB 属性无法解析。
 */
inline UsbLocation usb_location_from_video_device(
  const std::string & video_device,
  const std::filesystem::path & sysfs_root = "/sys/class/video4linux",
  const std::filesystem::path & device_root = "/dev",
  const std::filesystem::path & by_path_root = "/dev/v4l/by-path")
{
  if (!is_physical_video_device_path(video_device, by_path_root)) {
    throw std::invalid_argument(
            "video_device must use /dev/v4l/by-path/*-video-index0: " + video_device);
  }
  std::error_code error;  ///< 保留 filesystem 失败原因，避免异常丢失目标路径。
  /** @brief by-path 链接当前指向的内核视频节点。 */
  const auto resolved = std::filesystem::canonical(video_device, error);
  if (error || resolved.parent_path() != std::filesystem::canonical(device_root, error) ||
    !std::regex_match(resolved.filename().string(), std::regex(R"(video\d+)")))
  {
    throw std::runtime_error("video_device does not resolve to /dev/videoN: " + video_device);
  }
  /** @brief 视频节点对应的 USB 接口 sysfs 路径。 */
  auto current = std::filesystem::canonical(
    sysfs_root / resolved.filename() / "device", error);
  if (error) {
    throw std::runtime_error("cannot resolve sysfs device for " + video_device);
  }
  while (!current.empty()) {
    if (std::filesystem::is_regular_file(current / "busnum") &&
      std::filesystem::is_regular_file(current / "devnum"))
    {
      return UsbLocation{
        read_usb_number(current / "busnum"), read_usb_number(current / "devnum")};
    }
    if (current == current.root_path()) {
      break;
    }
    current = current.parent_path();
  }
  throw std::runtime_error("video_device is not associated with a USB device: " + video_device);
}

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__PHYSICAL_DEVICE_HPP_
