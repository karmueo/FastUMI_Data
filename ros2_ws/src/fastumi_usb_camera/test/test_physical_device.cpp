/**
 * @file test_physical_device.cpp
 * @brief 验证 C++ 采集节点使用的 by-path 与 sysfs USB 地址解析。
 */
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "fastumi_usb_camera/physical_device.hpp"

using fastumi_usb_camera::is_physical_video_device_path;
using fastumi_usb_camera::usb_location_from_video_device;

TEST(PhysicalDevice, ValidatesStableMainVideoPath)
{
  /** @brief 合法的稳定主视频路径。 */
  const std::string stable =
    "/dev/v4l/by-path/pci-test-usb-0:2.4:1.0-video-index0";
  EXPECT_TRUE(is_physical_video_device_path(stable));
  EXPECT_FALSE(is_physical_video_device_path("/dev/video0"));
  EXPECT_FALSE(is_physical_video_device_path(
      "/dev/v4l/by-path/pci-test-usb-0:2.4:1.0-video-index1"));
}

TEST(PhysicalDevice, ResolvesUsbBusAndAddressFromSysfs)
{
  /** @brief 本用例独占的临时根目录。 */
  const auto root = std::filesystem::temp_directory_path() /
    ("fastumi-physical-device-" + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count()));
  /** @brief 模拟字符设备根目录。 */
  const auto device_root = root / "dev";
  /** @brief 模拟 udev by-path 目录。 */
  const auto by_path_root = device_root / "v4l/by-path";
  /** @brief 模拟 video4linux sysfs 类目录。 */
  const auto sysfs_root = root / "sys/class/video4linux";
  /** @brief 模拟 USB 设备目录。 */
  const auto usb_device = root / "sys/devices/1-2.4";
  /** @brief 模拟 USB 视频接口目录。 */
  const auto usb_interface = usb_device / "1-2.4:1.0";
  /** @brief 动态内核视频节点。 */
  const auto video = device_root / "video7";
  /** @brief 稳定物理端口链接。 */
  const auto stable = by_path_root / "pci-test-usb-0:2.4:1.0-video-index0";
  std::filesystem::create_directories(by_path_root);
  std::filesystem::create_directories(sysfs_root / "video7");
  std::filesystem::create_directories(usb_interface);
  std::ofstream(video).put('\n');
  std::ofstream(usb_device / "busnum") << "1\n";
  std::ofstream(usb_device / "devnum") << "62\n";
  std::filesystem::create_symlink(video, stable);
  std::filesystem::create_symlink(usb_interface, sysfs_root / "video7/device");

  /** @brief 解析出的 USB 地址。 */
  const auto location = usb_location_from_video_device(
    stable.string(), sysfs_root, device_root, by_path_root);
  EXPECT_EQ(location.bus_number, 1);
  EXPECT_EQ(location.device_address, 62);
  EXPECT_THROW(
    usb_location_from_video_device(video.string(), sysfs_root, device_root, by_path_root),
    std::invalid_argument);
  std::filesystem::remove_all(root);
}
