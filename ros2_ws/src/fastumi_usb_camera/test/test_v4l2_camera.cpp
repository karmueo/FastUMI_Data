/**
 * @file test_v4l2_camera.cpp
 * @brief 验证 V4L2 显式路径及首个 USB 主视频节点发现规则。
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "fastumi_usb_camera/configuration.hpp"
#include "fastumi_usb_camera/v4l2_camera.hpp"

namespace
{

/** @brief 为每个用例创建并回收独立的伪 sysfs 目录。 */
class V4l2DiscoveryTest : public ::testing::Test
{
protected:
  /** @brief 创建当前进程唯一的临时目录。 */
  void SetUp() override
  {
    root_ = std::filesystem::temp_directory_path() /
      ("fastumi-v4l2-test-" + std::to_string(getpid()) + "-" +
      std::to_string(counter_++));
    std::filesystem::create_directories(video_class_root_ = root_ / "sys/class/video4linux");
    std::filesystem::create_directories(device_root_ = root_ / "dev");
  }

  /** @brief 删除当前用例创建的伪设备树。 */
  void TearDown() override
  {
    std::error_code error;  ///< 临时目录清理错误，测试结束时无需抛出。
    std::filesystem::remove_all(root_, error);
  }

  /**
   * @brief 创建一个可选关联 USB 设备的伪视频节点。
   * @param[in] video_name 视频节点名称。
   * @param[in] usb_name 伪 sysfs 设备名称。
   * @param[in] index V4L2 视频节点索引。
   * @param[in] is_usb 是否写入 USB 设备标识属性。
   */
  void add_camera(const std::string & video_name, const std::string & usb_name,
    const std::string & index = "0", bool is_usb = true)
  {
    const std::filesystem::path usb_device =
      root_ / "sys/devices" / usb_name;  ///< 伪 USB 设备目录。
    const std::filesystem::path interface_path =
      usb_device / (usb_name + ":1.0");  ///< 伪 UVC 接口目录。
    std::filesystem::create_directories(interface_path);
    if (is_usb) {
      std::ofstream(usb_device / "idVendor") << "1bcf\n";
      std::ofstream(usb_device / "idProduct") << "28c4\n";
    }

    const std::filesystem::path class_entry =
      video_class_root_ / video_name;  ///< 伪 video4linux 类节点。
    std::filesystem::create_directories(class_entry);
    std::ofstream(class_entry / "index") << index << '\n';
    std::filesystem::create_directory_symlink(interface_path, class_entry / "device");
  }

  static inline uint64_t counter_{0};  ///< 同一测试进程内的临时目录序号。
  std::filesystem::path root_;  ///< 当前用例的临时根目录。
  std::filesystem::path video_class_root_;  ///< 伪 video4linux sysfs 目录。
  std::filesystem::path device_root_;  ///< 伪 /dev 目录。
};

TEST_F(V4l2DiscoveryTest, ExplicitDeviceMustExist)
{
  fastumi_usb_camera::CameraConfiguration config;  ///< 待解析的显式设备配置。
  const std::filesystem::path device = device_root_ / "video7";  ///< 存在的伪视频节点。
  std::ofstream(device) << "";
  config.video_device = device.string();
  EXPECT_EQ(fastumi_usb_camera::discover_video_device(config), device.string());

  config.video_device = (device_root_ / "missing").string();
  EXPECT_THROW(fastumi_usb_camera::discover_video_device(config), std::runtime_error);
}

TEST_F(V4l2DiscoveryTest, SelectsLowestNumberedUsbPrimaryNode)
{
  add_camera("video10", "1-4");
  add_camera("video2", "1-3");
  add_camera("video1", "1-2", "1");
  add_camera("video0", "platform-camera", "0", false);
  fastumi_usb_camera::CameraConfiguration config;  ///< 使用自动选择的默认配置。
  EXPECT_EQ(
    fastumi_usb_camera::discover_video_device(config, video_class_root_, device_root_),
    (device_root_ / "video2").string());
}

TEST_F(V4l2DiscoveryTest, FailsWhenNoUsbPrimaryNodeExists)
{
  add_camera("video0", "platform-camera", "0", false);
  add_camera("video1", "1-2", "1");
  fastumi_usb_camera::CameraConfiguration config;  ///< 使用自动选择的默认配置。
  EXPECT_THROW(
    fastumi_usb_camera::discover_video_device(config, video_class_root_, device_root_),
    std::runtime_error);
}

}  // namespace
