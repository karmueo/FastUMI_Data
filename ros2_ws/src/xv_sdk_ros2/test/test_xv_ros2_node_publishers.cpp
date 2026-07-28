/**
 * @file test_xv_ros2_node_publishers.cpp
 * @brief 验证 XV ROS 2 节点只创建当前支持的数据话题。
 */

#include "xv_ros2_node.h"

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {
/**
 * @brief 创建设备话题名称。
 * @param sn 设备序列号。
 * @param topic 设备内的相对话题名称。
 * @return 完整 ROS 话题名称。
 */
std::string deviceTopic(const std::string &sn, const std::string &topic) {
  return "/xv_sdk/SN" + sn + "/" + topic;
}

/**
 * @brief 收集指定设备序列号下的 ROS 话题名称。
 * @param node ROS 2 节点。
 * @param sn 设备序列号。
 * @return 指定设备命名空间下的话题集合。
 */
std::set<std::string>
collectDeviceTopics(const std::shared_ptr<xvision_ros2_node> &node,
                    const std::string &sn) {
  /** 设备话题前缀。 */
  const std::string prefix = "/xv_sdk/SN" + sn + "/";
  /** ROS graph 当前可见的话题和类型列表。 */
  const auto topic_names_and_types = node->get_topic_names_and_types();
  /** 指定设备下的话题集合。 */
  std::set<std::string> device_topics;

  for (const auto &topic_entry : topic_names_and_types) {
    if (topic_entry.first.rfind(prefix, 0) == 0) {
      device_topics.insert(topic_entry.first);
    }
  }

  return device_topics;
}

/**
 * @brief 创建带显式布尔参数覆盖的测试节点。
 * @param parameter_overrides 参数名和值列表。
 * @return 完成设备配置读取的测试节点。
 */
std::shared_ptr<xvision_ros2_node> createConfiguredNode(
    const std::vector<std::pair<std::string, bool>> &parameter_overrides = {}) {
  /** 待测试的 ROS 2 节点。 */
  auto node = std::make_shared<xvision_ros2_node>();
  for (const auto &parameter_override : parameter_overrides) {
    node->declare_parameter<bool>(parameter_override.first,
                                  parameter_override.second);
  }

  node->get_device_config_parameters();
  return node;
}
} // namespace

/**
 * @brief 为需要 rclcpp 上下文的节点测试提供初始化和清理。
 */
class XvRos2NodePublisherTest : public ::testing::Test {
protected:
  /** @brief 初始化 ROS 2 上下文。 */
  static void SetUpTestSuite() {
    if (!rclcpp::ok()) {
      /** 测试进程的命令行参数数量。 */
      int argc = 1;
      /** 测试进程的伪命令行参数列表。 */
      const char *argv[] = {"test_xv_ros2_node_publishers"};
      rclcpp::init(argc, argv);
    }
  }

  /** @brief 清理 ROS 2 上下文。 */
  static void TearDownTestSuite() {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

/**
 * @brief 未创建设备 publisher 时，保留的发布函数应安全返回。
 */
TEST_F(XvRos2NodePublisherTest, SupportedPublishersIgnoreUnknownSerialNumber) {
  /** 未初始化设备话题的测试节点。 */
  auto node = std::make_shared<xvision_ros2_node>();
  /** 不存在于 publisher 映射中的设备序列号。 */
  const std::string unknown_sn = "unknown";
  /** 空 IMU 消息。 */
  const rosImu imu;
  /** 空图像消息。 */
  const rosImage image;
  /** 空相机内参消息。 */
  const rosCamInfo camera_info;

  EXPECT_NO_THROW(node->publishImu(unknown_sn, imu));
  EXPECT_NO_THROW(node->publishTofCameraImage(unknown_sn, image, camera_info));
  EXPECT_NO_THROW(
      node->publishTofIrCameraImage(unknown_sn, image, camera_info));
  EXPECT_NO_THROW(node->publishRGBCameraImage(unknown_sn, image, camera_info));
  EXPECT_NO_THROW(node->publishRGBFisheyeUndistortedCameraImage(
      unknown_sn, image, camera_info));
}

/**
 * @brief 默认配置只创建设备 IMU 和普通 RGB 话题。
 */
TEST_F(XvRos2NodePublisherTest, DefaultDeviceTopicsOnlyExposeImuAndRgb) {
  /** 默认配置的测试节点。 */
  auto node = createConfiguredNode();
  /** 测试设备序列号。 */
  const std::string sn = "DEFAULT";
  /** 默认应发布的设备话题。 */
  const std::set<std::string> expected_topics{
      deviceTopic(sn, "imu"),
      deviceTopic(sn, "rgb/image"),
      deviceTopic(sn, "rgb/camera_info"),
  };

  node->initTopicAndServer(sn);

  EXPECT_EQ(expected_topics, collectDeviceTopics(node, sn));
}

/**
 * @brief 默认配置只保留三个设备输出开关。
 */
TEST_F(XvRos2NodePublisherTest, DefaultDeviceConfigUsesSupportedSwitches) {
  /** 默认配置的测试节点。 */
  auto node = createConfiguredNode();

  EXPECT_TRUE(node->getConfig("rgb_enable"));
  EXPECT_FALSE(node->getConfig("tof_enable"));
  EXPECT_FALSE(node->getConfig("rgb_fisheye_undistort_enable"));
  EXPECT_FALSE(node->getConfig("rgbd_enable"));
}

/**
 * @brief 缺少 frame 参数时返回调用方传入的默认 frame id。
 */
TEST_F(XvRos2NodePublisherTest, FrameIdFallsBackToDefaultWhenParameterMissing) {
  /** 未声明 frame 参数的测试节点。 */
  auto node = std::make_shared<xvision_ros2_node>();

  EXPECT_EQ("rgb_optical_frame", node->getFrameID("rgb_optical_frame"));
}

/**
 * @brief 开启 RGB 鱼眼校正时创建对应图像和内参话题。
 */
TEST_F(XvRos2NodePublisherTest, RgbFisheyeUndistortedTopicsAppearWhenEnabled) {
  /** 开启 RGB 鱼眼校正输出的测试节点。 */
  auto node = createConfiguredNode({{"rgb_fisheye_undistort_enable", true}});
  /** 测试设备序列号。 */
  const std::string sn = "RGB_FISHEYE";

  node->initTopicAndServer(sn);

  /** 开启校正输出后的设备话题。 */
  const auto device_topics = collectDeviceTopics(node, sn);
  EXPECT_EQ(1U, device_topics.count(
                    deviceTopic(sn, "rgb_fisheye_undistorted/image")));
  EXPECT_EQ(1U, device_topics.count(
                    deviceTopic(sn, "rgb_fisheye_undistorted/camera_info")));
}

/**
 * @brief 关闭普通 RGB 时只保留默认启用的 IMU 话题。
 */
TEST_F(XvRos2NodePublisherTest, RgbDeviceTopicsCanBeDisabled) {
  /** 关闭普通 RGB 的测试节点。 */
  auto node = createConfiguredNode({{"rgb_enable", false}});
  /** 测试设备序列号。 */
  const std::string sn = "NO_RGB";
  /** 关闭普通 RGB 后应发布的设备话题。 */
  const std::set<std::string> expected_topics{
      deviceTopic(sn, "imu"),
  };

  node->initTopicAndServer(sn);

  EXPECT_EQ(expected_topics, collectDeviceTopics(node, sn));
}

/**
 * @brief 开启 ToF 时创建深度、IR 及各自内参话题。
 */
TEST_F(XvRos2NodePublisherTest, TofTopicsAppearWhenEnabled) {
  /** 开启 ToF 的测试节点。 */
  auto node = createConfiguredNode({{"tof_enable", true}});
  /** 测试设备序列号。 */
  const std::string sn = "TOF";
  /** 默认话题加 ToF 话题后的完整集合。 */
  const std::set<std::string> expected_topics{
      deviceTopic(sn, "imu"),
      deviceTopic(sn, "rgb/image"),
      deviceTopic(sn, "rgb/camera_info"),
      deviceTopic(sn, "tof/depth/image_rect_raw"),
      deviceTopic(sn, "tof/depth/camera_info"),
      deviceTopic(sn, "tof/ir/image_raw"),
      deviceTopic(sn, "tof/ir/camera_info"),
  };

  node->initTopicAndServer(sn);

  EXPECT_EQ(expected_topics, collectDeviceTopics(node, sn));
}

/**
 * @brief 旧参数即使传入也不会创建已移除话题。
 */
TEST_F(XvRos2NodePublisherTest, LegacyParametersDoNotCreateTopics) {
  /** 声明旧参数的测试节点。 */
  auto node = createConfiguredNode({
      {"slam_pose_enable", true},
      {"fisheye_enable", true},
      {"rgb_rectification_enable", true},
      {"rgb_registered_enable", true},
      {"rgbd_enable", true},
      {"rgbd_raw_enable", true},
      {"factory_rgbd_enable", true},
      {"rgb_point_cloud_enable", true},
      {"clamp_enable", true},
      {"camera_sync_enable", true},
  });
  /** 测试设备序列号。 */
  const std::string sn = "LEGACY";
  /** 旧参数不改变默认设备话题集合。 */
  const std::set<std::string> expected_topics{
      deviceTopic(sn, "imu"),
      deviceTopic(sn, "rgb/image"),
      deviceTopic(sn, "rgb/camera_info"),
  };

  node->initTopicAndServer(sn);

  EXPECT_EQ(expected_topics, collectDeviceTopics(node, sn));
  EXPECT_FALSE(node->getConfig("rgbd_enable"));
  EXPECT_FALSE(node->getConfig("camera_sync_enable"));
}
