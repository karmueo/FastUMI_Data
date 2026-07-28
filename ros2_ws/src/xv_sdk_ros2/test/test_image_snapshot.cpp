/**
 * @file test_image_snapshot.cpp
 * @brief 验证一次性截图的图像转换、文件策略和节点生命周期。
 */

#include "image_snapshot_node.h"
#include "image_snapshot_utils.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace {

/**
 * @brief 创建指定编码和原始数据的 ROS 图像。
 * @param width 图像宽度。
 * @param height 图像高度。
 * @param encoding ROS 图像编码。
 * @param step 每行字节数。
 * @param data 图像原始字节。
 * @return 构造完成的 ROS 图像。
 */
sensor_msgs::msg::Image makeImage(std::uint32_t width, std::uint32_t height,
                                  const std::string &encoding,
                                  std::uint32_t step,
                                  const std::vector<std::uint8_t> &data) {
  /** 构造得到的 ROS 图像。 */
  sensor_msgs::msg::Image image;
  image.width = width;
  image.height = height;
  image.encoding = encoding;
  image.is_bigendian = false;
  image.step = step;
  image.data = data;
  return image;
}

/** @brief 为截图测试提供 ROS 上下文和独立临时目录。 */
class ImageSnapshotTest : public ::testing::Test {
protected:
  /** @brief 初始化 ROS 2 测试上下文。 */
  static void SetUpTestSuite() {
    if (!rclcpp::ok()) {
      /** 测试进程命令行参数数量。 */
      int argc = 1;
      /** 测试进程伪命令行参数。 */
      const char *argv[] = {"test_image_snapshot"};
      rclcpp::init(argc, argv);
    }
  }

  /** @brief 清理 ROS 2 测试上下文。 */
  static void TearDownTestSuite() {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }

  /** @brief 为每个测试创建唯一临时目录名称。 */
  void SetUp() override {
    /** 用于隔离测试目录的单调时钟计数。 */
    const auto unique_value =
        std::chrono::steady_clock::now().time_since_epoch().count();
    test_directory_ =
        std::filesystem::temp_directory_path() /
        ("xv_image_snapshot_test_" + std::to_string(unique_value));
  }

  /** @brief 删除测试产生的临时目录。 */
  void TearDown() override {
    /** 清理临时目录时忽略的文件系统错误。 */
    std::error_code cleanup_error;
    std::filesystem::remove_all(test_directory_, cleanup_error);
  }

  /** 当前测试的临时输出目录。 */
  std::filesystem::path test_directory_;
};

/** @brief 验证 RGB8 保存后具有正确的 OpenCV BGR 通道顺序。 */
TEST_F(ImageSnapshotTest, SavesRgb8AsCorrectPngColor) {
  /** 单个红色 RGB 像素。 */
  const sensor_msgs::msg::Image image =
      makeImage(1, 1, sensor_msgs::image_encodings::RGB8, 3, {255, 0, 0});
  /** 固定输出文件名的保存选项。 */
  const xv_ros2::image_snapshot::SaveOptions options{test_directory_, "rgb.png",
                                                     false};
  /** 图像保存结果。 */
  const auto result = xv_ros2::image_snapshot::saveImage(image, options);

  ASSERT_TRUE(result.success) << result.error;
  /** 从磁盘重新读取的 BGR 图像。 */
  const cv::Mat saved =
      cv::imread(result.output_path.string(), cv::IMREAD_UNCHANGED);
  ASSERT_EQ(saved.type(), CV_8UC3);
  EXPECT_EQ(saved.at<cv::Vec3b>(0, 0), cv::Vec3b(0, 0, 255));
}

/** @brief 验证 mono16 PNG 保存过程保留原始像素值。 */
TEST_F(ImageSnapshotTest, PreservesMono16Values) {
  /** 两个小端序 16 位灰度像素。 */
  const sensor_msgs::msg::Image image = makeImage(
      2, 1, sensor_msgs::image_encodings::MONO16, 4, {0x34, 0x12, 0xcd, 0xab});
  /** mono16 图像保存选项。 */
  const xv_ros2::image_snapshot::SaveOptions options{test_directory_,
                                                     "mono16.png", false};
  /** 图像保存结果。 */
  const auto result = xv_ros2::image_snapshot::saveImage(image, options);

  ASSERT_TRUE(result.success) << result.error;
  /** 从磁盘重新读取的 16 位灰度图。 */
  const cv::Mat saved =
      cv::imread(result.output_path.string(), cv::IMREAD_UNCHANGED);
  ASSERT_EQ(saved.type(), CV_16UC1);
  EXPECT_EQ(saved.at<std::uint16_t>(0, 0), 0x1234);
  EXPECT_EQ(saved.at<std::uint16_t>(0, 1), 0xabcd);
}

/** @brief 验证 32FC1 自动使用 TIFF 并保留浮点深度。 */
TEST_F(ImageSnapshotTest, SavesFloatDepthAsTiff) {
  /** 两个浮点深度值。 */
  const float depth_values[2] = {1.25F, 3.5F};
  /** 浮点深度的原始字节。 */
  std::vector<std::uint8_t> depth_data(sizeof(depth_values));
  std::memcpy(depth_data.data(), depth_values, sizeof(depth_values));
  /** 浮点深度 ROS 图像。 */
  const sensor_msgs::msg::Image image =
      makeImage(2, 1, sensor_msgs::image_encodings::TYPE_32FC1,
                static_cast<std::uint32_t>(sizeof(depth_values)), depth_data);
  /** 使用自动文件名的保存选项。 */
  const xv_ros2::image_snapshot::SaveOptions options{test_directory_, "",
                                                     false};
  /** 图像保存结果。 */
  const auto result = xv_ros2::image_snapshot::saveImage(image, options);

  ASSERT_TRUE(result.success) << result.error;
  EXPECT_EQ(result.output_path.extension(), ".tiff");
  /** 从磁盘重新读取的浮点深度图。 */
  const cv::Mat saved =
      cv::imread(result.output_path.string(), cv::IMREAD_UNCHANGED);
  ASSERT_EQ(saved.type(), CV_32FC1);
  EXPECT_FLOAT_EQ(saved.at<float>(0, 0), depth_values[0]);
  EXPECT_FLOAT_EQ(saved.at<float>(0, 1), depth_values[1]);
}

/** @brief 验证目录创建、禁止覆盖和输入校验。 */
TEST_F(ImageSnapshotTest, EnforcesFilePolicies) {
  /** 单像素灰度图像。 */
  const sensor_msgs::msg::Image image =
      makeImage(1, 1, sensor_msgs::image_encodings::MONO8, 1, {42});
  /** 位于尚未创建子目录中的保存选项。 */
  const xv_ros2::image_snapshot::SaveOptions options{test_directory_ / "nested",
                                                     "frame.png", false};
  /** 第一次保存结果。 */
  const auto first_result = xv_ros2::image_snapshot::saveImage(image, options);
  /** 第二次保存结果。 */
  const auto second_result = xv_ros2::image_snapshot::saveImage(image, options);
  ASSERT_TRUE(first_result.success) << first_result.error;
  EXPECT_FALSE(second_result.success);
  EXPECT_NE(second_result.error.find("overwrite=false"), std::string::npos);

  /** 允许覆盖同一目标文件的保存选项。 */
  const xv_ros2::image_snapshot::SaveOptions overwrite_options{
      options.output_dir, options.filename, true};
  /** 允许覆盖时的保存结果。 */
  const auto overwrite_result =
      xv_ros2::image_snapshot::saveImage(image, overwrite_options);
  EXPECT_TRUE(overwrite_result.success) << overwrite_result.error;

  /** 未指定扩展名的保存选项。 */
  const xv_ros2::image_snapshot::SaveOptions extensionless_options{
      test_directory_, "extensionless", false};
  /** 自动补充 PNG 扩展名的保存结果。 */
  const auto extensionless_result =
      xv_ros2::image_snapshot::saveImage(image, extensionless_options);
  ASSERT_TRUE(extensionless_result.success) << extensionless_result.error;
  EXPECT_EQ(extensionless_result.output_path.extension(), ".png");

  /** 将已有文件误用作输出目录的保存选项。 */
  const xv_ros2::image_snapshot::SaveOptions file_as_directory_options{
      first_result.output_path, "child.png", false};
  /** 非法输出目录的保存结果。 */
  const auto invalid_directory_result =
      xv_ros2::image_snapshot::saveImage(image, file_as_directory_options);
  EXPECT_FALSE(invalid_directory_result.success);

  /** 使用不支持编码的图像。 */
  const sensor_msgs::msg::Image unsupported =
      makeImage(1, 1, "yuv422", 2, {0, 0});
  /** 不支持编码的保存结果。 */
  const auto unsupported_result =
      xv_ros2::image_snapshot::saveImage(unsupported, options);
  EXPECT_FALSE(unsupported_result.success);

  /** 尝试越出输出目录的保存选项。 */
  const xv_ros2::image_snapshot::SaveOptions unsafe_options{
      test_directory_, "../frame.png", false};
  /** 非法文件名的保存结果。 */
  const auto unsafe_result =
      xv_ros2::image_snapshot::saveImage(image, unsafe_options);
  EXPECT_FALSE(unsafe_result.success);
}

/** @brief 验证并发保存时禁止覆盖策略只允许一个写入者成功提交。 */
TEST_F(ImageSnapshotTest, AtomicallyRejectsConcurrentNonOverwriteSaves) {
  /** 并发写入者数量。 */
  constexpr std::size_t writer_count = 8;
  /** 所有写入者共用的保存选项。 */
  const xv_ros2::image_snapshot::SaveOptions options{
      test_directory_, "concurrent.png", false};
  /** 收集每个并发写入者的保存结果。 */
  std::vector<xv_ros2::image_snapshot::SaveResult> results(writer_count);
  /** 等待同时启动的写入者数量。 */
  std::atomic<std::size_t> ready_count{0};
  /** 控制全部写入者同时开始保存。 */
  std::atomic<bool> start{false};
  /** 承载并发保存任务的线程集合。 */
  std::vector<std::thread> writers;
  writers.reserve(writer_count);

  for (std::size_t writer_index = 0; writer_index < writer_count;
       ++writer_index) {
    writers.emplace_back([&, writer_index]() {
      /** 使用不同灰度值以检测最终文件是否完整来自单个写入者。 */
      const std::uint8_t pixel_value =
          static_cast<std::uint8_t>(writer_index + 1);
      /** 当前写入者保存的单像素灰度图像。 */
      const sensor_msgs::msg::Image image = makeImage(
          1, 1, sensor_msgs::image_encodings::MONO8, 1, {pixel_value});
      ready_count.fetch_add(1);
      while (!start.load()) {
        std::this_thread::yield();
      }
      results[writer_index] =
          xv_ros2::image_snapshot::saveImage(image, options);
    });
  }

  while (ready_count.load() != writer_count) {
    std::this_thread::yield();
  }
  start.store(true);
  for (std::thread &writer : writers) {
    writer.join();
  }

  /** 成功提交最终文件的写入者数量。 */
  std::size_t success_count = 0;
  for (const auto &result : results) {
    if (result.success) {
      ++success_count;
    } else {
      EXPECT_NE(result.error.find("overwrite=false"), std::string::npos);
    }
  }
  EXPECT_EQ(success_count, 1U);

  /** 并发保存后从最终路径读取的图像。 */
  const cv::Mat saved = cv::imread(
      (test_directory_ / options.filename).string(), cv::IMREAD_UNCHANGED);
  ASSERT_EQ(saved.type(), CV_8UC1);
  EXPECT_GE(saved.at<std::uint8_t>(0, 0), 1U);
  EXPECT_LE(saved.at<std::uint8_t>(0, 0), writer_count);

  /** 保存结束后目录中遗留的临时文件数量。 */
  std::size_t temporary_file_count = 0;
  for (const auto &entry : std::filesystem::directory_iterator(test_directory_)) {
    if (entry.path().filename().string().find(".tmp.") != std::string::npos) {
      ++temporary_file_count;
    }
  }
  EXPECT_EQ(temporary_file_count, 0U);
}

/** @brief 验证节点参数错误会立即进入失败状态。 */
TEST_F(ImageSnapshotTest, RejectsEmptyTopic) {
  /** 使用默认空话题参数的截图节点。 */
  auto node = std::make_shared<xv_ros2::image_snapshot::ImageSnapshotNode>();
  EXPECT_TRUE(node->finished());
  EXPECT_EQ(node->exitCode(), 2);
}

/** @brief 验证节点收到首帧后保存并结束。 */
TEST_F(ImageSnapshotTest, NodeSavesFirstFrameAndFinishes) {
  /** 测试图像话题。 */
  const std::string topic = "/test/image_snapshot/first_frame";
  /** 截图节点参数覆盖。 */
  const rclcpp::NodeOptions options =
      rclcpp::NodeOptions().parameter_overrides({
          rclcpp::Parameter("image_topic", topic),
          rclcpp::Parameter("output_dir", test_directory_.string()),
          rclcpp::Parameter("filename", "node.png"),
          rclcpp::Parameter("timeout_sec", 2.0),
      });
  /** 待测试的截图节点。 */
  auto snapshot_node =
      std::make_shared<xv_ros2::image_snapshot::ImageSnapshotNode>(options);
  /** 发布测试图像的 ROS 节点。 */
  auto publisher_node =
      std::make_shared<rclcpp::Node>("image_snapshot_test_publisher");
  /** 测试图像发布器。 */
  auto publisher = publisher_node->create_publisher<sensor_msgs::msg::Image>(
      topic, rclcpp::SensorDataQoS().keep_last(1));
  /** 驱动发布和订阅回调的执行器。 */
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(snapshot_node);
  executor.add_node(publisher_node);
  /** 待发布的灰度测试帧。 */
  const sensor_msgs::msg::Image image =
      makeImage(1, 1, sensor_msgs::image_encodings::MONO8, 1, {73});
  /** 测试等待截止时间。 */
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);

  while (!snapshot_node->finished() &&
         std::chrono::steady_clock::now() < deadline) {
    publisher->publish(image);
    executor.spin_once(std::chrono::milliseconds(20));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_TRUE(snapshot_node->finished());
  ASSERT_EQ(snapshot_node->exitCode(), 0);
  EXPECT_EQ(snapshot_node->outputPath(), test_directory_ / "node.png");
  EXPECT_TRUE(std::filesystem::exists(snapshot_node->outputPath()));
  executor.remove_node(publisher_node);
  executor.remove_node(snapshot_node);
}

/** @brief 验证等待首帧超时后节点返回失败状态。 */
TEST_F(ImageSnapshotTest, NodeTimesOutWithoutImage) {
  /** 超时测试节点参数覆盖。 */
  const rclcpp::NodeOptions options =
      rclcpp::NodeOptions().parameter_overrides({
          rclcpp::Parameter("image_topic", "/test/image_snapshot/timeout"),
          rclcpp::Parameter("output_dir", test_directory_.string()),
          rclcpp::Parameter("timeout_sec", 0.02),
      });
  /** 待测试的截图节点。 */
  auto node =
      std::make_shared<xv_ros2::image_snapshot::ImageSnapshotNode>(options);
  /** 驱动超时定时器的执行器。 */
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  /** 超时测试截止时间。 */
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (!node->finished() && std::chrono::steady_clock::now() < deadline) {
    executor.spin_once(std::chrono::milliseconds(20));
  }

  EXPECT_TRUE(node->finished());
  EXPECT_EQ(node->exitCode(), 4);
  executor.remove_node(node);
}

} // namespace
