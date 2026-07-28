/**
 * @file test_rgb_registered_utils.cpp
 * @brief 验证 RGB 视角配准深度图的基础工具函数。
 */

#include "rgb_registered_utils.h"

#include "rgb_fisheye_undistort_utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

/**
 * @brief 构造 16 位 ToF 深度图并验证毫米到米的转换。
 */
TEST(RgbRegisteredUtils, ConvertsDepth16MillimetersToMeters) {
  /** 深度原始值，单位为毫米。 */
  auto raw_depth =
    std::shared_ptr<std::uint8_t>(new std::uint8_t[sizeof(std::uint16_t) * 2],
                                    std::default_delete<std::uint8_t[]>());
  /** 16 位深度视图，用于写入测试样本。 */
  auto *depth_values = reinterpret_cast<std::uint16_t *>(raw_depth.get());
  depth_values[0] = 1250;
  depth_values[1] = 0;

  xv::DepthImage image;
  image.type = xv::DepthImage::Type::Depth_16;
  image.width = 2;
  image.height = 1;
  image.data = raw_depth;

  EXPECT_FLOAT_EQ(xv_ros2::rgb_registered::depthValueMeters(image, 0), 1.25F);
  EXPECT_FLOAT_EQ(xv_ros2::rgb_registered::depthValueMeters(image, 1), 0.0F);
}

/**
 * @brief 带上限的 16 位深度转换应过滤无效和超出 RGB registered 有效范围的值。
 */
TEST(RgbRegisteredUtils, FiltersDepth16ByMaximumValidMeters) {
  /** 深度原始值，单位为毫米。 */
  auto raw_depth =
    std::shared_ptr<std::uint8_t>(new std::uint8_t[sizeof(std::uint16_t) * 3],
                                    std::default_delete<std::uint8_t[]>());
  /** 16 位深度视图，用于写入测试样本。 */
  auto *depth_values = reinterpret_cast<std::uint16_t *>(raw_depth.get());
  depth_values[0] = 29999;
  depth_values[1] = 30000;
  depth_values[2] = 32010;

  /** 测试用 SDK 深度图。 */
  xv::DepthImage image;
  image.type = xv::DepthImage::Type::Depth_16;
  image.width = 3;
  image.height = 1;
  image.data = raw_depth;

  EXPECT_FLOAT_EQ(
    xv_ros2::rgb_registered::depthValueMeters(image, 0, 30.0), 29.999F);
  EXPECT_FLOAT_EQ(
    xv_ros2::rgb_registered::depthValueMeters(image, 1, 30.0), 0.0F);
  EXPECT_FLOAT_EQ(
    xv_ros2::rgb_registered::depthValueMeters(image, 2, 30.0), 0.0F);
}

/**
 * @brief 构造 32 位 ToF 深度图并验证米制深度保持不变。
 */
TEST(RgbRegisteredUtils, KeepsDepth32Meters) {
  /** 深度原始值，单位为米。 */
  auto raw_depth =
    std::shared_ptr<std::uint8_t>(new std::uint8_t[sizeof(float) * 2],
                                    std::default_delete<std::uint8_t[]>());
  /** 32 位深度视图，用于写入测试样本。 */
  auto *depth_values = reinterpret_cast<float *>(raw_depth.get());
  depth_values[0] = 2.5F;
  depth_values[1] = -1.0F;

  xv::DepthImage image;
  image.type = xv::DepthImage::Type::Depth_32;
  image.width = 2;
  image.height = 1;
  image.data = raw_depth;

  EXPECT_FLOAT_EQ(xv_ros2::rgb_registered::depthValueMeters(image, 0), 2.5F);
  EXPECT_FLOAT_EQ(xv_ros2::rgb_registered::depthValueMeters(image, 1), 0.0F);
}

/**
 * @brief 带上限的 32 位深度转换应过滤边界值、负值和非有限值。
 */
TEST(RgbRegisteredUtils, FiltersDepth32ByMaximumValidMetersAndFiniteValue) {
  /** 深度原始值，单位为米。 */
  auto raw_depth =
    std::shared_ptr<std::uint8_t>(new std::uint8_t[sizeof(float) * 5],
                                    std::default_delete<std::uint8_t[]>());
  /** 32 位深度视图，用于写入测试样本。 */
  auto *depth_values = reinterpret_cast<float *>(raw_depth.get());
  depth_values[0] = 29.5F;
  depth_values[1] = 30.0F;
  depth_values[2] = -1.0F;
  depth_values[3] = std::numeric_limits<float>::quiet_NaN();
  depth_values[4] = std::numeric_limits<float>::infinity();

  /** 测试用 SDK 深度图。 */
  xv::DepthImage image;
  image.type = xv::DepthImage::Type::Depth_32;
  image.width = 5;
  image.height = 1;
  image.data = raw_depth;

  EXPECT_FLOAT_EQ(
    xv_ros2::rgb_registered::depthValueMeters(image, 0, 30.0), 29.5F);
  EXPECT_FLOAT_EQ(
    xv_ros2::rgb_registered::depthValueMeters(image, 1, 30.0), 0.0F);
  EXPECT_FLOAT_EQ(
    xv_ros2::rgb_registered::depthValueMeters(image, 2, 30.0), 0.0F);
  EXPECT_FLOAT_EQ(
    xv_ros2::rgb_registered::depthValueMeters(image, 3, 30.0), 0.0F);
  EXPECT_FLOAT_EQ(
    xv_ros2::rgb_registered::depthValueMeters(image, 4, 30.0), 0.0F);
}

/**
 * @brief 验证 RGB 与 ToF 帧的时间近邻阈值为 33ms。
 */
TEST(RgbRegisteredUtils, MatchesFramesWithinThirtyThreeMilliseconds) {
  EXPECT_TRUE(xv_ros2::rgb_registered::isTimestampMatch(10.000, 10.033));
  EXPECT_FALSE(xv_ros2::rgb_registered::isTimestampMatch(10.000, 10.034));
}

/**
 * @brief 诊断统计应分别累计有效点、投影结果、越界点和同像素覆盖。
 */
TEST(RgbRegisteredUtils, TracksProjectionDiagnosticCounters) {
  /** RGB registered 投影诊断统计。 */
  xv_ros2::rgb_registered::ProjectionDiagnostics diagnostics;

  diagnostics.recordInvalidDepth();
  diagnostics.recordProjectionFailure();
  diagnostics.recordOutOfBoundsProjection();
  diagnostics.recordStoredDepth(false);
  diagnostics.recordStoredDepth(true);
  diagnostics.recordDiscardedOverlappedDepth();

  EXPECT_EQ(6U, diagnostics.total_points);
  EXPECT_EQ(5U, diagnostics.valid_depth_points);
  EXPECT_EQ(4U, diagnostics.projected_points);
  EXPECT_EQ(1U, diagnostics.projection_failed_points);
  EXPECT_EQ(1U, diagnostics.out_of_bounds_points);
  EXPECT_EQ(2U, diagnostics.stored_pixels);
  EXPECT_EQ(2U, diagnostics.overwritten_pixels);
}

/**
 * @brief 时间配对应从两侧缓存中选择全局最近的一对帧。
 */
TEST(RgbRegisteredUtils, FindsNearestTimestampPairAcrossBothCaches) {
  /** ToF 缓存时间戳，单位为秒。 */
  const std::vector<double> depth_timestamps = {10.031, 10.063};
  /** RGB 缓存时间戳，单位为秒。 */
  const std::vector<double> color_timestamps = {10.000, 10.033};

  /** 最近时间戳配对结果。 */
  const auto match = xv_ros2::rgb_registered::findNearestTimestampPair(
    depth_timestamps, color_timestamps,
    xv_ros2::rgb_registered::kMaxTimestampDiffSeconds);

  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(0U, match->depth_index);
  EXPECT_EQ(1U, match->color_index);
  EXPECT_NEAR(0.002, match->diff_seconds, 1e-9);
}

/**
 * @brief 稳定时间配对应等待两侧未来帧，避免提前选择较旧 RGB 帧。
 */
TEST(RgbRegisteredUtils, WaitsForFutureFramesBeforeStableTimestampMatch) {
  /** 仅有旧 RGB 时，尚不能确认 ToF 最近邻。 */
  const std::vector<double> incomplete_depth_timestamps = {10.362};
  /** 旧 RGB 时间戳，单位为秒。 */
  const std::vector<double> incomplete_color_timestamps = {10.333};

  EXPECT_FALSE(xv_ros2::rgb_registered::findStableNearestTimestampPair(
      incomplete_depth_timestamps, incomplete_color_timestamps,
      xv_ros2::rgb_registered::kMaxTimestampDiffSeconds).has_value());

  /** 已包含候选 RGB 后方 ToF 的时间戳缓存。 */
  const std::vector<double> depth_timestamps = {10.362, 10.395};
  /** 已包含候选 ToF 后方 RGB 的时间戳缓存。 */
  const std::vector<double> color_timestamps = {10.333, 10.370};

  /** 稳定最近时间戳配对结果。 */
  const auto match = xv_ros2::rgb_registered::findStableNearestTimestampPair(
    depth_timestamps, color_timestamps,
    xv_ros2::rgb_registered::kMaxTimestampDiffSeconds);

  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(0U, match->depth_index);
  EXPECT_EQ(1U, match->color_index);
  EXPECT_NEAR(0.008, match->diff_seconds, 1e-9);
}

/**
 * @brief ToF 点应先转 SDK IMU 坐标，再通过 Kalibr T_cam_imu 投影到 cam0
 * 校正图。
 */
TEST(RgbRegisteredUtils, ProjectsTofPointThroughKalibrCam0Frame) {
  /** 测试 Kalibr cam0 标定。 */
  xv_ros2::rgb_fisheye::KalibrCam0Calibration calibration;
  calibration.width = 1280;
  calibration.height = 1280;
  calibration.fx = 100.0;
  calibration.fy = 100.0;
  calibration.cx = 640.0;
  calibration.cy = 640.0;
  calibration.t_cam_imu = {1.0, 0.0, 0.0, 0.30, 0.0, 1.0, 0.0, -0.10,
    0.0, 0.0, 1.0, 0.20, 0.0, 0.0, 0.0, 1.0};

  /** SDK ToF 到 IMU 的外参。 */
  const xv::Transform tof_pose({0.10, 0.20, 0.30},
    {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
  /** ToF 相机坐标下的测试点。 */
  const xv::Vector3d tof_point = {0.20, 0.10, 1.50};
  /** 投影后的像素坐标。 */
  cv::Point2d pixel;
  /** Kalibr cam0 坐标下的深度。 */
  double kalibr_depth_meters = 0.0;

  ASSERT_TRUE(xv_ros2::rgb_registered::projectTofPointToKalibrUndistortedPixel(
      calibration, tof_pose, tof_point, &pixel, &kalibr_depth_meters));

  EXPECT_DOUBLE_EQ(670.0, pixel.x);
  EXPECT_DOUBLE_EQ(650.0, pixel.y);
  EXPECT_DOUBLE_EQ(2.0, kalibr_depth_meters);
}

/**
 * @brief 带旋转和平移的 ToF 外参应按 T_cam_imu * T_imu_tof 链路生效。
 */
TEST(RgbRegisteredUtils, ProjectsTofPointWithRotationAndTranslation)
{
  /** 测试 Kalibr cam0 标定。 */
  xv_ros2::rgb_fisheye::KalibrCam0Calibration calibration;
  calibration.width = 100;
  calibration.height = 100;
  calibration.fx = 100.0;
  calibration.fy = 100.0;
  calibration.cx = 50.0;
  calibration.cy = 50.0;
  calibration.t_cam_imu = {1.0, 0.0, 0.0, 0.20, 0.0, 1.0, 0.0, -0.20,
    0.0, 0.0, 1.0, 0.50, 0.0, 0.0, 0.0, 1.0};
  /** 绕 Z 轴旋转 90 度并平移的 ToF 到 IMU 外参。 */
  const xv::Transform tof_pose(
    {0.10, 0.20, 0.0},
    {0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0});
  /** ToF 相机坐标下的测试点。 */
  const xv::Vector3d tof_point = {0.20, 0.10, 2.0};
  /** 投影后的 RGB 像素坐标。 */
  cv::Point2d pixel;
  /** RGB 相机坐标系 Z 深度。 */
  double rgb_depth_meters = 0.0;

  ASSERT_TRUE(xv_ros2::rgb_registered::projectTofPointToKalibrUndistortedPixel(
      calibration, tof_pose, tof_point, &pixel, &rgb_depth_meters));
  EXPECT_DOUBLE_EQ(58.0, pixel.x);
  EXPECT_DOUBLE_EQ(58.0, pixel.y);
  EXPECT_DOUBLE_EQ(2.5, rgb_depth_meters);
}

/**
 * @brief 构造用于 ToF 投影测试的单位 PDM 标定。
 * @param width 图像宽度。
 * @param height 图像高度。
 * @return SDK ToF 标定。
 */
xv::Calibration makeIdentityTofCalibration(int width, int height)
{
  /** 单位 PDM 相机模型。 */
  xv::PolynomialDistortionCameraModel model;
  model.w = width;
  model.h = height;
  model.fx = 1.0;
  model.fy = 1.0;
  model.u0 = 0.0;
  model.v0 = 0.0;
  model.distor = {0.0, 0.0, 0.0, 0.0, 0.0};
  /** 测试用 SDK ToF 标定。 */
  xv::Calibration calibration;
  calibration.pose = xv::Transform(
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
  calibration.pdcm.push_back(model);
  return calibration;
}

/**
 * @brief 创建指定 16 位数据的 SDK 图像。
 * @param type SDK 图像类型。
 * @param values 像素值。
 * @return 持有独立数据的 SDK 图像。
 */
xv::DepthImage makeUint16Image(
  xv::DepthImage::Type type, const std::vector<std::uint16_t> & values)
{
  /** 图像数据缓存。 */
  auto data = std::shared_ptr<std::uint8_t>(
    new std::uint8_t[values.size() * sizeof(std::uint16_t)],
    std::default_delete<std::uint8_t[]>());
  std::memcpy(data.get(), values.data(), values.size() * sizeof(std::uint16_t));
  /** 测试 SDK 图像。 */
  xv::DepthImage image;
  image.type = type;
  image.width = values.size();
  image.height = 1;
  image.data = data;
  return image;
}

/**
 * @brief 创建指定二维尺寸的 16 位 SDK 图像。
 * @param type SDK 图像类型。
 * @param width 图像宽度。
 * @param height 图像高度。
 * @param values 行优先像素值。
 * @return 持有独立数据的 SDK 图像。
 */
xv::DepthImage makeUint16Image2d(
  xv::DepthImage::Type type, std::size_t width, std::size_t height,
  const std::vector<std::uint16_t> & values)
{
  EXPECT_EQ(width * height, values.size());
  /** 图像数据缓存。 */
  auto data = std::shared_ptr<std::uint8_t>(
    new std::uint8_t[values.size() * sizeof(std::uint16_t)],
    std::default_delete<std::uint8_t[]>());
  std::memcpy(data.get(), values.data(), values.size() * sizeof(std::uint16_t));
  /** 测试 SDK 图像。 */
  xv::DepthImage image;
  image.type = type;
  image.width = width;
  image.height = height;
  image.data = data;
  return image;
}

/**
 * @brief 构造不改变图像坐标的浮点 remap。
 * @param width 图像宽度。
 * @param height 图像高度。
 * @param map_x 输出 x remap。
 * @param map_y 输出 y remap。
 */
void makeIdentityRemap(int width, int height, cv::Mat *map_x, cv::Mat *map_y)
{
  ASSERT_NE(nullptr, map_x);
  ASSERT_NE(nullptr, map_y);
  *map_x = cv::Mat(height, width, CV_32FC1);
  *map_y = cv::Mat(height, width, CV_32FC1);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      map_x->at<float>(y, x) = static_cast<float>(x);
      map_y->at<float>(y, x) = static_cast<float>(y);
    }
  }
}

/**
 * @brief 构造与单位 ToF 投影一致的 Kalibr 针孔标定。
 * @param width 图像宽度。
 * @param height 图像高度。
 * @return 单位 Kalibr 标定。
 */
xv_ros2::rgb_fisheye::KalibrCam0Calibration makeIdentityRgbCalibration(
  int width, int height)
{
  /** 单位 RGB 针孔标定。 */
  xv_ros2::rgb_fisheye::KalibrCam0Calibration calibration;
  calibration.width = width;
  calibration.height = height;
  calibration.fx = 1.0;
  calibration.fy = 1.0;
  calibration.cx = 0.0;
  calibration.cy = 0.0;
  return calibration;
}

/**
 * @brief 恒等内外参应建立与 ToF 同尺寸、逐像素同源的公共虚拟网格。
 */
TEST(RgbRegisteredUtils, CreatesIdentityVirtualRgbdModel)
{
  /** 测试图像宽度。 */
  constexpr int kWidth = 4;
  /** 测试图像高度。 */
  constexpr int kHeight = 3;
  /** 单位 RGB remap。 */
  cv::Mat map_x;
  /** 单位 RGB remap。 */
  cv::Mat map_y;
  makeIdentityRemap(kWidth, kHeight, &map_x, &map_y);
  /** 构造出的公共虚拟模型。 */
  xv_ros2::rgb_registered::VirtualRgbdModel model;
  /** 模型构造错误信息。 */
  std::string error;

  ASSERT_TRUE(xv_ros2::rgb_registered::createVirtualRgbdModel(
      makeIdentityRgbCalibration(kWidth, kHeight),
      makeIdentityTofCalibration(kWidth, kHeight), kWidth, kHeight,
      map_x, map_y, &model, &error)) << error;
  EXPECT_EQ(4U, model.width);
  EXPECT_EQ(3U, model.height);
  EXPECT_DOUBLE_EQ(1.0, model.fx);
  EXPECT_DOUBLE_EQ(1.0, model.fy);
  EXPECT_DOUBLE_EQ(0.0, model.cx);
  EXPECT_DOUBLE_EQ(0.0, model.cy);
  ASSERT_EQ(12U, model.tof_source_indices.size());
  ASSERT_EQ(12U, model.fallback_rgb_pixels.size());
  for (std::size_t index = 0; index < model.tof_source_indices.size(); ++index) {
    EXPECT_EQ(index, model.tof_source_indices[index]);
    EXPECT_TRUE(std::isfinite(model.fallback_rgb_pixels[index].x));
    EXPECT_TRUE(std::isfinite(model.fallback_rgb_pixels[index].y));
  }
}

/**
 * @brief 标定分辨率与运行时分辨率不同时虚拟网格仍应映射到正确源像素。
 */
TEST(RgbRegisteredUtils, CreatesVirtualModelFromScaledTofCalibration)
{
  /** 运行时图像宽度。 */
  constexpr int kWidth = 4;
  /** 运行时图像高度。 */
  constexpr int kHeight = 3;
  /** 原生标定分辨率为运行时两倍的 ToF 标定。 */
  xv::Calibration tof_calibration = makeIdentityTofCalibration(8, 6);
  tof_calibration.pdcm[0].fx = 2.0;
  tof_calibration.pdcm[0].fy = 2.0;
  tof_calibration.pdcm[0].u0 = 0.5;
  tof_calibration.pdcm[0].v0 = 0.5;
  /** 单位 RGB remap x。 */
  cv::Mat map_x;
  /** 单位 RGB remap y。 */
  cv::Mat map_y;
  makeIdentityRemap(kWidth, kHeight, &map_x, &map_y);
  /** 构造出的虚拟模型。 */
  xv_ros2::rgb_registered::VirtualRgbdModel model;

  ASSERT_TRUE(xv_ros2::rgb_registered::createVirtualRgbdModel(
      makeIdentityRgbCalibration(kWidth, kHeight), tof_calibration,
      kWidth, kHeight, map_x, map_y, &model));
  ASSERT_EQ(12U, model.tof_source_indices.size());
  for (std::size_t index = 0; index < model.tof_source_indices.size(); ++index) {
    EXPECT_EQ(index, model.tof_source_indices[index]);
  }
}

/**
 * @brief 带径向畸变的 ToF PDM 模型应完成射线域虚拟映射。
 */
TEST(RgbRegisteredUtils, CreatesVirtualModelFromDistortedTofCalibration)
{
  /** 测试图像宽度。 */
  constexpr int kWidth = 6;
  /** 测试图像高度。 */
  constexpr int kHeight = 4;
  /** 带径向畸变的 ToF PDM 标定。 */
  xv::Calibration tof_calibration = makeIdentityTofCalibration(
    kWidth, kHeight);
  tof_calibration.pdcm[0].fx = 3.0;
  tof_calibration.pdcm[0].fy = 3.0;
  tof_calibration.pdcm[0].u0 = 2.5;
  tof_calibration.pdcm[0].v0 = 1.5;
  tof_calibration.pdcm[0].distor[0] = 0.05;
  /** 与 ToF 无畸变射线域相容的 RGB 标定。 */
  auto rgb_calibration = makeIdentityRgbCalibration(kWidth, kHeight);
  rgb_calibration.fx = 3.0;
  rgb_calibration.fy = 3.0;
  rgb_calibration.cx = 2.5;
  rgb_calibration.cy = 1.5;
  /** 单位 RGB remap x。 */
  cv::Mat map_x;
  /** 单位 RGB remap y。 */
  cv::Mat map_y;
  makeIdentityRemap(kWidth, kHeight, &map_x, &map_y);
  /** 构造出的虚拟模型。 */
  xv_ros2::rgb_registered::VirtualRgbdModel model;
  /** 模型构造错误信息。 */
  std::string error;

  ASSERT_TRUE(xv_ros2::rgb_registered::createVirtualRgbdModel(
      rgb_calibration, tof_calibration, kWidth, kHeight,
      map_x, map_y, &model, &error)) << error;
  EXPECT_EQ(static_cast<std::size_t>(kWidth), model.width);
  EXPECT_EQ(static_cast<std::size_t>(kHeight), model.height);
  EXPECT_EQ(static_cast<std::size_t>(kWidth * kHeight),
    model.tof_source_indices.size());
  EXPECT_EQ(static_cast<std::size_t>(kWidth * kHeight),
    model.fallback_rgb_pixels.size());
}

/**
 * @brief RGB remap 黑边应被裁出公共视场并保持最终回退映射 100% 有效。
 */
TEST(RgbRegisteredUtils, CropsVirtualFieldOfViewToValidRgbRemap)
{
  /** 测试图像宽度。 */
  constexpr int kWidth = 6;
  /** 测试图像高度。 */
  constexpr int kHeight = 4;
  /** 带无效左右边界的 remap x。 */
  cv::Mat map_x;
  /** 带无效左右边界的 remap y。 */
  cv::Mat map_y;
  makeIdentityRemap(kWidth, kHeight, &map_x, &map_y);
  for (int y = 0; y < kHeight; ++y) {
    map_x.at<float>(y, 0) = -1.0F;
    map_x.at<float>(y, kWidth - 1) = -1.0F;
  }
  /** 裁取公共视场后的虚拟模型。 */
  xv_ros2::rgb_registered::VirtualRgbdModel model;
  /** 模型构造错误信息。 */
  std::string error;

  ASSERT_TRUE(xv_ros2::rgb_registered::createVirtualRgbdModel(
      makeIdentityRgbCalibration(kWidth, kHeight),
      makeIdentityTofCalibration(kWidth, kHeight), kWidth, kHeight,
      map_x, map_y, &model, &error)) << error;
  EXPECT_GE(model.min_x, 1.0);
  EXPECT_LE(model.max_x, 4.0);
  EXPECT_GT(model.fx, 1.0);
  ASSERT_EQ(static_cast<std::size_t>(kWidth * kHeight),
    model.fallback_rgb_pixels.size());
  for (const cv::Point2f & pixel : model.fallback_rgb_pixels) {
    EXPECT_GE(pixel.x, 1.0F);
    EXPECT_LE(pixel.x, 4.0F);
    EXPECT_GE(pixel.y, 0.0F);
    EXPECT_LE(pixel.y, 3.0F);
  }
}

/**
 * @brief 无公共有效 RGB remap 时应拒绝创建虚拟相机。
 */
TEST(RgbRegisteredUtils, RejectsVirtualModelWithoutCommonRgbFieldOfView)
{
  /** 全部指向原图外的 x remap。 */
  cv::Mat map_x(3, 4, CV_32FC1, cv::Scalar(-1.0F));
  /** 单位 y remap。 */
  cv::Mat map_y;
  /** 临时单位 x remap。 */
  cv::Mat unused_map_x;
  makeIdentityRemap(4, 3, &unused_map_x, &map_y);
  /** 不应成功创建的虚拟模型。 */
  xv_ros2::rgb_registered::VirtualRgbdModel model;
  /** 模型构造错误信息。 */
  std::string error;

  EXPECT_FALSE(xv_ros2::rgb_registered::createVirtualRgbdModel(
      makeIdentityRgbCalibration(4, 3), makeIdentityTofCalibration(4, 3),
      4U, 3U, map_x, map_y, &model, &error));
  EXPECT_NE(std::string::npos, error.find("common field of view"));
}

/**
 * @brief 有效深度使用完整平移外参取色，无效深度使用旋转回退并保留 IR。
 */
TEST(RgbRegisteredUtils, SamplesVirtualRgbWithTranslationAndInvalidDepthFallback)
{
  /** 测试图像宽度。 */
  constexpr int kWidth = 4;
  /** 测试图像高度。 */
  constexpr int kHeight = 3;
  /** RGB 标定，x 平移使有效深度采样向右移动一个像素。 */
  auto rgb_calibration = makeIdentityRgbCalibration(kWidth, kHeight);
  rgb_calibration.t_cam_imu[3] = 1.0;
  /** 单位 RGB remap x。 */
  cv::Mat map_x;
  /** 单位 RGB remap y。 */
  cv::Mat map_y;
  makeIdentityRemap(kWidth, kHeight, &map_x, &map_y);
  /** 公共虚拟模型。 */
  xv_ros2::rgb_registered::VirtualRgbdModel model;
  /** 单位 ToF 标定。 */
  const xv::Calibration tof_calibration =
    makeIdentityTofCalibration(kWidth, kHeight);
  ASSERT_TRUE(xv_ros2::rgb_registered::createVirtualRgbdModel(
      rgb_calibration, tof_calibration, kWidth, kHeight,
      map_x, map_y, &model));

  /** 按 x 坐标编码红色通道的校正 RGB 图。 */
  cv::Mat rgb(kHeight, kWidth, CV_8UC3);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      rgb.at<cv::Vec3b>(y, x) = cv::Vec3b(
        static_cast<std::uint8_t>(x * 10),
        static_cast<std::uint8_t>(y * 20), 7U);
    }
  }
  /** 第一个像素无效，第二个像素深度为一米。 */
  std::vector<std::uint16_t> depth_values(kWidth * kHeight, 0U);
  depth_values[1] = 1000U;
  /** 每个原始 ToF 像素唯一的 IR 值。 */
  std::vector<std::uint16_t> ir_values(kWidth * kHeight, 0U);
  for (std::size_t index = 0; index < ir_values.size(); ++index) {
    ir_values[index] = static_cast<std::uint16_t>(100U + index);
  }
  /** 二维 ToF 深度图。 */
  const xv::DepthImage depth = makeUint16Image2d(
    xv::DepthImage::Type::Depth_16, kWidth, kHeight, depth_values);
  /** 二维 ToF IR 图。 */
  const xv::DepthImage ir = makeUint16Image2d(
    xv::DepthImage::Type::IR, kWidth, kHeight, ir_values);
  /** 虚拟网格三路输出。 */
  xv_ros2::rgb_registered::RgbToTofImages output;
  /** 配准错误信息。 */
  std::string error;

  ASSERT_TRUE(xv_ros2::rgb_registered::registerRgbToVirtualGrid(
      model, rgb_calibration, tof_calibration, depth, ir, rgb, 30.0,
      &output, &error)) << error;
  ASSERT_EQ(static_cast<std::size_t>(kWidth * kHeight * 3), output.rgb.size());
  EXPECT_EQ(0U, output.rgb[0]);
  EXPECT_FLOAT_EQ(0.0F, output.depth_meters[0]);
  EXPECT_EQ(100U, output.ir[0]);
  EXPECT_EQ(20U, output.rgb[3]);
  EXPECT_FLOAT_EQ(1.0F, output.depth_meters[1]);
  EXPECT_EQ(101U, output.ir[1]);
}

/**
 * @brief 虚拟网格 depth 与 IR 应始终从同一个最近邻 ToF 源索引复制。
 */
TEST(RgbRegisteredUtils, KeepsVirtualDepthAndIrFromSameTofSource)
{
  /** 测试图像宽度。 */
  constexpr int kWidth = 6;
  /** 测试图像高度。 */
  constexpr int kHeight = 4;
  /** 带无效左右边界的 remap x。 */
  cv::Mat map_x;
  /** 单位 remap y。 */
  cv::Mat map_y;
  makeIdentityRemap(kWidth, kHeight, &map_x, &map_y);
  for (int y = 0; y < kHeight; ++y) {
    map_x.at<float>(y, 0) = -1.0F;
    map_x.at<float>(y, kWidth - 1) = -1.0F;
  }
  /** 公共虚拟模型。 */
  xv_ros2::rgb_registered::VirtualRgbdModel model;
  /** 单位 RGB 标定。 */
  const auto rgb_calibration = makeIdentityRgbCalibration(kWidth, kHeight);
  /** 单位 ToF 标定。 */
  const xv::Calibration tof_calibration =
    makeIdentityTofCalibration(kWidth, kHeight);
  ASSERT_TRUE(xv_ros2::rgb_registered::createVirtualRgbdModel(
      rgb_calibration, tof_calibration, kWidth, kHeight,
      map_x, map_y, &model));

  /** 用源索引编码的深度原始值。 */
  std::vector<std::uint16_t> depth_values(kWidth * kHeight, 0U);
  /** 用源索引编码的 IR 原始值。 */
  std::vector<std::uint16_t> ir_values(kWidth * kHeight, 0U);
  for (std::size_t index = 0; index < depth_values.size(); ++index) {
    depth_values[index] = static_cast<std::uint16_t>(1000U + index);
    ir_values[index] = static_cast<std::uint16_t>(2000U + index);
  }
  /** 二维 ToF 深度图。 */
  const xv::DepthImage depth = makeUint16Image2d(
    xv::DepthImage::Type::Depth_16, kWidth, kHeight, depth_values);
  /** 二维 ToF IR 图。 */
  const xv::DepthImage ir = makeUint16Image2d(
    xv::DepthImage::Type::IR, kWidth, kHeight, ir_values);
  /** 非零常量 RGB 图。 */
  const cv::Mat rgb(kHeight, kWidth, CV_8UC3, cv::Scalar(9U, 8U, 7U));
  /** 虚拟网格输出。 */
  xv_ros2::rgb_registered::RgbToTofImages output;

  ASSERT_TRUE(xv_ros2::rgb_registered::registerRgbToVirtualGrid(
      model, rgb_calibration, tof_calibration, depth, ir, rgb, 30.0,
      &output));
  for (std::size_t index = 0; index < model.tof_source_indices.size(); ++index) {
    /** 当前虚拟像素对应的 ToF 源索引。 */
    const std::size_t source_index = model.tof_source_indices[index];
    EXPECT_FLOAT_EQ(
      static_cast<float>(depth_values[source_index]) * 0.001F,
      output.depth_meters[index]);
    EXPECT_EQ(ir_values[source_index], output.ir[index]);
    EXPECT_EQ(9U, output.rgb[index * 3U]);
    EXPECT_EQ(8U, output.rgb[index * 3U + 1U]);
    EXPECT_EQ(7U, output.rgb[index * 3U + 2U]);
  }
}

/**
 * @brief 深度与 IR 应投影到同一 RGB 像素并保持同源。
 */
TEST(RgbRegisteredUtils, RegistersDepthAndIrToSameRgbGrid)
{
  /** 单位 Kalibr 输出标定。 */
  xv_ros2::rgb_fisheye::KalibrCam0Calibration calibration;
  calibration.width = 2;
  calibration.height = 1;
  calibration.fx = 1.0;
  calibration.fy = 1.0;
  calibration.cx = 0.0;
  calibration.cy = 0.0;
  /** 两像素毫米制深度图。 */
  const xv::DepthImage depth = makeUint16Image(
    xv::DepthImage::Type::Depth_16, {2000U, 1000U});
  /** 与深度同网格的 IR 图。 */
  const xv::DepthImage ir = makeUint16Image(
    xv::DepthImage::Type::IR, {10U, 20U});
  /** 输出对齐图像。 */
  xv_ros2::rgb_registered::RegisteredTofImages output;
  /** 投影错误信息。 */
  std::string error;

  ASSERT_TRUE(xv_ros2::rgb_registered::registerTofImagesToRgb(
      calibration, makeIdentityTofCalibration(2, 1), depth, &ir, 30.0,
      &output, nullptr, &error)) << error;
  ASSERT_EQ(2U, output.depth_meters.size());
  ASSERT_EQ(2U, output.ir.size());
  EXPECT_FLOAT_EQ(2.0F, output.depth_meters[0]);
  EXPECT_FLOAT_EQ(1.0F, output.depth_meters[1]);
  EXPECT_EQ(10U, output.ir[0]);
  EXPECT_EQ(20U, output.ir[1]);
}

/**
 * @brief 运行时深度分辨率与 SDK 标定分辨率不同时应缩放像素后反投影。
 */
TEST(RgbRegisteredUtils, AdaptsCalibrationModelToRuntimeDepthResolution)
{
  /** 两像素 RGB 输出网格。 */
  xv_ros2::rgb_fisheye::KalibrCam0Calibration calibration;
  calibration.width = 2;
  calibration.height = 1;
  calibration.fx = 1.0;
  calibration.fy = 1.0;
  calibration.cx = 0.0;
  calibration.cy = 0.0;
  /** 原生标定分辨率为运行时深度两倍的 SDK ToF 标定。 */
  xv::Calibration tof_calibration = makeIdentityTofCalibration(4, 2);
  tof_calibration.pdcm[0].fx = 2.0;
  tof_calibration.pdcm[0].fy = 2.0;
  tof_calibration.pdcm[0].u0 = 0.5;
  tof_calibration.pdcm[0].v0 = 0.5;
  /** 运行时两像素深度图。 */
  const xv::DepthImage depth = makeUint16Image(
    xv::DepthImage::Type::Depth_16, {1000U, 2000U});
  /** 与运行时深度同分辨率的 IR 图。 */
  const xv::DepthImage ir = makeUint16Image(
    xv::DepthImage::Type::IR, {10U, 20U});
  /** 配准输出。 */
  xv_ros2::rgb_registered::RegisteredTofImages output;

  ASSERT_TRUE(xv_ros2::rgb_registered::registerTofImagesToRgb(
      calibration, tof_calibration, depth, &ir, 30.0, &output));
  ASSERT_EQ(2U, output.depth_meters.size());
  EXPECT_FLOAT_EQ(1.0F, output.depth_meters[0]);
  EXPECT_FLOAT_EQ(2.0F, output.depth_meters[1]);
  EXPECT_EQ(10U, output.ir[0]);
  EXPECT_EQ(20U, output.ir[1]);
}

/**
 * @brief RGB 应采样到 ToF 原生网格，深度无效时使用旋转回退并保留 IR。
 */
TEST(RgbRegisteredUtils, SamplesRgbOnTofGridAndKeepsIrForInvalidDepth)
{
  /** 与测试 RGB 图一致的单位 Kalibr 标定。 */
  xv_ros2::rgb_fisheye::KalibrCam0Calibration calibration;
  calibration.width = 2;
  calibration.height = 1;
  calibration.fx = 1.0;
  calibration.fy = 1.0;
  calibration.cx = 0.0;
  calibration.cy = 0.0;
  /** 两像素校正 RGB 图。 */
  cv::Mat rgb(1, 2, CV_8UC3);
  rgb.at<cv::Vec3b>(0, 0) = cv::Vec3b(10U, 20U, 30U);
  rgb.at<cv::Vec3b>(0, 1) = cv::Vec3b(40U, 50U, 60U);
  /** 第一像素有效、第二像素无效的 ToF 深度。 */
  const xv::DepthImage depth = makeUint16Image(
    xv::DepthImage::Type::Depth_16, {1000U, 0U});
  /** 深度无效位置仍有强度的 ToF IR。 */
  const xv::DepthImage ir = makeUint16Image(
    xv::DepthImage::Type::IR, {100U, 200U});
  /** ToF 网格三路输出。 */
  xv_ros2::rgb_registered::RgbToTofImages output;

  ASSERT_TRUE(xv_ros2::rgb_registered::registerRgbToTofGrid(
      calibration, makeIdentityTofCalibration(2, 1), depth, ir, rgb, 30.0,
      &output));
  ASSERT_EQ(6U, output.rgb.size());
  EXPECT_EQ(10U, output.rgb[0]);
  EXPECT_EQ(20U, output.rgb[1]);
  EXPECT_EQ(30U, output.rgb[2]);
  EXPECT_EQ(40U, output.rgb[3]);
  EXPECT_EQ(50U, output.rgb[4]);
  EXPECT_EQ(60U, output.rgb[5]);
  ASSERT_EQ(2U, output.depth_meters.size());
  EXPECT_FLOAT_EQ(1.0F, output.depth_meters[0]);
  EXPECT_FLOAT_EQ(0.0F, output.depth_meters[1]);
  ASSERT_EQ(2U, output.ir.size());
  EXPECT_EQ(100U, output.ir[0]);
  EXPECT_EQ(200U, output.ir[1]);
}

/**
 * @brief 多个 ToF 点落入同一 RGB 像素时应保留最近深度及其 IR。
 */
TEST(RgbRegisteredUtils, ZBufferKeepsIrFromNearestDepth)
{
  /** 将两个源像素压缩到同一目标像素的标定。 */
  xv_ros2::rgb_fisheye::KalibrCam0Calibration calibration;
  calibration.width = 1;
  calibration.height = 1;
  calibration.fx = 0.1;
  calibration.fy = 1.0;
  calibration.cx = 0.0;
  calibration.cy = 0.0;
  /** 两个候选深度。 */
  const xv::DepthImage depth = makeUint16Image(
    xv::DepthImage::Type::Depth_16, {2000U, 1000U});
  /** 候选深度对应 IR。 */
  const xv::DepthImage ir = makeUint16Image(
    xv::DepthImage::Type::IR, {100U, 200U});
  /** 输出对齐图像。 */
  xv_ros2::rgb_registered::RegisteredTofImages output;

  ASSERT_TRUE(xv_ros2::rgb_registered::registerTofImagesToRgb(
      calibration, makeIdentityTofCalibration(2, 1), depth, &ir, 30.0,
      &output));
  ASSERT_EQ(1U, output.depth_meters.size());
  EXPECT_FLOAT_EQ(1.0F, output.depth_meters[0]);
  EXPECT_EQ(200U, output.ir[0]);
}

/**
 * @brief IR 与深度尺寸不一致时应拒绝配准。
 */
TEST(RgbRegisteredUtils, RejectsMismatchedDepthAndIrResolution)
{
  /** 单位 Kalibr 输出标定。 */
  xv_ros2::rgb_fisheye::KalibrCam0Calibration calibration;
  calibration.width = 2;
  calibration.height = 1;
  calibration.fx = 1.0;
  calibration.fy = 1.0;
  /** 两像素深度图。 */
  const xv::DepthImage depth = makeUint16Image(
    xv::DepthImage::Type::Depth_16, {1000U, 1000U});
  /** 单像素 IR 图。 */
  const xv::DepthImage ir = makeUint16Image(
    xv::DepthImage::Type::IR, {100U});
  /** 输出对齐图像。 */
  xv_ros2::rgb_registered::RegisteredTofImages output;
  /** 配准错误信息。 */
  std::string error;

  EXPECT_FALSE(xv_ros2::rgb_registered::registerTofImagesToRgb(
      calibration, makeIdentityTofCalibration(2, 1), depth, &ir, 30.0,
      &output, nullptr, &error));
  EXPECT_NE(std::string::npos, error.find("resolution"));
}

/**
 * @brief 无效深度及投影越界位置应同时保持 depth 和 IR 为零。
 */
TEST(RgbRegisteredUtils, LeavesDepthAndIrZeroForInvalidOrOutOfBoundsPoints)
{
  /** 单像素 Kalibr 输出网格。 */
  xv_ros2::rgb_fisheye::KalibrCam0Calibration calibration;
  calibration.width = 1;
  calibration.height = 1;
  calibration.fx = 1.0;
  calibration.fy = 1.0;
  calibration.cx = 0.0;
  calibration.cy = 0.0;
  /** 首点深度无效，第二点会投影到输出网格外。 */
  const xv::DepthImage depth = makeUint16Image(
    xv::DepthImage::Type::Depth_16, {0U, 1000U});
  /** 两个源点均带有非零 IR，用于确认不会单独写入。 */
  const xv::DepthImage ir = makeUint16Image(
    xv::DepthImage::Type::IR, {123U, 456U});
  /** 对齐输出。 */
  xv_ros2::rgb_registered::RegisteredTofImages output;

  ASSERT_TRUE(xv_ros2::rgb_registered::registerTofImagesToRgb(
      calibration, makeIdentityTofCalibration(2, 1), depth, &ir, 30.0,
      &output));
  ASSERT_EQ(1U, output.depth_meters.size());
  ASSERT_EQ(1U, output.ir.size());
  EXPECT_FLOAT_EQ(0.0F, output.depth_meters[0]);
  EXPECT_EQ(0U, output.ir[0]);
}

/**
 * @brief 三路同步应选择相对深度时间最接近且已被未来帧确认的组合。
 */
TEST(RgbRegisteredUtils, FindsStableRgbDepthIrTriple)
{
  /** 深度时间戳。 */
  const std::vector<double> depth_timestamps = {10.100, 10.140};
  /** RGB 时间戳。 */
  const std::vector<double> color_timestamps = {10.070, 10.102};
  /** IR 时间戳。 */
  const std::vector<double> ir_timestamps = {10.099, 10.141};

  /** 稳定三帧匹配。 */
  const auto match = xv_ros2::rgb_registered::findStableNearestTimestampTriple(
    depth_timestamps, color_timestamps, ir_timestamps,
    xv_ros2::rgb_registered::kMaxTimestampDiffSeconds);

  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(0U, match->depth_index);
  EXPECT_EQ(1U, match->color_index);
  EXPECT_EQ(0U, match->ir_index);
  EXPECT_NEAR(0.002, match->color_diff_seconds, 1e-9);
  EXPECT_NEAR(0.001, match->ir_diff_seconds, 1e-9);
}

/**
 * @brief 三路同步应限制任意两帧的最大时间差，避免 RGB 与 IR 相差 66ms。
 */
TEST(RgbRegisteredUtils, RejectsTripleWhoseEndpointsExceedTolerance)
{
  /** 深度时间戳位于首组 RGB 与 IR 中间。 */
  const std::vector<double> depth_timestamps = {10.033, 10.100};
  /** 首帧 RGB 早于深度 33ms。 */
  const std::vector<double> color_timestamps = {10.000, 10.100};
  /** 首帧 IR 晚于深度 33ms。 */
  const std::vector<double> ir_timestamps = {10.066, 10.100};

  /** 端点相差 66ms 的首组无效，应选择后续完整同步帧。 */
  const auto match = xv_ros2::rgb_registered::findStableNearestTimestampTriple(
    depth_timestamps, color_timestamps, ir_timestamps,
    xv_ros2::rgb_registered::kMaxTimestampDiffSeconds);

  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(1U, match->depth_index);
  EXPECT_EQ(1U, match->color_index);
  EXPECT_EQ(1U, match->ir_index);
}
