/**
 * @file test_factory_rgbd_utils.cpp
 * @brief 验证 factory RGB-D 标定解析、投影、深度对齐和帧匹配工具。
 */

#include "factory_rgbd_utils.h"
#include "rgb_registered_utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace
{
/**
 * @brief 构造用于测试的最小 factory 标定。
 * @return 具备近似针孔行为和单位外参的标定对象。
 */
xv_ros2::factory_rgbd::FactoryCalibration makeSimpleCalibration()
{
    /** 测试用 factory 标定。 */
    xv_ros2::factory_rgbd::FactoryCalibration calibration;
    calibration.rgb.width = 3;
    calibration.rgb.height = 3;
    calibration.rgb.fx = 1.0;
    calibration.rgb.fy = 1.0;
    calibration.rgb.u0 = 1.0;
    calibration.rgb.v0 = 1.0;
    calibration.rgb.eu = 1.0;
    calibration.rgb.ev = 1.0;
    calibration.rgb.alpha = 0.0;
    calibration.rgb.beta = 1.0;

    calibration.tof.width = 2;
    calibration.tof.height = 1;
    calibration.tof.fx = 1000.0;
    calibration.tof.fy = 1000.0;
    calibration.tof.u0 = 0.5;
    calibration.tof.v0 = 0.0;
    calibration.tof.distor = {0.0, 0.0, 0.0, 0.0, 0.0};

    calibration.t_rgb_tof = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0};
    return calibration;
}

/**
 * @brief 构造 16 位 ToF 深度图。
 * @param width 图像宽度。
 * @param height 图像高度。
 * @param millimeters 深度值，单位为毫米。
 * @return SDK 深度图对象。
 */
xv::DepthImage makeDepth16(std::size_t width,
                           std::size_t height,
                           const std::vector<std::uint16_t> &millimeters)
{
    /** 深度字节缓存。 */
    auto raw_depth = std::shared_ptr<std::uint8_t>(
        new std::uint8_t[sizeof(std::uint16_t) * millimeters.size()],
        std::default_delete<std::uint8_t[]>());
    /** 16 位深度视图。 */
    auto *depth_values = reinterpret_cast<std::uint16_t *>(raw_depth.get());
    for (std::size_t index = 0; index < millimeters.size(); ++index)
    {
        depth_values[index] = millimeters[index];
    }

    /** SDK 深度图对象。 */
    xv::DepthImage image;
    image.type = xv::DepthImage::Type::Depth_16;
    image.width = width;
    image.height = height;
    image.data = raw_depth;
    return image;
}
}  // namespace

/**
 * @brief factory YAML 应能解析出 RGB/ToF 内参和 T_rgb_tof。
 */
TEST(FactoryRgbdUtils, LoadsFactoryCalibrationYaml)
{
    /** 标定文件路径。 */
    const std::string calibration_path =
        std::string(XV_SDK_ROS2_TEST_FIXTURE_DIR) +
        "/factory_calibration.yaml";
    /** 解析错误信息。 */
    std::string error;
    /** 解析得到的标定。 */
    const auto calibration =
        xv_ros2::factory_rgbd::loadFactoryCalibration(calibration_path, &error);

    ASSERT_TRUE(calibration.has_value()) << error;
    EXPECT_EQ(1280, calibration->rgb.width);
    EXPECT_EQ(1280, calibration->rgb.height);
    EXPECT_DOUBLE_EQ(396.109802246, calibration->rgb.fx);
    EXPECT_DOUBLE_EQ(0.68319350481, calibration->rgb.alpha);
    EXPECT_EQ(640, calibration->tof.width);
    EXPECT_EQ(480, calibration->tof.height);
    EXPECT_DOUBLE_EQ(509.14440918, calibration->tof.fx);
    EXPECT_DOUBLE_EQ(0.0349495597184, calibration->tof.distor[0]);
    EXPECT_NEAR(0.00365931550617841, calibration->t_rgb_tof[3], 1e-12);
}

/**
 * @brief factory 配对应等待未来 RGB 帧后选择真正的最近邻。
 */
TEST(FactoryRgbdUtils, WaitsForFutureRgbFrameBeforeMatching)
{
    /** ToF 缓存时间戳，单位为秒。 */
    const std::vector<double> depth_timestamps{10.030};
    /** 仅包含先到 RGB 帧的不完整缓存。 */
    const std::vector<double> incomplete_color_timestamps{10.000};

    EXPECT_FALSE(xv_ros2::rgb_registered::findStableNearestTimestampPair(
        depth_timestamps, incomplete_color_timestamps, 0.033).has_value());

    /** 加入未来 RGB 帧后的缓存。 */
    const std::vector<double> color_timestamps{10.000, 10.033};

    EXPECT_FALSE(xv_ros2::rgb_registered::findStableNearestTimestampPair(
        depth_timestamps, color_timestamps, 0.033).has_value());

    /** 加入未来 ToF 帧后的完整缓存。 */
    const std::vector<double> complete_depth_timestamps{10.030, 10.063};
    /** 未来帧确认后的最近邻配对。 */
    const auto match = xv_ros2::rgb_registered::findStableNearestTimestampPair(
        complete_depth_timestamps, color_timestamps, 0.033);

    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(0U, match->depth_index);
    EXPECT_EQ(1U, match->color_index);
    EXPECT_NEAR(0.003, match->diff_seconds, 1e-9);
}

/**
 * @brief SEUCM 投影中心点应落在 RGB 主点。
 */
TEST(FactoryRgbdUtils, ProjectsSeucmCenterPoint)
{
    /** 测试标定。 */
    auto calibration = makeSimpleCalibration();
    /** RGB 像素坐标。 */
    std::array<double, 2> pixel{};

    ASSERT_TRUE(xv_ros2::factory_rgbd::projectRgbSeucm(
        calibration.rgb, {0.0, 0.0, 1.0}, pixel));

    EXPECT_NEAR(1.0, pixel[0], 1e-9);
    EXPECT_NEAR(1.0, pixel[1], 1e-9);
}

/**
 * @brief ToF PDCM 反投影应输出米制 ToF 三维点。
 */
TEST(FactoryRgbdUtils, BackProjectsTofPdcmCenterPoint)
{
    /** 测试标定。 */
    auto calibration = makeSimpleCalibration();
    /** ToF 三维点。 */
    std::array<double, 3> point{};

    ASSERT_TRUE(xv_ros2::factory_rgbd::backProjectTofPdcm(
        calibration.tof, 0.5, 0.0, 2.0, point));

    EXPECT_NEAR(0.0, point[0], 1e-9);
    EXPECT_NEAR(0.0, point[1], 1e-9);
    EXPECT_NEAR(2.0, point[2], 1e-9);
}

/**
 * @brief 多个 ToF 点落到同一 RGB 像素时应保留更近深度。
 */
TEST(FactoryRgbdUtils, AlignedDepthKeepsNearestPoint)
{
    /** 测试标定。 */
    const auto calibration = makeSimpleCalibration();
    /** 两个会落到同一 RGB 像素的 ToF 深度。 */
    const xv::DepthImage depth_image = makeDepth16(2, 1, {2000, 1000});
    /** 输出 RGB 深度缓存。 */
    std::vector<float> aligned_depth;

    ASSERT_TRUE(xv_ros2::factory_rgbd::alignDepthToRgb(
        calibration, depth_image, aligned_depth));

    ASSERT_EQ(9U, aligned_depth.size());
    EXPECT_FLOAT_EQ(1.0F, aligned_depth[4]);
}

/**
 * @brief 无效深度和越界投影不应写入输出深度。
 */
TEST(FactoryRgbdUtils, InvalidAndOutOfBoundsDepthStayZero)
{
    /** 测试标定。 */
    auto calibration = makeSimpleCalibration();
    calibration.t_rgb_tof[3] = 100.0;
    /** 无效和越界样本深度。 */
    const xv::DepthImage depth_image = makeDepth16(2, 1, {0, 1000});
    /** 输出 RGB 深度缓存。 */
    std::vector<float> aligned_depth;

    ASSERT_TRUE(xv_ros2::factory_rgbd::alignDepthToRgb(
        calibration, depth_image, aligned_depth));

    EXPECT_TRUE(std::all_of(aligned_depth.begin(), aligned_depth.end(),
                            [](float depth) { return depth == 0.0F; }));
}

/**
 * @brief RGB/ToF 最近邻匹配应遵守调用方传入的时间阈值。
 */
TEST(FactoryRgbdUtils, FindsNearestTimestampWithinTolerance)
{
    /** 候选帧时间戳。 */
    const std::vector<double> timestamps{9.950, 10.020, 10.040};
    /** 33ms 阈值内的匹配结果。 */
    const auto matched =
        xv_ros2::factory_rgbd::findNearestTimestampIndex(timestamps, 10.000, 0.033);
    /** 阈值外的匹配结果。 */
    const auto rejected =
        xv_ros2::factory_rgbd::findNearestTimestampIndex(timestamps, 9.900, 0.033);

    ASSERT_TRUE(matched.has_value());
    EXPECT_EQ(1U, *matched);
    EXPECT_FALSE(rejected.has_value());
}
