/**
 * @file test_rgbd_image_utils.cpp
 * @brief 验证 SDK RGBD 图像拆分为 ROS RGB 与深度图像的转换逻辑。
 */

#include "rgbd_image_utils.h"

#include <gtest/gtest.h>
#include <sensor_msgs/image_encodings.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace
{
/**
 * @brief 向 RGBD 像素缓存写入一个 RGB + float 深度像素。
 * @param data RGBD 像素缓存。
 * @param pixel_index 像素下标。
 * @param red 红色通道。
 * @param green 绿色通道。
 * @param blue 蓝色通道。
 * @param depth_meters 米制深度值。
 */
void writeRgbdPixel(std::vector<std::uint8_t> &data,
                    std::size_t pixel_index,
                    std::uint8_t red,
                    std::uint8_t green,
                    std::uint8_t blue,
                    float depth_meters)
{
    /** 单个 SDK RGBD 像素的字节数。 */
    constexpr std::size_t kPixelStride = 7;
    /** 当前像素起始偏移。 */
    const std::size_t offset = pixel_index * kPixelStride;

    data[offset] = red;
    data[offset + 1] = green;
    data[offset + 2] = blue;
    std::memcpy(data.data() + offset + 3, &depth_meters, sizeof(depth_meters));
}

/**
 * @brief 从字节缓存读取 float。
 * @param data 字节缓存。
 * @param offset 起始偏移。
 * @return 读取到的 float 值。
 */
float readFloat(const std::vector<std::uint8_t> &data, std::size_t offset)
{
    /** 读取出的 float 值。 */
    float value = 0.0F;
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}
}  // namespace

/**
 * @brief SDK RGBD 彩色部分应拆分为 RGB8 图像。
 */
TEST(RgbdImageUtilsTest, ConvertsDepthColorImageToRgb8Image)
{
    /** SDK RGBD 图像原始字节缓存。 */
    auto data = std::make_shared<std::vector<std::uint8_t>>(2U * 7U);
    writeRgbdPixel(*data, 0U, 1U, 2U, 3U, 1.25F);
    writeRgbdPixel(*data, 1U, 4U, 5U, 6U, 2.5F);

    /** SDK RGBD 图像。 */
    xv::DepthColorImage image;
    image.width = 2U;
    image.height = 1U;
    image.hostTimestamp = 42.25;
    image.data = std::shared_ptr<const std::uint8_t>(data, data->data());

    /** 转换后的 ROS RGB 图像。 */
    const sensor_msgs::msg::Image ros_image = xv_ros2::rgbd::toRosRGBDColorImage(image, "rgbd_frame");

    EXPECT_EQ("rgbd_frame", ros_image.header.frame_id);
    EXPECT_EQ(42, ros_image.header.stamp.sec);
    EXPECT_EQ(250000000U, ros_image.header.stamp.nanosec);
    EXPECT_EQ(1U, ros_image.height);
    EXPECT_EQ(2U, ros_image.width);
    EXPECT_EQ(sensor_msgs::image_encodings::RGB8, ros_image.encoding);
    EXPECT_FALSE(ros_image.is_bigendian);
    EXPECT_EQ(6U, ros_image.step);
    EXPECT_EQ((std::vector<std::uint8_t>{1U, 2U, 3U, 4U, 5U, 6U}), ros_image.data);
}

/**
 * @brief SDK RGBD 深度部分应拆分为 32FC1 米制深度图像。
 */
TEST(RgbdImageUtilsTest, ConvertsDepthColorImageTo32FloatDepthImage)
{
    /** SDK RGBD 图像原始字节缓存。 */
    auto data = std::make_shared<std::vector<std::uint8_t>>(2U * 7U);
    writeRgbdPixel(*data, 0U, 1U, 2U, 3U, 1.25F);
    writeRgbdPixel(*data, 1U, 4U, 5U, 6U, 2.5F);

    /** SDK RGBD 图像。 */
    xv::DepthColorImage image;
    image.width = 2U;
    image.height = 1U;
    image.hostTimestamp = 42.25;
    image.data = std::shared_ptr<const std::uint8_t>(data, data->data());

    /** 转换后的 ROS 深度图像。 */
    const sensor_msgs::msg::Image ros_image = xv_ros2::rgbd::toRosRGBDDepthImage(image, "rgbd_frame");

    EXPECT_EQ("rgbd_frame", ros_image.header.frame_id);
    EXPECT_EQ(42, ros_image.header.stamp.sec);
    EXPECT_EQ(250000000U, ros_image.header.stamp.nanosec);
    EXPECT_EQ(1U, ros_image.height);
    EXPECT_EQ(2U, ros_image.width);
    EXPECT_EQ(sensor_msgs::image_encodings::TYPE_32FC1, ros_image.encoding);
    EXPECT_FALSE(ros_image.is_bigendian);
    EXPECT_EQ(8U, ros_image.step);
    ASSERT_EQ(2U * sizeof(float), ros_image.data.size());
    EXPECT_FLOAT_EQ(1.25F, readFloat(ros_image.data, 0U));
    EXPECT_FLOAT_EQ(2.5F, readFloat(ros_image.data, sizeof(float)));
}
