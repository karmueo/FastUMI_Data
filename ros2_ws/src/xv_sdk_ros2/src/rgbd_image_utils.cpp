/**
 * @file rgbd_image_utils.cpp
 * @brief 实现 SDK DepthColorImage 原始兼容数据到 RGB8 与 32FC1 图像的拆分。
 */

#include "rgbd_image_utils.h"
#include <sensor_msgs/image_encodings.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
/** 单个 SDK RGBD 像素的 RGB 字节数。 */
constexpr std::size_t kRgbBytesPerPixel = 3U;
/** 单个 SDK RGBD 像素的 depth 字节数。 */
constexpr std::size_t kDepthBytesPerPixel = sizeof(float);
/** 单个 SDK RGBD 像素的总字节数。 */
constexpr std::size_t kRgbdBytesPerPixel = kRgbBytesPerPixel + kDepthBytesPerPixel;
}  // namespace

namespace xv_ros2
{
namespace rgbd
{
/**
 * @brief 将 SDK RGBD 帧中的 RGB 字节拆分为 ROS RGB8 图像。
 * @param xvDepthColorImage SDK RGBD 图像。
 * @param frame_id ROS 图像坐标系。
 * @param stamp 已转换到 Unix 时间域的 ROS 时间戳。
 * @return RGB8 编码的 ROS 图像。
 */
sensor_msgs::msg::Image toRosRGBDColorImage(const xv::DepthColorImage &xvDepthColorImage,
                                            const std::string &frame_id,
                                            const builtin_interfaces::msg::Time &stamp)
{
    /** 转换后的 ROS RGB 图像。 */
    sensor_msgs::msg::Image rosImage;
    rosImage.header.stamp = stamp;
    rosImage.header.frame_id = frame_id;
    rosImage.height = static_cast<std::uint32_t>(xvDepthColorImage.height);
    rosImage.width = static_cast<std::uint32_t>(xvDepthColorImage.width);
    rosImage.encoding = sensor_msgs::image_encodings::RGB8;
    rosImage.is_bigendian = false;
    rosImage.step = rosImage.width * kRgbBytesPerPixel;

    /** RGBD 像素数量。 */
    const std::size_t pixel_count = xvDepthColorImage.width * xvDepthColorImage.height;
    rosImage.data.resize(pixel_count * kRgbBytesPerPixel);

    if (!xvDepthColorImage.data)
    {
        return rosImage;
    }

    /** SDK RGBD 原始像素指针。 */
    const std::uint8_t *rgbd_data = xvDepthColorImage.data.get();
    for (std::size_t pixel_index = 0U; pixel_index < pixel_count; ++pixel_index)
    {
        /** 当前 SDK RGBD 像素的源偏移。 */
        const std::size_t src_offset = pixel_index * kRgbdBytesPerPixel;
        /** 当前 ROS RGB 像素的目标偏移。 */
        const std::size_t dst_offset = pixel_index * kRgbBytesPerPixel;
        std::memcpy(rosImage.data.data() + dst_offset,
                    rgbd_data + src_offset,
                    kRgbBytesPerPixel);
    }

    return rosImage;
}

/**
 * @brief 将 SDK RGBD 帧中的 float 深度拆分为 ROS 32FC1 图像。
 * @param xvDepthColorImage SDK RGBD 图像。
 * @param frame_id ROS 图像坐标系。
 * @param stamp 已转换到 Unix 时间域的 ROS 时间戳。
 * @return 32FC1 编码的 ROS 深度图，单位为米。
 */
sensor_msgs::msg::Image toRosRGBDDepthImage(const xv::DepthColorImage &xvDepthColorImage,
                                            const std::string &frame_id,
                                            const builtin_interfaces::msg::Time &stamp)
{
    /** 转换后的 ROS 深度图像。 */
    sensor_msgs::msg::Image rosImage;
    rosImage.header.stamp = stamp;
    rosImage.header.frame_id = frame_id;
    rosImage.height = static_cast<std::uint32_t>(xvDepthColorImage.height);
    rosImage.width = static_cast<std::uint32_t>(xvDepthColorImage.width);
    rosImage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    rosImage.is_bigendian = false;
    rosImage.step = rosImage.width * kDepthBytesPerPixel;

    /** RGBD 像素数量。 */
    const std::size_t pixel_count = xvDepthColorImage.width * xvDepthColorImage.height;
    rosImage.data.resize(pixel_count * kDepthBytesPerPixel);

    if (!xvDepthColorImage.data)
    {
        return rosImage;
    }

    /** SDK RGBD 原始像素指针。 */
    const std::uint8_t *rgbd_data = xvDepthColorImage.data.get();
    for (std::size_t pixel_index = 0U; pixel_index < pixel_count; ++pixel_index)
    {
        /** 当前 SDK RGBD 像素的源偏移。 */
        const std::size_t src_offset = pixel_index * kRgbdBytesPerPixel + kRgbBytesPerPixel;
        /** 当前 ROS 深度像素的目标偏移。 */
        const std::size_t dst_offset = pixel_index * kDepthBytesPerPixel;
        std::memcpy(rosImage.data.data() + dst_offset,
                    rgbd_data + src_offset,
                    kDepthBytesPerPixel);
    }

    return rosImage;
}
}  // namespace rgbd
}  // namespace xv_ros2
