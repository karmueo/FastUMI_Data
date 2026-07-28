/**
 * @file rgbd_image_utils.h
 * @brief 声明 SDK DepthColorImage 原始兼容数据的拆分转换工具。
 */

#ifndef __RGBD_IMAGE_UTILS_H__
#define __RGBD_IMAGE_UTILS_H__

#include <sensor_msgs/msg/image.hpp>
#include <xv-sdk.h>

#include <string>

namespace xv_ros2
{
namespace rgbd
{
/**
 * @brief 将 SDK DepthColorImage 的 RGB 字节拆分为 ROS RGB8 图像。
 * @param xvDepthColorImage SDK RGBD 图像，像素布局为 RGB 3 字节加 float 深度 4 字节。
 * @param frame_id ROS 图像坐标系。
 * @return RGB8 编码的 ROS 图像。
 */
sensor_msgs::msg::Image toRosRGBDColorImage(const xv::DepthColorImage &xvDepthColorImage,
                                            const std::string &frame_id);

/**
 * @brief 将 SDK DepthColorImage 的 float 深度字节拆分为 ROS 32FC1 深度图。
 * @param xvDepthColorImage SDK RGBD 图像，像素布局为 RGB 3 字节加 float 深度 4 字节。
 * @param frame_id ROS 图像坐标系。
 * @return 32FC1 编码的 ROS 深度图，单位为米。
 */
sensor_msgs::msg::Image toRosRGBDDepthImage(const xv::DepthColorImage &xvDepthColorImage,
                                            const std::string &frame_id);
}  // namespace rgbd
}  // namespace xv_ros2

#endif  // __RGBD_IMAGE_UTILS_H__
