/**
 * @file frame_utils.hpp
 * @brief 声明 ROS 2 图像与 IMU 帧转换及时间映射工具。
 * @author 待确认
 * @date 创建：待确认
 * @date 修改：2026-08-13
 */

#ifndef TOF_STEREO_CAMERA__FRAME_UTILS_HPP_
#define TOF_STEREO_CAMERA__FRAME_UTILS_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "stereo_camera/stereo_camera.h"

namespace tof_stereo_camera {

constexpr int kItofDepthStreamId = 2;  ///< iTOF 深度图在 SDK 中使用的流 ID。
constexpr int kItofGrayStreamId = 7;   ///< iTOF 灰度图在 SDK 中使用的流 ID。

/**
 * @brief 判断 iTOF 时间匹配状态是否允许发布。
 * @param[in] match_state SDK 的 `STEREO_MATCH_*` 匹配状态。
 * @return 状态为精确、近似或陈旧匹配时返回 true。
 */
bool IsPublishableTofMatchState(int match_state);

/**
 * @brief 将 SDK RGB 帧转换为独立持有的 `bgr8` ROS 图像。
 * @param[in] frame 待转换的 YUYV 或 NV12 SDK 帧；payload 仅在本次调用期间读取。
 * @param[out] message 接收转换结果，不可为 `nullptr`。
 * @param[out] error 接收失败原因，不可为 `nullptr`。
 * @return 转换成功时返回 true。
 */
bool ConvertRgbFrame(const stereo_camera_frame_t &frame,
                     sensor_msgs::msg::Image *message,
                     std::string *error);

/**
 * @brief 复制 SDK 单通道 16 位图像到 ROS 消息。
 * @param[in] frame 待复制的 SDK 帧；payload 仅在本次调用期间读取。
 * @param[in] encoding 写入 ROS 消息的图像编码。
 * @param[out] message 接收独立 payload 副本，不可为 `nullptr`。
 * @param[out] error 接收失败原因，不可为 `nullptr`。
 * @return payload 尺寸有效并完成复制时返回 true。
 */
bool CopyMono16Frame(const stereo_camera_frame_t &frame,
                     const std::string &encoding,
                     sensor_msgs::msg::Image *message,
                     std::string *error);

/**
 * @brief 复制并解码 SDK IMU 批量 payload，同时返回序列元数据诊断。
 * @param[in] frame SDK 输出的 IMU 帧；payload 在下次 SDK 解析前有效。
 * @param[out] samples 接收独立副本的连续 IMU 样本，不可为 `nullptr`。
 * @param[out] error 接收结构错误或非阻断性序列诊断，不可为 `nullptr`。
 * @return payload 结构有效时返回 true；空批次同样返回 true。
 * @note 序列不一致只写入 `error`，不会令本函数失败。
 */
bool DecodeImuFrame(
    const stereo_camera_frame_t &frame,
    std::vector<stereo_camera_imu_data_t> *samples,
    std::string *error);

/**
 * @class TimestampMapper
 * @brief 将 SDK `CLOCK_MONOTONIC` 时间戳映射到 ROS 时钟。
 * @details 映射锚点在创建时固定；无效或溢出的 SDK 时间戳使用调用方回退时间。
 */
class TimestampMapper {
 public:
  /**
   * @brief 创建固定锚点的时间映射器。
   * @param[in] sdk_anchor_ns SDK 单调时钟锚点，单位为纳秒。
   * @param[in] ros_anchor_ns 与 SDK 锚点对应的 ROS 时钟时间，单位为纳秒。
   */
  TimestampMapper(std::int64_t sdk_anchor_ns, std::int64_t ros_anchor_ns);

  /**
   * @brief 映射无符号 SDK 纳秒时间戳。
   * @param[in] sdk_timestamp_ns SDK `CLOCK_MONOTONIC` 时间戳，0 表示无效。
   * @param[in] fallback 无效或溢出时返回的 ROS 时间。
   * @return 映射后的 ROS 时间或 `fallback`。
   */
  rclcpp::Time Map(std::uint64_t sdk_timestamp_ns,
                   const rclcpp::Time &fallback) const;

  /**
   * @brief 映射 IMU 样本使用的有符号 SDK 纳秒时间戳。
   * @param[in] sdk_timestamp_ns IMU 样本时间戳；非正值表示无效。
   * @param[in] fallback 无效或溢出时返回的 ROS 时间。
   * @return 映射后的 ROS 时间或 `fallback`。
   */
  rclcpp::Time MapImuSample(std::int64_t sdk_timestamp_ns,
                            const rclcpp::Time &fallback) const;

 private:
  std::int64_t sdk_anchor_ns_;  ///< SDK 单调时钟锚点，单位为纳秒。
  std::int64_t ros_anchor_ns_;  ///< 对应的 ROS 时钟锚点，单位为纳秒。
};

}  // namespace tof_stereo_camera

#endif  // TOF_STEREO_CAMERA__FRAME_UTILS_HPP_
