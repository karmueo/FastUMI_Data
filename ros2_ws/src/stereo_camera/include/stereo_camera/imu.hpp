/**
 * @file imu.hpp
 * @brief 声明 Ego IMU 数据包解析、SI 消息转换及 UVC 读取接口。
 */
#pragma once

#include "sensor_msgs/msg/imu.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace stereo_camera {
/** @brief 普通 IMU 的原始测量，保留传感器轴向。 */
struct ImuSample {
  std::uint32_t video_index = 0; ///< 视频帧序号，不是 IMU 样本序号。
  std::array<std::int16_t, 3> acceleration{}; ///< 三轴加速度 ADC 值。
  std::array<std::int16_t, 3> angular_velocity{}; ///< 三轴角速度 ADC 值。
};
/**
 * @brief 解析大端数据包，忽略扩展 IMU 和尾部。
 * @param[in] data 数据包首地址。
 * @param[in] size 数据包字节数，至少为 27。
 * @return 普通 IMU 原始测量。
 * @throws std::invalid_argument 指针为空或长度不足。
 */
ImuSample ParseImuPacket(const std::uint8_t *data, std::size_t size);
/**
 * @brief 按 ±8 g、±2000 dps 换算为 SI 消息，不应用偏置或姿态估计。
 * @param[in] sample 原始测量。
 * @param[in] stamp 主机读取完成时的系统时间。
 * @param[in] frame_id 原始传感器坐标系。
 * @return 加速度为 m/s²、角速度为 rad/s 的消息，协方差未知。
 */
sensor_msgs::msg::Imu
BuildImuMessage(const ImuSample &sample,
                const builtin_interfaces::msg::Time &stamp,
                const std::string &frame_id);
/** @brief 独占 UVC 控制描述符；必须在视频流启动后创建，仅由单一线程读取。 */
class UvcImu {
public:
  /**
   * @brief 打开设备并查询控制长度，不修改视频流配置。
   * @param[in] device_path 相机设备路径。
   * @throws std::system_error 打开失败。
   * @throws std::runtime_error 设备报告的数据长度不足。
   */
  explicit UvcImu(const std::string &device_path);
  /** @brief 关闭独占的控制描述符。 */
  ~UvcImu();
  UvcImu(const UvcImu &) = delete; ///< 禁止复制描述符所有权。
  UvcImu &operator=(const UvcImu &) = delete; ///< 禁止复制赋值。
  /**
   * @brief 执行一次 GET_CUR 并解析普通 IMU。
   * @return 当前控制响应中的测量，不保证是新样本。
   * @throws std::system_error ioctl 失败，错误码供调用方判断断开。
   */
  ImuSample Read();

private:
  int descriptor_ = -1;              ///< 独占设备描述符。
  std::vector<std::uint8_t> buffer_; ///< 设备控制响应缓冲区。
};
} // namespace stereo_camera
