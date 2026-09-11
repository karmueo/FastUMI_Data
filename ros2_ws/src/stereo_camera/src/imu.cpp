/**
 * @file imu.cpp
 * @brief 实现 Ego IMU 的 UVC 查询、大端解析和标准单位转换。
 */
#include "stereo_camera/imu.hpp"

#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <linux/usb/video.h>
#include <linux/uvcvideo.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <system_error>
#include <unistd.h>

namespace stereo_camera {
namespace {
/**
 * @brief 查询固定的 IMU 扩展控制，遇到信号中断时重试。
 * @param[in] descriptor 设备描述符。
 * @param[in] request UVC 请求类型。
 * @param[out] data 响应缓冲区。
 * @param[in] size 缓冲区字节数。
 * @return ioctl 返回值，失败时保留 errno。
 */
int Query(int descriptor, std::uint8_t request, std::uint8_t *data,
          std::uint16_t size) {
  /** 保存设备固定扩展单元及控制选择器。 */
  uvc_xu_control_query query{};
  query.unit = 3;
  query.selector = 1;
  query.query = request;
  query.size = size;
  query.data = data;
  /** 保存系统调用结果。 */
  int result;
  do {
    result = ioctl(descriptor, UVCIOC_CTRL_QUERY, &query);
  } while (result < 0 && errno == EINTR);
  return result;
}
/**
 * @brief 从两个大端字节解码有符号整数，避免越界有符号转换。
 * @param[in] data 两个可读字节。
 * @return 解码后的 16 位有符号值。
 */
std::int16_t ReadSigned(const std::uint8_t *data) {
  /** 保存未符号扩展的 ADC 位模式。 */
  const int value = (static_cast<int>(data[0]) << 8) | data[1];
  return static_cast<std::int16_t>(value >= 32768 ? value - 65536 : value);
}
} // namespace

/** @brief 解析普通 IMU；参数、返回值及异常见头文件。 */
ImuSample ParseImuPacket(const std::uint8_t *data, std::size_t size) {
  if (data == nullptr || size < 27) {
    throw std::invalid_argument("IMU 数据包必须至少包含 27 字节");
  }
  /** 保存视频序号及普通三轴测量。 */
  ImuSample sample;
  sample.video_index = (static_cast<std::uint32_t>(data[0]) << 16) |
                       (static_cast<std::uint32_t>(data[1]) << 8) | data[2];
  /** 遍历传感器的三个原始轴。 */
  for (std::size_t axis = 0; axis < 3; ++axis) {
    sample.acceleration[axis] = ReadSigned(data + 3 + axis * 2);
    sample.angular_velocity[axis] = ReadSigned(data + 9 + axis * 2);
  }
  return sample;
}

/** @brief 构造未校正的 SI 消息；参数及返回值见头文件。 */
sensor_msgs::msg::Imu
BuildImuMessage(const ImuSample &sample,
                const builtin_interfaces::msg::Time &stamp,
                const std::string &frame_id) {
  /** 加速度 ADC 到 m/s² 的换算比例。 */
  constexpr double acceleration_scale = 8.0 / 32768.0 * 9.80665;
  /** 角速度 ADC 到 rad/s 的换算比例。 */
  const double gyro_scale = 2000.0 / 32768.0 * std::acos(-1.0) / 180.0;
  /** 保存标准消息，默认零协方差表示未知。 */
  sensor_msgs::msg::Imu message;
  message.header.stamp = stamp;
  message.header.frame_id = frame_id;
  message.orientation.w = 1.0;
  message.orientation_covariance[0] = -1.0;
  message.linear_acceleration.x = sample.acceleration[0] * acceleration_scale;
  message.linear_acceleration.y = sample.acceleration[1] * acceleration_scale;
  message.linear_acceleration.z = sample.acceleration[2] * acceleration_scale;
  message.angular_velocity.x = sample.angular_velocity[0] * gyro_scale;
  message.angular_velocity.y = sample.angular_velocity[1] * gyro_scale;
  message.angular_velocity.z = sample.angular_velocity[2] * gyro_scale;
  return message;
}

/** @brief 打开独立控制描述符并获取长度；失败时释放已取得资源。 */
UvcImu::UvcImu(const std::string &device_path) {
  descriptor_ = open(device_path.c_str(), O_RDWR | O_CLOEXEC);
  if (descriptor_ < 0) {
    throw std::system_error(errno, std::generic_category(), "打开 IMU 设备");
  }
  try {
    /** UVC GET_LEN 返回小端长度，与测量数据的大端格式不同。 */
    std::uint8_t length_bytes[2]{};
    /** 固件不支持长度查询时使用参考程序的 64 字节默认值。 */
    std::size_t length = 64;
    if (Query(descriptor_, UVC_GET_LEN, length_bytes, 2) == 0) {
      length =
          length_bytes[0] | (static_cast<std::size_t>(length_bytes[1]) << 8);
    }
    if (length < 27) {
      throw std::runtime_error("IMU 控制响应长度不足 27 字节");
    }
    buffer_.resize(length);
  } catch (...) {
    close(descriptor_);
    descriptor_ = -1;
    throw;
  }
}
/** @brief 释放设备描述符。 */
UvcImu::~UvcImu() {
  if (descriptor_ >= 0) {
    close(descriptor_);
  }
}
/** @brief 读取并解析一次控制响应；失败时抛出包含 errno 的异常。 */
ImuSample UvcImu::Read() {
  if (Query(descriptor_, UVC_GET_CUR, buffer_.data(),
            static_cast<std::uint16_t>(buffer_.size())) < 0) {
    throw std::system_error(errno, std::generic_category(), "读取 IMU GET_CUR");
  }
  return ParseImuPacket(buffer_.data(), buffer_.size());
}
} // namespace stereo_camera
