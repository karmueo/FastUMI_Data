/**
 * @file test_imu.cpp
 * @brief 验证 IMU 协议解析、SI 单位及 ROS 消息有效性标记。
 */
#include "stereo_camera/imu.hpp"
#include <array>
#include <cerrno>
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <system_error>

namespace stereo_camera {
/** @brief 检查大端序号、正负边界、普通组偏移和尾部兼容。 */
TEST(ImuPacket, BigEndianSignedAndTrailingBytes) {
  /** 包含普通组边界值的设备响应，扩展组不参与消息转换。 */
  std::array<std::uint8_t, 64> packet{0x12, 0x34, 0x56, 0x80, 0x00,
                                      0x7f, 0xff, 0xff, 0xff, 0x00,
                                      0x00, 0x12, 0x34, 0xfe, 0xdc};
  packet[15] = 0x55;
  /** 最小有效包解析结果。 */
  const auto sample = ParseImuPacket(packet.data(), 27);
  EXPECT_EQ(sample.video_index, 0x123456U);
  EXPECT_EQ(sample.acceleration,
            (std::array<std::int16_t, 3>{-32768, 32767, -1}));
  EXPECT_EQ(sample.angular_velocity,
            (std::array<std::int16_t, 3>{0, 4660, -292}));
  EXPECT_EQ(ParseImuPacket(packet.data(), packet.size()).acceleration,
            sample.acceleration);
  EXPECT_EQ(ParseImuPacket(packet.data(), packet.size()).angular_velocity,
            sample.angular_velocity);
}
/** @brief 短包和空指针不能解析，避免越界访问。 */
TEST(ImuPacket, RejectInvalidBuffer) {
  /** 足够大的缓冲区，用不同声明长度覆盖边界。 */
  std::array<std::uint8_t, 27> packet{};
  EXPECT_THROW(ParseImuPacket(nullptr, 27), std::invalid_argument);
  EXPECT_THROW(ParseImuPacket(packet.data(), 0), std::invalid_argument);
  EXPECT_THROW(ParseImuPacket(packet.data(), 26), std::invalid_argument);
  EXPECT_NO_THROW(ParseImuPacket(packet.data(), 27));
}
/** @brief 验证重力、角速度单位、轴向、时间戳及未知协方差。 */
TEST(ImuMessage, SiUnitsAndUnavailableOrientation) {
  /** 包含 ±1 g 及角速度满量程边界的原始测量。 */
  ImuSample sample;
  sample.acceleration = {4096, -4096, 0};
  sample.angular_velocity = {-32768, 16384, 32767};
  /** 指定主机接收时间，确保转换不会重新取时间。 */
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 123;
  stamp.nanosec = 456;
  /** 待检查的 SI 单位消息。 */
  const auto message = BuildImuMessage(sample, stamp, "imu_test");
  EXPECT_EQ(message.header.stamp, stamp);
  EXPECT_EQ(message.header.frame_id, "imu_test");
  EXPECT_DOUBLE_EQ(message.linear_acceleration.x, 9.80665);
  EXPECT_DOUBLE_EQ(message.linear_acceleration.y, -9.80665);
  EXPECT_DOUBLE_EQ(message.linear_acceleration.z, 0.0);
  EXPECT_NEAR(message.angular_velocity.x, -2000.0 * std::acos(-1.0) / 180.0,
              1e-12);
  EXPECT_NEAR(message.angular_velocity.y, 1000.0 * std::acos(-1.0) / 180.0,
              1e-12);
  EXPECT_NEAR(message.angular_velocity.z,
              32767.0 / 32768.0 * 2000.0 * std::acos(-1.0) / 180.0, 1e-12);
  EXPECT_EQ(message.orientation_covariance[0], -1.0);
  EXPECT_EQ(message.linear_acceleration_covariance, (std::array<double, 9>{}));
  EXPECT_EQ(message.angular_velocity_covariance, (std::array<double, 9>{}));
}
/** @brief 无法打开控制设备时传播系统错误，不创建可读取实例。 */
TEST(UvcImu, RejectUnopenableDevice) {
  EXPECT_THROW(UvcImu("/dev/null/not_a_device"), std::system_error);
}
/** @brief 不支持长度查询时回退；失败的 GET_CUR 不得发布零填充假数据。 */
TEST(UvcImu, UnsupportedControlReadThrows) {
  /** Linux 空设备可打开但不支持 UVC 控制，用于检查异常路径。 */
  UvcImu imu("/dev/null");
  try {
    imu.Read();
    FAIL() << "GET_CUR 应失败";
  } catch (const std::system_error &error) {
    EXPECT_EQ(error.code().value(), ENOTTY);
  }
}
} // namespace stereo_camera
