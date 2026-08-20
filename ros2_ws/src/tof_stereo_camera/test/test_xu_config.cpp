/**
 * @file test_xu_config.cpp
 * @brief stereo_camera SDK XU 启动配置的无硬件回归测试。
 * @author 待确认
 * @date 创建：2026-08-20
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tof_stereo_camera/xu_config.hpp"

namespace tof_stereo_camera {
namespace {

/** @brief 保存伪 XU 调用接收到的一条命令。 */
struct CommandRecord {
  stereo_camera_xu_cmd_t command{};     ///< SDK XU 命令枚举。
  std::vector<unsigned char> parameter; ///< 命令参数的独立字节副本。
};

std::vector<CommandRecord> g_command_records; ///< 当前测试记录的命令列表。
int g_failed_call = -1; ///< 需要模拟失败的调用序号；-1 表示全部成功。

/**
 * @brief 记录 XU 参数并按测试设置返回成功或设备拒绝。
 * @param[in] camera 伪相机句柄。
 * @param[in] command 命令枚举。
 * @param[in] parameter 命令参数地址。
 * @param[in] parameter_size 命令参数字节数。
 * @param[out] ack 接收模拟 ACK。
 * @return 正常调用返回 0，指定失败调用返回 -1。
 */
int FakeXuCommand(stereo_camera_t *camera, stereo_camera_xu_cmd_t command,
                  const void *parameter, std::size_t parameter_size, int *ack) {
  if (camera == nullptr || parameter == nullptr || ack == nullptr) {
    return -1;
  }
  CommandRecord record; ///< 保存本次调用的命令和参数副本。
  record.command = command;
  const auto *bytes = static_cast<const unsigned char *>(parameter);
  record.parameter.assign(bytes, bytes + parameter_size);
  g_command_records.push_back(std::move(record));
  const int call_index = static_cast<int>(g_command_records.size()) - 1;
  if (call_index == g_failed_call) {
    *ack = 2;
    return -1;
  }
  *ack = 0;
  return 0;
}

/**
 * @brief 从伪调用记录中还原一个平凡可复制的参数结构体。
 * @tparam Parameter SDK 参数结构体类型。
 * @param[in] record 保存参数字节的命令记录。
 * @return 从记录字节复制出的结构体。
 */
template <typename Parameter>
Parameter DecodeParameter(const CommandRecord &record) {
  Parameter parameter{}; ///< 接收测试命令参数的结构体。
  EXPECT_EQ(record.parameter.size(), sizeof(parameter));
  if (record.parameter.size() == sizeof(parameter)) {
    std::memcpy(&parameter, record.parameter.data(), sizeof(parameter));
  }
  return parameter;
}

/** @brief 每个测试前清空伪 SDK 调用状态。 */
class XuConfigTest : public ::testing::Test {
protected:
  /** @brief 重置命令记录和失败注入位置。 */
  void SetUp() override {
    g_command_records.clear();
    g_failed_call = -1;
  }
};

/** @brief 验证三个视频开关的全部组合均映射到固定流 ID 位。 */
TEST_F(XuConfigTest, BuildsEveryVideoStreamMaskCombination) {
  for (unsigned int combination = 0U; combination < 8U; ++combination) {
    const bool rgb_enabled = (combination & 0x1U) != 0U;
    const bool depth_enabled = (combination & 0x2U) != 0U;
    const bool gray_enabled = (combination & 0x4U) != 0U;
    const std::uint32_t expected_mask = (rgb_enabled ? 0x01U : 0U) |
                                        (depth_enabled ? 0x04U : 0U) |
                                        (gray_enabled ? 0x80U : 0U);
    XuConfiguration configuration; ///< 接收当前开关组合的 XU 配置。
    std::string error;             ///< 接收意外校验错误。
    ASSERT_TRUE(BuildXuConfiguration(rgb_enabled, depth_enabled, gray_enabled,
                                     false, 100, 100, &configuration, &error))
        << error;
    EXPECT_EQ(configuration.stream_mask, expected_mask);
  }
}

/** @brief 验证默认全流组合生成协议规定的 0x85。 */
TEST_F(XuConfigTest, BuildsDefaultFullStreamMask) {
  XuConfiguration configuration; ///< 接收默认 XU 配置。
  std::string error;             ///< 接收意外校验错误。
  ASSERT_TRUE(BuildXuConfiguration(true, true, true, true, 100, 100,
                                   &configuration, &error));
  EXPECT_EQ(configuration.stream_mask, 0x85U);
}

/** @brief 验证 SDK 文档列出的全部 IMU 频率均被接受。 */
TEST_F(XuConfigTest, AcceptsDocumentedImuFrequencies) {
  const std::vector<int> accel_frequencies{12,  25,  50,  100,
                                           200, 400, 800, 1600};
  const std::vector<int> gyro_frequencies{25,  50,  100,  200,
                                          400, 800, 1600, 3200};
  for (const int accel_hz : accel_frequencies) {
    XuConfiguration configuration; ///< 接收合法加速度计频率配置。
    std::string error;             ///< 接收意外校验错误。
    EXPECT_TRUE(BuildXuConfiguration(true, true, true, true, accel_hz, 100,
                                     &configuration, &error));
  }
  for (const int gyro_hz : gyro_frequencies) {
    XuConfiguration configuration; ///< 接收合法陀螺仪频率配置。
    std::string error;             ///< 接收意外校验错误。
    EXPECT_TRUE(BuildXuConfiguration(true, true, true, true, 100, gyro_hz,
                                     &configuration, &error));
  }
}

/** @brief 验证非法、负数和超范围 IMU 频率在调用 SDK 前被拒绝。 */
TEST_F(XuConfigTest, RejectsUnsupportedImuFrequencies) {
  const std::vector<int> invalid_frequencies{-1, 0, 13, 3201};
  for (const int frequency : invalid_frequencies) {
    XuConfiguration configuration; ///< 接收不应生成的 XU 配置。
    std::string error;             ///< 接收频率校验诊断。
    EXPECT_FALSE(BuildXuConfiguration(true, true, true, true, frequency, 100,
                                      &configuration, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_FALSE(BuildXuConfiguration(true, true, true, true, 100, frequency,
                                      &configuration, &error));
    EXPECT_FALSE(error.empty());
  }
}

/** @brief 验证当前固件不允许 IMU 作为唯一启用的数据流。 */
TEST_F(XuConfigTest, RejectsImuWithoutVideoCarrier) {
  XuConfiguration configuration; ///< 接收不应生成的 XU 配置。
  std::string error;             ///< 接收视频载体约束诊断。

  EXPECT_FALSE(BuildXuConfiguration(false, false, false, true, 100, 100,
                                    &configuration, &error));
  EXPECT_NE(error.find("at least one enabled video stream"), std::string::npos);
}

/** @brief 验证两条 XU 命令的顺序、类型、尺寸和字段值。 */
TEST_F(XuConfigTest, AppliesTypedCommandsInOrder) {
  XuConfiguration configuration; ///< 待下发的测试配置。
  configuration.accel_hz = 200U;
  configuration.gyro_hz = 400U;
  configuration.stream_mask = 0x05U;
  std::string error; ///< 接收意外下发错误。
  auto *camera = reinterpret_cast<stereo_camera_t *>(
      std::uintptr_t{1}); ///< 非空且不会被伪命令解引用的句柄。

  ASSERT_TRUE(
      ApplyXuConfiguration(camera, configuration, &error, &FakeXuCommand))
      << error;
  ASSERT_EQ(g_command_records.size(), 2U);
  EXPECT_EQ(g_command_records[0].command, STEREO_CAMERA_XU_CMD_STREAM_MASK);
  const auto stream_mask =
      DecodeParameter<stereo_camera_xu_stream_mask_param_t>(
          g_command_records[0]);
  EXPECT_EQ(stream_mask.stream_mask, 0x05U);
  EXPECT_EQ(g_command_records[1].command, STEREO_CAMERA_XU_CMD_IMU_FREQ);
  const auto imu_frequency =
      DecodeParameter<stereo_camera_xu_imu_freq_param_t>(g_command_records[1]);
  EXPECT_EQ(imu_frequency.accel_hz, 200U);
  EXPECT_EQ(imu_frequency.gyro_hz, 400U);
}

/** @brief 验证第二条命令失败会返回 ACK 诊断并阻止启动流程继续。 */
TEST_F(XuConfigTest, ReportsSecondCommandFailure) {
  g_failed_call = 1;
  const XuConfiguration configuration{}; ///< 待下发的默认测试配置。
  std::string error;                     ///< 接收设备拒绝诊断。
  auto *camera = reinterpret_cast<stereo_camera_t *>(
      std::uintptr_t{1}); ///< 非空且不会被伪命令解引用的句柄。

  EXPECT_FALSE(
      ApplyXuConfiguration(camera, configuration, &error, &FakeXuCommand));
  EXPECT_EQ(g_command_records.size(), 2U);
  EXPECT_NE(error.find("imu_freq"), std::string::npos);
  EXPECT_NE(error.find("accel_hz=100"), std::string::npos);
  EXPECT_NE(error.find("gyro_hz=100"), std::string::npos);
  EXPECT_NE(error.find("result=-1"), std::string::npos);
  EXPECT_NE(error.find("ack=2"), std::string::npos);
}

/** @brief 验证首条命令失败后不会下发 IMU 频率。 */
TEST_F(XuConfigTest, StopsAfterFirstCommandFailure) {
  g_failed_call = 0;
  const XuConfiguration configuration{}; ///< 待下发的默认测试配置。
  std::string error;                     ///< 接收设备拒绝诊断。
  auto *camera = reinterpret_cast<stereo_camera_t *>(
      std::uintptr_t{1}); ///< 非空且不会被伪命令解引用的句柄。

  EXPECT_FALSE(
      ApplyXuConfiguration(camera, configuration, &error, &FakeXuCommand));
  ASSERT_EQ(g_command_records.size(), 1U);
  EXPECT_EQ(g_command_records[0].command, STEREO_CAMERA_XU_CMD_STREAM_MASK);
  EXPECT_NE(error.find("stream_mask=0x0"), std::string::npos);
}

} // namespace
} // namespace tof_stereo_camera
