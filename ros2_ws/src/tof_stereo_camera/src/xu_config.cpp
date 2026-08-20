/**
 * @file xu_config.cpp
 * @brief 实现 stereo_camera SDK XU 启动配置的校验、掩码生成和命令下发。
 * @author 待确认
 * @date 创建：2026-08-20
 */

#include "tof_stereo_camera/xu_config.hpp"

#include <array>
#include <cstddef>
#include <sstream>

#include "tof_stereo_camera/stream_ids.hpp"

namespace tof_stereo_camera {
namespace {

/** @brief SDK 支持的加速度计采样频率集合。 */
constexpr std::array<int, 8> kSupportedAccelFrequencies{12,  25,  50,  100,
                                                        200, 400, 800, 1600};
/** @brief SDK 支持的陀螺仪采样频率集合。 */
constexpr std::array<int, 8> kSupportedGyroFrequencies{25,  50,  100,  200,
                                                       400, 800, 1600, 3200};

/**
 * @brief 判断整数是否位于给定的固定集合中。
 * @tparam Size 集合元素数量。
 * @param[in] value 待检查的整数。
 * @param[in] supported 合法整数集合。
 * @return 找到相同整数时返回 true。
 */
template <std::size_t Size>
bool Contains(int value, const std::array<int, Size> &supported) {
  for (const int candidate : supported) {
    if (candidate == value) {
      return true;
    }
  }
  return false;
}

/**
 * @brief 为一个启用的视频流设置对应掩码位。
 * @param[in,out] mask 接收掩码位的变量。
 * @param[in] enabled 是否启用该流。
 * @param[in] stream_id SDK 视频流 ID。
 */
void AddStreamMaskBit(std::uint32_t *mask, bool enabled, int stream_id) {
  if (enabled) {
    *mask |= std::uint32_t{1} << static_cast<unsigned int>(stream_id);
  }
}

/**
 * @brief 调用一条 SDK XU 命令并生成包含 ACK 的诊断信息。
 * @param[in] camera 相机句柄。
 * @param[in] command SDK 命令函数。
 * @param[in] command_id XU 命令枚举。
 * @param[in] command_name 用于诊断的命令名称。
 * @param[in] parameter 命令参数地址。
 * @param[in] parameter_size 命令参数字节数。
 * @param[out] error 接收失败原因。
 * @return SDK 返回成功且 ACK 为 0 时返回 true。
 */
bool ExecuteCommand(stereo_camera_t *camera, XuCommandFunction command,
                    stereo_camera_xu_cmd_t command_id, const char *command_name,
                    const void *parameter, std::size_t parameter_size,
                    std::string *error) {
  int ack = -1; ///< 设备 ACK；-1 表示 SDK 未返回有效 ACK。
  const int result =
      command(camera, command_id, parameter, parameter_size, &ack);
  if (result == 0 && ack == 0) {
    return true;
  }
  std::ostringstream diagnostic; ///< 汇总 SDK 返回值和设备 ACK 的诊断文本。
  diagnostic << "XU command " << command_name << " failed: result=" << result
             << ", ack=" << ack;
  *error = diagnostic.str();
  return false;
}

} // namespace

/** @copydoc BuildXuConfiguration */
bool BuildXuConfiguration(bool enable_rgb, bool enable_itof_depth,
                          bool enable_itof_gray, bool enable_imu, int accel_hz,
                          int gyro_hz, XuConfiguration *configuration,
                          std::string *error) {
  if (configuration == nullptr || error == nullptr) {
    return false;
  }
  error->clear();
  if (!Contains(accel_hz, kSupportedAccelFrequencies)) {
    *error = "imu_accel_hz must be one of 12, 25, 50, 100, 200, 400, 800, "
             "1600";
    return false;
  }
  if (!Contains(gyro_hz, kSupportedGyroFrequencies)) {
    *error = "imu_gyro_hz must be one of 25, 50, 100, 200, 400, 800, 1600, "
             "3200";
    return false;
  }

  std::uint32_t stream_mask = 0U; ///< 按启用开关生成的视频流 ID 掩码。
  AddStreamMaskBit(&stream_mask, enable_rgb, kRgbStreamId);
  AddStreamMaskBit(&stream_mask, enable_itof_depth, kItofDepthStreamId);
  AddStreamMaskBit(&stream_mask, enable_itof_gray, kItofGrayStreamId);
  if (enable_imu && stream_mask == 0U) {
    *error = "enable_imu=true requires at least one enabled video stream "
             "because the current firmware carries IMU data in composite "
             "video frames";
    return false;
  }
  configuration->accel_hz = static_cast<std::uint32_t>(accel_hz);
  configuration->gyro_hz = static_cast<std::uint32_t>(gyro_hz);
  configuration->stream_mask = stream_mask;
  return true;
}

/** @copydoc ApplyXuConfiguration */
bool ApplyXuConfiguration(stereo_camera_t *camera,
                          const XuConfiguration &configuration,
                          std::string *error, XuCommandFunction command) {
  if (error == nullptr) {
    return false;
  }
  error->clear();
  if (camera == nullptr) {
    *error = "cannot apply XU configuration to a null camera";
    return false;
  }
  if (command == nullptr) {
    *error = "cannot apply XU configuration with a null command function";
    return false;
  }

  const stereo_camera_xu_stream_mask_param_t stream_mask{
      configuration.stream_mask};
  if (!ExecuteCommand(camera, command, STEREO_CAMERA_XU_CMD_STREAM_MASK,
                      "stream_mask", &stream_mask, sizeof(stream_mask),
                      error)) {
    std::ostringstream diagnostic; ///< 补充失败命令请求的视频流掩码。
    diagnostic << *error << ", stream_mask=0x" << std::hex
               << configuration.stream_mask;
    *error = diagnostic.str();
    return false;
  }

  const stereo_camera_xu_imu_freq_param_t imu_frequency{configuration.accel_hz,
                                                        configuration.gyro_hz};
  if (ExecuteCommand(camera, command, STEREO_CAMERA_XU_CMD_IMU_FREQ, "imu_freq",
                     &imu_frequency, sizeof(imu_frequency), error)) {
    return true;
  }
  std::ostringstream diagnostic; ///< 补充失败命令请求的两个 IMU 频率。
  diagnostic << *error << ", accel_hz=" << configuration.accel_hz
             << ", gyro_hz=" << configuration.gyro_hz;
  *error = diagnostic.str();
  return false;
}

} // namespace tof_stereo_camera
