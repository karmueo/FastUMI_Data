/**
 * @file xu_config.hpp
 * @brief 声明 stereo_camera SDK XU 启动配置的校验与下发接口。
 * @author 待确认
 * @date 创建：2026-08-20
 */

#ifndef TOF_STEREO_CAMERA__XU_CONFIG_HPP_
#define TOF_STEREO_CAMERA__XU_CONFIG_HPP_

#include <cstdint>
#include <string>

#include "stereo_camera/stereo_camera.h"

namespace tof_stereo_camera {

/** @brief SDK XU 命令函数签名，用于真机调用和单元测试替身。 */
using XuCommandFunction = int (*)(stereo_camera_t *, stereo_camera_xu_cmd_t,
                                  const void *, size_t, int *);

/** @brief 保存节点启动前需要下发的全部 SDK XU 配置。 */
struct XuConfiguration {
  std::uint32_t accel_hz = 100U;  ///< 加速度计采样频率，单位 Hz。
  std::uint32_t gyro_hz = 100U;   ///< 陀螺仪采样频率，单位 Hz。
  std::uint32_t stream_mask = 0U; ///< 设备端视频流 ID 位掩码。
};

/**
 * @brief 校验频率参数并构造 SDK XU 配置。
 * @param[in] enable_rgb 是否启用 RGB 设备流。
 * @param[in] enable_itof_depth 是否启用 iTOF 深度设备流。
 * @param[in] enable_itof_gray 是否启用 iTOF 灰度设备流。
 * @param[in] enable_imu 是否启用 ROS IMU 输出。
 * @param[in] accel_hz 加速度计采样频率，单位 Hz。
 * @param[in] gyro_hz 陀螺仪采样频率，单位 Hz。
 * @param[out] configuration 接收构造结果，不可为 `nullptr`。
 * @param[out] error 接收失败原因，不可为 `nullptr`。
 * @return 所有参数有效时返回 true。
 */
bool BuildXuConfiguration(bool enable_rgb, bool enable_itof_depth,
                          bool enable_itof_gray, bool enable_imu, int accel_hz,
                          int gyro_hz, XuConfiguration *configuration,
                          std::string *error);

/**
 * @brief 按固定顺序向已打开的相机下发视频流掩码和 IMU 频率。
 * @param[in] camera 已打开且尚未开始推流的相机句柄。
 * @param[in] configuration 已校验的 XU 配置。
 * @param[out] error 接收包含命令返回值和 ACK 的失败原因，不可为 `nullptr`。
 * @param[in] command SDK XU 命令函数；测试可传入替身。
 * @return 两条命令均成功且 ACK 为 0 时返回 true。
 * @note 第一条失败后不会继续下发第二条命令。
 */
bool ApplyXuConfiguration(
    stereo_camera_t *camera, const XuConfiguration &configuration,
    std::string *error, XuCommandFunction command = &stereo_camera_xu_command);

} // namespace tof_stereo_camera

#endif // TOF_STEREO_CAMERA__XU_CONFIG_HPP_
