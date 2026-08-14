/**
 * @file frame_utils.hpp
 * @brief 声明 ROS 2 图像与 IMU 帧转换及时间映射工具。
 * @author 待确认
 * @date 创建：待确认
 * @date 修改：2026-08-14
 */

#ifndef TOF_STEREO_CAMERA__FRAME_UTILS_HPP_
#define TOF_STEREO_CAMERA__FRAME_UTILS_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "stereo_camera/stereo_camera.h"

namespace tof_stereo_camera {

constexpr int kItofDepthStreamId = 2; ///< iTOF 深度图在 SDK 中使用的流 ID。
constexpr int kItofGrayStreamId = 7; ///< iTOF 灰度图在 SDK 中使用的流 ID。

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
                     sensor_msgs::msg::Image *message, std::string *error);

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
                     sensor_msgs::msg::Image *message, std::string *error);

/**
 * @brief 复制并解码 SDK IMU 批量 payload，同时返回序列元数据诊断。
 * @param[in] frame SDK 输出的 IMU 帧；payload 在下次 SDK 解析前有效。
 * @param[out] samples 接收独立副本的连续 IMU 样本，不可为 `nullptr`。
 * @param[out] error 接收结构错误或非阻断性序列诊断，不可为 `nullptr`。
 * @return payload 结构有效时返回 true；空批次同样返回 true。
 * @note 序列不一致只写入 `error`，不会令本函数失败。
 */
bool DecodeImuFrame(const stereo_camera_frame_t &frame,
                    std::vector<stereo_camera_imu_data_t> *samples,
                    std::string *error);

/** @brief 描述外层复合帧的同步处理状态。 */
enum class FrameTimestampStatus {
  kCalibrating,         ///< 正在采集初始标定观测。
  kReady,               ///< 新的唯一时间戳已准备好发布。
  kInvalidHostFallback, ///< 无效或溢出时间戳使用主机接收时间。
  kReset,               ///< 设备时钟 epoch 不连续，已重新标定。
  kDropped,   ///< 单调性下界与接收上界冲突，当前帧已丢弃。
  kDuplicate, ///< 连续重复时间戳复用前一外层帧快照。
};

/** @brief 保存外层复合帧的不可变同步快照。 */
struct FrameTimestampResult {
  FrameTimestampStatus status =
      FrameTimestampStatus::kCalibrating; ///< 处理状态。
  std::int64_t steady_ns = 0; ///< 映射的主机稳态时间，单位纳秒。
  std::int64_t system_ns = 0; ///< 映射的 Unix 系统时间，单位纳秒。
  std::int64_t offset_ns = 0; ///< 该帧 SDK 到稳态时钟的偏移。
  bool ready = false;         ///< 有效 SDK 时间戳是否允许发布。
  bool has_offset_snapshot = false; ///< `offset_ns` 是否有效。
  bool newly_locked = false;        ///< 本帧是否刚完成 epoch 锁定。
  bool receive_clamped = false; ///< 映射时间晚于接收时间而被钳制。
};

/** @brief 描述 IMU 样本映射结果。 */
struct ImuTimestampResult {
  std::int64_t steady_ns = 0; ///< 映射的主机稳态时间，单位纳秒。
  std::int64_t system_ns = 0; ///< 映射的 Unix 系统时间，单位纳秒。
  bool valid_sample = false;  ///< 样本时间戳和外层快照有效。
  bool future_clamped = false; ///< 样本晚于外层帧而被钳制。
};

/** @brief 以有限斜率将 SDK 微秒设备时钟同步到主机稳态及系统时钟。 */
class TimestampMapper {
public:
  /** @brief 从配对的稳态/系统时钟锚点及同步参数构造同步器。 */
  TimestampMapper(std::int64_t steady_anchor_ns, std::int64_t system_anchor_ns,
                  std::size_t calibration_frames, std::size_t window_frames,
                  double max_slew_ppm);
  /** @brief 观察并映射一个外层复合帧。 */
  FrameTimestampResult ObserveFrame(std::uint64_t sdk_timestamp_us,
                                    std::int64_t receive_steady_ns);
  /** @brief 用所属外层帧不可变快照映射一个 IMU 样本。 */
  ImuTimestampResult
  MapImuSample(std::int64_t sdk_timestamp_us,
               const FrameTimestampResult &frame_result) const;
  /** @brief 返回当前 epoch 是否完成初始锁定。 */
  bool ready() const;
  /** @brief 返回最近锁定的 SDK 到主机稳态时钟偏移，单位纳秒。 */
  std::int64_t detected_epoch_offset_ns() const;
  /** @brief 返回最近锁定的 SDK 到 Unix 系统时钟偏移，单位纳秒。 */
  std::int64_t detected_system_epoch_offset_ns() const;

private:
  /** @brief 以当前观测开始新的设备时钟 epoch。 */
  void StartEpoch(std::int64_t sdk_ns, std::int64_t receive_steady_ns,
                  std::int64_t candidate_offset_ns);
  /** @brief 添加候选偏移并保持滚动窗口大小。 */
  void PushCandidate(std::int64_t candidate_offset_ns);
  /** @brief 返回候选窗口中的最小偏移。 */
  std::int64_t MinimumCandidate() const;
  /** @brief 用固定配对锚点把稳态时间转换为 Unix 系统时间。 */
  std::int64_t SystemFromSteady(std::int64_t steady_ns) const;
  /** @brief 以主机接收时间回退，并维护全生命周期的发布单调下界。 */
  FrameTimestampResult HostFallback(std::int64_t receive_steady_ns);
  /** @brief 丢弃冲突输出后清空当前 epoch 的标定状态。 */
  void RestartCalibration();
  /** @brief 返回缓存快照的重复帧结果。 */
  FrameTimestampResult DuplicateResult() const;

  std::int64_t steady_anchor_ns_;  ///< 主机稳态锚点。
  std::int64_t system_anchor_ns_;  ///< Unix 系统时钟锚点。
  std::size_t calibration_frames_; ///< 初始锁定所需唯一观测数。
  std::size_t window_frames_;      ///< 滚动最小偏移窗口大小。
  double max_slew_ppm_;            ///< 偏移修正上限，单位 ppm。
  std::deque<std::int64_t> candidates_; ///< 最近唯一观测的候选偏移。
  bool ready_ = false;                  ///< 当前 epoch 是否已锁定。
  bool have_previous_unique_ = false; ///< 是否已有唯一有效外层帧。
  std::int64_t previous_sdk_ns_ = 0;  ///< 上一个唯一 SDK 时间戳。
  std::int64_t previous_receive_ns_ = 0; ///< 上一个唯一接收稳态时间。
  std::int64_t applied_offset_ns_ = 0;   ///< 当前 SDK 到稳态偏移。
  bool have_previous_ready_ = false; ///< 是否已有全生命周期内可发布外层帧。
  std::int64_t previous_ready_steady_ns_ =
      0; ///< 全生命周期上个可发布外层时间。
  bool have_cached_result_ = false; ///< 是否缓存最近调用结果。
  std::uint64_t cached_sdk_timestamp_us_ = 0; ///< 缓存对应原始 SDK 时间戳。
  FrameTimestampResult cached_result_;        ///< 最近外层帧结果。
};

} // namespace tof_stereo_camera

#endif // TOF_STEREO_CAMERA__FRAME_UTILS_HPP_
