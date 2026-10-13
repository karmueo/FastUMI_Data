/**
 * @file clock_mapping.hpp
 * @brief 把 V4L2 ``CLOCK_MONOTONIC`` 缓冲区时间映射到 ROS（系统）时间，并检测时间跳变。
 *
 * 映射通过成对采样建立：读取 ROS 时间前后各读一次单调时钟，以两次读数的中点作为
 * 对应时刻，区间宽度即本次采样的误差界；多次采样取误差界最小的一次。
 * ``t_ros = t_mono + offset``。不得把单调时钟纳秒值直接当作 ROS Unix 时间，也不得与
 * Tracker 标定时间偏移混用。
 *
 * 周期检查重新采样：误差界超限时保持原偏移并报告；偏移变化不超过 ``max_step_ns``
 * 时跟随（NTP 缓慢校正）；超过时视为系统时间跳变或时间基准切换，重新建立映射并
 * 递增代号 ``generation``，下游据此让当前采集失败，在新映射下开始新 episode。
 * 采样次数、误差门限和跳变门限须在目标机器实测确定。
 */

#ifndef FASTUMI_USB_CAMERA__CLOCK_MAPPING_HPP_
#define FASTUMI_USB_CAMERA__CLOCK_MAPPING_HPP_

#include <time.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace fastumi_usb_camera
{

/** @brief 返回当前时钟纳秒数的读取函数。 */
using ClockReader = std::function<int64_t()>;

/** @return 当前 ``CLOCK_MONOTONIC`` 纳秒数，与 V4L2 单调缓冲区时间同一时钟。 */
inline int64_t monotonic_now_ns()
{
  timespec value{};
  clock_gettime(CLOCK_MONOTONIC, &value);
  return static_cast<int64_t>(value.tv_sec) * 1000000000LL + value.tv_nsec;
}

/** @brief 一次成对采样的结果。 */
struct ClockOffsetSample
{
  int64_t offset_ns{0};  ///< ``t_ros − t_mono_midpoint``。
  int64_t uncertainty_ns{0};  ///< 两次单调时钟读数的区间宽度（误差界）。
};

/**
 * @brief 成对采样 ``attempts`` 次并返回误差界最小的一次。
 * @throws std::invalid_argument ``attempts`` 不为正。
 */
inline ClockOffsetSample sample_clock_offset(
  const ClockReader & monotonic, const ClockReader & ros, int attempts)
{
  if (attempts <= 0) {
    throw std::invalid_argument("clock mapping attempts must be positive");
  }
  ClockOffsetSample best{0, std::numeric_limits<int64_t>::max()};
  for (int index = 0; index < attempts; ++index) {
    const int64_t before = monotonic();
    const int64_t ros_ns = ros();
    const int64_t after = monotonic();
    const int64_t width = after - before;
    if (width < 0) {
      continue;  // 单调时钟不应回退；忽略该次采样。
    }
    if (width < best.uncertainty_ns) {
      best = {ros_ns - (before + width / 2), width};
    }
  }
  if (best.uncertainty_ns == std::numeric_limits<int64_t>::max()) {
    throw std::runtime_error("monotonic clock went backwards during every mapping sample");
  }
  return best;
}

/** @brief 时钟映射门限。 */
struct ClockMappingConfig
{
  int attempts{7};  ///< 每次建立或检查映射的成对采样次数。
  int64_t max_uncertainty_ns{200000};  ///< 可接受的单次映射误差界。
  int64_t max_step_ns{2000000};  ///< 两次检查间可跟随的最大偏移变化，超过视为跳变。
};

/** @brief 周期检查的结论。 */
enum class ClockCheckStatus
{
  kOk,  ///< 偏移在门限内跟随更新。
  kUncertain,  ///< 采样误差界超限，保持原偏移。
  kJump,  ///< 偏移突变：已重新建立映射并递增代号。
};

/** @brief 周期检查结果，供日志与诊断记录。 */
struct ClockCheckResult
{
  ClockCheckStatus status{ClockCheckStatus::kOk};
  int64_t offset_change_ns{0};  ///< 本次采样偏移减去原偏移。
  int64_t uncertainty_ns{0};  ///< 本次采样误差界。
  uint64_t generation{0};  ///< 检查后的映射代号。
};

/** @brief 线程安全的单调时钟→ROS 时间映射；读取无锁，检查由单一线程调用。 */
class MonotonicToRosClock
{
public:
  /** @brief 保存时钟读取函数与门限，尚未建立映射。 */
  MonotonicToRosClock(ClockReader monotonic, ClockReader ros, ClockMappingConfig config = {})
  : monotonic_(std::move(monotonic)), ros_(std::move(ros)), config_(config)
  {
    if (!monotonic_ || !ros_) {
      throw std::invalid_argument("clock readers must not be empty");
    }
    if (config_.max_uncertainty_ns <= 0 || config_.max_step_ns <= 0) {
      throw std::invalid_argument("clock mapping thresholds must be positive");
    }
  }

  /**
   * @brief 建立初始映射。
   * @throws std::runtime_error 采样误差界超过门限。
   */
  void initialize()
  {
    const ClockOffsetSample sample = sample_clock_offset(monotonic_, ros_, config_.attempts);
    if (sample.uncertainty_ns > config_.max_uncertainty_ns) {
      throw std::runtime_error(
              "clock mapping uncertainty " + std::to_string(sample.uncertainty_ns) +
              " ns exceeds " + std::to_string(config_.max_uncertainty_ns) + " ns");
    }
    offset_ns_.store(sample.offset_ns);
    uncertainty_ns_.store(sample.uncertainty_ns);
    generation_.store(1);
  }

  /** @brief 重新采样并按门限更新映射；见文件说明。 */
  ClockCheckResult check()
  {
    if (generation_.load() == 0) {
      throw std::logic_error("clock mapping has not been initialized");
    }
    const ClockOffsetSample sample = sample_clock_offset(monotonic_, ros_, config_.attempts);
    ClockCheckResult result;
    result.uncertainty_ns = sample.uncertainty_ns;
    result.offset_change_ns = sample.offset_ns - offset_ns_.load();
    if (sample.uncertainty_ns > config_.max_uncertainty_ns) {
      result.status = ClockCheckStatus::kUncertain;
    } else if (std::llabs(result.offset_change_ns) > config_.max_step_ns) {
      offset_ns_.store(sample.offset_ns);
      uncertainty_ns_.store(sample.uncertainty_ns);
      generation_.fetch_add(1);
      result.status = ClockCheckStatus::kJump;
    } else {
      offset_ns_.store(sample.offset_ns);
      uncertainty_ns_.store(sample.uncertainty_ns);
    }
    result.generation = generation_.load();
    return result;
  }

  /** @return 单调时钟纳秒对应的 ROS 纳秒时间。 */
  int64_t to_ros_ns(int64_t monotonic_ns) const
  {
    return monotonic_ns + offset_ns_.load();
  }

  int64_t offset_ns() const {return offset_ns_.load();}
  int64_t uncertainty_ns() const {return uncertainty_ns_.load();}
  /** @return 映射代号；0 表示未建立，每次跳变后重建递增。 */
  uint64_t generation() const {return generation_.load();}
  const ClockMappingConfig & config() const {return config_;}

private:
  ClockReader monotonic_;
  ClockReader ros_;
  ClockMappingConfig config_;
  std::atomic<int64_t> offset_ns_{0};
  std::atomic<int64_t> uncertainty_ns_{0};
  std::atomic<uint64_t> generation_{0};
};

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__CLOCK_MAPPING_HPP_
