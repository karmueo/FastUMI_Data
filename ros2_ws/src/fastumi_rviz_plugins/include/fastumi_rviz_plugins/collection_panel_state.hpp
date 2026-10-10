/**
 * @file collection_panel_state.hpp
 * @brief 定义不依赖 Qt/ROS 的采集 Panel 门控、请求跟踪、分页和文本格式化逻辑。
 */

#ifndef FASTUMI_RVIZ_PLUGINS__COLLECTION_PANEL_STATE_HPP_
#define FASTUMI_RVIZ_PLUGINS__COLLECTION_PANEL_STATE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace fastumi_rviz_plugins
{

/** @brief 超过该毫秒数未收到状态即视为与后端断开。 */
constexpr int64_t kCollectionStatusTimeoutMs = 3000;
/** @brief 记录列表每页条数。 */
constexpr uint32_t kCollectionPageSize = 10;
/** @brief 任务名最大字节数的保守上限；后端按字符数做权威校验。 */
constexpr std::size_t kMaxTaskNameBytes = 256;

/** @brief 后端按当前状态计算的按钮可用性，面板不自行推断生命周期。 */
struct BackendFlags
{
  bool can_start{false};
  bool can_stop{false};
  bool can_save{false};
  bool can_cancel{false};
  bool can_stop_and_save{false};
  bool can_stop_and_cancel{false};
};

/** @brief 面板各按钮最终是否可点击。 */
struct ButtonGate
{
  bool start{false};
  bool stop_and_save{false};
  bool stop_and_cancel{false};
  bool save{false};
  bool cancel{false};
};

/**
 * @brief 判断任务名在客户端侧是否可提交；后端仍会做权威校验。
 * @param task_name 任务名输入框内容。
 * @return 去除空白后非空，且不含路径分隔符、控制字符，不以点开头。
 */
inline bool task_name_acceptable(const std::string & task_name)
{
  const auto first = task_name.find_first_not_of(" \t");
  if (first == std::string::npos) {
    return false;
  }
  const auto last = task_name.find_last_not_of(" \t");
  const std::string trimmed = task_name.substr(first, last - first + 1);
  if (trimmed.size() > kMaxTaskNameBytes || trimmed.front() == '.') {
    return false;
  }
  return std::none_of(trimmed.begin(), trimmed.end(), [](unsigned char c) {
    return c == '/' || c == '\\' || c < 32 || c == 127;
  });
}

/**
 * @brief 计算按钮可用性：仅当已连接、没有在途请求且后端允许时才可点击。
 * @param flags 最近一次新鲜状态中的后端可用性。
 * @param connected 状态是否新鲜。
 * @param request_pending 面板是否存在未完成的修改请求。
 * @param task_name_ok 任务名是否可提交。
 */
inline ButtonGate compute_button_gate(
  const BackendFlags & flags, bool connected, bool request_pending, bool task_name_ok)
{
  const bool idle_ui = connected && !request_pending;
  ButtonGate gate;
  gate.start = idle_ui && flags.can_start && task_name_ok;
  gate.stop_and_save = idle_ui && flags.can_stop_and_save;
  gate.stop_and_cancel = idle_ui && flags.can_stop_and_cancel;
  gate.save = idle_ui && flags.can_save;
  gate.cancel = idle_ui && flags.can_cancel;
  return gate;
}

/**
 * @brief 判断收到的状态是否应覆盖当前显示；版本回退的过期状态被丢弃。
 * @param have_status 是否已经应用过状态。
 * @param applied_version 已应用状态的版本号。
 * @param incoming_version 新状态版本号。
 */
inline bool should_apply_status(
  bool have_status, uint64_t applied_version, uint64_t incoming_version)
{
  return !have_status || incoming_version >= applied_version;
}

/** @brief 判断最近状态是否仍新鲜。 */
inline bool status_is_fresh(bool have_status, int64_t now_ms, int64_t last_status_ms)
{
  return have_status && (now_ms - last_status_ms) <= kCollectionStatusTimeoutMs;
}

/**
 * @brief 判断服务响应是否对应面板当前在途请求。
 * @param pending_id 当前在途请求 ID，空表示没有在途请求。
 * @param response_id 响应携带的请求 ID。
 */
inline bool response_matches_pending(
  const std::string & pending_id, const std::string & response_id)
{
  return !pending_id.empty() && pending_id == response_id;
}

/**
 * @brief 判断权威状态是否已经反映了在途请求的结果。
 * @param pending_id 当前在途请求 ID。
 * @param last_request_id 状态中最近处理的请求 ID。
 */
inline bool status_confirms_pending(
  const std::string & pending_id, const std::string & last_request_id)
{
  return !pending_id.empty() && pending_id == last_request_id;
}

/**
 * @brief 生成跨面板实例唯一的请求 ID。
 * @param nonce 面板实例随机串。
 * @param counter 单调递增计数。
 * @param operation 操作名，便于排查。
 */
inline std::string make_request_id(
  const std::string & nonce, uint64_t counter, const std::string & operation)
{
  return "panel-" + nonce + "-" + std::to_string(counter) + "-" + operation;
}

/** @brief 总页数，至少为 1。 */
inline uint32_t page_count(uint64_t total, uint32_t page_size = kCollectionPageSize)
{
  if (page_size == 0 || total == 0) {
    return 1;
  }
  return static_cast<uint32_t>((total + page_size - 1) / page_size);
}

/** @brief 把页码限制在 [0, 总页数 - 1]。 */
inline uint32_t clamp_page(uint32_t page, uint64_t total, uint32_t page_size = kCollectionPageSize)
{
  return std::min(page, page_count(total, page_size) - 1);
}

/** @brief 列表响应是否对应最近一次发出的查询；过期响应不得覆盖列表。 */
inline bool list_response_is_current(uint64_t latest_epoch, uint64_t response_epoch)
{
  return latest_epoch == response_epoch;
}

/** @brief 不可用（NaN 或负值哨兵）时返回占位符，否则按小数位格式化。 */
inline std::string format_metric(double value, int decimals, const std::string & unit)
{
  if (!std::isfinite(value)) {
    return "不可用";
  }
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(decimals) << value << " " << unit;
  return stream.str();
}

/** @brief 夹爪百分比文本；无效时显示占位符。 */
inline std::string format_gripper_percent(bool valid, double percent)
{
  if (!valid || !std::isfinite(percent)) {
    return "--";
  }
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(1) << std::clamp(percent, 0.0, 100.0) << " %";
  return stream.str();
}

/** @brief 单路输入在面板中的状态等级，决定颜色。 */
enum class StreamLevel
{
  kOk,
  kWarning,
  kError,
};

/** @brief 单路输入的状态文本及等级。 */
struct StreamVerdict
{
  StreamLevel level{StreamLevel::kError};
  std::string text;
};

/**
 * @brief 把发现、新鲜度、有效性和匹配情况归约为一条状态。
 * @param discovered 话题是否有发布者。
 * @param fresh 数据是否在超时内。
 * @param valid 内容是否有效。
 * @param matched 是否找到配对样本。
 */
inline StreamVerdict stream_verdict(bool discovered, bool fresh, bool valid, bool matched)
{
  if (!discovered) {
    return {StreamLevel::kError, "无发布者"};
  }
  if (!fresh) {
    return {StreamLevel::kError, "数据过期"};
  }
  if (!valid) {
    return {StreamLevel::kError, "数据无效"};
  }
  if (!matched) {
    return {StreamLevel::kWarning, "未匹配"};
  }
  return {StreamLevel::kOk, "正常"};
}

/** @brief 把后端报警码翻译为中文提示；未知码原样返回。 */
inline std::string alarm_text(const std::string & code)
{
  static const struct { const char * prefix; const char * label; } kStreams[] = {
    {"IMAGE_", "图像"}, {"TRACKER_", "Tracker"}, {"GRIPPER_", "夹爪"}};
  static const struct { const char * suffix; const char * label; } kKinds[] = {
    {"NOT_DISCOVERED", "话题无发布者"}, {"STALE", "数据过期"}, {"INVALID", "数据无效"},
    {"OUT_OF_ORDER", "时间戳乱序"}, {"UNMATCHED", "未找到匹配样本"}, {"LAG", "处理滞后"},
    {"GAP", "疑似丢帧"}};
  if (code == "DISK_LOW") {
    return "磁盘空间不足";
  }
  if (code == "WRITER_QUEUE_HIGH") {
    return "写入队列积压";
  }
  for (const auto & stream : kStreams) {
    const std::string prefix(stream.prefix);
    if (code.rfind(prefix, 0) != 0) {
      continue;
    }
    const std::string rest = code.substr(prefix.size());
    for (const auto & kind : kKinds) {
      if (rest == kind.suffix) {
        return std::string(stream.label) + kind.label;
      }
    }
  }
  return code;
}

/** @brief 后端状态名到中文显示文本。 */
inline std::string state_text(const std::string & state)
{
  if (state == "idle") {return "空闲";}
  if (state == "starting") {return "正在开始";}
  if (state == "recording") {return "采集中";}
  if (state == "stopping") {return "正在停止";}
  if (state == "pending") {return "待保存";}
  if (state == "saving") {return "正在保存";}
  if (state == "cancelling") {return "正在取消";}
  if (state == "error") {return "故障（只能取消）";}
  return state;
}

/** @brief 把秒数格式化为 HH:MM:SS。 */
inline std::string format_duration(double seconds)
{
  const int64_t total = static_cast<int64_t>(std::max(0.0, std::isfinite(seconds) ? seconds : 0.0));
  std::ostringstream stream;
  stream << std::setfill('0') << std::setw(2) << total / 3600 << ":" << std::setw(2)
         << (total / 60) % 60 << ":" << std::setw(2) << total % 60;
  return stream.str();
}

/** @brief 把字节数格式化为 GiB 或 MiB。 */
inline std::string format_bytes(uint64_t bytes)
{
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(1);
  if (bytes >= (1ULL << 30)) {
    stream << static_cast<double>(bytes) / static_cast<double>(1ULL << 30) << " GiB";
  } else {
    stream << static_cast<double>(bytes) / static_cast<double>(1ULL << 20) << " MiB";
  }
  return stream.str();
}

}  // namespace fastumi_rviz_plugins

#endif  // FASTUMI_RVIZ_PLUGINS__COLLECTION_PANEL_STATE_HPP_
