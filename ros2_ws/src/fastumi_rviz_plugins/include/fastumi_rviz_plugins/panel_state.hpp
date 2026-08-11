/**
 * @file panel_state.hpp
 * @brief 定义不依赖 Qt/ROS 的回放标注 Panel 状态门控归约。
 */

#ifndef FASTUMI_RVIZ_PLUGINS__PANEL_STATE_HPP_
#define FASTUMI_RVIZ_PLUGINS__PANEL_STATE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace fastumi_rviz_plugins
{

/** @brief 描述全局按键需要触发或吞掉的 Panel 操作。 */
enum class ShortcutAction
{
  /** @brief 非快捷键，继续交给 RViz/Qt 处理。 */
  kNone,
  /** @brief 已识别快捷键的自动重复事件，仅吞掉且不触发操作。 */
  kSuppressAutoRepeat,
  /** @brief 切换 episode START/STOP。 */
  kToggleEpisode,
  /** @brief 切换回放 Pause/Resume。 */
  kTogglePlayback,
  /** @brief 冻结当前已完成标注并结束回放保存。 */
  kFinishAndSave,
  /** @brief 将回放倍率设为慢放 0.5×。 */
  kSetSlowRate,
  /** @brief 将回放倍率恢复为正常 1×。 */
  kSetNormalRate,
  /** @brief 将回放倍率设为快放 2×。 */
  kSetFastRate,
};

/** @brief 保存播放暂停状态和服务请求在途状态。 */
struct PlaybackUiState
{
  /** @brief 播放器当前是否处于暂停状态。 */
  bool paused{true};
  /** @brief 是否存在尚未完成的 Pause/Resume 请求。 */
  bool request_pending{false};
};

/** @brief 保存播放器倍率和 SetRate 请求状态。 */
struct RateUiState
{
  /** @brief 最近一次由播放器确认的倍率。 */
  double rate{1.0};
  /** @brief 是否存在尚未完成的 SetRate 请求。 */
  bool request_pending{false};
};

/** @brief 描述一次拖拽 Seek 的串行执行阶段。 */
enum class SeekStage
{
  /** @brief 没有拖拽或服务请求。 */
  kIdle,
  /** @brief 播放器原本暂停，当前只跟随用户拖拽。 */
  kDragging,
  /** @brief 播放器原本播放，正在等待 Pause 完成。 */
  kPausing,
  /** @brief 已释放滑块，正在等待 Seek 完成。 */
  kSeeking,
  /** @brief Seek 已结束，正在恢复拖拽前的播放状态。 */
  kResuming,
};

/** @brief 保存拖拽 Seek 的阶段、目标和恢复策略。 */
struct SeekUiState
{
  /** @brief 当前串行执行阶段。 */
  SeekStage stage{SeekStage::kIdle};
  /** @brief 完成 Seek 后是否需要恢复播放。 */
  bool restore_playback{false};
  /** @brief 用户是否已经释放滑块。 */
  bool release_received{false};
  /** @brief 经过边界钳制的绝对 bag 时间，单位为纳秒。 */
  int64_t target_time_ns{0};
};

/** @brief 时间轴滑块的固定归一化最大刻度。 */
inline constexpr int kTimelineSliderMaximum = 1000000;

/**
 * @brief 判断带版本的权威快照能否覆盖 Panel 当前标注状态。
 * @param current_revision Panel 已应用的最大 revision。
 * @param incoming_revision 新快照携带的 revision。
 * @return 新快照不旧于当前状态时返回 true。
 */
inline bool should_apply_annotation_snapshot(
  const uint64_t current_revision, const uint64_t incoming_revision)
{
  return incoming_revision >= current_revision;
}

/**
 * @brief 判断列表响应是否由当前查询代次发出。
 * @param current_epoch Panel 当前列表查询代次。
 * @param incoming_epoch 发出该列表请求时捕获的代次。
 * @return 两个代次相同时返回 true。
 */
inline bool annotation_query_is_current(
  const uint64_t current_epoch, const uint64_t incoming_epoch)
{
  return incoming_epoch == current_epoch;
}

/**
 * @brief 判断变更请求是否已被目标 revision 的权威快照确认。
 * @param request_pending 是否存在等待快照确认的变更请求。
 * @param service_succeeded 是否已经收到该变更服务的成功响应。
 * @param minimum_revision 本次请求至少需要到达的 revision。
 * @param incoming_revision 新快照携带的 revision。
 * @return 请求在途且新快照达到目标 revision 时返回 true。
 */
inline bool annotation_request_satisfied(
  const bool request_pending, const bool service_succeeded,
  const uint64_t minimum_revision, const uint64_t incoming_revision)
{
  return request_pending && service_succeeded &&
         incoming_revision >= minimum_revision;
}

/**
 * @brief 归约 episode 控制按钮的时钟、服务发现和在途请求门控条件。
 * @param clock_ready 是否已收到非零回放时钟。
 * @param services_ready START/STOP 服务是否均已发现。
 * @param request_pending 是否已有未完成 episode 服务请求。
 * @param annotation_time_safe 当前回放时间是否不早于最后一条边界。
 * @return 所有条件满足时返回 true。
 */
inline bool episode_control_enabled(
  const bool clock_ready, const bool services_ready, const bool request_pending,
  const bool annotation_time_safe)
{
  /** @brief 三项硬门控缺一不可，避免零时钟或并发重复标注。 */
  return clock_ready && services_ready && !request_pending && annotation_time_safe;
}

/**
 * @brief 判断当前回放时间是否允许新增 episode 边界。
 * @param current_time_ns 当前绝对回放时间。
 * @param last_event_time_ns 最后一条已保存边界时间；零代表无标记。
 * @return 无历史边界或当前时间不早于末次边界时返回 true。
 */
inline bool annotation_time_safe(
  const int64_t current_time_ns, const int64_t last_event_time_ns)
{
  return last_event_time_ns <= 0 || current_time_ns >= last_event_time_ns;
}

/**
 * @brief 将已由 Qt 判定的 Space/Enter/Ctrl+S 状态归类为 Panel 快捷操作。
 * @param is_space 是否为 Space。
 * @param is_enter 是否为主键盘 Return 或小键盘 Enter。
 * @param is_finish 是否为无额外修饰键的 Ctrl+S。
 * @param is_left 是否为左方向键。
 * @param is_up 是否为上方向键。
 * @param is_right 是否为右方向键。
 * @param auto_repeat 是否为按键自动重复事件。
 * @return 需要触发、吞掉或继续传递的操作。
 */
inline ShortcutAction classify_shortcut(
  const bool is_space, const bool is_enter, const bool is_finish, const bool is_left,
  const bool is_up, const bool is_right, const bool auto_repeat)
{
  /** @brief 识别 episode、播放、结束保存和固定三档倍率快捷键。 */
  const bool recognized = is_space || is_enter || is_finish || is_left || is_up || is_right;
  if (!recognized) {
    return ShortcutAction::kNone;
  }
  if (auto_repeat) {
    return ShortcutAction::kSuppressAutoRepeat;
  }
  if (is_space) {
    return ShortcutAction::kToggleEpisode;
  }
  if (is_enter) {
    return ShortcutAction::kTogglePlayback;
  }
  if (is_finish) {
    return ShortcutAction::kFinishAndSave;
  }
  if (is_left) {
    return ShortcutAction::kSetSlowRate;
  }
  return is_up ? ShortcutAction::kSetNormalRate : ShortcutAction::kSetFastRate;
}

/**
 * @brief 判断手动结束保存按钮和 Ctrl+S 是否可以发起请求。
 * @param service_ready Python 结束保存服务和回放时钟是否就绪。
 * @param episode_active 当前是否存在未闭合 episode。
 * @param request_pending 是否存在任一改变状态的请求。
 * @return 可以安全冻结标注并结束回放时返回 true。
 */
inline bool finish_control_enabled(
  const bool service_ready, const bool episode_active, const bool request_pending)
{
  return service_ready && !episode_active && !request_pending;
}

/**
 * @brief 判断删除所有标记按钮是否可以发起请求。
 * @param service_ready Python 清空服务和回放时钟是否就绪。
 * @param has_annotations 当前是否存在任意 START、STOP 或 ABORT。
 * @param request_pending 是否存在任一改变状态的请求。
 * @return 可以显示确认框并提交清空请求时返回 true。
 */
inline bool clear_control_enabled(
  const bool service_ready, const bool has_annotations, const bool request_pending)
{
  return service_ready && has_annotations && !request_pending;
}

/**
 * @brief 判断已完成 episode 列表是否允许点击跳转。
 * @param clock_ready 是否已有有效 bag 时钟。
 * @param metadata_ready 是否已有有效起点和持续时间。
 * @param episode_active 当前是否存在活动 episode。
 * @param services_ready Pause 和 Seek 服务是否均可用。
 * @param request_pending 是否存在任一改变状态的请求。
 * @return 可以安全执行首帧暂停跳转时返回 true。
 */
inline bool episode_list_control_enabled(
  const bool clock_ready, const bool metadata_ready, const bool episode_active,
  const bool services_ready, const bool request_pending)
{
  return clock_ready && metadata_ready && !episode_active && services_ready &&
         !request_pending;
}

/**
 * @brief 判断右键删除选中 episode 是否可用。
 * @param service_ready 单条删除服务是否已发现。
 * @param has_selection 列表当前是否选中可删除项。
 * @param episode_active 当前是否存在活动 episode。
 * @param request_pending 是否存在任一改变状态的请求或确认框。
 * @return 可以显示删除确认框并提交请求时返回 true。
 */
inline bool delete_episode_control_enabled(
  const bool service_ready, const bool has_selection, const bool episode_active,
  const bool request_pending)
{
  return service_ready && has_selection && !episode_active && !request_pending;
}

/**
 * @brief 返回固定倍率快捷操作对应的倍率。
 * @param action 已分类的快捷操作。
 * @return 0.5、1.0 或 2.0；非倍率操作返回 0.0。
 */
inline double rate_for_shortcut(const ShortcutAction action)
{
  switch (action) {
    case ShortcutAction::kSetSlowRate:
      return 0.5;
    case ShortcutAction::kSetNormalRate:
      return 1.0;
    case ShortcutAction::kSetFastRate:
      return 2.0;
    default:
      return 0.0;
  }
}

/**
 * @brief 归约播放控制的服务发现和在途请求门禁。
 * @param service_ready 当前状态对应的 Pause 或 Resume 服务是否就绪。
 * @param request_pending 是否已有未完成播放请求。
 * @return 可以通过按钮或 Enter 发起切换时返回 true。
 */
inline bool playback_control_enabled(
  const bool service_ready, const bool request_pending)
{
  return service_ready && !request_pending;
}

/**
 * @brief 完成播放请求；只有成功响应才提交目标暂停状态。
 * @param state 请求发出后的 UI 状态。
 * @param target_paused 本次请求成功后应提交的暂停状态。
 * @param request_succeeded 服务 future 是否成功完成。
 * @return 已解除在途标志的播放 UI 状态。
 */
inline PlaybackUiState complete_playback_request(
  PlaybackUiState state, const bool target_paused, const bool request_succeeded)
{
  if (request_succeeded) {
    state.paused = target_paused;
  }
  state.request_pending = false;
  return state;
}

/**
 * @brief 完成 SetRate 请求；只有服务接受目标倍率时才提交显示状态。
 * @param state 请求发出后的倍率 UI 状态。
 * @param target_rate 请求的目标倍率。
 * @param request_succeeded future 完成且响应 success 为 true 时取 true。
 * @return 已解除在途门禁的倍率状态。
 */
inline RateUiState complete_rate_request(
  RateUiState state, const double target_rate, const bool request_succeeded)
{
  if (request_succeeded) {
    state.rate = target_rate;
  }
  state.request_pending = false;
  return state;
}

/**
 * @brief 把绝对 bag 时间转换为归一化滑块刻度。
 * @param start_time_ns bag 起始绝对时间。
 * @param duration_ns bag 持续时间。
 * @param time_ns 当前绝对时间。
 * @return 0 到 kTimelineSliderMaximum 之间的刻度。
 */
inline int slider_value_for_time(
  const int64_t start_time_ns, const int64_t duration_ns, const int64_t time_ns)
{
  if (duration_ns <= 0) {
    return 0;
  }
  /** @brief 相对时间先钳制，避免 epoch 纳秒参与乘法。 */
  const int64_t elapsed_ns = std::clamp(time_ns - start_time_ns, int64_t{0}, duration_ns);
  /** @brief long double 防止持续纳秒与百万刻度相乘溢出。 */
  const long double normalized =
    static_cast<long double>(elapsed_ns) * kTimelineSliderMaximum /
    static_cast<long double>(duration_ns);
  return static_cast<int>(std::llround(normalized));
}

/**
 * @brief 把归一化滑块刻度转换为绝对 bag 时间。
 * @param start_time_ns bag 起始绝对时间。
 * @param duration_ns bag 持续时间。
 * @param slider_value 归一化滑块刻度。
 * @return 钳制在 bag 首尾之间的绝对纳秒时间。
 */
inline int64_t time_for_slider_value(
  const int64_t start_time_ns, const int64_t duration_ns, const int slider_value)
{
  /** @brief 输入刻度限制在有效时间轴范围。 */
  const int clamped_value = std::clamp(slider_value, 0, kTimelineSliderMaximum);
  /** @brief 先计算相对时间，保留 64 位 epoch 起始值。 */
  const long double elapsed_ns =
    static_cast<long double>(duration_ns) * clamped_value /
    static_cast<long double>(kTimelineSliderMaximum);
  return start_time_ns + static_cast<int64_t>(std::llround(elapsed_ns));
}

/**
 * @brief 将 Seek 目标限制在最后事件边界和 bag 末尾之间。
 * @param requested_time_ns 用户请求的绝对时间。
 * @param minimum_time_ns bag 起点或最后一条 episode 事件时间。
 * @param end_time_ns bag 末尾绝对时间。
 * @return 安全的绝对 Seek 时间。
 */
inline int64_t clamp_seek_time(
  const int64_t requested_time_ns, const int64_t minimum_time_ns, const int64_t end_time_ns)
{
  return std::clamp(requested_time_ns, minimum_time_ns, end_time_ns);
}

/**
 * @brief 将相对纳秒格式化为固定宽度时分秒毫秒。
 * @param duration_ns 非负相对时长。
 * @return `HH:MM:SS.mmm` 文本。
 */
inline std::string format_duration_ns(const int64_t duration_ns)
{
  /** @brief 总毫秒向零截断，避免显示超过当前回放位置。 */
  const int64_t total_ms = std::max<int64_t>(0, duration_ns) / 1000000;
  /** @brief 各时间字段从总毫秒依次分解。 */
  const int64_t hours = total_ms / 3600000;
  const int64_t minutes = (total_ms / 60000) % 60;
  const int64_t seconds = (total_ms / 1000) % 60;
  const int64_t milliseconds = total_ms % 1000;
  /** @brief 固定宽度输出便于播放时稳定显示。 */
  std::ostringstream stream;
  stream << std::setfill('0') << std::setw(2) << hours << ":" <<
    std::setw(2) << minutes << ":" << std::setw(2) << seconds << "." <<
    std::setw(3) << milliseconds;
  return stream.str();
}

/**
 * @brief 判断时间轴是否允许开始拖拽。
 * @param clock_ready 是否已有有效 bag 时钟。
 * @param metadata_ready 是否已有有效起点和持续时间。
 * @param episode_active 当前是否存在活动 episode。
 * @param services_ready Pause、Resume 和 Seek 是否全部可用。
 * @param request_pending 是否已有播放或 Seek 请求在途。
 * @return 可以安全开始拖拽时返回 true。
 */
inline bool timeline_control_enabled(
  const bool clock_ready, const bool metadata_ready, const bool episode_active,
  const bool services_ready, const bool request_pending)
{
  return clock_ready && metadata_ready && !episode_active && services_ready && !request_pending;
}

/**
 * @brief 开始 Seek，并根据原播放状态和调用策略决定是否恢复播放。
 * @param playback_paused 请求前播放器是否已经暂停。
 * @param restore_playback 播放中发起时，Seek 后是否恢复播放。
 * @return 初始 Pause/拖拽阶段及恢复策略。
 */
inline SeekUiState begin_seek_sequence(
  const bool playback_paused, const bool restore_playback = true)
{
  return SeekUiState{
    playback_paused ? SeekStage::kDragging : SeekStage::kPausing,
    !playback_paused && restore_playback, false, 0};
}

/** @brief 记录释放目标；暂停已完成时立即进入 Seek 阶段。 */
inline SeekUiState release_seek_sequence(SeekUiState state, const int64_t target_time_ns)
{
  state.release_received = true;
  state.target_time_ns = target_time_ns;
  if (state.stage == SeekStage::kDragging) {
    state.stage = SeekStage::kSeeking;
  }
  return state;
}

/** @brief 完成临时 Pause；失败时取消整次 Seek。 */
inline SeekUiState complete_seek_pause(SeekUiState state, const bool request_succeeded)
{
  if (!request_succeeded) {
    state.stage = SeekStage::kIdle;
  } else {
    state.stage = state.release_received ? SeekStage::kSeeking : SeekStage::kDragging;
  }
  return state;
}

/** @brief 完成 Seek；原本播放时进入恢复阶段，原本暂停时结束。 */
inline SeekUiState complete_seek_request(SeekUiState state)
{
  state.stage = state.restore_playback ? SeekStage::kResuming : SeekStage::kIdle;
  return state;
}

/** @brief 完成恢复播放请求并结束拖拽序列。 */
inline SeekUiState complete_seek_resume(SeekUiState state)
{
  state.stage = SeekStage::kIdle;
  return state;
}

}  // namespace fastumi_rviz_plugins

#endif  // FASTUMI_RVIZ_PLUGINS__PANEL_STATE_HPP_
