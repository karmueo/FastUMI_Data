/**
 * @file test_panel_gating.cpp
 * @brief 验证回放标注 Panel 的 episode 控制状态门控归约。
 */

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>

#include "fastumi_rviz_plugins/panel_state.hpp"

/** @brief 验证零时钟、未发现服务和在途请求都会禁止 episode 控制。 */
TEST(ReplayAnnotationPanel, EpisodeControlGating)
{
  /** @brief 三种关键安全门控组合。 */
  EXPECT_FALSE(fastumi_rviz_plugins::episode_control_enabled(false, true, false, true));
  EXPECT_FALSE(fastumi_rviz_plugins::episode_control_enabled(true, false, false, true));
  EXPECT_FALSE(fastumi_rviz_plugins::episode_control_enabled(true, true, true, true));
  EXPECT_FALSE(fastumi_rviz_plugins::episode_control_enabled(true, true, false, false));
  EXPECT_TRUE(fastumi_rviz_plugins::episode_control_enabled(true, true, false, true));
}

/** @brief 验证旧快照不能覆盖新状态，变更请求等待目标 revision。 */
TEST(ReplayAnnotationPanel, AnnotationRevisionRejectsStaleSnapshots)
{
  EXPECT_FALSE(fastumi_rviz_plugins::should_apply_annotation_snapshot(8, 7));
  EXPECT_TRUE(fastumi_rviz_plugins::should_apply_annotation_snapshot(8, 8));
  EXPECT_TRUE(fastumi_rviz_plugins::should_apply_annotation_snapshot(8, 9));
  EXPECT_FALSE(fastumi_rviz_plugins::annotation_request_satisfied(true, false, 9, 10));
  EXPECT_FALSE(fastumi_rviz_plugins::annotation_request_satisfied(true, true, 9, 8));
  EXPECT_TRUE(fastumi_rviz_plugins::annotation_request_satisfied(true, true, 9, 9));
  EXPECT_FALSE(fastumi_rviz_plugins::annotation_request_satisfied(false, true, 9, 10));
}

/** @brief 验证服务成功前发出的列表查询即使 revision 较高也不能确认该服务。 */
TEST(ReplayAnnotationPanel, AnnotationQueryEpochRejectsPreMutationResponse)
{
  EXPECT_TRUE(fastumi_rviz_plugins::annotation_query_is_current(4, 4));
  EXPECT_FALSE(fastumi_rviz_plugins::annotation_query_is_current(5, 4));
  EXPECT_FALSE(fastumi_rviz_plugins::annotation_query_is_current(4, 5));
}

/** @brief 验证 Space、双 Enter、Ctrl+S、三枚方向键和自动重复映射互不冲突。 */
TEST(ReplayAnnotationPanel, GlobalShortcutClassification)
{
  /** @brief 使用简短别名提高各按键组合断言的可读性。 */
  using ShortcutAction = fastumi_rviz_plugins::ShortcutAction;
  EXPECT_EQ(
    fastumi_rviz_plugins::classify_shortcut(true, false, false, false, false, false, false),
    ShortcutAction::kToggleEpisode);
  EXPECT_EQ(
    fastumi_rviz_plugins::classify_shortcut(false, true, false, false, false, false, false),
    ShortcutAction::kTogglePlayback);
  EXPECT_EQ(
    fastumi_rviz_plugins::classify_shortcut(false, false, true, false, false, false, false),
    ShortcutAction::kFinishAndSave);
  EXPECT_EQ(
    fastumi_rviz_plugins::classify_shortcut(false, false, false, true, false, false, false),
    ShortcutAction::kSetSlowRate);
  EXPECT_EQ(
    fastumi_rviz_plugins::classify_shortcut(false, false, false, false, true, false, false),
    ShortcutAction::kSetNormalRate);
  EXPECT_EQ(
    fastumi_rviz_plugins::classify_shortcut(false, false, false, false, false, true, false),
    ShortcutAction::kSetFastRate);
  EXPECT_EQ(
    fastumi_rviz_plugins::classify_shortcut(false, false, false, false, false, false, false),
    ShortcutAction::kNone);
  EXPECT_EQ(
    fastumi_rviz_plugins::classify_shortcut(true, false, false, false, false, false, true),
    ShortcutAction::kSuppressAutoRepeat);
  EXPECT_EQ(
    fastumi_rviz_plugins::classify_shortcut(false, false, false, false, false, true, true),
    ShortcutAction::kSuppressAutoRepeat);
}

/** @brief 验证手动结束保存要求服务就绪、无活动 episode 且所有请求均空闲。 */
TEST(ReplayAnnotationPanel, FinishAndSaveControlGating)
{
  EXPECT_FALSE(fastumi_rviz_plugins::finish_control_enabled(false, false, false));
  EXPECT_FALSE(fastumi_rviz_plugins::finish_control_enabled(true, true, false));
  EXPECT_FALSE(fastumi_rviz_plugins::finish_control_enabled(true, false, true));
  EXPECT_TRUE(fastumi_rviz_plugins::finish_control_enabled(true, false, false));
}

/** @brief 验证删除按钮只在存在标记、服务就绪且无并发请求时可用。 */
TEST(ReplayAnnotationPanel, ClearAnnotationsControlGating)
{
  EXPECT_FALSE(fastumi_rviz_plugins::clear_control_enabled(false, true, false));
  EXPECT_FALSE(fastumi_rviz_plugins::clear_control_enabled(true, false, false));
  EXPECT_FALSE(fastumi_rviz_plugins::clear_control_enabled(true, true, true));
  EXPECT_TRUE(fastumi_rviz_plugins::clear_control_enabled(true, true, false));
}

/** @brief 验证固定三档倍率及失败时不提交目标倍率。 */
TEST(ReplayAnnotationPanel, FixedRateShortcutsAndCompletion)
{
  using ShortcutAction = fastumi_rviz_plugins::ShortcutAction;
  EXPECT_DOUBLE_EQ(fastumi_rviz_plugins::rate_for_shortcut(ShortcutAction::kSetSlowRate), 0.5);
  EXPECT_DOUBLE_EQ(fastumi_rviz_plugins::rate_for_shortcut(ShortcutAction::kSetNormalRate), 1.0);
  EXPECT_DOUBLE_EQ(fastumi_rviz_plugins::rate_for_shortcut(ShortcutAction::kSetFastRate), 2.0);

  const fastumi_rviz_plugins::RateUiState pending{1.0, true};
  const auto succeeded = fastumi_rviz_plugins::complete_rate_request(pending, 2.0, true);
  const auto failed = fastumi_rviz_plugins::complete_rate_request(pending, 2.0, false);
  EXPECT_DOUBLE_EQ(succeeded.rate, 2.0);
  EXPECT_FALSE(succeeded.request_pending);
  EXPECT_DOUBLE_EQ(failed.rate, 1.0);
  EXPECT_FALSE(failed.request_pending);
}

/** @brief 验证绝对纳秒、归一化滑块与末次事件边界钳制。 */
TEST(ReplayAnnotationPanel, TimelineConversionAndBoundaryClamp)
{
  constexpr int64_t start_ns = 1786347834309694563LL;
  constexpr int64_t duration_ns = 54384407757LL;
  const int middle = fastumi_rviz_plugins::slider_value_for_time(
    start_ns, duration_ns, start_ns + duration_ns / 2);
  EXPECT_NEAR(middle, fastumi_rviz_plugins::kTimelineSliderMaximum / 2, 1);
  const int64_t reconstructed = fastumi_rviz_plugins::time_for_slider_value(
    start_ns, duration_ns, middle);
  EXPECT_NEAR(
    static_cast<double>(reconstructed),
    static_cast<double>(start_ns + duration_ns / 2),
    60000.0);
  EXPECT_EQ(
    fastumi_rviz_plugins::clamp_seek_time(start_ns + 10, start_ns, start_ns + duration_ns),
    start_ns + 10);
  EXPECT_EQ(
    fastumi_rviz_plugins::clamp_seek_time(start_ns - 1, start_ns + 10, start_ns + duration_ns),
    start_ns + 10);
  EXPECT_EQ(fastumi_rviz_plugins::format_duration_ns(3723004000000LL), "01:02:03.004");
}

/** @brief 验证播放中拖拽的 Pause→Seek→Resume 串行状态转换。 */
TEST(ReplayAnnotationPanel, SeekSequenceHandlesEarlyReleaseAndFailures)
{
  using SeekStage = fastumi_rviz_plugins::SeekStage;
  auto state = fastumi_rviz_plugins::begin_seek_sequence(false);
  EXPECT_EQ(state.stage, SeekStage::kPausing);
  EXPECT_TRUE(state.restore_playback);

  state = fastumi_rviz_plugins::release_seek_sequence(state, 1234);
  EXPECT_EQ(state.stage, SeekStage::kPausing);
  EXPECT_TRUE(state.release_received);
  EXPECT_EQ(state.target_time_ns, 1234);

  state = fastumi_rviz_plugins::complete_seek_pause(state, true);
  EXPECT_EQ(state.stage, SeekStage::kSeeking);
  state = fastumi_rviz_plugins::complete_seek_request(state);
  EXPECT_EQ(state.stage, SeekStage::kResuming);
  state = fastumi_rviz_plugins::complete_seek_resume(state);
  EXPECT_EQ(state.stage, SeekStage::kIdle);

  auto failed_pause = fastumi_rviz_plugins::begin_seek_sequence(false);
  failed_pause = fastumi_rviz_plugins::complete_seek_pause(failed_pause, false);
  EXPECT_EQ(failed_pause.stage, SeekStage::kIdle);

  auto paused_seek = fastumi_rviz_plugins::begin_seek_sequence(true);
  EXPECT_EQ(paused_seek.stage, SeekStage::kDragging);
  paused_seek = fastumi_rviz_plugins::release_seek_sequence(paused_seek, 5678);
  EXPECT_EQ(paused_seek.stage, SeekStage::kSeeking);
  paused_seek = fastumi_rviz_plugins::complete_seek_request(paused_seek);
  EXPECT_EQ(paused_seek.stage, SeekStage::kIdle);

  /** @brief 列表跳转即使原本播放，也会 Pause→Seek 后保持暂停。 */
  auto list_seek = fastumi_rviz_plugins::begin_seek_sequence(false, false);
  EXPECT_EQ(list_seek.stage, SeekStage::kPausing);
  EXPECT_FALSE(list_seek.restore_playback);
  list_seek = fastumi_rviz_plugins::release_seek_sequence(list_seek, 42);
  list_seek = fastumi_rviz_plugins::complete_seek_pause(list_seek, true);
  list_seek = fastumi_rviz_plugins::complete_seek_request(list_seek);
  EXPECT_EQ(list_seek.stage, SeekStage::kIdle);
}

/** @brief 验证列表回看门控和回退后的新增标记时间保护。 */
TEST(ReplayAnnotationPanel, EpisodeListJumpAndAnnotationTimeGating)
{
  EXPECT_FALSE(fastumi_rviz_plugins::annotation_time_safe(99, 100));
  EXPECT_TRUE(fastumi_rviz_plugins::annotation_time_safe(100, 100));
  EXPECT_TRUE(fastumi_rviz_plugins::annotation_time_safe(1, 0));
  EXPECT_FALSE(fastumi_rviz_plugins::episode_list_control_enabled(
    true, true, true, true, false));
  EXPECT_FALSE(fastumi_rviz_plugins::episode_list_control_enabled(
    true, true, false, true, true));
  EXPECT_TRUE(fastumi_rviz_plugins::episode_list_control_enabled(
    true, true, false, true, false));
  EXPECT_FALSE(fastumi_rviz_plugins::delete_episode_control_enabled(
    true, true, true, false));
  EXPECT_TRUE(fastumi_rviz_plugins::delete_episode_control_enabled(
    true, true, false, false));
}

/** @brief 验证活动 episode、元数据缺失和请求在途都会禁止时间轴。 */
TEST(ReplayAnnotationPanel, TimelineControlGating)
{
  EXPECT_FALSE(fastumi_rviz_plugins::timeline_control_enabled(true, true, true, true, false));
  EXPECT_FALSE(fastumi_rviz_plugins::timeline_control_enabled(true, false, false, true, false));
  EXPECT_FALSE(fastumi_rviz_plugins::timeline_control_enabled(true, true, false, false, false));
  EXPECT_FALSE(fastumi_rviz_plugins::timeline_control_enabled(true, true, false, true, true));
  EXPECT_TRUE(fastumi_rviz_plugins::timeline_control_enabled(true, true, false, true, false));
}

/** @brief 验证播放服务发现和在途请求共同控制按钮及 Enter 可用性。 */
TEST(ReplayAnnotationPanel, PlaybackControlGating)
{
  EXPECT_FALSE(fastumi_rviz_plugins::playback_control_enabled(false, false));
  EXPECT_FALSE(fastumi_rviz_plugins::playback_control_enabled(true, true));
  EXPECT_TRUE(fastumi_rviz_plugins::playback_control_enabled(true, false));
}

/** @brief 验证播放请求成功才提交状态，失败仍会解除在途门禁。 */
TEST(ReplayAnnotationPanel, PlaybackCompletionCommitsOnlySuccess)
{
  /** @brief 模拟从暂停状态发起 Resume 的在途 UI 状态。 */
  const fastumi_rviz_plugins::PlaybackUiState pending{true, true};
  /** @brief 成功 Resume 后应进入播放状态并解除 pending。 */
  const auto succeeded = fastumi_rviz_plugins::complete_playback_request(
    pending, false, true);
  /** @brief 失败 Resume 后应保留暂停状态但解除 pending。 */
  const auto failed = fastumi_rviz_plugins::complete_playback_request(
    pending, false, false);
  EXPECT_FALSE(succeeded.paused);
  EXPECT_FALSE(succeeded.request_pending);
  EXPECT_TRUE(failed.paused);
  EXPECT_FALSE(failed.request_pending);
}

/** @brief 验证默认显示启用，Panel 只从带 revision 的权威服务恢复状态。 */
TEST(ReplayAnnotationPanel, DefaultConfigurationUsesAuthoritativeRevisionedState)
{
  /** @brief 加载源码配置和 Panel 实现，防止安装前配置回归。 */
  const std::string source_root = FASTUMI_RVIZ_PLUGIN_SOURCE_DIR;
  std::ifstream config_file(source_root + "/config/replay_annotation.rviz");
  std::stringstream config_buffer;
  config_buffer << config_file.rdbuf();
  const std::string config = config_buffer.str();
  std::ifstream panel_file(source_root + "/src/replay_annotation_panel.cpp");
  std::stringstream panel_buffer;
  panel_buffer << panel_file.rdbuf();
  const std::string panel = panel_buffer.str();
  EXPECT_NE(config.find("Name: root"), std::string::npos);
  EXPECT_TRUE(config_file.good() || !config.empty());
  EXPECT_NE(config.find("Reliability Policy: Best Effort"), std::string::npos);
  EXPECT_NE(config.find("Class: rviz_default_plugins/Image\n      Enabled: true"), std::string::npos);
  /** @brief Pose 显示块允许在 Class 与 Enabled 之间包含颜色等配置。 */
  const std::size_t pose_block_start = config.find("Class: rviz_default_plugins/Pose");
  ASSERT_NE(pose_block_start, std::string::npos);
  /** @brief 下一显示块边界用于避免误匹配其他 Display 的 Enabled。 */
  const std::size_t pose_block_end = config.find("\n    - Class:", pose_block_start + 1);
  /** @brief 独立截取 Pose 配置块以验证其默认启用状态。 */
  const std::string pose_block = config.substr(pose_block_start, pose_block_end - pose_block_start);
  EXPECT_NE(pose_block.find("Enabled: true"), std::string::npos);
  EXPECT_NE(config.find("Class: rviz_default_plugins/TF\n      Enabled: true"), std::string::npos);
  EXPECT_EQ(panel.find("create_subscription<fastumi_interfaces::msg::EpisodeEvent>"), std::string::npos);
  EXPECT_NE(panel.find("response->revision"), std::string::npos);
  EXPECT_NE(panel.find("完整 Episode 数"), std::string::npos);
  EXPECT_NE(panel.find("删除所有标记"), std::string::npos);
  EXPECT_NE(panel.find("已标注 Episode"), std::string::npos);
  EXPECT_NE(panel.find("setContextMenuPolicy(Qt::CustomContextMenu)"), std::string::npos);
  EXPECT_NE(panel.find("QMessageBox::warning"), std::string::npos);
  EXPECT_NE(panel.find("clear_confirmation_active_ = true"), std::string::npos);
  EXPECT_NE(panel.find("Qt::Key_Return"), std::string::npos);
  EXPECT_NE(panel.find("Qt::Key_Enter"), std::string::npos);
}
