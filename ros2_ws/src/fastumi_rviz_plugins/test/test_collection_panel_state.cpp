/**
 * @file test_collection_panel_state.cpp
 * @brief 验证采集 Panel 的按钮门控、过期响应丢弃、分页与文本格式化。
 */

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "fastumi_rviz_plugins/collection_panel_state.hpp"

namespace fp = fastumi_rviz_plugins;

/** @brief 构造一个全部允许的后端标志。 */
static fp::BackendFlags all_allowed()
{
  fp::BackendFlags flags;
  flags.can_start = flags.can_stop = flags.can_save = flags.can_cancel = true;
  flags.can_stop_and_save = flags.can_stop_and_cancel = true;
  return flags;
}

/** @brief 按钮可用性必须由后端标志决定，面板不得自行放行。 */
TEST(CollectionPanelState, ButtonsFollowBackendFlags)
{
  fp::BackendFlags flags;
  flags.can_stop_and_save = true;
  flags.can_stop_and_cancel = true;
  const auto gate = fp::compute_button_gate(flags, true, false, true);
  EXPECT_FALSE(gate.start);
  EXPECT_TRUE(gate.stop_and_save);
  EXPECT_TRUE(gate.stop_and_cancel);
  EXPECT_FALSE(gate.save);
  EXPECT_FALSE(gate.cancel);
}

/** @brief 失联、请求在途或任务名非法时禁止提交，避免重复操作。 */
TEST(CollectionPanelState, DisconnectedOrPendingDisablesEverything)
{
  const auto flags = all_allowed();
  for (const auto & gate : {
      fp::compute_button_gate(flags, false, false, true),
      fp::compute_button_gate(flags, true, true, true)})
  {
    EXPECT_FALSE(gate.start);
    EXPECT_FALSE(gate.stop_and_save);
    EXPECT_FALSE(gate.stop_and_cancel);
    EXPECT_FALSE(gate.save);
    EXPECT_FALSE(gate.cancel);
  }
  const auto invalid_task = fp::compute_button_gate(flags, true, false, false);
  EXPECT_FALSE(invalid_task.start);
  EXPECT_TRUE(invalid_task.stop_and_save);
}

/** @brief 任务名客户端校验与后端规则一致的拒绝项。 */
TEST(CollectionPanelState, TaskNameValidation)
{
  EXPECT_TRUE(fp::task_name_acceptable("pick_place"));
  EXPECT_TRUE(fp::task_name_acceptable("  抓取任务-2  "));
  EXPECT_FALSE(fp::task_name_acceptable(""));
  EXPECT_FALSE(fp::task_name_acceptable("   "));
  EXPECT_FALSE(fp::task_name_acceptable("a/b"));
  EXPECT_FALSE(fp::task_name_acceptable("a\\b"));
  EXPECT_FALSE(fp::task_name_acceptable(".hidden"));
  EXPECT_FALSE(fp::task_name_acceptable("a\nb"));
  EXPECT_FALSE(fp::task_name_acceptable(std::string(300, 'x')));
}

/** @brief 版本回退的过期状态不能覆盖显示；首个状态总是接受。 */
TEST(CollectionPanelState, StaleStatusIsRejected)
{
  EXPECT_TRUE(fp::should_apply_status(false, 100, 1));
  EXPECT_TRUE(fp::should_apply_status(true, 7, 7));
  EXPECT_TRUE(fp::should_apply_status(true, 7, 8));
  EXPECT_FALSE(fp::should_apply_status(true, 7, 6));
}

/** @brief 状态超过 3 s 未更新视为失联；未收到过状态也视为失联。 */
TEST(CollectionPanelState, FreshnessTimeout)
{
  EXPECT_FALSE(fp::status_is_fresh(false, 1000, 1000));
  EXPECT_TRUE(fp::status_is_fresh(true, 1000 + fp::kCollectionStatusTimeoutMs, 1000));
  EXPECT_FALSE(fp::status_is_fresh(true, 1001 + fp::kCollectionStatusTimeoutMs, 1000));
}

/** @brief 过期响应（请求 ID 不匹配或没有在途请求）不得改变面板状态。 */
TEST(CollectionPanelState, ResponseMatchingDropsStaleResponses)
{
  EXPECT_TRUE(fp::response_matches_pending("panel-a-1-start", "panel-a-1-start"));
  EXPECT_FALSE(fp::response_matches_pending("panel-a-2-stop", "panel-a-1-start"));
  EXPECT_FALSE(fp::response_matches_pending("", ""));
  EXPECT_TRUE(fp::status_confirms_pending("x", "x"));
  EXPECT_FALSE(fp::status_confirms_pending("x", "y"));
  EXPECT_FALSE(fp::status_confirms_pending("", ""));
}

/** @brief 请求 ID 含实例随机串和计数，跨实例与跨请求唯一。 */
TEST(CollectionPanelState, RequestIdsAreUnique)
{
  EXPECT_NE(fp::make_request_id("aa", 1, "start"), fp::make_request_id("aa", 2, "start"));
  EXPECT_NE(fp::make_request_id("aa", 1, "start"), fp::make_request_id("bb", 1, "start"));
  EXPECT_EQ(fp::make_request_id("aa", 3, "save"), "panel-aa-3-save");
}

/** @brief 分页计算：空列表仍有 1 页，越界页码被夹紧。 */
TEST(CollectionPanelState, Paging)
{
  EXPECT_EQ(fp::page_count(0), 1u);
  EXPECT_EQ(fp::page_count(10), 1u);
  EXPECT_EQ(fp::page_count(11), 2u);
  EXPECT_EQ(fp::clamp_page(5, 11), 1u);
  EXPECT_EQ(fp::clamp_page(0, 0), 0u);
  EXPECT_TRUE(fp::list_response_is_current(4, 4));
  EXPECT_FALSE(fp::list_response_is_current(5, 4));
}

/** @brief 无效夹爪与不可用指标显示占位符而不是数字。 */
TEST(CollectionPanelState, UnavailableMetricsUsePlaceholders)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(fp::format_gripper_percent(false, 50.0), "--");
  EXPECT_EQ(fp::format_gripper_percent(true, nan), "--");
  EXPECT_EQ(fp::format_gripper_percent(true, 50.04), "50.0 %");
  EXPECT_EQ(fp::format_gripper_percent(true, 120.0), "100.0 %");
  EXPECT_EQ(fp::format_metric(nan, 2, "ms"), "不可用");
  EXPECT_EQ(fp::format_metric(1.234, 2, "ms"), "1.23 ms");
}

/** @brief 单路状态按发现、新鲜度、有效性、匹配的优先级归约。 */
TEST(CollectionPanelState, StreamVerdictPriority)
{
  using fp::StreamLevel;
  EXPECT_EQ(fp::stream_verdict(false, true, true, true).text, "无发布者");
  EXPECT_EQ(fp::stream_verdict(true, false, true, true).text, "数据过期");
  EXPECT_EQ(fp::stream_verdict(true, true, false, true).text, "数据无效");
  EXPECT_EQ(fp::stream_verdict(true, true, true, false).level, StreamLevel::kWarning);
  EXPECT_EQ(fp::stream_verdict(true, true, true, true).level, StreamLevel::kOk);
}

/** @brief 报警码翻译覆盖三路输入的全部种类，未知码原样保留。 */
TEST(CollectionPanelState, AlarmText)
{
  EXPECT_EQ(fp::alarm_text("IMAGE_STALE"), "图像数据过期");
  EXPECT_EQ(fp::alarm_text("TRACKER_UNMATCHED"), "Tracker未找到匹配样本");
  EXPECT_EQ(fp::alarm_text("GRIPPER_INVALID"), "夹爪数据无效");
  EXPECT_EQ(fp::alarm_text("GRIPPER_LAG"), "夹爪处理滞后");
  EXPECT_EQ(fp::alarm_text("IMAGE_GAP"), "图像疑似丢帧");
  EXPECT_EQ(fp::alarm_text("DISK_LOW"), "磁盘空间不足");
  EXPECT_EQ(fp::alarm_text("SOMETHING_NEW"), "SOMETHING_NEW");
}

/** @brief 状态名翻译和时长、容量格式化。 */
TEST(CollectionPanelState, TextFormatting)
{
  EXPECT_EQ(fp::state_text("recording"), "采集中");
  EXPECT_EQ(fp::state_text("pending"), "待保存");
  EXPECT_EQ(fp::state_text("unknown"), "unknown");
  EXPECT_EQ(fp::format_duration(3725.9), "01:02:05");
  EXPECT_EQ(fp::format_duration(std::nan("")), "00:00:00");
  EXPECT_EQ(fp::format_bytes(3ULL << 30), "3.0 GiB");
  EXPECT_EQ(fp::format_bytes(5ULL << 20), "5.0 MiB");
}

/** @brief 状态名必须与后端 CollectionStatus 常量一致，防止两端漂移。 */
TEST(CollectionPanelState, StateNamesMatchInterfaceDefinition)
{
  std::ifstream definition(
    std::string(FASTUMI_RVIZ_PLUGIN_SOURCE_DIR) +
    "/../fastumi_interfaces/msg/CollectionStatus.msg");
  ASSERT_TRUE(definition.good());
  std::stringstream buffer;
  buffer << definition.rdbuf();
  for (const char * state : {"idle", "starting", "recording", "stopping", "pending",
      "saving", "cancelling", "error"})
  {
    EXPECT_NE(buffer.str().find(std::string("=") + state), std::string::npos) << state;
    EXPECT_NE(fp::state_text(state), std::string(state)) << state;
  }
}
