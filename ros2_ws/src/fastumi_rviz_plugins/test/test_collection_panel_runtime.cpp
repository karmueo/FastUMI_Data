/**
 * @file test_collection_panel_runtime.cpp
 * @brief 验证真实 Qt 采集面板的请求跟踪、滚动布局及输入框与全局快捷键的交互。
 */

#include <gtest/gtest.h>

#include <QApplication>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QTimer>

#include "fastumi_rviz_plugins/collection_panel.hpp"

namespace fastumi_rviz_plugins
{

/** @brief 通过友元访问面板内部状态，执行真实响应处理回归测试。 */
class CollectionPanelTestPeer
{
public:
  /** @brief 旧请求已由状态确认时，旧响应不能结束或覆盖新请求。 */
  static void expect_old_response_preserves_new_request()
  {
    /** @brief 无需 ROS 节点的真实 Qt 面板。 */
    CollectionPanel panel;
    /** @brief 已被状态确认、尚未收到服务响应的旧请求。 */
    const std::string old_id = panel.begin_request("start", 25000);
    /** @brief 模拟先于服务响应到达的权威状态。 */
    fastumi_interfaces::msg::CollectionStatus status;
    status.state = "recording";
    status.state_version = 1;
    status.last_request_id = old_id;
    status.last_result_code = "OK";
    panel.apply_status(status);
    ASSERT_TRUE(panel.pending_request_id_.empty());
    /** @brief 用户在旧响应到达前发出的新请求。 */
    const std::string new_id = panel.begin_request("stop_and_save", 75000);
    /** @brief 新请求的等待提示，旧响应不能覆盖该文本。 */
    const QString waiting_message = panel.message_label_->text();
    /** @brief 延迟到达的旧请求服务响应。 */
    fastumi_interfaces::msg::CollectionResult old_result;
    old_result.request_id = old_id;
    old_result.accepted = true;
    old_result.completed = true;
    old_result.code = "OK";
    panel.handle_result("开始", old_result);

    EXPECT_EQ(panel.pending_request_id_, new_id);
    EXPECT_TRUE(panel.request_timer_->isActive());
    EXPECT_EQ(panel.message_label_->text(), waiting_message);
    EXPECT_TRUE(panel.confirmed_request_id_.empty());

    /** @brief 当前请求的响应应正常结束等待。 */
    fastumi_interfaces::msg::CollectionResult new_result;
    new_result.request_id = new_id;
    new_result.accepted = true;
    new_result.completed = true;
    new_result.code = "OK";
    panel.handle_result("停止并保存", new_result);
    EXPECT_TRUE(panel.pending_request_id_.empty());
    EXPECT_FALSE(panel.request_timer_->isActive());
  }

  /** @brief 面板内容可滚动，最小高度不能撑大 RViz 主窗口（否则无法最大化）。 */
  static void expect_small_minimum_height()
  {
    CollectionPanel panel;
    EXPECT_LE(panel.minimumSizeHint().height(), 300);
  }

  /** @brief 可开始时 Space 走开始请求；自动重复被吞掉且不重复触发。 */
  static void expect_space_starts_collection()
  {
    CollectionPanel panel;
    fastumi_interfaces::msg::CollectionStatus status;
    status.state = "idle";
    status.state_version = 1;
    status.can_start = true;
    panel.apply_status(status);
    panel.message_label_->clear();

    QKeyEvent repeat(QEvent::KeyPress, Qt::Key_Space, Qt::NoModifier, " ", true);
    EXPECT_TRUE(QApplication::sendEvent(&panel, &repeat));
    EXPECT_TRUE(panel.message_label_->text().isEmpty());

    QKeyEvent space(QEvent::KeyPress, Qt::Key_Space, Qt::NoModifier, " ");
    EXPECT_TRUE(QApplication::sendEvent(&panel, &space));
    /** @brief 测试中没有 ROS 客户端，开始请求应以服务不可用结束。 */
    EXPECT_TRUE(panel.message_label_->text().contains("采集服务不可用"));
    EXPECT_TRUE(panel.pending_request_id_.empty());
  }

  /** @brief 筛选框回车只应用筛选，焦点交还后下一次回车才触发保存。 */
  static void expect_filter_return_does_not_save()
  {
    /** @brief 真实面板，保留输入框到父控件的按键传播路径。 */
    CollectionPanel panel;
    /** @brief 后端处于待保存状态，允许全局 Enter 保存。 */
    fastumi_interfaces::msg::CollectionStatus status;
    status.state = "pending";
    status.state_version = 1;
    status.can_save = true;
    panel.apply_status(status);
    panel.resize(800, 1200);
    panel.show();
    panel.activateWindow();
    QApplication::processEvents();
    panel.filter_edit_->setFocus();
    QApplication::processEvents();
    ASSERT_TRUE(panel.filter_edit_->hasFocus());
    panel.message_label_->clear();

    /** @brief 输入框使用的首次回车，必须在传播结束前保留焦点。 */
    QKeyEvent input_return(QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier);
    QApplication::sendEvent(panel.filter_edit_, &input_return);
    EXPECT_TRUE(panel.message_label_->text().isEmpty());
    EXPECT_TRUE(panel.pending_request_id_.empty());
    QApplication::processEvents();
    EXPECT_FALSE(panel.filter_edit_->hasFocus());

    /** @brief 焦点交还后的独立回车应触发保存，未连接服务时显示发送失败。 */
    QKeyEvent shortcut_return(QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier);
    QApplication::sendEvent(&panel, &shortcut_return);
    EXPECT_TRUE(panel.message_label_->text().contains("采集服务不可用"));
  }
};

}  // namespace fastumi_rviz_plugins

/** @brief 使用无界面 Qt 应用验证状态先确认、旧响应后到的真实回调顺序。 */
TEST(CollectionPanelRuntime, DelayedConfirmedResponsePreservesNewRequest)
{
  /** @brief QApplication 所需的参数数量。 */
  int argc = 1;
  /** @brief QApplication 使用的程序名缓冲区。 */
  char program_name[] = "test_collection_panel_runtime";
  /** @brief QApplication 使用的参数指针数组。 */
  char * argv[] = {program_name, nullptr};
  /** @brief 测试期间持有 Qt 应用，允许创建控件和启动计时器。 */
  QApplication application(argc, argv);
  fastumi_rviz_plugins::CollectionPanelTestPeer::expect_old_response_preserves_new_request();
  fastumi_rviz_plugins::CollectionPanelTestPeer::expect_small_minimum_height();
  fastumi_rviz_plugins::CollectionPanelTestPeer::expect_space_starts_collection();
  fastumi_rviz_plugins::CollectionPanelTestPeer::expect_filter_return_does_not_save();
}
