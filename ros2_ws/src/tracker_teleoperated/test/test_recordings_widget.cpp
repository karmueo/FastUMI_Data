/**
 * @file test_recordings_widget.cpp
 * @brief 使用真实 Qt 菜单和 ROS 2 测试服务验证已保存记录的单条删除流程。
 */
#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <QApplication>
#include <QLabel>
#include <QMenu>
#include <QMessageBox>
#include <QPushButton>
#include <QTest>
#include <QTimer>
#include <QTreeWidget>
#include <rclcpp/rclcpp.hpp>
#include <fastumi_interfaces/srv/delete_recording.hpp>
#include <fastumi_interfaces/srv/list_recordings.hpp>

#include "tracker_teleoperated/recordings_widget.hpp"

namespace
{
/** @brief 创建测试进程共享的 Qt 应用实例。 @return Qt 应用。 */
QApplication * application()
{
  static int argc = 1;  ///< Qt 保留到进程结束的参数数量。
  static char name[] = "test_recordings_widget";  ///< Qt 应用名称。
  static char * argv[] = {name, nullptr};  ///< Qt 保留到进程结束的参数数组。
  static QApplication app(argc, argv);  ///< 测试进程唯一应用实例。
  return &app;
}

/** @brief 确保测试进程共享的 ROS context 已初始化。 */
void initializeRos()
{
  if (!rclcpp::ok()) {
    int argc = 0;  ///< 测试不向 ROS 传递命令行参数。
    char ** argv = nullptr;  ///< 空 ROS 参数数组。
    rclcpp::init(argc, argv);
  }
}

/**
 * @brief 在处理 Qt 和 ROS 事件的同时等待断言条件。
 * @param[in] executor 测试节点执行器。
 * @param[in] predicate 完成条件。
 * @param[in] timeout_ms 最长等待毫秒数。
 * @return 条件是否在期限内成立。
 */
bool waitUntil(
  rclcpp::executors::SingleThreadedExecutor & executor,
  const std::function<bool()> & predicate, int timeout_ms = 3000)
{
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::milliseconds(timeout_ms);  ///< 本轮等待期限。
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(std::chrono::milliseconds(5));
    QApplication::processEvents();
    if (predicate()) {return true;}
    QTest::qWait(5);
  }
  return predicate();
}

/**
 * @brief 在右键菜单中选择删除，并点击指定确认按钮。
 * @param[in] tree 记录树。
 * @param[in] item 目标记录叶子节点。
 * @param[in] confirmation 确认框按钮。
 * @return 是否成功触发菜单和确认框。
 */
bool chooseDelete(
  QTreeWidget * tree, QTreeWidgetItem * item,
  QMessageBox::StandardButton confirmation)
{
  bool menu_seen = false;  ///< 是否找到并触发删除菜单项。
  bool confirmation_seen = false;  ///< 是否找到并点击确认框按钮。
  QTimer::singleShot(20, [&menu_seen]() {
    for (auto * top_level : QApplication::topLevelWidgets()) {
      auto * menu = qobject_cast<QMenu *>(top_level);  ///< offscreen 平台中的可见右键菜单。
      if (!menu || !menu->isVisible() || menu->actions().isEmpty()) {continue;}
      menu_seen = true;
      menu->setActiveAction(menu->actions().front());
      QTest::keyClick(menu, Qt::Key_Return);
      return;
    }
  });
  QTimer::singleShot(100, [confirmation, &confirmation_seen]() {
    for (auto * top_level : QApplication::topLevelWidgets()) {
      auto * box = qobject_cast<QMessageBox *>(top_level);  ///< offscreen 平台中的可见确认框。
      if (!box || !box->isVisible()) {continue;}
      auto * button = box->button(confirmation);  ///< 测试选择的确认按钮。
      if (!button) {return;}
      confirmation_seen = true;
      button->click();
      return;
    }
  });
  QTimer::singleShot(500, []() {
    for (auto * top_level : QApplication::topLevelWidgets()) {
      if (auto * menu = qobject_cast<QMenu *>(top_level); menu && menu->isVisible()) {
        menu->close();
      }
      if (auto * box = qobject_cast<QMessageBox *>(top_level); box && box->isVisible()) {
        box->reject();
      }
    }
  });
  const auto position = tree->visualItemRect(item).center();  ///< 目标记录在视口中的位置。
  QMetaObject::invokeMethod(tree, "customContextMenuRequested", Qt::DirectConnection,
    Q_ARG(QPoint, position));
  return menu_seen && confirmation_seen;
}

/**
 * @brief 打开记录右键菜单并读取删除动作可用状态。
 * @param[in] tree 记录树。
 * @param[in] item 目标记录叶子节点。
 * @param[out] menu_seen 是否成功观察到菜单。
 * @return 删除动作是否可用。
 */
bool deleteActionEnabled(QTreeWidget * tree, QTreeWidgetItem * item, bool & menu_seen)
{
  bool action_enabled = false;  ///< 右键删除动作的可用状态。
  menu_seen = false;
  QTimer::singleShot(20, [&menu_seen, &action_enabled]() {
    for (auto * top_level : QApplication::topLevelWidgets()) {
      auto * menu = qobject_cast<QMenu *>(top_level);  ///< 当前可见的记录右键菜单。
      if (!menu || !menu->isVisible() || menu->actions().isEmpty()) {continue;}
      menu_seen = true;
      action_enabled = menu->actions().front()->isEnabled();
      menu->close();
      return;
    }
  });
  QTimer::singleShot(500, []() {
    for (auto * top_level : QApplication::topLevelWidgets()) {
      if (auto * menu = qobject_cast<QMenu *>(top_level); menu && menu->isVisible()) {
        menu->close();
      }
    }
  });
  const auto position = tree->visualItemRect(item).center();  ///< 目标记录在视口中的位置。
  QMetaObject::invokeMethod(tree, "customContextMenuRequested", Qt::DirectConnection,
    Q_ARG(QPoint, position));
  return action_enabled;
}

/** @brief 构造一条远端已保存记录摘要。 @return 固定测试记录。 */
fastumi_interfaces::msg::RecordingInfo recording()
{
  fastumi_interfaces::msg::RecordingInfo value;  ///< 测试记录摘要。
  value.recording_id = "recording-1";
  value.dir_name = "task";
  value.name = "sample";
  value.relative_path = "task/sample/episode_1";
  value.duration = 1.5;
  value.joint_samples = 10;
  value.action_samples = 9;
  value.image_frames = 8;
  return value;
}

/** @brief 管理测试服务、权威列表和收到的删除请求。 */
class RecordingServices
{
public:
  /** @brief 创建隔离命名空间中的列表与删除服务。 */
  RecordingServices()
  {
    static unsigned counter = 0;  ///< 避免不同用例节点重名。
    prefix = "/test_recordings_" + std::to_string(++counter);
    node = std::make_shared<rclcpp::Node>("recordings_services_" + std::to_string(counter));
    records = {recording()};
    list_service = node->create_service<fastumi_interfaces::srv::ListRecordings>(
      prefix + "/list",
      [this](
        const std::shared_ptr<fastumi_interfaces::srv::ListRecordings::Request>,
        std::shared_ptr<fastumi_interfaces::srv::ListRecordings::Response> response) {
        ++list_requests;
        response->success = true;
        response->code = "OK";
        response->recordings = records;
        response->total = records.size();
      });
    delete_service = node->create_service<fastumi_interfaces::srv::DeleteRecording>(
      prefix + "/delete",
      [this](
        const std::shared_ptr<fastumi_interfaces::srv::DeleteRecording::Request> request,
        std::shared_ptr<fastumi_interfaces::srv::DeleteRecording::Response> response) {
        ++delete_requests;
        deleted_recording_id = request->recording_id;
        response->success = delete_succeeds;
        response->recording_id = request->recording_id;
        response->code = delete_succeeds ? "DELETED" : "DELETE_FAILED";
        response->message = delete_succeeds ? "删除成功" : "测试删除失败";
        if (delete_succeeds) {
          records.erase(std::remove_if(records.begin(), records.end(),
            [&request](const auto & value) {return value.recording_id == request->recording_id;}),
            records.end());
        }
      });
    executor.add_node(node);
  }

  /** @brief 从执行器移除测试节点。 */
  ~RecordingServices() {executor.remove_node(node);}

  std::string prefix;  ///< 测试服务前缀。
  rclcpp::Node::SharedPtr node;  ///< 同时承载测试服务和控件客户端的节点。
  rclcpp::executors::SingleThreadedExecutor executor;  ///< GUI 线程轮询的测试执行器。
  rclcpp::Service<fastumi_interfaces::srv::ListRecordings>::SharedPtr list_service;  ///< 列表服务。
  rclcpp::Service<fastumi_interfaces::srv::DeleteRecording>::SharedPtr delete_service;  ///< 删除服务。
  std::vector<fastumi_interfaces::msg::RecordingInfo> records;  ///< 测试端权威记录。
  std::string deleted_recording_id;  ///< 最近删除请求携带的 UUID。
  int list_requests{0};  ///< 已收到的列表请求数。
  int delete_requests{0};  ///< 已收到的删除请求数。
  bool delete_succeeds{true};  ///< 控制删除服务响应。
};

/**
 * @brief 初始化控件并等待首轮列表完成。
 * @param[in] services 测试服务。
 * @param[in,out] widget 被测控件。
 * @return 数据树。
 */
QTreeWidget * initializeWidget(
  RecordingServices & services, tracker_teleoperated::RecordingsWidget & widget)
{
  widget.resize(800, 400);
  widget.show();
  widget.initialize(services.node, services.prefix);
  /** @brief 等待服务发现后重试一次，覆盖初始化瞬间尚未发现服务的情况。 */
  EXPECT_TRUE(waitUntil(services.executor, [&services]() {
    return services.list_service != nullptr && services.delete_service != nullptr;
  }, 500));
  widget.requestRefresh();
  auto * tree = widget.findChild<QTreeWidget *>("recordings_tree");  ///< 被测记录树。
  EXPECT_NE(tree, nullptr);
  EXPECT_TRUE(waitUntil(services.executor, [tree]() {
    return tree && tree->topLevelItemCount() == 1 && tree->topLevelItem(0)->childCount() == 1;
  }));
  return tree;
}
}  // namespace

/** @brief 确认删除发送稳定 UUID，成功后自动刷新权威列表。 */
TEST(RecordingsWidget, DeleteConfirmedRecordingAndRefresh)
{
  application();
  initializeRos();
  RecordingServices services;  ///< 列表和删除测试服务。
  tracker_teleoperated::RecordingsWidget widget;  ///< 被测已保存数据控件。
  auto * tree = initializeWidget(services, widget);  ///< 已加载一条记录的数据树。
  ASSERT_NE(tree, nullptr);
  auto * item = tree->topLevelItem(0)->child(0);  ///< 待删除记录。
  ASSERT_TRUE(chooseDelete(tree, item, QMessageBox::Yes));
  EXPECT_EQ(tree->currentItem(), item);
  bool pending_menu_seen = false;  ///< 删除请求在途时是否显示右键菜单。
  EXPECT_FALSE(deleteActionEnabled(tree, item, pending_menu_seen));
  EXPECT_TRUE(pending_menu_seen);
  EXPECT_TRUE(waitUntil(services.executor, [&services]() {return services.delete_requests == 1;}));
  EXPECT_EQ(services.deleted_recording_id, "recording-1");
  EXPECT_TRUE(waitUntil(services.executor, [tree]() {return tree->topLevelItemCount() == 0;}));
  EXPECT_EQ(tree->currentItem(), nullptr);
  auto * status = widget.findChild<QLabel *>("recordings_status");  ///< 列表状态提示。
  ASSERT_NE(status, nullptr);
  EXPECT_TRUE(status->text().contains("暂无已保存数据"));
  EXPECT_GE(services.list_requests, 2);
}

/** @brief 取消确认、分组节点和空白区域都不会调用删除服务。 */
TEST(RecordingsWidget, IgnoreCancelledGroupAndBlankDeletion)
{
  application();
  initializeRos();
  RecordingServices services;  ///< 列表和删除测试服务。
  tracker_teleoperated::RecordingsWidget widget;  ///< 被测已保存数据控件。
  auto * tree = initializeWidget(services, widget);  ///< 已加载一条记录的数据树。
  ASSERT_NE(tree, nullptr);
  auto * root = tree->topLevelItem(0);  ///< 不可删除的任务分组。
  auto * item = root->child(0);  ///< 可弹出确认框的记录节点。
  widget.requestRefresh();
  bool refresh_menu_seen = false;  ///< 列表刷新期间是否显示右键菜单。
  EXPECT_FALSE(deleteActionEnabled(tree, item, refresh_menu_seen));
  EXPECT_TRUE(refresh_menu_seen);
  auto * refresh = widget.findChild<QPushButton *>("recordings_refresh");  ///< 列表刷新完成标志。
  ASSERT_NE(refresh, nullptr);
  EXPECT_TRUE(waitUntil(services.executor, [refresh]() {return refresh->isEnabled();}));
  root = tree->topLevelItem(0);
  item = tree->topLevelItem(0)->child(0);
  ASSERT_TRUE(chooseDelete(tree, item, QMessageBox::Cancel));
  QMetaObject::invokeMethod(tree, "customContextMenuRequested", Qt::DirectConnection,
    Q_ARG(QPoint, tree->visualItemRect(root).center()));
  QMetaObject::invokeMethod(tree, "customContextMenuRequested", Qt::DirectConnection,
    Q_ARG(QPoint, QPoint(tree->viewport()->width() - 2, tree->viewport()->height() - 2)));
  services.executor.spin_some(std::chrono::milliseconds(20));
  EXPECT_EQ(services.delete_requests, 0);
  EXPECT_EQ(tree->topLevelItemCount(), 1);
}

/** @brief 菜单打开期间列表节点失效时，不访问旧节点或发送删除请求。 */
TEST(RecordingsWidget, IgnoreRecordRemovedWhileMenuIsOpen)
{
  application();
  initializeRos();
  RecordingServices services;  ///< 列表和删除测试服务。
  tracker_teleoperated::RecordingsWidget widget;  ///< 被测已保存数据控件。
  auto * tree = initializeWidget(services, widget);  ///< 已加载一条记录的数据树。
  ASSERT_NE(tree, nullptr);
  bool menu_seen = false;  ///< 是否观察到记录右键菜单。
  QTimer::singleShot(20, &widget, [tree, &menu_seen]() {
    tree->clear();
    for (auto * top_level : QApplication::topLevelWidgets()) {
      auto * menu = qobject_cast<QMenu *>(top_level);  ///< 节点销毁后仍打开的菜单。
      if (!menu || !menu->isVisible() || menu->actions().isEmpty()) {continue;}
      menu_seen = true;
      menu->setActiveAction(menu->actions().front());
      QTest::keyClick(menu, Qt::Key_Return);
      return;
    }
  });
  QTimer::singleShot(500, &widget, []() {
    for (auto * top_level : QApplication::topLevelWidgets()) {
      if (auto * menu = qobject_cast<QMenu *>(top_level); menu && menu->isVisible()) {
        menu->close();
      }
    }
  });
  auto * item = tree->topLevelItem(0)->child(0);  ///< 菜单打开时命中的原始节点。
  const auto position = tree->visualItemRect(item).center();  ///< 原始节点的菜单位置。
  QMetaObject::invokeMethod(tree, "customContextMenuRequested", Qt::DirectConnection,
    Q_ARG(QPoint, position));
  EXPECT_TRUE(menu_seen);
  EXPECT_EQ(services.delete_requests, 0);
}

/** @brief 删除失败保留现有列表并显示服务端错误。 */
TEST(RecordingsWidget, PreserveListOnDeleteFailure)
{
  application();
  initializeRos();
  RecordingServices services;  ///< 返回失败的删除测试服务。
  services.delete_succeeds = false;
  tracker_teleoperated::RecordingsWidget widget;  ///< 被测已保存数据控件。
  auto * tree = initializeWidget(services, widget);  ///< 已加载一条记录的数据树。
  ASSERT_NE(tree, nullptr);
  ASSERT_TRUE(chooseDelete(tree, tree->topLevelItem(0)->child(0), QMessageBox::Yes));
  auto * status = widget.findChild<QLabel *>("recordings_status");  ///< 删除错误提示。
  ASSERT_NE(status, nullptr);
  EXPECT_TRUE(waitUntil(services.executor, [status]() {
    return status->text().contains("测试删除失败");
  }));
  EXPECT_EQ(tree->topLevelItemCount(), 1);
  EXPECT_EQ(tree->topLevelItem(0)->childCount(), 1);
  auto * refresh = widget.findChild<QPushButton *>("recordings_refresh");  ///< 失败后恢复的刷新按钮。
  ASSERT_NE(refresh, nullptr);
  EXPECT_TRUE(refresh->isEnabled());
}

/** @brief 删除服务离线时仍显示记录菜单，但删除动作保持禁用。 */
TEST(RecordingsWidget, DisableDeletionWhenServiceUnavailable)
{
  application();
  initializeRos();
  RecordingServices services;  ///< 初始可用且随后下线的删除测试服务。
  tracker_teleoperated::RecordingsWidget widget;  ///< 被测已保存数据控件。
  auto * tree = initializeWidget(services, widget);  ///< 已加载一条记录的数据树。
  ASSERT_NE(tree, nullptr);
  services.delete_service.reset();
  for (int attempt = 0; attempt < 20; ++attempt) {
    services.executor.spin_some(std::chrono::milliseconds(5));
    QTest::qWait(10);
  }
  bool menu_seen = false;  ///< 服务离线时是否仍可查看右键菜单。
  EXPECT_FALSE(deleteActionEnabled(tree, tree->topLevelItem(0)->child(0), menu_seen));
  EXPECT_EQ(services.delete_requests, 0);
}
