/** @file test_panel.cpp
 * @brief 使用真实 Qt 事件验证面板快捷键焦点、自动重复与文本编辑保护。
 */
#include <gtest/gtest.h>
#include <QApplication>
#include <QComboBox>
#include <QDialog>
#include <QDockWidget>
#include <QKeyEvent>
#include <QLineEdit>
#include <QMainWindow>
#include <QMenu>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QShortcut>
#include <QSignalSpy>
#include <QSpinBox>
#include <QTest>
#include "tracker_teleoperated/teleop_panel.hpp"

/** @brief 创建测试进程共享的 Qt 应用实例。 */
QApplication * application()
{
  /** @brief Qt 必须保留参数内存到进程结束。 */
  static int argc = 1;
  static char name[] = "test_panel";
  static char * argv[] = {name, nullptr};
  static QApplication app(argc, argv);
  return &app;
}

/** @brief RViz 窗口内每个物理按键只产生一个操作，按住按键不重复触发。 */
TEST(TeleopPanel, ShortcutsAndRepeat)
{
  application();
  tracker_teleoperated::TeleopPanel panel;
  QSignalSpy spy(&panel, SIGNAL(actionRequested(QString)));
  panel.show();
  panel.activateWindow();
  panel.setFocus();
  QApplication::processEvents();
  QTest::keyClick(&panel, Qt::Key_Space);
  ASSERT_EQ(spy.count(), 1);
  EXPECT_EQ(spy.takeFirst().at(0).toString(), "toggle");
  QKeyEvent repeat(QEvent::KeyPress, Qt::Key_A, Qt::NoModifier, "a", true);
  QApplication::sendEvent(&panel, &repeat);
  EXPECT_EQ(spy.count(), 0);
  QTest::keyClick(&panel, Qt::Key_C, Qt::ShiftModifier);
  ASSERT_EQ(spy.count(), 1);
  EXPECT_EQ(spy.takeFirst().at(0).toString(), "calibrate");
  QTest::keyClick(&panel, Qt::Key_C, Qt::ControlModifier);
  EXPECT_EQ(spy.count(), 0);
}

/** @brief 输入框与其他顶层窗口保持原生按键行为，不触发运动或录制。 */
TEST(TeleopPanel, EditorsAndOutsideFocus)
{
  application();
  tracker_teleoperated::TeleopPanel panel;
  QLineEdit edit(&panel);
  QLineEdit outside;
  QSignalSpy spy(&panel, SIGNAL(actionRequested(QString)));
  panel.show();
  edit.show();
  panel.activateWindow();
  edit.setFocus();
  QApplication::processEvents();
  QTest::keyClicks(&edit, "ach sbq");
  EXPECT_EQ(edit.text(), "ach sbq");
  EXPECT_EQ(spy.count(), 0);
  outside.show();
  outside.activateWindow();
  outside.setFocus();
  QApplication::processEvents();
  QTest::keyClicks(&outside, "a");
  EXPECT_EQ(spy.count(), 0);
}

/** @brief 三维视图等普通 RViz 控件无需先选中插件，隐藏插件后也保持快捷键。 */
TEST(TeleopPanel, WholeRvizWindowAndHiddenPanel)
{
  application();
  QMainWindow rviz;  ///< 模拟 RViz 主窗口。
  auto * view = new QWidget(&rviz);  ///< 模拟三维视图或其他普通面板。
  view->setFocusPolicy(Qt::StrongFocus);
  rviz.setCentralWidget(view);
  tracker_teleoperated::TeleopPanel panel(&rviz);  ///< 安装应用级过滤器的插件实例。
  QSignalSpy spy(&panel, SIGNAL(actionRequested(QString)));
  rviz.show();
  view->setFocus();
  rviz.activateWindow();
  QApplication::processEvents();
  QTest::keyClick(view, Qt::Key_H);
  ASSERT_EQ(spy.count(), 1);
  EXPECT_EQ(spy.takeFirst().at(0).toString(), "home");
  panel.hide();
  view->setFocus();
  QTest::keyClick(view, Qt::Key_A);
  ASSERT_EQ(spy.count(), 1);
  EXPECT_EQ(spy.takeFirst().at(0).toString(), "record");
}

/** @brief 浮动停靠窗口仍属于同一 RViz，其普通控件可直接使用快捷键。 */
TEST(TeleopPanel, FloatingDockBelongsToRviz)
{
  application();
  QMainWindow rviz;  ///< 模拟 RViz 主窗口。
  tracker_teleoperated::TeleopPanel panel(&rviz);  ///< 窗口级快捷键拥有者。
  QSignalSpy spy(&panel, SIGNAL(actionRequested(QString)));
  auto * dock = new QDockWidget("浮动显示", &rviz);  ///< 模拟浮动视频或 Displays 面板。
  auto * content = new QWidget(dock);  ///< 浮动面板内的普通焦点控件。
  content->setFocusPolicy(Qt::StrongFocus);
  dock->setWidget(content);
  rviz.addDockWidget(Qt::RightDockWidgetArea, dock);
  rviz.show();
  dock->setFloating(true);
  dock->show();
  dock->activateWindow();
  content->setFocus();
  QApplication::processEvents();
  QTest::keyClick(content, Qt::Key_Space);
  ASSERT_EQ(spy.count(), 1);
  EXPECT_EQ(spy.takeFirst().at(0).toString(), "toggle");
}

/** @brief 遥操按键抢占 RViz 同键快捷键，不可执行时也不会落回原快捷键。 */
TEST(TeleopPanel, OverridesConflictingRvizShortcut)
{
  application();
  QMainWindow rviz;  ///< 模拟带原有快捷键的 RViz 主窗口。
  auto * view = new QWidget(&rviz);  ///< 接收键盘事件的普通控件。
  view->setFocusPolicy(Qt::StrongFocus);
  rviz.setCentralWidget(view);
  tracker_teleoperated::TeleopPanel panel(&rviz);  ///< 优先消费遥操按键的插件。
  QShortcut conflicting(QKeySequence(Qt::Key_A), &rviz);  ///< 模拟 RViz 的冲突快捷键。
  QSignalSpy action_spy(&panel, SIGNAL(actionRequested(QString)));
  QSignalSpy shortcut_spy(&conflicting, SIGNAL(activated()));
  rviz.show();
  rviz.activateWindow();
  view->setFocus();
  QApplication::processEvents();
  QTest::keyClick(view, Qt::Key_A);
  EXPECT_EQ(action_spy.count(), 1);
  EXPECT_EQ(shortcut_spy.count(), 0);
}

/** @brief 可编辑控件、下拉框和对话框保留原行为，只读诊断区仍可使用快捷键。 */
TEST(TeleopPanel, ProtectsEditorsAndDialogsButAllowsReadonlyDetails)
{
  application();
  QMainWindow rviz;  ///< 模拟 RViz 主窗口。
  tracker_teleoperated::TeleopPanel panel(&rviz);  ///< 被测插件实例。
  QSignalSpy spy(&panel, SIGNAL(actionRequested(QString)));
  auto * edit = new QLineEdit(&rviz);  ///< 可编辑文本输入。
  rviz.setCentralWidget(edit);
  rviz.show();
  panel.show();
  rviz.activateWindow();
  edit->setFocus();
  QApplication::processEvents();
  QTest::keyClicks(edit, "ach sbq");
  EXPECT_EQ(edit->text(), "ach sbq");
  EXPECT_EQ(spy.count(), 0);
  auto * details = panel.findChild<QPlainTextEdit *>();  ///< 面板内只读诊断区域。
  ASSERT_NE(details, nullptr);
  ASSERT_TRUE(details->isReadOnly());
  details->setFocus();
  QTest::keyClick(details, Qt::Key_C);
  ASSERT_EQ(spy.count(), 1);
  EXPECT_EQ(spy.takeFirst().at(0).toString(), "calibrate");
  QDialog dialog(&rviz);  ///< 属于 RViz 的模态对话框。
  QPushButton dialog_button("确认", &dialog);  ///< 对话框原生操作按钮。
  dialog.setModal(true);
  dialog.show();
  dialog.activateWindow();
  dialog_button.setFocus();
  QApplication::processEvents();
  QTest::keyClick(&dialog_button, Qt::Key_Space);
  EXPECT_EQ(spy.count(), 0);
  dialog.close();
  QSpinBox spin(&rviz);  ///< RViz 参数面板中的数值输入控件。
  spin.show();
  spin.setFocus();
  rviz.activateWindow();
  QApplication::processEvents();
  QTest::keyClick(&spin, Qt::Key_A);
  EXPECT_EQ(spy.count(), 0);
  QMenu menu(&rviz);  ///< RViz 主窗口弹出的菜单。
  menu.addAction("测试操作");
  menu.popup(rviz.mapToGlobal(QPoint(20, 20)));
  QApplication::processEvents();
  QTest::keyClick(&menu, Qt::Key_Q);
  EXPECT_EQ(spy.count(), 0);
  menu.close();
}

/** @brief 多个面板实例只消费一次，最后实例卸载后前一实例继续接管。 */
TEST(TeleopPanel, MultipleInstancesDoNotDuplicate)
{
  application();
  QMainWindow rviz;  ///< 共享两个插件实例的 RViz 主窗口。
  auto * view = new QWidget(&rviz);  ///< 普通焦点控件。
  view->setFocusPolicy(Qt::StrongFocus);
  rviz.setCentralWidget(view);
  tracker_teleoperated::TeleopPanel first(&rviz);  ///< 较早安装的事件过滤器。
  QSignalSpy first_spy(&first, SIGNAL(actionRequested(QString)));
  rviz.show();
  rviz.activateWindow();
  view->setFocus();
  {
    tracker_teleoperated::TeleopPanel second(&rviz);  ///< 后安装且优先接收事件的实例。
    QSignalSpy second_spy(&second, SIGNAL(actionRequested(QString)));
    QApplication::processEvents();
    QTest::keyClick(view, Qt::Key_S);
    EXPECT_EQ(first_spy.count() + second_spy.count(), 1);
  }
  /** @brief 第一实例在前一轮可能接管事件，记录其当前计数。 */
  const int first_count = first_spy.count();
  view->setFocus();
  QTest::keyClick(view, Qt::Key_H);
  ASSERT_EQ(first_spy.count(), first_count + 1);
  EXPECT_EQ(first_spy.at(first_spy.count() - 1).at(0).toString(), "home");
}

/** @brief 插件卸载会移除过滤器，使 RViz 原有快捷键恢复。 */
TEST(TeleopPanel, UnloadRestoresRvizShortcut)
{
  application();
  QMainWindow rviz;  ///< 模拟 RViz 主窗口。
  auto * view = new QWidget(&rviz);  ///< 快捷键上下文内的普通控件。
  view->setFocusPolicy(Qt::StrongFocus);
  rviz.setCentralWidget(view);
  QShortcut original(QKeySequence(Qt::Key_Q), &rviz);  ///< 模拟 RViz 原有快捷键。
  QSignalSpy shortcut_spy(&original, SIGNAL(activated()));
  rviz.show();
  rviz.activateWindow();
  view->setFocus();
  {
    tracker_teleoperated::TeleopPanel panel(&rviz);  ///< 临时加载的插件。
    QSignalSpy action_spy(&panel, SIGNAL(actionRequested(QString)));
    QApplication::processEvents();
    QTest::keyClick(view, Qt::Key_Q);
    EXPECT_EQ(action_spy.count(), 1);
    EXPECT_EQ(shortcut_spy.count(), 0);
  }
  view->setFocus();
  QTest::keyClick(view, Qt::Key_Q);
  EXPECT_EQ(shortcut_spy.count(), 1);
}

/** @brief 视频下拉框及弹出列表获得焦点时不触发遥操快捷键。 */
TEST(TeleopPanel, VideoComboKeepsNativeKeys)
{
  application();
  tracker_teleoperated::TeleopPanel panel;
  QSignalSpy spy(&panel, SIGNAL(actionRequested(QString)));
  panel.show();
  panel.activateWindow();
  auto * combo = panel.findChild<QComboBox *>("wrist_video_source");  ///< 面板内真实视频源下拉框。
  ASSERT_NE(combo, nullptr);
  combo->addItems({"/camera/a", "/camera/b"});
  combo->setEnabled(true);
  combo->setFocus();
  QApplication::processEvents();
  QTest::keyClick(combo, Qt::Key_Space);
  QTest::keyClick(combo, Qt::Key_A);
  QTest::keyClick(combo, Qt::Key_Down);
  EXPECT_EQ(spy.count(), 0);
}
