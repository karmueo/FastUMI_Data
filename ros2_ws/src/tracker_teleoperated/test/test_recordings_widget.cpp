/**
 * @file test_recordings_widget.cpp
 * @brief 验证历史记录筛选、后台刷新、目录打开及浏览状态保持。
 */
#include <atomic>
#include <filesystem>
#include <gtest/gtest.h>
#include <QApplication>
#include <QDir>
#include <QFile>
#include <QLabel>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>
#include <QTreeWidget>
#include "tracker_teleoperated/recordings_widget.hpp"

/** @brief 复用快捷键测试的 Qt 应用。 @return 测试进程中的应用实例。 */
QApplication * application();

namespace
{
/** @brief 在临时目录创建记录文件。 @param[in] root 任务根目录。 @param[in] episode 记录目录名。 @param[in] video 是否包含视频。 */
void createEpisode(const QString & root, const QString & episode, bool video = false)
{
  ASSERT_TRUE(QDir().mkpath(root + "/" + episode));
  /** @brief 文件内容无需读取，测试只验证已提交目录发现。 */
  QFile hdf5(root + "/" + episode + "/proprio.hdf5");
  ASSERT_TRUE(hdf5.open(QIODevice::WriteOnly));
  hdf5.write("fixture");
  hdf5.close();
  if (video) {
    /** @brief 可选视频文件。 */
    QFile mp4(root + "/" + episode + "/gripper.mp4");
    ASSERT_TRUE(mp4.open(QIODevice::WriteOnly));
  }
}

/** @brief 等待一次后台扫描结果，Qt 事件循环仍运行。 @param[in] widget 被测控件。 @return 是否在时限内收到结果。 */
bool refreshAndWait(tracker_teleoperated::RecordingsWidget & widget)
{
  /** @brief 在提交请求之前监听完成信号。 */
  QSignalSpy spy(&widget, &tracker_teleoperated::RecordingsWidget::refreshed);
  widget.requestRefresh();
  /** @brief 首次 setDirectory 与手动请求可能合并为两轮，等待后续结果落稳。 */
  const bool completed = spy.wait(3000);
  QTest::qWait(120);
  return completed;
}
}  // namespace

/** @brief 仅已提交且包含 HDF5 的数字记录参与倒序排序，视频可以缺省。 */
TEST(Recordings, ScanCommittedAndNumericOrder)
{
  /** @brief 完全隔离用户真实数据。 */
  QTemporaryDir temporary;
  createEpisode(temporary.path(), "episode_2");
  createEpisode(temporary.path(), "episode_10", true);
  createEpisode(temporary.path(), "episode_99999999999999999999999999");
  createEpisode(temporary.path(), ".episode_11.pending");
  createEpisode(temporary.path(), "episode_bad");
  QDir().mkpath(temporary.path() + "/episode_12");
  /** @brief 常规扫描未被取消。 */
  std::atomic_bool cancelled{false};
  /** @brief 首次历史扫描。 */
  const auto result = tracker_teleoperated::scanRecordings(temporary.path(), cancelled);
  EXPECT_TRUE(result.error.isEmpty());
  ASSERT_EQ(result.recordings.size(), 3);
  EXPECT_TRUE(result.recordings[0].path.endsWith("episode_99999999999999999999999999"));
  EXPECT_TRUE(result.recordings[1].path.endsWith("episode_10"));
  EXPECT_EQ(result.recordings[1].files.size(), 2);
  EXPECT_TRUE(result.recordings[2].path.endsWith("episode_2"));
  EXPECT_EQ(result.recordings[2].files, QStringList{"proprio.hdf5"});
  EXPECT_FALSE(result.recordings[2].modified.isEmpty());
  cancelled.store(true);
  EXPECT_TRUE(tracker_teleoperated::scanRecordings(temporary.path(), cancelled).recordings.isEmpty());
}

/** @brief 不存在的任务目录为空；权限和非目录错误必须明确报告。 */
TEST(Recordings, MissingAndUnreadableDirectory)
{
  /** @brief 测试数据根目录。 */
  QTemporaryDir temporary;
  /** @brief 本轮不取消扫描。 */
  std::atomic_bool cancelled{false};
  EXPECT_TRUE(tracker_teleoperated::scanRecordings(temporary.path() + "/missing", cancelled).error.isEmpty());
  createEpisode(temporary.path(), "episode_0");
  EXPECT_FALSE(tracker_teleoperated::scanRecordings(temporary.path() + "/episode_0/proprio.hdf5", cancelled).error.isEmpty());
  /** @brief 暂时移除任务目录权限，结束前恢复以便清理。 */
  const auto path = std::filesystem::path(temporary.path().toStdString());
  std::filesystem::permissions(path, std::filesystem::perms::none);
  /** @brief 保存扫描结果后立即恢复权限。 */
  const auto result = tracker_teleoperated::scanRecordings(temporary.path(), cancelled);
  std::filesystem::permissions(path, std::filesystem::perms::owner_all);
  EXPECT_FALSE(result.error.isEmpty());
}

/** @brief 真实双击事件使用编码正确的本地 URL，文件条目打开其父目录。 */
TEST(Recordings, OpenDirectoriesAndErrors)
{
  application();
  /** @brief 中文空格覆盖常见任务命名，不经 shell 转义。 */
  QTemporaryDir temporary;
  const auto root_path = temporary.path() + "/中文 任务";  ///< 当前任务绝对路径。
  createEpisode(root_path, "episode_3", true);
  QUrl opened;  ///< 捕获传给系统打开器的 URL。
  bool success = true;  ///< 控制系统打开器成功与失败。
  tracker_teleoperated::RecordingsWidget widget(nullptr, [&](const QUrl & url) {opened = url; return success;});  ///< 待测试控件。
  widget.resize(700, 350);
  widget.setDirectory(root_path);
  widget.show();
  ASSERT_TRUE(refreshAndWait(widget));
  auto * tree = widget.findChild<QTreeWidget *>("recordings_tree");  ///< 当前数据树。
  auto * status = widget.findChild<QLabel *>("recordings_status");  ///< 错误与计数标签。
  auto * root = tree->topLevelItem(0);  ///< 当前任务节点。
  ASSERT_EQ(root->childCount(), 1);
  auto * episode = root->child(0);  ///< 已保存的第三条记录。
  ASSERT_EQ(episode->childCount(), 2);
  QTest::mouseClick(tree->viewport(), Qt::LeftButton, Qt::NoModifier, tree->visualItemRect(episode).center());
  QTest::mouseDClick(tree->viewport(), Qt::LeftButton, Qt::NoModifier, tree->visualItemRect(episode).center());
  EXPECT_EQ(opened.toLocalFile(), root_path + "/episode_3");
  EXPECT_TRUE(opened.isLocalFile());
  EXPECT_FALSE(episode->isExpanded());
  tree->itemDoubleClicked(episode->child(0), 0);
  EXPECT_EQ(opened.toLocalFile(), root_path + "/episode_3");
  tree->itemDoubleClicked(root, 0);
  EXPECT_EQ(opened.toLocalFile(), root_path);
  success = false;
  tree->itemDoubleClicked(episode, 0);
  EXPECT_TRUE(status->text().contains("未能打开"));
  QDir(root_path + "/episode_3").removeRecursively();
  opened = QUrl();
  tree->itemDoubleClicked(episode, 0);
  EXPECT_TRUE(opened.isEmpty());
  EXPECT_TRUE(status->text().contains("目录不存在"));
}

/** @brief 刷新保留展开和选择；保存临时目录原子提交后才出现。 */
TEST(Recordings, RefreshPreservesSelectionAndCommitVisibility)
{
  application();
  QTemporaryDir temporary;  ///< 隔离任务目录。
  createEpisode(temporary.path(), "episode_2");
  tracker_teleoperated::RecordingsWidget widget;  ///< 后台扫描与界面更新的实际控件。
  widget.setDirectory(temporary.path());
  ASSERT_TRUE(refreshAndWait(widget));
  auto * tree = widget.findChild<QTreeWidget *>("recordings_tree");  ///< 记录列表。
  auto * episode = tree->topLevelItem(0)->child(0);  ///< 选择已有文件以测试刷新恢复。
  episode->setExpanded(true);
  tree->setCurrentItem(episode->child(0));
  createEpisode(temporary.path(), ".episode_3.pending", true);
  ASSERT_TRUE(refreshAndWait(widget));
  EXPECT_EQ(tree->topLevelItem(0)->childCount(), 1);
  ASSERT_TRUE(QDir().rename(temporary.path() + "/.episode_3.pending", temporary.path() + "/episode_3"));
  ASSERT_TRUE(refreshAndWait(widget));
  ASSERT_EQ(tree->topLevelItem(0)->childCount(), 2);
  EXPECT_EQ(tree->topLevelItem(0)->child(0)->text(0), "episode_3");
  EXPECT_TRUE(tree->topLevelItem(0)->child(1)->isExpanded());
  ASSERT_NE(tree->currentItem(), nullptr);
  EXPECT_EQ(tree->currentItem()->data(0, Qt::UserRole).toString(), temporary.path() + "/episode_2/proprio.hdf5");
  createEpisode(temporary.path(), ".episode_4.failed");
  QDir(temporary.path() + "/.episode_4.failed").removeRecursively();
  ASSERT_TRUE(refreshAndWait(widget));
  EXPECT_EQ(tree->topLevelItem(0)->childCount(), 2);
}

/** @brief 可见时周期发现新增记录，无需记录节点持续运行。 */
TEST(Recordings, VisiblePeriodicRefresh)
{
  application();
  QTemporaryDir temporary;  ///< 初始为空的任务目录。
  tracker_teleoperated::RecordingsWidget widget;  ///< 可见的数据浏览页签。
  widget.setDirectory(temporary.path());
  widget.show();
  ASSERT_TRUE(refreshAndWait(widget));
  createEpisode(temporary.path(), "episode_0");
  auto * tree = widget.findChild<QTreeWidget *>("recordings_tree");  ///< 定时刷新的目标树。
  for (int attempt = 0; attempt < 60 && tree->topLevelItem(0)->childCount() == 0; ++attempt) {QTest::qWait(50);}  ///< 等待最多三秒。
  EXPECT_EQ(tree->topLevelItem(0)->childCount(), 1);
}

/** @brief 切换目录丢弃旧结果，后台扫描进行时销毁也安全回收。 */
TEST(Recordings, DirectoryChangeAndUnload)
{
  application();
  QTemporaryDir first;  ///< 首个扫描目录。
  QTemporaryDir second;  ///< 用户切换后的目标目录。
  createEpisode(first.path(), "episode_1");
  createEpisode(second.path(), "episode_7");
  tracker_teleoperated::RecordingsWidget widget;  ///< 异步结果不得串入旧目录。
  widget.setDirectory(first.path());
  widget.setDirectory(second.path());
  ASSERT_TRUE(refreshAndWait(widget));
  auto * tree = widget.findChild<QTreeWidget *>("recordings_tree");  ///< 当前任务树。
  ASSERT_EQ(tree->topLevelItemCount(), 1);
  EXPECT_EQ(tree->topLevelItem(0)->data(0, Qt::UserRole).toString(), second.path());
  ASSERT_EQ(tree->topLevelItem(0)->childCount(), 1);
  EXPECT_EQ(tree->topLevelItem(0)->child(0)->text(0), "episode_7");
  widget.setDirectory("");
  EXPECT_EQ(tree->topLevelItemCount(), 0);
  for (int attempt = 0; attempt < 10; ++attempt) {  ///< 重复模拟 RViz 添加及卸载面板。
    tracker_teleoperated::RecordingsWidget transient;  ///< 析构时必须等待自身后台代码结束。
    transient.setDirectory(first.path());
    transient.requestRefresh();
  }
  QApplication::processEvents();
}
