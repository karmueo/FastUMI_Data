/**
 * @file recordings_widget.hpp
 * @brief 声明本机已保存记录的异步目录扫描与树形浏览控件。
 */
#ifndef TRACKER_TELEOPERATED__RECORDINGS_WIDGET_HPP_
#define TRACKER_TELEOPERATED__RECORDINGS_WIDGET_HPP_

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <QStringList>
#include <QUrl>
#include <QVector>
#include <QWidget>

class QLabel;
class QPushButton;
class QShowEvent;
class QTimer;
class QTreeWidget;
class QTreeWidgetItem;

namespace tracker_teleoperated
{
/** @brief 一条已经完成目录提交的录制，仅保存展示所需元数据。 */
struct SavedRecording
{
  QString path;  ///< Episode 绝对路径。
  QString modified;  ///< HDF5 修改时间，按本地时区显示。
  QStringList files;  ///< 已存在的数据文件名称。
};

/** @brief 一次目录扫描的值语义结果，不包含 GUI 对象。 */
struct RecordingScan
{
  QString directory;  ///< 本次扫描的任务绝对目录。
  QVector<SavedRecording> recordings;  ///< 按 episode 数字编号倒序排列的记录。
  QString error;  ///< 目录无法读取时的具体错误，正常为空。
};

/**
 * @brief 扫描任务目录直属的已完成记录，不解析 HDF5 或视频内容。
 * @param[in] directory 任务绝对目录。
 * @param[in] cancelled 调用方控制的取消标记，在目录项之间检查。
 * @return 已保存记录或文件系统错误；不存在的目录返回空列表。
 * @note 仅在后台或测试线程调用；文件系统调用本身不可中断。
 */
RecordingScan scanRecordings(const QString & directory, const std::atomic_bool & cancelled);

/** @brief 独占单个后台扫描任务，所有控件更新和打开目录操作均在 GUI 线程执行。 */
class RecordingsWidget : public QWidget
{
  Q_OBJECT
public:
  /** @brief 可注入的系统目录打开器，便于测试路径和失败提示。 */
  using DirectoryOpener = std::function<bool(const QUrl &)>;
  /** @brief 初始化树与刷新定时器。 @param[in] parent 父控件。 @param[in] opener 为空时使用系统文件管理器。 */
  explicit RecordingsWidget(QWidget * parent = nullptr, DirectoryOpener opener = {});
  /** @brief 取消并等待后台扫描退出，确保插件卸载后没有任务执行其代码。 */
  ~RecordingsWidget() override;
  /** @brief 设置当前任务目录；变化时清除旧树并异步扫描。 @param[in] directory 管理器提供的绝对目录。 */
  void setDirectory(const QString & directory);
  /** @brief 请求异步刷新，扫描期间的请求合并为一次后续刷新。 */
  void requestRefresh();

Q_SIGNALS:
  /** @brief 当前目录的一次扫描结果已经更新到界面。 */
  void refreshed();

protected:
  /** @brief 页签变为可见时刷新历史记录。 @param[in] event Qt 显示事件。 */
  void showEvent(QShowEvent * event) override;

private:
  /** @brief 非阻塞检查后台结果并在 GUI 线程应用，过时结果直接丢弃。 */
  void collectResult();
  /** @brief 重建树并恢复选择、展开和滚动状态。 @param[in] result 当前目录的扫描结果。 */
  void displayResult(const RecordingScan & result);
  /** @brief 打开条目对应目录，目录已删除或系统调用失败时提示。 @param[in] item 被双击的树项。 */
  void openItem(QTreeWidgetItem * item);

  QTreeWidget * tree_;  ///< 当前任务、episode 和文件组成的三层树。
  QLabel * status_;  ///< 数量、刷新进度或错误说明。
  QPushButton * refresh_;  ///< 可手动刷新历史文件的按钮。
  QTimer * poll_;  ///< GUI 线程检查扫描结果，不等待磁盘。
  QTimer * periodic_;  ///< 数据页签可见时每两秒刷新。
  QString directory_;  ///< 当前本机任务目录。
  DirectoryOpener opener_;  ///< 系统文件管理器调用或测试替身。
  std::future<RecordingScan> scan_;  ///< 最多一个后台扫描任务。
  std::shared_ptr<std::atomic_bool> cancelled_;  ///< 与后台共享的取消标记。
  bool refresh_pending_{false};  ///< 当前扫描结束后是否需要补一次刷新。
};
}  // namespace tracker_teleoperated
#endif
