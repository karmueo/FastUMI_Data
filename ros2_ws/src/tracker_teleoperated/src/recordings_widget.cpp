/**
 * @file recordings_widget.cpp
 * @brief 在后台枚举已完成记录，在 GUI 线程展示并打开本机目录。
 */
#include "tracker_teleoperated/recordings_widget.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFileInfo>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSet>
#include <QShowEvent>
#include <QTimer>
#include <QTreeWidget>
#include <QTreeWidgetItemIterator>
#include <QVBoxLayout>

namespace tracker_teleoperated
{
namespace
{
/** @brief 标准文件系统接口在错误时保留具体系统原因。 */
namespace fs = std::filesystem;

/** @brief 比较任意长度的非负十进制编号。 @param[in] path Episode 路径。 @return 去除前导零的编号。 */
QString episodeNumber(const QString & path)
{
  /** @brief 目录名中 episode_ 后的十进制部分。 */
  auto number = QFileInfo(path).fileName().mid(8);
  while (number.size() > 1 && number.startsWith('0')) {number.remove(0, 1);}
  return number;
}

/** @brief 创建带完整路径信息的树项。 @param[in] parent 父树项。 @param[in] path 本项绝对路径。 @param[in] directory 双击后打开的目录。 @return 新树项，由树控件管理。 */
QTreeWidgetItem * makeItem(QTreeWidgetItem * parent, const QString & path, const QString & directory)
{
  /** @brief 将展示名称与实际路径分离，避免按文字拼接目录。 */
  auto * item = new QTreeWidgetItem(parent, {QFileInfo(path).fileName()});
  item->setData(0, Qt::UserRole, path);
  item->setData(0, Qt::UserRole + 1, directory);
  item->setToolTip(0, path);
  return item;
}
}  // namespace

/** @copydoc scanRecordings */
RecordingScan scanRecordings(const QString & directory, const std::atomic_bool & cancelled)
{
  /** @brief 后台只创建值对象，不持有或访问面板。 */
  RecordingScan result{directory, {}, {}};
  try {
    /** @brief 本机 UTF-8 路径与严格的已提交目录名称。 */
    const fs::path root(directory.toStdString());
    /** @brief 隐藏临时目录及非数字目录不会进入记录列表。 */
    const QRegularExpression pattern("^episode_[0-9]+$");
    if (!fs::exists(root)) {return result;}
    for (const auto & entry : fs::directory_iterator(root)) {
      if (cancelled.load()) {return result;}
      /** @brief 当前目录项的绝对路径，仅处理当前任务的直属记录。 */
      const auto path = QString::fromStdString(entry.path().string());
      if (!pattern.match(QFileInfo(path).fileName()).hasMatch()) {continue;}
      /** @brief 删除竞态使用错误码处理，真实读取错误由外层统一报告。 */
      std::error_code error;
      if (!entry.is_directory(error)) {
        if (error && error != std::errc::no_such_file_or_directory) {throw fs::filesystem_error("读取记录目录", entry.path(), error);}
        continue;
      }
      if (!fs::is_regular_file(entry.path() / "proprio.hdf5")) {continue;}
      /** @brief HDF5 提交时间作为列表的修改时间，无需打开数据内容。 */
      SavedRecording record{path, QFileInfo(path + "/proprio.hdf5").lastModified().toString("yyyy-MM-dd HH:mm:ss"), {"proprio.hdf5"}};
      if (fs::is_regular_file(entry.path() / "gripper.mp4")) {record.files.append("gripper.mp4");}
      result.recordings.append(record);
    }
    std::sort(result.recordings.begin(), result.recordings.end(),
      [](const SavedRecording & left, const SavedRecording & right) {
        /** @brief 用字符串长度和字典序实现无整数溢出的数字排序。 */
        const auto a = episodeNumber(left.path);
        /** @brief 右侧记录的规范编号。 */
        const auto b = episodeNumber(right.path);
        if (a.size() != b.size()) {return a.size() > b.size();}
        return a == b ? left.path > right.path : a > b;
      });
  } catch (const std::exception & error) {
    result.recordings.clear();
    result.error = "读取保存目录失败：" + QString::fromUtf8(error.what());
  }
  return result;
}

/** @copydoc RecordingsWidget::RecordingsWidget */
RecordingsWidget::RecordingsWidget(QWidget * parent, DirectoryOpener opener)
: QWidget(parent), opener_(opener ? std::move(opener) : DirectoryOpener(QDesktopServices::openUrl))
{
  /** @brief 数据页签的独立布局。 */
  auto * layout = new QVBoxLayout(this);
  /** @brief 浏览说明与手动刷新共用一行。 */
  auto * toolbar = new QHBoxLayout();
  status_ = new QLabel("等待管理器提供本机任务目录", this);
  status_->setObjectName("recordings_status");
  status_->setWordWrap(true);
  status_->setTextFormat(Qt::PlainText);
  refresh_ = new QPushButton("刷新", this);
  refresh_->setObjectName("recordings_refresh");
  refresh_->setEnabled(false);
  toolbar->addWidget(status_, 1);
  toolbar->addWidget(refresh_);
  layout->addLayout(toolbar);
  tree_ = new QTreeWidget(this);
  tree_->setObjectName("recordings_tree");
  tree_->setHeaderLabels({"当前任务 / 已保存记录", "修改时间"});
  tree_->setExpandsOnDoubleClick(false);
  tree_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
  tree_->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
  layout->addWidget(tree_);
  /** @brief 明确双击的本机文件管理器行为。 */
  auto * hint = new QLabel("双击条目打开对应文件夹；点击箭头展开数据文件。", this);
  hint->setWordWrap(true);
  layout->addWidget(hint);
  poll_ = new QTimer(this);
  periodic_ = new QTimer(this);
  connect(refresh_, &QPushButton::clicked, this, &RecordingsWidget::requestRefresh);
  connect(tree_, &QTreeWidget::itemDoubleClicked, this, [this](QTreeWidgetItem * item, int) {openItem(item);});
  connect(poll_, &QTimer::timeout, this, &RecordingsWidget::collectResult);
  connect(periodic_, &QTimer::timeout, this, [this]() {if (isVisible()) {requestRefresh();}});
  periodic_->start(2000);
}

/** @copydoc RecordingsWidget::~RecordingsWidget */
RecordingsWidget::~RecordingsWidget()
{
  poll_->stop();
  periodic_->stop();
  if (cancelled_) {cancelled_->store(true);}
  if (scan_.valid()) {scan_.wait();}
}

/** @copydoc RecordingsWidget::setDirectory */
void RecordingsWidget::setDirectory(const QString & directory)
{
  if (directory == directory_) {return;}
  directory_ = directory;
  tree_->clear();
  if (cancelled_) {cancelled_->store(true);}
  refresh_->setEnabled(!directory_.isEmpty() && QDir::isAbsolutePath(directory_));
  if (!refresh_->isEnabled()) {
    status_->setText("等待管理器提供有效的本机任务绝对目录");
    return;
  }
  requestRefresh();
}

/** @copydoc RecordingsWidget::requestRefresh */
void RecordingsWidget::requestRefresh()
{
  if (!refresh_->isEnabled()) {return;}
  if (scan_.valid()) {refresh_pending_ = true; return;}
  refresh_pending_ = false;
  cancelled_ = std::make_shared<std::atomic_bool>(false);
  status_->setText("正在刷新已保存记录…");
  scan_ = std::async(std::launch::async, [directory = directory_, cancelled = cancelled_]() {
    return scanRecordings(directory, *cancelled);
  });
  poll_->start(50);
}

/** @copydoc RecordingsWidget::showEvent */
void RecordingsWidget::showEvent(QShowEvent * event)
{
  QWidget::showEvent(event);
  requestRefresh();
}

/** @copydoc RecordingsWidget::collectResult */
void RecordingsWidget::collectResult()
{
  if (!scan_.valid() || scan_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {return;}
  /** @brief 只有已完成的结果才取出，GUI 不等待文件系统。 */
  const auto result = scan_.get();
  poll_->stop();
  if (!cancelled_->load() && result.directory == directory_) {
    displayResult(result);
    emit refreshed();
  }
  if (refresh_pending_) {requestRefresh();}
}

/** @copydoc RecordingsWidget::displayResult */
void RecordingsWidget::displayResult(const RecordingScan & result)
{
  /** @brief 保存当前展开路径，避免周期刷新打断浏览。 */
  QSet<QString> expanded;
  /** @brief 保留文件或目录选择，而非依赖易变化的行编号。 */
  const auto selected = tree_->currentItem() ? tree_->currentItem()->data(0, Qt::UserRole).toString() : QString();
  /** @brief 保留当前滚动位置。 */
  const int scroll = tree_->verticalScrollBar()->value();
  /** @brief 首次显示时默认展开任务根目录。 */
  const bool first = tree_->topLevelItemCount() == 0;
  for (QTreeWidgetItemIterator item(tree_); *item; ++item) {
    if ((*item)->isExpanded()) {expanded.insert((*item)->data(0, Qt::UserRole).toString());}
  }
  tree_->clear();
  /** @brief 当前任务根节点显示任务名，完整路径可悬停查看。 */
  auto * root = makeItem(nullptr, result.directory, result.directory);
  tree_->addTopLevelItem(root);
  for (const auto & record : result.recordings) {
    /** @brief 每条记录保留实际打开路径。 */
    auto * episode = makeItem(root, record.path, record.path);
    episode->setText(1, record.modified);
    for (const auto & file : record.files) {makeItem(episode, record.path + "/" + file, record.path);}
  }
  for (QTreeWidgetItemIterator item(tree_); *item; ++item) {
    /** @brief 当前树项用于匹配扫描前的状态。 */
    const auto path = (*item)->data(0, Qt::UserRole).toString();
    (*item)->setExpanded(expanded.contains(path) || (first && *item == root));
    if (path == selected) {tree_->setCurrentItem(*item);}
  }
  tree_->verticalScrollBar()->setValue(scroll);
  status_->setText(!result.error.isEmpty() ? result.error : result.recordings.isEmpty() ?
    "暂无已保存数据" : QString("已保存 %1 条记录（本机当前任务）").arg(result.recordings.size()));
  status_->setToolTip(result.directory);
}

/** @copydoc RecordingsWidget::openItem */
void RecordingsWidget::openItem(QTreeWidgetItem * item)
{
  if (!item) {return;}
  /** @brief 文件条目保存其父目录，不将界面文字当作命令执行。 */
  const auto directory = item->data(0, Qt::UserRole + 1).toString();
  if (!QFileInfo(directory).isDir()) {
    status_->setText("无法打开文件夹，目录不存在或不可访问：" + directory);
    return;
  }
  if (!opener_(QUrl::fromLocalFile(directory))) {
    status_->setText("系统文件管理器未能打开：" + directory);
  }
}
}  // namespace tracker_teleoperated
