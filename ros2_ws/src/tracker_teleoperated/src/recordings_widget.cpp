/**
 * @file recordings_widget.cpp
 * @brief 通过远端分页服务展示已保存记录摘要。
 */
#include "tracker_teleoperated/recordings_widget.hpp"

#include <algorithm>
#include <map>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QScrollBar>
#include <QShowEvent>
#include <QTreeWidget>
#include <QVBoxLayout>

namespace tracker_teleoperated
{
RecordingsWidget::RecordingsWidget(QWidget * parent) : QWidget(parent)
{
  auto * layout = new QVBoxLayout(this);  ///< 当前页签的纵向布局。
  auto * toolbar = new QHBoxLayout();  ///< 状态与刷新按钮布局。
  status_ = new QLabel("等待远端录制服务", this);
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
  tree_->setHeaderLabels({"任务 / 记录", "时长", "采样 / 视频帧"});
  tree_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
  tree_->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
  tree_->header()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
  layout->addWidget(tree_);
  auto * hint = new QLabel("列表来自远端录制服务；路径相对于录制端数据根目录。", this);  ///< 使用说明。
  hint->setWordWrap(true);
  layout->addWidget(hint);
  connect(refresh_, &QPushButton::clicked, this, &RecordingsWidget::requestRefresh);
}

void RecordingsWidget::initialize(
  const rclcpp::Node::SharedPtr & node, const std::string & prefix)
{
  client_ = node->create_client<fastumi_interfaces::srv::ListRecordings>(prefix + "/list");
  refresh_->setEnabled(true);
  requestRefresh();
}

void RecordingsWidget::requestRefresh()
{
  if (!client_) {return;}
  if (!client_->service_is_ready()) {
    status_->setText("远端记录列表服务未就绪");
    return;
  }
  recordings_.clear();
  status_->setText("正在刷新远端记录…");
  requestPage(++generation_, 0);
}

void RecordingsWidget::showEvent(QShowEvent * event)
{
  QWidget::showEvent(event);
  requestRefresh();
}

void RecordingsWidget::requestPage(unsigned generation, std::uint32_t offset)
{
  auto request = std::make_shared<fastumi_interfaces::srv::ListRecordings::Request>();  ///< 当前分页请求。
  request->offset = offset;
  request->limit = 100;
  client_->async_send_request(request,
    [this, generation, offset](rclcpp::Client<fastumi_interfaces::srv::ListRecordings>::SharedFuture future) {
      if (generation != generation_) {return;}
      const auto result = future.get();  ///< 当前分页响应。
      if (!result->success) {
        status_->setText("读取远端记录失败：" + QString::fromStdString(result->message));
        return;
      }
      recordings_.insert(recordings_.end(), result->recordings.begin(), result->recordings.end());
      const auto next = static_cast<std::uint64_t>(offset) + result->recordings.size();  ///< 下一页偏移。
      if (next < result->total && !result->recordings.empty()) {
        requestPage(generation, static_cast<std::uint32_t>(next));
      } else {
        displayResult(result->total);
        emit refreshed();
      }
    });
}

void RecordingsWidget::displayResult(std::uint64_t total)
{
  const auto selected = tree_->currentItem() ?
    tree_->currentItem()->data(0, Qt::UserRole).toString() : QString();  ///< 刷新前的记录 ID。
  const int scroll = tree_->verticalScrollBar()->value();  ///< 刷新前滚动位置。
  tree_->clear();
  std::map<QString, QTreeWidgetItem *> tasks;  ///< 任务名到根节点的映射。
  for (const auto & recording : recordings_) {
    const auto task = QString::fromStdString(recording.dir_name + "/" + recording.name);  ///< 两级任务名。
    if (!tasks.count(task)) {
      tasks[task] = new QTreeWidgetItem(tree_, {task});
      tasks[task]->setExpanded(true);
    }
    auto * item = new QTreeWidgetItem(tasks[task], {
      QString::fromStdString(recording.relative_path),
      QString::number(recording.duration, 'f', 2) + " s",
      QString("关节 %1 / 动作 %2 / 视频 %3").arg(recording.joint_samples).arg(
        recording.action_samples).arg(recording.image_frames)});  ///< 单条远端记录。
    item->setData(0, Qt::UserRole, QString::fromStdString(recording.recording_id));
    item->setToolTip(0, QString::fromStdString(recording.recording_id));
    if (item->data(0, Qt::UserRole).toString() == selected) {tree_->setCurrentItem(item);}
  }
  tree_->verticalScrollBar()->setValue(scroll);
  status_->setText(recordings_.empty() ? "暂无已保存数据" :
    QString("已加载 %1 / %2 条远端记录").arg(recordings_.size()).arg(total));
}
}  // namespace tracker_teleoperated
