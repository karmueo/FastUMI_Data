/**
 * @file recordings_widget.cpp
 * @brief 通过远端分页服务展示已保存记录摘要。
 */
#include "tracker_teleoperated/recordings_widget.hpp"

#include <algorithm>
#include <exception>
#include <map>
#include <QAction>
#include <QHeaderView>
#include <QLabel>
#include <QMenu>
#include <QMessageBox>
#include <QPointer>
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
  tree_->setContextMenuPolicy(Qt::CustomContextMenu);
  tree_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
  tree_->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
  tree_->header()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
  layout->addWidget(tree_);
  auto * hint = new QLabel(
    "列表来自远端录制服务；路径相对于录制端数据根目录。右键记录可删除该条数据。",
    this);  ///< 使用说明。
  hint->setWordWrap(true);
  layout->addWidget(hint);
  connect(refresh_, &QPushButton::clicked, this, &RecordingsWidget::requestRefresh);
  connect(tree_, &QTreeWidget::customContextMenuRequested,
    this, &RecordingsWidget::showContextMenu);
}

void RecordingsWidget::initialize(
  const rclcpp::Node::SharedPtr & node, const std::string & prefix)
{
  client_ = node->create_client<fastumi_interfaces::srv::ListRecordings>(prefix + "/list");
  delete_client_ = node->create_client<fastumi_interfaces::srv::DeleteRecording>(prefix + "/delete");
  refreshControls();
  requestRefresh();
}

void RecordingsWidget::requestRefresh()
{
  if (!client_ || delete_pending_) {return;}
  if (!client_->service_is_ready()) {
    list_pending_ = false;
    status_->setText("远端记录列表服务未就绪");
    refreshControls();
    return;
  }
  list_pending_ = true;
  recordings_.clear();
  status_->setText("正在刷新远端记录…");
  refreshControls();
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
        list_pending_ = false;
        status_->setText("读取远端记录失败：" + QString::fromStdString(result->message));
        refreshControls();
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
  list_pending_ = false;
  status_->setText(recordings_.empty() ? "暂无已保存数据" :
    QString("已加载 %1 / %2 条远端记录").arg(recordings_.size()).arg(total));
  refreshControls();
}

/** @copydoc RecordingsWidget::showContextMenu */
void RecordingsWidget::showContextMenu(const QPoint & position)
{
  auto * item = tree_->itemAt(position);  ///< 右键位置命中的树节点。
  if (!item || !item->parent()) {return;}
  const auto recording_id = item->data(0, Qt::UserRole).toString();  ///< 记录稳定 UUID。
  const auto relative_path = item->text(0);  ///< 菜单打开前保存的远端相对路径。
  if (recording_id.isEmpty()) {return;}
  tree_->setCurrentItem(item);
  const bool can_delete = delete_client_ && delete_client_->service_is_ready() &&
    !list_pending_ && !delete_pending_ && !confirmation_active_;  ///< 当前是否允许删除。
  QMenu menu(this);  ///< 当前记录的右键菜单。
  auto * delete_action = menu.addAction("删除该条数据");  ///< 单条删除入口。
  delete_action->setEnabled(can_delete);
  auto * selected_action = menu.exec(tree_->viewport()->mapToGlobal(position));  ///< 用户选择的菜单项。
  if (selected_action == delete_action && can_delete) {
    auto * current_item = tree_->currentItem();  ///< 菜单关闭后重新获取仍有效的记录节点。
    if (current_item && current_item->parent() &&
      current_item->data(0, Qt::UserRole).toString() == recording_id) {
      requestDelete(recording_id, relative_path);
    }
  }
}

/** @copydoc RecordingsWidget::requestDelete */
void RecordingsWidget::requestDelete(const QString & recording_id, const QString & relative_path)
{
  if (recording_id.isEmpty() || !delete_client_ || !delete_client_->service_is_ready() ||
    list_pending_ || delete_pending_ || confirmation_active_) {return;}
  const auto recording_id_string = recording_id.toStdString();  ///< 与当前列表核对的记录 UUID。
  if (std::none_of(recordings_.begin(), recordings_.end(),
    [&recording_id_string](const auto & recording) {
      return recording.recording_id == recording_id_string;
    })) {return;}
  confirmation_active_ = true;
  refreshControls();
  const auto confirmation = QMessageBox::warning(
    this, "删除已保存数据",
    QString("将永久删除以下已保存数据：\n%1\n\nrecording_id：%2\n\n此操作不可撤销，是否继续？")
    .arg(relative_path, recording_id),
    QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);  ///< 用户的不可撤销操作确认。
  confirmation_active_ = false;
  if (confirmation != QMessageBox::Yes || !delete_client_ ||
    !delete_client_->service_is_ready() || list_pending_ || delete_pending_) {
    refreshControls();
    return;
  }

  ++generation_;
  const auto delete_generation = ++delete_generation_;  ///< 当前删除请求代次。
  list_pending_ = false;
  delete_pending_ = true;
  status_->setText("正在删除远端记录：" + relative_path);
  refreshControls();
  auto request = std::make_shared<fastumi_interfaces::srv::DeleteRecording::Request>();  ///< 删除请求。
  request->recording_id = recording_id_string;
  QPointer<RecordingsWidget> widget(this);  ///< 控件销毁后异步响应不再访问界面。
  delete_client_->async_send_request(request,
    [widget, delete_generation, relative_path](
      rclcpp::Client<fastumi_interfaces::srv::DeleteRecording>::SharedFuture future) {
      bool success = false;  ///< 删除服务是否确认成功。
      QString message;  ///< 服务端说明或调用异常。
      try {
        const auto result = future.get();  ///< 删除服务响应。
        success = result->success;
        message = QString::fromStdString(result->message);
      } catch (const std::exception & error) {
        message = QString::fromUtf8(error.what());
      }
      if (widget.isNull() || delete_generation != widget->delete_generation_) {return;}
      widget->delete_pending_ = false;
      if (success) {
        widget->tree_->setCurrentItem(nullptr);
        widget->status_->setText("已删除远端记录：" + relative_path + "；正在刷新列表…");
        widget->refreshControls();
        widget->requestRefresh();
      } else {
        widget->status_->setText("删除远端记录失败：" +
          (message.isEmpty() ? QString("未知错误") : message));
        widget->refreshControls();
      }
    });
}

/** @copydoc RecordingsWidget::refreshControls */
void RecordingsWidget::refreshControls()
{
  refresh_->setEnabled(client_ && !list_pending_ && !delete_pending_ && !confirmation_active_);
}
}  // namespace tracker_teleoperated
