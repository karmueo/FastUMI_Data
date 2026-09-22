/**
 * @file video_source_widget.cpp
 * @brief 实现本机相机设备选择、异步切换确认及固定角色预览同步。
 */
#include "tracker_teleoperated/video_source_widget.hpp"

#include <algorithm>

#include <QComboBox>
#include <QAbstractItemView>
#include <QCollator>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QPointer>
#include <QSignalBlocker>
#include <QStringList>
#include <QTimer>
#include <QVariant>

#include <rviz_common/display.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/display_group.hpp>
#include <rviz_common/properties/property.hpp>

namespace tracker_teleoperated
{
namespace
{
/** @brief ROS 2 原始图像的规范消息类型名称。 */
const std::string kImageType = "sensor_msgs/msg/Image";
/** @brief RViz 默认 Image Display 的插件标识。 */
const QString kImageDisplayClass = "rviz_default_plugins/Image";

/**
 * @brief 在根显示组中按稳定名称查找 Image Display。
 * @param[in] context RViz 显示上下文，可为空。
 * @param[in] name 默认配置中的显示名称。
 * @return 找到的 Image Display；上下文、名称或类型不匹配时返回 `nullptr`。
 * @note 返回值由 RViz 持有，只能在 GUI 线程短时使用。
 */
rviz_common::Display * findImageDisplay(
  rviz_common::DisplayContext * context, const QString & name)
{
  if (!context || !context->getRootDisplayGroup()) {return nullptr;}
  /** @brief RViz 根组中的显示数量。 */
  const int count = context->getRootDisplayGroup()->numDisplays();
  for (int index = 0; index < count; ++index) {
    /** @brief 当前候选显示仍由根显示组持有。 */
    auto * display = context->getRootDisplayGroup()->getDisplayAt(index);
    if (display && display->getName() == name && display->getClassId() == kImageDisplayClass) {
      return display;
    }
  }
  return nullptr;
}

}  // namespace

/** @copydoc VideoSourceWidget::VideoSourceWidget */
VideoSourceWidget::VideoSourceWidget(
  QWidget * parent, TopicReader reader, TopicWriter writer)
: QWidget(parent), topic_reader_(std::move(reader)), topic_writer_(std::move(writer))
{
  /** @brief 两路选择与刷新按钮使用紧凑网格，避免压缩节点表格。 */
  auto * layout = new QGridLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  /** @brief 末端预览标签。 */
  auto * wrist_label = new QLabel("末端视频", this);
  /** @brief UMI 预览标签。 */
  auto * umi_label = new QLabel("UMI 视频", this);
  wrist_combo_ = new QComboBox(this);
  wrist_combo_->setObjectName("wrist_video_source");
  wrist_combo_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
  wrist_combo_->setMinimumContentsLength(18);
  wrist_combo_->setEnabled(false);
  umi_combo_ = new QComboBox(this);
  umi_combo_->setObjectName("umi_video_source");
  umi_combo_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
  umi_combo_->setMinimumContentsLength(18);
  refresh_button_ = new QPushButton("刷新设备", this);
  refresh_button_->setObjectName("refresh_video_sources");
  status_ = new QLabel("UMI 视频用于夹爪预测；末端视频用于数据录制。", this);
  status_->setObjectName("video_source_status");
  status_->setWordWrap(true);
  status_->setTextFormat(Qt::PlainText);
  layout->addWidget(wrist_label, 0, 0);
  layout->addWidget(wrist_combo_, 0, 1);
  layout->addWidget(umi_label, 1, 0);
  layout->addWidget(umi_combo_, 1, 1);
  layout->addWidget(refresh_button_, 0, 2, 2, 1);
  layout->addWidget(status_, 2, 0, 1, 3);
  layout->setColumnStretch(1, 1);
  refresh_timer_ = new QTimer(this);
  refresh_timer_->setInterval(2000);
  connect(refresh_timer_, &QTimer::timeout, this, &VideoSourceWidget::refreshSources);
  connect(refresh_button_, &QPushButton::clicked, this, [this]() {
      if (refresh_client_ && refresh_client_->service_is_ready()) {
        refresh_client_->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>(),
          [](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture) {});
      }
      refreshSources();
    });
  connect(umi_combo_, QOverload<int>::of(&QComboBox::activated), this,
    [this](int) {applySelection("UMI 视频", umi_combo_);});
  refresh_timer_->start();
}

/** @copydoc VideoSourceWidget::initialize */
void VideoSourceWidget::initialize(
  const rclcpp::Node::SharedPtr & node,
  rviz_common::DisplayContext * context)
{
  managed_ = true;
  refresh_client_ = node->create_client<std_srvs::srv::Trigger>("/tracker_teleoperated/refresh_camera_devices");
  device_status_ = node->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
    "/tracker_teleoperated/camera_devices", rclcpp::QoS(1).transient_local(),
    [this](const diagnostic_msgs::msg::DiagnosticArray & message) {updateDevices(message);});
  source_client_ = node->create_client<rcl_interfaces::srv::SetParametersAtomically>(
    "/tracker_component_manager/set_parameters_atomically");
  source_status_ = node->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
    "/tracker_teleoperated/components/status", rclcpp::QoS(1).transient_local(),
    [this](const diagnostic_msgs::msg::DiagnosticArray & message) {updateSources(message);});
  if (context || !topic_reader_) {topic_reader_ = [context](const QString & name) {
      /** @brief 每次重新查找显示，避免缓存被 RViz 删除的对象。 */
      auto * display = findImageDisplay(context, name);
      return display ? display->subProp("Topic")->getValue().toString() : QString();
    };}
  if (context || !topic_writer_) {topic_writer_ = [context](const QString & name, const QString & topic) {
      /** @brief 切换前重新验证显示仍存在且类型正确。 */
      auto * display = findImageDisplay(context, name);
      if (!display) {return false;}
      display->reset();
      display->setTopic(topic, QString::fromStdString(kImageType));
      return true;
    };}
  refreshSources();
}

/** @copydoc VideoSourceWidget::refreshSources */
void VideoSourceWidget::refreshSources()
{
  /** @brief 管理器目录已过滤元数据，按稳定物理端口排序。 */
  QStringList topics;
  for (const auto & device : devices_) {topics.append(device.first);}
  QCollator collator;  ///< 支持 2.3、2.10 等端口链的自然数字顺序。
  collator.setNumericMode(true);
  std::sort(topics.begin(), topics.end(), [&collator](const QString & left, const QString & right) {
      return collator.compare(left, right) < 0;
    });
  /** @brief 末端显示固定为管理器发布的解码话题。 */
  const auto wrist_topic = QString::fromStdString(sources_["末端视频"].values["image_topic"]);  ///< 解码输出。
  const bool wrist_found = topic_reader_ && !topic_reader_("末端视频").isEmpty();
  if (wrist_found && !wrist_topic.isEmpty() && topic_reader_("末端视频") != wrist_topic && topic_writer_) {
    topic_writer_("末端视频", wrist_topic);
  }
  {
    const QSignalBlocker blocker(wrist_combo_);  ///< 阻止只读内容变化触发选择。
    wrist_combo_->clear();
    wrist_combo_->addItem(wrist_topic.isEmpty() ? "等待解码节点状态" : wrist_topic);
    wrist_combo_->setEnabled(false);
  }
  /** @brief UMI 显示查找结果。 */
  const bool umi_found = updateCombo(umi_combo_, "UMI 视频", topics);
  if (!wrist_found || !umi_found) {
    /** @brief 缺失显示名称列表用于直接排查自定义 RViz 配置。 */
    QStringList missing;
    if (!wrist_found) {missing.append("末端视频");}
    if (!umi_found) {missing.append("UMI 视频");}
    status_->setText("未找到 RViz Image Display：" + missing.join("、"));
  } else {
    /** @brief 各路状态按行显示，设备刷新保留失败原因。 */
    QStringList messages;
    for (const auto & name : {QString("UMI 视频"), QString("末端视频")}) {
      auto & source = sources_[name];  ///< 本路诊断和请求状态。
      QString message = name + (name == "UMI 视频" ? "用于夹爪预测" : "来自远端 H.264 解码");  ///< 本路输入及操作提示。
      if (managed_) {
        message += "：" + QString::fromStdString(
          name == "UMI 视频" ? source.values["video_device"] : source.values["image_topic"]);
        if (source.pending || source.values["source_state"] == "pending" ||
          source.values["source_state"] == "checking") {message += "；正在切换或核对";}
        if (source.values["source_state"] == "deferred") {message += "；等待节点启动";}
        if (!source.values["source_reason"].empty()) {message += "；" + QString::fromStdString(source.values["source_reason"]);}
        if (!source.values["source_error"].empty()) {message += "；" + QString::fromStdString(source.values["source_error"]);}
        if (!source.error.isEmpty()) {message += "；" + source.error;}
        if (std::chrono::steady_clock::now() - manager_seen_ > std::chrono::seconds(2)) {
          message += "；等待管理器状态";
        }
      }
      messages.append(message);
    }
    status_->setText(messages.join("\n"));
    if (!scan_error_.isEmpty()) {status_->setText(status_->text() + "\n" + scan_error_);}
  }
}

/** @copydoc VideoSourceWidget::updateCombo */
bool VideoSourceWidget::updateCombo(
  QComboBox * combo, const QString & display_name,
  const QStringList & image_topics)
{
  if (!combo || !topic_reader_) {
    if (combo) {combo->setEnabled(false);}
    return false;
  }
  /** @brief 读取当前显示属性，用于识别 Displays 中的用户修改。 */
  QString display_topic = topic_reader_(display_name);  ///< 当前预览话题。
  QString current = QString::fromStdString(sources_[display_name].values["video_device"]);  ///< 已确认设备。
  if (display_topic.isEmpty()) {
    combo->setEnabled(false);
    return false;
  }
  /** @brief 管理器确认设备，Displays 始终恢复到固定角色话题。 */
  bool editable = false;
  if (managed_) {
    auto & source = sources_[display_name];  ///< 当前视频源的确认状态。
    const auto now = std::chrono::steady_clock::now();  ///< 单调时间用于连接和请求超时。
    if (source.pending && !source.accepted && now > source.deadline) {
      source_client_->remove_pending_request(source.request_id);
      source.pending = false;
      ++source.sequence;
      source.error = "切换确认超时，以管理器实际输入为准";
    }
    editable = now - manager_seen_ < std::chrono::seconds(2) && !session_busy_ &&
      source.values["source_editable"] == "true" && !source.pending &&
      (recording_state_.empty() || recording_state_ == "idle");
    for (const auto & entry : sources_) {if (entry.second.pending) {editable = false;}}
    const QString actual = QString::fromStdString(source.values["image_topic"]);  ///< 已确认输入。
    if (!actual.isEmpty() && actual != display_topic) {
      if (topic_writer_) {topic_writer_(display_name, actual);}
    }
    const QString previous_device = property((display_name + "_device").toUtf8()).toString();  ///< 上次确认设备。
    if (!current.isEmpty() && current != previous_device && topic_writer_ && !actual.isEmpty()) {
      topic_writer_(display_name, actual);
    }
    setProperty((display_name + "_device").toUtf8(), current);
    setProperty((display_name + "_confirmed").toUtf8(), actual);
    editable = editable && !source.pending;
  }
  /** @brief 弹出列表中的当前项尚未确认，周期诊断不能重置用户选择。 */
  if (combo->view()->isVisible()) {
    if (!editable) {combo->hidePopup();}
    combo->setEnabled(editable);
    return true;
  }
  /** @brief 阻止列表重建触发预览切换。 */
  const QSignalBlocker blocker(combo);
  combo->clear();
  for (const auto & topic : image_topics) {combo->addItem(topic + "  " + devices_[topic], topic);}
  if (!current.isEmpty() && !image_topics.contains(current)) {
    combo->addItem(current + "（设备不存在）", current);
  }
  /** @brief 当前确认设备在重建后的位置。 */
  const int current_index = combo->findData(current);
  combo->setCurrentIndex(current_index);
  combo->setEnabled(editable);
  combo->setToolTip(current);
  return true;
}

/** @copydoc VideoSourceWidget::applySelection */
void VideoSourceWidget::applySelection(
  const QString & display_name, QComboBox * combo)
{
  if (!combo || !topic_writer_) {return;}
  /** @brief 条目包含设备名称和错误说明，设备路径单独保存在用户数据中。 */
  const QString topic = combo->currentData().toString();
  if (topic.isEmpty()) {return;}
  requestSource(display_name, topic);
  refreshSources();
}

/** @copydoc VideoSourceWidget::requestSource */
void VideoSourceWidget::requestSource(const QString & name, const QString & topic)
{
  auto & source = sources_[name];  ///< 本路请求状态。
  if (source.pending) {return;}
  if (!source_client_ || !source_client_->service_is_ready()) {
    source.error = "管理器参数服务不可用";
    return;
  }
  source.pending = true;
  source.accepted = false;
  source.requested = topic;
  source.error.clear();
  source.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
  const unsigned sequence = ++source.sequence;  ///< 回调所属代次。
  auto request = std::make_shared<rcl_interfaces::srv::SetParametersAtomically::Request>();  ///< 单路更新请求。
  request->parameters.push_back(rclcpp::Parameter(
    "umi_video_device", topic.toStdString()).to_parameter_msg());
  const QPointer<VideoSourceWidget> guard(this);  ///< 卸载后不再访问控件。
  source.request_id = source_client_->async_send_request(request,
    [guard, name, sequence](rclcpp::Client<rcl_interfaces::srv::SetParametersAtomically>::SharedFuture future) {
      if (!guard || guard->sources_[name].sequence != sequence) {return;}
      auto & state = guard->sources_[name];  ///< 仍存活且属于该请求的状态。
      const auto result = future.get()->result;  ///< 已完成的参数服务响应。
      if (!result.successful) {state.pending = false; state.error = QString::fromStdString(result.reason);}
      else {state.accepted = true;}
      guard->refreshSources();
    }).request_id;
}

/** @copydoc VideoSourceWidget::updateSources */
void VideoSourceWidget::updateSources(const diagnostic_msgs::msg::DiagnosticArray & message)
{
  managed_ = true;
  manager_seen_ = std::chrono::steady_clock::now();
  for (const auto & status : message.status) {
    if (status.name == "recorder") {
      for (const auto & value : status.values) {
        if (value.key == "recording_state") {recording_state_ = value.value;}
      }
    }
    if (status.name == "session") {
      for (const auto & value : status.values) {if (value.key == "busy") {session_busy_ = value.value == "true";}}
      continue;
    }
    if (status.name != "umi_camera" && status.name != "wrist_decoder") {continue;}
    auto & source = sources_[status.name == "umi_camera" ? "UMI 视频" : "末端视频"];  ///< 本路共享状态。
    const auto previous = source.values["video_device"];  ///< 其他面板也可能完成设备切换。
    source.values.clear();
    for (const auto & value : status.values) {source.values[value.key] = value.value;}
    if (source.values["video_device"] != previous) {source.error.clear();}
    if (source.pending && source.values["requested_video_device"] == source.requested.toStdString() &&
      source.values["source_state"] != "pending" && source.values["source_state"] != "checking") {
      source.pending = false;
    }
  }
  refreshSources();
}

/** @copydoc VideoSourceWidget::umiBusy */
bool VideoSourceWidget::umiBusy() const
{
  if (!managed_) {return false;}
  if (session_busy_) {return true;}
  for (const auto & source : sources_) {if (source.second.pending) {return true;}}
  const auto item = sources_.find("UMI 视频");  ///< UMI 源的最近状态。
  if (item == sources_.end() || std::chrono::steady_clock::now() - manager_seen_ > std::chrono::seconds(2)) {return true;}
  const auto state = item->second.values.find("source_state");  ///< 输入切换状态。
  return item->second.pending || state == item->second.values.end() ||
         state->second == "pending" || state->second == "checking";
}

/** @copydoc VideoSourceWidget::updateDevices */
void VideoSourceWidget::updateDevices(const diagnostic_msgs::msg::DiagnosticArray & message)
{
  devices_.clear();
  scan_error_.clear();
  for (const auto & item : message.status) {
    if (item.name == "scan") {scan_error_ = QString::fromStdString(item.message); continue;}
    QString label;  ///< 名称、物理端口及当前内核节点。
    QString port;  ///< Hub 物理端口链。
    QString kernel_device;  ///< 当前动态内核节点，仅用于排查。
    for (const auto & value : item.values) {
      if (value.key == "name") {label = QString::fromStdString(value.value);}
      if (value.key == "physical_port") {port = QString::fromStdString(value.value);}
      if (value.key == "kernel_device") {kernel_device = QString::fromStdString(value.value);}
    }
    if (!port.isEmpty()) {label += "（端口 " + port + (kernel_device.isEmpty() ? "" : "，" + kernel_device) + "）";}
    if (!item.message.empty()) {label += "（" + QString::fromStdString(item.message) + "）";}
    devices_[QString::fromStdString(item.name)] = label;
  }
  refreshSources();
}

/** @copydoc VideoSourceWidget::save */
void VideoSourceWidget::save(rviz_common::Config config) const
{
  const auto source = sources_.find("UMI 视频");  ///< 已确认 UMI 状态。
  if (source != sources_.end()) {
    const auto value = source->second.values.find("video_device");  ///< 当前 UMI 设备。
    if (value != source->second.values.end()) {config.mapSetValue("UmiVideoDevice", QString::fromStdString(value->second));}
  }
}

/** @copydoc VideoSourceWidget::load */
void VideoSourceWidget::load(const rviz_common::Config & config)
{
  QString value;  ///< 保存配置中的 UMI 设备，仅作为收到管理器前的显示初值。
  if (config.mapGetString("UmiVideoDevice", &value)) {
    sources_["UMI 视频"].values["video_device"] = value.toStdString();
  }
}
}  // namespace tracker_teleoperated
