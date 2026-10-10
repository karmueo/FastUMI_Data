/**
 * @file collection_panel.cpp
 * @brief 实现 UMI 服务化采集 RViz2 Panel 的异步服务调用、状态刷新与列表管理。
 */

#include "fastumi_rviz_plugins/collection_panel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <random>
#include <sstream>
#include <string>

#include <QDateTime>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QMetaObject>
#include <QProgressBar>
#include <QPushButton>
#include <QStringList>
#include <QTimer>
#include <QVBoxLayout>

#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

namespace fastumi_rviz_plugins
{

namespace
{

/** @brief 开始请求的超时：预检与暂存通常在数秒内完成。 */
constexpr int kStartTimeoutMs = 25000;
/** @brief 停止/保存/取消请求的超时：覆盖后端排空 MCAP 的最长等待。 */
constexpr int kStopTimeoutMs = 75000;
/** @brief 删除与列表请求的超时。 */
constexpr int kShortTimeoutMs = 20000;
/** @brief 状态新鲜度检查周期。 */
constexpr int kConnectionCheckMs = 500;
/** @brief 状态等级对应的显示颜色。 */
constexpr const char * kOkColor = "#2e9e44";
constexpr const char * kWarningColor = "#d98e04";
constexpr const char * kErrorColor = "#d33232";

/** @brief 单调毫秒时钟，仅用于新鲜度判断。 */
int64_t steady_now_ms()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

/** @brief 生成 6 位十六进制串，区分同时存在的多个面板实例。 */
std::string random_nonce()
{
  std::random_device device;
  std::ostringstream stream;
  stream << std::hex << (device() & 0xFFFFFFu);
  return stream.str();
}

/** @brief 返回状态等级对应的颜色。 */
const char * level_color(StreamLevel level)
{
  switch (level) {
    case StreamLevel::kOk: return kOkColor;
    case StreamLevel::kWarning: return kWarningColor;
    default: return kErrorColor;
  }
}

}  // namespace

/** @brief 构造控件、布局和定时器。 */
CollectionPanel::CollectionPanel(QWidget * parent)
: rviz_common::Panel(parent),
  connection_label_(new QLabel("未连接", this)),
  state_label_(new QLabel("--", this)),
  calibration_label_(new QLabel("--", this)),
  duration_label_(new QLabel("00:00:00", this)),
  message_label_(new QLabel("", this)),
  task_edit_(new QLineEdit("task", this)),
  name_edit_(new QLineEdit(this)),
  start_button_(new QPushButton("开始采集", this)),
  stop_and_save_button_(new QPushButton("停止并保存", this)),
  stop_and_cancel_button_(new QPushButton("停止并取消", this)),
  save_button_(new QPushButton("保存", this)),
  cancel_button_(new QPushButton("取消", this)),
  image_label_(new QLabel("--", this)),
  tracker_label_(new QLabel("--", this)),
  gripper_label_(new QLabel("--", this)),
  gripper_bar_(new QProgressBar(this)),
  counts_label_(new QLabel("--", this)),
  alarms_label_(new QLabel("", this)),
  blockers_label_(new QLabel("", this)),
  filter_edit_(new QLineEdit(this)),
  record_list_(new QListWidget(this)),
  page_label_(new QLabel("第 1/1 页（共 0 条）", this)),
  previous_button_(new QPushButton("上一页", this)),
  next_button_(new QPushButton("下一页", this)),
  delete_button_(new QPushButton("删除选中记录", this)),
  connection_timer_(new QTimer(this)),
  request_timer_(new QTimer(this)),
  nonce_(random_nonce()),
  callback_gate_(std::make_shared<CallbackGate>())
{
  {
    std::lock_guard<std::mutex> lock(callback_gate_->mutex);
    callback_gate_->panel = this;
  }
  auto * layout = new QVBoxLayout(this);

  auto * overview = new QFormLayout();
  overview->addRow("后端连接", connection_label_);
  overview->addRow("采集状态", state_label_);
  overview->addRow("采集时长", duration_label_);
  overview->addRow("标定状态", calibration_label_);
  layout->addLayout(overview);
  state_label_->setStyleSheet("font-size: 16px; font-weight: bold;");

  task_edit_->setPlaceholderText("任务名（用作目录名）");
  name_edit_->setPlaceholderText("采集名称（可选）");
  auto * inputs = new QFormLayout();
  inputs->addRow("任务名", task_edit_);
  inputs->addRow("采集名称", name_edit_);
  layout->addLayout(inputs);

  auto * main_buttons = new QHBoxLayout();
  main_buttons->addWidget(start_button_);
  main_buttons->addWidget(stop_and_save_button_);
  main_buttons->addWidget(stop_and_cancel_button_);
  layout->addLayout(main_buttons);
  auto * pending_buttons = new QHBoxLayout();
  pending_buttons->addWidget(save_button_);
  pending_buttons->addWidget(cancel_button_);
  layout->addLayout(pending_buttons);
  save_button_->setToolTip("待保存状态下发布为正式记录");
  cancel_button_->setToolTip("待保存或故障状态下丢弃本次采集数据");
  message_label_->setWordWrap(true);
  blockers_label_->setWordWrap(true);
  blockers_label_->setStyleSheet(QString("color: %1;").arg(kWarningColor));
  layout->addWidget(blockers_label_);
  layout->addWidget(message_label_);

  auto * health_box = new QGroupBox("设备与数据健康", this);
  auto * health_layout = new QFormLayout(health_box);
  for (QLabel * label : {image_label_, tracker_label_, gripper_label_}) {
    label->setTextFormat(Qt::RichText);
    label->setWordWrap(true);
  }
  tracker_label_->setToolTip(
    "到达差为配对双方到达采集节点的时间差，包含传输与排队，不是纯算法耗时");
  gripper_label_->setToolTip(tracker_label_->toolTip());
  gripper_bar_->setRange(0, 1000);
  gripper_bar_->setValue(0);
  gripper_bar_->setFormat("--");
  health_layout->addRow("图像", image_label_);
  health_layout->addRow("Tracker", tracker_label_);
  health_layout->addRow("夹爪", gripper_label_);
  health_layout->addRow("夹爪开合", gripper_bar_);
  health_layout->addRow("已写入", counts_label_);
  layout->addWidget(health_box);
  alarms_label_->setWordWrap(true);
  alarms_label_->setStyleSheet(QString("color: %1; font-weight: bold;").arg(kErrorColor));
  layout->addWidget(alarms_label_);

  auto * records_box = new QGroupBox("已保存记录", this);
  auto * records_layout = new QVBoxLayout(records_box);
  filter_edit_->setPlaceholderText("按任务名筛选，回车应用");
  records_layout->addWidget(filter_edit_);
  record_list_->setMinimumHeight(120);
  records_layout->addWidget(record_list_);
  auto * paging = new QHBoxLayout();
  paging->addWidget(previous_button_);
  paging->addWidget(page_label_, 1);
  paging->addWidget(next_button_);
  records_layout->addLayout(paging);
  records_layout->addWidget(delete_button_);
  layout->addWidget(records_box);
  setLayout(layout);

  connect(start_button_, &QPushButton::clicked, this, &CollectionPanel::request_start);
  connect(stop_and_save_button_, &QPushButton::clicked, this,
    &CollectionPanel::request_stop_and_save);
  connect(stop_and_cancel_button_, &QPushButton::clicked, this,
    &CollectionPanel::request_stop_and_cancel);
  connect(save_button_, &QPushButton::clicked, this, &CollectionPanel::request_save);
  connect(cancel_button_, &QPushButton::clicked, this, &CollectionPanel::request_cancel);
  connect(delete_button_, &QPushButton::clicked, this,
    &CollectionPanel::request_delete_selected);
  connect(previous_button_, &QPushButton::clicked, this, &CollectionPanel::previous_page);
  connect(next_button_, &QPushButton::clicked, this, &CollectionPanel::next_page);
  connect(filter_edit_, &QLineEdit::returnPressed, this, &CollectionPanel::apply_task_filter);
  connect(task_edit_, &QLineEdit::textChanged, this, &CollectionPanel::refresh_controls);
  connect(record_list_, &QListWidget::itemSelectionChanged, this,
    &CollectionPanel::refresh_controls);

  request_timer_->setSingleShot(true);
  connect(request_timer_, &QTimer::timeout, this, &CollectionPanel::on_request_timeout);
  connection_timer_->setInterval(kConnectionCheckMs);
  connect(connection_timer_, &QTimer::timeout, this, &CollectionPanel::check_connection);
  connection_timer_->start();
  refresh_controls();
}

/** @brief 先撤销 ROS 回调再停止定时器，避免回调访问已销毁的控件。 */
CollectionPanel::~CollectionPanel()
{
  status_subscription_.reset();
  if (callback_gate_ != nullptr) {
    std::lock_guard<std::mutex> lock(callback_gate_->mutex);
    callback_gate_->panel = nullptr;
  }
  callback_gate_.reset();
  connection_timer_->stop();
  request_timer_->stop();
}

/** @brief 创建状态订阅和服务客户端。 */
void CollectionPanel::onInitialize()
{
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
  namespace srv = fastumi_interfaces::srv;
  start_client_ = node_->create_client<srv::CollectionStart>("/fastumi/collection/start");
  stop_and_save_client_ = node_->create_client<srv::CollectionStopAndSave>(
    "/fastumi/collection/stop_and_save");
  stop_and_cancel_client_ = node_->create_client<srv::CollectionStopAndCancel>(
    "/fastumi/collection/stop_and_cancel");
  save_client_ = node_->create_client<srv::CollectionSave>("/fastumi/collection/save");
  cancel_client_ = node_->create_client<srv::CollectionCancel>("/fastumi/collection/cancel");
  delete_client_ = node_->create_client<srv::CollectionDelete>("/fastumi/collection/delete");
  list_client_ = node_->create_client<srv::CollectionList>("/fastumi/collection/list");
  get_status_client_ = node_->create_client<srv::CollectionGetStatus>(
    "/fastumi/collection/get_status");
  /** @brief 状态为 best-effort 周期发布；回调只弱持有 gate。 */
  const std::weak_ptr<CallbackGate> weak_gate(callback_gate_);
  status_subscription_ = node_->create_subscription<fastumi_interfaces::msg::CollectionStatus>(
    "/fastumi/collection/status", rclcpp::QoS(10).best_effort(),
    [weak_gate](const fastumi_interfaces::msg::CollectionStatus::SharedPtr message) {
      const auto gate = weak_gate.lock();
      if (gate == nullptr) {return;}
      std::lock_guard<std::mutex> lock(gate->mutex);
      if (gate->panel == nullptr) {return;}
      const fastumi_interfaces::msg::CollectionStatus copy = *message;
      CollectionPanel * panel = gate->panel;
      QMetaObject::invokeMethod(panel, [panel, copy]() {panel->apply_status(copy);},
        Qt::QueuedConnection);
    });
  request_status();
}

/** @brief 周期检查状态新鲜度；失联后标记断开并持续尝试查询。 */
void CollectionPanel::check_connection()
{
  const bool fresh = status_is_fresh(have_status_, steady_now_ms(), last_status_ms_);
  if (!fresh) {
    if (have_status_) {
      /** @brief 失联后重置版本跟踪：后端重启后版本号会从头开始。 */
      have_status_ = false;
      was_disconnected_ = true;
      show_message("与采集后端失联，等待重新连接…", true);
    }
    connection_label_->setText("未连接（等待 /fastumi/collection）");
    connection_label_->setStyleSheet(QString("color: %1;").arg(kErrorColor));
    request_status();
  }
  refresh_controls();
}

/** @brief 查询权威状态；已有查询在途或服务未就绪时跳过。 */
void CollectionPanel::request_status()
{
  if (get_status_client_ == nullptr || status_query_in_flight_ ||
    !get_status_client_->service_is_ready())
  {
    return;
  }
  status_query_in_flight_ = true;
  const std::weak_ptr<CallbackGate> weak_gate(callback_gate_);
  get_status_client_->async_send_request(
    std::make_shared<fastumi_interfaces::srv::CollectionGetStatus::Request>(),
    [weak_gate](rclcpp::Client<fastumi_interfaces::srv::CollectionGetStatus>::SharedFuture future) {
      const auto gate = weak_gate.lock();
      if (gate == nullptr) {return;}
      std::lock_guard<std::mutex> lock(gate->mutex);
      if (gate->panel == nullptr) {return;}
      const fastumi_interfaces::msg::CollectionStatus copy = future.get()->status;
      CollectionPanel * panel = gate->panel;
      QMetaObject::invokeMethod(panel, [panel, copy]() {
        panel->status_query_in_flight_ = false;
        panel->apply_status(copy);
      }, Qt::QueuedConnection);
    });
  /** @brief 服务无响应时在超时后允许重新查询。 */
  QTimer::singleShot(kShortTimeoutMs, this, [this]() {status_query_in_flight_ = false;});
}

/** @brief 应用一条权威状态：丢弃版本回退的过期快照。 */
void CollectionPanel::apply_status(const fastumi_interfaces::msg::CollectionStatus & status)
{
  if (!should_apply_status(have_status_, status_.state_version, status.state_version)) {
    return;
  }
  const bool reconnected = !have_status_ || was_disconnected_;
  status_ = status;
  have_status_ = true;
  was_disconnected_ = false;
  last_status_ms_ = steady_now_ms();

  connection_label_->setText("已连接");
  connection_label_->setStyleSheet(QString("color: %1;").arg(kOkColor));
  state_label_->setText(QString::fromStdString(state_text(status.state)));
  state_label_->setToolTip(QString("状态版本 %1").arg(status.state_version));
  duration_label_->setText(QString::fromStdString(format_duration(status.duration_s)));
  calibration_label_->setText(
    status.calibration_status == "calibrated" ?
    "已携带外参快照" : "未标定（转换为 HDF5 前需提供通过验收的外参）");

  apply_stream(image_label_, status.image, false);
  apply_stream(tracker_label_, status.tracker, true);
  apply_stream(gripper_label_, status.gripper, true);
  const std::string gripper_text =
    format_gripper_percent(status.gripper_valid, status.gripper_percent);
  gripper_bar_->setFormat(QString::fromStdString(gripper_text));
  gripper_bar_->setValue(
    status.gripper_valid && std::isfinite(status.gripper_percent) ?
    static_cast<int>(std::clamp(status.gripper_percent, 0.0, 100.0) * 10.0) : 0);
  counts_label_->setText(
    QString("图像 %1 / Tracker %2 / 夹爪 %3　队列 %4/%5　可用磁盘 %6")
    .arg(status.image_messages).arg(status.tracker_messages).arg(status.gripper_messages)
    .arg(status.writer_queue_depth).arg(status.writer_queue_capacity)
    .arg(QString::fromStdString(format_bytes(status.free_disk_bytes))));

  QStringList alarms;
  for (const auto & code : status.alarms) {
    alarms << QString::fromStdString(alarm_text(code));
  }
  alarms_label_->setText(alarms.isEmpty() ? "" : "报警：" + alarms.join("；"));

  QStringList blockers;
  for (const auto & blocker : status.start_blockers) {
    blockers << QString::fromStdString(blocker);
  }
  blockers_label_->setText(
    blockers.isEmpty() ? "" : "暂不能开始：" + blockers.join("；"));

  if (status.state == "error") {
    show_message("采集故障：" + status.last_error, true);
  }
  if (status_confirms_pending(pending_request_id_, status.last_request_id)) {
    /** @brief 权威状态已经反映在途请求：即使响应丢失也结束等待。 */
    const bool ok = status.last_result_code == "OK";
    request_timer_->stop();
    confirmed_request_id_ = pending_request_id_;
    pending_request_id_.clear();
    show_message(
      ok ? "操作完成" : "操作失败：" + status.last_error, !ok);
  }
  /** @brief 重连或回到空闲且版本变化后刷新记录列表。 */
  if (reconnected ||
    (status.state == "idle" && status.state_version != list_refreshed_version_))
  {
    list_refreshed_version_ = status.state_version;
    request_list();
  }
  refresh_controls();
}

/** @brief 刷新单路输入的富文本显示。 */
void CollectionPanel::apply_stream(
  QLabel * label, const fastumi_interfaces::msg::CollectionStreamHealth & stream,
  bool show_pair)
{
  const StreamVerdict verdict =
    stream_verdict(stream.discovered, stream.fresh, stream.valid, !show_pair || stream.matched);
  QString text = QString("<span style='color:%1'>● %2</span>")
    .arg(level_color(verdict.level)).arg(QString::fromStdString(verdict.text));
  if (!stream.detail.empty() && verdict.level != StreamLevel::kOk) {
    text += "（" + QString::fromStdString(stream.detail).toHtmlEscaped() + "）";
  }
  text += "　" + QString::fromStdString(format_metric(stream.rate_hz, 1, "Hz"));
  text += "　延迟 " + QString::fromStdString(format_metric(stream.message_age_ms, 0, "ms"));
  if (show_pair) {
    text += "<br>源时间差 " +
      QString::fromStdString(format_metric(stream.source_delta_ms, 2, "ms")) +
      "　到达差 " +
      QString::fromStdString(format_metric(stream.pair_arrival_delta_ms, 2, "ms"));
  }
  if (stream.out_of_order > 0) {
    text += QString("　乱序 %1 次").arg(stream.out_of_order);
  }
  label->setText(text);
}

/** @brief 开始跟踪修改请求：生成唯一 ID、记录在途并启动超时。 */
std::string CollectionPanel::begin_request(const std::string & operation, int timeout_ms)
{
  pending_request_id_ = make_request_id(nonce_, ++request_counter_, operation);
  pending_operation_ = operation;
  request_timer_->start(timeout_ms);
  show_message("正在执行：" + operation + "…", false);
  /** @brief 延后到事件循环空闲再刷新：避免在被点击按钮自己的 clicked 处理中
   *  同步禁用该按钮。pending_request_id_ 已设置，期间重复点击会被各处理函数拒绝。 */
  QTimer::singleShot(0, this, &CollectionPanel::refresh_controls);
  return pending_request_id_;
}

/** @brief 请求无法发送：清除在途标记并提示。 */
void CollectionPanel::fail_request(const std::string & request_id, const std::string & message)
{
  if (request_id == pending_request_id_) {
    pending_request_id_.clear();
    request_timer_->stop();
  }
  show_message(message, true);
  refresh_controls();
}

/** @brief 发送修改请求并把响应经 gate 排队回 GUI 线程。 */
template<typename ClientPtr, typename RequestPtr>
void CollectionPanel::submit_modify(
  const ClientPtr & client, const RequestPtr & request, const std::string & operation)
{
  const std::string request_id = request->request_id;
  if (client == nullptr || !client->service_is_ready()) {
    fail_request(request_id, "采集服务不可用，请确认 collection_node 已启动");
    return;
  }
  using ClientType = typename ClientPtr::element_type;
  const std::weak_ptr<CallbackGate> weak_gate(callback_gate_);
  client->async_send_request(
    request,
    [weak_gate, operation](typename ClientType::SharedFuture future) {
      const auto gate = weak_gate.lock();
      if (gate == nullptr) {return;}
      std::lock_guard<std::mutex> lock(gate->mutex);
      if (gate->panel == nullptr) {return;}
      const ResultMessage copy = future.get()->result;
      CollectionPanel * panel = gate->panel;
      QMetaObject::invokeMethod(panel, [panel, operation, copy]() {
        panel->handle_result(operation, copy);
      }, Qt::QueuedConnection);
    });
}

/** @brief 应用修改请求的响应；与在途请求不匹配的过期响应被忽略。 */
void CollectionPanel::handle_result(const std::string & operation, const ResultMessage & result)
{
  /** @brief 响应是否对应当前请求，避免旧响应结束后续请求的等待。 */
  const bool matches_current = response_matches_pending(pending_request_id_, result.request_id);
  const bool confirmed_earlier = response_matches_pending(confirmed_request_id_, result.request_id);
  if (!matches_current && !confirmed_earlier) {
    return;
  }
  /** @brief 权威状态先到时响应仍携带更详细的结果（如保存路径），予以采用。 */
  if (confirmed_earlier) {
    confirmed_request_id_.clear();
  }
  if (!matches_current && !pending_request_id_.empty()) {
    return;
  }
  const bool done = result.accepted && result.completed;
  const bool in_progress = result.accepted && !result.completed && result.code == "IN_PROGRESS";
  if (matches_current && !in_progress) {
    request_timer_->stop();
    pending_request_id_.clear();
  }
  if (done) {
    show_message(operation + " 完成：" + result.message, false);
  } else if (in_progress) {
    show_message("相同请求仍在执行，等待后端状态…", false);
  } else {
    show_message(
      operation + " 失败 [" + result.code + "]：" + result.message, true);
  }
  request_status();
  refresh_controls();
}

/** @brief 在途请求超时：不再等待，并查询权威状态判断真实结果。 */
void CollectionPanel::on_request_timeout()
{
  if (pending_request_id_.empty()) {
    return;
  }
  show_message("请求超时：" + pending_operation_ + "。已刷新后端状态，请核对后重试", true);
  pending_request_id_.clear();
  status_query_in_flight_ = false;
  request_status();
  refresh_controls();
}

/** @brief 显示一条操作结果。 */
void CollectionPanel::show_message(const std::string & text, bool error)
{
  message_label_->setStyleSheet(QString("color: %1;").arg(error ? kErrorColor : kOkColor));
  message_label_->setText(QString::fromStdString(text));
}

/** @brief 根据后端可用性、连接和在途请求刷新按钮。 */
void CollectionPanel::refresh_controls()
{
  BackendFlags flags;
  flags.can_start = status_.can_start;
  flags.can_stop = status_.can_stop;
  flags.can_save = status_.can_save;
  flags.can_cancel = status_.can_cancel;
  flags.can_stop_and_save = status_.can_stop_and_save;
  flags.can_stop_and_cancel = status_.can_stop_and_cancel;
  const bool connected = status_is_fresh(have_status_, steady_now_ms(), last_status_ms_);
  const bool pending = !pending_request_id_.empty();
  const ButtonGate gate = compute_button_gate(
    flags, connected, pending, task_name_acceptable(task_edit_->text().toStdString()));
  start_button_->setEnabled(gate.start);
  stop_and_save_button_->setEnabled(gate.stop_and_save);
  stop_and_cancel_button_->setEnabled(gate.stop_and_cancel);
  save_button_->setEnabled(gate.save);
  cancel_button_->setEnabled(gate.cancel);
  const bool recording_or_busy = connected && status_.state != "idle";
  task_edit_->setEnabled(!recording_or_busy);
  name_edit_->setEnabled(!recording_or_busy);
  delete_button_->setEnabled(
    connected && !pending && record_list_->currentRow() >= 0);
  previous_button_->setEnabled(connected && page_ > 0);
  next_button_->setEnabled(
    connected && page_ + 1 < page_count(total_records_));
}

/** @brief 点击开始：校验任务名后提交 start。 */
void CollectionPanel::request_start()
{
  if (!pending_request_id_.empty()) {return;}
  const std::string task = task_edit_->text().trimmed().toStdString();
  if (!task_name_acceptable(task)) {
    show_message("任务名不能为空，且不能含路径分隔符或以点开头", true);
    return;
  }
  auto request = std::make_shared<fastumi_interfaces::srv::CollectionStart::Request>();
  request->request_id = begin_request("start", kStartTimeoutMs);
  request->task_name = task;
  request->name = name_edit_->text().trimmed().toStdString();
  submit_modify(start_client_, request, "开始");
}

/** @brief 点击停止并保存。 */
void CollectionPanel::request_stop_and_save()
{
  if (!pending_request_id_.empty()) {return;}
  auto request = std::make_shared<fastumi_interfaces::srv::CollectionStopAndSave::Request>();
  request->collection_uuid = status_.collection_uuid;
  request->request_id = begin_request("stop_and_save", kStopTimeoutMs);
  submit_modify(stop_and_save_client_, request, "停止并保存");
}

/** @brief 二次确认后停止并取消。 */
void CollectionPanel::request_stop_and_cancel()
{
  if (!pending_request_id_.empty()) {return;}
  if (QMessageBox::question(
      this, "停止并取消", "将停止采集并丢弃本次全部数据，确定吗？") != QMessageBox::Yes)
  {
    return;
  }
  auto request = std::make_shared<fastumi_interfaces::srv::CollectionStopAndCancel::Request>();
  request->collection_uuid = status_.collection_uuid;
  request->request_id = begin_request("stop_and_cancel", kStopTimeoutMs);
  submit_modify(stop_and_cancel_client_, request, "停止并取消");
}

/** @brief 保存待保存采集。 */
void CollectionPanel::request_save()
{
  if (!pending_request_id_.empty()) {return;}
  auto request = std::make_shared<fastumi_interfaces::srv::CollectionSave::Request>();
  request->collection_uuid = status_.collection_uuid;
  request->request_id = begin_request("save", kStopTimeoutMs);
  submit_modify(save_client_, request, "保存");
}

/** @brief 二次确认后取消待保存或故障采集。 */
void CollectionPanel::request_cancel()
{
  if (!pending_request_id_.empty()) {return;}
  if (QMessageBox::question(
      this, "取消采集", "将丢弃本次全部数据，确定吗？") != QMessageBox::Yes)
  {
    return;
  }
  auto request = std::make_shared<fastumi_interfaces::srv::CollectionCancel::Request>();
  request->collection_uuid = status_.collection_uuid;
  request->request_id = begin_request("cancel", kStopTimeoutMs);
  submit_modify(cancel_client_, request, "取消");
}

/** @brief 二次确认后删除选中记录（移入回收区）。 */
void CollectionPanel::request_delete_selected()
{
  if (!pending_request_id_.empty()) {return;}
  const int row = record_list_->currentRow();
  if (row < 0 || static_cast<std::size_t>(row) >= record_uuids_.size()) {
    return;
  }
  const QString description = record_list_->item(row)->text();
  if (QMessageBox::question(
      this, "删除记录",
      "将把以下记录移入回收区（本面板不提供恢复）：\n\n" + description + "\n\n确定删除吗？") !=
    QMessageBox::Yes)
  {
    return;
  }
  auto request = std::make_shared<fastumi_interfaces::srv::CollectionDelete::Request>();
  request->collection_uuid = record_uuids_[static_cast<std::size_t>(row)];
  request->request_id = begin_request("delete", kShortTimeoutMs);
  submit_modify(delete_client_, request, "删除");
}

/** @brief 应用任务筛选并回到第一页。 */
void CollectionPanel::apply_task_filter()
{
  page_ = 0;
  request_list();
}

/** @brief 上一页。 */
void CollectionPanel::previous_page()
{
  if (page_ > 0) {
    --page_;
    request_list();
  }
}

/** @brief 下一页。 */
void CollectionPanel::next_page()
{
  if (page_ + 1 < page_count(total_records_)) {
    ++page_;
    request_list();
  }
}

/** @brief 请求当前页记录；用查询代次丢弃过期响应。 */
void CollectionPanel::request_list()
{
  if (list_client_ == nullptr || !list_client_->service_is_ready()) {
    return;
  }
  applied_filter_ = filter_edit_->text().trimmed().toStdString();
  const uint64_t epoch = ++list_epoch_;
  auto request = std::make_shared<fastumi_interfaces::srv::CollectionList::Request>();
  request->task_name = applied_filter_;
  request->offset = page_ * kCollectionPageSize;
  request->limit = kCollectionPageSize;
  const std::weak_ptr<CallbackGate> weak_gate(callback_gate_);
  list_client_->async_send_request(
    request,
    [weak_gate, epoch](
      rclcpp::Client<fastumi_interfaces::srv::CollectionList>::SharedFuture future) {
      const auto gate = weak_gate.lock();
      if (gate == nullptr) {return;}
      std::lock_guard<std::mutex> lock(gate->mutex);
      if (gate->panel == nullptr) {return;}
      const auto response = future.get();
      const bool success = response->success;
      const std::string message = response->message;
      const auto records = response->records;
      const uint64_t total = response->total;
      CollectionPanel * panel = gate->panel;
      QMetaObject::invokeMethod(panel, [panel, epoch, success, message, records, total]() {
        panel->apply_list(epoch, success, message, records, total);
      }, Qt::QueuedConnection);
    });
}

/** @brief 应用记录列表；过期查询的响应被丢弃。 */
void CollectionPanel::apply_list(
  uint64_t epoch, bool success, const std::string & message,
  const std::vector<fastumi_interfaces::msg::CollectionRecord> & records, uint64_t total)
{
  if (!list_response_is_current(list_epoch_, epoch)) {
    return;
  }
  if (!success) {
    show_message("读取记录列表失败：" + message, true);
    return;
  }
  total_records_ = total;
  const uint32_t clamped = clamp_page(page_, total);
  if (clamped != page_) {
    /** @brief 删除后当前页可能越界：回退到最后一页重新查询。 */
    page_ = clamped;
    request_list();
    return;
  }
  record_list_->clear();
  record_uuids_.clear();
  for (const auto & record : records) {
    const QDateTime created = QDateTime::fromSecsSinceEpoch(
      static_cast<qint64>(record.created_at), Qt::UTC);
    QString line = QString("%1　%2%3　%4 帧　%5　%6")
      .arg(created.toString("yyyy-MM-dd HH:mm:ss'Z'"))
      .arg(QString::fromStdString(record.task_name))
      .arg(record.name.empty() ? QString() : "／" + QString::fromStdString(record.name))
      .arg(record.image_messages)
      .arg(QString::fromStdString(format_duration(record.duration_s)))
      .arg(QString::fromStdString(format_bytes(record.size_bytes)));
    if (!record.calibrated) {line += "　[未标定]";}
    if (record.has_alarms) {line += "　[有报警]";}
    auto * item = new QListWidgetItem(line, record_list_);
    item->setToolTip(QString::fromStdString(record.relative_path + "\n" + record.collection_uuid));
    record_uuids_.push_back(record.collection_uuid);
  }
  page_label_->setText(
    QString("第 %1/%2 页（共 %3 条）").arg(page_ + 1).arg(page_count(total)).arg(total));
  refresh_controls();
}

}  // namespace fastumi_rviz_plugins

PLUGINLIB_EXPORT_CLASS(fastumi_rviz_plugins::CollectionPanel, rviz_common::Panel)
