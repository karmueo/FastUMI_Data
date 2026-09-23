/**
 * @file teleop_panel.cpp
 * @brief 实现 Tracker 遥操面板、RViz 窗口级快捷键及非阻塞 ROS 服务调用。
 */
#include "tracker_teleoperated/teleop_panel.hpp"
#include "tracker_teleoperated/recordings_widget.hpp"
#include "tracker_teleoperated/video_source_widget.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <QApplication>
#include <QAbstractSpinBox>
#include <QColor>
#include <QComboBox>
#include <QDialog>
#include <QEvent>
#include <QGridLayout>
#include <QHeaderView>
#include <QKeyEvent>
#include <QKeySequenceEdit>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QTextEdit>
#include <QTabWidget>
#include <QTimer>
#include <QTreeWidget>
#include <QUuid>
#include <QVBoxLayout>
#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/window_manager_interface.hpp>

namespace tracker_teleoperated
{
namespace
{
/** @brief 控制接口公共前缀。 */
const std::string prefix = "/tracker_teleoperated/";
/** @brief 稳定时钟不受 ROS 模拟时间和系统时钟调整影响。 */
using Clock = std::chrono::steady_clock;
/** @brief 将稳定业务枚举翻译为界面文字。 @param state 进程状态枚举。 @return 中文名称或原值。 */
QString translated(const std::string & state)
{
  /** @brief 进程与录制状态对应的中文展示。 */
  static const std::map<std::string, QString> names = {
    {"starting", "启动中"}, {"running", "运行中"}, {"stopped", "已停止"},
    {"stopping", "停止中"}, {"external", "外部节点"}, {"failed", "启动失败"},
    {"exited", "进程退出"}, {"idle", "空闲"}, {"recording", "录制中"}, {"saving", "保存中"}};
  return names.count(state) ? names.at(state) : QString::fromStdString(state);
}

/**
 * @brief 判断一个窗口或控件是否由指定 RViz 主窗口拥有。
 * @param[in] widget 当前活动窗口或焦点控件，可为空。
 * @param[in] rviz_window 当前面板所属 RViz 主窗口，可为空。
 * @return `widget` 等于主窗口，或其 Qt 父控件链最终到达主窗口时为 true。
 */
bool belongsToRviz(QWidget * widget, QWidget * rviz_window)
{
  for (auto * current = widget; current; current = current->parentWidget()) {
    if (current == rviz_window) {return true;}
  }
  return false;
}

/**
 * @brief 判断焦点链是否需要保留控件原有键盘输入。
 * @param[in] focus 当前焦点控件，可为空。
 * @param[in] rviz_window 遍历焦点父链时的停止边界。
 * @return 可编辑控件、下拉框、菜单或对话框获得焦点时为 true。
 */
bool protectsKeyboardInput(QWidget * focus, QWidget * rviz_window)
{
  for (auto * widget = focus; widget; widget = widget->parentWidget()) {
    if (auto * edit = qobject_cast<QLineEdit *>(widget); edit && !edit->isReadOnly()) {return true;}
    if (auto * edit = qobject_cast<QTextEdit *>(widget); edit && !edit->isReadOnly()) {return true;}
    if (auto * edit = qobject_cast<QPlainTextEdit *>(widget); edit && !edit->isReadOnly()) {return true;}
    if (qobject_cast<QAbstractSpinBox *>(widget) || qobject_cast<QComboBox *>(widget) ||
      qobject_cast<QKeySequenceEdit *>(widget) || qobject_cast<QMenu *>(widget) ||
      qobject_cast<QDialog *>(widget)) {return true;}
    if (widget == rviz_window) {break;}
  }
  return false;
}
}  // namespace

/** @brief 创建组件表格、按钮及诊断区，并安装窗口级过滤器。 @param[in] parent 管理控件生命周期的父窗口。 */
TeleopPanel::TeleopPanel(QWidget * parent) : rviz_common::Panel(parent)
{
  setFocusPolicy(Qt::StrongFocus);
  setMinimumWidth(520);
  /** @brief 面板主布局。 */
  auto * layout = new QVBoxLayout(this);
  hint_ = new QLabel("RViz 窗口内可直接使用快捷键；输入框和对话框中不触发。\n"
    "回车：开始遥操和录制；再次按下则停止录制并回位。\n"
    "Backspace：取消当前录制并回到初始位姿。\n"
    "workspace 标定：起点 C → 向上移动 C → 向前移动 C。\n"
    "reference_eef：跟踪稳定后按空格建立参考。", this);
  hint_->setWordWrap(true);
  layout->addWidget(hint_);
  video_sources_ = new VideoSourceWidget(this);
  layout->addWidget(video_sources_);
  /** @brief 页签共用下方控制按钮，切换数据浏览时仍可暂停或保存。 */
  auto * tabs = new QTabWidget(this);
  tabs->setObjectName("teleop_tabs");
  /** @brief 状态表格和详细诊断放在同一页签。 */
  auto * components = new QWidget(tabs);
  /** @brief 节点页签布局。 */
  auto * components_layout = new QVBoxLayout(components);
  tree_ = new QTreeWidget(components);
  tree_->setColumnCount(6);
  tree_->setHeaderLabels({"组件", "归属", "进程", "数据", "启动", "停止"});
  tree_->setRootIsDecorated(false);
  tree_->setMinimumHeight(270);
  tree_->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
  components_layout->addWidget(tree_);
  tabs->addTab(components, "节点状态");
  recordings_ = new RecordingsWidget(tabs);
  tabs->addTab(recordings_, "已保存数据");
  layout->addWidget(tabs, 1);
  connect(tree_, &QTreeWidget::itemSelectionChanged, this, &TeleopPanel::showDetails);
  /** @brief 所有按钮映射到相同的操作信号。 */
  auto * grid = new QGridLayout();
  /** @brief 面板操作与快捷键标签。 */
  const std::vector<std::pair<QString, QString>> actions = {
    {"start_all", "启动全部"}, {"stop_all", "停止全部"}, {"toggle", "启用遥操（空格）"},
    {"pause", "暂停／取消操作（S）"}, {"calibrate", "标定采样（C）"}, {"home", "回位（H）"},
    {"record", "开始录制（A）"}, {"discard", "取消并回位（Backspace）"}, {"quit", "保存并退出（Q）"}};
  for (size_t i = 0; i < actions.size(); ++i) {
    /** @brief 本项按钮及固定操作标识。 */
    auto * button = new QPushButton(actions[i].second, this);
    /** @brief 捕获值语义的操作标识，避免按钮回调引用循环变量。 */
    const auto action = actions[i].first;
    button->setObjectName(action);
    buttons_[action] = button;
    grid->addWidget(button, static_cast<int>(i / 2), static_cast<int>(i % 2));
    connect(button, &QPushButton::clicked, this, [this, action]() {emit actionRequested(action);});
  }
  layout->addLayout(grid);
  control_status_ = new QLabel("遥操：等待状态", this);
  record_status_ = new QLabel("录制：等待状态", this);
  request_status_ = new QLabel("等待管理节点", this);
  for (auto * label : {control_status_, record_status_, request_status_}) {
    label->setWordWrap(true);
    label->setTextFormat(Qt::PlainText);
    layout->addWidget(label);
  }
  details_ = new QPlainTextEdit(this);
  details_->setReadOnly(true);
  details_->setMaximumHeight(150);
  details_->setPlaceholderText("选择组件查看异常、话题、消息间隔和日志路径");
  components_layout->addWidget(details_);
  spin_timer_ = new QTimer(this);
  heartbeat_timer_ = new QTimer(this);
  connect(this, &TeleopPanel::actionRequested, this, &TeleopPanel::dispatch);
  connect(spin_timer_, &QTimer::timeout, this, [this]() {
    if (node_ && !rclcpp::ok(node_->get_node_base_interface()->get_context())) {
      spin_timer_->stop();
      heartbeat_timer_->stop();
      return;
    }
    if (executor_) {executor_->spin_some(std::chrono::milliseconds(5));}
    refresh();
  });
  connect(heartbeat_timer_, &QTimer::timeout, this, [this]() {
    if (heartbeat_ && component_values_.count("teleop") &&
      component_values_.at("teleop").at("ownership") == "local" &&
      Clock::now() - manager_seen_ < std::chrono::seconds(2)) {
      heartbeat_->publish(std_msgs::msg::Empty());
    }
  });
  qApp->installEventFilter(this);
  refresh();
}

/** @brief 所有回调都在 GUI 线程执行，停止轮询即可同步解除生命周期。 */
TeleopPanel::~TeleopPanel()
{
  qApp->removeEventFilter(this);
  spin_timer_->stop();
  heartbeat_timer_->stop();
  /** @brief 子控件持有 ROS 客户端和订阅，必须在面板节点与 context 之前释放。 */
  delete recordings_;
  recordings_ = nullptr;
  delete video_sources_;
  video_sources_ = nullptr;
  if (executor_ && node_ && rclcpp::ok(node_->get_node_base_interface()->get_context())) {
    executor_->remove_node(node_);
  }
  executor_.reset();
  subscriptions_.clear();
  triggers_.clear();
  booleans_.clear();
  recording_status_subscription_.reset();
  recording_start_.reset();
  recording_stop_.reset();
  recording_cancel_.reset();
  recording_cancel_request_.reset();
  recording_get_request_.reset();
  recording_status_client_.reset();
  control_enable_.reset();
  control_disable_.reset();
  control_get_generation_.reset();
}

/** @copydoc TeleopPanel::save */
void TeleopPanel::save(rviz_common::Config config) const
{
  rviz_common::Panel::save(config);
  video_sources_->save(config);
}

/** @copydoc TeleopPanel::load */
void TeleopPanel::load(const rviz_common::Config & config)
{
  rviz_common::Panel::load(config);
  video_sources_->load(config);
}

/** @copydoc TeleopPanel::onInitialize */
void TeleopPanel::onInitialize()
{
  /** @brief 节点编号允许安全卸载后重新添加面板。 */
  static std::atomic<unsigned> counter{0};
  /** @brief 复用 RViz context，但不占用 RViz 节点的 executor。 */
  auto context = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node()->get_node_base_interface()->get_context();
  /** @brief 主窗口用于限定应用级事件过滤器，包含其浮动停靠窗口。 */
  auto * window_manager = getDisplayContext()->getWindowManager();
  rviz_window_ = window_manager ? window_manager->getParentWindow() : window();
  /** @brief 配置独立 ROS 节点，避免继承 RViz 的节点重命名参数。 */
  rclcpp::NodeOptions options;
  options.context(context);
  options.use_global_arguments(false);
  node_ = std::make_shared<rclcpp::Node>("tracker_teleop_panel_" + std::to_string(++counter), options);
  video_sources_->initialize(node_, getDisplayContext());
  /** @brief 让面板 executor 与 RViz 共用 context 的退出生命周期。 */
  rclcpp::ExecutorOptions executor_options;
  executor_options.context = context;
  executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>(executor_options);
  executor_->add_node(node_);
  heartbeat_ = node_->create_publisher<std_msgs::msg::Empty>(prefix + "panel_heartbeat", 10);
  /** @brief 业务状态支持晚加入面板；新鲜度仍由周期消息决定。 */
  auto qos = rclcpp::QoS(1).transient_local();
  subscriptions_.push_back(node_->create_subscription<std_msgs::msg::Bool>(prefix + "enabled", qos,
    [this](const std_msgs::msg::Bool & m) {enabled_ = m.data; control_seen_ = Clock::now();}));
  subscriptions_.push_back(node_->create_subscription<std_msgs::msg::String>(prefix + "status", qos,
    [this](const std_msgs::msg::String & m) {control_status_->setText(QString::fromStdString(m.data));}));
  subscriptions_.push_back(node_->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(prefix + "components/status", qos,
    [this](const diagnostic_msgs::msg::DiagnosticArray & m) {updateComponents(m);}));
  for (const auto & name : {"start_all", "stop_all", "shutdown_session", "calibrate_workspace", "return_home"}) {
    triggers_[name] = node_->create_client<std_srvs::srv::Trigger>(prefix + name);
  }
  booleans_["set_enabled"] = node_->create_client<std_srvs::srv::SetBool>(prefix + "set_enabled");
  control_enable_ = node_->create_client<fastumi_interfaces::srv::SetTeleopGeneration>(prefix + "enable");
  control_disable_ = node_->create_client<fastumi_interfaces::srv::SetTeleopGeneration>(prefix + "disable");
  control_get_generation_ = node_->create_client<fastumi_interfaces::srv::GetTeleopGeneration>(
    prefix + "get_generation");
  spin_timer_->start(50);
  heartbeat_timer_->start(100);
}

/** @copydoc TeleopPanel::eventFilter */
bool TeleopPanel::eventFilter(QObject * watched, QEvent * event)
{
  if (event->type() != QEvent::KeyPress && event->type() != QEvent::KeyRelease &&
    event->type() != QEvent::ShortcutOverride) {return Panel::eventFilter(watched, event);}
  /** @brief 尚未初始化的测试面板使用自身顶层窗口作为作用域。 */
  auto * rviz_window = rviz_window_ ? rviz_window_.data() : window();
  /** @brief 活动窗口必须属于当前 RViz；浮动停靠窗口通过 Qt 父链识别。 */
  auto * active = QApplication::activeWindow();
  auto * focus = QApplication::focusWidget();
  if (!rviz_window || !active || !belongsToRviz(active, rviz_window) ||
    (focus && !belongsToRviz(focus, rviz_window))) {
    return false;
  }
  /** @brief 模态窗口、弹出菜单及编辑控件保留原始按键行为。 */
  if (QApplication::activeModalWidget() || QApplication::activePopupWidget() ||
    qobject_cast<QDialog *>(active) || protectsKeyboardInput(focus, rviz_window)) {
    return false;
  }
  /** @brief 只允许无修饰键或 Shift 大写输入，保留 Ctrl+C 等编辑行为。 */
  auto * key = static_cast<QKeyEvent *>(event);
  if (key->modifiers() & (Qt::ControlModifier | Qt::AltModifier | Qt::MetaModifier)) {return false;}
  /** @brief 快捷键映射到与按钮相同的固定动作。 */
  const std::map<int, QString> actions = {{Qt::Key_Return, "combined"}, {Qt::Key_Enter, "combined"},
    {Qt::Key_Space, "toggle"}, {Qt::Key_S, "pause"},
    {Qt::Key_C, "calibrate"}, {Qt::Key_H, "home"}, {Qt::Key_A, "record"},
    {Qt::Key_Backspace, "discard"}, {Qt::Key_Q, "quit"}};
  if (!actions.count(key->key())) {return false;}
  event->accept();
  if (event->type() == QEvent::KeyPress && !key->isAutoRepeat()) {
    emit actionRequested(actions.at(key->key()));
  }
  return true;
}

/** @brief 分派可用操作，暂停可抢占普通请求。 @param action 固定操作标识。 */
void TeleopPanel::dispatch(const QString & action)
{
  refresh();
  if (!node_ || (buttons_.count(action) && !buttons_[action]->isEnabled())) {return;}
  if (action == "combined") {
    combinedAction();
  } else if (action == "toggle" || action == "pause") {
    if (action == "pause" && combined_operation_ == CombinedOperation::Starting) {
      beginCombinedRollback(combined_sequence_, "人工暂停组合启动");
      refresh();
      return;
    }
    const bool enable = action == "toggle" && !enabled_;  ///< 本次明确的启停目标。
    const auto generation_floor = enable ? 0 : manual_enable_generation_;  ///< 覆盖在途启用代次。
    const auto sequence = beginRequest();  ///< 普通启停请求代次。
    manual_enable_pending_ = enable;
    manual_enable_generation_ = 0;
    requestControl(enable,
      [this, sequence](bool success, std::uint64_t, const std::string & message) {
        if (sequence != sequence_) {return;}
        manual_enable_pending_ = false;
        finishRequest(sequence, success, message);
      },
      [this, sequence]() {return sequence == sequence_;},
      [this, enable](std::uint64_t generation) {
        if (enable) {manual_enable_generation_ = generation;}
      }, generation_floor);
  } else if (action == "record") {
    recordingAction(action);
  } else if (action == "discard") {
    discardCombinedAction();
  } else {
    /** @brief 将用户操作映射为固定服务，避免界面拼接任意命令。 */
    const std::map<QString, std::string> services = {{"start_all", "start_all"}, {"stop_all", "stop_all"},
      {"quit", "shutdown_session"}, {"calibrate", "calibrate_workspace"}, {"home", "return_home"}};
    if (services.count(action)) {trigger(services.at(action));}
  }
  refresh();
}

/** @brief 记录服务超时并推进请求代次。 @return 本次请求的递增序号。 */
unsigned TeleopPanel::beginRequest()
{
  pending_ = true;
  request_deadline_ = Clock::now() + std::chrono::seconds(3);
  request_status_->setText("请求处理中…");
  return ++sequence_;
}

/** @brief 仅最新请求更新提示。 @param sequence 请求序号。 @param success 服务是否成功。 @param message 服务端说明。 */
void TeleopPanel::finishRequest(unsigned sequence, bool success, const std::string & message)
{
  if (sequence != sequence_) {return;}
  pending_ = false;
  request_status_->setText((success ? "" : "失败：") + QString::fromStdString(message));
  refresh();
}

/** @brief 非阻塞调用 Trigger，失败时更新提示。 @param service 相对于控制命名空间的服务名称。 */
void TeleopPanel::trigger(const std::string & service)
{
  /** @brief 保留本次异步调用的客户端。 */
  auto client = triggers_.at(service);
  if (!client->service_is_ready()) {request_status_->setText("服务未就绪"); return;}
  /** @brief 请求代次用于丢弃迟到的响应。 */
  const auto sequence = beginRequest();
  client->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>(),
    [this, sequence](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
      /** @brief 由 executor 确认已完成的服务响应。 */
      const auto result = future.get();
      finishRequest(sequence, result->success, result->message);
    });
}

/** @brief 异步调用组件或遥操启停服务。 @param service 相对服务名称。 @param value 目标启停状态。 */
void TeleopPanel::setBool(const std::string & service, bool value)
{
  if (!booleans_.count(service)) {
    booleans_[service] = node_->create_client<std_srvs::srv::SetBool>(prefix + service);
  }
  /** @brief 保留本次异步调用的客户端。 */
  auto client = booleans_.at(service);
  if (!client->service_is_ready()) {request_status_->setText("服务未就绪，请稍后重试"); return;}
  /** @brief 包含目标启停状态的服务请求。 */
  auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
  request->data = value;
  /** @brief 请求代次用于丢弃迟到的响应。 */
  const auto sequence = beginRequest();
  client->async_send_request(request, [this, sequence](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture future) {
    /** @brief 已完成的服务响应，不会在 GUI 线程等待。 */
    const auto result = future.get();
    finishRequest(sequence, result->success, result->message);
  });
}

/** @copydoc TeleopPanel::requestControl */
void TeleopPanel::requestControl(bool enable,
  const std::function<void(bool, std::uint64_t, const std::string &)> & done,
  const std::function<bool()> & allowed,
  const std::function<void(std::uint64_t)> & sent,
  std::uint64_t generation_floor)
{
  if (!control_get_generation_ || !control_get_generation_->service_is_ready()) {
    done(false, 0, "遥操代次查询服务未就绪");
    return;
  }
  control_get_generation_->async_send_request(
    std::make_shared<fastumi_interfaces::srv::GetTeleopGeneration::Request>(),
    [this, enable, done, allowed, sent, generation_floor](
      rclcpp::Client<fastumi_interfaces::srv::GetTeleopGeneration>::SharedFuture future) {
      if (!allowed()) {return;}
      const auto state = future.get();  ///< 控制节点当前最高代次。
      const auto highest_generation = std::max(
        state->operation_generation, generation_floor);  ///< 包含可能尚未到达服务端的启用请求。
      if (highest_generation == std::numeric_limits<std::uint64_t>::max()) {
        done(false, 0, "遥操操作代次已耗尽");
        return;
      }
      const auto generation = highest_generation + 1;  ///< 本次请求代次。
      auto client = enable ? control_enable_ : control_disable_;  ///< 目标启停服务。
      if (!client || !client->service_is_ready()) {
        done(false, generation, "遥操启停服务未就绪");
        return;
      }
      auto request = std::make_shared<fastumi_interfaces::srv::SetTeleopGeneration::Request>();
      request->operation_generation = generation;
      sent(generation);
      client->async_send_request(request,
        [done, allowed, enable, generation](
          rclcpp::Client<fastumi_interfaces::srv::SetTeleopGeneration>::SharedFuture result_future) {
          if (!allowed()) {return;}
          const auto result = result_future.get();  ///< 控制节点持久化后的启停结果。
          done(result->success && result->enabled == enable, generation, result->message);
        });
    });
}

/** @copydoc TeleopPanel::recordingAction */
void TeleopPanel::recordingAction(const QString & action)
{
  const auto sequence = ++recording_sequence_;  ///< 当前录制操作代次。
  manual_start_pending_ = action != "discard" && record_state_ != "recording";
  record_pending_ = true;
  record_deadline_ = Clock::now() + std::chrono::seconds(3);
  request_status_->setText("录制请求处理中…");
  if (action == "discard") {
    auto request = std::make_shared<fastumi_interfaces::srv::CancelRecording::Request>();  ///< 取消请求。
    request->recording_id = recording_id_;
    recording_cancel_->async_send_request(request,
      [this, sequence](rclcpp::Client<fastumi_interfaces::srv::CancelRecording>::SharedFuture future) {
        const auto result = future.get();  ///< 取消响应。
        if (sequence != recording_sequence_) {return;}
        if (!result->success) {
          record_pending_ = false;
          request_status_->setText("取消录制失败：" + QString::fromStdString(result->message));
        }
      });
    return;
  }
  if (record_state_ == "recording") {
    auto request = std::make_shared<fastumi_interfaces::srv::StopRecording::Request>();  ///< 停止请求。
    request->recording_id = recording_id_;
    recording_stop_->async_send_request(request,
      [this, sequence](rclcpp::Client<fastumi_interfaces::srv::StopRecording>::SharedFuture future) {
        const auto result = future.get();  ///< 停止响应。
        if (sequence != recording_sequence_) {return;}
        if (!result->success && result->code != "ALREADY_STOPPING" && result->code != "ALREADY_SAVED") {
          record_pending_ = false;
          request_status_->setText("停止录制失败：" + QString::fromStdString(result->message));
        }
      });
    return;
  }
  auto request = std::make_shared<fastumi_interfaces::srv::StartRecording::Request>();  ///< 开始请求。
  manual_request_id_ = QUuid::createUuid().toString(QUuid::WithoutBraces).toLower().toStdString();
  request->request_id = manual_request_id_;
  request->dir_name = recording_dir_name_;
  request->name = recording_name_;
  const auto request_id = manual_request_id_;  ///< 用于关联超时后的迟到响应。
  recording_start_->async_send_request(request,
    [this, sequence, request_id](rclcpp::Client<fastumi_interfaces::srv::StartRecording>::SharedFuture future) {
      const auto result = future.get();  ///< 开始响应。
      if (sequence != recording_sequence_) {
        if (combined_operation_ == CombinedOperation::RollingBack &&
          combined_request_id_ == request_id && result->success) {
          combined_recording_id_ = result->recording_id;
        }
        return;
      }
      manual_start_pending_ = false;
      if (result->success) {
        recording_id_ = result->recording_id;
        if (record_state_ == "recording") {record_pending_ = false;}
      } else {
        record_pending_ = false;
        request_status_->setText("开始录制失败：" + QString::fromStdString(result->message));
      }
    });
}

/** @copydoc TeleopPanel::combinedAction */
void TeleopPanel::combinedAction()
{
  if (combined_operation_ != CombinedOperation::Idle) {
    request_status_->setText("回车组合操作正在处理中，请勿重复触发");
    return;
  }
  if (record_state_ == "idle") {
    if (!buttons_["toggle"]->isEnabled() || !buttons_["record"]->isEnabled()) {
      request_status_->setText("回车启动不可用，请检查遥操、录像、相机及管理节点状态");
      return;
    }
    startCombinedAction();
    return;
  }
  if (record_state_ == "recording") {
    if (!buttons_["home"]->isEnabled() || !buttons_["record"]->isEnabled()) {
      request_status_->setText("回车停止不可用，请检查回位、录像及管理节点状态");
      return;
    }
    stopCombinedAction();
    return;
  }
  request_status_->setText(record_state_ == "saving" ?
    "录像正在保存，请等待完成后再按回车" : "录像状态未知，暂时不能执行回车组合操作");
}

/** @copydoc TeleopPanel::startCombinedAction */
void TeleopPanel::startCombinedAction()
{
  const auto sequence = ++combined_sequence_;  ///< 本次组合启动代次。
  combined_operation_ = CombinedOperation::Starting;
  combined_control_done_ = false;
  combined_control_success_ = false;
  combined_record_done_ = false;
  combined_record_success_ = false;
  combined_fallback_done_ = false;
  combined_fallback_started_ = false;
  combined_fallback_success_ = false;
  combined_message_.clear();
  combined_recording_id_.clear();
  combined_request_id_ = QUuid::createUuid().toString(QUuid::WithoutBraces).toLower().toStdString();
  combined_enable_generation_ = 0;
  recovery_pause_control_ = true;
  record_pending_ = true;
  record_deadline_ = Clock::now() + std::chrono::seconds(3);
  combined_deadline_ = Clock::now() + std::chrono::seconds(5);
  request_status_->setText("正在同时启用遥操和开始录制…");

  requestControl(true,
    [this, sequence](bool success, std::uint64_t, const std::string & message) {
      if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::Starting) {return;}
      combined_control_done_ = true;
      combined_control_success_ = success;
      if (!success) {combined_message_ = "启用遥操失败：" + QString::fromStdString(message);}
      evaluateCombinedStart(sequence);
    },
    [this, sequence]() {
      return sequence == combined_sequence_ && combined_operation_ == CombinedOperation::Starting;
    },
    [this](std::uint64_t generation) {combined_enable_generation_ = generation;});

  auto record_request = std::make_shared<fastumi_interfaces::srv::StartRecording::Request>();  ///< 录制启动请求。
  record_request->request_id = combined_request_id_;
  record_request->dir_name = recording_dir_name_;
  record_request->name = recording_name_;
  recording_start_->async_send_request(record_request,
    [this, sequence](rclcpp::Client<fastumi_interfaces::srv::StartRecording>::SharedFuture future) {
      if (sequence != combined_sequence_) {return;}
      const auto result = future.get();  ///< 录制启动响应。
      if (combined_operation_ == CombinedOperation::RollingBack) {
        if (result->success) {combined_recording_id_ = result->recording_id;}
        return;
      }
      if (combined_operation_ != CombinedOperation::Starting) {return;}
      combined_record_done_ = true;
      combined_record_success_ = result->success;
      if (result->success) {
        combined_recording_id_ = result->recording_id;
      } else {
        if (!combined_message_.isEmpty()) {combined_message_ += "；";}
        combined_message_ += "开始录制失败：" + QString::fromStdString(result->message);
      }
      evaluateCombinedStart(sequence);
    });
  refresh();
}

/** @copydoc TeleopPanel::evaluateCombinedStart */
void TeleopPanel::evaluateCombinedStart(unsigned sequence)
{
  if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::Starting ||
    !combined_control_done_ || !combined_record_done_) {return;}
  if (combined_control_success_ && combined_record_success_) {
    finishCombinedAction("遥操和录制已同时启动");
    return;
  }
  beginCombinedRollback(sequence, combined_message_);
}

/** @copydoc TeleopPanel::beginCombinedRollback */
void TeleopPanel::beginCombinedRollback(unsigned sequence, const QString & reason)
{
  if (sequence != combined_sequence_) {return;}
  combined_operation_ = CombinedOperation::RollingBack;
  combined_message_ = reason.isEmpty() ? "组合启动失败" : reason;
  recovery_record_done_ = combined_request_id_.empty();
  recovery_control_done_ = !recovery_pause_control_;
  recovery_completed_id_.clear();
  recovery_next_ = Clock::time_point{};
  record_pending_ = false;
  request_status_->setText(combined_message_ + "；正在恢复安全状态…");
  advanceCombinedRecovery();
}

/** @copydoc TeleopPanel::evaluateCombinedRollback */
void TeleopPanel::evaluateCombinedRollback(unsigned sequence)
{
  if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::RollingBack ||
    !recovery_control_done_ || !recovery_record_done_) {return;}
  QString result = "组合启动失败：" + combined_message_;  ///< 最终回滚结果。
  if (recovery_pause_control_) {result += "；遥操已暂停";}
  if (!combined_request_id_.empty()) {result += "；启动请求已核对";}
  if (!recovery_completed_id_.empty()) {
    result += "；已保存数据保留，请人工核对 UUID：" + QString::fromStdString(recovery_completed_id_);
  }
  finishCombinedAction(result);
}

/** @copydoc TeleopPanel::advanceCombinedRecovery */
void TeleopPanel::advanceCombinedRecovery()
{
  if (combined_operation_ != CombinedOperation::RollingBack || Clock::now() < recovery_next_) {return;}
  recovery_next_ = Clock::now() + std::chrono::seconds(1);
  const auto sequence = combined_sequence_;  ///< 当前恢复流程代次。
  if (!recovery_record_done_ && !combined_request_id_.empty()) {
    if (recording_cancel_request_ && recording_cancel_request_->service_is_ready()) {
      recording_cancel_request_->prune_pending_requests();
      auto request = std::make_shared<fastumi_interfaces::srv::CancelRecordingRequest::Request>();
      request->request_id = combined_request_id_;
      recording_cancel_request_->async_send_request(request,
        [this, sequence](rclcpp::Client<fastumi_interfaces::srv::CancelRecordingRequest>::SharedFuture future) {
          if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::RollingBack) {return;}
          const auto result = future.get();  ///< 撤销已接受仍需查询终态。
          if (!result->success && result->state != "completed") {
            request_status_->setText("撤销启动请求未确认，正在重试：" + QString::fromStdString(result->message));
          }
        });
    }
    if (recording_get_request_ && recording_get_request_->service_is_ready()) {
      recording_get_request_->prune_pending_requests();
      auto request = std::make_shared<fastumi_interfaces::srv::GetRecordingRequest::Request>();
      request->request_id = combined_request_id_;
      recording_get_request_->async_send_request(request,
        [this, sequence](rclcpp::Client<fastumi_interfaces::srv::GetRecordingRequest>::SharedFuture future) {
          if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::RollingBack) {return;}
          const auto result = future.get();  ///< 按请求 UUID 查询的权威终态。
          if (!result->found) {return;}
          if (result->state == "cancelled" || result->state == "failed" || result->state == "completed") {
            recovery_record_done_ = true;
            if (result->state == "completed") {recovery_completed_id_ = result->recording_id;}
            if (result->state == "failed") {
              combined_message_ += "；录制请求失败：" + QString::fromStdString(result->message);
            }
            evaluateCombinedRollback(sequence);
          }
        });
    }
  }
  if (!recovery_control_done_ && control_get_generation_ && control_get_generation_->service_is_ready()) {
    control_get_generation_->prune_pending_requests();
    control_get_generation_->async_send_request(
      std::make_shared<fastumi_interfaces::srv::GetTeleopGeneration::Request>(),
      [this, sequence](rclcpp::Client<fastumi_interfaces::srv::GetTeleopGeneration>::SharedFuture future) {
        if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::RollingBack) {return;}
        const auto state = future.get();  ///< 当前遥操代次和实际启停状态。
        if (!state->enabled && state->operation_generation >= combined_enable_generation_) {
          recovery_control_done_ = true;
          evaluateCombinedRollback(sequence);
          return;
        }
        const auto highest_generation = std::max(
          state->operation_generation, combined_enable_generation_);  ///< 覆盖在途启用请求的代次。
        if (!control_disable_ || !control_disable_->service_is_ready() ||
          highest_generation == std::numeric_limits<std::uint64_t>::max()) {return;}
        auto request = std::make_shared<fastumi_interfaces::srv::SetTeleopGeneration::Request>();
        request->operation_generation = highest_generation + 1;
        control_disable_->async_send_request(request,
          [this, sequence](rclcpp::Client<fastumi_interfaces::srv::SetTeleopGeneration>::SharedFuture result_future) {
            if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::RollingBack) {return;}
            const auto result = result_future.get();  ///< 暂停响应需下一轮权威查询确认。
            if (!result->success) {
              request_status_->setText("暂停遥操未确认，正在重试：" + QString::fromStdString(result->message));
            }
          });
      });
  }
  evaluateCombinedRollback(sequence);
}

/** @copydoc TeleopPanel::stopCombinedAction */
void TeleopPanel::stopCombinedAction()
{
  const auto sequence = ++combined_sequence_;  ///< 本次组合停止代次。
  const auto recording_id = recording_id_;  ///< 必须停止的当前录制 UUID。
  combined_operation_ = CombinedOperation::Stopping;
  combined_control_done_ = false;
  combined_control_success_ = false;
  combined_record_done_ = false;
  combined_record_success_ = false;
  combined_fallback_done_ = false;
  combined_fallback_started_ = false;
  combined_fallback_success_ = false;
  combined_message_.clear();
  record_pending_ = true;
  record_deadline_ = Clock::now() + std::chrono::seconds(3);
  combined_deadline_ = Clock::now() + std::chrono::seconds(5);
  request_status_->setText("正在停止录制并回到初始位姿…");

  auto stop_request = std::make_shared<fastumi_interfaces::srv::StopRecording::Request>();  ///< 录像停止请求。
  stop_request->recording_id = recording_id;
  recording_stop_->async_send_request(stop_request,
    [this, sequence](rclcpp::Client<fastumi_interfaces::srv::StopRecording>::SharedFuture future) {
      if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::Stopping) {return;}
      const auto result = future.get();  ///< 录像停止响应。
      combined_record_done_ = true;
      combined_record_success_ = result->success || result->code == "ALREADY_STOPPING" ||
        result->code == "ALREADY_SAVED";
      if (!combined_record_success_) {
        record_pending_ = false;
        combined_message_ = "停止录像失败：" + QString::fromStdString(result->message);
      }
      evaluateCombinedStop(sequence);
    });

  triggers_.at("return_home")->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>(),
    [this, sequence](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
      if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::Stopping) {return;}
      const auto result = future.get();  ///< 回位响应。
      combined_control_done_ = true;
      combined_control_success_ = result->success;
      if (!result->success) {
        if (!combined_message_.isEmpty()) {combined_message_ += "；";}
        combined_message_ += "回位失败：" + QString::fromStdString(result->message);
      }
      evaluateCombinedStop(sequence);
    });
  refresh();
}

/** @copydoc TeleopPanel::evaluateCombinedStop */
void TeleopPanel::evaluateCombinedStop(unsigned sequence)
{
  if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::Stopping) {return;}
  if (combined_control_done_ && !combined_control_success_ && !combined_fallback_started_) {
    combined_fallback_started_ = true;
    requestControl(false,
      [this, sequence](bool success, std::uint64_t, const std::string & message) {
        if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::Stopping) {return;}
        combined_fallback_done_ = true;
        combined_fallback_success_ = success;
        if (!success) {combined_message_ += "；暂停遥操失败：" + QString::fromStdString(message);}
        evaluateCombinedStop(sequence);
      },
      [this, sequence]() {
        return sequence == combined_sequence_ && combined_operation_ == CombinedOperation::Stopping;
      });
    return;
  }
  if (!combined_control_done_ || !combined_record_done_) {return;}
  if (!combined_control_success_ && !combined_fallback_done_) {return;}

  QString result;  ///< 组合停止的最终状态说明。
  if (combined_record_success_ && combined_control_success_) {
    result = "录像已停止并进入保存，机械臂已暂停遥操并开始回位";
  } else {
    result = combined_message_.isEmpty() ? "回车停止操作未完全成功" : combined_message_;
    if (!combined_control_success_ && combined_fallback_success_) {result += "；遥操已暂停";}
    if (combined_record_success_) {result += "；录像已停止并进入保存";}
  }
  finishCombinedAction(result);
}

/** @copydoc TeleopPanel::discardCombinedAction */
void TeleopPanel::discardCombinedAction()
{
  if (combined_operation_ != CombinedOperation::Idle) {
    request_status_->setText("组合操作正在处理中，请勿重复触发");
    return;
  }
  if (record_state_ != "recording") {
    request_status_->setText(record_state_ == "saving" ?
      "录像正在保存，不能取消当前数据" : "当前没有正在录制的数据");
    return;
  }
  if (!buttons_["discard"]->isEnabled()) {
    request_status_->setText("取消并回位不可用，请检查回位、录像及管理节点状态");
    return;
  }

  const auto sequence = ++combined_sequence_;  ///< 本次组合取消代次。
  const auto recording_id = recording_id_;  ///< 必须取消的当前录制 UUID。
  combined_operation_ = CombinedOperation::Discarding;
  combined_control_done_ = false;
  combined_control_success_ = false;
  combined_record_done_ = false;
  combined_record_success_ = false;
  combined_fallback_done_ = false;
  combined_fallback_started_ = false;
  combined_fallback_success_ = false;
  combined_message_.clear();
  record_pending_ = true;
  record_deadline_ = Clock::now() + std::chrono::seconds(3);
  combined_deadline_ = Clock::now() + std::chrono::seconds(5);
  request_status_->setText("正在取消当前录制并回到初始位姿…");

  auto cancel_request = std::make_shared<fastumi_interfaces::srv::CancelRecording::Request>();  ///< 录像取消请求。
  cancel_request->recording_id = recording_id;
  recording_cancel_->async_send_request(cancel_request,
    [this, sequence](rclcpp::Client<fastumi_interfaces::srv::CancelRecording>::SharedFuture future) {
      if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::Discarding) {return;}
      const auto result = future.get();  ///< 录像取消响应。
      combined_record_done_ = true;
      combined_record_success_ = result->success;
      if (!result->success) {
        record_pending_ = false;
        combined_message_ = "取消录像失败：" + QString::fromStdString(result->message);
      }
      evaluateCombinedDiscard(sequence);
    });

  triggers_.at("return_home")->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>(),
    [this, sequence](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
      if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::Discarding) {return;}
      const auto result = future.get();  ///< 回位响应。
      combined_control_done_ = true;
      combined_control_success_ = result->success;
      if (!result->success) {
        if (!combined_message_.isEmpty()) {combined_message_ += "；";}
        combined_message_ += "回位失败：" + QString::fromStdString(result->message);
      }
      evaluateCombinedDiscard(sequence);
    });
  refresh();
}

/** @copydoc TeleopPanel::evaluateCombinedDiscard */
void TeleopPanel::evaluateCombinedDiscard(unsigned sequence)
{
  if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::Discarding) {return;}
  if (combined_control_done_ && !combined_control_success_ && !combined_fallback_started_) {
    combined_fallback_started_ = true;
    requestControl(false,
      [this, sequence](bool success, std::uint64_t, const std::string & message) {
        if (sequence != combined_sequence_ || combined_operation_ != CombinedOperation::Discarding) {return;}
        combined_fallback_done_ = true;
        combined_fallback_success_ = success;
        if (!success) {combined_message_ += "；暂停遥操失败：" + QString::fromStdString(message);}
        evaluateCombinedDiscard(sequence);
      },
      [this, sequence]() {
        return sequence == combined_sequence_ && combined_operation_ == CombinedOperation::Discarding;
      });
    return;
  }
  if (!combined_control_done_ || !combined_record_done_) {return;}
  if (!combined_control_success_ && !combined_fallback_done_) {return;}

  QString result;  ///< 组合取消的最终状态说明。
  if (combined_record_success_ && combined_control_success_) {
    result = "当前录像已取消，机械臂已暂停遥操并开始回位";
  } else {
    result = combined_message_.isEmpty() ? "取消并回位操作未完全成功" : combined_message_;
    if (!combined_control_success_ && combined_fallback_success_) {result += "；遥操已暂停";}
    if (combined_record_success_) {result += "；当前录像已取消";}
    if (combined_control_success_) {result += "；机械臂已开始回位";}
  }
  finishCombinedAction(result);
}

/** @copydoc TeleopPanel::recoverDiscardAfterTimeout */
void TeleopPanel::recoverDiscardAfterTimeout()
{
  if (!recording_status_client_ || !recording_status_client_->service_is_ready()) {
    request_status_->setText("取消并回位响应超时；已请求暂停遥操，录制状态查询服务未就绪");
    return;
  }
  const auto sequence = ++recording_sequence_;  ///< 本次超时恢复查询代次。
  recording_status_client_->async_send_request(
    std::make_shared<fastumi_interfaces::srv::GetRecordingStatus::Request>(),
    [this, sequence](rclcpp::Client<fastumi_interfaces::srv::GetRecordingStatus>::SharedFuture future) {
      if (sequence != recording_sequence_) {return;}
      const auto status = future.get()->status;  ///< 远端权威录制状态。
      updateRecordingStatus(status);
      if (status.state == "recording" && recording_cancel_ && recording_cancel_->service_is_ready()) {
        request_status_->setText("取消并回位响应超时；已暂停遥操，正在重试取消当前录像");
        recordingAction("discard");
      } else if (status.state == "recording") {
        request_status_->setText("取消并回位响应超时；已暂停遥操，取消服务未就绪");
      } else {
        request_status_->setText("取消并回位响应超时；已暂停遥操，远端录像已不在录制状态");
      }
    });
}

/** @copydoc TeleopPanel::finishCombinedAction */
void TeleopPanel::finishCombinedAction(const QString & message)
{
  combined_operation_ = CombinedOperation::Idle;
  request_status_->setText(message);
  refresh();
}

/** @copydoc TeleopPanel::configureRecording */
void TeleopPanel::configureRecording(
  const std::string & prefix_value, const std::string & dir_name, const std::string & name)
{
  std::string prefix_value_normalized = prefix_value;  ///< 去除结尾斜杠后的服务前缀。
  while (prefix_value_normalized.size() > 1 && prefix_value_normalized.back() == '/') {
    prefix_value_normalized.pop_back();
  }
  recording_dir_name_ = dir_name;
  recording_name_ = name;
  if (prefix_value_normalized.empty() || prefix_value_normalized == recording_prefix_) {return;}
  recording_prefix_ = prefix_value_normalized;
  recording_start_ = node_->create_client<fastumi_interfaces::srv::StartRecording>(
    recording_prefix_ + "/start");
  recording_stop_ = node_->create_client<fastumi_interfaces::srv::StopRecording>(
    recording_prefix_ + "/stop");
  recording_cancel_ = node_->create_client<fastumi_interfaces::srv::CancelRecording>(
    recording_prefix_ + "/cancel");
  recording_cancel_request_ = node_->create_client<fastumi_interfaces::srv::CancelRecordingRequest>(
    recording_prefix_ + "/cancel_request");
  recording_get_request_ = node_->create_client<fastumi_interfaces::srv::GetRecordingRequest>(
    recording_prefix_ + "/get_request");
  recording_status_client_ = node_->create_client<fastumi_interfaces::srv::GetRecordingStatus>(
    recording_prefix_ + "/get_status");
  recording_status_subscription_ = node_->create_subscription<fastumi_interfaces::msg::RecordingStatus>(
    recording_prefix_ + "/status", rclcpp::QoS(1).reliable().transient_local(),
    [this](const fastumi_interfaces::msg::RecordingStatus & message) {
      updateRecordingStatus(message);
    });
  recordings_->initialize(node_, recording_prefix_);
}

/** @copydoc TeleopPanel::queryRecordingStatus */
void TeleopPanel::queryRecordingStatus()
{
  const auto sequence = ++recording_sequence_;  ///< 当前权威状态查询代次。
  if (!recording_status_client_ || !recording_status_client_->service_is_ready()) {
    request_status_->setText("录制请求超时，状态查询服务未就绪");
    return;
  }
  recording_status_client_->async_send_request(
    std::make_shared<fastumi_interfaces::srv::GetRecordingStatus::Request>(),
    [this, sequence](rclcpp::Client<fastumi_interfaces::srv::GetRecordingStatus>::SharedFuture future) {
      if (sequence != recording_sequence_) {return;}
      updateRecordingStatus(future.get()->status);
      request_status_->setText("已按远端权威状态核对录制请求");
    });
}

/** @copydoc TeleopPanel::updateRecordingStatus */
void TeleopPanel::updateRecordingStatus(
  const fastumi_interfaces::msg::RecordingStatus & message)
{
  const auto previous = record_state_;  ///< 用于识别保存完成的上一状态。
  record_state_ = message.state;
  recording_id_ = message.recording_id;
  last_completed_recording_id_ = message.last_completed.recording_id;
  record_seen_ = Clock::now();
  if (previous != record_state_) {
    if (!manual_start_pending_) {
      record_pending_ = false;
      ++recording_sequence_;
    }
  }
  if (previous == "saving" && record_state_ == "idle") {recordings_->requestRefresh();}
  record_status_->setText(QString("录制：%1，时长 %2 s，关节 %3，视频 %4，丢帧 %5%6")
    .arg(translated(record_state_)).arg(message.duration, 0, 'f', 1)
    .arg(message.joint_samples).arg(message.saved_frames).arg(message.dropped_frames)
    .arg(message.last_error.empty() ? "" : "；错误：" + QString::fromStdString(message.last_error)));
}

/** @brief 消息过期后禁用运动和录制入口；暂停服务仍可用以便人工停止。 */
void TeleopPanel::refresh()
{
  /** @brief 本轮统一使用的单调时刻。 */
  const auto now = Clock::now();
  if (pending_ && now > request_deadline_) {
    pending_ = false;
    ++sequence_;
    for (auto & entry : triggers_) {entry.second->prune_pending_requests();}
    for (auto & entry : booleans_) {entry.second->prune_pending_requests();}
    if (manual_enable_pending_) {
      manual_enable_pending_ = false;
      combined_request_id_.clear();
      combined_enable_generation_ = manual_enable_generation_;
      recovery_pause_control_ = true;
      beginCombinedRollback(++combined_sequence_, "遥操启用响应超时");
    } else {
      request_status_->setText("服务响应超时，请检查节点状态");
    }
  }
  if (record_pending_ && now > record_deadline_ && combined_operation_ == CombinedOperation::Idle) {
    record_pending_ = false;
    if (manual_start_pending_) {
      manual_start_pending_ = false;
      ++recording_sequence_;
      combined_request_id_ = manual_request_id_;
      combined_enable_generation_ = 0;
      recovery_pause_control_ = false;
      beginCombinedRollback(++combined_sequence_, "开始录制响应超时");
    } else {
      request_status_->setText("录制状态未变化，正在查询远端权威状态");
      queryRecordingStatus();
    }
  }
  if (combined_operation_ == CombinedOperation::RollingBack) {advanceCombinedRecovery();}
  if (combined_operation_ == CombinedOperation::Starting && now > combined_deadline_) {
    beginCombinedRollback(combined_sequence_, "回车组合启动响应超时");
  } else if ((combined_operation_ == CombinedOperation::Stopping ||
    combined_operation_ == CombinedOperation::Discarding) && now > combined_deadline_) {
    const auto timed_out_operation = combined_operation_;  ///< 用于选择对应的安全恢复动作。
    ++combined_sequence_;
    combined_operation_ = CombinedOperation::Idle;
    requestControl(false, [this](bool success, std::uint64_t, const std::string & message) {
      if (!success) {request_status_->setText("暂停遥操失败：" + QString::fromStdString(message));}
    });
    if (timed_out_operation == CombinedOperation::Discarding) {
      recoverDiscardAfterTimeout();
    } else if (record_state_ == "recording" && recording_stop_ && recording_stop_->service_is_ready()) {
      recordingAction("record");
    }
    if (timed_out_operation != CombinedOperation::Discarding) {
      request_status_->setText("回车组合操作响应超时；已请求暂停遥操并停止当前录像，请核对状态");
    }
  }
  /** @brief 管理诊断必须持续更新，断连后禁止新增操作。 */
  const bool manager = node_ && now - manager_seen_ < std::chrono::seconds(2);
  /** @brief 暂停操作只要求本会话归属，不依赖周期状态是否新鲜。 */
  const bool owns_control = node_ && component_values_.count("teleop") &&
    component_values_.at("teleop").at("ownership") == "local";
  /** @brief 启用和回位额外要求控制节点的周期状态新鲜。 */
  const bool control = owns_control && now - control_seen_ < std::chrono::seconds(2);
  /** @brief 远端录制只要求权威状态新鲜和相应服务可用。 */
  const bool record = node_ && now - record_seen_ < std::chrono::seconds(2) &&
    component_values_.count("recorder") &&
    component_values_.at("recorder").at("recording_enabled") == "true";
  /** @brief 普通操作需要管理器空闲且没有在途请求。 */
  const bool combined_idle = combined_operation_ == CombinedOperation::Idle;
  /** @brief 普通操作在组合流程中保持禁用，避免交叉修改同一状态。 */
  const bool available = manager && !pending_ && !manager_busy_ && combined_idle;
  if (!manager && manager_seen_ != Clock::time_point{}) {
    request_status_->setText("管理节点状态超时，组件状态已过期");
    for (int i = 0; i < tree_->topLevelItemCount(); ++i) {
      tree_->topLevelItem(i)->setText(3, "状态过期");
      tree_->topLevelItem(i)->setForeground(3, QColor("#b07800"));
    }
  }
  for (auto & entry : buttons_) {entry.second->setEnabled(false);}
  for (const auto & action : {"start_all", "stop_all", "quit"}) {
    buttons_[action]->setEnabled(manager && available);
  }
  if (node_) {
    const bool generation_ready = control_get_generation_ && control_get_generation_->service_is_ready();
    buttons_["pause"]->setEnabled(
      owns_control && generation_ready && control_disable_ && control_disable_->service_is_ready());
    buttons_["toggle"]->setEnabled(control && available && generation_ready &&
      control_enable_ && control_enable_->service_is_ready());
    buttons_["calibrate"]->setEnabled(control && available && triggers_.at("calibrate_workspace")->service_is_ready());
    buttons_["home"]->setEnabled(control && available && triggers_.at("return_home")->service_is_ready());
  }
  const bool start_ready = recording_start_ && recording_start_->service_is_ready();  ///< 开始服务发现状态。
  const bool stop_ready = recording_stop_ && recording_stop_->service_is_ready();  ///< 停止服务发现状态。
  const bool cancel_ready = recording_cancel_ && recording_cancel_->service_is_ready();  ///< 取消服务发现状态。
  buttons_["record"]->setEnabled(record && available && !record_pending_ &&
    ((record_state_ == "recording" && stop_ready) || (record_state_ == "idle" && start_ready)) &&
    !video_sources_->umiBusy());
  buttons_["discard"]->setEnabled(
    record && control && available && !record_pending_ && record_state_ == "recording" && cancel_ready &&
    triggers_.at("return_home")->service_is_ready());
  buttons_["toggle"]->setText(enabled_ ? "暂停遥操（空格）" : "启用遥操（空格）");
  buttons_["record"]->setText(record_state_ == "recording" ? "停止并保存（A）" :
    record_state_ == "saving" ? "保存中…" : "开始录制（A）");
  for (auto & entry : component_buttons_) {
    /** @brief 该行对应的最新归属、模式和进程状态。 */
    const auto & values = component_values_[entry.first];
    /** @brief 决定是否允许停止进程的归属信息。 */
    const auto owner = values.at("ownership");
    /** @brief 独立于数据健康的进程状态。 */
    const auto state = values.at("process_state");
    entry.second.first->setEnabled(manager && available && values.at("mode") == "auto" &&
      owner != "external" && (state == "stopped" || state == "failed" || state == "exited"));
    entry.second.second->setEnabled(manager && available && owner == "local" &&
      (state == "starting" || state == "running"));
  }
}

/** @brief 保留行与焦点并更新诊断快照。 @param message 管理器的完整组件诊断。 */
void TeleopPanel::updateComponents(const diagnostic_msgs::msg::DiagnosticArray & message)
{
  /** @brief 管理器重新连接时清除上一轮断连提示。 */
  const bool reconnected = Clock::now() - manager_seen_ >= std::chrono::seconds(2);
  manager_seen_ = Clock::now();
  for (const auto & status : message.status) {
    /** @brief 将诊断键值字段索引化，界面不解析中文消息。 */
    std::map<std::string, std::string> values;
    for (const auto & value : status.values) {values[value.key] = value.value;}
    if (status.name == "session") {
      manager_busy_ = values["busy"] == "true";
      if (reconnected || manager_busy_ || status.message != last_session_message_) {
        request_status_->setText(QString::fromStdString(status.message));
      }
      last_session_message_ = status.message;
      if (values["shutdown"] == "true") {
        spin_timer_->stop();
        heartbeat_timer_->stop();
        QTimer::singleShot(0, qApp, &QCoreApplication::quit);
      }
      continue;
    }
    /** @brief 组件的稳定行标识。 */
    const auto id = QString::fromStdString(status.name);
    /** @brief 复用现有行，保留选择和键盘焦点。 */
    QTreeWidgetItem * row = nullptr;
    for (int i = 0; i < tree_->topLevelItemCount(); ++i) {
      if (tree_->topLevelItem(i)->data(0, Qt::UserRole).toString() == id) {row = tree_->topLevelItem(i); break;}
    }
    if (!row) {
      row = new QTreeWidgetItem(tree_);
      row->setData(0, Qt::UserRole, id);
      /** @brief 仅自动管理且未运行的组件可启动。 */
      auto * start = new QPushButton("启动", tree_);
      /** @brief 仅本会话拥有的运行组件可停止。 */
      auto * stop = new QPushButton("停止", tree_);
      tree_->setItemWidget(row, 4, start);
      tree_->setItemWidget(row, 5, stop);
      component_buttons_[id] = {start, stop};
      /** @brief 组件对应的固定启停服务全名。 */
      const auto service = "components/" + status.name + "/set_running";
      booleans_[service] = node_->create_client<std_srvs::srv::SetBool>(prefix + service);
      connect(start, &QPushButton::clicked, this, [this, service]() {setBool(service, true);});
      connect(stop, &QPushButton::clicked, this, [this, service]() {setBool(service, false);});
    }
    component_values_[id] = values;
    if (id == "recorder" && values.count("service_prefix")) {
      configureRecording(
        values["service_prefix"], values["recording_dir_name"], values["recording_name"]);
    }
    row->setText(0, QString::fromStdString(values["label"]));
    row->setText(1, values["ownership"] == "local" ? "本次启动" :
      values["ownership"] == "external" ? "外部／只读" : "未管理");
    row->setText(2, translated(values["process_state"]));
    row->setText(3, values["mode"] == "disabled" ? "已禁用" : values["healthy"] == "true" ? "正常" : "异常／无数据");
    row->setForeground(3, status.level == 0 ? QColor("#228b22") : status.level == 1 ? QColor("#b07800") : QColor("#c03030"));
    /** @brief 组合可复制的异常原因与诊断字段。 */
    QString detail = QString::fromStdString(status.message);
    for (const auto & value : status.values) {
      detail += "\n" + QString::fromStdString(value.key + ": " + value.value);
    }
    details_by_id_[id] = detail;
    row->setToolTip(0, QString::fromStdString(status.message));
  }
  showDetails();
  refresh();
}

/** @brief 显示关键话题、数据年龄、退出码和日志位置。 */
void TeleopPanel::showDetails()
{
  if (tree_->currentItem()) {
    details_->setPlainText(details_by_id_[tree_->currentItem()->data(0, Qt::UserRole).toString()]);
  }
}
}  // namespace tracker_teleoperated
PLUGINLIB_EXPORT_CLASS(tracker_teleoperated::TeleopPanel, rviz_common::Panel)
