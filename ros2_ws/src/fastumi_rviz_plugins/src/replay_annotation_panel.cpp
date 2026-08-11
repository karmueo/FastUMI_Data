/**
 * @file replay_annotation_panel.cpp
 * @brief 实现 FastUMI 回放标注 RViz2 Panel 的 Qt 与 ROS 安全交接。
 */

#include "fastumi_rviz_plugins/replay_annotation_panel.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <string>

#include <QAction>
#include <QApplication>
#include <QEvent>
#include <QFormLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QMetaObject>
#include <QProgressBar>
#include <QPushButton>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>

#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

namespace fastumi_rviz_plugins
{

namespace
{

/**
 * @brief 把 ROS 时间消息转换为绝对纳秒。
 * @param stamp ROS 时间消息。
 * @return 有符号 64 位绝对纳秒。
 */
int64_t stamp_to_ns(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL +
         static_cast<int64_t>(stamp.nanosec);
}

}  // namespace

/** @brief 构造 Panel 控件、布局、定时器和全局事件过滤器。 */
ReplayAnnotationPanel::ReplayAnnotationPanel(QWidget * parent)
: rviz_common::Panel(parent), gripper_label_(new QLabel("--", this)),
  gripper_bar_(new QProgressBar(this)),
  gripper_valid_label_(new QLabel("无数据", this)),
  pose_label_(new QLabel("--", this)), freshness_label_(new QLabel("无数据", this)),
  episode_label_(new QLabel("episode 0：空闲", this)),
  completed_episode_count_label_(new QLabel("0", this)),
  episode_list_(new QListWidget(this)),
  episode_button_(new QPushButton("开始 episode（Space）", this)),
  playback_button_(new QPushButton("继续回放（Enter）", this)),
  finish_button_(new QPushButton("结束并保存（Ctrl+S）", this)),
  clear_button_(new QPushButton("删除所有标记", this)),
  timeline_slider_(new QSlider(Qt::Horizontal, this)),
  timeline_label_(new QLabel("00:00:00.000 / 00:00:00.000", this)),
  rate_label_(new QLabel("1.00×", this)),
  playback_hint_label_(new QLabel("← 0.5×　↑ 1×　→ 2×", this)),
  readiness_timer_(new QTimer(this)),
  player_state_timer_(new QTimer(this)),
  annotation_list_timer_(new QTimer(this)),
  callback_gate_(std::make_shared<CallbackGate>())
{
  /** @brief callback gate 初始指向当前 Panel，析构时会在互斥锁内置空。 */
  {
    std::lock_guard<std::mutex> lock(callback_gate_->mutex);
    callback_gate_->panel = this;
  }
  /** @brief 垂直布局保存状态区和两个鼠标控制按钮。 */
  auto * layout = new QVBoxLayout(this);
  /** @brief 表单布局压缩状态字段，便于窄 Dock 显示。 */
  auto * values = new QFormLayout();
  gripper_bar_->setRange(0, 1000);
  timeline_slider_->setRange(0, kTimelineSliderMaximum);
  timeline_slider_->setSingleStep(kTimelineSliderMaximum / 1000);
  timeline_slider_->setPageStep(kTimelineSliderMaximum / 100);
  values->addRow("夹爪开度", gripper_label_);
  values->addRow("开度条", gripper_bar_);
  values->addRow("夹爪有效", gripper_valid_label_);
  values->addRow("Tracker", pose_label_);
  values->addRow("Tracker 新鲜度", freshness_label_);
  values->addRow("Episode", episode_label_);
  values->addRow("完整 Episode 数", completed_episode_count_label_);
  values->addRow("回放时间", timeline_label_);
  values->addRow("播放倍率", rate_label_);
  layout->addLayout(values);
  layout->addWidget(timeline_slider_);
  layout->addWidget(playback_hint_label_);
  /** @brief 列表显示所有 STOP 闭合项，点击跳转、右键删除。 */
  layout->addWidget(new QLabel("已标注 Episode（单击跳转，右键删除）", this));
  episode_list_->setMinimumHeight(110);
  episode_list_->setContextMenuPolicy(Qt::CustomContextMenu);
  layout->addWidget(episode_list_);
  layout->addWidget(episode_button_);
  layout->addWidget(playback_button_);
  layout->addWidget(clear_button_);
  layout->addWidget(finish_button_);
  setLayout(layout);
  connect(episode_button_, &QPushButton::clicked, this,
    &ReplayAnnotationPanel::request_episode_toggle);
  connect(playback_button_, &QPushButton::clicked, this,
    &ReplayAnnotationPanel::request_playback_toggle);
  connect(finish_button_, &QPushButton::clicked, this,
    &ReplayAnnotationPanel::request_finish_and_save);
  connect(clear_button_, &QPushButton::clicked, this,
    &ReplayAnnotationPanel::request_clear_annotations);
  connect(episode_list_, &QListWidget::itemClicked, this,
    &ReplayAnnotationPanel::jump_to_episode);
  connect(episode_list_, &QListWidget::customContextMenuRequested, this,
    &ReplayAnnotationPanel::show_episode_context_menu);
  clear_button_->setToolTip("删除当前回放会话内全部 START、STOP 和 ABORT 标记");
  connect(timeline_slider_, &QSlider::sliderPressed, this,
    &ReplayAnnotationPanel::begin_timeline_drag);
  connect(timeline_slider_, &QSlider::sliderMoved, this,
    &ReplayAnnotationPanel::preview_timeline_drag);
  connect(timeline_slider_, &QSlider::sliderReleased, this,
    &ReplayAnnotationPanel::release_timeline_drag);
  readiness_timer_->setInterval(150);
  connect(readiness_timer_, &QTimer::timeout, this,
    &ReplayAnnotationPanel::refresh_controls);
  readiness_timer_->start();
  player_state_timer_->setInterval(500);
  connect(player_state_timer_, &QTimer::timeout, this,
    &ReplayAnnotationPanel::poll_player_state);
  player_state_timer_->start();
  annotation_list_timer_->setInterval(250);
  connect(annotation_list_timer_, &QTimer::timeout, this,
    &ReplayAnnotationPanel::poll_episode_annotations);
  annotation_list_timer_->start();
  qApp->installEventFilter(this);
  refresh_controls();
}

/** @brief 安全撤销 ROS 回调、Qt 定时器和应用级事件过滤器。 */
ReplayAnnotationPanel::~ReplayAnnotationPanel()
{
  /** @brief 先注销 ROS 回调，再阻塞正在运行的回调并撤销其 Panel 访问权。 */
  gripper_subscription_.reset();
  pose_subscription_.reset();
  clock_subscription_.reset();
  if (callback_gate_ != nullptr) {
    std::lock_guard<std::mutex> lock(callback_gate_->mutex);
    callback_gate_->panel = nullptr;
  }
  callback_gate_.reset();
  readiness_timer_->stop();
  player_state_timer_->stop();
  annotation_list_timer_->stop();
  /** @brief 析构前撤销应用级过滤器，避免 Qt 向已销毁对象投递按键。 */
  if (qApp != nullptr) {
    qApp->removeEventFilter(this);
  }
}

/** @brief 读取回放参数并创建 Player 服务客户端和数据订阅。 */
void ReplayAnnotationPanel::onInitialize()
{
  /** @brief RViz 的 ROS 抽象保证所有接口归属于同一个 executor。 */
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
  if (!node_->has_parameter("replay_start_time_ns")) {
    node_->declare_parameter<int64_t>("replay_start_time_ns", 0);
  }
  if (!node_->has_parameter("replay_duration_ns")) {
    node_->declare_parameter<int64_t>("replay_duration_ns", 0);
  }
  if (!node_->has_parameter("replay_initial_rate")) {
    node_->declare_parameter<double>("replay_initial_rate", 1.0);
  }
  /** @brief Python 编排层提供的精确时间轴和 CLI 初始倍率。 */
  replay_start_time_ns_ = node_->get_parameter("replay_start_time_ns").as_int();
  replay_duration_ns_ = node_->get_parameter("replay_duration_ns").as_int();
  playback_rate_ = node_->get_parameter("replay_initial_rate").as_double();
  current_time_ns_ = replay_start_time_ns_;
  update_timeline(current_time_ns_, true);
  rate_label_->setText(QString::number(playback_rate_, 'f', 2) + "×");
  start_client_ = node_->create_client<std_srvs::srv::Trigger>("/fastumi/episode/start");
  stop_client_ = node_->create_client<std_srvs::srv::Trigger>("/fastumi/episode/stop");
  finish_client_ = node_->create_client<std_srvs::srv::Trigger>(
    "/fastumi/replay/finish_and_save");
  clear_client_ = node_->create_client<std_srvs::srv::Trigger>(
    "/fastumi/replay/clear_annotations");
  list_annotations_client_ =
    node_->create_client<fastumi_interfaces::srv::ListEpisodeAnnotations>(
    "/fastumi/replay/list_episode_annotations");
  delete_annotation_client_ =
    node_->create_client<fastumi_interfaces::srv::DeleteEpisodeAnnotation>(
    "/fastumi/replay/delete_episode_annotation");
  pause_client_ = node_->create_client<rosbag2_interfaces::srv::Pause>("/rosbag2_player/pause");
  resume_client_ = node_->create_client<rosbag2_interfaces::srv::Resume>("/rosbag2_player/resume");
  seek_client_ = node_->create_client<rosbag2_interfaces::srv::Seek>("/rosbag2_player/seek");
  set_rate_client_ = node_->create_client<rosbag2_interfaces::srv::SetRate>(
    "/rosbag2_player/set_rate");
  get_rate_client_ = node_->create_client<rosbag2_interfaces::srv::GetRate>(
    "/rosbag2_player/get_rate");
  is_paused_client_ = node_->create_client<rosbag2_interfaces::srv::IsPaused>(
    "/rosbag2_player/is_paused");
  /** @brief 所有订阅只弱持有 gate，不能捕获裸 Panel 指针。 */
  const std::weak_ptr<CallbackGate> weak_gate(callback_gate_);
  gripper_subscription_ = node_->create_subscription<fastumi_interfaces::msg::GripperState>(
    "/gripper/state", 20,
    [weak_gate](const fastumi_interfaces::msg::GripperState::SharedPtr message) {
      const auto gate = weak_gate.lock();
      if (gate == nullptr) { return; }
      std::lock_guard<std::mutex> lock(gate->mutex);
      if (gate->panel != nullptr) { gate->panel->handle_gripper(message); }
    });
  pose_subscription_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/vive_tracker/pose", 20,
    [weak_gate](const geometry_msgs::msg::PoseStamped::SharedPtr message) {
      const auto gate = weak_gate.lock();
      if (gate == nullptr) { return; }
      std::lock_guard<std::mutex> lock(gate->mutex);
      if (gate->panel != nullptr) { gate->panel->handle_pose(message); }
    });
  clock_subscription_ = node_->create_subscription<rosgraph_msgs::msg::Clock>(
    "/clock", rclcpp::ClockQoS(),
    [weak_gate](const rosgraph_msgs::msg::Clock::SharedPtr message) {
      const auto gate = weak_gate.lock();
      if (gate == nullptr) { return; }
      std::lock_guard<std::mutex> lock(gate->mutex);
      if (gate->panel != nullptr) { gate->panel->handle_clock(message); }
    });
  refresh_controls();
}

/** @brief 将全局键盘事件分发到 episode、播放、保存和倍率控制。 */
bool ReplayAnnotationPanel::eventFilter(QObject * watched, QEvent * event)
{
  /** @brief 仅 KeyPress 参与快捷键分类，KeyRelease 继续交给 Qt。 */
  if (event->type() == QEvent::KeyPress) {
    /** @brief 当前应用级键盘事件。 */
    auto * key_event = static_cast<QKeyEvent *>(event);
    /** @brief Space 保持 episode START/STOP 语义。 */
    const bool is_space = key_event->key() == Qt::Key_Space;
    /** @brief 主键盘 Return 和数字小键盘 Enter 共享播放切换语义。 */
    const bool is_enter =
      key_event->key() == Qt::Key_Return || key_event->key() == Qt::Key_Enter;
    /** @brief Ctrl+S 只负责结束并保存，避免与普通 S 键的 RViz 操作冲突。 */
    const bool is_finish = key_event->key() == Qt::Key_S &&
      key_event->modifiers() == Qt::ControlModifier;
    /** @brief 左、上、右方向键分别固定选择 0.5×、1× 和 2×。 */
    const bool is_left = key_event->key() == Qt::Key_Left;
    const bool is_up = key_event->key() == Qt::Key_Up;
    const bool is_right = key_event->key() == Qt::Key_Right;
    /** @brief 纯状态分类结果决定触发操作、吞掉重复事件或继续传递。 */
    const ShortcutAction action = classify_shortcut(
      is_space, is_enter, is_finish, is_left, is_up, is_right,
      key_event->isAutoRepeat());
    switch (action) {
      case ShortcutAction::kSuppressAutoRepeat:
        return true;
      case ShortcutAction::kToggleEpisode:
        request_episode_toggle();
        return true;
      case ShortcutAction::kTogglePlayback:
        request_playback_toggle();
        return true;
      case ShortcutAction::kFinishAndSave:
        request_finish_and_save();
        return true;
      case ShortcutAction::kSetSlowRate:
      case ShortcutAction::kSetNormalRate:
      case ShortcutAction::kSetFastRate:
        request_rate(rate_for_shortcut(action));
        return true;
      case ShortcutAction::kNone:
        break;
    }
  }
  return rviz_common::Panel::eventFilter(watched, event);
}

/** @brief 根据服务发现、episode 和请求阶段刷新所有交互门禁。 */
void ReplayAnnotationPanel::refresh_controls()
{
  /** @brief 时间轴的非空阶段统一阻止 episode 和普通播放切换。 */
  const bool seek_busy = seek_state_.stage != SeekStage::kIdle;
  /** @brief 任一改变播放器或 episode 状态的请求都会阻止结束保存。 */
  const bool state_request_pending = request_pending_ || playback_request_pending_ ||
    rate_request_pending_ || seek_busy || finish_request_pending_ || clear_request_pending_ ||
    clear_confirmation_active_ || delete_request_pending_ || delete_confirmation_active_;
  /** @brief episode 控制同时依赖有效时钟、两个服务发现和无在途请求。 */
  const bool services_ready = start_client_ != nullptr && stop_client_ != nullptr &&
    start_client_->service_is_ready() && stop_client_->service_is_ready();
  const bool time_safe = annotation_time_safe(current_time_ns_, last_event_time_ns_);
  const bool enabled = episode_control_enabled(
    clock_ready_, services_ready, request_pending_, time_safe) && !seek_busy &&
    !playback_request_pending_ && !finish_request_pending_ && !clear_request_pending_ &&
    !clear_confirmation_active_ && !delete_request_pending_ && !delete_confirmation_active_;
  episode_button_->setEnabled(enabled);
  episode_button_->setText(episode_active_ ? "结束 episode（Space）" : "开始 episode（Space）");
  episode_button_->setToolTip(time_safe ? QString() :
    "当前正在回看较早帧；回放到最后一条标记之后才能继续打标");
  /** @brief 当前状态只依赖下一步需要调用的 Resume 或 Pause 服务。 */
  const bool playback_service_ready = playback_paused_ ?
    (resume_client_ != nullptr && resume_client_->service_is_ready()) :
    (pause_client_ != nullptr && pause_client_->service_is_ready());
  playback_button_->setEnabled(playback_control_enabled(
    playback_service_ready, playback_request_pending_) && !seek_busy &&
    !finish_request_pending_ && !clear_request_pending_ && !clear_confirmation_active_ &&
    !delete_request_pending_ && !delete_confirmation_active_);
  playback_button_->setText(
    playback_paused_ ? "继续回放（Enter）" : "暂停回放（Enter）");
  /** @brief 播放中拖拽需要 Pause、Seek 和 Resume 全部可用。 */
  const bool timeline_services_ready = pause_client_ != nullptr && resume_client_ != nullptr &&
    seek_client_ != nullptr && pause_client_->service_is_ready() &&
    resume_client_->service_is_ready() && seek_client_->service_is_ready();
  const bool metadata_ready = replay_start_time_ns_ > 0 && replay_duration_ns_ > 0;
  /** @brief 按下到释放之间保持滑块可交互，释放后的服务阶段才禁用。 */
  const bool drag_in_progress =
    (seek_state_.stage == SeekStage::kDragging || seek_state_.stage == SeekStage::kPausing) &&
    !seek_state_.release_received;
  const bool may_begin_drag = timeline_control_enabled(
    clock_ready_, metadata_ready, episode_active_, timeline_services_ready,
    seek_busy || playback_request_pending_ || finish_request_pending_ || clear_request_pending_ ||
    clear_confirmation_active_ || delete_request_pending_ || delete_confirmation_active_);
  timeline_slider_->setEnabled(drag_in_progress || may_begin_drag);
  /** @brief 列表跳转允许回看早于末次边界的位置，并始终保持暂停。 */
  const bool list_services_ready = pause_client_ != nullptr && seek_client_ != nullptr &&
    pause_client_->service_is_ready() && seek_client_->service_is_ready();
  const bool list_jump_enabled = episode_list_control_enabled(
    clock_ready_, metadata_ready, episode_active_, list_services_ready,
    state_request_pending);
  const bool delete_service_ready = delete_annotation_client_ != nullptr &&
    delete_annotation_client_->service_is_ready();
  const bool list_delete_enabled = delete_episode_control_enabled(
    delete_service_ready, true, episode_active_, state_request_pending);
  episode_list_->setEnabled(
    !episode_entries_.empty() && (list_jump_enabled || list_delete_enabled));
  /** @brief 结束按钮只有在 Python 服务就绪、时钟有效且所有边界已闭合时开放。 */
  const bool finish_service_ready = clock_ready_ && finish_client_ != nullptr &&
    finish_client_->service_is_ready();
  finish_button_->setEnabled(finish_control_enabled(
    finish_service_ready, episode_active_, state_request_pending));
  finish_button_->setText(finish_request_pending_ ?
    "正在结束并保存…" : "结束并保存（Ctrl+S）");
  /** @brief 清空允许删除活动 START，但必须已经存在标记且没有并发状态请求。 */
  const bool clear_service_ready = clock_ready_ && clear_client_ != nullptr &&
    clear_client_->service_is_ready();
  clear_button_->setEnabled(clear_control_enabled(
    clear_service_ready, has_annotations_, state_request_pending));
  clear_button_->setText(clear_request_pending_ ?
    "正在删除所有标记…" : "删除所有标记");
}

/** @brief 调用 START 或 STOP 服务，等待更高 revision 的权威快照提交 UI 状态。 */
void ReplayAnnotationPanel::request_episode_toggle()
{
  /** @brief 在服务未来完成前锁住控件，防止鼠标和全局空格重复提交。 */
  if (
    request_pending_ || playback_request_pending_ || finish_request_pending_ ||
    clear_request_pending_ || clear_confirmation_active_ ||
    delete_request_pending_ || delete_confirmation_active_ ||
    seek_state_.stage != SeekStage::kIdle || !clock_ready_ ||
    !annotation_time_safe(current_time_ns_, last_event_time_ns_) ||
    start_client_ == nullptr || stop_client_ == nullptr)
  {
    return;
  }
  auto client = episode_active_ ? stop_client_ : start_client_;
  if (!client->service_is_ready()) {
    refresh_controls();
    return;
  }
  request_pending_ = true;
  episode_request_succeeded_ = false;
  episode_request_min_revision_ = annotation_revision_ + 1;
  refresh_controls();
  QPointer<ReplayAnnotationPanel> panel(this);
  client->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>(),
    [panel](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
      /** @brief Trigger 结果决定继续等待权威 revision 或立即解除门控。 */
      bool request_succeeded = false;
      /** @brief 服务端状态或异常文本。 */
      std::string message;
      try {
        const auto response = future.get();
        request_succeeded = response->success;
        message = response->message;
      } catch (const std::exception & error) {
        message = error.what();
      }
      if (panel.isNull()) { return; }
      QMetaObject::invokeMethod(panel, [panel, request_succeeded, message]() {
        if (panel.isNull()) { return; }
        if (request_succeeded) {
          /** @brief 使服务提交前已经发出的列表响应全部失效。 */
          ++panel->annotation_query_epoch_;
          panel->episode_request_succeeded_ = true;
          panel->playback_hint_label_->setText("边界已提交，正在同步权威状态…");
          panel->poll_episode_annotations();
        } else {
          panel->request_pending_ = false;
          panel->episode_request_succeeded_ = false;
          panel->playback_hint_label_->setText(QString::fromStdString(message));
          if (panel->node_ != nullptr) {
            RCLCPP_WARN(
              panel->node_->get_logger(), "episode 请求失败: %s", message.c_str());
          }
        }
        panel->refresh_controls();
      }, Qt::QueuedConnection);
    });
}

/** @brief 根据当前暂停状态调用 Resume 或 Pause 服务。 */
void ReplayAnnotationPanel::request_playback_toggle()
{
  /** @brief 按钮和双 Enter 共用该入口，Space 不参与播放控制。 */
  if (playback_request_pending_ || finish_request_pending_ || clear_request_pending_ ||
    clear_confirmation_active_ || delete_request_pending_ || delete_confirmation_active_ ||
    seek_state_.stage != SeekStage::kIdle)
  {
    return;
  }
  /** @brief 请求成功后应提交的目标状态；Resume 为 false，Pause 为 true。 */
  const bool target_paused = !playback_paused_;
  if (playback_paused_) {
    if (resume_client_ == nullptr || !resume_client_->service_is_ready()) {
      refresh_controls();
      return;
    }
    playback_request_pending_ = true;
    refresh_controls();
    /** @brief 异步回调通过 QPointer 避免访问已析构 Panel。 */
    QPointer<ReplayAnnotationPanel> panel(this);
    resume_client_->async_send_request(std::make_shared<rosbag2_interfaces::srv::Resume::Request>(),
      [panel, target_paused](
        rclcpp::Client<rosbag2_interfaces::srv::Resume>::SharedFuture future)
      {
        /** @brief future 是否成功完成，决定是否提交目标播放状态。 */
        bool request_succeeded = true;
        /** @brief 失败时传递到 GUI 线程日志的异常文本。 */
        std::string error_message;
        try {
          future.get();
        } catch (const std::exception & error) {
          request_succeeded = false;
          error_message = error.what();
        }
        if (panel.isNull()) { return; }
        QMetaObject::invokeMethod(panel, [panel, target_paused, request_succeeded, error_message]() {
          if (panel.isNull()) { return; }
          /** @brief 归约 future 结果并始终解除播放请求门禁。 */
          const PlaybackUiState completed = complete_playback_request(
            PlaybackUiState{panel->playback_paused_, panel->playback_request_pending_},
            target_paused, request_succeeded);
          panel->playback_paused_ = completed.paused;
          panel->playback_request_pending_ = completed.request_pending;
          if (!request_succeeded && panel->node_ != nullptr) {
            RCLCPP_WARN(
              panel->node_->get_logger(), "继续回放请求失败: %s", error_message.c_str());
          }
          panel->refresh_controls();
        }, Qt::QueuedConnection);
      });
  } else {
    if (pause_client_ == nullptr || !pause_client_->service_is_ready()) {
      refresh_controls();
      return;
    }
    playback_request_pending_ = true;
    refresh_controls();
    /** @brief 异步回调通过 QPointer 避免访问已析构 Panel。 */
    QPointer<ReplayAnnotationPanel> panel(this);
    pause_client_->async_send_request(std::make_shared<rosbag2_interfaces::srv::Pause::Request>(),
      [panel, target_paused](
        rclcpp::Client<rosbag2_interfaces::srv::Pause>::SharedFuture future)
      {
        /** @brief future 是否成功完成，决定是否提交目标暂停状态。 */
        bool request_succeeded = true;
        /** @brief 失败时传递到 GUI 线程日志的异常文本。 */
        std::string error_message;
        try {
          future.get();
        } catch (const std::exception & error) {
          request_succeeded = false;
          error_message = error.what();
        }
        if (panel.isNull()) { return; }
        QMetaObject::invokeMethod(panel, [panel, target_paused, request_succeeded, error_message]() {
          if (panel.isNull()) { return; }
          /** @brief 归约 future 结果并始终解除播放请求门禁。 */
          const PlaybackUiState completed = complete_playback_request(
            PlaybackUiState{panel->playback_paused_, panel->playback_request_pending_},
            target_paused, request_succeeded);
          panel->playback_paused_ = completed.paused;
          panel->playback_request_pending_ = completed.request_pending;
          if (!request_succeeded && panel->node_ != nullptr) {
            RCLCPP_WARN(
              panel->node_->get_logger(), "暂停回放请求失败: %s", error_message.c_str());
          }
          panel->refresh_controls();
        }, Qt::QueuedConnection);
      });
  }
}

/** @brief 调用 Python 编排层服务，冻结已完成 episode 并结束回放。 */
void ReplayAnnotationPanel::request_finish_and_save()
{
  /** @brief 与所有改变状态的异步请求互斥，服务端会再次原子校验活动状态。 */
  if (
    finish_request_pending_ || clear_request_pending_ || clear_confirmation_active_ ||
    delete_request_pending_ || delete_confirmation_active_ ||
    request_pending_ || playback_request_pending_ ||
    rate_request_pending_ || seek_state_.stage != SeekStage::kIdle ||
    episode_active_ || !clock_ready_ || finish_client_ == nullptr ||
    !finish_client_->service_is_ready())
  {
    refresh_controls();
    return;
  }
  finish_request_pending_ = true;
  playback_hint_label_->setText("正在冻结标注并结束回放，请勿按 Ctrl+C…");
  refresh_controls();
  QPointer<ReplayAnnotationPanel> panel(this);
  finish_client_->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>(),
    [panel](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
      /** @brief 服务响应或异常决定是否保持结束门禁。 */
      bool request_succeeded = false;
      /** @brief 服务端返回的用户可读状态或异常文本。 */
      std::string message;
      try {
        const auto response = future.get();
        request_succeeded = response->success;
        message = response->message;
      } catch (const std::exception & error) {
        message = error.what();
      }
      if (panel.isNull()) { return; }
      QMetaObject::invokeMethod(panel, [panel, request_succeeded, message]() {
        if (panel.isNull()) { return; }
        /** @brief 接受后保持锁定直至 RViz 由编排层关闭，拒绝时恢复交互。 */
        panel->finish_request_pending_ = request_succeeded;
        panel->playback_hint_label_->setText(QString::fromStdString(message));
        if (!request_succeeded && panel->node_ != nullptr) {
          RCLCPP_WARN(
            panel->node_->get_logger(), "结束保存请求失败: %s", message.c_str());
        }
        panel->refresh_controls();
      }, Qt::QueuedConnection);
    });
}

/** @brief 二次确认后调用 Python 编排层服务原子删除全部标记。 */
void ReplayAnnotationPanel::request_clear_annotations()
{
  /** @brief 服务发现和所有状态请求共同门控，活动 episode 可随全部标记一并删除。 */
  if (
    clear_request_pending_ || clear_confirmation_active_ || finish_request_pending_ ||
    delete_request_pending_ || delete_confirmation_active_ ||
    request_pending_ ||
    playback_request_pending_ || rate_request_pending_ ||
    seek_state_.stage != SeekStage::kIdle || !has_annotations_ ||
    !clock_ready_ || clear_client_ == nullptr || !clear_client_->service_is_ready())
  {
    refresh_controls();
    return;
  }
  clear_confirmation_active_ = true;
  refresh_controls();
  const QMessageBox::StandardButton confirmation = QMessageBox::warning(
    this,
    "删除所有标记",
    "将删除当前会话内全部 START、STOP 和 ABORT 标记，并把 episode 编号重置为 0。\n"
    "此操作不可撤销，是否继续？",
    QMessageBox::Yes | QMessageBox::Cancel,
    QMessageBox::Cancel);
  clear_confirmation_active_ = false;
  if (confirmation != QMessageBox::Yes) {
    refresh_controls();
    return;
  }
  /** @brief 确认框停留期间状态可能变化，提交前再次执行完整门控。 */
  if (
    finish_request_pending_ || request_pending_ || playback_request_pending_ ||
    rate_request_pending_ || delete_request_pending_ || delete_confirmation_active_ ||
    seek_state_.stage != SeekStage::kIdle ||
    clear_client_ == nullptr || !clear_client_->service_is_ready())
  {
    refresh_controls();
    return;
  }
  clear_request_pending_ = true;
  clear_request_succeeded_ = false;
  clear_request_min_revision_ = annotation_revision_ + 1;
  playback_hint_label_->setText("正在删除全部标记…");
  refresh_controls();
  QPointer<ReplayAnnotationPanel> panel(this);
  clear_client_->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>(),
    [panel](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
      /** @brief 服务响应或异常决定是否重置 Panel 本地状态。 */
      bool request_succeeded = false;
      /** @brief 服务端返回的清空结果或异常文本。 */
      std::string message;
      try {
        const auto response = future.get();
        request_succeeded = response->success;
        message = response->message;
      } catch (const std::exception & error) {
        message = error.what();
      }
      if (panel.isNull()) { return; }
      QMetaObject::invokeMethod(panel, [panel, request_succeeded, message]() {
        if (panel.isNull()) { return; }
        if (request_succeeded) {
          /** @brief 仅允许服务提交后发出的列表请求确认清空结果。 */
          ++panel->annotation_query_epoch_;
          panel->clear_request_succeeded_ = true;
          panel->playback_hint_label_->setText("标记已删除，正在同步权威状态…");
          panel->poll_episode_annotations();
        } else {
          panel->clear_request_pending_ = false;
          panel->clear_request_succeeded_ = false;
          panel->playback_hint_label_->setText(QString::fromStdString(message));
        }
        if (!request_succeeded && panel->node_ != nullptr) {
          RCLCPP_WARN(
            panel->node_->get_logger(), "删除所有标记失败: %s", message.c_str());
        }
        panel->refresh_controls();
      }, Qt::QueuedConnection);
    });
}

/** @brief 定时向 Python 后端查询全部正常闭合 episode。 */
void ReplayAnnotationPanel::poll_episode_annotations()
{
  if (node_ == nullptr || list_query_pending_ ||
    list_annotations_client_ == nullptr ||
    !list_annotations_client_->service_is_ready())
  {
    return;
  }
  list_query_pending_ = true;
  /** @brief 请求发出时的代次，用于识别服务变更前的在途查询。 */
  const uint64_t query_epoch = annotation_query_epoch_;
  QPointer<ReplayAnnotationPanel> panel(this);
  list_annotations_client_->async_send_request(
    std::make_shared<fastumi_interfaces::srv::ListEpisodeAnnotations::Request>(),
    [panel, query_epoch](
      rclcpp::Client<fastumi_interfaces::srv::ListEpisodeAnnotations>::SharedFuture future)
    {
      /** @brief 成功响应中的完整列表副本。 */
      std::vector<fastumi_interfaces::msg::EpisodeAnnotation> episodes;
      /** @brief 异常只结束本轮轮询，下一周期自动重试。 */
      bool request_succeeded = false;
      /** @brief 权威状态快照是 Panel episode 状态的唯一事实来源。 */
      uint32_t next_episode_index = 0;
      /** @brief 查询时是否存在活动 episode。 */
      bool episode_active = false;
      /** @brief 查询时的最后边界时间。 */
      int64_t last_event_time_ns = 0;
      /** @brief 查询时是否存在任意边界。 */
      bool has_annotations = false;
      /** @brief 查询响应对应的后端单调状态版本。 */
      uint64_t revision = 0;
      try {
        const auto response = future.get();
        episodes = response->episodes;
        next_episode_index = response->next_episode_index;
        episode_active = response->episode_active;
        last_event_time_ns = stamp_to_ns(response->last_event_time);
        has_annotations = response->has_annotations;
        revision = response->revision;
        request_succeeded = true;
      } catch (const std::exception &) {
        request_succeeded = false;
      }
      if (panel.isNull()) { return; }
      QMetaObject::invokeMethod(panel,
        [panel, request_succeeded, episodes, next_episode_index,
          episode_active, last_event_time_ns, has_annotations, revision, query_epoch]() {
          if (panel.isNull()) { return; }
          panel->list_query_pending_ = false;
          if (request_succeeded && annotation_query_is_current(
              panel->annotation_query_epoch_, query_epoch))
          {
            panel->apply_annotation_snapshot(
              episodes, next_episode_index, episode_active,
              last_event_time_ns, has_annotations, revision);
          }
          panel->refresh_controls();
        }, Qt::QueuedConnection);
    });
}

/** @brief 用服务端权威摘要重建 episode 列表并保留选择。 */
void ReplayAnnotationPanel::apply_episode_annotations(
  const std::vector<fastumi_interfaces::msg::EpisodeAnnotation> & episodes)
{
  /** @brief 记录重建前的选中索引；无选择时使用最大值哨兵。 */
  uint32_t selected_index = std::numeric_limits<uint32_t>::max();
  if (episode_list_->currentItem() != nullptr) {
    selected_index = episode_list_->currentItem()->data(Qt::UserRole).toUInt();
  }
  /** @brief 将 ROS 消息转换为便于比较和渲染的稳定条目。 */
  std::vector<EpisodeListEntry> replacement_entries;
  replacement_entries.reserve(episodes.size());
  for (const auto & episode : episodes) {
    replacement_entries.push_back(EpisodeListEntry{
      episode.episode_index,
      stamp_to_ns(episode.start_time),
      stamp_to_ns(episode.end_time)});
  }
  completed_episode_count_ = static_cast<uint32_t>(replacement_entries.size());
  completed_episode_count_label_->setText(QString::number(completed_episode_count_));
  if (replacement_entries == episode_entries_) {
    return;
  }
  episode_entries_ = replacement_entries;
  episode_list_->clear();
  for (const auto & entry : episode_entries_) {
    /** @brief 列表时间以 bag 起点为零，便于和时间轴直接对应。 */
    const int64_t relative_start_ns = std::max<int64_t>(
      0, entry.start_time_ns - replay_start_time_ns_);
    const int64_t duration_ns = std::max<int64_t>(
      0, entry.end_time_ns - entry.start_time_ns);
    /** @brief 每项保存索引和绝对 START 时间，显示文本无需反向解析。 */
    auto * item = new QListWidgetItem(
      QString("episode %1　首帧 %2　时长 %3")
      .arg(entry.episode_index)
      .arg(QString::fromStdString(format_duration_ns(relative_start_ns)))
      .arg(QString::fromStdString(format_duration_ns(duration_ns))),
      episode_list_);
    item->setData(Qt::UserRole, entry.episode_index);
    item->setData(Qt::UserRole + 1, static_cast<qlonglong>(entry.start_time_ns));
    item->setToolTip("单击跳到首帧；右键删除这条完整 episode");
    if (entry.episode_index == selected_index) {
      episode_list_->setCurrentItem(item);
    }
  }
}

/** @brief 只应用不旧于本地版本的完整后端权威 episode 状态。 */
void ReplayAnnotationPanel::apply_annotation_snapshot(
  const std::vector<fastumi_interfaces::msg::EpisodeAnnotation> & episodes,
  const uint32_t next_episode_index, const bool episode_active,
  const int64_t last_event_time_ns, const bool has_annotations,
  const uint64_t revision)
{
  if (!should_apply_annotation_snapshot(annotation_revision_, revision)) {
    return;
  }
  annotation_revision_ = revision;
  episode_index_ = next_episode_index;
  episode_active_ = episode_active;
  last_event_time_ns_ = last_event_time_ns;
  has_annotations_ = has_annotations;
  apply_episode_annotations(episodes);
  episode_label_->setText(
    QString("episode %1：%2").arg(next_episode_index)
    .arg(episode_active ? "活动" : "空闲"));

  /** @brief 变更服务成功后，只有达到目标 revision 才解除对应门控。 */
  if (annotation_request_satisfied(
      request_pending_, episode_request_succeeded_,
      episode_request_min_revision_, revision))
  {
    request_pending_ = false;
    episode_request_succeeded_ = false;
    playback_hint_label_->setText("episode 状态已同步");
  }
  if (annotation_request_satisfied(
      clear_request_pending_, clear_request_succeeded_,
      clear_request_min_revision_, revision))
  {
    clear_request_pending_ = false;
    clear_request_succeeded_ = false;
    playback_hint_label_->setText("已删除所有标记，权威状态已同步");
  }
  if (annotation_request_satisfied(
      delete_request_pending_, delete_request_succeeded_,
      delete_request_min_revision_, revision))
  {
    delete_request_pending_ = false;
    delete_request_succeeded_ = false;
    playback_hint_label_->setText("episode 删除结果已同步");
  }
}

/** @brief 点选列表项后开始保持暂停的首帧跳转。 */
void ReplayAnnotationPanel::jump_to_episode(QListWidgetItem * item)
{
  if (item == nullptr || !episode_list_->isEnabled()) {
    return;
  }
  begin_episode_jump(
    item->data(Qt::UserRole).toUInt(),
    item->data(Qt::UserRole + 1).toLongLong());
}

/** @brief 发起无需末次边界钳制、完成后保持暂停的列表 Seek。 */
void ReplayAnnotationPanel::begin_episode_jump(
  const uint32_t episode_index, const int64_t start_time_ns)
{
  const bool metadata_ready = replay_start_time_ns_ > 0 && replay_duration_ns_ > 0;
  const bool services_ready = pause_client_ != nullptr && seek_client_ != nullptr &&
    pause_client_->service_is_ready() && seek_client_->service_is_ready();
  const bool state_request_pending = request_pending_ || playback_request_pending_ ||
    rate_request_pending_ || finish_request_pending_ || clear_request_pending_ ||
    clear_confirmation_active_ || delete_request_pending_ || delete_confirmation_active_ ||
    seek_state_.stage != SeekStage::kIdle;
  if (!episode_list_control_enabled(
      clock_ready_, metadata_ready, episode_active_, services_ready,
      state_request_pending))
  {
    refresh_controls();
    return;
  }
  /** @brief 服务端边界仍限制在 bag 首尾，但允许早于最后一条标记。 */
  const int64_t target_time_ns = std::clamp(
    start_time_ns, replay_start_time_ns_, replay_start_time_ns_ + replay_duration_ns_);
  seek_request_succeeded_ = false;
  seek_target_was_clamped_ = target_time_ns != start_time_ns;
  episode_list_seek_active_ = true;
  episode_list_seek_index_ = episode_index;
  seek_state_ = begin_seek_sequence(playback_paused_, false);
  seek_state_ = release_seek_sequence(seek_state_, target_time_ns);
  playback_hint_label_->setText(
    QString("正在跳转到 episode %1 首帧…").arg(episode_index));
  refresh_controls();
  if (seek_state_.stage == SeekStage::kPausing) {
    request_seek_pause();
  } else if (seek_state_.stage == SeekStage::kSeeking) {
    request_seek();
  }
}

/** @brief 在右键位置显示选中 episode 的删除菜单。 */
void ReplayAnnotationPanel::show_episode_context_menu(const QPoint & position)
{
  /** @brief 右键位置必须命中真实列表项。 */
  QListWidgetItem * item = episode_list_->itemAt(position);
  if (item == nullptr) {
    return;
  }
  episode_list_->setCurrentItem(item);
  const uint32_t episode_index = item->data(Qt::UserRole).toUInt();
  const bool service_ready = delete_annotation_client_ != nullptr &&
    delete_annotation_client_->service_is_ready();
  const bool request_pending = request_pending_ || playback_request_pending_ ||
    rate_request_pending_ || finish_request_pending_ || clear_request_pending_ ||
    clear_confirmation_active_ || delete_request_pending_ || delete_confirmation_active_ ||
    seek_state_.stage != SeekStage::kIdle;
  const bool can_delete = delete_episode_control_enabled(
    service_ready, true, episode_active_, request_pending);
  QMenu menu(this);
  QAction * delete_action = menu.addAction(
    QString("删除 episode %1").arg(episode_index));
  delete_action->setEnabled(can_delete);
  /** @brief 菜单的嵌套事件循环期间屏蔽全局快捷键。 */
  delete_confirmation_active_ = true;
  refresh_controls();
  QAction * selected_action = menu.exec(episode_list_->viewport()->mapToGlobal(position));
  delete_confirmation_active_ = false;
  refresh_controls();
  if (selected_action == delete_action && can_delete) {
    request_delete_episode(episode_index);
  }
}

/** @brief 二次确认并请求后端原子删除指定完整 episode。 */
void ReplayAnnotationPanel::request_delete_episode(const uint32_t episode_index)
{
  const bool service_ready = delete_annotation_client_ != nullptr &&
    delete_annotation_client_->service_is_ready();
  const bool request_pending = request_pending_ || playback_request_pending_ ||
    rate_request_pending_ || finish_request_pending_ || clear_request_pending_ ||
    clear_confirmation_active_ || delete_request_pending_ || delete_confirmation_active_ ||
    seek_state_.stage != SeekStage::kIdle;
  if (!delete_episode_control_enabled(
      service_ready, true, episode_active_, request_pending))
  {
    refresh_controls();
    return;
  }
  delete_confirmation_active_ = true;
  refresh_controls();
  const QMessageBox::StandardButton confirmation = QMessageBox::warning(
    this,
    QString("删除 episode %1").arg(episode_index),
    QString("将删除 episode %1 的 START/STOP 标记，并自动重编号后续 episode。\n"
      "此操作不可撤销，是否继续？").arg(episode_index),
    QMessageBox::Yes | QMessageBox::Cancel,
    QMessageBox::Cancel);
  delete_confirmation_active_ = false;
  if (confirmation != QMessageBox::Yes || episode_active_ ||
    delete_annotation_client_ == nullptr ||
    !delete_annotation_client_->service_is_ready())
  {
    refresh_controls();
    return;
  }
  delete_request_pending_ = true;
  delete_request_succeeded_ = false;
  delete_request_min_revision_ = annotation_revision_ + 1;
  playback_hint_label_->setText(
    QString("正在删除 episode %1…").arg(episode_index));
  refresh_controls();
  auto request =
    std::make_shared<fastumi_interfaces::srv::DeleteEpisodeAnnotation::Request>();
  request->episode_index = episode_index;
  QPointer<ReplayAnnotationPanel> panel(this);
  delete_annotation_client_->async_send_request(request,
    [panel](
      rclcpp::Client<fastumi_interfaces::srv::DeleteEpisodeAnnotation>::SharedFuture future)
    {
      /** @brief 响应中的完整列表和状态只在成功时提交。 */
      bool request_succeeded = false;
      /** @brief 服务端用户可读结果或异常。 */
      std::string message;
      /** @brief 删除后的完整 episode 摘要。 */
      std::vector<fastumi_interfaces::msg::EpisodeAnnotation> episodes;
      /** @brief 删除后的下一 episode 索引。 */
      uint32_t next_episode_index = 0;
      /** @brief 删除响应对应权威状态是否仍有活动 episode。 */
      bool episode_active = false;
      /** @brief 删除后的最后边界绝对时间。 */
      int64_t last_event_time_ns = 0;
      /** @brief 删除后是否仍存在任意边界。 */
      bool has_annotations = false;
      /** @brief 删除响应对应的后端单调状态版本。 */
      uint64_t revision = 0;
      try {
        const auto response = future.get();
        request_succeeded = response->success;
        message = response->message;
        episodes = response->episodes;
        next_episode_index = response->next_episode_index;
        episode_active = response->episode_active;
        last_event_time_ns = stamp_to_ns(response->last_event_time);
        has_annotations = response->has_annotations;
        revision = response->revision;
      } catch (const std::exception & error) {
        message = error.what();
      }
      if (panel.isNull()) { return; }
      QMetaObject::invokeMethod(panel,
        [panel, request_succeeded, message, episodes, next_episode_index,
          episode_active, last_event_time_ns, has_annotations, revision]() {
          if (panel.isNull()) { return; }
          if (request_succeeded) {
            /** @brief 被删条目的选择不转移到重编号后的另一条 episode。 */
            panel->episode_list_->setCurrentRow(-1);
            /** @brief 删除响应本身是提交后快照，同时废弃更早的列表请求。 */
            ++panel->annotation_query_epoch_;
            panel->delete_request_succeeded_ = true;
            panel->apply_annotation_snapshot(
              episodes, next_episode_index, episode_active,
              last_event_time_ns, has_annotations, revision);
            if (panel->delete_request_pending_) {
              panel->poll_episode_annotations();
            }
          } else {
            panel->delete_request_pending_ = false;
            panel->delete_request_succeeded_ = false;
            panel->playback_hint_label_->setText(QString::fromStdString(message));
          }
          if (!request_succeeded && panel->node_ != nullptr) {
            RCLCPP_WARN(
              panel->node_->get_logger(), "删除单条 episode 失败: %s", message.c_str());
          }
          panel->refresh_controls();
        }, Qt::QueuedConnection);
    });
}

/** @brief 调用 SetRate 并在响应成功后提交固定目标倍率。 */
void ReplayAnnotationPanel::request_rate(const double target_rate)
{
  /** @brief 倍率切换允许在活动 episode 中使用，Seek 或 SetRate 在途时拒绝并发。 */
  if (
    rate_request_pending_ || finish_request_pending_ || clear_request_pending_ ||
    clear_confirmation_active_ || delete_request_pending_ || delete_confirmation_active_ ||
    seek_state_.stage != SeekStage::kIdle ||
    set_rate_client_ == nullptr || !set_rate_client_->service_is_ready())
  {
    return;
  }
  if (std::abs(playback_rate_ - target_rate) < 1e-9) {
    playback_hint_label_->setText(QString("当前已是 %1×").arg(target_rate, 0, 'f', 2));
    return;
  }
  rate_request_pending_ = true;
  playback_hint_label_->setText(QString("正在切换到 %1×…").arg(target_rate, 0, 'f', 2));
  /** @brief SetRate 请求只携带一个严格正倍率。 */
  auto request = std::make_shared<rosbag2_interfaces::srv::SetRate::Request>();
  request->rate = target_rate;
  QPointer<ReplayAnnotationPanel> panel(this);
  set_rate_client_->async_send_request(request,
    [panel, target_rate](rclcpp::Client<rosbag2_interfaces::srv::SetRate>::SharedFuture future) {
      /** @brief success 字段和异常共同决定是否提交倍率。 */
      bool request_succeeded = false;
      /** @brief 失败时转发到 GUI 线程的诊断文本。 */
      std::string error_message;
      try {
        request_succeeded = future.get()->success;
      } catch (const std::exception & error) {
        error_message = error.what();
      }
      if (panel.isNull()) { return; }
      QMetaObject::invokeMethod(panel,
        [panel, target_rate, request_succeeded, error_message]() {
          if (panel.isNull()) { return; }
          const RateUiState completed = complete_rate_request(
            RateUiState{panel->playback_rate_, panel->rate_request_pending_},
            target_rate, request_succeeded);
          panel->playback_rate_ = completed.rate;
          panel->rate_request_pending_ = completed.request_pending;
          panel->rate_label_->setText(
            QString::number(panel->playback_rate_, 'f', 2) + "×");
          panel->playback_hint_label_->setText(request_succeeded ?
            QString("播放倍率：%1×").arg(target_rate, 0, 'f', 2) :
            QString("倍率切换失败，保持 %1×").arg(panel->playback_rate_, 0, 'f', 2));
          if (!request_succeeded && panel->node_ != nullptr) {
            RCLCPP_WARN(
              panel->node_->get_logger(), "设置回放倍率失败: %s",
              error_message.empty() ? "播放器拒绝目标倍率" : error_message.c_str());
          }
          panel->refresh_controls();
        }, Qt::QueuedConnection);
    });
}

/** @brief 开始拖拽；播放中先进入异步暂停阶段。 */
void ReplayAnnotationPanel::begin_timeline_drag()
{
  if (!timeline_slider_->isEnabled() || seek_state_.stage != SeekStage::kIdle) {
    return;
  }
  seek_request_succeeded_ = false;
  seek_target_was_clamped_ = false;
  seek_state_ = begin_seek_sequence(playback_paused_);
  playback_hint_label_->setText(
    playback_paused_ ? "拖拽后保持暂停" : "正在临时暂停回放…");
  refresh_controls();
  if (seek_state_.stage == SeekStage::kPausing) {
    request_seek_pause();
  }
}

/** @brief 显示拖拽预览并限制末次 episode 边界。 */
void ReplayAnnotationPanel::preview_timeline_drag(const int slider_value)
{
  if (seek_state_.stage == SeekStage::kIdle) {
    return;
  }
  /** @brief 滑块刻度先换成绝对时间，再限制到末次事件边界。 */
  const int64_t requested_time_ns = time_for_slider_value(
    replay_start_time_ns_, replay_duration_ns_, slider_value);
  /** @brief 未产生事件时允许回到 bag 起点。 */
  const int64_t minimum_time_ns = std::max(replay_start_time_ns_, last_event_time_ns_);
  const int64_t target_time_ns = clamp_seek_time(
    requested_time_ns, minimum_time_ns, replay_start_time_ns_ + replay_duration_ns_);
  /** @brief 边界钳制后的刻度同步回滑块，避免释放位置与请求不一致。 */
  const int target_slider_value = slider_value_for_time(
    replay_start_time_ns_, replay_duration_ns_, target_time_ns);
  if (target_slider_value != timeline_slider_->sliderPosition()) {
    timeline_slider_->setSliderPosition(target_slider_value);
  }
  update_timeline(target_time_ns, false);
  if (target_time_ns != requested_time_ns) {
    playback_hint_label_->setText("已限制到最后一条 episode 边界");
  }
}

/** @brief 记录最终目标并在播放器暂停后发起 Seek。 */
void ReplayAnnotationPanel::release_timeline_drag()
{
  if (seek_state_.stage == SeekStage::kIdle) {
    return;
  }
  /** @brief 释放位置再次执行边界钳制，防止信号顺序绕过 sliderMoved。 */
  const int64_t requested_time_ns = time_for_slider_value(
    replay_start_time_ns_, replay_duration_ns_, timeline_slider_->sliderPosition());
  const int64_t minimum_time_ns = std::max(replay_start_time_ns_, last_event_time_ns_);
  const int64_t target_time_ns = clamp_seek_time(
    requested_time_ns, minimum_time_ns, replay_start_time_ns_ + replay_duration_ns_);
  seek_target_was_clamped_ = target_time_ns != requested_time_ns;
  seek_state_ = release_seek_sequence(seek_state_, target_time_ns);
  if (seek_state_.stage == SeekStage::kSeeking) {
    request_seek();
  } else {
    playback_hint_label_->setText("等待暂停完成后跳转…");
  }
  refresh_controls();
}

/** @brief 为播放中的拖拽异步提交临时 Pause。 */
void ReplayAnnotationPanel::request_seek_pause()
{
  if (pause_client_ == nullptr || !pause_client_->service_is_ready()) {
    finish_seek_sequence("暂停服务尚未就绪，已取消跳转");
    return;
  }
  QPointer<ReplayAnnotationPanel> panel(this);
  pause_client_->async_send_request(std::make_shared<rosbag2_interfaces::srv::Pause::Request>(),
    [panel](rclcpp::Client<rosbag2_interfaces::srv::Pause>::SharedFuture future) {
      /** @brief 空响应服务以 future 是否抛异常判定完成。 */
      bool request_succeeded = true;
      /** @brief 失败诊断文本。 */
      std::string error_message;
      try {
        future.get();
      } catch (const std::exception & error) {
        request_succeeded = false;
        error_message = error.what();
      }
      if (panel.isNull()) { return; }
      QMetaObject::invokeMethod(panel, [panel, request_succeeded, error_message]() {
        if (panel.isNull()) { return; }
        panel->seek_state_ = complete_seek_pause(panel->seek_state_, request_succeeded);
        if (!request_succeeded) {
          if (panel->node_ != nullptr) {
            RCLCPP_WARN(
              panel->node_->get_logger(), "拖拽前暂停回放失败: %s", error_message.c_str());
          }
          panel->finish_seek_sequence("暂停失败，已取消跳转");
          return;
        }
        panel->playback_paused_ = true;
        if (panel->seek_state_.stage == SeekStage::kSeeking) {
          panel->request_seek();
        } else {
          panel->playback_hint_label_->setText("已暂停，释放滑块后跳转");
          panel->refresh_controls();
        }
      }, Qt::QueuedConnection);
    });
}

/** @brief 将归一化滑块目标作为绝对 ROS 时间提交给 Seek 服务。 */
void ReplayAnnotationPanel::request_seek()
{
  if (seek_client_ == nullptr || !seek_client_->service_is_ready()) {
    seek_request_succeeded_ = false;
    seek_state_ = complete_seek_request(seek_state_);
    if (seek_state_.stage == SeekStage::kResuming) {
      request_seek_resume();
    } else {
      finish_seek_sequence("Seek 服务尚未就绪，已取消跳转");
    }
    return;
  }
  playback_hint_label_->setText("正在跳转…");
  /** @brief Seek.srv 使用绝对 ROS 时间。 */
  auto request = std::make_shared<rosbag2_interfaces::srv::Seek::Request>();
  request->time.sec = static_cast<int32_t>(seek_state_.target_time_ns / 1000000000LL);
  request->time.nanosec = static_cast<uint32_t>(seek_state_.target_time_ns % 1000000000LL);
  QPointer<ReplayAnnotationPanel> panel(this);
  seek_client_->async_send_request(request,
    [panel](rclcpp::Client<rosbag2_interfaces::srv::Seek>::SharedFuture future) {
      /** @brief Seek 必须同时满足 future 完成和 response.success。 */
      bool request_succeeded = false;
      /** @brief 失败诊断文本。 */
      std::string error_message;
      try {
        request_succeeded = future.get()->success;
      } catch (const std::exception & error) {
        error_message = error.what();
      }
      if (panel.isNull()) { return; }
      QMetaObject::invokeMethod(panel, [panel, request_succeeded, error_message]() {
        if (panel.isNull()) { return; }
        panel->seek_request_succeeded_ = request_succeeded;
        if (request_succeeded) {
          panel->current_time_ns_ = panel->seek_state_.target_time_ns;
          panel->update_timeline(panel->current_time_ns_, true);
        } else if (panel->node_ != nullptr) {
          RCLCPP_WARN(
            panel->node_->get_logger(), "回放 Seek 失败: %s",
            error_message.empty() ? "播放器拒绝目标时间" : error_message.c_str());
        }
        panel->seek_state_ = complete_seek_request(panel->seek_state_);
        if (panel->seek_state_.stage == SeekStage::kResuming) {
          panel->request_seek_resume();
        } else {
          if (!request_succeeded) {
            panel->finish_seek_sequence("跳转失败，保持原位置");
          } else if (panel->episode_list_seek_active_) {
            panel->finish_seek_sequence(
              QString("已跳转到 episode %1 首帧，保持暂停")
              .arg(panel->episode_list_seek_index_));
          } else if (panel->seek_target_was_clamped_) {
            panel->finish_seek_sequence("已限制到最后一条 episode 边界，保持暂停");
          } else {
            panel->finish_seek_sequence("跳转完成，保持暂停");
          }
        }
      }, Qt::QueuedConnection);
    });
}

/** @brief Seek 完成后恢复拖拽前的播放状态。 */
void ReplayAnnotationPanel::request_seek_resume()
{
  if (resume_client_ == nullptr || !resume_client_->service_is_ready()) {
    playback_paused_ = true;
    seek_state_ = complete_seek_resume(seek_state_);
    finish_seek_sequence("恢复播放服务未就绪，当前保持暂停");
    return;
  }
  playback_hint_label_->setText(
    seek_request_succeeded_ ? "跳转完成，正在恢复播放…" : "跳转失败，正在恢复播放…");
  QPointer<ReplayAnnotationPanel> panel(this);
  resume_client_->async_send_request(std::make_shared<rosbag2_interfaces::srv::Resume::Request>(),
    [panel](rclcpp::Client<rosbag2_interfaces::srv::Resume>::SharedFuture future) {
      /** @brief Resume 空响应以异常作为失败信号。 */
      bool request_succeeded = true;
      /** @brief 失败诊断文本。 */
      std::string error_message;
      try {
        future.get();
      } catch (const std::exception & error) {
        request_succeeded = false;
        error_message = error.what();
      }
      if (panel.isNull()) { return; }
      QMetaObject::invokeMethod(panel, [panel, request_succeeded, error_message]() {
        if (panel.isNull()) { return; }
        panel->playback_paused_ = !request_succeeded;
        panel->seek_state_ = complete_seek_resume(panel->seek_state_);
        if (!request_succeeded && panel->node_ != nullptr) {
          RCLCPP_WARN(
            panel->node_->get_logger(), "Seek 后恢复播放失败: %s", error_message.c_str());
        }
        if (!request_succeeded) {
          panel->finish_seek_sequence("恢复播放失败，当前保持暂停");
        } else if (panel->seek_request_succeeded_) {
          panel->finish_seek_sequence(panel->seek_target_was_clamped_ ?
            "已限制到最后一条 episode 边界，已恢复播放" :
            "跳转完成，已恢复播放");
        } else {
          panel->finish_seek_sequence("跳转失败，已恢复原播放状态");
        }
      }, Qt::QueuedConnection);
    });
}

/** @brief 结束 Seek 状态机并恢复基于最新时钟的控件显示。 */
void ReplayAnnotationPanel::finish_seek_sequence(const QString & status_text)
{
  seek_state_ = SeekUiState{};
  episode_list_seek_active_ = false;
  episode_list_seek_index_ = 0;
  update_timeline(current_time_ns_, true);
  playback_hint_label_->setText(status_text);
  refresh_controls();
}

/** @brief 把绝对时间映射到滑块和当前/总时长文本。 */
void ReplayAnnotationPanel::update_timeline(
  const int64_t absolute_time_ns, const bool update_slider)
{
  if (replay_start_time_ns_ <= 0 || replay_duration_ns_ <= 0) {
    timeline_label_->setText("00:00:00.000 / 00:00:00.000");
    return;
  }
  /** @brief 显示时间始终限制在 MCAP 边界。 */
  const int64_t elapsed_ns = std::clamp(
    absolute_time_ns - replay_start_time_ns_, int64_t{0}, replay_duration_ns_);
  if (update_slider) {
    timeline_slider_->setValue(slider_value_for_time(
      replay_start_time_ns_, replay_duration_ns_, absolute_time_ns));
  }
  timeline_label_->setText(QString::fromStdString(
    format_duration_ns(elapsed_ns) + " / " + format_duration_ns(replay_duration_ns_)));
}

/** @brief 定时查询 Player 的权威 Pause 和 Rate 状态。 */
void ReplayAnnotationPanel::poll_player_state()
{
  if (node_ == nullptr || seek_state_.stage != SeekStage::kIdle) {
    return;
  }
  if (
    !pause_query_pending_ && !playback_request_pending_ && is_paused_client_ != nullptr &&
    is_paused_client_->service_is_ready())
  {
    pause_query_pending_ = true;
    QPointer<ReplayAnnotationPanel> panel(this);
    is_paused_client_->async_send_request(
      std::make_shared<rosbag2_interfaces::srv::IsPaused::Request>(),
      [panel](rclcpp::Client<rosbag2_interfaces::srv::IsPaused>::SharedFuture future) {
        /** @brief 查询失败只解除查询门禁，不覆盖最近确认状态。 */
        bool request_succeeded = false;
        /** @brief 成功响应中的暂停事实。 */
        bool paused = true;
        try {
          paused = future.get()->paused;
          request_succeeded = true;
        } catch (const std::exception &) {
          request_succeeded = false;
        }
        if (panel.isNull()) { return; }
        QMetaObject::invokeMethod(panel, [panel, request_succeeded, paused]() {
          if (panel.isNull()) { return; }
          panel->pause_query_pending_ = false;
          if (request_succeeded && !panel->playback_request_pending_ &&
            panel->seek_state_.stage == SeekStage::kIdle)
          {
            panel->playback_paused_ = paused;
          }
          panel->refresh_controls();
        }, Qt::QueuedConnection);
      });
  }
  if (
    !rate_query_pending_ && !rate_request_pending_ && get_rate_client_ != nullptr &&
    get_rate_client_->service_is_ready())
  {
    rate_query_pending_ = true;
    QPointer<ReplayAnnotationPanel> panel(this);
    get_rate_client_->async_send_request(
      std::make_shared<rosbag2_interfaces::srv::GetRate::Request>(),
      [panel](rclcpp::Client<rosbag2_interfaces::srv::GetRate>::SharedFuture future) {
        /** @brief 查询成功后提交的播放器倍率。 */
        double rate = 0.0;
        try {
          rate = future.get()->rate;
        } catch (const std::exception &) {
          rate = 0.0;
        }
        if (panel.isNull()) { return; }
        QMetaObject::invokeMethod(panel, [panel, rate]() {
          if (panel.isNull()) { return; }
          panel->rate_query_pending_ = false;
          if (rate > 0.0 && !panel->rate_request_pending_) {
            panel->playback_rate_ = rate;
            panel->rate_label_->setText(QString::number(rate, 'f', 2) + "×");
          }
        }, Qt::QueuedConnection);
      });
  }
}

/** @brief 把夹爪状态安全排队到 GUI 线程。 */
void ReplayAnnotationPanel::handle_gripper(const fastumi_interfaces::msg::GripperState::SharedPtr message)
{
  /** @brief ROS 回调只复制值并排队，绝不直接触碰 QWidget。 */
  const float openness = message->filtered_openness;
  const bool valid = message->valid;
  QMetaObject::invokeMethod(this, [this, openness, valid]() {
    const bool finite = std::isfinite(openness);
    const float clamped = finite ? std::clamp(openness, 0.0F, 1.0F) : 0.0F;
    gripper_label_->setText(finite ? QString::number(clamped, 'f', 3) : "NaN");
    gripper_bar_->setValue(static_cast<int>(clamped * 1000.0F));
    gripper_valid_label_->setText(valid ? "有效" : "无效");
  }, Qt::QueuedConnection);
}

/** @brief 把 Tracker 位姿安全排队到 GUI 线程。 */
void ReplayAnnotationPanel::handle_pose(const geometry_msgs::msg::PoseStamped::SharedPtr message)
{
  /** @brief 以米和 xyzw 显示 Tracker 位姿，时效代表收到新样本。 */
  const auto pose = message->pose;
  QMetaObject::invokeMethod(this, [this, pose]() {
    pose_label_->setText(QString("p=(%1, %2, %3) q=(%4, %5, %6, %7)")
      .arg(pose.position.x, 0, 'f', 3).arg(pose.position.y, 0, 'f', 3)
      .arg(pose.position.z, 0, 'f', 3).arg(pose.orientation.x, 0, 'f', 3)
      .arg(pose.orientation.y, 0, 'f', 3).arg(pose.orientation.z, 0, 'f', 3)
      .arg(pose.orientation.w, 0, 'f', 3));
    freshness_label_->setText("已收到当前回放样本");
  }, Qt::QueuedConnection);
}

/** @brief 使用非零 `/clock` 更新回放进度事实。 */
void ReplayAnnotationPanel::handle_clock(const rosgraph_msgs::msg::Clock::SharedPtr message)
{
  /** @brief 只有非零 bag 时钟才允许提交边界，避免系统时钟污染事件。 */
  const bool nonzero = message->clock.sec > 0 || message->clock.nanosec > 0;
  if (!nonzero) { return; }
  /** @brief ROS 时间消息转换为绝对纳秒，作为进度事实来源。 */
  const int64_t timestamp_ns = static_cast<int64_t>(message->clock.sec) * 1000000000LL +
    static_cast<int64_t>(message->clock.nanosec);
  QMetaObject::invokeMethod(this, [this, timestamp_ns]() {
    clock_ready_ = true;
    current_time_ns_ = timestamp_ns;
    if (seek_state_.stage == SeekStage::kIdle) {
      update_timeline(current_time_ns_, true);
    }
    refresh_controls();
  }, Qt::QueuedConnection);
}

}  // namespace fastumi_rviz_plugins

PLUGINLIB_EXPORT_CLASS(fastumi_rviz_plugins::ReplayAnnotationPanel, rviz_common::Panel)
