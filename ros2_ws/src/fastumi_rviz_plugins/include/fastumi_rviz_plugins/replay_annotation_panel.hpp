/**
 * @file replay_annotation_panel.hpp
 * @brief 声明线程安全更新的 FastUMI 回放标注 RViz2 Panel。
 */

#ifndef FASTUMI_RVIZ_PLUGINS__REPLAY_ANNOTATION_PANEL_HPP_
#define FASTUMI_RVIZ_PLUGINS__REPLAY_ANNOTATION_PANEL_HPP_

#include <memory>
#include <mutex>
#include <vector>

#include <QPoint>
#include <QPointer>
#include <rviz_common/panel.hpp>

#include "fastumi_rviz_plugins/panel_state.hpp"

#include <fastumi_interfaces/msg/episode_annotation.hpp>
#include <fastumi_interfaces/msg/gripper_state.hpp>
#include <fastumi_interfaces/srv/delete_episode_annotation.hpp>
#include <fastumi_interfaces/srv/list_episode_annotations.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_interfaces/srv/pause.hpp>
#include <rosbag2_interfaces/srv/get_rate.hpp>
#include <rosbag2_interfaces/srv/is_paused.hpp>
#include <rosbag2_interfaces/srv/resume.hpp>
#include <rosbag2_interfaces/srv/seek.hpp>
#include <rosbag2_interfaces/srv/set_rate.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <std_srvs/srv/trigger.hpp>

class QLabel;
class QListWidget;
class QListWidgetItem;
class QProgressBar;
class QPushButton;
class QSlider;
class QString;
class QTimer;

namespace fastumi_rviz_plugins
{

/** @brief 为回放 MCAP 提供状态显示、episode 和播放控制的面板。 */
class ReplayAnnotationPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  /** @brief 构造尚未绑定 RViz ROS 节点的面板。 */
  explicit ReplayAnnotationPanel(QWidget * parent = nullptr);
  /** @brief 取消 Qt 事件过滤器，释放 ROS 接口。 */
  ~ReplayAnnotationPanel() override;

  /** @brief 在获得 RViz DisplayContext 后创建订阅和服务客户端。 */
  void onInitialize() override;

protected:
  /** @brief 捕获全局 Space、双 Enter、Ctrl+S 和三档倍率方向键。 */
  bool eventFilter(QObject * watched, QEvent * event) override;

private Q_SLOTS:
  /** @brief 根据当前空闲/活动状态调用 START 或 STOP 服务。 */
  void request_episode_toggle();
  /** @brief 使用 rosbag2_player 的 pause/resume 服务切换播放状态。 */
  void request_playback_toggle();
  /** @brief 请求冻结当前已闭合 episode、停止回放并保存输出。 */
  void request_finish_and_save();
  /** @brief 二次确认后删除当前会话内全部 episode 标记。 */
  void request_clear_annotations();
  /** @brief 开始时间轴拖拽，播放中先异步暂停。 */
  void begin_timeline_drag();
  /** @brief 更新拖拽预览并钳制末次 episode 边界。 */
  void preview_timeline_drag(int slider_value);
  /** @brief 释放时间轴并在暂停完成后执行 Seek。 */
  void release_timeline_drag();
  /** @brief 定时从 rosbag2 Player 同步暂停状态和倍率。 */
  void poll_player_state();
  /** @brief 定时从 Python 后端同步完整 episode 权威列表。 */
  void poll_episode_annotations();
  /** @brief 点选列表项后暂停并跳到该 episode 的 START 帧。 */
  void jump_to_episode(QListWidgetItem * item);
  /** @brief 在列表项右键位置提供单条删除菜单。 */
  void show_episode_context_menu(const QPoint & position);

private:
  /** @brief 协调 ROS 回调与 Panel 析构的共享访问屏障。 */
  struct CallbackGate
  {
    /** @brief 保护 panel 指针和 callback 到析构的交接。 */
    std::mutex mutex;
    /** @brief 非空时允许回调向该存活 Panel 排队 Qt 更新。 */
    ReplayAnnotationPanel * panel{nullptr};
  };

  /** @brief 保存列表渲染和跳转需要的稳定 episode 摘要。 */
  struct EpisodeListEntry
  {
    /** @brief 当前事件序列内的 episode 索引。 */
    uint32_t episode_index{0};
    /** @brief START 边界绝对时间。 */
    int64_t start_time_ns{0};
    /** @brief STOP 边界绝对时间。 */
    int64_t end_time_ns{0};

    /** @brief 比较权威列表快照，避免轮询时无谓重建 QListWidget。 */
    bool operator==(const EpisodeListEntry & other) const
    {
      return episode_index == other.episode_index &&
             start_time_ns == other.start_time_ns &&
             end_time_ns == other.end_time_ns;
    }
  };

  /** @brief 在 GUI 线程设置各按钮的可用性和文字。 */
  void refresh_controls();
  /** @brief 请求固定目标倍率并在成功响应后更新显示。 */
  void request_rate(double target_rate);
  /** @brief 为播放中的拖拽提交临时 Pause。 */
  void request_seek_pause();
  /** @brief 提交当前拖拽目标的绝对 Seek 请求。 */
  void request_seek();
  /** @brief Seek 结束后恢复拖拽前的播放状态。 */
  void request_seek_resume();
  /** @brief 根据绝对时间刷新滑块和当前/总时长标签。 */
  void update_timeline(int64_t absolute_time_ns, bool update_slider);
  /** @brief 用后端响应替换列表并尽量保留当前选中索引。 */
  void apply_episode_annotations(
    const std::vector<fastumi_interfaces::msg::EpisodeAnnotation> & episodes);
  /** @brief 按 revision 原子应用后端权威 episode 状态。 */
  void apply_annotation_snapshot(
    const std::vector<fastumi_interfaces::msg::EpisodeAnnotation> & episodes,
    uint32_t next_episode_index, bool episode_active, int64_t last_event_time_ns,
    bool has_annotations, uint64_t revision);
  /** @brief 发起保持暂停的 episode 首帧 Seek 序列。 */
  void begin_episode_jump(uint32_t episode_index, int64_t start_time_ns);
  /** @brief 二次确认并请求后端原子删除指定完整 episode。 */
  void request_delete_episode(uint32_t episode_index);
  /** @brief 取消或完成 Seek 序列，并将滑块恢复到最新时钟。 */
  void finish_seek_sequence(const QString & status_text);
  /** @brief 将夹爪状态回调排队到 GUI 线程显示。 */
  void handle_gripper(const fastumi_interfaces::msg::GripperState::SharedPtr message);
  /** @brief 将 Tracker 位姿回调排队到 GUI 线程显示。 */
  void handle_pose(const geometry_msgs::msg::PoseStamped::SharedPtr message);
  /** @brief 将有效模拟时钟状态排队到 GUI 线程。 */
  void handle_clock(const rosgraph_msgs::msg::Clock::SharedPtr message);

  /** @brief RViz 提供的共享 ROS 节点。 */
  rclcpp::Node::SharedPtr node_;
  /** @brief START 服务客户端。 */
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr start_client_;
  /** @brief STOP 服务客户端。 */
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stop_client_;
  /** @brief Python 编排层手动结束保存服务客户端。 */
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr finish_client_;
  /** @brief Python 编排层删除全部标记服务客户端。 */
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr clear_client_;
  /** @brief Python 编排层完整 episode 列表服务客户端。 */
  rclcpp::Client<fastumi_interfaces::srv::ListEpisodeAnnotations>::SharedPtr
    list_annotations_client_;
  /** @brief Python 编排层单条 episode 删除服务客户端。 */
  rclcpp::Client<fastumi_interfaces::srv::DeleteEpisodeAnnotation>::SharedPtr
    delete_annotation_client_;
  /** @brief rosbag2 pause 服务客户端。 */
  rclcpp::Client<rosbag2_interfaces::srv::Pause>::SharedPtr pause_client_;
  /** @brief rosbag2 resume 服务客户端。 */
  rclcpp::Client<rosbag2_interfaces::srv::Resume>::SharedPtr resume_client_;
  /** @brief rosbag2 Seek 服务客户端。 */
  rclcpp::Client<rosbag2_interfaces::srv::Seek>::SharedPtr seek_client_;
  /** @brief rosbag2 SetRate 服务客户端。 */
  rclcpp::Client<rosbag2_interfaces::srv::SetRate>::SharedPtr set_rate_client_;
  /** @brief rosbag2 GetRate 服务客户端。 */
  rclcpp::Client<rosbag2_interfaces::srv::GetRate>::SharedPtr get_rate_client_;
  /** @brief rosbag2 IsPaused 服务客户端。 */
  rclcpp::Client<rosbag2_interfaces::srv::IsPaused>::SharedPtr is_paused_client_;
  /** @brief 夹爪开度订阅。 */
  rclcpp::Subscription<fastumi_interfaces::msg::GripperState>::SharedPtr gripper_subscription_;
  /** @brief Tracker 位姿订阅。 */
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
  /** @brief 模拟时钟订阅。 */
  rclcpp::Subscription<rosgraph_msgs::msg::Clock>::SharedPtr clock_subscription_;
  /** @brief 由 Panel 与每个 ROS 回调弱持有的析构同步屏障。 */
  std::shared_ptr<CallbackGate> callback_gate_;

  /** @brief 归一化夹爪开度数值标签。 */
  QLabel * gripper_label_;
  /** @brief 归一化夹爪开度进度条。 */
  QProgressBar * gripper_bar_;
  /** @brief 夹爪检测有效性标签。 */
  QLabel * gripper_valid_label_;
  /** @brief Tracker 位置和四元数标签。 */
  QLabel * pose_label_;
  /** @brief Tracker 最近样本状态标签。 */
  QLabel * freshness_label_;
  /** @brief episode 索引和状态标签。 */
  QLabel * episode_label_;
  /** @brief 当前会话已正常闭合的完整 episode 数标签。 */
  QLabel * completed_episode_count_label_;
  /** @brief 当前会话全部正常闭合 episode 的权威列表。 */
  QListWidget * episode_list_;
  /** @brief episode 切换按钮。 */
  QPushButton * episode_button_;
  /** @brief 鼠标播放暂停/继续按钮。 */
  QPushButton * playback_button_;
  /** @brief 手动结束回放并保存已完成标注的按钮。 */
  QPushButton * finish_button_;
  /** @brief 删除当前回放会话全部标记的按钮。 */
  QPushButton * clear_button_;
  /** @brief 可拖拽的归一化回放时间轴。 */
  QSlider * timeline_slider_;
  /** @brief 当前相对时间和总时长标签。 */
  QLabel * timeline_label_;
  /** @brief 当前播放器倍率标签。 */
  QLabel * rate_label_;
  /** @brief 快捷键和 Seek 状态提示。 */
  QLabel * playback_hint_label_;
  /** @brief GUI 线程定时刷新服务发现状态。 */
  QTimer * readiness_timer_;
  /** @brief 定时查询播放器事实状态。 */
  QTimer * player_state_timer_;
  /** @brief 定时查询后端权威 episode 列表。 */
  QTimer * annotation_list_timer_;
  /** @brief 事件服务在途标志，阻止重复输入。 */
  bool request_pending_{false};
  /** @brief /clock 是否已到达非零 bag 时间。 */
  bool clock_ready_{false};
  /** @brief 当前 episode 是否活动。 */
  bool episode_active_{false};
  /** @brief 当前 episode 的从零开始索引。 */
  uint32_t episode_index_{0};
  /** @brief 当前标注会话内收到 STOP 的完整 episode 数。 */
  uint32_t completed_episode_count_{0};
  /** @brief 播放器当前是否暂停，初始值对应 --start-paused。 */
  bool playback_paused_{true};
  /** @brief Pause/Resume 服务在途标志，阻止按钮或 Enter 重复提交。 */
  bool playback_request_pending_{false};
  /** @brief 手动结束保存请求在途或已接受标志。 */
  bool finish_request_pending_{false};
  /** @brief 删除所有标记服务请求在途标志。 */
  bool clear_request_pending_{false};
  /** @brief 删除确认框活动标志，阻止其嵌套事件循环触发全局快捷键。 */
  bool clear_confirmation_active_{false};
  /** @brief 列表查询服务请求在途标志，防止轮询堆积。 */
  bool list_query_pending_{false};
  /** @brief 单条 episode 删除服务请求在途标志。 */
  bool delete_request_pending_{false};
  /** @brief 单条删除确认框活动标志，阻止全局快捷键重入。 */
  bool delete_confirmation_active_{false};
  /** @brief 当前是否存在任意 START、STOP 或 ABORT 标记。 */
  bool has_annotations_{false};
  /** @brief 已应用的最大后端权威标注 revision。 */
  uint64_t annotation_revision_{0};
  /** @brief 列表查询代次，用于丢弃服务变更提交前发出的晚到响应。 */
  uint64_t annotation_query_epoch_{0};
  /** @brief START/STOP 请求等待确认的最小 revision。 */
  uint64_t episode_request_min_revision_{0};
  /** @brief 是否已收到当前 START/STOP 服务请求的成功响应。 */
  bool episode_request_succeeded_{false};
  /** @brief 清空请求等待确认的最小 revision。 */
  uint64_t clear_request_min_revision_{0};
  /** @brief 是否已收到当前清空服务请求的成功响应。 */
  bool clear_request_succeeded_{false};
  /** @brief 单条删除请求等待确认的最小 revision。 */
  uint64_t delete_request_min_revision_{0};
  /** @brief 是否已收到当前单条删除服务请求的成功响应。 */
  bool delete_request_succeeded_{false};
  /** @brief 最近一次从后端取得的完整 episode 列表。 */
  std::vector<EpisodeListEntry> episode_entries_;
  /** @brief 最近一次由播放器确认的倍率。 */
  double playback_rate_{1.0};
  /** @brief SetRate 请求在途标志。 */
  bool rate_request_pending_{false};
  /** @brief GetRate 查询在途标志。 */
  bool rate_query_pending_{false};
  /** @brief IsPaused 查询在途标志。 */
  bool pause_query_pending_{false};
  /** @brief MCAP 第一条消息的绝对时间。 */
  int64_t replay_start_time_ns_{0};
  /** @brief MCAP 首尾消息之间的持续时间。 */
  int64_t replay_duration_ns_{0};
  /** @brief 最近一次收到的绝对 /clock 时间。 */
  int64_t current_time_ns_{0};
  /** @brief 最后一条 START、STOP 或 ABORT 的绝对时间。 */
  int64_t last_event_time_ns_{0};
  /** @brief 当前拖拽 Seek 的串行状态。 */
  SeekUiState seek_state_;
  /** @brief 最近一次 Seek 响应是否成功，用于恢复播放后的提示。 */
  bool seek_request_succeeded_{false};
  /** @brief 本次释放目标是否因末次 episode 边界被钳制。 */
  bool seek_target_was_clamped_{false};
  /** @brief 当前 Seek 是否由 episode 列表触发并要求保持暂停。 */
  bool episode_list_seek_active_{false};
  /** @brief 当前列表跳转目标的 episode 索引，用于完成提示。 */
  uint32_t episode_list_seek_index_{0};
};

}  // namespace fastumi_rviz_plugins

#endif  // FASTUMI_RVIZ_PLUGINS__REPLAY_ANNOTATION_PANEL_HPP_
