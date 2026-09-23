/**
 * @file teleop_panel.hpp
 * @brief 声明在 GUI 线程执行 ROS 回调的遥操与数采管理面板。
 */
#ifndef TRACKER_TELEOPERATED__TELEOP_PANEL_HPP_
#define TRACKER_TELEOPERATED__TELEOP_PANEL_HPP_

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <QPointer>
#include <QWidget>
#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <fastumi_interfaces/msg/recording_status.hpp>
#include <fastumi_interfaces/srv/cancel_recording.hpp>
#include <fastumi_interfaces/srv/cancel_recording_request.hpp>
#include <fastumi_interfaces/srv/get_recording_request.hpp>
#include <fastumi_interfaces/srv/get_recording_status.hpp>
#include <fastumi_interfaces/srv/get_teleop_generation.hpp>
#include <fastumi_interfaces/srv/set_teleop_generation.hpp>
#include <fastumi_interfaces/srv/start_recording.hpp>
#include <fastumi_interfaces/srv/stop_recording.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>

class QLabel;
class QPlainTextEdit;
class QPushButton;
class QTimer;
class QTreeWidget;
class QEvent;

namespace tracker_teleoperated
{
class RecordingsWidget;
class VideoSourceWidget;
/** @brief 只负责 ROS 交互和显示，不拥有硬件进程的 RViz 面板。 */
class TeleopPanel : public rviz_common::Panel
{
  Q_OBJECT
  friend class TeleopPanelRecoveryHarness;
public:
  /** @brief 构造界面并安装 RViz 窗口级快捷键过滤器。 @param[in] parent RViz 提供的父控件。 */
  explicit TeleopPanel(QWidget * parent = nullptr);
  /** @brief 停止定时器后移除 executor，防止卸载后的回调访问。 */
  ~TeleopPanel() override;
  /** @brief 创建独立 ROS 节点并绑定 RViz 使用的 ROS context。 */
  void onInitialize() override;
  /** @brief 保存两路相机设备选择。 @param[out] config RViz 面板配置。 */
  void save(rviz_common::Config config) const override;
  /** @brief 恢复相机选择的显示初值。 @param[in] config RViz 面板配置。 */
  void load(const rviz_common::Config & config) override;

Q_SIGNALS:
  /** @brief 按钮与快捷键共用的操作意图。 @param action 固定命令标识。 */
  void actionRequested(const QString & action);

protected:
  /**
   * @brief 在所属 RViz 窗口内优先处理控制按键，同时保护编辑控件和对话框。
   * @param[in] watched 事件对象。
   * @param[in] event 原始事件。
   * @return 已消费遥操快捷键时为 true。
   */
  bool eventFilter(QObject * watched, QEvent * event) override;

private:
  /** @brief 回车组合操作的内部执行阶段。 */
  enum class CombinedOperation
  {
    Idle,       ///< 当前没有组合操作。
    Starting,   ///< 正在同时启用遥操和开始录制。
    Stopping,   ///< 正在停止录制并执行回位。
    Discarding, ///< 正在取消当前录制并执行回位。
    RollingBack ///< 启动部分失败后正在恢复安全状态。
  };

  /** @brief 在 GUI 线程处理操作意图。 @param action 固定操作标识。 */
  void dispatch(const QString & action);
  /** @brief 异步调用 Trigger。 @param service 相对服务名称。 */
  void trigger(const std::string & service);
  /** @brief 异步调用组件 SetBool。 @param service 相对服务名称。 @param value 目标启停状态。 */
  void setBool(const std::string & service, bool value);
  /** @brief 查询最高代次后执行遥操启停，暂停时可覆盖已发出的启用代次。 */
  void requestControl(bool enable,
    const std::function<void(bool, std::uint64_t, const std::string &)> & done,
    const std::function<bool()> & allowed = []() {return true;},
    const std::function<void(std::uint64_t)> & sent = [](std::uint64_t) {},
    std::uint64_t generation_floor = 0);
  /** @brief 按当前权威状态调用开始、停止或取消服务。 @param action record 或 discard。 */
  void recordingAction(const QString & action);
  /** @brief 按远端录制状态启动或停止回车组合流程。 */
  void combinedAction();
  /** @brief 同时请求启用遥操和开始录制。 */
  void startCombinedAction();
  /** @brief 检查组合启动的两个响应，并在部分失败时开始回滚。 @param sequence 组合请求代次。 */
  void evaluateCombinedStart(unsigned sequence);
  /** @brief 进入按请求 UUID 撤销录制及暂停遥操的恢复流程。 @param sequence 组合请求代次。 @param reason 启动失败原因。 */
  void beginCombinedRollback(unsigned sequence, const QString & reason);
  /** @brief 检查组合启动回滚是否完成。 @param sequence 组合请求代次。 */
  void evaluateCombinedRollback(unsigned sequence);
  /** @brief 周期查询并撤销本次启动请求，直到录像和遥操恢复安全状态。 */
  void advanceCombinedRecovery();
  /** @brief 同时请求停止录制和回到初始位姿。 */
  void stopCombinedAction();
  /** @brief 检查组合停止结果，并在回位失败时补发暂停。 @param sequence 组合请求代次。 */
  void evaluateCombinedStop(unsigned sequence);
  /** @brief 同时请求取消当前录制和回到初始位姿。 */
  void discardCombinedAction();
  /** @brief 检查组合取消结果，并在回位失败时补发暂停。 @param sequence 组合请求代次。 */
  void evaluateCombinedDiscard(unsigned sequence);
  /** @brief 组合取消超时后查询权威状态，并在仍录制时重试取消。 */
  void recoverDiscardAfterTimeout();
  /** @brief 结束当前组合操作并刷新界面。 @param message 显示给操作者的最终结果。 */
  void finishCombinedAction(const QString & message);
  /** @brief 查询远端权威录制状态，用于请求超时后的状态核对。 */
  void queryRecordingStatus();
  /** @brief 按管理器配置创建远端录制客户端。 @param prefix 服务前缀。 @param dir_name 任务目录名。 @param name 任务名。 */
  void configureRecording(
    const std::string & prefix, const std::string & dir_name, const std::string & name);
  /** @brief 应用远端权威状态。 @param message 录制状态消息。 */
  void updateRecordingStatus(const fastumi_interfaces::msg::RecordingStatus & message);
  /** @brief 根据服务、消息新鲜度及当前操作刷新按钮。 */
  void refresh();
  /** @brief 更新节点表格与诊断详情。 @param message 管理器诊断快照。 */
  void updateComponents(const diagnostic_msgs::msg::DiagnosticArray & message);
  /** @brief 将选中行的详细状态显示在文本区。 */
  void showDetails();
  /** @brief 启动请求超时计时。 @return 本次请求编号。 */
  unsigned beginRequest();
  /** @brief 处理最新请求结果。 @param sequence 请求代次。 @param success 是否成功。 @param message 服务说明。 */
  void finishRequest(unsigned sequence, bool success, const std::string & message);

  /** @brief 所有 ROS 回调都通过 GUI 定时器中的 spin_some 执行。 */
  rclcpp::Node::SharedPtr node_;
  /** @brief 不创建后台线程，析构时同步撤销回调。 */
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  /** @brief 面板存活心跳。 */
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr heartbeat_;
  /** @brief 远端开始录制服务。 */
  rclcpp::Client<fastumi_interfaces::srv::StartRecording>::SharedPtr recording_start_;
  /** @brief 远端停止录制服务。 */
  rclcpp::Client<fastumi_interfaces::srv::StopRecording>::SharedPtr recording_stop_;
  /** @brief 远端取消录制服务。 */
  rclcpp::Client<fastumi_interfaces::srv::CancelRecording>::SharedPtr recording_cancel_;
  /** @brief 按请求 UUID 撤销尚未完成的启动。 */
  rclcpp::Client<fastumi_interfaces::srv::CancelRecordingRequest>::SharedPtr recording_cancel_request_;
  /** @brief 查询请求 UUID 的权威终态。 */
  rclcpp::Client<fastumi_interfaces::srv::GetRecordingRequest>::SharedPtr recording_get_request_;
  /** @brief 本机遥操的启用、暂停和最高代次客户端。 */
  rclcpp::Client<fastumi_interfaces::srv::SetTeleopGeneration>::SharedPtr control_enable_;
  rclcpp::Client<fastumi_interfaces::srv::SetTeleopGeneration>::SharedPtr control_disable_;
  rclcpp::Client<fastumi_interfaces::srv::GetTeleopGeneration>::SharedPtr control_get_generation_;
  /** @brief 远端状态查询服务。 */
  rclcpp::Client<fastumi_interfaces::srv::GetRecordingStatus>::SharedPtr recording_status_client_;
  /** @brief 远端权威录制状态订阅。 */
  rclcpp::Subscription<fastumi_interfaces::msg::RecordingStatus>::SharedPtr recording_status_subscription_;
  /** @brief 用基类容器保留所有订阅生命周期。 */
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions_;
  /** @brief 按服务名缓存 Trigger 客户端。 */
  std::map<std::string, rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr> triggers_;
  /** @brief 按服务名缓存 SetBool 客户端。 */
  std::map<std::string, rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr> booleans_;
  /** @brief 控制按钮按固定操作标识索引。 */
  std::map<QString, QPushButton *> buttons_;
  /** @brief 本机当前任务的已保存数据浏览页签。 */
  RecordingsWidget * recordings_;
  /** @brief 两路 RViz 图像预览的话题选择控件。 */
  VideoSourceWidget * video_sources_;
  /** @brief 当前面板所属的 RViz 主窗口；窗口销毁后自动清空。 */
  QPointer<QWidget> rviz_window_;
  /** @brief 节点状态与启停按钮表格。 */
  QTreeWidget * tree_;
  /** @brief 选中组件的诊断详情。 */
  QPlainTextEdit * details_;
  /** @brief 遥操业务状态说明。 */
  QLabel * control_status_;
  /** @brief 录制进度、保存路径与错误。 */
  QLabel * record_status_;
  /** @brief 服务响应与会话收尾进度。 */
  QLabel * request_status_;
  /** @brief 提示焦点范围与标定操作。 */
  QLabel * hint_;
  /** @brief GUI 线程轮询 ROS 与按钮状态。 */
  QTimer * spin_timer_;
  /** @brief GUI 线程发送 10 Hz 心跳。 */
  QTimer * heartbeat_timer_;
  /** @brief 组件诊断详情的快照。 */
  std::map<QString, QString> details_by_id_;
  /** @brief 各行启动和停止按钮。 */
  std::map<QString, std::pair<QPushButton *, QPushButton *>> component_buttons_;
  /** @brief 各组件最新管理归属、模式和进程状态。 */
  std::map<QString, std::map<std::string, std::string>> component_values_;
  /** @brief 已确认的遥操启用状态。 */
  bool enabled_{false};
  /** @brief 管理节点是否正在暂停、保存或停止。 */
  bool manager_busy_{false};
  /** @brief 服务请求是否在途。 */
  bool pending_{false};
  /** @brief 录制命令是否等待状态变化。 */
  bool record_pending_{false};
  /** @brief 当前回车组合操作阶段。 */
  CombinedOperation combined_operation_{CombinedOperation::Idle};
  /** @brief 当前组合操作的遥操或回位请求是否已有响应。 */
  bool combined_control_done_{false};
  /** @brief 当前组合操作的遥操或回位请求是否成功。 */
  bool combined_control_success_{false};
  /** @brief 当前组合操作的录制请求是否已有响应。 */
  bool combined_record_done_{false};
  /** @brief 当前组合操作的录制请求是否成功。 */
  bool combined_record_success_{false};
  /** @brief 回位失败后的暂停兜底请求是否已有响应。 */
  bool combined_fallback_done_{false};
  /** @brief 回位失败后的暂停兜底请求是否已经发出。 */
  bool combined_fallback_started_{false};
  /** @brief 回位失败后的暂停兜底请求是否成功。 */
  bool combined_fallback_success_{false};
  /** @brief 正在等待撤销的启动请求 UUID。 */
  std::string combined_request_id_;
  /** @brief 上次回车启用请求的代次；恢复时必须由更高或同代次暂停覆盖。 */
  std::uint64_t combined_enable_generation_{0};
  /** @brief 启动恢复期间录像和遥操是否确认安全。 */
  bool recovery_record_done_{false};
  bool recovery_control_done_{false};
  /** @brief 单独按 A 超时恢复时不改变遥操启停。 */
  bool recovery_pause_control_{true};
  /** @brief 组合恢复中的已保存记录 UUID，需人工决定是否删除。 */
  std::string recovery_completed_id_;
  /** @brief 避免周期恢复请求密集重发。 */
  std::chrono::steady_clock::time_point recovery_next_{};
  /** @brief 单独按 A 启动请求的 UUID。 */
  std::string manual_request_id_;
  /** @brief 最近普通录制操作是否正在启动。 */
  bool manual_start_pending_{false};
  /** @brief 普通遥操启用请求是否等待响应。 */
  bool manual_enable_pending_{false};
  /** @brief 普通启用超时后暂停必须覆盖的代次。 */
  std::uint64_t manual_enable_generation_{0};
  /** @brief 最新固定录制状态。 */
  std::string record_state_;
  /** @brief 当前远端录制 UUID。 */
  std::string recording_id_;
  /** @brief 最近完成的远端录制 UUID。 */
  std::string last_completed_recording_id_;
  /** @brief 管理器提供的远端录制服务前缀。 */
  std::string recording_prefix_;
  /** @brief 开始录制请求使用的可选任务目录名。 */
  std::string recording_dir_name_;
  /** @brief 开始录制请求使用的可选任务名。 */
  std::string recording_name_;
  /** @brief 记录会话进度变化，避免覆盖普通服务响应。 */
  std::string last_session_message_;
  /** @brief 防止旧服务结果覆盖当前操作。 */
  unsigned sequence_{0};
  /** @brief 防止迟到的录制服务响应覆盖当前录制操作。 */
  unsigned recording_sequence_{0};
  /** @brief 防止迟到的组合服务响应覆盖后续操作。 */
  unsigned combined_sequence_{0};
  /** @brief 组合操作失败及回滚结果的累计说明。 */
  QString combined_message_;
  /** @brief 组合启动成功后用于精确停止的远端录制 UUID。 */
  std::string combined_recording_id_;
  /** @brief 最近管理诊断接收时间。 */
  std::chrono::steady_clock::time_point manager_seen_{};
  /** @brief 最近遥操业务心跳时间。 */
  std::chrono::steady_clock::time_point control_seen_{};
  /** @brief 最近记录状态接收时间。 */
  std::chrono::steady_clock::time_point record_seen_{};
  /** @brief 服务调用的超时期限。 */
  std::chrono::steady_clock::time_point request_deadline_{};
  /** @brief 录制命令的超时期限。 */
  std::chrono::steady_clock::time_point record_deadline_{};
  /** @brief 回车组合操作的超时期限。 */
  std::chrono::steady_clock::time_point combined_deadline_{};
};
}  // namespace tracker_teleoperated
#endif
