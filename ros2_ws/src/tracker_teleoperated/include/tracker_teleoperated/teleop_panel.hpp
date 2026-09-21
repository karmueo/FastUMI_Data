/**
 * @file teleop_panel.hpp
 * @brief 声明在 GUI 线程执行 ROS 回调的遥操与数采管理面板。
 */
#ifndef TRACKER_TELEOPERATED__TELEOP_PANEL_HPP_
#define TRACKER_TELEOPERATED__TELEOP_PANEL_HPP_

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <QPointer>
#include <QWidget>
#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
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
  /** @brief 在 GUI 线程处理操作意图。 @param action 固定操作标识。 */
  void dispatch(const QString & action);
  /** @brief 异步调用 Trigger。 @param service 相对服务名称。 */
  void trigger(const std::string & service);
  /** @brief 异步调用 SetBool。 @param service 相对服务名称。 @param value 目标启停状态。 */
  void setBool(const std::string & service, bool value);
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
  /** @brief 录制兼容命令发布端。 */
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr record_command_;
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
  /** @brief 最新固定录制状态。 */
  std::string record_state_;
  /** @brief 记录会话进度变化，避免覆盖普通服务响应。 */
  std::string last_session_message_;
  /** @brief 防止旧服务结果覆盖当前操作。 */
  unsigned sequence_{0};
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
};
}  // namespace tracker_teleoperated
#endif
