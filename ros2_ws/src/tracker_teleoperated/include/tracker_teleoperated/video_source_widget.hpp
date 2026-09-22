/**
 * @file video_source_widget.hpp
 * @brief 声明本机 UMI 设备选择和远端末端视频只读状态控件。
 */
#ifndef TRACKER_TELEOPERATED__VIDEO_SOURCE_WIDGET_HPP_
#define TRACKER_TELEOPERATED__VIDEO_SOURCE_WIDGET_HPP_

#include <functional>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <QStringList>
#include <QWidget>
#include <rclcpp/rclcpp.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rcl_interfaces/srv/set_parameters_atomically.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <rviz_common/config.hpp>

class QComboBox;
class QLabel;
class QPushButton;
class QTimer;

namespace rviz_common
{
class DisplayContext;
}  // namespace rviz_common

namespace tracker_teleoperated
{
/**
 * @brief 管理“UMI 视频”和“末端视频”两个角色的本机相机设备。
 * @details 管理器确认实际输入后同步显示；不修改 YAML 配置。
 */
class VideoSourceWidget : public QWidget
{
  Q_OBJECT
public:
  /** @brief 按 RViz 显示名称读取当前预览话题的可替换函数类型。 */
  using TopicReader = std::function<QString(const QString &)>;
  /** @brief 按 RViz 显示名称切换预览话题的可替换函数类型。 */
  using TopicWriter = std::function<bool(const QString &, const QString &)>;

  /**
   * @brief 创建两路下拉框、状态提示及自动刷新定时器。
   * @param[in] parent 父控件，由 Qt 管理生命周期。
   * @param[in] reader 测试可注入的显示话题读取函数；生产环境可留空。
   * @param[in] writer 测试可注入的显示话题切换函数；生产环境可留空。
   */
  explicit VideoSourceWidget(
    QWidget * parent = nullptr, TopicReader reader = {}, TopicWriter writer = {});

  /**
   * @brief 绑定面板 ROS 节点和 RViz 显示上下文。
   * @param[in] node 用于订阅设备目录和提交切换的面板节点。
   * @param[in] context 用于动态查找两个 Image Display 的 RViz 上下文。
   * @note 两个对象均由面板或 RViz 持有，且生命周期覆盖本控件。
   */
  void initialize(
    const rclcpp::Node::SharedPtr & node,
    rviz_common::DisplayContext * context);

  /** @brief 刷新设备列表并恢复固定角色的预览话题。 */
  void refreshSources();

  /** @brief 接收管理器确认的源状态，供生产订阅与测试共用。 @param[in] message 组件诊断。 */
  void updateSources(const diagnostic_msgs::msg::DiagnosticArray & message);

  /** @brief 更新本机采集设备目录。 @param[in] message 后台扫描结果。 */
  void updateDevices(const diagnostic_msgs::msg::DiagnosticArray & message);
  /** @brief 保存已确认设备，供下次 launch 在自动启动前恢复。 @param[out] config RViz 配置。 */
  void save(rviz_common::Config config) const;
  /** @brief 读取设备显示初值，实际运行仍由管理器确认。 @param[in] config RViz 配置。 */
  void load(const rviz_common::Config & config);

  /** @brief UMI 切换未确认时阻止开始录制。 @return 是否正在切换或失去管理状态。 */
  bool umiBusy() const;

private:
  /**
   * @brief 更新单个下拉框，同时保留当前确认设备和用户选中项。
   * @param[in,out] combo 目标下拉框。
   * @param[in] display_name 对应 RViz Image Display 名称。
   * @param[in] image_topics 当前可用的设备路径。
   * @return 成功找到对应 Image Display 时为 true。
   */
  bool updateCombo(
    QComboBox * combo, const QString & display_name,
    const QStringList & image_topics);

  /**
   * @brief 提交用户选择，生产环境等待管理器确认后同步显示。
   * @param[in] display_name 对应显示名称。
   * @param[in] combo 提供实际设备路径的下拉框。
   */
  void applySelection(const QString & display_name, QComboBox * combo);

  /** @brief 异步提交单路源参数。 @param[in] name 显示名称。 @param[in] topic 新设备路径。 */
  void requestSource(const QString & name, const QString & topic);

  /** @brief 保存单路异步请求及管理器确认信息。 */
  struct SourceState
  {
    std::map<std::string, std::string> values;  ///< 管理器最近一次诊断字段。
    QString error;  ///< 本面板提交错误，保留到下次用户操作。
    QString requested;  ///< 本面板等待确认的设备。
    bool pending{false};  ///< 是否等待参数服务响应或管理器确认。
    bool accepted{false};  ///< 管理器已接受后等待诊断，不对硬件启动套用服务超时。
    unsigned sequence{0};  ///< 拒绝迟到回调的请求代次。
    int64_t request_id{0};  ///< 用于移除已超时的参数请求。
    std::chrono::steady_clock::time_point deadline{};  ///< 请求响应期限。
  };
  std::map<QString, SourceState> sources_;  ///< 两路独立状态，管理器统一仲裁。
  bool managed_{false};  ///< 生产模式要求管理器确认后切换显示。
  bool session_busy_{false};  ///< 会话退出或停止时禁止修改输入。
  std::chrono::steady_clock::time_point manager_seen_{};  ///< 最近诊断时间。
  rclcpp::Client<rcl_interfaces::srv::SetParametersAtomically>::SharedPtr source_client_;  ///< 管理器参数客户端。
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr source_status_;  ///< 实际输入订阅。
  std::string recording_state_;  ///< 最近录制状态。
  std::map<QString, QString> devices_;  ///< 设备路径对应名称及访问错误。
  QString scan_error_;  ///< 最近目录扫描错误。
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr device_status_;  ///< 本机设备目录。
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr refresh_client_;  ///< 后台刷新请求。

  QComboBox * wrist_combo_;  ///< 只读显示末端编码与解码链路。
  QComboBox * umi_combo_;  ///< 选择 UMI 相机的本机采集设备。
  QLabel * status_;  ///< 显示实际输入、不可操作原因及切换错误。
  QPushButton * refresh_button_;  ///< 用户主动刷新后台设备目录。
  QTimer * refresh_timer_;  ///< 每两秒同步设备目录和 RViz 属性。
  TopicReader topic_reader_;  ///< RViz 显示当前话题读取函数。
  TopicWriter topic_writer_;  ///< RViz 显示话题切换函数。
};
}  // namespace tracker_teleoperated
#endif  // TRACKER_TELEOPERATED__VIDEO_SOURCE_WIDGET_HPP_
