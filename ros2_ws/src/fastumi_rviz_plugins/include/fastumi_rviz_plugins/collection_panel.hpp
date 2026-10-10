/**
 * @file collection_panel.hpp
 * @brief 声明 UMI 服务化采集的 RViz2 控制面板。
 */

#ifndef FASTUMI_RVIZ_PLUGINS__COLLECTION_PANEL_HPP_
#define FASTUMI_RVIZ_PLUGINS__COLLECTION_PANEL_HPP_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rviz_common/panel.hpp>

#include "fastumi_rviz_plugins/collection_panel_state.hpp"

#include <fastumi_interfaces/msg/collection_record.hpp>
#include <fastumi_interfaces/msg/collection_result.hpp>
#include <fastumi_interfaces/msg/collection_status.hpp>
#include <fastumi_interfaces/srv/collection_cancel.hpp>
#include <fastumi_interfaces/srv/collection_delete.hpp>
#include <fastumi_interfaces/srv/collection_get_status.hpp>
#include <fastumi_interfaces/srv/collection_list.hpp>
#include <fastumi_interfaces/srv/collection_save.hpp>
#include <fastumi_interfaces/srv/collection_start.hpp>
#include <fastumi_interfaces/srv/collection_stop_and_cancel.hpp>
#include <fastumi_interfaces/srv/collection_stop_and_save.hpp>
#include <rclcpp/rclcpp.hpp>

class QLabel;
class QLineEdit;
class QListWidget;
class QProgressBar;
class QPushButton;
class QTimer;

namespace fastumi_rviz_plugins
{

/**
 * @brief 控制 UMI 采集的开始、停止、保存、取消和删除，并显示设备健康与记录列表。
 *
 * 面板不推断生命周期：按钮可用性来自后端 `CollectionStatus.can_*`；所有修改请求
 * 异步发送并带超时，状态版本回退的过期快照与不匹配的过期响应都会被丢弃；
 * 与后端失联后重新连接会重新查询权威状态和记录列表。
 */
class CollectionPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  /** @brief 构造尚未绑定 RViz ROS 节点的面板。 */
  explicit CollectionPanel(QWidget * parent = nullptr);
  /** @brief 撤销 ROS 回调并停止全部定时器。 */
  ~CollectionPanel() override;

  /** @brief 获得 RViz 的 ROS 节点后创建订阅、客户端和定时器。 */
  void onInitialize() override;

private Q_SLOTS:
  /** @brief 点击“开始”：校验任务名后提交 start 请求。 */
  void request_start();
  /** @brief 点击“停止并保存”。 */
  void request_stop_and_save();
  /** @brief 二次确认后提交“停止并取消”。 */
  void request_stop_and_cancel();
  /** @brief 点击“保存”（待保存状态）。 */
  void request_save();
  /** @brief 二次确认后提交“取消”（待保存或故障状态）。 */
  void request_cancel();
  /** @brief 二次确认后删除列表选中的已保存记录。 */
  void request_delete_selected();
  /** @brief 按任务筛选并回到第一页后刷新列表。 */
  void apply_task_filter();
  /** @brief 翻到上一页。 */
  void previous_page();
  /** @brief 翻到下一页。 */
  void next_page();
  /** @brief 周期检查状态新鲜度并在失联后重新查询。 */
  void check_connection();
  /** @brief 在途请求超时：清除等待并查询权威状态。 */
  void on_request_timeout();
  /** @brief 任务名或选择变化后刷新按钮可用性。 */
  void refresh_controls();

private:
  /** @brief 允许回归测试检查真实 Qt 面板的响应时序与计时器。 */
  friend class CollectionPanelTestPeer;

  /** @brief 协调 ROS 回调与面板析构的共享访问屏障。 */
  struct CallbackGate
  {
    /** @brief 保护 panel 指针和回调到析构的交接。 */
    std::mutex mutex;
    /** @brief 非空时允许回调向存活的面板排队 Qt 更新。 */
    CollectionPanel * panel{nullptr};
  };

  /** @brief 修改类请求的统一结果回调类型。 */
  using ResultMessage = fastumi_interfaces::msg::CollectionResult;

  /** @brief 在 GUI 线程应用一条权威状态。 */
  void apply_status(const fastumi_interfaces::msg::CollectionStatus & status);
  /** @brief 刷新单路输入的显示。 */
  void apply_stream(
    QLabel * label, const fastumi_interfaces::msg::CollectionStreamHealth & stream,
    bool show_pair);
  /** @brief 查询权威状态（首次连接、重连、超时后）。 */
  void request_status();
  /** @brief 请求记录列表当前页。 */
  void request_list();
  /** @brief 在 GUI 线程应用记录列表。 */
  void apply_list(
    uint64_t epoch, bool success, const std::string & message,
    const std::vector<fastumi_interfaces::msg::CollectionRecord> & records, uint64_t total);
  /** @brief 开始跟踪一条修改请求并启动超时计时，返回新的请求 ID。 */
  std::string begin_request(const std::string & operation, int timeout_ms);
  /** @brief 应用修改请求的服务响应。 */
  void handle_result(const std::string & operation, const ResultMessage & result);
  /** @brief 请求发送失败（服务不可用）时清除在途标记。 */
  void fail_request(const std::string & request_id, const std::string & message);
  /** @brief 显示一条操作结果消息。 */
  void show_message(const std::string & text, bool error);
  /** @brief 异步发送一条修改请求，并把响应排队回 GUI 线程。 */
  template<typename ClientPtr, typename RequestPtr>
  void submit_modify(
    const ClientPtr & client, const RequestPtr & request, const std::string & operation);

  /** @brief RViz 提供的 ROS 节点。 */
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<fastumi_interfaces::msg::CollectionStatus>::SharedPtr status_subscription_;
  rclcpp::Client<fastumi_interfaces::srv::CollectionStart>::SharedPtr start_client_;
  rclcpp::Client<fastumi_interfaces::srv::CollectionStopAndSave>::SharedPtr stop_and_save_client_;
  rclcpp::Client<fastumi_interfaces::srv::CollectionStopAndCancel>::SharedPtr
    stop_and_cancel_client_;
  rclcpp::Client<fastumi_interfaces::srv::CollectionSave>::SharedPtr save_client_;
  rclcpp::Client<fastumi_interfaces::srv::CollectionCancel>::SharedPtr cancel_client_;
  rclcpp::Client<fastumi_interfaces::srv::CollectionDelete>::SharedPtr delete_client_;
  rclcpp::Client<fastumi_interfaces::srv::CollectionList>::SharedPtr list_client_;
  rclcpp::Client<fastumi_interfaces::srv::CollectionGetStatus>::SharedPtr get_status_client_;

  QLabel * connection_label_;
  QLabel * state_label_;
  QLabel * calibration_label_;
  QLabel * duration_label_;
  QLabel * message_label_;
  QLineEdit * task_edit_;
  QLineEdit * name_edit_;
  QPushButton * start_button_;
  QPushButton * stop_and_save_button_;
  QPushButton * stop_and_cancel_button_;
  QPushButton * save_button_;
  QPushButton * cancel_button_;
  QLabel * image_label_;
  QLabel * tracker_label_;
  QLabel * gripper_label_;
  QProgressBar * gripper_bar_;
  QLabel * counts_label_;
  QLabel * alarms_label_;
  QLabel * blockers_label_;
  QLineEdit * filter_edit_;
  QListWidget * record_list_;
  QLabel * page_label_;
  QPushButton * previous_button_;
  QPushButton * next_button_;
  QPushButton * delete_button_;
  QTimer * connection_timer_;
  QTimer * request_timer_;

  /** @brief 最近一次已应用状态。 */
  fastumi_interfaces::msg::CollectionStatus status_;
  bool have_status_{false};
  int64_t last_status_ms_{0};
  /** @brief 状态曾经失联，重新收到状态后需要重新查询列表。 */
  bool was_disconnected_{true};
  bool status_query_in_flight_{false};
  /** @brief 已为哪个状态版本刷新过列表，避免重复查询。 */
  uint64_t list_refreshed_version_{0};

  /** @brief 面板实例随机串和请求计数，生成跨实例唯一的请求 ID。 */
  std::string nonce_;
  uint64_t request_counter_{0};
  /** @brief 在途修改请求 ID，空表示没有。 */
  std::string pending_request_id_;
  std::string pending_operation_;
  /** @brief 已被权威状态确认、仍可能收到详细响应的请求 ID。 */
  std::string confirmed_request_id_;

  /** @brief 当前页码（从 0 开始）、总条数和查询代次。 */
  uint32_t page_{0};
  uint64_t total_records_{0};
  uint64_t list_epoch_{0};
  std::string applied_filter_;
  /** @brief 列表项对应的记录 UUID。 */
  std::vector<std::string> record_uuids_;

  std::shared_ptr<CallbackGate> callback_gate_;
};

}  // namespace fastumi_rviz_plugins

#endif  // FASTUMI_RVIZ_PLUGINS__COLLECTION_PANEL_HPP_
