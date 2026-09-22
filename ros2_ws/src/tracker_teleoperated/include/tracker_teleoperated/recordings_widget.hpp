/**
 * @file recordings_widget.hpp
 * @brief 声明通过远端录制服务分页浏览已保存数据的控件。
 */
#ifndef TRACKER_TELEOPERATED__RECORDINGS_WIDGET_HPP_
#define TRACKER_TELEOPERATED__RECORDINGS_WIDGET_HPP_

#include <memory>
#include <vector>

#include <QWidget>
#include <rclcpp/rclcpp.hpp>
#include <fastumi_interfaces/msg/recording_info.hpp>
#include <fastumi_interfaces/srv/list_recordings.hpp>

class QLabel;
class QPushButton;
class QShowEvent;
class QTreeWidget;

namespace tracker_teleoperated
{
/** @brief 使用 ListRecordings 服务展示远端已提交记录。 */
class RecordingsWidget : public QWidget
{
  Q_OBJECT
public:
  /** @brief 创建远端记录树和刷新按钮。 @param[in] parent 父控件。 */
  explicit RecordingsWidget(QWidget * parent = nullptr);
  /** @brief 绑定面板 ROS 节点并创建列表客户端。 @param[in] node 面板节点。 @param[in] prefix 录制服务前缀。 */
  void initialize(const rclcpp::Node::SharedPtr & node, const std::string & prefix = "/fastumi/recording");
  /** @brief 请求从 offset=0 开始异步分页刷新。 */
  void requestRefresh();

Q_SIGNALS:
  /** @brief 一次完整远端列表刷新完成。 */
  void refreshed();

protected:
  /** @brief 页签显示时刷新远端列表。 @param[in] event Qt 显示事件。 */
  void showEvent(QShowEvent * event) override;

private:
  /** @brief 请求一页最多 100 条记录。 @param[in] generation 刷新代次。 @param[in] offset 页偏移。 */
  void requestPage(unsigned generation, std::uint32_t offset);
  /** @brief 用已聚合的远端摘要重建树。 @param[in] total 服务报告的总数。 */
  void displayResult(std::uint64_t total);

  QTreeWidget * tree_;  ///< 任务、记录和摘要字段组成的树。
  QLabel * status_;  ///< 刷新状态或服务错误。
  QPushButton * refresh_;  ///< 手动刷新按钮。
  rclcpp::Client<fastumi_interfaces::srv::ListRecordings>::SharedPtr client_;  ///< 远端列表服务。
  std::vector<fastumi_interfaces::msg::RecordingInfo> recordings_;  ///< 当前刷新聚合结果。
  unsigned generation_{0};  ///< 丢弃迟到分页响应的代次。
};
}  // namespace tracker_teleoperated
#endif
