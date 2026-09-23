/**
 * @file test_panel_recovery.cpp
 * @brief 使用本地 ROS 2 假服务验证面板启动失败后的请求撤销与遥操代次恢复。
 */
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <QApplication>
#include <QLabel>
#include <QTest>
#include <QTimer>
#include <rclcpp/rclcpp.hpp>
#include <fastumi_interfaces/srv/cancel_recording_request.hpp>
#include <fastumi_interfaces/srv/get_recording_request.hpp>
#include <fastumi_interfaces/srv/get_teleop_generation.hpp>
#include <fastumi_interfaces/srv/set_teleop_generation.hpp>
#include <fastumi_interfaces/srv/start_recording.hpp>

#include "tracker_teleoperated/teleop_panel.hpp"

namespace tracker_teleoperated
{
/** @brief 为恢复流程测试绑定客户端并读取私有状态。 */
class TeleopPanelRecoveryHarness
{
public:
  /** @brief 为测试面板设置独立 ROS 节点与服务客户端。 */
  static void initialize(TeleopPanel & panel, const rclcpp::Node::SharedPtr & node)
  {
    panel.node_ = node;
    panel.executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    panel.executor_->add_node(node);
    panel.control_get_generation_ = node->create_client<fastumi_interfaces::srv::GetTeleopGeneration>(
      "/tracker_teleoperated/get_generation");
    panel.control_enable_ = node->create_client<fastumi_interfaces::srv::SetTeleopGeneration>(
      "/tracker_teleoperated/enable");
    panel.control_disable_ = node->create_client<fastumi_interfaces::srv::SetTeleopGeneration>(
      "/tracker_teleoperated/disable");
    panel.recording_start_ = node->create_client<fastumi_interfaces::srv::StartRecording>(
      "/test_panel_recording/start");
    panel.recording_cancel_request_ =
      node->create_client<fastumi_interfaces::srv::CancelRecordingRequest>(
      "/test_panel_recording/cancel_request");
    panel.recording_get_request_ = node->create_client<fastumi_interfaces::srv::GetRecordingRequest>(
      "/test_panel_recording/get_request");
    panel.spin_timer_->start(10);
  }

  /** @brief 模拟普通启动或组合启动已超时。 */
  static void beginRecovery(TeleopPanel & panel, const std::string & request_id,
    std::uint64_t enable_generation, bool pause_control)
  {
    panel.combined_request_id_ = request_id;
    panel.combined_enable_generation_ = enable_generation;
    panel.recovery_pause_control_ = pause_control;
    panel.beginCombinedRollback(++panel.combined_sequence_, "测试启动超时");
  }

  /** @brief 返回组合操作是否结束。 */
  static bool idle(const TeleopPanel & panel)
  {
    return panel.combined_operation_ == TeleopPanel::CombinedOperation::Idle;
  }

  /** @brief 返回面板显示的最终结果。 */
  static QString message(const TeleopPanel & panel) {return panel.request_status_->text();}

  /** @brief 调用真实组合启动流程。 */
  static void start(TeleopPanel & panel) {panel.startCombinedAction();}

  /** @brief 调用单独录制启动流程。 */
  static void startRecording(TeleopPanel & panel) {panel.recordingAction("record");}

  /** @brief 将组合启动的等待时间推进到超时。 */
  static void expireCombined(TeleopPanel & panel)
  {
    panel.combined_deadline_ = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    panel.refresh();
  }

  /** @brief 将单独录制的等待时间推进到超时。 */
  static void expireRecording(TeleopPanel & panel)
  {
    panel.record_deadline_ = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    panel.refresh();
  }

  /** @brief 读取本次组合启动发出的遥操代次。 */
  static std::uint64_t enableGeneration(const TeleopPanel & panel)
  {
    return panel.combined_enable_generation_;
  }

  /** @brief 在启用请求尚未被服务端处理时发出更高代次的人工暂停。 */
  static void pauseAfterEnable(TeleopPanel & panel, std::uint64_t enable_generation)
  {
    panel.requestControl(false,
      [](bool, std::uint64_t, const std::string &) {},
      []() {return true;},
      [](std::uint64_t) {},
      enable_generation);
  }

  /** @brief 所需服务是否已被面板发现。 */
  static bool ready(const TeleopPanel & panel)
  {
    return panel.control_get_generation_->service_is_ready() &&
      panel.control_enable_->service_is_ready() &&
      panel.control_disable_->service_is_ready() &&
      panel.recording_start_->service_is_ready() &&
      panel.recording_cancel_request_->service_is_ready() &&
      panel.recording_get_request_->service_is_ready();
  }
};
}  // namespace tracker_teleoperated

namespace
{
/** @brief 创建测试进程唯一的 Qt 应用。 @return Qt 应用。 */
QApplication * application()
{
  static int argc = 1;  ///< Qt 启动参数数量。
  static char name[] = "test_panel_recovery";  ///< Qt 应用名称。
  static char * argv[] = {name, nullptr};  ///< Qt 启动参数数组。
  static QApplication app(argc, argv);  ///< 全部用例共用的应用。
  return &app;
}

/** @brief 确保 ROS context 在测试进程中只初始化一次。 */
void initializeRos()
{
  if (!rclcpp::ok()) {
    int argc = 0;  ///< 测试不传递 ROS 参数。
    char ** argv = nullptr;  ///< 空参数数组。
    rclcpp::init(argc, argv);
  }
}

/** @brief 模拟录制服务请求台账和本机遥操代次门控。 */
class FakeServices
{
public:
  /** @brief 创建隔离的服务节点。 */
  FakeServices()
  {
    static unsigned counter = 0;  ///< 避免用例间节点重名。
    node = std::make_shared<rclcpp::Node>("panel_recovery_services_" + std::to_string(++counter));
    executor.add_node(node);
  }

  /** @brief 从执行器移除服务节点。 */
  ~FakeServices() {executor.remove_node(node);}

  /** @brief 启动所有假服务，可在用例中延迟以模拟离线。 */
  void start()
  {
    services.push_back(node->create_service<fastumi_interfaces::srv::StartRecording>(
      "/test_panel_recording/start",
      [this](const std::shared_ptr<fastumi_interfaces::srv::StartRecording::Request> request,
        std::shared_ptr<fastumi_interfaces::srv::StartRecording::Response> response) {
        request_id = request->request_id;
        request_state = "recording";
        response->success = true;
        response->recording_id = "recording-1";
      }));
    services.push_back(node->create_service<fastumi_interfaces::srv::CancelRecordingRequest>(
      "/test_panel_recording/cancel_request",
      [this](const std::shared_ptr<fastumi_interfaces::srv::CancelRecordingRequest::Request> request,
        std::shared_ptr<fastumi_interfaces::srv::CancelRecordingRequest::Response> response) {
        ++cancel_calls;
        request_id = request->request_id;
        if (request_state == "saving") {
          response->code = "CANCEL_ACCEPTED";
          response->state = "saving";
        } else if (request_state == "completed") {
          response->code = "COMPLETED";
          response->state = "completed";
        } else {
          request_state = "cancelled";
          response->success = true;
          response->state = "cancelled";
        }
        response->recording_id = "recording-1";
      }));
    services.push_back(node->create_service<fastumi_interfaces::srv::GetRecordingRequest>(
      "/test_panel_recording/get_request",
      [this](const std::shared_ptr<fastumi_interfaces::srv::GetRecordingRequest::Request>,
        std::shared_ptr<fastumi_interfaces::srv::GetRecordingRequest::Response> response) {
        response->found = !request_id.empty();
        response->state = request_state;
        response->recording_id = "recording-1";
      }));
    services.push_back(node->create_service<fastumi_interfaces::srv::GetTeleopGeneration>(
      "/tracker_teleoperated/get_generation",
      [this](const std::shared_ptr<fastumi_interfaces::srv::GetTeleopGeneration::Request>,
        std::shared_ptr<fastumi_interfaces::srv::GetTeleopGeneration::Response> response) {
        response->operation_generation = generation;
        response->enabled = enabled;
      }));
    services.push_back(node->create_service<fastumi_interfaces::srv::SetTeleopGeneration>(
      "/tracker_teleoperated/enable",
      [this](const std::shared_ptr<fastumi_interfaces::srv::SetTeleopGeneration::Request> request,
        std::shared_ptr<fastumi_interfaces::srv::SetTeleopGeneration::Response> response) {
        if (request->operation_generation > generation) {
          generation = request->operation_generation;
          enabled = allow_enable;
          response->success = allow_enable;
        }
        response->enabled = enabled;
        response->operation_generation = generation;
        response->message = allow_enable ? "已启用" : "测试拒绝启用";
      }));
    services.push_back(node->create_service<fastumi_interfaces::srv::SetTeleopGeneration>(
      "/tracker_teleoperated/disable",
      [this](const std::shared_ptr<fastumi_interfaces::srv::SetTeleopGeneration::Request> request,
        std::shared_ptr<fastumi_interfaces::srv::SetTeleopGeneration::Response> response) {
        ++disable_calls;
        if (request->operation_generation >= generation) {
          generation = request->operation_generation;
          enabled = false;
          response->success = true;
        }
        response->enabled = enabled;
        response->operation_generation = generation;
      }));
  }

  rclcpp::Node::SharedPtr node;  ///< 提供所有假服务的节点。
  rclcpp::executors::SingleThreadedExecutor executor;  ///< 单线程假服务执行器。
  std::vector<rclcpp::ServiceBase::SharedPtr> services;  ///< 服务生命周期持有者。
  std::string request_id;  ///< 收到的启动请求 UUID。
  std::string request_state{"pending"};  ///< 假请求台账状态。
  std::uint64_t generation{0};  ///< 假控制节点最高代次。
  bool enabled{false};  ///< 假控制节点启用状态。
  bool allow_enable{true};  ///< 是否允许启动遥操。
  int cancel_calls{0};  ///< 收到的按请求 UUID 撤销次数。
  int disable_calls{0};  ///< 收到的带代次暂停次数。
};

/**
 * @brief 同时推进 Qt 和假 ROS 服务直到满足条件。
 * @param[in] services 假服务执行器。
 * @param[in] predicate 完成条件。
 * @param[in] timeout_ms 最长等待时间。
 * @return 是否满足完成条件。
 */
bool waitUntil(FakeServices & services, const std::function<bool()> & predicate,
  int timeout_ms = 3500)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    services.executor.spin_some(std::chrono::milliseconds(5));
    QApplication::processEvents();
    if (predicate()) {return true;}
    QTest::qWait(5);
  }
  return predicate();
}
}  // namespace

/** @brief 启用失败后以请求 UUID 撤销已启动的录像，并确认遥操暂停。 */
TEST(TeleopPanelRecovery, StartFailureCancelsRecording)
{
  application();
  initializeRos();
  FakeServices services;  ///< 允许录制但拒绝启用遥操的服务。
  services.allow_enable = false;
  services.start();
  tracker_teleoperated::TeleopPanel panel;  ///< 被测面板。
  auto client = std::make_shared<rclcpp::Node>("panel_recovery_client_1");  ///< 面板测试节点。
  tracker_teleoperated::TeleopPanelRecoveryHarness::initialize(panel, client);
  ASSERT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::ready(panel);
  }));
  tracker_teleoperated::TeleopPanelRecoveryHarness::start(panel);
  ASSERT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::idle(panel);
  }));
  EXPECT_FALSE(services.request_id.empty());
  EXPECT_EQ(services.request_state, "cancelled");
  EXPECT_GE(services.cancel_calls, 1);
  EXPECT_FALSE(services.enabled);
}

/** @brief 保存中撤销仍需等待请求终态，不提前开放操作。 */
TEST(TeleopPanelRecovery, WaitsForSavingCancellationTerminalState)
{
  application();
  initializeRos();
  FakeServices services;  ///< 撤销先返回接受、随后才进入终态的服务。
  services.request_state = "saving";
  services.start();
  tracker_teleoperated::TeleopPanel panel;  ///< 被测面板。
  auto client = std::make_shared<rclcpp::Node>("panel_recovery_client_2");  ///< 面板测试节点。
  tracker_teleoperated::TeleopPanelRecoveryHarness::initialize(panel, client);
  ASSERT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::ready(panel);
  }));
  tracker_teleoperated::TeleopPanelRecoveryHarness::beginRecovery(panel, "request-2", 1, true);
  EXPECT_TRUE(waitUntil(services, [&services]() {return services.cancel_calls > 0;}));
  EXPECT_FALSE(tracker_teleoperated::TeleopPanelRecoveryHarness::idle(panel));
  services.request_state = "cancelled";
  EXPECT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::idle(panel);
  }));
}

/** @brief 服务暂时离线时保持恢复，重新出现后撤销迟到的启动。 */
TEST(TeleopPanelRecovery, RecoversAfterServicesReturn)
{
  application();
  initializeRos();
  FakeServices services;  ///< 开始时没有服务的节点。
  tracker_teleoperated::TeleopPanel panel;  ///< 被测面板。
  auto client = std::make_shared<rclcpp::Node>("panel_recovery_client_3");  ///< 面板测试节点。
  tracker_teleoperated::TeleopPanelRecoveryHarness::initialize(panel, client);
  tracker_teleoperated::TeleopPanelRecoveryHarness::beginRecovery(panel, "request-3", 0, false);
  EXPECT_FALSE(tracker_teleoperated::TeleopPanelRecoveryHarness::idle(panel));
  services.start();
  EXPECT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::idle(panel);
  }));
  EXPECT_GE(services.cancel_calls, 1);
  EXPECT_EQ(services.disable_calls, 0);
}

/** @brief 录制已保存时保留数据并展示 UUID。 */
TEST(TeleopPanelRecovery, PreservesCompletedRecording)
{
  application();
  initializeRos();
  FakeServices services;  ///< 请求已保存完成的服务。
  services.request_state = "completed";
  services.start();
  tracker_teleoperated::TeleopPanel panel;  ///< 被测面板。
  auto client = std::make_shared<rclcpp::Node>("panel_recovery_client_4");  ///< 面板测试节点。
  tracker_teleoperated::TeleopPanelRecoveryHarness::initialize(panel, client);
  ASSERT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::ready(panel);
  }));
  tracker_teleoperated::TeleopPanelRecoveryHarness::beginRecovery(panel, "request-4", 0, false);
  EXPECT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::idle(panel);
  }));
  EXPECT_TRUE(tracker_teleoperated::TeleopPanelRecoveryHarness::message(panel).contains("recording-1"));
}

/** @brief 正常组合启动时录制 UUID 与遥操启用代次都被服务接收。 */
TEST(TeleopPanelRecovery, StartsRecordingAndTeleop)
{
  application();
  initializeRos();
  FakeServices services;  ///< 正常接受两个启动请求的服务。
  services.start();
  tracker_teleoperated::TeleopPanel panel;  ///< 被测面板。
  auto client = std::make_shared<rclcpp::Node>("panel_recovery_client_5");  ///< 面板测试节点。
  tracker_teleoperated::TeleopPanelRecoveryHarness::initialize(panel, client);
  ASSERT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::ready(panel);
  }));
  tracker_teleoperated::TeleopPanelRecoveryHarness::start(panel);
  ASSERT_TRUE(waitUntil(services, [&panel, &services]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::idle(panel) && services.enabled;
  }));
  EXPECT_FALSE(services.request_id.empty());
  EXPECT_EQ(services.generation, 1u);
  EXPECT_EQ(services.cancel_calls, 0);
}

/** @brief 组合启动超时后到达的录制响应仍按 UUID 撤销。 */
TEST(TeleopPanelRecovery, CancelsLateRecordingStart)
{
  application();
  initializeRos();
  FakeServices services;  ///< 先积压启动请求再恢复响应的服务。
  services.start();
  tracker_teleoperated::TeleopPanel panel;  ///< 被测面板。
  auto client = std::make_shared<rclcpp::Node>("panel_recovery_client_6");  ///< 面板测试节点。
  tracker_teleoperated::TeleopPanelRecoveryHarness::initialize(panel, client);
  ASSERT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::ready(panel);
  }));
  tracker_teleoperated::TeleopPanelRecoveryHarness::start(panel);
  tracker_teleoperated::TeleopPanelRecoveryHarness::expireCombined(panel);
  EXPECT_FALSE(tracker_teleoperated::TeleopPanelRecoveryHarness::idle(panel));
  ASSERT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::idle(panel);
  }));
  EXPECT_FALSE(services.request_id.empty());
  EXPECT_EQ(services.request_state, "cancelled");
  EXPECT_GE(services.cancel_calls, 1);
  EXPECT_FALSE(services.enabled);
}

/** @brief 录制成功但启用响应迟到时，恢复流程使用更高代次暂停。 */
TEST(TeleopPanelRecovery, FencesLateTeleopEnable)
{
  application();
  initializeRos();
  FakeServices services;  ///< 启用请求已发出但服务暂未处理。
  services.start();
  tracker_teleoperated::TeleopPanel panel;  ///< 被测面板。
  auto client = std::make_shared<rclcpp::Node>("panel_recovery_client_7");  ///< 面板测试节点。
  tracker_teleoperated::TeleopPanelRecoveryHarness::initialize(panel, client);
  ASSERT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::ready(panel);
  }));
  tracker_teleoperated::TeleopPanelRecoveryHarness::start(panel);
  services.executor.spin_some(std::chrono::milliseconds(5));
  ASSERT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::enableGeneration(panel) > 0;
  }));
  tracker_teleoperated::TeleopPanelRecoveryHarness::expireCombined(panel);
  ASSERT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::idle(panel);
  }));
  EXPECT_GE(services.disable_calls, 1);
  EXPECT_GT(services.generation,
    tracker_teleoperated::TeleopPanelRecoveryHarness::enableGeneration(panel));
  EXPECT_FALSE(services.enabled);
  EXPECT_EQ(services.request_state, "cancelled");
}

/** @brief 普通暂停须覆盖尚未到达控制节点的启用代次。 */
TEST(TeleopPanelRecovery, ManualPauseFencesPendingEnable)
{
  application();
  initializeRos();
  FakeServices services;  ///< 控制节点仍报告旧代次。
  services.start();
  tracker_teleoperated::TeleopPanel panel;  ///< 被测面板。
  auto client = std::make_shared<rclcpp::Node>("panel_recovery_client_pause");  ///< 面板测试节点。
  tracker_teleoperated::TeleopPanelRecoveryHarness::initialize(panel, client);
  ASSERT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::ready(panel);
  }));
  tracker_teleoperated::TeleopPanelRecoveryHarness::pauseAfterEnable(panel, 1);
  ASSERT_TRUE(waitUntil(services, [&services]() {return services.disable_calls > 0;}));
  EXPECT_EQ(services.generation, 2u);
  EXPECT_FALSE(services.enabled);
}

/** @brief 单独按 A 的启动超时也按 UUID 撤销迟到录制。 */
TEST(TeleopPanelRecovery, CancelsTimedOutManualStart)
{
  application();
  initializeRos();
  FakeServices services;  ///< 暂不执行单独开始请求的服务。
  services.start();
  tracker_teleoperated::TeleopPanel panel;  ///< 被测面板。
  auto client = std::make_shared<rclcpp::Node>("panel_recovery_client_8");  ///< 面板测试节点。
  tracker_teleoperated::TeleopPanelRecoveryHarness::initialize(panel, client);
  ASSERT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::ready(panel);
  }));
  tracker_teleoperated::TeleopPanelRecoveryHarness::startRecording(panel);
  tracker_teleoperated::TeleopPanelRecoveryHarness::expireRecording(panel);
  EXPECT_FALSE(tracker_teleoperated::TeleopPanelRecoveryHarness::idle(panel));
  ASSERT_TRUE(waitUntil(services, [&panel]() {
    return tracker_teleoperated::TeleopPanelRecoveryHarness::idle(panel);
  }));
  EXPECT_FALSE(services.request_id.empty());
  EXPECT_EQ(services.request_state, "cancelled");
  EXPECT_GE(services.cancel_calls, 1);
  EXPECT_EQ(services.disable_calls, 0);
}
