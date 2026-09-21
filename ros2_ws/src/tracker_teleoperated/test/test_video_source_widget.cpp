/**
 * @file test_video_source_widget.cpp
 * @brief 验证设备目录、角色预览、录制互锁和配置保存。
 */
#include <gtest/gtest.h>
#include <QApplication>
#include <QComboBox>
#include <QLabel>
#include <QTest>
#include "tracker_teleoperated/video_source_widget.hpp"

/** @brief 复用 Qt 测试进程。 @return 应用实例。 */
QApplication * application();

namespace
{
/** @brief 默认 UMI 相机物理端口。 */
const std::string kUmiDevice =
  "/dev/v4l/by-path/pci-test-usb-0:2.4:1.0-video-index0";
/** @brief 默认末端相机物理端口。 */
const std::string kWristDevice =
  "/dev/v4l/by-path/pci-test-usb-0:2.3:1.0-video-index0";
/** @brief 测试切换使用的备用物理端口。 */
const std::string kAltDevice =
  "/dev/v4l/by-path/pci-test-usb-0:2.5:1.0-video-index0";

/** @brief 构造双相机诊断。 @param[in] editable 是否允许选择。 @return 状态快照。 */
diagnostic_msgs::msg::DiagnosticArray status(bool editable)
{
  diagnostic_msgs::msg::DiagnosticArray message;  ///< 双角色状态。
  for (const auto & name : {"umi_camera", "wrist_camera"}) {
    diagnostic_msgs::msg::DiagnosticStatus item;  ///< 单相机诊断。
    item.name = name;
    for (const auto & pair : std::map<std::string, std::string>{
        {"video_device", name == std::string("umi_camera") ? kUmiDevice : kWristDevice},
        {"image_topic", "/" + std::string(name) + "/image_raw"},
        {"source_editable", editable ? "true" : "false"}, {"source_state", "ready"},
        {"source_reason", editable ? "" : "录制期间禁止切换"}}) {
      diagnostic_msgs::msg::KeyValue value;  ///< 机器可读字段。
      value.key = pair.first; value.value = pair.second; item.values.push_back(value);
    }
    message.status.push_back(item);
  }
  return message;
}

/** @brief 构造数字顺序不同的设备目录。 @return 扫描结果。 */
diagnostic_msgs::msg::DiagnosticArray devices()
{
  diagnostic_msgs::msg::DiagnosticArray message;  ///< 模拟扫描结果。
  for (const auto & entry : std::map<std::string, std::string>{
      {kWristDevice, "2.3"}, {kUmiDevice, "2.4"}, {kAltDevice, "2.5"}}) {
    diagnostic_msgs::msg::DiagnosticStatus item;  ///< USB 采集入口。
    item.name = entry.first;
    for (const auto & pair : std::map<std::string, std::string>{
        {"name", "USB Camera"}, {"physical_port", entry.second},
        {"kernel_device", entry.second == "2.3" ? "/dev/video0" : "/dev/video2"}}) {
      diagnostic_msgs::msg::KeyValue value;  ///< 设备目录字段。
      value.key = pair.first; value.value = pair.second; item.values.push_back(value);
    }
    message.status.push_back(item);
  }
  return message;
}
}

/** @brief 设备数字排序、缺失保留和固定角色话题覆盖 Displays 修改。 */
TEST(VideoSources, DeviceCatalogAndStableRoleTopics)
{
  application();
  std::map<QString, QString> displays{{"UMI 视频", "/old"}, {"末端视频", "/old"}};  ///< 模拟显示。
  tracker_teleoperated::VideoSourceWidget widget(nullptr,
    [&displays](const QString & name) {return displays[name];},
    [&displays](const QString & name, const QString & topic) {displays[name] = topic; return true;});  ///< 控件。
  widget.updateDevices(devices());
  widget.updateSources(status(true));
  auto * combo = widget.findChild<QComboBox *>("umi_video_source");  ///< UMI 设备选择。
  ASSERT_NE(combo, nullptr);
  EXPECT_EQ(combo->itemData(0).toString(), QString::fromStdString(kWristDevice));
  EXPECT_EQ(combo->itemData(2).toString(), QString::fromStdString(kAltDevice));
  EXPECT_TRUE(combo->itemText(0).contains("USB Camera"));
  EXPECT_TRUE(combo->itemText(0).contains("端口 2.3"));
  EXPECT_EQ(combo->currentData().toString(), QString::fromStdString(kUmiDevice));
  EXPECT_EQ(displays["UMI 视频"], "/umi_camera/image_raw");
  displays["UMI 视频"] = "/other";
  widget.refreshSources();
  EXPECT_EQ(displays["UMI 视频"], "/umi_camera/image_raw");
  widget.updateDevices(diagnostic_msgs::msg::DiagnosticArray());
  EXPECT_EQ(combo->currentData().toString(), QString::fromStdString(kUmiDevice));
  EXPECT_TRUE(combo->currentText().contains("设备不存在"));
}

/** @brief 录制和外部归属禁用两路选择，配置保存确认设备。 */
TEST(VideoSources, ReadOnlyAndPersistedDevices)
{
  application();
  tracker_teleoperated::VideoSourceWidget widget(nullptr,
    [](const QString &) {return QString("/image");},
    [](const QString &, const QString &) {return true;});  ///< 控件。
  widget.updateDevices(devices());
  widget.updateSources(status(false));
  EXPECT_FALSE(widget.findChild<QComboBox *>("umi_video_source")->isEnabled());
  EXPECT_FALSE(widget.findChild<QComboBox *>("wrist_video_source")->isEnabled());
  rviz_common::Config saved;  ///< 待保存配置。
  widget.save(saved);
  QString value;  ///< 保存的设备路径。
  EXPECT_TRUE(saved.mapGetString("UmiVideoDevice", &value));
  EXPECT_EQ(value, QString::fromStdString(kUmiDevice));
  tracker_teleoperated::VideoSourceWidget restored;  ///< 第二实例读取保存值。
  restored.load(saved);
  rviz_common::Config copied;  ///< 恢复后重新保存。
  restored.save(copied);
  EXPECT_TRUE(copied.mapGetString("WristVideoDevice", &value));
  EXPECT_EQ(value, QString::fromStdString(kWristDevice));
  widget.updateSources(status(true));
  EXPECT_TRUE(widget.findChild<QComboBox *>("umi_video_source")->isEnabled());
  EXPECT_FALSE(widget.wristBusy());
}

/** @brief 显示缺失时禁用选择，目录错误保留明确提示。 */
TEST(VideoSources, MissingDisplay)
{
  application();
  tracker_teleoperated::VideoSourceWidget widget(nullptr,
    [](const QString &) {return QString();});  ///< 没有默认显示的控件。
  widget.updateSources(status(true));
  EXPECT_FALSE(widget.findChild<QComboBox *>("umi_video_source")->isEnabled());
  EXPECT_TRUE(widget.findChild<QLabel *>("video_source_status")->text().contains("未找到"));
}

/** @brief 真实参数服务确认设备请求，验证去重、两路互锁和卸载回调安全。 */
TEST(VideoSources, AsyncDeviceRequestAndUnload)
{
  application();
  rclcpp::init(0, nullptr);
  {
    auto node = std::make_shared<rclcpp::Node>("video_device_widget_test");  ///< 面板测试节点。
    unsigned calls = 0;  ///< 后端接收请求次数。
    auto service = node->create_service<rcl_interfaces::srv::SetParametersAtomically>(
      "/tracker_component_manager/set_parameters_atomically",
      [&calls](const std::shared_ptr<rcl_interfaces::srv::SetParametersAtomically::Request> request,
      std::shared_ptr<rcl_interfaces::srv::SetParametersAtomically::Response> response) {
        ++calls;
        EXPECT_EQ(request->parameters.at(0).name, "umi_video_device");
        EXPECT_EQ(request->parameters.at(0).value.string_value, kAltDevice);
        response->result.successful = true;
      });  ///< 本机模拟管理器参数服务。
    rclcpp::executors::SingleThreadedExecutor executor;  ///< 在测试线程中执行 ROS 回调。
    executor.add_node(node);
    auto widget = std::make_unique<tracker_teleoperated::VideoSourceWidget>(nullptr,
      [](const QString &) {return QString("/image");},
      [](const QString &, const QString &) {return true;});  ///< 支持模拟显示的真实 ROS 控件。
    widget->initialize(node, nullptr);
    widget->updateDevices(devices());
    widget->updateSources(status(true));
    for (int i = 0; i < 10; ++i) {executor.spin_some(); QTest::qWait(20);}
    auto * combo = widget->findChild<QComboBox *>("umi_video_source");  ///< 用户操作入口。
    combo->setCurrentIndex(combo->findData(QString::fromStdString(kAltDevice)));
    QMetaObject::invokeMethod(combo, "activated", Q_ARG(int, combo->currentIndex()));
    QMetaObject::invokeMethod(combo, "activated", Q_ARG(int, combo->currentIndex()));
    EXPECT_TRUE(widget->wristBusy());
    EXPECT_FALSE(widget->findChild<QComboBox *>("wrist_video_source")->isEnabled());
    for (int i = 0; i < 20 && calls == 0; ++i) {executor.spin_some(); QTest::qWait(20);}
    executor.spin_some();
    EXPECT_EQ(calls, 1u);
    EXPECT_TRUE(widget->wristBusy());
    auto confirmed = status(true);  ///< 后端最终确认，与参数接受响应分开。
    diagnostic_msgs::msg::KeyValue requested;  ///< 当前请求设备。
    requested.key = "requested_video_device"; requested.value = kAltDevice;
    confirmed.status[0].values.push_back(requested);
    for (auto & value : confirmed.status[0].values) {
      if (value.key == "video_device") {value.value = kAltDevice;}
    }
    widget->updateSources(confirmed);
    EXPECT_FALSE(widget->wristBusy());
    QMetaObject::invokeMethod(combo, "activated", Q_ARG(int, combo->currentIndex()));
    widget.reset();
    for (int i = 0; i < 5; ++i) {executor.spin_some(); QTest::qWait(20);}
    executor.remove_node(node);
  }
  rclcpp::shutdown();
}
