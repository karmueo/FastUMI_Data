/**
 * @file test_exposure_priority.cpp
 * @brief 用可注入控制接口验证曝光优先读取、关闭、读回及各种失败的记录。
 */

#include <gtest/gtest.h>

#include <cerrno>
#include <stdexcept>

#include "fastumi_usb_camera/exposure_priority.hpp"

using fastumi_usb_camera::configure_exposure_priority;
using fastumi_usb_camera::ControlIo;
using fastumi_usb_camera::ExposurePriorityRequest;

namespace
{

/** @brief 模拟设备：可配置查询/读/写错误、写入是否生效以及读取失败时的伪值。 */
struct FakeDevice
{
  int query_error{0};
  bool disabled_flag{false};
  int get_error{0};
  int set_error{0};
  bool write_takes_effect{true};
  int value{1};
  int bogus_value_on_error{0};  ///< 读取失败时仍写入结构体的伪值（模拟 EIO 后打印 0）。

  ControlIo io()
  {
    ControlIo control;
    control.query = [this](v4l2_queryctrl & query) {
        if (disabled_flag) {
          query.flags |= V4L2_CTRL_FLAG_DISABLED;
        }
        return query_error;
      };
    control.get = [this](v4l2_control & ctrl) {
        if (get_error != 0) {
          ctrl.value = bogus_value_on_error;
          return get_error;
        }
        ctrl.value = value;
        return 0;
      };
    control.set = [this](v4l2_control & ctrl) {
        if (set_error != 0) {
          return set_error;
        }
        if (write_takes_effect) {
          value = ctrl.value;
        }
        return 0;
      };
    return control;
  }
};

}  // namespace

TEST(ExposurePriority, DisableIsConfirmedOnlyAfterReadback)
{
  FakeDevice device;
  const auto status = configure_exposure_priority(device.io(), ExposurePriorityRequest::kDisable);
  EXPECT_EQ(status.state, "disabled_confirmed");
  EXPECT_TRUE(status.confirmed_disabled());
  EXPECT_EQ(status.value, 0);
}

TEST(ExposurePriority, ReadbackStillOneIsNotDisabled)
{
  FakeDevice device;
  device.write_takes_effect = false;
  const auto status = configure_exposure_priority(device.io(), ExposurePriorityRequest::kDisable);
  EXPECT_EQ(status.state, "still_enabled");
  EXPECT_EQ(status.value, 1);
  EXPECT_FALSE(status.confirmed_disabled());
}

TEST(ExposurePriority, MissingControlIsUnsupported)
{
  FakeDevice device;
  device.query_error = EINVAL;
  EXPECT_EQ(
    configure_exposure_priority(device.io(), ExposurePriorityRequest::kDisable).state,
    "unsupported");
  FakeDevice disabled;
  disabled.disabled_flag = true;
  EXPECT_EQ(
    configure_exposure_priority(disabled.io(), ExposurePriorityRequest::kLeave).state,
    "unsupported");
}

TEST(ExposurePriority, EioReadIsUnknownNotZero)
{
  FakeDevice device;
  device.get_error = EIO;
  device.bogus_value_on_error = 0;
  const auto status = configure_exposure_priority(device.io(), ExposurePriorityRequest::kLeave);
  EXPECT_EQ(status.state, "read_error");
  EXPECT_EQ(status.value, -1);
  EXPECT_NE(status.summary().find("value=unknown"), std::string::npos);
  EXPECT_FALSE(status.confirmed_disabled());
}

TEST(ExposurePriority, WriteAndReadbackErrorsAreRecorded)
{
  FakeDevice write_failure;
  write_failure.set_error = EIO;
  EXPECT_EQ(
    configure_exposure_priority(write_failure.io(), ExposurePriorityRequest::kDisable).state,
    "write_error");
  FakeDevice readback_failure;
  readback_failure.get_error = EIO;
  const auto status = configure_exposure_priority(
    readback_failure.io(), ExposurePriorityRequest::kDisable);
  EXPECT_EQ(status.state, "readback_error");
  EXPECT_FALSE(status.confirmed_disabled());
  FakeDevice query_failure;
  query_failure.query_error = EIO;
  EXPECT_EQ(
    configure_exposure_priority(query_failure.io(), ExposurePriorityRequest::kLeave).state,
    "query_error");
}

TEST(ExposurePriority, LeaveRecordsCurrentValue)
{
  FakeDevice device;
  const auto status = configure_exposure_priority(device.io(), ExposurePriorityRequest::kLeave);
  EXPECT_EQ(status.state, "unchanged");
  EXPECT_EQ(status.value, 1);
  EXPECT_THROW(
    fastumi_usb_camera::parse_exposure_priority_request("off"), std::invalid_argument);
}
