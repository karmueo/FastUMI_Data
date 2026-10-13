/**
 * @file exposure_priority.hpp
 * @brief 读取或关闭 ``V4L2_CID_EXPOSURE_AUTO_PRIORITY``，并如实记录控制状态。
 *
 * 该控制为 1 时，自动曝光可以通过动态降低帧率来延长曝光；设为 0 禁止这种降帧，
 * 不要求改成手动曝光。只有写入成功且读回为 0 才记为 ``disabled_confirmed``。
 * 读取失败（例如 EIO）时状态为未知，绝不把失败后打印的数值当作关闭成功；设备不
 * 支持、写入失败或读回仍为 1 都保留明确标记，帧率门限不因此放宽。
 * 设备访问经 :struct:`ControlIo` 注入，测试可模拟各种失败。
 */

#ifndef FASTUMI_USB_CAMERA__EXPOSURE_PRIORITY_HPP_
#define FASTUMI_USB_CAMERA__EXPOSURE_PRIORITY_HPP_

#include <linux/videodev2.h>

#include <cerrno>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

namespace fastumi_usb_camera
{

/** @brief 控制访问接口：每个函数返回 0 表示成功，否则返回 errno。 */
struct ControlIo
{
  std::function<int(v4l2_queryctrl &)> query;
  std::function<int(v4l2_control &)> get;
  std::function<int(v4l2_control &)> set;
};

/** @brief 启动时对曝光优先控制的请求。 */
enum class ExposurePriorityRequest
{
  kLeave,  ///< 不修改，只读取并记录当前值。
  kDisable,  ///< 写入 0 并读回确认。
};

/**
 * @brief 解析 ``exposure_auto_priority`` 参数。
 * @throws std::invalid_argument 取值不是 ``leave`` 或 ``disable``。
 */
inline ExposurePriorityRequest parse_exposure_priority_request(const std::string & text)
{
  if (text == "leave") {
    return ExposurePriorityRequest::kLeave;
  }
  if (text == "disable") {
    return ExposurePriorityRequest::kDisable;
  }
  throw std::invalid_argument("exposure_auto_priority must be leave or disable");
}

/** @brief 控制状态记录。 */
struct ExposurePriorityStatus
{
  /**
   * 状态：``unsupported``、``query_error``、``read_error``（当前值未知）、
   * ``unchanged``（已读取当前值）、``write_error``、``readback_error``、
   * ``still_enabled``（写入后读回仍非 0）或 ``disabled_confirmed``。
   */
  std::string state{"not_checked"};
  int value{-1};  ///< 成功读取的控制值；未知时为 -1。
  std::string detail;  ///< 失败的 errno 文本等补充说明。

  /** @return 是否已写入并读回确认关闭。 */
  bool confirmed_disabled() const {return state == "disabled_confirmed";}

  /** @return 单行描述，例如 ``read_error(value=unknown, Input/output error)``。 */
  std::string summary() const
  {
    std::string text = state + "(value=" + (value < 0 ? std::string("unknown") :
      std::to_string(value));
    if (!detail.empty()) {
      text += ", " + detail;
    }
    return text + ")";
  }
};

/** @brief 按请求读取或关闭曝光优先控制，返回如实的状态。 */
inline ExposurePriorityStatus configure_exposure_priority(
  const ControlIo & io, ExposurePriorityRequest request)
{
  ExposurePriorityStatus status;
  v4l2_queryctrl query{};
  query.id = V4L2_CID_EXPOSURE_AUTO_PRIORITY;
  const int query_error = io.query(query);
  if (query_error == EINVAL || (query_error == 0 && (query.flags & V4L2_CTRL_FLAG_DISABLED) != 0U)) {
    status.state = "unsupported";
    return status;
  }
  if (query_error != 0) {
    status.state = "query_error";
    status.detail = std::strerror(query_error);
    return status;
  }
  v4l2_control control{};
  control.id = V4L2_CID_EXPOSURE_AUTO_PRIORITY;
  if (request == ExposurePriorityRequest::kLeave) {
    const int read_error = io.get(control);
    if (read_error != 0) {
      status.state = "read_error";
      status.detail = std::strerror(read_error);
      return status;
    }
    status.state = "unchanged";
    status.value = control.value;
    return status;
  }
  control.value = 0;
  const int write_error = io.set(control);
  if (write_error != 0) {
    status.state = "write_error";
    status.detail = std::strerror(write_error);
    return status;
  }
  v4l2_control readback{};
  readback.id = V4L2_CID_EXPOSURE_AUTO_PRIORITY;
  readback.value = -1;
  const int readback_error = io.get(readback);
  if (readback_error != 0) {
    status.state = "readback_error";
    status.detail = std::strerror(readback_error);
    return status;
  }
  status.value = readback.value;
  status.state = readback.value == 0 ? "disabled_confirmed" : "still_enabled";
  return status;
}

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__EXPOSURE_PRIORITY_HPP_
