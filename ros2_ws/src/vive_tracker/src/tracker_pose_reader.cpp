/**
 * @file tracker_pose_reader.cpp
 * @brief 实现 OpenVR 生命周期管理、Tracker 枚举和当前位姿采样。
 */

#include "vive_tracker/tracker_pose_reader.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <openvr.h>

#include "vive_tracker/pose_math.hpp"

namespace vive_tracker {
namespace {

/**
 * @brief 将公共跟踪原点转换为 OpenVR 枚举。
 * @param origin 公共跟踪原点。
 * @return 对应的 OpenVR 跟踪原点。
 */
vr::ETrackingUniverseOrigin ToOpenVrOrigin(TrackingOrigin origin) noexcept {
  switch (origin) {
  case TrackingOrigin::kSeated:
    return vr::TrackingUniverseSeated;
  case TrackingOrigin::kRaw:
    return vr::TrackingUniverseRawAndUncalibrated;
  case TrackingOrigin::kStanding:
  default:
    return vr::TrackingUniverseStanding;
  }
}

/**
 * @brief 将 OpenVR 跟踪结果转换为公共状态。
 * @param result OpenVR 跟踪结果。
 * @return 对应的公共跟踪状态。
 */
TrackingState ToTrackingState(vr::ETrackingResult result) noexcept {
  switch (result) {
  case vr::TrackingResult_Uninitialized:
    return TrackingState::kUninitialized;
  case vr::TrackingResult_Calibrating_InProgress:
    return TrackingState::kCalibratingInProgress;
  case vr::TrackingResult_Calibrating_OutOfRange:
    return TrackingState::kCalibratingOutOfRange;
  case vr::TrackingResult_Running_OK:
    return TrackingState::kRunningOk;
  case vr::TrackingResult_Running_OutOfRange:
    return TrackingState::kRunningOutOfRange;
  case vr::TrackingResult_Fallback_RotationOnly:
    return TrackingState::kFallbackRotationOnly;
  default:
    return TrackingState::kUnknown;
  }
}

/**
 * @brief 读取 Tracker 序列号，并在属性不可用时生成稳定的索引标识。
 * @param vr_system 已初始化的 OpenVR 系统接口。
 * @param device_index OpenVR 设备索引。
 * @return Tracker 序列号或基于设备索引生成的占位标识。
 */
std::string ReadSerialNumber(vr::IVRSystem *vr_system,
                             vr::TrackedDeviceIndex_t device_index) {
  /** 第一次属性查询的错误状态。 */
  vr::ETrackedPropertyError property_error = vr::TrackedProp_Success;
  /** 包含字符串终止符的属性缓冲区长度。 */
  const std::uint32_t buffer_size = vr_system->GetStringTrackedDeviceProperty(
      device_index, vr::Prop_SerialNumber_String, nullptr, 0, &property_error);
  if (buffer_size == 0) {
    return "unknown-index-" + std::to_string(device_index);
  }

  /** 保存序列号文本的缓冲区。 */
  std::vector<char> buffer(buffer_size, '\0');
  property_error = vr::TrackedProp_Success;
  vr_system->GetStringTrackedDeviceProperty(
      device_index, vr::Prop_SerialNumber_String, buffer.data(), buffer_size,
      &property_error);
  if (property_error != vr::TrackedProp_Success || buffer.front() == '\0') {
    return "unknown-index-" + std::to_string(device_index);
  }
  return std::string(buffer.data());
}

} // namespace

/**
 * @brief 保存不暴露到公共头文件的 OpenVR 系统接口。
 */
class TrackerPoseReader::Impl {
public:
  /** 已初始化的 OpenVR 系统接口。 */
  vr::IVRSystem *vr_system{nullptr};
};

/**
 * @brief 创建尚未初始化的 Tracker 位姿读取器。
 */
TrackerPoseReader::TrackerPoseReader() : impl_(std::make_unique<Impl>()) {}

/**
 * @brief 关闭已经初始化的 OpenVR 会话。
 */
TrackerPoseReader::~TrackerPoseReader() {
  if (impl_->vr_system != nullptr) {
    vr::VR_Shutdown();
    impl_->vr_system = nullptr;
  }
}

/**
 * @brief 以后台应用类型连接到已经运行的 SteamVR。
 * @param error_message 初始化失败时接收可读错误信息；允许传入 nullptr。
 * @return 初始化成功或读取器已经初始化时返回 true。
 */
bool TrackerPoseReader::Initialize(std::string *error_message) {
  if (impl_->vr_system != nullptr) {
    return true;
  }

  /** OpenVR 初始化错误码。 */
  vr::EVRInitError init_error = vr::VRInitError_None;
  impl_->vr_system = vr::VR_Init(&init_error, vr::VRApplication_Background);
  if (init_error != vr::VRInitError_None || impl_->vr_system == nullptr) {
    impl_->vr_system = nullptr;
    if (error_message != nullptr) {
      /** OpenVR 对初始化错误的英文说明。 */
      const char *error_description =
          vr::VR_GetVRInitErrorAsEnglishDescription(init_error);
      *error_message = error_description != nullptr ? error_description
                                                    : "Unknown OpenVR error";
    }
    return false;
  }
  return true;
}

/**
 * @brief 查询读取器是否已经成功连接 OpenVR。
 * @return 已初始化返回 true。
 */
bool TrackerPoseReader::IsInitialized() const noexcept {
  return impl_->vr_system != nullptr;
}

/**
 * @brief 读取当前所有 Generic Tracker 的状态和位姿。
 * @param origin 本次查询使用的 SteamVR 跟踪原点。
 * @return 当前会话中所有 Generic Tracker 的采样结果；未初始化时返回空数组。
 */
std::vector<TrackerPoseSample>
TrackerPoseReader::ReadPoses(TrackingOrigin origin) const {
  /** 本次返回的 Tracker 采样结果。 */
  std::vector<TrackerPoseSample> samples{};
  if (impl_->vr_system == nullptr) {
    return samples;
  }

  /** OpenVR 当前会话中全部设备的位姿数组。 */
  std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount> poses{};
  impl_->vr_system->GetDeviceToAbsoluteTrackingPose(
      ToOpenVrOrigin(origin), 0.0F, poses.data(),
      static_cast<std::uint32_t>(poses.size()));

  /** 本批位姿读取完成后的主机系统时间。 */
  const auto sample_time = std::chrono::system_clock::now();
  /** 本批位姿共用的 Unix 纳秒时间戳。 */
  const std::int64_t sample_time_unix_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          sample_time.time_since_epoch())
          .count();

  for (vr::TrackedDeviceIndex_t device_index = 0;
       device_index < vr::k_unMaxTrackedDeviceCount; ++device_index) {
    if (impl_->vr_system->GetTrackedDeviceClass(device_index) !=
        vr::TrackedDeviceClass_GenericTracker) {
      continue;
    }

    /** 当前设备的 OpenVR 位姿记录。 */
    const vr::TrackedDevicePose_t &openvr_pose = poses[device_index];
    /** 当前 Tracker 的公共采样记录。 */
    TrackerPoseSample sample{};
    sample.sample_time_unix_ns = sample_time_unix_ns;
    sample.device_index = device_index;
    sample.serial_number = ReadSerialNumber(impl_->vr_system, device_index);
    sample.device_connected = openvr_pose.bDeviceIsConnected;
    sample.pose_valid =
        openvr_pose.bDeviceIsConnected && openvr_pose.bPoseIsValid;
    sample.tracking_state = ToTrackingState(openvr_pose.eTrackingResult);
    if (sample.pose_valid) {
      sample.pose =
          ConvertOpenVrMatrixToPose(openvr_pose.mDeviceToAbsoluteTracking);
    }
    samples.push_back(std::move(sample));
  }
  return samples;
}

/**
 * @brief 将跟踪状态转换为稳定的输出字符串。
 * @param state 跟踪状态。
 * @return 对应的英文小写状态字符串。
 */
std::string_view TrackingStateToString(TrackingState state) noexcept {
  switch (state) {
  case TrackingState::kUninitialized:
    return "uninitialized";
  case TrackingState::kCalibratingInProgress:
    return "calibrating_in_progress";
  case TrackingState::kCalibratingOutOfRange:
    return "calibrating_out_of_range";
  case TrackingState::kRunningOk:
    return "running_ok";
  case TrackingState::kRunningOutOfRange:
    return "running_out_of_range";
  case TrackingState::kFallbackRotationOnly:
    return "fallback_rotation_only";
  case TrackingState::kUnknown:
  default:
    return "unknown";
  }
}

} // namespace vive_tracker
