/**
 * @file tracker_pose_reader.cpp
 * @brief 实现 OpenVR 生命周期管理、Tracker 枚举和当前位姿采样。
 */

#include "vive_tracker/tracker_pose_reader.hpp"

#include <dlfcn.h>

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

/** 随节点安装、在运行时隔离加载的 OpenVR 客户端库名称。 */
constexpr char kOpenVrLibraryName[] = "libopenvr_api.so";

/** OpenVR 初始化入口的函数指针类型。 */
using OpenVrInitFunction = std::uint32_t (*)(vr::EVRInitError *,
                                             vr::EVRApplicationType,
                                             const char *);
/** OpenVR 关闭入口的函数指针类型。 */
using OpenVrShutdownFunction = void (*)();
/** OpenVR 通用接口查询入口的函数指针类型。 */
using OpenVrGetInterfaceFunction = void *(*)(const char *, vr::EVRInitError *);
/** OpenVR 接口版本检查入口的函数指针类型。 */
using OpenVrIsInterfaceValidFunction = bool (*)(const char *);
/** OpenVR 初始化错误说明入口的函数指针类型。 */
using OpenVrErrorDescriptionFunction = const char *(*)(vr::EVRInitError);

/**
 * @brief 从动态库解析一个函数入口。
 * @tparam FunctionType 函数指针类型。
 * @param library_handle 已打开的动态库句柄。
 * @param symbol_name 待解析的导出符号名称。
 * @param function 接收函数地址的指针，不能为空。
 * @param error_message 解析失败时接收错误信息；允许传入 nullptr。
 * @return 解析成功返回 true。
 */
template <typename FunctionType>
bool ResolveOpenVrFunction(void *library_handle, const char *symbol_name,
                           FunctionType *function,
                           std::string *error_message) {
  dlerror();
  /** dlsym 返回的未类型化函数地址。 */
  void *symbol_address = dlsym(library_handle, symbol_name);
  /** dlsym 线程局部错误说明。 */
  const char *dynamic_loader_error = dlerror();
  if (dynamic_loader_error != nullptr || symbol_address == nullptr) {
    if (error_message != nullptr) {
      *error_message = "failed to resolve " + std::string(symbol_name) +
                       " from " + kOpenVrLibraryName + ": " +
                       (dynamic_loader_error != nullptr ? dynamic_loader_error
                                                        : "symbol not found");
    }
    return false;
  }
  *function = reinterpret_cast<FunctionType>(symbol_address);
  return true;
}

/**
 * @brief 局部加载 OpenVR API，避免其私有 C++ 异常符号污染 ROS 进程。
 */
class OpenVrApi {
public:
  /**
   * @brief 创建尚未加载动态库的 API 包装器。
   */
  OpenVrApi() = default;

  /**
   * @brief 关闭尚未结束的 OpenVR 会话并卸载动态库。
   */
  ~OpenVrApi() { Unload(); }

  OpenVrApi(const OpenVrApi &) = delete;
  OpenVrApi &operator=(const OpenVrApi &) = delete;

  /**
   * @brief 以局部深度绑定方式加载 OpenVR 及所需入口。
   * @param error_message 加载失败时接收错误信息；允许传入 nullptr。
   * @return 全部入口加载成功返回 true。
   */
  bool Load(std::string *error_message) {
    if (library_handle_ != nullptr) {
      return true;
    }

    /** 限制 OpenVR 符号可见性并优先使用其内部依赖的加载标志。 */
    int dynamic_loader_flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
    dynamic_loader_flags |= RTLD_DEEPBIND;
#endif
    library_handle_ = dlopen(kOpenVrLibraryName, dynamic_loader_flags);
    if (library_handle_ == nullptr) {
      if (error_message != nullptr) {
        /** dlopen 线程局部错误说明。 */
        const char *dynamic_loader_error = dlerror();
        *error_message =
            "failed to load " + std::string(kOpenVrLibraryName) + ": " +
            (dynamic_loader_error != nullptr ? dynamic_loader_error
                                             : "unknown dynamic loader error");
      }
      return false;
    }

    if (!ResolveOpenVrFunction(library_handle_, "VR_InitInternal2",
                               &init_function_, error_message) ||
        !ResolveOpenVrFunction(library_handle_, "VR_ShutdownInternal",
                               &shutdown_function_, error_message) ||
        !ResolveOpenVrFunction(library_handle_, "VR_GetGenericInterface",
                               &get_interface_function_, error_message) ||
        !ResolveOpenVrFunction(library_handle_, "VR_IsInterfaceVersionValid",
                               &is_interface_valid_function_, error_message) ||
        !ResolveOpenVrFunction(
            library_handle_, "VR_GetVRInitErrorAsEnglishDescription",
            &error_description_function_, error_message)) {
      Unload();
      return false;
    }
    return true;
  }

  /**
   * @brief 初始化后台 OpenVR 会话并取得系统接口。
   * @param init_error 接收 OpenVR 初始化错误码，不能为空。
   * @return 初始化成功时返回系统接口，否则返回 nullptr。
   */
  vr::IVRSystem *Initialize(vr::EVRInitError *init_error) {
    /** OpenVR 内部会话令牌，仅用于确认初始化入口已经执行。 */
    const std::uint32_t session_token =
        init_function_(init_error, vr::VRApplication_Background, nullptr);
    (void)session_token;
    if (*init_error == vr::VRInitError_None &&
        !is_interface_valid_function_(vr::IVRSystem_Version)) {
      *init_error = vr::VRInitError_Init_InterfaceNotFound;
    }

    /** 通过版本化接口名称取得的 OpenVR 系统对象。 */
    vr::IVRSystem *vr_system = nullptr;
    if (*init_error == vr::VRInitError_None) {
      vr_system = static_cast<vr::IVRSystem *>(
          get_interface_function_(vr::IVRSystem_Version, init_error));
    }
    if (*init_error == vr::VRInitError_None && vr_system != nullptr) {
      *init_error = vr_system->SetSDKVersion(
          vr::k_nSteamVRVersionMajor, vr::k_nSteamVRVersionMinor,
          vr::k_nSteamVRVersionBuild);
    }
    if (*init_error != vr::VRInitError_None || vr_system == nullptr) {
      shutdown_function_();
      return nullptr;
    }
    session_active_ = true;
    return vr_system;
  }

  /**
   * @brief 返回 OpenVR 初始化错误的英文说明。
   * @param init_error OpenVR 初始化错误码。
   * @return OpenVR 管理的只读说明字符串。
   */
  const char *GetErrorDescription(vr::EVRInitError init_error) const {
    return error_description_function_(init_error);
  }

  /**
   * @brief 正常关闭当前 OpenVR 会话。
   */
  void Shutdown() {
    if (session_active_ && shutdown_function_ != nullptr) {
      shutdown_function_();
      session_active_ = false;
    }
  }

private:
  /**
   * @brief 卸载 OpenVR 动态库并清空全部函数入口。
   */
  void Unload() {
    Shutdown();
    if (library_handle_ != nullptr) {
      dlclose(library_handle_);
      library_handle_ = nullptr;
    }
    init_function_ = nullptr;
    shutdown_function_ = nullptr;
    get_interface_function_ = nullptr;
    is_interface_valid_function_ = nullptr;
    error_description_function_ = nullptr;
  }

  /** OpenVR 动态库句柄。 */
  void *library_handle_{nullptr};
  /** OpenVR 初始化入口。 */
  OpenVrInitFunction init_function_{nullptr};
  /** OpenVR 关闭入口。 */
  OpenVrShutdownFunction shutdown_function_{nullptr};
  /** OpenVR 通用接口查询入口。 */
  OpenVrGetInterfaceFunction get_interface_function_{nullptr};
  /** OpenVR 接口版本检查入口。 */
  OpenVrIsInterfaceValidFunction is_interface_valid_function_{nullptr};
  /** OpenVR 初始化错误说明入口。 */
  OpenVrErrorDescriptionFunction error_description_function_{nullptr};
  /** 当前是否存在需要正常关闭的 OpenVR 会话。 */
  bool session_active_{false};
};

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
  /** 隔离加载并调用 OpenVR API 的包装器。 */
  OpenVrApi openvr_api{};
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
    impl_->openvr_api.Shutdown();
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

  if (!impl_->openvr_api.Load(error_message)) {
    return false;
  }

  /** OpenVR 初始化错误码。 */
  vr::EVRInitError init_error = vr::VRInitError_None;
  impl_->vr_system = impl_->openvr_api.Initialize(&init_error);
  if (init_error != vr::VRInitError_None || impl_->vr_system == nullptr) {
    impl_->vr_system = nullptr;
    if (error_message != nullptr) {
      /** OpenVR 对初始化错误的英文说明。 */
      const char *error_description =
          impl_->openvr_api.GetErrorDescription(init_error);
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
 * @return 当前查询的统一时间上下文和所有 Generic Tracker 采样；未初始化时样本为空。
 */
TrackerPoseBatch TrackerPoseReader::ReadPoses(TrackingOrigin origin) const {
  /** 本次返回的统一查询批次。 */
  TrackerPoseBatch batch{};
  if (impl_->vr_system == nullptr) {
    return batch;
  }

  /** OpenVR 当前会话中全部设备的位姿数组。 */
  std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount> poses{};
  /** 查询前读取的系统时钟，用于异常时的同调用区间回退。 */
  const auto system_before = std::chrono::system_clock::now();
  /** 查询前读取的稳定时钟，用于估计 OpenVR 调用中点。 */
  const auto steady_before = std::chrono::steady_clock::now();
  impl_->vr_system->GetDeviceToAbsoluteTrackingPose(
      ToOpenVrOrigin(origin), 0.0F, poses.data(),
      static_cast<std::uint32_t>(poses.size()));
  /** 查询后读取的稳定时钟，用于估计 OpenVR 调用中点。 */
  const auto steady_after = std::chrono::steady_clock::now();
  /** 查询后读取的系统时钟，用于异常时的同调用区间回退。 */
  const auto system_after = std::chrono::system_clock::now();
  batch.timing.steady_before_ns = std::chrono::duration_cast<
      std::chrono::nanoseconds>(steady_before.time_since_epoch()).count();
  batch.timing.steady_after_ns = std::chrono::duration_cast<
      std::chrono::nanoseconds>(steady_after.time_since_epoch()).count();
  batch.timing.system_before_ns = std::chrono::duration_cast<
      std::chrono::nanoseconds>(system_before.time_since_epoch()).count();
  batch.timing.system_after_ns = std::chrono::duration_cast<
      std::chrono::nanoseconds>(system_after.time_since_epoch()).count();

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
    batch.samples.push_back(std::move(sample));
  }
  return batch;
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
