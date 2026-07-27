/**
 * @file vive_tracker_node.cpp
 * @brief 实现 VIVE Tracker 绝对位姿、首帧里程计、轨迹和多级 TF 发布。
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <fastumi_interfaces/msg/tracker_status.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include "vive_tracker/path_history.hpp"
#include "vive_tracker/pose_math.hpp"
#include "vive_tracker/tracker_pose_reader.hpp"
#include "vive_tracker/tracker_pose_types.hpp"

namespace vive_tracker {
namespace {

/** 失去目标时相邻告警之间的最小毫秒数。 */
constexpr std::int64_t kWarningThrottleMs = 1000;
/** 节点允许的最大采样频率，单位为 Hz。 */
constexpr double kMaximumPublishRateHz = 1000.0;

/**
 * @brief 将参数字符串解析为 SteamVR 跟踪原点。
 * @param value 待解析的参数值。
 * @return 对应的跟踪原点枚举。
 * @throws std::invalid_argument 参数值不受支持时抛出。
 */
TrackingOrigin ParseTrackingOrigin(const std::string &value) {
  if (value == "standing") {
    return TrackingOrigin::kStanding;
  }
  if (value == "seated") {
    return TrackingOrigin::kSeated;
  }
  if (value == "raw") {
    return TrackingOrigin::kRaw;
  }
  throw std::invalid_argument(
      "tracking_origin must be standing, seated, or raw");
}

/**
 * @brief 验证 TF 坐标系参数适合作为未带前导斜杠的 frame ID。
 * @param value 待验证的 frame ID。
 * @param parameter_name 对应的 ROS 参数名称。
 * @throws std::invalid_argument frame ID 为空或带前导斜杠时抛出。
 */
void ValidateFrameId(const std::string &value,
                     const std::string &parameter_name) {
  if (value.empty()) {
    throw std::invalid_argument(parameter_name + " must not be empty");
  }
  if (value.front() == '/') {
    throw std::invalid_argument(parameter_name +
                                " must not start with a slash");
  }
}

/**
 * @brief 将公共位姿填充到 ROS Pose 消息。
 * @param source OpenVR 位姿转换后的公共位姿。
 * @param target 接收数值的 ROS 位姿消息，不能为空。
 */
void FillRosPose(const Pose &source, geometry_msgs::msg::Pose *target) {
  target->position.x = source.position.x;
  target->position.y = source.position.y;
  target->position.z = source.position.z;
  target->orientation.x = source.orientation.x;
  target->orientation.y = source.orientation.y;
  target->orientation.z = source.orientation.z;
  target->orientation.w = source.orientation.w;
}

/**
 * @brief 将内部 OpenVR 跟踪状态转换为公共 ROS 消息枚举。
 * @param state 内部跟踪状态。
 * @return fastumi_interfaces/TrackerStatus 对应的数值。
 */
std::uint8_t ToStatusMessageState(TrackingState state) noexcept {
  using Status = fastumi_interfaces::msg::TrackerStatus;
  switch (state) {
  case TrackingState::kUninitialized:
    return Status::TRACKING_UNINITIALIZED;
  case TrackingState::kCalibratingInProgress:
    return Status::TRACKING_CALIBRATING_IN_PROGRESS;
  case TrackingState::kCalibratingOutOfRange:
    return Status::TRACKING_CALIBRATING_OUT_OF_RANGE;
  case TrackingState::kRunningOk:
    return Status::TRACKING_RUNNING_OK;
  case TrackingState::kRunningOutOfRange:
    return Status::TRACKING_RUNNING_OUT_OF_RANGE;
  case TrackingState::kFallbackRotationOnly:
    return Status::TRACKING_FALLBACK_ROTATION_ONLY;
  case TrackingState::kUnknown:
  default:
    return Status::TRACKING_UNKNOWN;
  }
}

/**
 * @brief 发布指定 VIVE Tracker 的绝对位姿、首帧里程计、轨迹与多级 TF。
 */
class ViveTrackerNode : public rclcpp::Node {
public:
  /**
   * @brief 读取并验证参数，初始化 OpenVR 和全部发布接口。
   * @throws std::invalid_argument 参数无效时抛出。
   * @throws std::runtime_error OpenVR 初始化失败时抛出。
   */
  ViveTrackerNode() : Node("pose_publisher") {
    serial_ = declare_parameter<std::string>("serial", "LHR-B77A06A7");
    publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 30.0);
    /** 尚未解析的 OpenVR 跟踪原点参数。 */
    const std::string tracking_origin_text =
        declare_parameter<std::string>("tracking_origin", "standing");
    openvr_frame_ =
        declare_parameter<std::string>("openvr_frame", "steamvr_tracking");
    parent_frame_ =
        declare_parameter<std::string>("parent_frame", "steamvr_tracking_ros");
    odom_frame_ =
        declare_parameter<std::string>("odom_frame", "vive_tracker_odom");
    child_frame_ =
        declare_parameter<std::string>("child_frame", "vive_tracker");
    /** 尚未转换为容器长度类型的轨迹点上限。 */
    const std::int64_t max_path_points_parameter =
        declare_parameter<std::int64_t>("max_path_points", 3000);

    ValidateParameters(max_path_points_parameter);
    tracking_origin_ = ParseTrackingOrigin(tracking_origin_text);
    max_path_points_ = static_cast<std::size_t>(max_path_points_parameter);

    /** OpenVR 初始化失败时返回的可读说明。 */
    std::string openvr_error{};
    if (!pose_reader_.Initialize(&openvr_error)) {
      throw std::runtime_error("failed to initialize OpenVR: " + openvr_error);
    }

    pose_publisher_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        "pose", rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    status_publisher_ =
        create_publisher<fastumi_interfaces::msg::TrackerStatus>(
            "status", rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>(
        "odom", rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    /** 确保新启动的 RViz 能立即收到完整轨迹的发布策略。 */
    rclcpp::QoS path_qos(rclcpp::KeepLast(1));
    path_qos.reliable().transient_local();
    path_publisher_ = create_publisher<nav_msgs::msg::Path>("path", path_qos);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    static_tf_broadcaster_ =
        std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    PublishTrackingFrameTransform();

    path_message_.header.frame_id = parent_frame_;
    /** 根据发布频率计算得到的壁钟采样周期。 */
    const auto sample_period =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / publish_rate_hz_));
    sample_timer_ = create_wall_timer(
        sample_period, std::bind(&ViveTrackerNode::SampleAndPublish, this));

    RCLCPP_INFO(get_logger(),
                "Publishing Tracker %s at %.3f Hz in frame %s converted "
                "from %s; path limit is %zu points",
                serial_.c_str(), publish_rate_hz_, parent_frame_.c_str(),
                openvr_frame_.c_str(), max_path_points_);
  }

private:
  /**
   * @brief 验证全部启动参数之间的约束。
   * @param max_path_points_parameter 尚未转换的轨迹点上限。
   * @throws std::invalid_argument 任意参数无效时抛出。
   */
  void ValidateParameters(std::int64_t max_path_points_parameter) const {
    if (serial_.empty()) {
      throw std::invalid_argument("serial must not be empty");
    }
    if (!std::isfinite(publish_rate_hz_) || publish_rate_hz_ <= 0.0 ||
        publish_rate_hz_ > kMaximumPublishRateHz) {
      throw std::invalid_argument("publish_rate_hz must be in (0, 1000]");
    }
    ValidateFrameId(openvr_frame_, "openvr_frame");
    ValidateFrameId(parent_frame_, "parent_frame");
    ValidateFrameId(odom_frame_, "odom_frame");
    ValidateFrameId(child_frame_, "child_frame");
    if (openvr_frame_ == parent_frame_ || openvr_frame_ == odom_frame_ ||
        openvr_frame_ == child_frame_ || parent_frame_ == odom_frame_ ||
        parent_frame_ == child_frame_ || odom_frame_ == child_frame_) {
      throw std::invalid_argument(
          "openvr_frame, parent_frame, odom_frame, and child_frame must be "
          "different");
    }
    if (max_path_points_parameter <= 0) {
      throw std::invalid_argument("max_path_points must be greater than zero");
    }
    if (static_cast<std::uint64_t>(max_path_points_parameter) >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      throw std::invalid_argument("max_path_points is too large");
    }
  }

  /**
   * @brief 发布原始 OpenVR 全局坐标系到新 ROS 跟踪坐标系的静态 TF。
   */
  void PublishTrackingFrameTransform() {
    tracking_frame_transform_.header.stamp = now();
    tracking_frame_transform_.header.frame_id = openvr_frame_;
    tracking_frame_transform_.child_frame_id = parent_frame_;
    /** 新 ROS 跟踪坐标系在 OpenVR 全局坐标系中的固定方向。 */
    const Quaternion frame_orientation =
        GetRosTrackingFrameOrientationInOpenVr();
    tracking_frame_transform_.transform.rotation.x = frame_orientation.x;
    tracking_frame_transform_.transform.rotation.y = frame_orientation.y;
    tracking_frame_transform_.transform.rotation.z = frame_orientation.z;
    tracking_frame_transform_.transform.rotation.w = frame_orientation.w;
    static_tf_broadcaster_->sendTransform(tracking_frame_transform_);
  }

  /**
   * @brief 使用第一条有效位姿初始化里程计坐标系并发布静态 TF。
   * @param initial_pose Tracker 在 ROS 全局跟踪坐标系中的首帧位姿。
   * @param stamp 首帧采样时间戳。
   */
  void InitializeOdomFrame(const Pose &initial_pose,
                           const rclcpp::Time &stamp) {
    initial_pose_ = initial_pose;

    /** ROS 全局跟踪坐标系到首帧里程计坐标系的静态变换。 */
    geometry_msgs::msg::TransformStamped odom_frame_transform{};
    odom_frame_transform.header.stamp = stamp;
    odom_frame_transform.header.frame_id = parent_frame_;
    odom_frame_transform.child_frame_id = odom_frame_;
    odom_frame_transform.transform.translation.x = initial_pose.position.x;
    odom_frame_transform.transform.translation.y = initial_pose.position.y;
    odom_frame_transform.transform.translation.z = initial_pose.position.z;
    odom_frame_transform.transform.rotation.x = initial_pose.orientation.x;
    odom_frame_transform.transform.rotation.y = initial_pose.orientation.y;
    odom_frame_transform.transform.rotation.z = initial_pose.orientation.z;
    odom_frame_transform.transform.rotation.w = initial_pose.orientation.w;

    /** 确保晚加入的订阅者能够同时收到完整的两级静态 TF。 */
    const std::vector<geometry_msgs::msg::TransformStamped> static_transforms{
        tracking_frame_transform_, odom_frame_transform};
    static_tf_broadcaster_->sendTransform(static_transforms);
    RCLCPP_INFO(get_logger(),
                "Initialized odometry frame %s from the first valid pose",
                odom_frame_.c_str());
  }

  /**
   * @brief 读取目标 Tracker，并在位姿有效时发布 Pose、Odom、Path 和 TF。
   */
  void SampleAndPublish() {
    /** 当前 OpenVR 会话中全部 Generic Tracker 的采样。 */
    const std::vector<TrackerPoseSample> samples =
        pose_reader_.ReadPoses(tracking_origin_);
    /** 与配置序列号匹配的采样迭代器。 */
    const auto target_sample =
        std::find_if(samples.cbegin(), samples.cend(),
                     [this](const TrackerPoseSample &sample) {
                       return sample.serial_number == serial_;
                     });

    if (target_sample == samples.cend()) {
      /** 设备缺失状态使用当前系统时钟，确保录制端仍能观察到采样。 */
      fastumi_interfaces::msg::TrackerStatus status_message{};
      status_message.header.stamp = now();
      status_message.header.frame_id = parent_frame_;
      status_message.serial_number = serial_;
      status_message.device_connected = false;
      status_message.pose_valid = false;
      status_message.tracking_state =
          fastumi_interfaces::msg::TrackerStatus::TRACKING_UNKNOWN;
      status_publisher_->publish(status_message);
      RCLCPP_WARN_THROTTLE(get_logger(), steady_clock_, kWarningThrottleMs,
                           "Tracker %s was not found; waiting for the device",
                           serial_.c_str());
      return;
    }
    /** 位姿与状态消息共享一次 OpenVR 采样的 Unix 时间戳。 */
    const rclcpp::Time sample_stamp(target_sample->sample_time_unix_ns,
                                    RCL_SYSTEM_TIME);
    fastumi_interfaces::msg::TrackerStatus status_message{};
    status_message.header.stamp = sample_stamp;
    status_message.header.frame_id = parent_frame_;
    status_message.serial_number = serial_;
    status_message.device_connected = target_sample->device_connected;
    status_message.pose_valid = target_sample->pose_valid;
    status_message.tracking_state =
        ToStatusMessageState(target_sample->tracking_state);
    status_publisher_->publish(status_message);

    if (!target_sample->pose_valid) {
      /** 用于日志输出的稳定跟踪状态文本。 */
      const std::string tracking_state(
          TrackingStateToString(target_sample->tracking_state));
      RCLCPP_WARN_THROTTLE(
          get_logger(), steady_clock_, kWarningThrottleMs,
          "Tracker %s has no valid pose: connected=%s tracking=%s",
          serial_.c_str(), target_sample->device_connected ? "true" : "false",
          tracking_state.c_str());
      return;
    }

    /** 本次发布使用的当前位姿消息。 */
    geometry_msgs::msg::PoseStamped pose_message{};
    pose_message.header.stamp = sample_stamp;
    pose_message.header.frame_id = parent_frame_;
    /** 将位姿表达从原始 OpenVR 全局坐标系转换到新 ROS 跟踪坐标系。 */
    const Pose ros_pose = ConvertOpenVrPoseToRosPose(target_sample->pose);
    FillRosPose(ros_pose, &pose_message.pose);

    if (!initial_pose_.has_value()) {
      InitializeOdomFrame(ros_pose, pose_message.header.stamp);
    }
    /** 当前 Tracker 相对于首帧里程计坐标系的位姿。 */
    const Pose odom_pose = CalculateRelativePose(*initial_pose_, ros_pose);
    /** 本次发布的首帧归零里程计消息。 */
    nav_msgs::msg::Odometry odom_message{};
    odom_message.header.stamp = pose_message.header.stamp;
    odom_message.header.frame_id = odom_frame_;
    odom_message.child_frame_id = child_frame_;
    FillRosPose(odom_pose, &odom_message.pose.pose);

    path_message_.header.stamp = pose_message.header.stamp;
    AppendPoseToBoundedPath(pose_message, max_path_points_, &path_message_);

    /** 与里程计位姿具有相同时间戳和坐标数据的动态 TF。 */
    geometry_msgs::msg::TransformStamped transform_message{};
    transform_message.header = odom_message.header;
    transform_message.child_frame_id = child_frame_;
    transform_message.transform.translation.x = odom_pose.position.x;
    transform_message.transform.translation.y = odom_pose.position.y;
    transform_message.transform.translation.z = odom_pose.position.z;
    transform_message.transform.rotation = odom_message.pose.pose.orientation;

    pose_publisher_->publish(pose_message);
    odom_publisher_->publish(odom_message);
    path_publisher_->publish(path_message_);
    tf_broadcaster_->sendTransform(transform_message);
  }

  /** 待读取的 Tracker 序列号。 */
  std::string serial_{};
  /** OpenVR 位姿采样和 ROS 发布频率，单位为 Hz。 */
  double publish_rate_hz_{30.0};
  /** SteamVR 查询使用的跟踪原点。 */
  TrackingOrigin tracking_origin_{TrackingOrigin::kStanding};
  /** OpenVR 返回位姿时使用的原始全局坐标系。 */
  std::string openvr_frame_{};
  /** Pose、Path 和静态里程计原点使用的轴向重排父坐标系。 */
  std::string parent_frame_{};
  /** 第一条有效位姿定义的固定里程计坐标系。 */
  std::string odom_frame_{};
  /** TF 和 Odometry 使用的 Tracker 子坐标系。 */
  std::string child_frame_{};
  /** 轨迹允许保留的最大点数。 */
  std::size_t max_path_points_{3000};
  /** 管理 OpenVR 会话并读取 Tracker 位姿的对象。 */
  TrackerPoseReader pose_reader_{};
  /** 本次节点生命周期内用于定义里程计原点的首帧有效位姿。 */
  std::optional<Pose> initial_pose_{};
  /** 当前已经积累的有限长度轨迹。 */
  nav_msgs::msg::Path path_message_{};
  /** 当前绝对位姿发布器。 */
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
      pose_publisher_{};
  /** 每次 OpenVR 采样的连接和跟踪质量发布器。 */
  rclcpp::Publisher<fastumi_interfaces::msg::TrackerStatus>::SharedPtr
      status_publisher_{};
  /** 首帧归零里程计发布器。 */
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_{};
  /** 历史轨迹发布器。 */
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_{};
  /** 当前 Tracker 动态坐标变换发布器。 */
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_{};
  /** 原始 OpenVR 全局坐标系到新 ROS 跟踪坐标系的静态变换发布器。 */
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_{};
  /** 需要与首帧里程计静态变换一起重发的全局轴变换。 */
  geometry_msgs::msg::TransformStamped tracking_frame_transform_{};
  /** 周期执行 OpenVR 采样的壁钟定时器。 */
  rclcpp::TimerBase::SharedPtr sample_timer_{};
  /** 限频日志使用、不受 ROS 时间配置影响的稳定时钟。 */
  rclcpp::Clock steady_clock_{RCL_STEADY_TIME};
};

} // namespace
} // namespace vive_tracker

/**
 * @brief ROS 2 VIVE Tracker 位姿发布节点入口。
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @return 正常退出返回 EXIT_SUCCESS，初始化失败返回 EXIT_FAILURE。
 */
int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  try {
    /** 节点实例在 ROS 关闭前保持 OpenVR 会话有效。 */
    const auto node = std::make_shared<vive_tracker::ViveTrackerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception &exception) {
    RCLCPP_FATAL(rclcpp::get_logger("vive_tracker_node"), "%s",
                 exception.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}
