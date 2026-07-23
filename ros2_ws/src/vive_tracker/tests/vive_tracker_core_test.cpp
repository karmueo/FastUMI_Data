/**
 * @file vive_tracker_core_test.cpp
 * @brief 验证 OpenVR 位姿、多级坐标转换和 ROS 2 有界轨迹行为。
 */

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <gtest/gtest.h>
#include <openvr.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>

#include "vive_tracker/path_history.hpp"
#include "vive_tracker/pose_math.hpp"
#include "vive_tracker/tracker_pose_types.hpp"

namespace {

/** 浮点比较使用的绝对误差。 */
constexpr double kTolerance = 1.0e-6;
/** 90 度旋转四元数的非零分量。 */
constexpr double kSqrtHalf = 0.70710678118654752440;

/**
 * @brief 创建单位旋转和零平移矩阵。
 * @return OpenVR 3×4 单位变换矩阵。
 */
vr::HmdMatrix34_t MakeIdentityMatrix() {
  /** 单位变换矩阵。 */
  vr::HmdMatrix34_t matrix{};
  matrix.m[0][0] = 1.0F;
  matrix.m[1][1] = 1.0F;
  matrix.m[2][2] = 1.0F;
  return matrix;
}

/**
 * @brief 计算四元数范数。
 * @param quaternion 待检查的四元数。
 * @return 四元数的欧几里得范数。
 */
double QuaternionNorm(const vive_tracker::Quaternion &quaternion) {
  return std::sqrt(quaternion.x * quaternion.x + quaternion.y * quaternion.y +
                   quaternion.z * quaternion.z + quaternion.w * quaternion.w);
}

/**
 * @brief 计算两个四元数对应旋转的复合结果。
 * @param left 左侧旋转四元数。
 * @param right 右侧旋转四元数。
 * @return 先应用 right、再应用 left 的组合四元数。
 */
vive_tracker::Quaternion
MultiplyQuaternions(const vive_tracker::Quaternion &left,
                    const vive_tracker::Quaternion &right) {
  return vive_tracker::Quaternion{
      left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
      left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
      left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
      left.w * right.w - left.x * right.x - left.y * right.y -
          left.z * right.z};
}

/**
 * @brief 计算单位四元数的逆旋转。
 * @param quaternion 待求逆的单位四元数。
 * @return 输入四元数的共轭。
 */
vive_tracker::Quaternion
ConjugateQuaternion(const vive_tracker::Quaternion &quaternion) {
  return vive_tracker::Quaternion{-quaternion.x, -quaternion.y, -quaternion.z,
                                  quaternion.w};
}

/**
 * @brief 验证单位旋转和平移提取。
 */
TEST(ViveTrackerPoseMath, ConvertsIdentityAndTranslation) {
  /** 带测试平移的单位旋转矩阵。 */
  vr::HmdMatrix34_t matrix = MakeIdentityMatrix();
  matrix.m[0][3] = 1.25F;
  matrix.m[1][3] = -2.5F;
  matrix.m[2][3] = 0.125F;

  /** 转换后的测试位姿。 */
  const vive_tracker::Pose pose =
      vive_tracker::ConvertOpenVrMatrixToPose(matrix);
  EXPECT_NEAR(pose.position.x, 1.25, kTolerance);
  EXPECT_NEAR(pose.position.y, -2.5, kTolerance);
  EXPECT_NEAR(pose.position.z, 0.125, kTolerance);
  EXPECT_NEAR(pose.orientation.x, 0.0, kTolerance);
  EXPECT_NEAR(pose.orientation.y, 0.0, kTolerance);
  EXPECT_NEAR(pose.orientation.z, 0.0, kTolerance);
  EXPECT_NEAR(pose.orientation.w, 1.0, kTolerance);
}

/**
 * @brief 验证绕 Y 轴 90 度旋转和四元数归一化。
 */
TEST(ViveTrackerPoseMath, ConvertsRotationAndNormalizesQuaternion) {
  /** 绕 Y 轴旋转 90 度的矩阵。 */
  vr::HmdMatrix34_t matrix{};
  matrix.m[0][2] = 1.0F;
  matrix.m[1][1] = 1.0F;
  matrix.m[2][0] = -1.0F;

  /** 转换后的测试位姿。 */
  const vive_tracker::Pose pose =
      vive_tracker::ConvertOpenVrMatrixToPose(matrix);
  EXPECT_NEAR(pose.orientation.x, 0.0, kTolerance);
  EXPECT_NEAR(std::abs(pose.orientation.y), kSqrtHalf, kTolerance);
  EXPECT_NEAR(pose.orientation.z, 0.0, kTolerance);
  EXPECT_NEAR(std::abs(pose.orientation.w), kSqrtHalf, kTolerance);
  EXPECT_NEAR(QuaternionNorm(pose.orientation), 1.0, kTolerance);
}

/**
 * @brief 验证 OpenVR 位置分量和基准方向映射到新 ROS 跟踪坐标系。
 */
TEST(ViveTrackerPoseMath, ConvertsOpenVrPositionToRosTrackingFrame) {
  /** 包含三个不同位置分量且旋转为单位四元数的 OpenVR 位姿。 */
  const vive_tracker::Pose openvr_pose{vive_tracker::Vector3{1.0, 2.0, 3.0},
                                       vive_tracker::Quaternion{}};

  /** 在新 ROS 跟踪坐标系中表达的位姿。 */
  const vive_tracker::Pose ros_pose =
      vive_tracker::ConvertOpenVrPoseToRosPose(openvr_pose);
  EXPECT_NEAR(ros_pose.position.x, 3.0, kTolerance);
  EXPECT_NEAR(ros_pose.position.y, 1.0, kTolerance);
  EXPECT_NEAR(ros_pose.position.z, 2.0, kTolerance);
  EXPECT_NEAR(ros_pose.orientation.x, 0.5, kTolerance);
  EXPECT_NEAR(ros_pose.orientation.y, 0.5, kTolerance);
  EXPECT_NEAR(ros_pose.orientation.z, 0.5, kTolerance);
  EXPECT_NEAR(ros_pose.orientation.w, 0.5, kTolerance);
}

/**
 * @brief 验证原始正 Y 旋转轴在新全局坐标系中对应正 Z。
 */
TEST(ViveTrackerPoseMath, MapsOpenVrPositiveYRotationToRosPositiveZ) {
  /** 原始单位方向转换后的基准位姿。 */
  const vive_tracker::Pose baseline_pose =
      vive_tracker::ConvertOpenVrPoseToRosPose(vive_tracker::Pose{});
  /** 绕 OpenVR 正 Y 轴旋转 90 度的位姿。 */
  const vive_tracker::Pose openvr_pose{
      vive_tracker::Vector3{},
      vive_tracker::Quaternion{0.0, kSqrtHalf, 0.0, kSqrtHalf}};
  /** 在新全局坐标系中表达的旋转后位姿。 */
  const vive_tracker::Pose ros_pose =
      vive_tracker::ConvertOpenVrPoseToRosPose(openvr_pose);
  /** 相对基准方向的旋转增量。 */
  const vive_tracker::Quaternion delta_rotation = MultiplyQuaternions(
      ros_pose.orientation, ConjugateQuaternion(baseline_pose.orientation));

  EXPECT_NEAR(delta_rotation.x, 0.0, kTolerance);
  EXPECT_NEAR(delta_rotation.y, 0.0, kTolerance);
  EXPECT_NEAR(std::abs(delta_rotation.z), kSqrtHalf, kTolerance);
  EXPECT_NEAR(std::abs(delta_rotation.w), kSqrtHalf, kTolerance);
  EXPECT_NEAR(QuaternionNorm(delta_rotation), 1.0, kTolerance);
}

/**
 * @brief 验证原始正 Z 旋转轴在新全局坐标系中对应正 X。
 */
TEST(ViveTrackerPoseMath, MapsOpenVrPositiveZRotationToRosPositiveX) {
  /** 原始单位方向转换后的基准位姿。 */
  const vive_tracker::Pose baseline_pose =
      vive_tracker::ConvertOpenVrPoseToRosPose(vive_tracker::Pose{});
  /** 绕 OpenVR 正 Z 轴旋转 90 度的位姿。 */
  const vive_tracker::Pose openvr_pose{
      vive_tracker::Vector3{},
      vive_tracker::Quaternion{0.0, 0.0, kSqrtHalf, kSqrtHalf}};
  /** 在新全局坐标系中表达的旋转后位姿。 */
  const vive_tracker::Pose ros_pose =
      vive_tracker::ConvertOpenVrPoseToRosPose(openvr_pose);
  /** 相对基准方向的旋转增量。 */
  const vive_tracker::Quaternion delta_rotation = MultiplyQuaternions(
      ros_pose.orientation, ConjugateQuaternion(baseline_pose.orientation));

  EXPECT_NEAR(std::abs(delta_rotation.x), kSqrtHalf, kTolerance);
  EXPECT_NEAR(delta_rotation.y, 0.0, kTolerance);
  EXPECT_NEAR(delta_rotation.z, 0.0, kTolerance);
  EXPECT_NEAR(std::abs(delta_rotation.w), kSqrtHalf, kTolerance);
  EXPECT_NEAR(QuaternionNorm(delta_rotation), 1.0, kTolerance);
}

/**
 * @brief 验证静态全局变换和转换后的动态位姿能够还原原始方向。
 */
TEST(ViveTrackerPoseMath, ComposesStaticAndDynamicFramesToOriginalPose) {
  /** 新 ROS 跟踪坐标系在 OpenVR 全局坐标系中的静态方向。 */
  const vive_tracker::Quaternion static_orientation =
      vive_tracker::GetRosTrackingFrameOrientationInOpenVr();
  /** 原始单位 Tracker 位姿在新 ROS 跟踪坐标系中的方向。 */
  const vive_tracker::Quaternion dynamic_orientation =
      vive_tracker::ConvertOpenVrPoseToRosPose(vive_tracker::Pose{})
          .orientation;
  /** 两级 TF 组合后得到的 OpenVR 全局到 Tracker 方向。 */
  const vive_tracker::Quaternion composed_orientation =
      MultiplyQuaternions(static_orientation, dynamic_orientation);

  EXPECT_NEAR(composed_orientation.x, 0.0, kTolerance);
  EXPECT_NEAR(composed_orientation.y, 0.0, kTolerance);
  EXPECT_NEAR(composed_orientation.z, 0.0, kTolerance);
  EXPECT_NEAR(std::abs(composed_orientation.w), 1.0, kTolerance);
  EXPECT_NEAR(QuaternionNorm(composed_orientation), 1.0, kTolerance);
}

/**
 * @brief 验证轨迹只保留最近 3000 个采样。
 */
TEST(ViveTrackerPathHistory, KeepsNewestThreeThousandPoints) {
  /** 测试使用的轨迹消息。 */
  nav_msgs::msg::Path path{};
  /** 计划追加的总点数。 */
  constexpr std::size_t kTotalPoints = 3005;
  /** 轨迹保留点数上限。 */
  constexpr std::size_t kMaximumPoints = 3000;

  for (std::size_t point_index = 0; point_index < kTotalPoints; ++point_index) {
    /** 使用 X 位置记录原始序号的测试位姿。 */
    geometry_msgs::msg::PoseStamped pose{};
    pose.pose.position.x = static_cast<double>(point_index);
    vive_tracker::AppendPoseToBoundedPath(pose, kMaximumPoints, &path);
  }

  ASSERT_EQ(path.poses.size(), kMaximumPoints);
  EXPECT_DOUBLE_EQ(path.poses.front().pose.position.x, 5.0);
  EXPECT_DOUBLE_EQ(path.poses.back().pose.position.x, 3004.0);
}

/**
 * @brief 验证轨迹工具拒绝零长度上限。
 */
TEST(ViveTrackerPathHistory, RejectsZeroPointLimit) {
  /** 测试使用的轨迹消息。 */
  nav_msgs::msg::Path path{};
  /** 测试使用的位姿消息。 */
  geometry_msgs::msg::PoseStamped pose{};
  EXPECT_THROW(vive_tracker::AppendPoseToBoundedPath(pose, 0, &path),
               std::invalid_argument);
}

} // namespace
