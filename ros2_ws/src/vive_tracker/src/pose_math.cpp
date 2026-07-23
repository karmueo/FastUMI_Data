/**
 * @file pose_math.cpp
 * @brief 实现 OpenVR 位姿解析和轴向重排后的 ROS 跟踪坐标转换。
 */

#include "vive_tracker/pose_math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <openvr.h>

namespace vive_tracker {
namespace {

/** 旋转矩阵的固定维度。 */
constexpr std::size_t kRotationDimension = 3;
/** 三维旋转矩阵。 */
using RotationMatrix =
    std::array<std::array<double, kRotationDimension>, kRotationDimension>;
/** OpenVR 全局坐标分量到 ROS 跟踪坐标分量的基变换矩阵。 */
constexpr RotationMatrix kOpenVrToRosBasis{
    std::array<double, kRotationDimension>{0.0, 0.0, 1.0},
    std::array<double, kRotationDimension>{1.0, 0.0, 0.0},
    std::array<double, kRotationDimension>{0.0, 1.0, 0.0}};

/**
 * @brief 将四元数归一化，并为退化输入返回单位四元数。
 * @param quaternion 待归一化的四元数。
 * @return 归一化后的四元数。
 */
Quaternion NormalizeQuaternion(const Quaternion &quaternion) noexcept {
  /** 四元数的欧几里得范数。 */
  const double norm =
      std::sqrt(quaternion.x * quaternion.x + quaternion.y * quaternion.y +
                quaternion.z * quaternion.z + quaternion.w * quaternion.w);
  /** 避免除以零的最小范数。 */
  constexpr double kMinimumNorm = 1.0e-12;
  if (norm <= kMinimumNorm) {
    return Quaternion{};
  }
  return Quaternion{quaternion.x / norm, quaternion.y / norm,
                    quaternion.z / norm, quaternion.w / norm};
}

/**
 * @brief 将三维旋转矩阵转换为单位四元数。
 * @param matrix 待转换的正交旋转矩阵。
 * @return 采用 xyzw 顺序的单位四元数。
 */
Quaternion RotationMatrixToQuaternion(const RotationMatrix &matrix) noexcept {
  /** 旋转矩阵对角线之和。 */
  const double trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
  /** 尚未归一化的四元数。 */
  Quaternion quaternion{};

  if (trace > 0.0) {
    /** 正迹分支使用的缩放系数。 */
    const double scale = 2.0 * std::sqrt(std::max(0.0, trace + 1.0));
    quaternion.w = 0.25 * scale;
    quaternion.x = (matrix[2][1] - matrix[1][2]) / scale;
    quaternion.y = (matrix[0][2] - matrix[2][0]) / scale;
    quaternion.z = (matrix[1][0] - matrix[0][1]) / scale;
  } else if (matrix[0][0] > matrix[1][1] && matrix[0][0] > matrix[2][2]) {
    /** X 分量占优分支使用的缩放系数。 */
    const double scale =
        2.0 * std::sqrt(std::max(0.0, 1.0 + matrix[0][0] - matrix[1][1] -
                                          matrix[2][2]));
    quaternion.w = (matrix[2][1] - matrix[1][2]) / scale;
    quaternion.x = 0.25 * scale;
    quaternion.y = (matrix[0][1] + matrix[1][0]) / scale;
    quaternion.z = (matrix[0][2] + matrix[2][0]) / scale;
  } else if (matrix[1][1] > matrix[2][2]) {
    /** Y 分量占优分支使用的缩放系数。 */
    const double scale =
        2.0 * std::sqrt(std::max(0.0, 1.0 - matrix[0][0] + matrix[1][1] -
                                          matrix[2][2]));
    quaternion.w = (matrix[0][2] - matrix[2][0]) / scale;
    quaternion.x = (matrix[0][1] + matrix[1][0]) / scale;
    quaternion.y = 0.25 * scale;
    quaternion.z = (matrix[1][2] + matrix[2][1]) / scale;
  } else {
    /** Z 分量占优分支使用的缩放系数。 */
    const double scale =
        2.0 * std::sqrt(std::max(0.0, 1.0 - matrix[0][0] - matrix[1][1] +
                                          matrix[2][2]));
    quaternion.w = (matrix[1][0] - matrix[0][1]) / scale;
    quaternion.x = (matrix[0][2] + matrix[2][0]) / scale;
    quaternion.y = (matrix[1][2] + matrix[2][1]) / scale;
    quaternion.z = 0.25 * scale;
  }

  return NormalizeQuaternion(quaternion);
}

/**
 * @brief 将单位四元数转换为三维旋转矩阵。
 * @param quaternion 待转换的四元数。
 * @return 对应的正交旋转矩阵。
 */
RotationMatrix
QuaternionToRotationMatrix(const Quaternion &quaternion) noexcept {
  /** 归一化后的输入四元数。 */
  const Quaternion rotation = NormalizeQuaternion(quaternion);
  /** 四元数各分量的常用乘积。 */
  const double xx = rotation.x * rotation.x;
  const double xy = rotation.x * rotation.y;
  const double xz = rotation.x * rotation.z;
  const double xw = rotation.x * rotation.w;
  const double yy = rotation.y * rotation.y;
  const double yz = rotation.y * rotation.z;
  const double yw = rotation.y * rotation.w;
  const double zz = rotation.z * rotation.z;
  const double zw = rotation.z * rotation.w;

  return RotationMatrix{
      std::array<double, kRotationDimension>{1.0 - 2.0 * (yy + zz),
                                             2.0 * (xy - zw), 2.0 * (xz + yw)},
      std::array<double, kRotationDimension>{
          2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)},
      std::array<double, kRotationDimension>{2.0 * (xz - yw), 2.0 * (yz + xw),
                                             1.0 - 2.0 * (xx + yy)}};
}

/**
 * @brief 将旋转矩阵的父坐标基从 OpenVR 转换为 ROS 跟踪坐标基。
 * @param openvr_rotation OpenVR 全局坐标系到原始 Tracker 子坐标系的旋转。
 * @return ROS 跟踪坐标系到原始 Tracker 子坐标系的旋转。
 */
RotationMatrix
ConvertOpenVrRotationToRos(const RotationMatrix &openvr_rotation) noexcept {
  /** 基变换后的旋转矩阵。 */
  RotationMatrix ros_rotation{};
  for (std::size_t row = 0; row < kRotationDimension; ++row) {
    for (std::size_t column = 0; column < kRotationDimension; ++column) {
      for (std::size_t openvr_row = 0; openvr_row < kRotationDimension;
           ++openvr_row) {
        ros_rotation[row][column] += kOpenVrToRosBasis[row][openvr_row] *
                                     openvr_rotation[openvr_row][column];
      }
    }
  }
  return ros_rotation;
}

} // namespace

/**
 * @brief 将 OpenVR 的 3×4 刚体变换矩阵转换为原始 OpenVR 位姿。
 * @param matrix OpenVR 返回的设备到绝对跟踪坐标系变换矩阵。
 * @return 位置单位为米、四元数顺序为 xyzw 的 OpenVR 位姿。
 */
Pose ConvertOpenVrMatrixToPose(const vr::HmdMatrix34_t &matrix) noexcept {
  /** 转换后的 OpenVR 位姿。 */
  Pose pose{};
  pose.position.x = static_cast<double>(matrix.m[0][3]);
  pose.position.y = static_cast<double>(matrix.m[1][3]);
  pose.position.z = static_cast<double>(matrix.m[2][3]);

  /** OpenVR 矩阵中的三维旋转部分。 */
  RotationMatrix rotation{};
  for (std::size_t row = 0; row < kRotationDimension; ++row) {
    for (std::size_t column = 0; column < kRotationDimension; ++column) {
      rotation[row][column] = static_cast<double>(matrix.m[row][column]);
    }
  }
  pose.orientation = RotationMatrixToQuaternion(rotation);
  return pose;
}

/**
 * @brief 将 OpenVR 位姿转换到轴向重排后的 ROS 跟踪坐标系。
 * @param openvr_pose 使用 VIVE Tracker 3.0 原始坐标轴的 OpenVR 位姿。
 * @return 将父坐标系原始正 Z、正 X、正 Y 依次映射到 ROS 正 X、正 Y、正 Z
 * 的位姿。
 */
Pose ConvertOpenVrPoseToRosPose(const Pose &openvr_pose) noexcept {
  /** 在 ROS 跟踪坐标系中表达的 Tracker 位姿。 */
  Pose ros_pose{};
  ros_pose.position.x = openvr_pose.position.z;
  ros_pose.position.y = openvr_pose.position.x;
  ros_pose.position.z = openvr_pose.position.y;

  /** 原始 Tracker 子坐标系在 OpenVR 全局坐标系中的旋转。 */
  const RotationMatrix openvr_rotation =
      QuaternionToRotationMatrix(openvr_pose.orientation);
  /** 原始 Tracker 子坐标系在 ROS 跟踪坐标系中的旋转。 */
  const RotationMatrix ros_rotation =
      ConvertOpenVrRotationToRos(openvr_rotation);
  ros_pose.orientation = RotationMatrixToQuaternion(ros_rotation);
  return ros_pose;
}

/**
 * @brief 获取轴向重排后的 ROS 跟踪坐标系在 OpenVR 全局坐标系中的方向。
 * @return 用于发布 OpenVR 全局坐标系到 ROS 跟踪坐标系静态 TF 的单位四元数。
 */
Quaternion GetRosTrackingFrameOrientationInOpenVr() noexcept {
  /** ROS 跟踪坐标系到 OpenVR 全局坐标系的旋转，即分量变换矩阵的转置。 */
  RotationMatrix openvr_from_ros{};
  for (std::size_t row = 0; row < kRotationDimension; ++row) {
    for (std::size_t column = 0; column < kRotationDimension; ++column) {
      openvr_from_ros[row][column] = kOpenVrToRosBasis[column][row];
    }
  }
  return RotationMatrixToQuaternion(openvr_from_ros);
}

} // namespace vive_tracker
