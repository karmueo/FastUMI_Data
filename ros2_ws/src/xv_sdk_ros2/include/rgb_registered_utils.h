/**
 * @file rgb_registered_utils.h
 * @brief 提供 RGB/ToF 配准、公共虚拟相机和时间同步基础工具函数。
 */

#ifndef RGB_REGISTERED_UTILS_H_
#define RGB_REGISTERED_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <xv-sdk.h>

namespace xv_ros2::rgb_fisheye {
  struct KalibrCam0Calibration;
} // namespace xv_ros2::rgb_fisheye

namespace xv_ros2 {
  namespace rgb_registered {

/**
 * @brief RGB 与 ToF 帧允许匹配的最大时间差，单位为秒。
 */
    constexpr double kMaxTimestampDiffSeconds = 0.033;

/**
 * @brief RGB registered 最近邻时间戳配对结果。
 */
    struct TimestampPairMatch
    {
  /** ToF 时间戳数组中的索引。 */
      std::size_t depth_index = 0;
  /** RGB 时间戳数组中的索引。 */
      std::size_t color_index = 0;
  /** 两帧时间差，单位为秒。 */
      double diff_seconds = 0.0;
    };

/**
 * @brief RGB、ToF 深度与 ToF IR 最近邻时间戳匹配结果。
 */
    struct TimestampTripleMatch
    {
  /** ToF 深度缓存中的索引。 */
      std::size_t depth_index = 0;
  /** RGB 缓存中的索引。 */
      std::size_t color_index = 0;
  /** ToF IR 缓存中的索引。 */
      std::size_t ir_index = 0;
  /** RGB 与深度的时间差，单位为秒。 */
      double color_diff_seconds = 0.0;
  /** IR 与深度的时间差，单位为秒。 */
      double ir_diff_seconds = 0.0;
    };

/**
 * @brief 投影到 Kalibr RGB 网格的 ToF 深度与 IR 数据。
 */
    struct RegisteredTofImages
    {
  /** 输出图像宽度。 */
      std::size_t width = 0;
  /** 输出图像高度。 */
      std::size_t height = 0;
  /** RGB 相机坐标系 Z 深度，单位为米，0 表示无效。 */
      std::vector < float > depth_meters;
  /** 与深度同源的 ToF IR 强度，0 表示无对应投影。 */
      std::vector < std::uint16_t > ir;
    };

/**
 * @brief 保存公共虚拟网格或兼容 ToF 原生网格的 RGB、深度与 IR 数据。
 */
    struct RgbToTofImages
    {
  /** 输出图像宽度。 */
      std::size_t width = 0;
  /** 输出图像高度。 */
      std::size_t height = 0;
  /** 行优先 RGB8 数据。 */
      std::vector < std::uint8_t > rgb;
  /** 米制深度，0 表示无效。 */
      std::vector < float > depth_meters;
  /** 与深度同源的 IR 强度。 */
      std::vector < std::uint16_t > ir;
    };

/**
 * @brief RGBD 三路输出使用的 ToF 坐标系公共虚拟针孔相机及预映射。
 */
    struct VirtualRgbdModel
    {
  /** 虚拟相机输出宽度。 */
      std::size_t width = 0;
  /** 虚拟相机输出高度。 */
      std::size_t height = 0;
  /** 虚拟针孔相机 x 方向焦距。 */
      double fx = 0.0;
  /** 虚拟针孔相机 y 方向焦距。 */
      double fy = 0.0;
  /** 虚拟针孔相机 x 方向主点。 */
      double cx = 0.0;
  /** 虚拟针孔相机 y 方向主点。 */
      double cy = 0.0;
  /** 公共视场归一化最小 x。 */
      double min_x = 0.0;
  /** 公共视场归一化最大 x。 */
      double max_x = 0.0;
  /** 公共视场归一化最小 y。 */
      double min_y = 0.0;
  /** 公共视场归一化最大 y。 */
      double max_y = 0.0;
  /** 每个虚拟像素对应的 ToF 原始图像行优先索引。 */
      std::vector < std::size_t > tof_source_indices;
  /** 每个虚拟像素在 ToF 坐标系下、z=1 的射线。 */
      std::vector < cv::Point3f > virtual_rays;
  /** 深度无效时使用的 Kalibr 校正 RGB 采样位置。 */
      std::vector < cv::Point2f > fallback_rgb_pixels;
  /** Kalibr 校正 RGB 图中可安全双线性采样的像素掩码。 */
      std::vector < std::uint8_t > rgb_valid_mask;
  /** RGB 有效掩码宽度。 */
      std::size_t rgb_width = 0;
  /** RGB 有效掩码高度。 */
      std::size_t rgb_height = 0;
    };

/**
 * @brief 记录 RGB registered 深度投影过程中的诊断计数。
 */
    struct ProjectionDiagnostics
    {
  /** 遍历过的 ToF 像素数量。 */
      std::size_t total_points = 0;
  /** 深度值有效的 ToF 像素数量。 */
      std::size_t valid_depth_points = 0;
  /** 成功投影到 Kalibr cam0 图像平面的 ToF 点数量。 */
      std::size_t projected_points = 0;
  /** 反投影、外参变换或 Kalibr 投影失败的点数量。 */
      std::size_t projection_failed_points = 0;
  /** 投影后落在 RGB registered 图像范围外的点数量。 */
      std::size_t out_of_bounds_points = 0;
  /** 写入输出深度图的像素数量。 */
      std::size_t stored_pixels = 0;
  /** 写入时覆盖同一像素已有深度的次数。 */
      std::size_t overwritten_pixels = 0;

  /**
   * @brief 记录一个无效深度点。
   */
      void recordInvalidDepth();

  /**
   * @brief 记录一个深度有效但投影失败的点。
   */
      void recordProjectionFailure();

  /**
   * @brief 记录一个成功投影但落在图像范围外的点。
   */
      void recordOutOfBoundsProjection();

  /**
   * @brief 记录一个成功写入输出深度图的点。
   * @param overwrote_existing_depth 是否覆盖了同一像素已有深度。
   */
      void recordStoredDepth(bool overwrote_existing_depth);

  /**
   * @brief 记录一个投影到已有深度像素但未写入的点。
   */
      void recordDiscardedOverlappedDepth();
    };

/**
 * @brief 将 ToF 深度图指定像素转换为米制深度。
 * @param image SDK ToF 深度图，支持 Depth_16 与 Depth_32。
 * @param index 以行优先顺序表示的像素索引。
 * @return 有效深度的米制值；无效、越界或不支持的深度返回 0.0。
 */
    float depthValueMeters(const xv::DepthImage & image, std::size_t index);

/**
 * @brief 将 ToF 深度图指定像素转换为米制深度，并按最大有效深度过滤。
 * @param image SDK ToF 深度图，支持 Depth_16 与 Depth_32。
 * @param index 以行优先顺序表示的像素索引。
 * @param max_valid_depth_meters 最大有效深度，单位为米；大于等于该值返回
 * 0.0。
 * @return 有效且小于最大深度的米制值；无效、越界、不支持或超限返回 0.0。
 */
    float depthValueMeters(
      const xv::DepthImage & image, std::size_t index,
      double max_valid_depth_meters);

/**
 * @brief 判断两帧时间戳是否满足 RGB/ToF 近邻同步阈值。
 * @param lhs 第一帧 host 时间戳，单位为秒。
 * @param rhs 第二帧 host 时间戳，单位为秒。
 * @return 时间差小于等于 33ms 时返回 true。
 */
    bool isTimestampMatch(double lhs, double rhs);

/**
 * @brief 从 ToF 与 RGB 两侧缓存中查找时间差最小的配对。
 * @param depth_timestamps ToF 时间戳列表，单位为秒。
 * @param color_timestamps RGB 时间戳列表，单位为秒。
 * @param tolerance_sec 最大允许时间差，单位为秒。
 * @return 存在满足阈值的配对时返回索引和时间差，否则返回空。
 */
    std::optional < TimestampPairMatch > findNearestTimestampPair(
      const std::vector < double > &depth_timestamps,
      const std::vector < double > &color_timestamps,
      double tolerance_sec);

/**
 * @brief 从两侧缓存中查找已被未来帧确认的最近邻时间戳配对。
 * @param depth_timestamps ToF 时间戳列表，单位为秒。
 * @param color_timestamps RGB 时间戳列表，单位为秒。
 * @param tolerance_sec 最大允许时间差，单位为秒。
 * @return 存在稳定配对时返回索引和时间差，否则返回空。
 */
    std::optional < TimestampPairMatch > findStableNearestTimestampPair(
      const std::vector < double > &depth_timestamps,
      const std::vector < double > &color_timestamps,
      double tolerance_sec);

/**
 * @brief 从 RGB、ToF 深度和 ToF IR 缓存中查找稳定的最近邻三帧组。
 * @param depth_timestamps ToF 深度时间戳列表，单位为秒。
 * @param color_timestamps RGB 时间戳列表，单位为秒。
 * @param ir_timestamps ToF IR 时间戳列表，单位为秒。
 * @param tolerance_sec 任意两路帧的最大允许时间差，单位为秒。
 * @return 存在已由未来帧确认的三帧组时返回索引和时间差，否则返回空。
 */
    std::optional < TimestampTripleMatch > findStableNearestTimestampTriple(
      const std::vector < double > &depth_timestamps,
      const std::vector < double > &color_timestamps,
      const std::vector < double > &ir_timestamps,
      double tolerance_sec);

/**
 * @brief 将 ToF 相机点按 SDK ToF 外参和 Kalibr T_cam_imu 投影到 cam0 校正图。
 * @param calibration Kalibr cam0 标定，包含 T_cam_imu 和无畸变针孔内参。
 * @param tof_pose SDK ToF 到 SDK IMU 坐标系的外参。
 * @param tof_point ToF 相机坐标系下的三维点，单位为米。
 * @param pixel 输出 Kalibr 校正 RGB 图像像素坐标；不可为 nullptr。
 * @param kalibr_depth_meters 输出 Kalibr cam0 坐标下的 z 深度，单位为米；可为
 * nullptr。
 * @return 投影成功且像素位于图像范围内时返回 true。
 */
    bool projectTofPointToKalibrUndistortedPixel(
      const xv_ros2::rgb_fisheye::KalibrCam0Calibration & calibration,
      const xv::Transform & tof_pose, const xv::Vector3d & tof_point,
      cv::Point2d *pixel, double *kalibr_depth_meters = nullptr);

/**
 * @brief 将 ToF 深度及可选 IR 投影到 Kalibr 校正 RGB 网格。
 * @param calibration Kalibr cam0 标定。
 * @param tof_calibration SDK ToF 内外参。
 * @param depth_image ToF 深度图，支持 Depth_16 与 Depth_32。
 * @param ir_image 同分辨率 ToF IR 图；传入 nullptr 时只生成深度。
 * @param max_valid_depth_meters 最大有效深度，单位为米。
 * @param output 输出对齐后的深度与 IR；不可为 nullptr。
 * @param diagnostics 可选投影诊断计数。
 * @param error 失败原因；可为 nullptr。
 * @return 投影流程可执行时返回 true；标定、类型或尺寸非法时返回 false。
 */
    bool registerTofImagesToRgb(
      const xv_ros2::rgb_fisheye::KalibrCam0Calibration & calibration,
      const xv::Calibration & tof_calibration,
      const xv::DepthImage & depth_image,
      const xv::DepthImage *ir_image,
      double max_valid_depth_meters,
      RegisteredTofImages *output,
      ProjectionDiagnostics *diagnostics = nullptr,
      std::string *error = nullptr);

/**
 * @brief 将 Kalibr 校正 RGB 采样到 ToF 原生网格，并保留同像素 depth/IR。
 * @param calibration Kalibr cam0 标定。
 * @param tof_calibration SDK ToF 内外参。
 * @param depth_image ToF 深度图。
 * @param ir_image 与深度同尺寸的 ToF IR 图。
 * @param undistorted_rgb Kalibr 校正后的 CV_8UC3 RGB 图。
 * @param max_valid_depth_meters 最大有效深度，单位为米。
 * @param output 输出 ToF 网格三路数据；不可为 nullptr。
 * @param error 失败原因；可为 nullptr。
 * @return 输入与标定可用时返回 true。
 */
    bool registerRgbToTofGrid(
      const xv_ros2::rgb_fisheye::KalibrCam0Calibration & calibration,
      const xv::Calibration & tof_calibration,
      const xv::DepthImage & depth_image,
      const xv::DepthImage & ir_image,
      const cv::Mat & undistorted_rgb,
      double max_valid_depth_meters,
      RgbToTofImages *output,
      std::string *error = nullptr);

/**
 * @brief 构造 ToF 坐标系下的 RGB/ToF 最大公共视场虚拟针孔模型。
 * @param calibration Kalibr cam0 标定。
 * @param tof_calibration SDK ToF 内外参。
 * @param tof_width 运行时 ToF 宽度。
 * @param tof_height 运行时 ToF 高度。
 * @param rgb_remap_x Kalibr 校正图到原始 RGB 的 x remap。
 * @param rgb_remap_y Kalibr 校正图到原始 RGB 的 y remap。
 * @param output 输出虚拟相机及逐像素预映射；不可为 nullptr。
 * @param error 失败原因；可为 nullptr。
 * @return 找到完整公共视场并建立 100% RGB 回退映射时返回 true。
 */
    bool createVirtualRgbdModel(
      const xv_ros2::rgb_fisheye::KalibrCam0Calibration & calibration,
      const xv::Calibration & tof_calibration,
      std::size_t tof_width,
      std::size_t tof_height,
      const cv::Mat & rgb_remap_x,
      const cv::Mat & rgb_remap_y,
      VirtualRgbdModel *output,
      std::string *error = nullptr);

/**
 * @brief 按预计算虚拟相机映射生成逐像素对应的 RGB、depth 和 IR。
 * @param model 公共虚拟相机及预映射。
 * @param calibration Kalibr cam0 标定。
 * @param tof_calibration SDK ToF 内外参。
 * @param depth_image ToF 原始深度图。
 * @param ir_image ToF 原始 IR 图。
 * @param undistorted_rgb Kalibr 校正 RGB 图。
 * @param max_valid_depth_meters 最大有效深度，单位为米。
 * @param output 输出虚拟网格三路数据；不可为 nullptr。
 * @param error 失败原因；可为 nullptr。
 * @return 输入尺寸与预映射一致且处理成功时返回 true。
 */
    bool registerRgbToVirtualGrid(
      const VirtualRgbdModel & model,
      const xv_ros2::rgb_fisheye::KalibrCam0Calibration & calibration,
      const xv::Calibration & tof_calibration,
      const xv::DepthImage & depth_image,
      const xv::DepthImage & ir_image,
      const cv::Mat & undistorted_rgb,
      double max_valid_depth_meters,
      RgbToTofImages *output,
      std::string *error = nullptr);

  } // namespace rgb_registered
} // namespace xv_ros2

#endif // RGB_REGISTERED_UTILS_H_
