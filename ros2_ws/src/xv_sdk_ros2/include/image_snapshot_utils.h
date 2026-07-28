/**
 * @file image_snapshot_utils.h
 * @brief 声明 ROS 图像单帧转换、命名和落盘工具。
 */

#pragma once

#include <chrono>
#include <filesystem>
#include <string>

#include <sensor_msgs/msg/image.hpp>

namespace xv_ros2::image_snapshot {

/** @brief 单帧图像的保存选项。 */
struct SaveOptions {
  /** 输出目录。 */
  std::filesystem::path output_dir;
  /** 输出文件名；为空时自动生成。 */
  std::string filename;
  /** 是否允许覆盖已有文件。 */
  bool overwrite{false};
};

/** @brief 单帧图像的保存结果。 */
struct SaveResult {
  /** 保存是否成功。 */
  bool success{false};
  /** 成功时的完整输出路径。 */
  std::filesystem::path output_path;
  /** 失败时的错误信息。 */
  std::string error;
};

/**
 * @brief 判断图像编码是否受截图工具支持。
 * @param encoding ROS 图像编码。
 * @return 支持时返回 true，否则返回 false。
 */
bool isSupportedEncoding(const std::string &encoding);

/**
 * @brief 根据图像编码和时间生成默认文件名。
 * @param encoding ROS 图像编码。
 * @param timestamp 用于文件名的系统时间。
 * @return 形如 snapshot_YYYYmmdd_HHMMSS_mmm.png 的文件名；32FC1 使用 tiff。
 */
std::string
makeDefaultFilename(const std::string &encoding,
                    const std::chrono::system_clock::time_point &timestamp);

/**
 * @brief 将一帧 ROS 图像保存到指定目录。
 * @param image 待保存的 ROS 图像。
 * @param options 输出目录、文件名和覆盖策略。
 * @return 保存结果，包含输出路径或错误信息。
 * @note 该函数可能创建输出目录，并通过临时文件完成最终写入。
 */
SaveResult saveImage(const sensor_msgs::msg::Image &image,
                     const SaveOptions &options);

} // namespace xv_ros2::image_snapshot
