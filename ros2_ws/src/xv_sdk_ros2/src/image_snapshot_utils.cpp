/**
 * @file image_snapshot_utils.cpp
 * @brief 实现 ROS 图像单帧转换、命名和安全落盘。
 */

#include "image_snapshot_utils.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <vector>

#include <unistd.h>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>

namespace xv_ros2::image_snapshot {
namespace {

/**
 * @brief 将 ROS 图像转换为适合 OpenCV 编码器写入的通道顺序。
 * @param image 待转换的 ROS 图像。
 * @param converted 输出 OpenCV 图像。
 * @param error 输出错误信息。
 * @return 转换成功时返回 true。
 */
bool convertForWriting(const sensor_msgs::msg::Image &image, cv::Mat *converted,
                       std::string *error) {
  if (converted == nullptr || error == nullptr) {
    return false;
  }

  try {
    /** 保留原始编码和位深的 OpenCV 图像。 */
    const cv_bridge::CvImagePtr cv_image =
        cv_bridge::toCvCopy(image, image.encoding);
    if (image.encoding == sensor_msgs::image_encodings::RGB8) {
      cv::cvtColor(cv_image->image, *converted, cv::COLOR_RGB2BGR);
    } else if (image.encoding == sensor_msgs::image_encodings::RGBA8) {
      cv::cvtColor(cv_image->image, *converted, cv::COLOR_RGBA2BGRA);
    } else {
      *converted = cv_image->image.clone();
    }
  } catch (const cv_bridge::Exception &exception) {
    *error = std::string("图像转换失败：") + exception.what();
    return false;
  } catch (const cv::Exception &exception) {
    *error = std::string("OpenCV 图像转换失败：") + exception.what();
    return false;
  }

  if (converted->empty()) {
    *error = "图像内容为空";
    return false;
  }
  return true;
}

/**
 * @brief 校验文件名只表示输出目录内的单个文件。
 * @param filename 待校验的文件名。
 * @return 合法时返回 true。
 */
bool isPlainFilename(const std::filesystem::path &filename) {
  return !filename.empty() && !filename.is_absolute() &&
         filename.filename() == filename && filename != "." && filename != "..";
}

/**
 * @brief 在输出目录中独占创建保留原扩展名的唯一临时文件。
 * @param output_path 最终输出路径。
 * @param temporary_path 输出创建成功的临时文件路径。
 * @param error 输出错误信息。
 * @return 创建成功时返回 true。
 */
bool createUniqueTemporaryFile(const std::filesystem::path &output_path,
                               std::filesystem::path *temporary_path,
                               std::string *error) {
  if (temporary_path == nullptr || error == nullptr) {
    return false;
  }

  /** 保留目标扩展名的唯一临时文件模板。 */
  const std::filesystem::path temporary_template =
      output_path.parent_path() /
      (output_path.stem().string() + ".tmp.XXXXXX" +
       output_path.extension().string());
  /** 临时文件模板的本地编码字符串。 */
  const std::string temporary_template_string = temporary_template.string();
  /** 提供给 mkstemps 修改的可写路径缓冲区。 */
  std::vector<char> template_buffer(temporary_template_string.begin(),
                                    temporary_template_string.end());
  template_buffer.push_back('\0');
  /** mkstemps 需要保留的文件扩展名长度。 */
  const int suffix_length =
      static_cast<int>(output_path.extension().string().size());
  /** 独占创建得到的临时文件描述符。 */
  const int temporary_fd = mkstemps(template_buffer.data(), suffix_length);
  if (temporary_fd == -1) {
    /** 临时文件创建失败时的系统错误码。 */
    const std::error_code create_error(errno, std::generic_category());
    *error = "创建临时图像文件失败：" + create_error.message();
    return false;
  }

  *temporary_path = std::filesystem::path(template_buffer.data());
  if (close(temporary_fd) != 0) {
    /** 关闭临时文件失败时的系统错误码。 */
    const std::error_code close_error(errno, std::generic_category());
    /** 清理无法安全交给 OpenCV 写入的临时文件。 */
    std::error_code cleanup_error;
    std::filesystem::remove(*temporary_path, cleanup_error);
    *error = "关闭临时图像文件失败：" + close_error.message();
    return false;
  }
  return true;
}

/**
 * @brief 原子提交临时文件，并按配置决定是否允许替换目标文件。
 * @param temporary_path 已完整写入的临时文件路径。
 * @param output_path 最终输出路径。
 * @param overwrite 是否允许替换已有目标。
 * @param error 输出错误信息。
 * @return 提交成功时返回 true。
 */
bool commitTemporaryFile(const std::filesystem::path &temporary_path,
                         const std::filesystem::path &output_path,
                         bool overwrite, std::string *error) {
  if (error == nullptr) {
    return false;
  }

  if (overwrite) {
    /** 覆盖提交过程的文件系统错误码。 */
    std::error_code rename_error;
    std::filesystem::rename(temporary_path, output_path, rename_error);
    if (rename_error) {
      *error = "提交输出文件失败：" + rename_error.message();
      return false;
    }
    return true;
  }

  if (link(temporary_path.c_str(), output_path.c_str()) != 0) {
    /** 原子创建最终路径失败时的系统错误码。 */
    const std::error_code link_error(errno, std::generic_category());
    if (link_error == std::errc::file_exists) {
      *error = "输出文件已存在且 overwrite=false：" + output_path.string();
    } else {
      *error = "提交输出文件失败：" + link_error.message();
    }
    return false;
  }

  /** 删除成功提交后保留的临时硬链接时的错误码。 */
  std::error_code cleanup_error;
  std::filesystem::remove(temporary_path, cleanup_error);
  if (cleanup_error) {
    *error = "清理已提交的临时文件失败：" + cleanup_error.message();
    return false;
  }
  return true;
}

} // namespace

/**
 * @brief 判断图像编码是否受截图工具支持。
 * @param encoding ROS 图像编码。
 * @return 支持时返回 true，否则返回 false。
 */
bool isSupportedEncoding(const std::string &encoding) {
  /** 节点明确支持的 ROS 图像编码列表。 */
  const std::array<std::string, 7> supported_encodings{
      sensor_msgs::image_encodings::RGB8,
      sensor_msgs::image_encodings::BGR8,
      sensor_msgs::image_encodings::RGBA8,
      sensor_msgs::image_encodings::BGRA8,
      sensor_msgs::image_encodings::MONO8,
      sensor_msgs::image_encodings::MONO16,
      sensor_msgs::image_encodings::TYPE_32FC1,
  };
  return std::find(supported_encodings.begin(), supported_encodings.end(),
                   encoding) != supported_encodings.end();
}

/**
 * @brief 根据图像编码和时间生成默认文件名。
 * @param encoding ROS 图像编码。
 * @param timestamp 用于文件名的系统时间。
 * @return 自动生成的截图文件名。
 */
std::string
makeDefaultFilename(const std::string &encoding,
                    const std::chrono::system_clock::time_point &timestamp) {
  /** 时间戳对应的秒级系统时间。 */
  const std::time_t seconds = std::chrono::system_clock::to_time_t(timestamp);
  /** 线程安全转换得到的本地时间。 */
  std::tm local_time{};
  localtime_r(&seconds, &local_time);
  /** 秒内毫秒数，用于降低文件名冲突概率。 */
  const auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          timestamp.time_since_epoch()) %
      1000;
  /** 浮点深度使用 TIFF，其余编码使用 PNG。 */
  const char *extension =
      encoding == sensor_msgs::image_encodings::TYPE_32FC1 ? ".tiff" : ".png";
  /** 自动文件名输出流。 */
  std::ostringstream filename;
  filename << "snapshot_" << std::put_time(&local_time, "%Y%m%d_%H%M%S") << "_"
           << std::setw(3) << std::setfill('0') << milliseconds.count()
           << extension;
  return filename.str();
}

/**
 * @brief 将一帧 ROS 图像保存到指定目录。
 * @param image 待保存的 ROS 图像。
 * @param options 输出目录、文件名和覆盖策略。
 * @return 保存结果，包含输出路径或错误信息。
 */
SaveResult saveImage(const sensor_msgs::msg::Image &image,
                     const SaveOptions &options) {
  /** 本次保存的返回结果。 */
  SaveResult result;
  if (options.output_dir.empty()) {
    result.error = "output_dir 不能为空";
    return result;
  }
  if (!isSupportedEncoding(image.encoding)) {
    result.error = "不支持的图像编码：" + image.encoding;
    return result;
  }

  /** 实际使用的输出文件名。 */
  std::filesystem::path filename =
      options.filename.empty()
          ? makeDefaultFilename(image.encoding,
                                std::chrono::system_clock::now())
          : options.filename;
  if (filename.extension().empty()) {
    /** 按图像编码补充的默认文件扩展名。 */
    const char *extension =
        image.encoding == sensor_msgs::image_encodings::TYPE_32FC1 ? ".tiff"
                                                                   : ".png";
    filename += extension;
  }
  if (!isPlainFilename(filename)) {
    result.error = "filename 必须是输出目录内的单个文件名";
    return result;
  }

  /** 文件系统操作的错误码。 */
  std::error_code filesystem_error;
  std::filesystem::create_directories(options.output_dir, filesystem_error);
  if (filesystem_error) {
    result.error = "创建输出目录失败：" + filesystem_error.message();
    return result;
  }
  if (!std::filesystem::is_directory(options.output_dir, filesystem_error) ||
      filesystem_error) {
    result.error = "output_dir 不是可用目录";
    return result;
  }

  /** 最终输出文件路径。 */
  const std::filesystem::path output_path = options.output_dir / filename;
  if (!options.overwrite &&
      std::filesystem::exists(output_path, filesystem_error)) {
    result.error = "输出文件已存在且 overwrite=false：" + output_path.string();
    return result;
  }
  if (filesystem_error) {
    result.error = "检查输出文件失败：" + filesystem_error.message();
    return result;
  }

  /** 适合 OpenCV 文件编码器的图像。 */
  cv::Mat converted;
  if (!convertForWriting(image, &converted, &result.error)) {
    return result;
  }

  /** 保留目标扩展名的唯一临时文件路径。 */
  std::filesystem::path temporary_path;
  if (!createUniqueTemporaryFile(output_path, &temporary_path, &result.error)) {
    return result;
  }
  try {
    if (!cv::imwrite(temporary_path.string(), converted)) {
      /** 清理编码器可能创建的不完整临时文件。 */
      std::error_code cleanup_error;
      std::filesystem::remove(temporary_path, cleanup_error);
      result.error = "OpenCV 无法写入图像文件：" + temporary_path.string();
      return result;
    }
  } catch (const cv::Exception &exception) {
    /** 清理编码器异常后可能残留的不完整临时文件。 */
    std::error_code cleanup_error;
    std::filesystem::remove(temporary_path, cleanup_error);
    result.error = std::string("写入图像失败：") + exception.what();
    return result;
  }

  if (!commitTemporaryFile(temporary_path, output_path, options.overwrite,
                           &result.error)) {
    /** 清理未完成的临时图像文件。 */
    std::error_code cleanup_error;
    std::filesystem::remove(temporary_path, cleanup_error);
    return result;
  }

  result.success = true;
  result.output_path = output_path;
  return result;
}

} // namespace xv_ros2::image_snapshot
