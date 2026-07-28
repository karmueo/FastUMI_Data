/**
 * @file rgb_registered_utils.cpp
 * @brief 实现 RGB/ToF 配准、公共虚拟相机和时间同步基础工具函数。
 */

#include "rgb_registered_utils.h"

#include "rgb_fisheye_undistort_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace
{
/**
 * @brief 写入可选错误信息。
 * @param error 错误信息指针。
 * @param message 错误内容。
 */
void setError(std::string *error, const std::string & message)
{
  if (error) {
    *error = message;
  }
}

/**
 * @brief SDK ToF 反投影模型及其原生标定分辨率。
 */
struct TofRaytraceModel
{
  /** 新版 SDK 通用相机模型。 */
  const xv::CameraModel *camera_model = nullptr;
  /** 旧版 SDK PDM 模型。 */
  const xv::PolynomialDistortionCameraModel *pdm_model = nullptr;
  /** 旧版 SDK UCM 模型。 */
  const xv::UnifiedCameraModel *ucm_model = nullptr;
  /** 标定模型原生宽度。 */
  std::size_t width = 0;
  /** 标定模型原生高度。 */
  std::size_t height = 0;
};

/**
 * @brief 计算标定模型分辨率与运行时分辨率的缩放距离。
 * @param model_width 标定模型宽度。
 * @param model_height 标定模型高度。
 * @param image_width 运行时图像宽度。
 * @param image_height 运行时图像高度。
 * @return 越小表示两个分辨率越接近。
 */
double resolutionScaleDistance(
  std::size_t model_width, std::size_t model_height,
  std::size_t image_width, std::size_t image_height)
{
  return std::abs(std::log(
           static_cast<double>(model_width) / static_cast<double>(image_width))) +
         std::abs(std::log(
           static_cast<double>(model_height) / static_cast<double>(image_height)));
}

/**
 * @brief 选择最适合运行时 ToF 分辨率的 SDK 标定模型。
 * @param calibration SDK ToF 标定。
 * @param width 运行时图像宽度。
 * @param height 运行时图像高度。
 * @return 优先精确匹配，否则返回缩放比例最接近的可用模型。
 */
TofRaytraceModel selectTofRaytraceModel(
  const xv::Calibration & calibration, std::size_t width, std::size_t height)
{
  for (const auto & model : calibration.camerasModel) {
    if (model && model->width() == static_cast<std::int32_t>(width) &&
      model->height() == static_cast<std::int32_t>(height))
    {
      return {model.get(), nullptr, nullptr, width, height};
    }
  }
  for (const auto & model : calibration.pdcm) {
    if (model.w == static_cast<int>(width) && model.h == static_cast<int>(height)) {
      return {nullptr, &model, nullptr, width, height};
    }
  }
  for (const auto & model : calibration.ucm) {
    if (model.w == static_cast<int>(width) && model.h == static_cast<int>(height)) {
      return {nullptr, nullptr, &model, width, height};
    }
  }

  /** 当前最接近运行时分辨率的模型。 */
  TofRaytraceModel best_model;
  /** 当前最小分辨率缩放距离。 */
  double best_distance = std::numeric_limits<double>::infinity();
  for (const auto & model : calibration.camerasModel) {
    if (!model || model->width() <= 0 || model->height() <= 0) {
      continue;
    }
    /** 当前通用模型的分辨率缩放距离。 */
    const double distance = resolutionScaleDistance(
      static_cast<std::size_t>(model->width()),
      static_cast<std::size_t>(model->height()), width, height);
    if (distance < best_distance) {
      best_distance = distance;
      best_model = {model.get(), nullptr, nullptr,
        static_cast<std::size_t>(model->width()),
        static_cast<std::size_t>(model->height())};
    }
  }
  for (const auto & model : calibration.pdcm) {
    if (model.w <= 0 || model.h <= 0) {
      continue;
    }
    /** 当前 PDM 模型的分辨率缩放距离。 */
    const double distance = resolutionScaleDistance(
      static_cast<std::size_t>(model.w), static_cast<std::size_t>(model.h),
      width, height);
    if (distance < best_distance) {
      best_distance = distance;
      best_model = {nullptr, &model, nullptr,
        static_cast<std::size_t>(model.w),
        static_cast<std::size_t>(model.h)};
    }
  }
  for (const auto & model : calibration.ucm) {
    if (model.w <= 0 || model.h <= 0) {
      continue;
    }
    /** 当前 UCM 模型的分辨率缩放距离。 */
    const double distance = resolutionScaleDistance(
      static_cast<std::size_t>(model.w), static_cast<std::size_t>(model.h),
      width, height);
    if (distance < best_distance) {
      best_distance = distance;
      best_model = {nullptr, nullptr, &model,
        static_cast<std::size_t>(model.w),
        static_cast<std::size_t>(model.h)};
    }
  }

  if (best_model.camera_model || best_model.pdm_model || best_model.ucm_model) {
    return best_model;
  }

  for (const auto & model : calibration.camerasModel) {
    if (model) {
      return {model.get(), nullptr, nullptr, width, height};
    }
  }
  if (!calibration.pdcm.empty()) {
    return {nullptr, &calibration.pdcm.front(), nullptr, width, height};
  }
  if (!calibration.ucm.empty()) {
    return {nullptr, nullptr, &calibration.ucm.front(), width, height};
  }
  return {};
}

/**
 * @brief 将运行时图像像素中心映射到标定模型的原生分辨率。
 * @param coordinate 运行时像素坐标。
 * @param image_size 运行时图像轴向尺寸。
 * @param model_size 标定模型轴向尺寸。
 * @return 标定模型分辨率下的像素坐标。
 */
double scalePixelCoordinate(
  double coordinate, std::size_t image_size, std::size_t model_size)
{
  return (coordinate + 0.5) * static_cast<double>(model_size) /
         static_cast<double>(image_size) - 0.5;
}

/**
 * @brief 使用 PDM 内参将像素反投影为相机射线。
 * @param model PDM 内参。
 * @param pixel 输入像素坐标。
 * @param ray 输出三维射线。
 * @return 反投影成功返回 true。
 */
bool raytracePdm(
  const xv::PolynomialDistortionCameraModel & model,
  const double *pixel, double *ray)
{
  /** 当前归一化 x 坐标。 */
  double x = (pixel[0] - model.u0) / model.fx;
  /** 当前归一化 y 坐标。 */
  double y = (pixel[1] - model.v0) / model.fy;
  /** 原始畸变归一化 x 坐标。 */
  const double x0 = x;
  /** 原始畸变归一化 y 坐标。 */
  const double y0 = y;

  for (int iteration = 0; iteration < 5; ++iteration) {
    /** 当前半径平方。 */
    const double r2 = x * x + y * y;
    /** 当前半径四次方。 */
    const double r4 = r2 * r2;
    /** 当前半径六次方。 */
    const double r6 = r4 * r2;
    /** 径向畸变比例。 */
    const double radial = 1.0 + model.distor[0] * r2 +
      model.distor[1] * r4 + model.distor[4] * r6;
    /** 切向畸变 x 分量。 */
    const double delta_x = 2.0 * model.distor[2] * x * y +
      model.distor[3] * (r2 + 2.0 * x * x);
    /** 切向畸变 y 分量。 */
    const double delta_y = model.distor[2] * (r2 + 2.0 * y * y) +
      2.0 * model.distor[3] * x * y;
    if (std::abs(radial) < 1e-12) {
      return false;
    }
    x = (x0 - delta_x) / radial;
    y = (y0 - delta_y) / radial;
  }

  ray[0] = x;
  ray[1] = y;
  ray[2] = 1.0;
  return true;
}

/**
 * @brief 使用 UCM 内参将像素反投影为相机射线。
 * @param model UCM 内参。
 * @param pixel 输入像素坐标。
 * @param ray 输出三维射线。
 * @return 反投影成功返回 true。
 */
bool raytraceUcm(
  const xv::UnifiedCameraModel & model, const double *pixel, double *ray)
{
  if (!std::isfinite(model.fx) || !std::isfinite(model.fy) ||
    !std::isfinite(model.xi) || std::abs(model.fx) < 1e-12 ||
    std::abs(model.fy) < 1e-12)
  {
    return false;
  }
  /** 归一化图像 x 坐标。 */
  const double x = (pixel[0] - model.u0) / model.fx;
  /** 归一化图像 y 坐标。 */
  const double y = (pixel[1] - model.v0) / model.fy;
  /** 归一化图像半径平方。 */
  const double radius_squared = x * x + y * y;
  /** UCM 反投影根号项。 */
  const double radicand =
    1.0 + (1.0 - model.xi * model.xi) * radius_squared;
  if (radicand < 0.0) {
    return false;
  }
  /** UCM 射线缩放系数。 */
  const double scale =
    (model.xi + std::sqrt(radicand)) / (1.0 + radius_squared);
  ray[0] = scale * x;
  ray[1] = scale * y;
  ray[2] = scale - model.xi;
  return std::isfinite(ray[0]) && std::isfinite(ray[1]) &&
         std::isfinite(ray[2]);
}

/**
 * @brief 使用选定 SDK 模型反投影运行时 ToF 像素。
 * @param model 已选定的 ToF 模型。
 * @param image_width 运行时图像宽度。
 * @param image_height 运行时图像高度。
 * @param pixel 运行时 ToF 像素。
 * @param ray 输出 ToF 相机射线。
 * @return 反投影成功返回 true。
 */
bool raytraceTofPixel(
  const TofRaytraceModel & model,
  std::size_t image_width, std::size_t image_height,
  const double *pixel, double *ray)
{
  /** 映射到标定模型原生分辨率的像素。 */
  const double calibration_pixel[2] = {
    scalePixelCoordinate(pixel[0], image_width, model.width),
    scalePixelCoordinate(pixel[1], image_height, model.height)};
  if (model.camera_model) {
    return model.camera_model->raytrace(calibration_pixel, ray);
  }
  if (model.pdm_model) {
    return raytracePdm(*model.pdm_model, calibration_pixel, ray);
  }
  return model.ucm_model && raytraceUcm(*model.ucm_model, calibration_pixel, ray);
}

/**
 * @brief 忽略平移，仅按相机旋转将 ToF 射线投影到 RGB 校正图。
 * @param calibration Kalibr RGB 标定。
 * @param tof_pose ToF 到 IMU 外参。
 * @param tof_ray ToF 相机射线。
 * @param pixel 输出 RGB 像素。
 * @return 投影位于 RGB 图像内时返回 true。
 */
bool projectTofRayByRotation(
  const xv_ros2::rgb_fisheye::KalibrCam0Calibration & calibration,
  const xv::Transform & tof_pose, const double *tof_ray, cv::Point2d *pixel);

/**
 * @brief 使用 PDM 内参将三维射线投影到标定分辨率像素。
 * @param model PDM 内参。
 * @param ray ToF 相机坐标系射线。
 * @param pixel 输出标定分辨率像素坐标。
 * @return 投影成功返回 true。
 */
bool projectPdm(
  const xv::PolynomialDistortionCameraModel & model,
  const double *ray, double *pixel)
{
  if (!ray || !pixel || !std::isfinite(ray[0]) || !std::isfinite(ray[1]) ||
    !std::isfinite(ray[2]) || ray[2] <= 1e-12 ||
    !std::isfinite(model.fx) || !std::isfinite(model.fy) ||
    std::abs(model.fx) < 1e-12 || std::abs(model.fy) < 1e-12)
  {
    return false;
  }
  /** 归一化 x 坐标。 */
  const double x = ray[0] / ray[2];
  /** 归一化 y 坐标。 */
  const double y = ray[1] / ray[2];
  /** 归一化半径平方。 */
  const double r2 = x * x + y * y;
  /** 归一化半径四次方。 */
  const double r4 = r2 * r2;
  /** 归一化半径六次方。 */
  const double r6 = r4 * r2;
  /** 径向畸变比例。 */
  const double radial = 1.0 + model.distor[0] * r2 +
    model.distor[1] * r4 + model.distor[4] * r6;
  /** 畸变后的归一化 x 坐标。 */
  const double distorted_x = x * radial + 2.0 * model.distor[2] * x * y +
    model.distor[3] * (r2 + 2.0 * x * x);
  /** 畸变后的归一化 y 坐标。 */
  const double distorted_y = y * radial + model.distor[2] *
    (r2 + 2.0 * y * y) + 2.0 * model.distor[3] * x * y;
  pixel[0] = model.fx * distorted_x + model.u0;
  pixel[1] = model.fy * distorted_y + model.v0;
  return std::isfinite(pixel[0]) && std::isfinite(pixel[1]);
}

/**
 * @brief 使用 UCM 内参将三维射线投影到标定分辨率像素。
 * @param model UCM 内参。
 * @param ray ToF 相机坐标系射线。
 * @param pixel 输出标定分辨率像素坐标。
 * @return 投影成功返回 true。
 */
bool projectUcm(
  const xv::UnifiedCameraModel & model, const double *ray, double *pixel)
{
  if (!ray || !pixel || !std::isfinite(ray[0]) || !std::isfinite(ray[1]) ||
    !std::isfinite(ray[2]) || !std::isfinite(model.xi) ||
    !std::isfinite(model.fx) || !std::isfinite(model.fy) ||
    std::abs(model.fx) < 1e-12 || std::abs(model.fy) < 1e-12)
  {
    return false;
  }
  /** 三维射线模长。 */
  const double norm = std::sqrt(
    ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]);
  /** UCM 投影分母。 */
  const double denominator = ray[2] + model.xi * norm;
  if (!std::isfinite(norm) || norm <= 1e-12 || denominator <= 1e-12) {
    return false;
  }
  pixel[0] = model.fx * ray[0] / denominator + model.u0;
  pixel[1] = model.fy * ray[1] / denominator + model.v0;
  return std::isfinite(pixel[0]) && std::isfinite(pixel[1]);
}

/**
 * @brief 使用选定 SDK 模型将 ToF 射线投影到运行时像素。
 * @param model 已选定的 ToF 模型。
 * @param image_width 运行时图像宽度。
 * @param image_height 运行时图像高度。
 * @param ray ToF 相机坐标系射线。
 * @param pixel 输出运行时像素坐标。
 * @return 投影成功返回 true。
 */
bool projectTofRay(
  const TofRaytraceModel & model,
  std::size_t image_width, std::size_t image_height,
  const double *ray, double *pixel)
{
  /** 标定模型原生分辨率像素。 */
  double calibration_pixel[2] = {0.0, 0.0};
  /** SDK 模型投影结果。 */
  bool projected = false;
  if (model.camera_model) {
    projected = model.camera_model->project(ray, calibration_pixel);
  } else if (model.pdm_model) {
    projected = projectPdm(*model.pdm_model, ray, calibration_pixel);
  } else if (model.ucm_model) {
    projected = projectUcm(*model.ucm_model, ray, calibration_pixel);
  }
  if (!projected || model.width == 0 || model.height == 0) {
    return false;
  }
  pixel[0] = (calibration_pixel[0] + 0.5) *
    static_cast<double>(image_width) / static_cast<double>(model.width) - 0.5;
  pixel[1] = (calibration_pixel[1] + 0.5) *
    static_cast<double>(image_height) / static_cast<double>(model.height) - 0.5;
  return std::isfinite(pixel[0]) && std::isfinite(pixel[1]);
}

/**
 * @brief 判断浮点像素可在图像内安全执行双线性采样。
 * @param pixel 浮点像素坐标。
 * @param width 图像宽度。
 * @param height 图像高度。
 * @param valid_mask 每个整数像素的有效掩码。
 * @return 四个双线性邻域像素均有效时返回 true。
 */
bool isBilinearSampleValid(
  const cv::Point2d & pixel, std::size_t width, std::size_t height,
  const std::vector<std::uint8_t> & valid_mask)
{
  if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) || width == 0 ||
    height == 0 || valid_mask.size() != width * height || pixel.x < 0.0 ||
    pixel.y < 0.0 || pixel.x > static_cast<double>(width - 1U) ||
    pixel.y > static_cast<double>(height - 1U))
  {
    return false;
  }
  /** 双线性邻域左侧 x。 */
  const std::size_t x0 = static_cast<std::size_t>(std::floor(pixel.x));
  /** 双线性邻域顶部 y。 */
  const std::size_t y0 = static_cast<std::size_t>(std::floor(pixel.y));
  /** 双线性邻域右侧 x。 */
  const std::size_t x1 = std::min(x0 + 1U, width - 1U);
  /** 双线性邻域底部 y。 */
  const std::size_t y1 = std::min(y0 + 1U, height - 1U);
  return valid_mask[y0 * width + x0] != 0U &&
         valid_mask[y0 * width + x1] != 0U &&
         valid_mask[y1 * width + x0] != 0U &&
         valid_mask[y1 * width + x1] != 0U;
}

/**
 * @brief 从 OpenCV remap 构造校正 RGB 的有效像素掩码。
 * @param map_x 校正图到原始图的 x remap。
 * @param map_y 校正图到原始图的 y remap。
 * @param raw_width 原始 RGB 宽度。
 * @param raw_height 原始 RGB 高度。
 * @param valid_mask 输出行优先有效掩码。
 * @return remap 类型、尺寸与原始分辨率合法时返回 true。
 */
bool buildRgbRemapValidityMask(
  const cv::Mat & map_x, const cv::Mat & map_y,
  std::size_t raw_width, std::size_t raw_height,
  std::vector<std::uint8_t> *valid_mask)
{
  if (!valid_mask || map_x.empty() || map_y.empty() ||
    map_x.type() != CV_32FC1 || map_y.type() != CV_32FC1 ||
    map_x.size() != map_y.size() || raw_width == 0 || raw_height == 0)
  {
    return false;
  }
  /** remap 输出像素总数。 */
  const std::size_t pixel_count = static_cast<std::size_t>(map_x.cols) *
    static_cast<std::size_t>(map_x.rows);
  valid_mask->assign(pixel_count, 0U);
  for (int y = 0; y < map_x.rows; ++y) {
    for (int x = 0; x < map_x.cols; ++x) {
      /** 原始 RGB x 采样坐标。 */
      const float source_x = map_x.at<float>(y, x);
      /** 原始 RGB y 采样坐标。 */
      const float source_y = map_y.at<float>(y, x);
      /** 校正图行优先索引。 */
      const std::size_t index = static_cast<std::size_t>(y) *
        static_cast<std::size_t>(map_x.cols) + static_cast<std::size_t>(x);
      if (std::isfinite(source_x) && std::isfinite(source_y) &&
        source_x >= 0.0F && source_y >= 0.0F &&
        source_x <= static_cast<float>(raw_width - 1U) &&
        source_y <= static_cast<float>(raw_height - 1U))
      {
        (*valid_mask)[index] = 1U;
      }
    }
  }
  return true;
}

/**
 * @brief 二值掩码中的轴对齐矩形。
 */
struct MaskRectangle
{
  /** 左边界，包含。 */
  std::size_t left = 0;
  /** 右边界，包含。 */
  std::size_t right = 0;
  /** 上边界，包含。 */
  std::size_t top = 0;
  /** 下边界，包含。 */
  std::size_t bottom = 0;
  /** 矩形面积。 */
  std::size_t area = 0;
};

/**
 * @brief 查找二值掩码中的最大轴对齐全有效矩形。
 * @param mask 行优先二值掩码。
 * @param width 掩码宽度。
 * @param height 掩码高度。
 * @return 最大面积矩形；不存在有效像素时 area 为 0。
 */
MaskRectangle largestValidRectangle(
  const std::vector<std::uint8_t> & mask,
  std::size_t width, std::size_t height)
{
  /** 各列截至当前行的连续有效高度。 */
  std::vector<std::size_t> histogram(width, 0U);
  /** 单调栈列索引。 */
  std::vector<std::size_t> stack;
  /** 当前最大矩形。 */
  MaskRectangle best;
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t column = 0; column < width; ++column) {
      histogram[column] = mask[row * width + column] != 0U ?
        histogram[column] + 1U : 0U;
    }
    stack.clear();
    for (std::size_t column = 0; column <= width; ++column) {
      /** 当前柱高；末尾哨兵为零。 */
      const std::size_t current_height =
        column < width ? histogram[column] : 0U;
      while (!stack.empty() && histogram[stack.back()] > current_height) {
        /** 弹出柱对应的矩形高度。 */
        const std::size_t rectangle_height = histogram[stack.back()];
        stack.pop_back();
        /** 矩形左边界。 */
        const std::size_t left = stack.empty() ? 0U : stack.back() + 1U;
        /** 矩形宽度。 */
        const std::size_t rectangle_width = column - left;
        /** 矩形面积。 */
        const std::size_t area = rectangle_width * rectangle_height;
        if (area > best.area) {
          best.left = left;
          best.right = column - 1U;
          best.bottom = row;
          best.top = row + 1U - rectangle_height;
          best.area = area;
        }
      }
      stack.push_back(column);
    }
  }
  return best;
}

/**
 * @brief 根据归一化视场边界计算虚拟针孔内参。
 * @param width 输出宽度。
 * @param height 输出高度。
 * @param min_x 最小归一化 x。
 * @param max_x 最大归一化 x。
 * @param min_y 最小归一化 y。
 * @param max_y 最大归一化 y。
 * @param fx 输出 x 焦距。
 * @param fy 输出 y 焦距。
 * @param cx 输出 x 主点。
 * @param cy 输出 y 主点。
 * @return 边界和尺寸合法时返回 true。
 */
bool computeVirtualIntrinsics(
  std::size_t width, std::size_t height,
  double min_x, double max_x, double min_y, double max_y,
  double *fx, double *fy, double *cx, double *cy)
{
  if (!fx || !fy || !cx || !cy || width < 2U || height < 2U ||
    !std::isfinite(min_x) || !std::isfinite(max_x) ||
    !std::isfinite(min_y) || !std::isfinite(max_y) ||
    max_x - min_x <= 1e-12 || max_y - min_y <= 1e-12)
  {
    return false;
  }
  *fx = static_cast<double>(width - 1U) / (max_x - min_x);
  *fy = static_cast<double>(height - 1U) / (max_y - min_y);
  *cx = -min_x * (*fx);
  *cy = -min_y * (*fy);
  return std::isfinite(*fx) && std::isfinite(*fy) &&
         std::isfinite(*cx) && std::isfinite(*cy);
}

/**
 * @brief 建立给定视场的虚拟像素预映射。
 * @param calibration Kalibr RGB 标定。
 * @param tof_calibration SDK ToF 标定。
 * @param tof_model 运行时 ToF 对应模型。
 * @param rgb_valid_mask 校正 RGB 有效掩码。
 * @param model 输入视场与尺寸，输出内参和逐像素映射。
 * @return 所有虚拟像素都具备有效 ToF/RGB 映射时返回 true。
 */
bool buildVirtualMappings(
  const xv_ros2::rgb_fisheye::KalibrCam0Calibration & calibration,
  const xv::Calibration & tof_calibration,
  const TofRaytraceModel & tof_model,
  const std::vector<std::uint8_t> & rgb_valid_mask,
  xv_ros2::rgb_registered::VirtualRgbdModel *model)
{
  if (!model || !computeVirtualIntrinsics(
      model->width, model->height, model->min_x, model->max_x,
      model->min_y, model->max_y, &model->fx, &model->fy,
      &model->cx, &model->cy))
  {
    return false;
  }
  /** 虚拟网格像素数量。 */
  const std::size_t pixel_count = model->width * model->height;
  model->tof_source_indices.assign(pixel_count, 0U);
  model->virtual_rays.assign(pixel_count, cv::Point3f());
  model->fallback_rgb_pixels.assign(pixel_count, cv::Point2f());
  for (std::size_t index = 0; index < pixel_count; ++index) {
    /** 虚拟像素 x 坐标。 */
    const std::size_t x = index % model->width;
    /** 虚拟像素 y 坐标。 */
    const std::size_t y = index / model->width;
    /** 当前虚拟针孔射线。 */
    const double ray[3] = {
      (static_cast<double>(x) - model->cx) / model->fx,
      (static_cast<double>(y) - model->cy) / model->fy, 1.0};
    /** 当前射线投影到运行时 ToF 的浮点像素。 */
    double tof_pixel[2] = {0.0, 0.0};
    if (!projectTofRay(
        tof_model, model->width, model->height, ray, tof_pixel))
    {
      return false;
    }
    /** 最近邻 ToF 源像素 x。 */
    const long source_x = static_cast<long>(std::floor(tof_pixel[0] + 0.5));
    /** 最近邻 ToF 源像素 y。 */
    const long source_y = static_cast<long>(std::floor(tof_pixel[1] + 0.5));
    if (source_x < 0 || source_y < 0 ||
      source_x >= static_cast<long>(model->width) ||
      source_y >= static_cast<long>(model->height))
    {
      return false;
    }
    /** 深度无效时使用的旋转 RGB 像素。 */
    cv::Point2d fallback_pixel;
    if (!projectTofRayByRotation(
        calibration, tof_calibration.pose, ray, &fallback_pixel) ||
      !isBilinearSampleValid(
        fallback_pixel, model->rgb_width, model->rgb_height,
        rgb_valid_mask))
    {
      return false;
    }
    model->tof_source_indices[index] = static_cast<std::size_t>(source_y) *
      model->width + static_cast<std::size_t>(source_x);
    model->virtual_rays[index] = cv::Point3f(
      static_cast<float>(ray[0]), static_cast<float>(ray[1]), 1.0F);
    model->fallback_rgb_pixels[index] = cv::Point2f(
      static_cast<float>(fallback_pixel.x),
      static_cast<float>(fallback_pixel.y));
  }
  model->rgb_valid_mask = rgb_valid_mask;
  return true;
}

/**
 * @brief 忽略平移，仅按相机旋转将 ToF 射线投影到 RGB 校正图。
 * @param calibration Kalibr RGB 标定。
 * @param tof_pose ToF 到 IMU 外参。
 * @param tof_ray ToF 相机射线。
 * @param pixel 输出 RGB 像素。
 * @return 投影位于 RGB 图像内时返回 true。
 */
bool projectTofRayByRotation(
  const xv_ros2::rgb_fisheye::KalibrCam0Calibration & calibration,
  const xv::Transform & tof_pose, const double *tof_ray, cv::Point2d *pixel)
{
  /** ToF 到 IMU 旋转矩阵。 */
  const auto & r_imu_tof = tof_pose.rotation();
  /** IMU 坐标系射线。 */
  const xv::Vector3d imu_ray = {
    r_imu_tof[0] * tof_ray[0] + r_imu_tof[1] * tof_ray[1] +
    r_imu_tof[2] * tof_ray[2],
    r_imu_tof[3] * tof_ray[0] + r_imu_tof[4] * tof_ray[1] +
    r_imu_tof[5] * tof_ray[2],
    r_imu_tof[6] * tof_ray[0] + r_imu_tof[7] * tof_ray[1] +
    r_imu_tof[8] * tof_ray[2]};
  /** Kalibr cam0 坐标系射线，仅应用旋转。 */
  const cv::Point3d rgb_ray(
    calibration.t_cam_imu[0] * imu_ray[0] +
    calibration.t_cam_imu[1] * imu_ray[1] +
    calibration.t_cam_imu[2] * imu_ray[2],
    calibration.t_cam_imu[4] * imu_ray[0] +
    calibration.t_cam_imu[5] * imu_ray[1] +
    calibration.t_cam_imu[6] * imu_ray[2],
    calibration.t_cam_imu[8] * imu_ray[0] +
    calibration.t_cam_imu[9] * imu_ray[1] +
    calibration.t_cam_imu[10] * imu_ray[2]);
  return xv_ros2::rgb_fisheye::projectUndistortedPixel(
    calibration, rgb_ray, pixel);
}

/**
 * @brief 双线性采样 RGB 校正图。
 * @param image CV_8UC3 RGB 图像。
 * @param pixel 浮点像素坐标。
 * @return RGB 三通道采样值。
 */
cv::Vec3b sampleRgbBilinear(const cv::Mat & image, const cv::Point2d & pixel)
{
  /** 左上整数像素 x。 */
  const int x0 = std::clamp(
    static_cast<int>(std::floor(pixel.x)), 0, image.cols - 1);
  /** 左上整数像素 y。 */
  const int y0 = std::clamp(
    static_cast<int>(std::floor(pixel.y)), 0, image.rows - 1);
  /** 右侧整数像素 x。 */
  const int x1 = std::min(x0 + 1, image.cols - 1);
  /** 下侧整数像素 y。 */
  const int y1 = std::min(y0 + 1, image.rows - 1);
  /** x 方向插值权重。 */
  const double weight_x = pixel.x - static_cast<double>(x0);
  /** y 方向插值权重。 */
  const double weight_y = pixel.y - static_cast<double>(y0);
  /** 双线性采样结果。 */
  cv::Vec3b result;
  for (int channel = 0; channel < 3; ++channel) {
    /** 上边缘插值值。 */
    const double top =
      (1.0 - weight_x) * image.at<cv::Vec3b>(y0, x0)[channel] +
      weight_x * image.at<cv::Vec3b>(y0, x1)[channel];
    /** 下边缘插值值。 */
    const double bottom =
      (1.0 - weight_x) * image.at<cv::Vec3b>(y1, x0)[channel] +
      weight_x * image.at<cv::Vec3b>(y1, x1)[channel];
    result[channel] = cv::saturate_cast<std::uint8_t>(
      (1.0 - weight_y) * top + weight_y * bottom);
  }
  return result;
}
}  // namespace

namespace xv_ros2
{
namespace rgb_registered
{

void ProjectionDiagnostics::recordInvalidDepth() {++total_points;}

void ProjectionDiagnostics::recordProjectionFailure()
{
  ++total_points;
  ++valid_depth_points;
  ++projection_failed_points;
}

void ProjectionDiagnostics::recordOutOfBoundsProjection()
{
  ++total_points;
  ++valid_depth_points;
  ++projected_points;
  ++out_of_bounds_points;
}

void ProjectionDiagnostics::recordStoredDepth(bool overwrote_existing_depth)
{
  ++total_points;
  ++valid_depth_points;
  ++projected_points;
  ++stored_pixels;
  if (overwrote_existing_depth) {
    ++overwritten_pixels;
  }
}

void ProjectionDiagnostics::recordDiscardedOverlappedDepth()
{
  ++total_points;
  ++valid_depth_points;
  ++projected_points;
  ++overwritten_pixels;
}

float depthValueMeters(const xv::DepthImage & image, std::size_t index)
{
  /** 深度图像素总数，用于越界保护。 */
  const std::size_t pixel_count = image.width * image.height;
  if (!image.data || index >= pixel_count) {
    return 0.0F;
  }

  if (image.type == xv::DepthImage::Type::Depth_16) {
    /** 16 位深度数组，SDK 约定单位为毫米。 */
    const auto *depth =
      reinterpret_cast<const std::uint16_t *>(image.data.get());
    /** 米制深度值。 */
    const float depth_meters = static_cast<float>(depth[index]) * 0.001F;
    return depth_meters > 0.0F ? depth_meters : 0.0F;
  }

  if (image.type == xv::DepthImage::Type::Depth_32) {
    /** 32 位深度数组，SDK 约定单位为米。 */
    const auto *depth = reinterpret_cast<const float *>(image.data.get());
    /** 米制深度值。 */
    const float depth_meters = depth[index];
    return std::isfinite(depth_meters) && depth_meters > 0.0F ? depth_meters :
           0.0F;
  }

  return 0.0F;
}

float depthValueMeters(
  const xv::DepthImage & image, std::size_t index,
  double max_valid_depth_meters)
{
  if (!std::isfinite(max_valid_depth_meters) ||
    max_valid_depth_meters <= 0.0)
  {
    return 0.0F;
  }

  /** 米制深度值。 */
  const float depth_meters = depthValueMeters(image, index);
  if (depth_meters <= 0.0F ||
    static_cast<double>(depth_meters) >= max_valid_depth_meters)
  {
    return 0.0F;
  }

  return depth_meters;
}

bool isTimestampMatch(double lhs, double rhs)
{
  return std::abs(lhs - rhs) <= kMaxTimestampDiffSeconds;
}

std::optional<TimestampPairMatch> findNearestTimestampPair(
  const std::vector<double> & depth_timestamps,
  const std::vector<double> & color_timestamps,
  double tolerance_sec)
{
  /** 当前最佳配对结果。 */
  TimestampPairMatch best_match;
  /** 当前最佳时间差，单位为秒。 */
  double best_diff = std::numeric_limits<double>::infinity();

  for (std::size_t depth_index = 0; depth_index < depth_timestamps.size();
    ++depth_index)
  {
    for (std::size_t color_index = 0; color_index < color_timestamps.size();
      ++color_index)
    {
      /** 当前 RGB/ToF 时间差，单位为秒。 */
      const double diff = std::abs(
        depth_timestamps[depth_index] - color_timestamps[color_index]);
      if (diff < best_diff) {
        best_diff = diff;
        best_match.depth_index = depth_index;
        best_match.color_index = color_index;
        best_match.diff_seconds = diff;
      }
    }
  }

  if (best_diff <= tolerance_sec) {
    return best_match;
  }
  return std::nullopt;
}

std::optional<TimestampPairMatch> findStableNearestTimestampPair(
  const std::vector<double> & depth_timestamps,
  const std::vector<double> & color_timestamps,
  double tolerance_sec)
{
  if (depth_timestamps.empty() || color_timestamps.empty()) {
    return std::nullopt;
  }

  /** 两侧缓存中的全局最近配对。 */
  const auto nearest_match = findNearestTimestampPair(
    depth_timestamps, color_timestamps, tolerance_sec);
  if (!nearest_match) {
    return std::nullopt;
  }

  /** 当前 ToF 缓存中最新的时间戳，单位为秒。 */
  const double latest_depth_timestamp =
    *std::max_element(depth_timestamps.begin(), depth_timestamps.end());
  /** 当前 RGB 缓存中最新的时间戳，单位为秒。 */
  const double latest_color_timestamp =
    *std::max_element(color_timestamps.begin(), color_timestamps.end());
  /** 最近配对中的 ToF 时间戳，单位为秒。 */
  const double depth_timestamp =
    depth_timestamps[nearest_match->depth_index];
  /** 最近配对中的 RGB 时间戳，单位为秒。 */
  const double color_timestamp =
    color_timestamps[nearest_match->color_index];
  if (latest_depth_timestamp < color_timestamp ||
    latest_color_timestamp < depth_timestamp)
  {
    return std::nullopt;
  }

  return nearest_match;
}

std::optional<TimestampTripleMatch> findStableNearestTimestampTriple(
  const std::vector<double> & depth_timestamps,
  const std::vector<double> & color_timestamps,
  const std::vector<double> & ir_timestamps,
  double tolerance_sec)
{
  if (depth_timestamps.empty() || color_timestamps.empty() ||
    ir_timestamps.empty() || !std::isfinite(tolerance_sec) ||
    tolerance_sec < 0.0)
  {
    return std::nullopt;
  }

  /** 当前最佳三帧组。 */
  TimestampTripleMatch best_match;
  /** 当前最佳最大时间差，单位为秒。 */
  double best_max_diff = std::numeric_limits<double>::infinity();
  for (std::size_t depth_index = 0; depth_index < depth_timestamps.size();
    ++depth_index)
  {
    for (std::size_t color_index = 0; color_index < color_timestamps.size();
      ++color_index)
    {
      /** 当前 RGB 与深度时间差。 */
      const double color_diff = std::abs(
        color_timestamps[color_index] - depth_timestamps[depth_index]);
      if (color_diff > tolerance_sec) {
        continue;
      }
      for (std::size_t ir_index = 0; ir_index < ir_timestamps.size(); ++ir_index) {
        /** 当前 IR 与深度时间差。 */
        const double ir_diff = std::abs(
          ir_timestamps[ir_index] - depth_timestamps[depth_index]);
        /** 当前 RGB 与 IR 时间差。 */
        const double color_ir_diff = std::abs(
          color_timestamps[color_index] - ir_timestamps[ir_index]);
        /** 当前三帧组最大时间差。 */
        const double max_diff = std::max(
          color_diff, std::max(ir_diff, color_ir_diff));
        if (ir_diff <= tolerance_sec && color_ir_diff <= tolerance_sec &&
          max_diff < best_max_diff)
        {
          best_max_diff = max_diff;
          best_match.depth_index = depth_index;
          best_match.color_index = color_index;
          best_match.ir_index = ir_index;
          best_match.color_diff_seconds = color_diff;
          best_match.ir_diff_seconds = ir_diff;
        }
      }
    }
  }

  if (!std::isfinite(best_max_diff)) {
    return std::nullopt;
  }

  /** 缓存中最新深度时间戳。 */
  const double latest_depth =
    *std::max_element(depth_timestamps.begin(), depth_timestamps.end());
  /** 缓存中最新 RGB 时间戳。 */
  const double latest_color =
    *std::max_element(color_timestamps.begin(), color_timestamps.end());
  /** 缓存中最新 IR 时间戳。 */
  const double latest_ir =
    *std::max_element(ir_timestamps.begin(), ir_timestamps.end());
  /** 候选深度时间戳。 */
  const double depth_timestamp = depth_timestamps[best_match.depth_index];
  /** 候选 RGB 时间戳。 */
  const double color_timestamp = color_timestamps[best_match.color_index];
  /** 候选 IR 时间戳。 */
  const double ir_timestamp = ir_timestamps[best_match.ir_index];
  if (latest_depth < std::max(color_timestamp, ir_timestamp) ||
    latest_color < depth_timestamp || latest_ir < depth_timestamp)
  {
    return std::nullopt;
  }
  return best_match;
}

bool projectTofPointToKalibrUndistortedPixel(
  const xv_ros2::rgb_fisheye::KalibrCam0Calibration & calibration,
  const xv::Transform & tof_pose, const xv::Vector3d & tof_point,
  cv::Point2d *pixel, double *kalibr_depth_meters)
{
  if (!pixel) {
    return false;
  }

  /** SDK IMU 坐标系下的三维点，单位为米。 */
  const xv::Vector3d imu_point = tof_pose * tof_point;
  /** Kalibr cam0 到 IMU 的齐次变换矩阵。 */
  const auto & t_cam_imu = calibration.t_cam_imu;
  /** Kalibr cam0 坐标系下的三维点，单位为米。 */
  const cv::Point3d kalibr_point(
    t_cam_imu[0] * imu_point[0] + t_cam_imu[1] * imu_point[1] +
    t_cam_imu[2] * imu_point[2] + t_cam_imu[3],
    t_cam_imu[4] * imu_point[0] + t_cam_imu[5] * imu_point[1] +
    t_cam_imu[6] * imu_point[2] + t_cam_imu[7],
    t_cam_imu[8] * imu_point[0] + t_cam_imu[9] * imu_point[1] +
    t_cam_imu[10] * imu_point[2] + t_cam_imu[11]);

  if (!xv_ros2::rgb_fisheye::projectUndistortedPixel(calibration, kalibr_point,
                                                     pixel))
  {
    return false;
  }

  if (kalibr_depth_meters) {
    *kalibr_depth_meters = kalibr_point.z;
  }
  return true;
}

bool registerTofImagesToRgb(
  const xv_ros2::rgb_fisheye::KalibrCam0Calibration & calibration,
  const xv::Calibration & tof_calibration,
  const xv::DepthImage & depth_image,
  const xv::DepthImage *ir_image,
  double max_valid_depth_meters,
  RegisteredTofImages *output,
  ProjectionDiagnostics *diagnostics,
  std::string *error)
{
  if (!output) {
    setError(error, "output is null");
    return false;
  }
  if (calibration.width <= 0 || calibration.height <= 0) {
    setError(error, "Kalibr output resolution is invalid");
    return false;
  }
  if (!depth_image.data || depth_image.width == 0 || depth_image.height == 0 ||
    (depth_image.type != xv::DepthImage::Type::Depth_16 &&
    depth_image.type != xv::DepthImage::Type::Depth_32))
  {
    setError(error, "ToF depth image is invalid");
    return false;
  }
  if (ir_image &&
    (!ir_image->data || ir_image->type != xv::DepthImage::Type::IR ||
    ir_image->width != depth_image.width ||
    ir_image->height != depth_image.height))
  {
    setError(error, "ToF IR image type or resolution does not match depth");
    return false;
  }

  /** 与运行时深度分辨率最适配的 SDK ToF 模型。 */
  const TofRaytraceModel raytrace_model = selectTofRaytraceModel(
    tof_calibration, depth_image.width, depth_image.height);
  if (!raytrace_model.camera_model && !raytrace_model.pdm_model &&
    !raytrace_model.ucm_model)
  {
    setError(error, "ToF calibration contains no usable camera model");
    return false;
  }

  output->width = static_cast<std::size_t>(calibration.width);
  output->height = static_cast<std::size_t>(calibration.height);
  /** 输出 RGB 网格像素数量。 */
  const std::size_t output_pixel_count = output->width * output->height;
  output->depth_meters.assign(output_pixel_count, 0.0F);
  output->ir.assign(output_pixel_count, 0U);
  /** 输入 IR 像素数组。 */
  const auto *ir_values = ir_image ?
    reinterpret_cast<const std::uint16_t *>(ir_image->data.get()) : nullptr;
  /** 输入 ToF 像素数量。 */
  const std::size_t tof_pixel_count = depth_image.width * depth_image.height;

  for (std::size_t index = 0; index < tof_pixel_count; ++index) {
    /** 当前有效米制深度。 */
    const float depth_meters = depthValueMeters(
      depth_image, index, max_valid_depth_meters);
    if (depth_meters <= 0.0F) {
      if (diagnostics) {
        diagnostics->recordInvalidDepth();
      }
      continue;
    }

    /** 当前运行时 ToF 像素坐标。 */
    const double tof_pixel[2] = {
      static_cast<double>(index % depth_image.width),
      static_cast<double>(index / depth_image.width)};
    /** 当前 ToF 反投影射线。 */
    double tof_ray[3] = {0.0, 0.0, 0.0};
    /** 当前反投影是否成功。 */
    const bool ray_ok = raytraceTofPixel(
      raytrace_model, depth_image.width, depth_image.height,
      tof_pixel, tof_ray);
    if (!ray_ok || std::abs(tof_ray[2]) < 1e-12) {
      if (diagnostics) {
        diagnostics->recordProjectionFailure();
      }
      continue;
    }

    /** ToF 相机坐标系三维点，单位为米。 */
    const xv::Vector3d tof_point = {
      tof_ray[0] * depth_meters / tof_ray[2],
      tof_ray[1] * depth_meters / tof_ray[2],
      static_cast<double>(depth_meters)};
    /** 投影后的 RGB 像素。 */
    cv::Point2d rgb_pixel;
    /** RGB 相机坐标系候选 Z 深度。 */
    double rgb_depth_meters = 0.0;
    if (!projectTofPointToKalibrUndistortedPixel(
        calibration, tof_calibration.pose, tof_point, &rgb_pixel,
        &rgb_depth_meters) || !std::isfinite(rgb_depth_meters) ||
      rgb_depth_meters <= 0.0)
    {
      if (diagnostics) {
        diagnostics->recordProjectionFailure();
      }
      continue;
    }

    /** 输出 RGB 像素 x 坐标。 */
    const int rgb_x = static_cast<int>(std::lround(rgb_pixel.x));
    /** 输出 RGB 像素 y 坐标。 */
    const int rgb_y = static_cast<int>(std::lround(rgb_pixel.y));
    if (rgb_x < 0 || rgb_y < 0 ||
      rgb_x >= static_cast<int>(output->width) ||
      rgb_y >= static_cast<int>(output->height))
    {
      if (diagnostics) {
        diagnostics->recordOutOfBoundsProjection();
      }
      continue;
    }

    /** 输出图像行优先索引。 */
    const std::size_t rgb_index = static_cast<std::size_t>(rgb_y) *
      output->width + static_cast<std::size_t>(rgb_x);
    /** 候选 Z 深度。 */
    const float candidate_depth = static_cast<float>(rgb_depth_meters);
    /** 输出像素是否已有深度。 */
    const bool has_depth = output->depth_meters[rgb_index] > 0.0F;
    if (!has_depth || candidate_depth < output->depth_meters[rgb_index]) {
      output->depth_meters[rgb_index] = candidate_depth;
      output->ir[rgb_index] = ir_values ? ir_values[index] : 0U;
      if (diagnostics) {
        diagnostics->recordStoredDepth(has_depth);
      }
    } else if (diagnostics) {
      diagnostics->recordDiscardedOverlappedDepth();
    }
  }
  return true;
}

bool createVirtualRgbdModel(
  const xv_ros2::rgb_fisheye::KalibrCam0Calibration & calibration,
  const xv::Calibration & tof_calibration,
  std::size_t tof_width,
  std::size_t tof_height,
  const cv::Mat & rgb_remap_x,
  const cv::Mat & rgb_remap_y,
  VirtualRgbdModel *output,
  std::string *error)
{
  if (!output) {
    setError(error, "output is null");
    return false;
  }
  if (tof_width < 2U || tof_height < 2U) {
    setError(error, "runtime ToF resolution is too small");
    return false;
  }
  if (calibration.width <= 1 || calibration.height <= 1 ||
    rgb_remap_x.cols != calibration.width ||
    rgb_remap_x.rows != calibration.height ||
    rgb_remap_y.cols != calibration.width ||
    rgb_remap_y.rows != calibration.height)
  {
    setError(error, "RGB remap does not match Kalibr resolution");
    return false;
  }

  /** 运行时分辨率对应的 ToF 投影模型。 */
  const TofRaytraceModel tof_model = selectTofRaytraceModel(
    tof_calibration, tof_width, tof_height);
  if (!tof_model.camera_model && !tof_model.pdm_model &&
    !tof_model.ucm_model)
  {
    setError(error, "ToF calibration contains no usable camera model");
    return false;
  }

  /** Kalibr 校正 RGB 图的有效 remap 掩码。 */
  std::vector<std::uint8_t> rgb_valid_mask;
  if (!buildRgbRemapValidityMask(
      rgb_remap_x, rgb_remap_y,
      static_cast<std::size_t>(calibration.width),
      static_cast<std::size_t>(calibration.height), &rgb_valid_mask))
  {
    setError(error, "RGB fisheye remap is invalid");
    return false;
  }

  /** ToF 原生视场最小归一化 x。 */
  double tof_min_x = std::numeric_limits<double>::infinity();
  /** ToF 原生视场最大归一化 x。 */
  double tof_max_x = -std::numeric_limits<double>::infinity();
  /** ToF 原生视场最小归一化 y。 */
  double tof_min_y = std::numeric_limits<double>::infinity();
  /** ToF 原生视场最大归一化 y。 */
  double tof_max_y = -std::numeric_limits<double>::infinity();
  /** 成功反投影的 ToF 像素数量。 */
  std::size_t valid_tof_rays = 0U;
  for (std::size_t y = 0; y < tof_height; ++y) {
    for (std::size_t x = 0; x < tof_width; ++x) {
      /** 当前运行时 ToF 像素。 */
      const double tof_pixel[2] = {
        static_cast<double>(x), static_cast<double>(y)};
      /** 当前 ToF 反投影射线。 */
      double tof_ray[3] = {0.0, 0.0, 0.0};
      if (!raytraceTofPixel(
          tof_model, tof_width, tof_height, tof_pixel, tof_ray) ||
        !std::isfinite(tof_ray[0]) || !std::isfinite(tof_ray[1]) ||
        !std::isfinite(tof_ray[2]) || tof_ray[2] <= 1e-12)
      {
        continue;
      }
      /** 当前归一化射线 x。 */
      const double normalized_x = tof_ray[0] / tof_ray[2];
      /** 当前归一化射线 y。 */
      const double normalized_y = tof_ray[1] / tof_ray[2];
      tof_min_x = std::min(tof_min_x, normalized_x);
      tof_max_x = std::max(tof_max_x, normalized_x);
      tof_min_y = std::min(tof_min_y, normalized_y);
      tof_max_y = std::max(tof_max_y, normalized_y);
      ++valid_tof_rays;
    }
  }
  if (valid_tof_rays == 0U || !std::isfinite(tof_min_x) ||
    !std::isfinite(tof_max_x) || !std::isfinite(tof_min_y) ||
    !std::isfinite(tof_max_y))
  {
    setError(error, "ToF camera model cannot raytrace the runtime image");
    return false;
  }

  /** 覆盖 ToF 原生视场的临时虚拟相机。 */
  VirtualRgbdModel provisional_model;
  provisional_model.width = tof_width;
  provisional_model.height = tof_height;
  provisional_model.min_x = tof_min_x;
  provisional_model.max_x = tof_max_x;
  provisional_model.min_y = tof_min_y;
  provisional_model.max_y = tof_max_y;
  provisional_model.rgb_width = static_cast<std::size_t>(calibration.width);
  provisional_model.rgb_height = static_cast<std::size_t>(calibration.height);
  if (!computeVirtualIntrinsics(
      provisional_model.width, provisional_model.height,
      provisional_model.min_x, provisional_model.max_x,
      provisional_model.min_y, provisional_model.max_y,
      &provisional_model.fx, &provisional_model.fy,
      &provisional_model.cx, &provisional_model.cy))
  {
    setError(error, "ToF normalized field of view is invalid");
    return false;
  }

  /** ToF 与 RGB 旋转投影同时有效的临时网格掩码。 */
  std::vector<std::uint8_t> common_mask(tof_width * tof_height, 0U);
  for (std::size_t index = 0; index < common_mask.size(); ++index) {
    /** 临时虚拟网格 x。 */
    const std::size_t x = index % tof_width;
    /** 临时虚拟网格 y。 */
    const std::size_t y = index / tof_width;
    /** 当前归一化虚拟射线。 */
    const double ray[3] = {
      (static_cast<double>(x) - provisional_model.cx) /
      provisional_model.fx,
      (static_cast<double>(y) - provisional_model.cy) /
      provisional_model.fy,
      1.0};
    /** 当前射线投影到运行时 ToF 的像素。 */
    double tof_pixel[2] = {0.0, 0.0};
    if (!projectTofRay(tof_model, tof_width, tof_height, ray, tof_pixel)) {
      continue;
    }
    /** 最近邻 ToF 源像素 x。 */
    const long source_x = static_cast<long>(std::floor(tof_pixel[0] + 0.5));
    /** 最近邻 ToF 源像素 y。 */
    const long source_y = static_cast<long>(std::floor(tof_pixel[1] + 0.5));
    if (source_x < 0 || source_y < 0 ||
      source_x >= static_cast<long>(tof_width) ||
      source_y >= static_cast<long>(tof_height))
    {
      continue;
    }
    /** 当前射线的 RGB 旋转回退坐标。 */
    cv::Point2d fallback_pixel;
    if (projectTofRayByRotation(
        calibration, tof_calibration.pose, ray, &fallback_pixel) &&
      isBilinearSampleValid(
        fallback_pixel, provisional_model.rgb_width,
        provisional_model.rgb_height, rgb_valid_mask))
    {
      common_mask[index] = 1U;
    }
  }

  /** 最大公共全有效矩形。 */
  const MaskRectangle common_rectangle = largestValidRectangle(
    common_mask, tof_width, tof_height);
  /** 公共矩形宽度。 */
  const std::size_t rectangle_width = common_rectangle.area > 0U ?
    common_rectangle.right - common_rectangle.left + 1U : 0U;
  /** 公共矩形高度。 */
  const std::size_t rectangle_height = common_rectangle.area > 0U ?
    common_rectangle.bottom - common_rectangle.top + 1U : 0U;
  if (rectangle_width < 2U || rectangle_height < 2U) {
    setError(error, "RGB and ToF have no usable common field of view");
    return false;
  }

  /** 公共视场候选模型。 */
  VirtualRgbdModel candidate;
  candidate.width = tof_width;
  candidate.height = tof_height;
  candidate.min_x =
    (static_cast<double>(common_rectangle.left) - provisional_model.cx) /
    provisional_model.fx;
  candidate.max_x =
    (static_cast<double>(common_rectangle.right) - provisional_model.cx) /
    provisional_model.fx;
  candidate.min_y =
    (static_cast<double>(common_rectangle.top) - provisional_model.cy) /
    provisional_model.fy;
  candidate.max_y =
    (static_cast<double>(common_rectangle.bottom) - provisional_model.cy) /
    provisional_model.fy;
  candidate.rgb_width = provisional_model.rgb_width;
  candidate.rgb_height = provisional_model.rgb_height;

  /** 逐次内收时保持不变的公共视场中心 x。 */
  const double center_x = (candidate.min_x + candidate.max_x) * 0.5;
  /** 逐次内收时保持不变的公共视场中心 y。 */
  const double center_y = (candidate.min_y + candidate.max_y) * 0.5;
  /** 初始公共视场半宽。 */
  const double initial_half_width = (candidate.max_x - candidate.min_x) * 0.5;
  /** 初始公共视场半高。 */
  const double initial_half_height = (candidate.max_y - candidate.min_y) * 0.5;
  /** 最大视场内收次数。 */
  constexpr std::size_t kMaxShrinkIterations = 100U;
  /** 每次视场边界保留比例。 */
  constexpr double kShrinkFactor = 0.99;
  for (std::size_t iteration = 0; iteration <= kMaxShrinkIterations;
    ++iteration)
  {
    /** 当前迭代视场缩放比例。 */
    const double scale = std::pow(kShrinkFactor, static_cast<double>(iteration));
    candidate.min_x = center_x - initial_half_width * scale;
    candidate.max_x = center_x + initial_half_width * scale;
    candidate.min_y = center_y - initial_half_height * scale;
    candidate.max_y = center_y + initial_half_height * scale;
    if (buildVirtualMappings(
        calibration, tof_calibration, tof_model, rgb_valid_mask, &candidate))
    {
      *output = std::move(candidate);
      return true;
    }
  }

  setError(error, "virtual RGBD mapping could not reach 100% RGB validity");
  return false;
}

bool registerRgbToVirtualGrid(
  const VirtualRgbdModel & model,
  const xv_ros2::rgb_fisheye::KalibrCam0Calibration & calibration,
  const xv::Calibration & tof_calibration,
  const xv::DepthImage & depth_image,
  const xv::DepthImage & ir_image,
  const cv::Mat & undistorted_rgb,
  double max_valid_depth_meters,
  RgbToTofImages *output,
  std::string *error)
{
  if (!output) {
    setError(error, "output is null");
    return false;
  }
  /** 虚拟网格像素数量。 */
  const std::size_t pixel_count = model.width * model.height;
  if (model.width == 0U || model.height == 0U ||
    model.tof_source_indices.size() != pixel_count ||
    model.virtual_rays.size() != pixel_count ||
    model.fallback_rgb_pixels.size() != pixel_count ||
    model.rgb_valid_mask.size() != model.rgb_width * model.rgb_height)
  {
    setError(error, "virtual RGBD model is incomplete");
    return false;
  }
  if (!depth_image.data || depth_image.width != model.width ||
    depth_image.height != model.height ||
    (depth_image.type != xv::DepthImage::Type::Depth_16 &&
    depth_image.type != xv::DepthImage::Type::Depth_32))
  {
    setError(error, "ToF depth image does not match virtual model");
    return false;
  }
  if (!ir_image.data || ir_image.type != xv::DepthImage::Type::IR ||
    ir_image.width != model.width || ir_image.height != model.height)
  {
    setError(error, "ToF IR image does not match virtual model");
    return false;
  }
  if (undistorted_rgb.empty() || undistorted_rgb.type() != CV_8UC3 ||
    undistorted_rgb.cols != static_cast<int>(model.rgb_width) ||
    undistorted_rgb.rows != static_cast<int>(model.rgb_height) ||
    calibration.width != undistorted_rgb.cols ||
    calibration.height != undistorted_rgb.rows)
  {
    setError(error, "undistorted RGB image does not match virtual model");
    return false;
  }

  output->width = model.width;
  output->height = model.height;
  output->rgb.assign(pixel_count * 3U, 0U);
  output->depth_meters.assign(pixel_count, 0.0F);
  output->ir.assign(pixel_count, 0U);
  /** 原始 ToF IR 强度数组。 */
  const auto *ir_values =
    reinterpret_cast<const std::uint16_t *>(ir_image.data.get());
  for (std::size_t index = 0; index < pixel_count; ++index) {
    /** 当前虚拟像素对应的原始 ToF 索引。 */
    const std::size_t source_index = model.tof_source_indices[index];
    if (source_index >= pixel_count) {
      setError(error, "virtual RGBD model contains an invalid ToF source index");
      return false;
    }
    /** 当前虚拟像素的米制深度。 */
    const float depth_meters = depthValueMeters(
      depth_image, source_index, max_valid_depth_meters);
    output->depth_meters[index] = depth_meters;
    output->ir[index] = ir_values[source_index];

    /** 当前像素的预计算 RGB 旋转回退坐标。 */
    cv::Point2d rgb_pixel(
      model.fallback_rgb_pixels[index].x,
      model.fallback_rgb_pixels[index].y);
    if (depth_meters > 0.0F) {
      /** 当前虚拟像素的 ToF 坐标系射线。 */
      const cv::Point3f & ray = model.virtual_rays[index];
      /** 当前虚拟像素的 ToF 三维点，单位为米。 */
      const xv::Vector3d tof_point = {
        static_cast<double>(ray.x) * depth_meters,
        static_cast<double>(ray.y) * depth_meters,
        static_cast<double>(depth_meters)};
      /** 完整旋转和平移外参得到的 RGB 像素。 */
      cv::Point2d projected_pixel;
      if (projectTofPointToKalibrUndistortedPixel(
          calibration, tof_calibration.pose, tof_point, &projected_pixel) &&
        isBilinearSampleValid(
          projected_pixel, model.rgb_width, model.rgb_height,
          model.rgb_valid_mask))
      {
        rgb_pixel = projected_pixel;
      }
    }

    if (!isBilinearSampleValid(
        rgb_pixel, model.rgb_width, model.rgb_height,
        model.rgb_valid_mask))
    {
      setError(error, "virtual RGBD fallback mapping lost RGB validity");
      return false;
    }
    /** 当前虚拟像素的 RGB 双线性采样值。 */
    const cv::Vec3b rgb_value = sampleRgbBilinear(
      undistorted_rgb, rgb_pixel);
    output->rgb[index * 3U] = rgb_value[0];
    output->rgb[index * 3U + 1U] = rgb_value[1];
    output->rgb[index * 3U + 2U] = rgb_value[2];
  }
  return true;
}

bool registerRgbToTofGrid(
  const xv_ros2::rgb_fisheye::KalibrCam0Calibration & calibration,
  const xv::Calibration & tof_calibration,
  const xv::DepthImage & depth_image,
  const xv::DepthImage & ir_image,
  const cv::Mat & undistorted_rgb,
  double max_valid_depth_meters,
  RgbToTofImages *output,
  std::string *error)
{
  if (!output) {
    setError(error, "output is null");
    return false;
  }
  if (undistorted_rgb.empty() || undistorted_rgb.type() != CV_8UC3 ||
    undistorted_rgb.cols != calibration.width ||
    undistorted_rgb.rows != calibration.height)
  {
    setError(error, "undistorted RGB image does not match Kalibr resolution");
    return false;
  }
  if (!depth_image.data || depth_image.width == 0 || depth_image.height == 0 ||
    (depth_image.type != xv::DepthImage::Type::Depth_16 &&
    depth_image.type != xv::DepthImage::Type::Depth_32))
  {
    setError(error, "ToF depth image is invalid");
    return false;
  }
  if (!ir_image.data || ir_image.type != xv::DepthImage::Type::IR ||
    ir_image.width != depth_image.width ||
    ir_image.height != depth_image.height)
  {
    setError(error, "ToF IR image type or resolution does not match depth");
    return false;
  }

  /** 运行时深度分辨率对应的 SDK ToF 模型。 */
  const TofRaytraceModel raytrace_model = selectTofRaytraceModel(
    tof_calibration, depth_image.width, depth_image.height);
  if (!raytrace_model.camera_model && !raytrace_model.pdm_model &&
    !raytrace_model.ucm_model)
  {
    setError(error, "ToF calibration contains no usable camera model");
    return false;
  }

  output->width = depth_image.width;
  output->height = depth_image.height;
  /** ToF 网格像素数量。 */
  const std::size_t pixel_count = output->width * output->height;
  output->rgb.assign(pixel_count * 3U, 0U);
  output->depth_meters.assign(pixel_count, 0.0F);
  /** 原始 ToF IR 数据。 */
  const auto *ir_values =
    reinterpret_cast<const std::uint16_t *>(ir_image.data.get());
  output->ir.assign(ir_values, ir_values + pixel_count);

  for (std::size_t index = 0; index < pixel_count; ++index) {
    /** 当前 ToF 深度，单位为米。 */
    const float depth_meters = depthValueMeters(
      depth_image, index, max_valid_depth_meters);
    output->depth_meters[index] = depth_meters;
    /** 当前 ToF 像素坐标。 */
    const double tof_pixel[2] = {
      static_cast<double>(index % output->width),
      static_cast<double>(index / output->width)};
    /** 当前 ToF 相机射线。 */
    double tof_ray[3] = {0.0, 0.0, 0.0};
    if (!raytraceTofPixel(
        raytrace_model, output->width, output->height, tof_pixel, tof_ray) ||
      std::abs(tof_ray[2]) < 1e-12)
    {
      continue;
    }

    /** 校正 RGB 图中的采样坐标。 */
    cv::Point2d rgb_pixel;
    /** 是否得到有效 RGB 采样位置。 */
    bool projected = false;
    if (depth_meters > 0.0F) {
      /** 当前 ToF 三维点，单位为米。 */
      const xv::Vector3d tof_point = {
        tof_ray[0] * depth_meters / tof_ray[2],
        tof_ray[1] * depth_meters / tof_ray[2],
        static_cast<double>(depth_meters)};
      projected = projectTofPointToKalibrUndistortedPixel(
        calibration, tof_calibration.pose, tof_point, &rgb_pixel);
    }
    if (!projected) {
      projected = projectTofRayByRotation(
        calibration, tof_calibration.pose, tof_ray, &rgb_pixel);
    }
    if (!projected) {
      continue;
    }

    /** 当前 ToF 像素对应的双线性 RGB 值。 */
    const cv::Vec3b rgb_value = sampleRgbBilinear(
      undistorted_rgb, rgb_pixel);
    output->rgb[index * 3U] = rgb_value[0];
    output->rgb[index * 3U + 1U] = rgb_value[1];
    output->rgb[index * 3U + 2U] = rgb_value[2];
  }
  return true;
}

} // namespace rgb_registered
} // namespace xv_ros2
