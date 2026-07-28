/**
 * @file factory_rgbd_utils.cpp
 * @brief 实现 factory RGB-D 标定解析、ToF 到 RGB 深度对齐和时间匹配工具。
 */

#include "factory_rgbd_utils.h"

#include "rgb_registered_utils.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>

namespace xv_ros2
{
namespace factory_rgbd
{
namespace
{
/**
 * @brief 写入错误信息。
 * @param error 错误输出指针，可为空。
 * @param message 错误说明。
 */
void setError(std::string *error, const std::string &message)
{
    if (error)
    {
        *error = message;
    }
}

/**
 * @brief 判断浮点数是否为有限值。
 * @param value 待检查数值。
 * @return 有限时返回 true。
 */
bool isFinite(double value)
{
    return std::isfinite(value);
}

/**
 * @brief 从 YAML 节点读取必需数值。
 * @param node YAML 节点。
 * @param key 字段名。
 * @param value 输出数值。
 * @param error 输出错误说明；可为空。
 * @return 读取成功返回 true。
 */
template <typename T>
bool readRequiredScalar(const YAML::Node &node,
                        const std::string &key,
                        T &value,
                        std::string *error)
{
    if (!node[key])
    {
        setError(error, "missing required key: " + key);
        return false;
    }

    try
    {
        value = node[key].as<T>();
    }
    catch (const YAML::Exception &exception)
    {
        setError(error, "invalid key " + key + ": " + exception.what());
        return false;
    }

    return true;
}

/**
 * @brief 校验 RGB SEUCM 内参是否可用于投影。
 * @param intrinsics RGB SEUCM 内参。
 * @return 可用返回 true。
 */
bool isValidRgbIntrinsics(const RgbSeucmIntrinsics &intrinsics)
{
    return intrinsics.width > 0 &&
           intrinsics.height > 0 &&
           intrinsics.fx > 0.0 &&
           intrinsics.fy > 0.0 &&
           intrinsics.beta > 0.0 &&
           isFinite(intrinsics.u0) &&
           isFinite(intrinsics.v0) &&
           isFinite(intrinsics.eu) &&
           isFinite(intrinsics.ev) &&
           isFinite(intrinsics.alpha) &&
           isFinite(intrinsics.beta);
}

/**
 * @brief 校验 ToF PDCM 内参是否可用于反投影。
 * @param intrinsics ToF PDCM 内参。
 * @return 可用返回 true。
 */
bool isValidTofIntrinsics(const TofPdcmIntrinsics &intrinsics)
{
    if (intrinsics.width <= 0 ||
        intrinsics.height <= 0 ||
        intrinsics.fx <= 0.0 ||
        intrinsics.fy <= 0.0 ||
        !isFinite(intrinsics.u0) ||
        !isFinite(intrinsics.v0))
    {
        return false;
    }

    return std::all_of(intrinsics.distor.begin(), intrinsics.distor.end(), isFinite);
}

/**
 * @brief 校验 T_rgb_tof 是否为有限齐次矩阵。
 * @param matrix row-major 齐次矩阵。
 * @return 可用返回 true。
 */
bool isValidTransform(const std::array<double, 16> &matrix)
{
    /** 齐次矩阵最后一行期望值。 */
    constexpr std::array<double, 4> kExpectedLastRow{0.0, 0.0, 0.0, 1.0};
    if (!std::all_of(matrix.begin(), matrix.end(), isFinite))
    {
        return false;
    }

    for (std::size_t index = 0; index < kExpectedLastRow.size(); ++index)
    {
        if (std::abs(matrix[12 + index] - kExpectedLastRow[index]) > 1e-9)
        {
            return false;
        }
    }

    return true;
}

/**
 * @brief 将 T_rgb_tof 应用于 ToF 坐标点。
 * @param transform row-major 齐次矩阵。
 * @param point_tof ToF 坐标点。
 * @return RGB 坐标点。
 */
std::array<double, 3> transformPoint(const std::array<double, 16> &transform,
                                     const std::array<double, 3> &point_tof)
{
    return {
        transform[0] * point_tof[0] + transform[1] * point_tof[1] +
            transform[2] * point_tof[2] + transform[3],
        transform[4] * point_tof[0] + transform[5] * point_tof[1] +
            transform[6] * point_tof[2] + transform[7],
        transform[8] * point_tof[0] + transform[9] * point_tof[1] +
            transform[10] * point_tof[2] + transform[11]};
}

}  // namespace

std::optional<FactoryCalibration> loadFactoryCalibration(const std::string &path,
                                                         std::string *error)
{
    /** 解析后的 factory 标定。 */
    FactoryCalibration calibration;
    try
    {
        /** factory YAML 根节点。 */
        const YAML::Node root = YAML::LoadFile(path);
        /** RGB 内参节点。 */
        const YAML::Node rgb = root["rgb"]["intrinsics"];
        /** ToF 内参节点。 */
        const YAML::Node tof = root["tof"]["intrinsics"];
        /** T_rgb_tof 外参矩阵节点。 */
        const YAML::Node transform = root["extrinsics"]["T_rgb_tof"]["matrix"];
        if (!rgb || !tof || !transform)
        {
            setError(error, "missing rgb/tof intrinsics or T_rgb_tof matrix");
            return std::nullopt;
        }

        /** RGB 模型名。 */
        std::string rgb_model;
        /** ToF 模型名。 */
        std::string tof_model;
        if (!readRequiredScalar(rgb, "model", rgb_model, error) ||
            !readRequiredScalar(tof, "model", tof_model, error))
        {
            return std::nullopt;
        }
        if (rgb_model != "SEUCM" || tof_model != "PDCM0")
        {
            setError(error, "unsupported factory calibration model");
            return std::nullopt;
        }

        if (!readRequiredScalar(rgb, "width", calibration.rgb.width, error) ||
            !readRequiredScalar(rgb, "height", calibration.rgb.height, error) ||
            !readRequiredScalar(rgb, "fx", calibration.rgb.fx, error) ||
            !readRequiredScalar(rgb, "fy", calibration.rgb.fy, error) ||
            !readRequiredScalar(rgb, "u0", calibration.rgb.u0, error) ||
            !readRequiredScalar(rgb, "v0", calibration.rgb.v0, error) ||
            !readRequiredScalar(rgb, "eu", calibration.rgb.eu, error) ||
            !readRequiredScalar(rgb, "ev", calibration.rgb.ev, error) ||
            !readRequiredScalar(rgb, "alpha", calibration.rgb.alpha, error) ||
            !readRequiredScalar(rgb, "beta", calibration.rgb.beta, error) ||
            !readRequiredScalar(tof, "width", calibration.tof.width, error) ||
            !readRequiredScalar(tof, "height", calibration.tof.height, error) ||
            !readRequiredScalar(tof, "fx", calibration.tof.fx, error) ||
            !readRequiredScalar(tof, "fy", calibration.tof.fy, error) ||
            !readRequiredScalar(tof, "u0", calibration.tof.u0, error) ||
            !readRequiredScalar(tof, "v0", calibration.tof.v0, error))
        {
            return std::nullopt;
        }

        /** ToF 畸变参数节点。 */
        const YAML::Node distor = tof["distor"];
        if (!distor || !distor.IsSequence() || distor.size() != calibration.tof.distor.size())
        {
            setError(error, "invalid tof distor");
            return std::nullopt;
        }
        for (std::size_t index = 0; index < calibration.tof.distor.size(); ++index)
        {
            calibration.tof.distor[index] = distor[index].as<double>();
        }

        if (!transform.IsSequence() || transform.size() != 4)
        {
            setError(error, "invalid T_rgb_tof matrix rows");
            return std::nullopt;
        }
        for (std::size_t row = 0; row < 4; ++row)
        {
            if (!transform[row].IsSequence() || transform[row].size() != 4)
            {
                setError(error, "invalid T_rgb_tof matrix columns");
                return std::nullopt;
            }
            for (std::size_t column = 0; column < 4; ++column)
            {
                calibration.t_rgb_tof[row * 4 + column] =
                    transform[row][column].as<double>();
            }
        }
    }
    catch (const YAML::Exception &exception)
    {
        setError(error, exception.what());
        return std::nullopt;
    }

    if (!isValidRgbIntrinsics(calibration.rgb) ||
        !isValidTofIntrinsics(calibration.tof) ||
        !isValidTransform(calibration.t_rgb_tof))
    {
        setError(error, "factory calibration values are invalid");
        return std::nullopt;
    }

    return calibration;
}

bool backProjectTofPdcm(const TofPdcmIntrinsics &intrinsics,
                        double u,
                        double v,
                        double depth_meters,
                        std::array<double, 3> &point_tof)
{
    if (!isValidTofIntrinsics(intrinsics) ||
        depth_meters <= 0.0 ||
        !isFinite(depth_meters))
    {
        return false;
    }

    /** 去畸变迭代的归一化 x 坐标。 */
    double x = (u - intrinsics.u0) / intrinsics.fx;
    /** 去畸变迭代的归一化 y 坐标。 */
    double y = (v - intrinsics.v0) / intrinsics.fy;
    /** 初始归一化 x 坐标。 */
    const double x0 = x;
    /** 初始归一化 y 坐标。 */
    const double y0 = y;
    /** 径向畸变 k1。 */
    const double k1 = intrinsics.distor[0];
    /** 径向畸变 k2。 */
    const double k2 = intrinsics.distor[1];
    /** 切向畸变 p1。 */
    const double p1 = intrinsics.distor[2];
    /** 切向畸变 p2。 */
    const double p2 = intrinsics.distor[3];
    /** 径向畸变 k3。 */
    const double k3 = intrinsics.distor[4];

    for (int iteration = 0; iteration < 5; ++iteration)
    {
        /** 半径平方。 */
        const double r2 = x * x + y * y;
        /** 半径四次方。 */
        const double r4 = r2 * r2;
        /** 半径六次方。 */
        const double r6 = r4 * r2;
        /** 径向畸变比例。 */
        const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        /** 切向畸变 x 分量。 */
        const double delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        /** 切向畸变 y 分量。 */
        const double delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        if (std::abs(radial) < 1e-12)
        {
            return false;
        }

        x = (x0 - delta_x) / radial;
        y = (y0 - delta_y) / radial;
    }

    point_tof = {x * depth_meters, y * depth_meters, depth_meters};
    return std::all_of(point_tof.begin(), point_tof.end(), isFinite);
}

bool projectRgbSeucm(const RgbSeucmIntrinsics &intrinsics,
                     const std::array<double, 3> &point_rgb,
                     std::array<double, 2> &pixel)
{
    if (!isValidRgbIntrinsics(intrinsics) ||
        point_rgb[2] <= 0.0 ||
        !std::all_of(point_rgb.begin(), point_rgb.end(), isFinite))
    {
        return false;
    }

    /** RGB 坐标点 x。 */
    const double x = point_rgb[0];
    /** RGB 坐标点 y。 */
    const double y = point_rgb[1];
    /** RGB 坐标点 z。 */
    const double z = point_rgb[2];
    /** SEUCM 广义距离。 */
    const double distance = std::sqrt(intrinsics.beta * (x * x + y * y) + z * z);
    /** SEUCM 投影分母。 */
    const double denominator = intrinsics.alpha * distance +
                               (1.0 - intrinsics.alpha) * z;
    if (denominator <= 0.0 || !isFinite(denominator))
    {
        return false;
    }

    pixel = {
        intrinsics.fx * (x / denominator) + intrinsics.u0,
        intrinsics.fy * (y / denominator) + intrinsics.v0};
    return std::all_of(pixel.begin(), pixel.end(), isFinite);
}

bool alignDepthToRgb(const FactoryCalibration &calibration,
                     const xv::DepthImage &depth_image,
                     std::vector<float> &aligned_depth)
{
    if (!isValidRgbIntrinsics(calibration.rgb) ||
        !isValidTofIntrinsics(calibration.tof) ||
        !isValidTransform(calibration.t_rgb_tof) ||
        !depth_image.data ||
        static_cast<int>(depth_image.width) != calibration.tof.width ||
        static_cast<int>(depth_image.height) != calibration.tof.height)
    {
        aligned_depth.clear();
        return false;
    }

    /** RGB 输出像素数量。 */
    const std::size_t rgb_pixel_count =
        static_cast<std::size_t>(calibration.rgb.width) *
        static_cast<std::size_t>(calibration.rgb.height);
    /** ToF 输入像素数量。 */
    const std::size_t tof_pixel_count =
        static_cast<std::size_t>(depth_image.width) *
        static_cast<std::size_t>(depth_image.height);
    aligned_depth.assign(rgb_pixel_count, 0.0F);

    for (std::size_t index = 0; index < tof_pixel_count; ++index)
    {
        /** 当前 ToF 深度，单位为米。 */
        const float depth_meters =
            xv_ros2::rgb_registered::depthValueMeters(depth_image, index);
        if (depth_meters <= 0.0F)
        {
            continue;
        }

        /** 当前 ToF 像素横坐标。 */
        const double tof_u = static_cast<double>(index % depth_image.width);
        /** 当前 ToF 像素纵坐标。 */
        const double tof_v = static_cast<double>(index / depth_image.width);
        /** ToF 坐标点。 */
        std::array<double, 3> point_tof{};
        if (!backProjectTofPdcm(calibration.tof, tof_u, tof_v, depth_meters, point_tof))
        {
            continue;
        }

        /** RGB 坐标点。 */
        const std::array<double, 3> point_rgb =
            transformPoint(calibration.t_rgb_tof, point_tof);
        /** RGB 像素坐标。 */
        std::array<double, 2> rgb_pixel{};
        if (!projectRgbSeucm(calibration.rgb, point_rgb, rgb_pixel))
        {
            continue;
        }

        /** 四舍五入后的 RGB 横坐标。 */
        const int rgb_x = static_cast<int>(std::lround(rgb_pixel[0]));
        /** 四舍五入后的 RGB 纵坐标。 */
        const int rgb_y = static_cast<int>(std::lround(rgb_pixel[1]));
        if (rgb_x < 0 ||
            rgb_y < 0 ||
            rgb_x >= calibration.rgb.width ||
            rgb_y >= calibration.rgb.height)
        {
            continue;
        }

        /** RGB 输出数组索引。 */
        const std::size_t rgb_index =
            static_cast<std::size_t>(rgb_y) * static_cast<std::size_t>(calibration.rgb.width) +
            static_cast<std::size_t>(rgb_x);
        /** RGB 坐标系候选深度，单位为米。 */
        const float candidate_depth = static_cast<float>(point_rgb[2]);
        if (candidate_depth > 0.0F &&
            (aligned_depth[rgb_index] == 0.0F || candidate_depth < aligned_depth[rgb_index]))
        {
            aligned_depth[rgb_index] = candidate_depth;
        }
    }

    return true;
}

std::optional<std::size_t> findNearestTimestampIndex(const std::vector<double> &timestamps,
                                                     double target_timestamp,
                                                     double tolerance_sec)
{
    if (timestamps.empty() || tolerance_sec < 0.0 || !isFinite(target_timestamp))
    {
        return std::nullopt;
    }

    /** 当前最佳匹配索引。 */
    std::optional<std::size_t> best_index;
    /** 当前最小时间差。 */
    double best_diff = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < timestamps.size(); ++index)
    {
        if (!isFinite(timestamps[index]))
        {
            continue;
        }

        /** 当前候选时间差。 */
        const double diff = std::abs(timestamps[index] - target_timestamp);
        if (diff < best_diff)
        {
            best_diff = diff;
            best_index = index;
        }
    }

    if (!best_index.has_value() || best_diff > tolerance_sec)
    {
        return std::nullopt;
    }

    return best_index;
}

}  // namespace factory_rgbd
}  // namespace xv_ros2
