/**
 * @file test_xv_ros2_node_stubs.cpp
 * @brief 为 xv_ros2_node 单测提供 XV SDK 与设备封装的最小测试桩。
 */

#include "xv_dev_wrapper.h"
#include "xv_ros2_node.h"

#include <map>
#include <memory>
#include <string>
#include <utility>

namespace xv
{
/**
 * @brief 测试桩：构造单位变换，满足标定对象默认构造。
 */
Transform::Transform()
    : details::Transform_<double>({0.0, 0.0, 0.0},
                                  {1.0, 0.0, 0.0,
                                   0.0, 1.0, 0.0,
                                   0.0, 0.0, 1.0})
{
}

/**
 * @brief 测试桩：忽略 SDK 日志级别设置。
 * @param level SDK 日志级别。
 */
void setLogLevel(LogLevel level)
{
    (void)level;
}

/**
 * @brief 测试桩：返回空设备列表，避免单测依赖真实硬件和 SDK 运行库。
 * @param timeout 设备枚举超时时间。
 * @param desc SDK 初始化描述。
 * @param stop 是否停止枚举的标志。
 * @param mode SLAM 启动模式。
 * @param support 设备支持能力。
 * @return 空设备映射表。
 */
std::map<std::string, std::shared_ptr<Device>> getDevices(double timeout,
                                                          const std::string &desc,
                                                          bool *stop,
                                                          SlamStartMode mode,
                                                          DeviceSupport support)
{
    (void)timeout;
    (void)desc;
    (void)stop;
    (void)mode;
    (void)support;
    return {};
}
}  // namespace xv

/**
 * @brief 构造测试用设备封装对象。
 * @param node ROS 2 节点指针。
 * @param device SDK 设备指针。
 * @param sn 设备序列号。
 * @param type 设备类型。
 */
xv_dev_wrapper::xv_dev_wrapper(xvision_ros2_node *node,
                               std::shared_ptr<xv::Device> device,
                               std::string sn,
                               int type)
    : m_node(node),
      m_device(std::move(device)),
      m_sn(std::move(sn)),
      m_type(type),
      m_rgbd_rect_pool(0),
      m_imu_pool(0)
{
}

/**
 * @brief 销毁测试用设备封装对象。
 */
xv_dev_wrapper::~xv_dev_wrapper() = default;

/**
 * @brief 测试桩：忽略 SDK 流启动。
 */
void xv_dev_wrapper::startDeviceStreams()
{
}

/**
 * @brief 测试桩：忽略 IMU 频率配置。
 * @param flag 是否启用高频 IMU。
 */
void xv_dev_wrapper::set_imu_flag(bool flag)
{
    (void)flag;
}

/**
 * @brief 测试桩：模拟启动方向流失败。
 * @return 固定返回 false。
 */
bool xv_dev_wrapper::startImuOri(void)
{
    return false;
}

/**
 * @brief 测试桩：模拟停止方向流失败。
 * @return 固定返回 false。
 */
bool xv_dev_wrapper::stopImuOri(void)
{
    return false;
}

/**
 * @brief 测试桩：模拟获取方向数据失败。
 * @param oriStamped 输出方向数据。
 * @param duration 预测时间。
 * @return 固定返回 false。
 */
bool xv_dev_wrapper::getImuOri(rosOrientationStamped &oriStamped,
                               const builtin_interfaces::msg::Duration &duration)
{
    (void)oriStamped;
    (void)duration;
    return false;
}

/**
 * @brief 测试桩：模拟按时间戳获取方向数据失败。
 * @param oriStamped 输出方向数据。
 * @param time 查询时间戳。
 * @return 固定返回 false。
 */
bool xv_dev_wrapper::getImuOriAt(rosOrientationStamped &oriStamped,
                                 const builtin_interfaces::msg::Time &time)
{
    (void)oriStamped;
    (void)time;
    return false;
}

/**
 * @brief 测试桩：模拟启动 SLAM 失败。
 * @return 固定返回 false。
 */
bool xv_dev_wrapper::start_slam(void)
{
    return false;
}

/**
 * @brief 测试桩：模拟停止 SLAM 失败。
 * @return 固定返回 false。
 */
bool xv_dev_wrapper::stop_slam(void)
{
    return false;
}

/**
 * @brief 测试桩：模拟获取 SLAM 位姿失败。
 * @param poseSteamped 输出位姿。
 * @param prediction 预测时间。
 * @return 固定返回 false。
 */
bool xv_dev_wrapper::slam_get_pose(geometry_msgs::msg::PoseStamped &poseSteamped,
                                   const builtin_interfaces::msg::Duration &prediction)
{
    (void)poseSteamped;
    (void)prediction;
    return false;
}

/**
 * @brief 测试桩：模拟按时间戳获取 SLAM 位姿失败。
 * @param poseSteamped 输出位姿。
 * @param time 查询时间戳。
 * @return 固定返回 false。
 */
bool xv_dev_wrapper::slam_get_pose_at(geometry_msgs::msg::PoseStamped &poseSteamped,
                                      const builtin_interfaces::msg::Time &time)
{
    (void)poseSteamped;
    (void)time;
    return false;
}

/**
 * @brief 测试桩：模拟保存地图失败。
 * @param filename 地图文件名。
 * @return 固定返回 false。
 */
bool xv_dev_wrapper::saveCslamMap(std::string filename)
{
    (void)filename;
    return false;
}

/**
 * @brief 测试桩：模拟加载地图失败。
 * @param filename 地图文件名。
 * @return 固定返回 false。
 */
bool xv_dev_wrapper::loadCslamMap(std::string filename)
{
    (void)filename;
    return false;
}

/**
 * @brief 测试桩：模拟启动手柄失败。
 * @param portAddress 端口地址。
 * @return 固定返回 false。
 */
bool xv_dev_wrapper::startController(std::string portAddress)
{
    (void)portAddress;
    return false;
}

/**
 * @brief 测试桩：模拟启动夹爪失败。
 * @return 固定返回 false。
 */
bool xv_dev_wrapper::start_clamp(void)
{
    return false;
}

/**
 * @brief 测试桩：模拟停止夹爪失败。
 * @return 固定返回 false。
 */
bool xv_dev_wrapper::stop_clamp(void)
{
    return false;
}
