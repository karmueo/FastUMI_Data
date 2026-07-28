/**
 * @file xv_ros2_node.h
 * @brief 声明 XV SDK ROS 2 节点，负责 topic、service 和设备封装对象管理。
 */

#ifndef __XV_ROS2_NODE_H__
#define __XV_ROS2_NODE_H__

#include <condition_variable>
#include <mutex>
#include <shared_mutex>

#include <rclcpp/executors/multi_threaded_executor.hpp>

#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "nav_msgs/msg/path.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include "xv_ros2_msgs/srv/get_orientation.hpp"
#include "xv_ros2_msgs/srv/get_orientation_at.hpp"
#include "xv_ros2_msgs/srv/get_pose.hpp"
#include "xv_ros2_msgs/srv/get_pose_at.hpp"
#include "xv_ros2_msgs/srv/save_map_and_switch_cslam.hpp"
#include "xv_ros2_msgs/srv/load_map_and_switch_cslam.hpp"
#include "xv_ros2_msgs/srv/controller_start.hpp"
#include "xv_ros2_msgs/srv/controller_stop.hpp"

#include <image_transport/image_transport.hpp>
#include "xv_dev_wrapper.h"

#include "./head.h"

using namespace xv;

class xvision_ros2_node : public rclcpp::Node
{
public:
    /** @brief 创建节点并初始化设备监控状态。 */
    xvision_ros2_node(void);
    /** @brief 停止设备监控线程并等待其退出。 */
    ~xvision_ros2_node() override;
    void init(void);
    void initTopicAndServer(std::string sn);
    void initFrameIDParameter(void);

    void get_frameId_parameters(void);
    void get_device_config_parameters(void);

    void printInfoMsg(const std::string msgString) const;
    void printErrorMsg(const std::string msgString) const;

    void publishImu(std::string sn, const rosImu &imuMsg);
    void publisheOrientation(std::string sn, const rosOrientationStamped &orientationMsg);
    void publishFEImage(std::string sn, const rosImage &image, const rosCamInfo &cameraInfo, xv_dev_wrapper::FE_IMAGE_TYPE imageType);
    void publishFEAntiDistortionImage(std::string sn, const rosImage &image, const rosCamInfo &cameraInfo, xv_dev_wrapper::FE_IMAGE_TYPE imageType);
    void publishSlamPose(std::string sn, const geometry_msgs::msg::PoseStamped &pose);
    void publishSlamTrajectory(std::string sn, const nav_msgs::msg::Path &path);
    void publishTofCameraImage(std::string sn, const rosImage &image, const rosCamInfo &cameraInfo);
    /**
     * @brief 发布 ToF 原始 IR 强度图和 CameraInfo。
     * @param sn 设备序列号。
     * @param image mono16 编码的原始 IR 强度图。
     * @param cameraInfo ToF 相机内参。
     */
    void publishTofIrCameraImage(std::string sn,
                                 const rosImage &image,
                                 const rosCamInfo &cameraInfo);
    void publishRGBCameraImage(std::string sn, const rosImage &image, const rosCamInfo &cameraInfo);
    void publishRGBRectCameraImage(std::string sn, const rosImage &image, const rosCamInfo &cameraInfo);
    /**
     * @brief 发布 Kalibr RGB 鱼眼校正后的图像和 CameraInfo。
     * @param sn 设备序列号。
     * @param image RGB8 校正图像。
     * @param cameraInfo 无畸变针孔模型 CameraInfo。
     */
    void publishRGBFisheyeUndistortedCameraImage(std::string sn,
                                                 const rosImage &image,
                                                 const rosCamInfo &cameraInfo);
    void publishSGBMImage(std::string sn, const rosImage &image, const rosCamInfo &cameraInfo);
    void publishSGBMRawImage(std::string sn, const rosImage &image, const rosCamInfo &cameraInfo);
    /**
     * @brief 发布逐像素对齐到公共虚拟相机网格的 RGB、深度和 IR 图像。
     * @param sn 设备序列号。
     * @param rgbImage RGB8 彩色图。
     * @param depthImage 32FC1 米制深度图。
     * @param irImage mono16 ToF IR 强度图。
     * @param rgbCameraInfo RGB 图像内参。
     * @param depthCameraInfo depth 图像内参。
     * @param irCameraInfo IR 图像内参。
     */
    void publishRGBDCameraImage(std::string sn,
                                const rosImage &rgbImage,
                                const rosImage &depthImage,
                                const rosImage &irImage,
                                const rosCamInfo &rgbCameraInfo,
                                const rosCamInfo &depthCameraInfo,
                                const rosCamInfo &irCameraInfo);
    void publishRGBDRawCameraImage(std::string sn, const rosImage &image, const rosCamInfo &cameraInfo);
    void publishRGBDRawCameraData(std::string sn, const rosColorDepth &orientationMsg);
    /**
     * @brief 发布 Kalibr 校正 RGB 图像和已对齐到校正 RGB 网格的深度图。
     * @param sn 设备序列号。
     * @param rgbImage Kalibr 校正后的 RGB8 彩色图。
     * @param depthImage 32FC1 米制深度图。
     * @param cameraInfo Kalibr 无畸变针孔相机内参。
     */
    void publishRGBRegisteredCameraImage(std::string sn,
                                         const rosImage &rgbImage,
                                         const rosImage &depthImage,
                                         const rosCamInfo &cameraInfo);
    /**
     * @brief 发布 factory RGB-D 对齐输出的 RGB 图像和深度图像。
     * @param sn 设备序列号。
     * @param rgbImage RGB8 彩色图。
     * @param depthImage 32FC1 米制深度图。
     */
    void publishFactoryRGBDCameraImage(std::string sn,
                                       const rosImage &rgbImage,
                                       const rosImage &depthImage);

    std::string getFrameID(const std::string &defaultId);
    bool getConfig(std::string configName);
    void broadcasterTfTransform(const geometry_msgs::msg::TransformStamped &transform);
    void publishLeftControllerPose(std::string sn, const geometry_msgs::msg::PoseStamped &pose);
    void publishRightControllerPose(std::string sn, const geometry_msgs::msg::PoseStamped &pose);
    void publishLeftControllerData(std::string sn, const rosController &controllerMsg);
    void publishRightControllerData(std::string sn, const rosController &controllerMsg);

    void publisheEvent(std::string sn, const rosEventData &eventMsg);
    void publisheButton(std::string sn, int buttonType, const rosButtonMsg &buttonMsg);
    void publishRGBPointCloud(std::string sn, rosPointCloud2 pointclouds, const rosCamInfo &cameraInfo);

    void publishClamp(std::string sn, rosClamp clamp);
private:
    /** @brief 监控设备连接状态，发现新设备后初始化其 ROS 接口。 */
    void watchDevices(void);
    /**
     * @brief 等待下一次设备扫描，析构请求停止时立即返回。
     * @param timeout 最长等待时间。
     * @return 收到停止请求时返回 true；等待超时时返回 false。
     */
    bool waitForDeviceWatchStop(std::chrono::milliseconds timeout);
    /**
     * @brief 查询指定序列号对应的图像 publisher。
     * @param publishers 图像 publisher 映射表。
     * @param sn 设备序列号。
     * @return publisher 存在且有效时返回其副本；否则返回空 publisher。
     */
    image_transport::Publisher findImagePublisher(
        const std::map<std::string, image_transport::Publisher> &publishers,
        const std::string &sn);
    /**
     * @brief 查询指定序列号对应的 camera_info publisher。
     * @param publishers camera_info publisher 映射表。
     * @param sn 设备序列号。
     * @return publisher 存在且有效时返回共享指针；否则返回空指针。
     */
    rclcpp::Publisher<rosCamInfo>::SharedPtr findCameraInfoPublisher(
        const std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> &publishers,
        const std::string &sn);
    /**
     * @brief 在线程安全保护下复制指定序列号对应的映射值。
     * @tparam Value 映射值类型。
     * @param values 设备序列号到目标对象的映射。
     * @param sn 设备序列号。
     * @return 找到时返回目标对象副本；否则返回默认构造值。
     */
    template<typename Value>
    Value findMappedValue(const std::map<std::string, Value> &values,
                          const std::string &sn)
    {
        /** 保护共享设备接口容器的读锁。 */
        std::shared_lock<std::shared_mutex> lock(m_deviceStateMutex);
        /** 指向目标映射值的迭代器。 */
        const auto value = values.find(sn);
        return value != values.end() ? value->second : Value{};
    }

private:
    // imu publisher and service.
    std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr> m_imuPublisher;

    // orientationStream publisher and service.
    std::map<std::string, rclcpp::Publisher<xv_ros2_msgs::msg::OrientationStamped>::SharedPtr> m_orientationPublisher;
    std::map<std::string, rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr> m_service_imuSensor_startOri;
    std::map<std::string, rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr> m_service_imuSensor_stopOri;
    std::map<std::string, rclcpp::Service<xv_ros2_msgs::srv::GetOrientation>::SharedPtr> m_service_imuSensor_getOri;
    std::map<std::string, rclcpp::Service<xv_ros2_msgs::srv::GetOrientationAt>::SharedPtr> m_service_imuSensor_getOriAt;

    // FE publisher and service.
    std::map<std::string, image_transport::Publisher> m_fisheyeImageLeftPublisher;
    std::map<std::string, image_transport::Publisher> m_fisheyeImageRightPublisher;
    std::map<std::string, image_transport::Publisher> m_fisheyeImageLeft2Publisher;
    std::map<std::string, image_transport::Publisher> m_fisheyeImageRight2Publisher;
    std::map<std::string, image_transport::Publisher> m_fisheyeAntiDistortionLeftImagePublisher;
    std::map<std::string, image_transport::Publisher> m_fisheyeAntiDistortionRightImagePublisher;
    // rclcpp::Publisher<rosImage>::SharedPtr m_fisheyeImagePublisher[2];
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_fisheyeImageLeftInfoPublisher;
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_fisheyeImageRightInfoPublisher;
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_fisheyeImageLeft2InfoPublisher;
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_fisheyeImageRight2InfoPublisher;
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_fisheyeAntiDistortionCamLeftInfoPublisher;
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_fisheyeAntiDistortionCamRightInfoPublisher;

    // slam publisher and service.
    std::map<std::string, rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr> m_slam_pose_publisher;
    std::map<std::string, rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr> m_slam_path_publisher;
    std::map<std::string, rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr> m_service_slam_start;
    std::map<std::string, rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr> m_service_slam_stop;
    std::map<std::string, rclcpp::Service<xv_ros2_msgs::srv::GetPose>::SharedPtr> m_service_slam_get_pose;
    std::map<std::string, rclcpp::Service<xv_ros2_msgs::srv::GetPoseAt>::SharedPtr> m_service_slam_get_pose_at;

    // tf2 static transform broadcaster
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> m_static_tf_broadcaster;

    // tof publisher and service.
    std::map<std::string, rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr> m_service_tof_start;
    std::map<std::string, rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr> m_service_tof_stop;
    std::map<std::string, image_transport::Publisher> m_tof_camera_publisher;
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_tof_cameraInfo_pub;
    /** ToF 原始 IR 强度图 publisher。 */
    std::map<std::string, image_transport::Publisher> m_tof_ir_camera_publisher;
    /** ToF 原始 IR 强度图 CameraInfo publisher。 */
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_tof_ir_cameraInfo_pub;

    std::map<std::string, rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr> m_service_rgb_start;
    std::map<std::string, rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr> m_service_rgb_stop;
    std::map<std::string, image_transport::Publisher> m_rgb_camera_publisher;
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_rgb_cameraInfo_pub;

    std::map<std::string, image_transport::Publisher> m_rgb_rect_camera_publisher;
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_rgb_rect_cameraInfo_pub;

    /** Kalibr RGB 鱼眼校正图像 publisher。 */
    std::map<std::string, image_transport::Publisher> m_rgb_fisheye_undistorted_publisher;
    /** Kalibr RGB 鱼眼校正 CameraInfo publisher。 */
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_rgb_fisheye_undistorted_cameraInfo_pub;

    std::map<std::string, image_transport::Publisher> m_sgbm_camera_publisher;
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_sgbm_cameraInfo_pub;
    std::map<std::string, image_transport::Publisher> m_sgbm_raw_camera_publisher;
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_sgbm_raw_cameraInfo_pub;

    /** 公共虚拟网格 RGBD 输出的 RGB 图像 publisher。 */
    std::map<std::string, image_transport::Publisher> m_rgbd_rgb_camera_publisher;
    /** 公共虚拟网格 RGBD 输出的 depth 图像 publisher。 */
    std::map<std::string, image_transport::Publisher> m_rgbd_depth_camera_publisher;
    /** SDK RGBD 拆分输出的 RGB camera_info publisher。 */
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_rgbd_rgb_cameraInfo_pub;
    /** SDK RGBD 拆分输出的 depth camera_info publisher。 */
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_rgbd_depth_cameraInfo_pub;
    /** 公共虚拟网格 RGBD 输出的 IR 图像 publisher。 */
    std::map<std::string, image_transport::Publisher> m_rgbd_ir_camera_publisher;
    /** 公共虚拟网格 RGBD 输出的 IR CameraInfo publisher。 */
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_rgbd_ir_cameraInfo_pub;

    std::map<std::string, image_transport::Publisher> m_rgbd_raw_camera_publisher;
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_rgbd_raw_cameraInfo_pub;

    std::map<std::string, rclcpp::Publisher<xv_ros2_msgs::msg::ColorDepth>::SharedPtr> m_rgbd_raw_data_publisher;

    /** RGB registered 输出的 Kalibr 校正彩色图 publisher。 */
    std::map<std::string, image_transport::Publisher> m_rgb_registered_image_publisher;
    /** RGB registered 输出的校正网格深度图 publisher。 */
    std::map<std::string, image_transport::Publisher> m_rgb_registered_depth_publisher;
    /** RGB registered 输出的 Kalibr 无畸变相机内参 publisher。 */
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_rgb_registered_cameraInfo_pub;

    /** factory RGB-D 输出的 RGB 图像 publisher。 */
    std::map<std::string, image_transport::Publisher> m_factory_rgbd_rgb_publisher;
    /** factory RGB-D 输出的深度图 publisher。 */
    std::map<std::string, image_transport::Publisher> m_factory_rgbd_depth_publisher;

    std::map<std::string, rclcpp::Service<xv_ros2_msgs::srv::SaveMapAndSwitchCslam>::SharedPtr> m_cslam_save_map;
    std::map<std::string, rclcpp::Service<xv_ros2_msgs::srv::LoadMapAndSwitchCslam>::SharedPtr> m_cslam_load_map;

    std::map<std::string, rclcpp::Service<xv_ros2_msgs::srv::ControllerStart>::SharedPtr> m_controller_start;
    std::map<std::string, rclcpp::Service<xv_ros2_msgs::srv::ControllerStop>::SharedPtr> m_controller_stop;
    std::map<std::string, rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr> m_controller_left_pose_publisher;
    std::map<std::string, rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr> m_controller_right_pose_publisher;
    std::map<std::string, rclcpp::Publisher<xv_ros2_msgs::msg::Controller>::SharedPtr> m_controller_left_data_publisher;
    std::map<std::string, rclcpp::Publisher<xv_ros2_msgs::msg::Controller>::SharedPtr> m_controller_right_data_publisher;

    std::map<std::string, rclcpp::Publisher<xv_ros2_msgs::msg::EventData>::SharedPtr> m_eventPublisher;

    std::map<std::string, rclcpp::Publisher<xv_ros2_msgs::msg::ButtonMsg>::SharedPtr> m_buttonPublisher1;
    std::map<std::string, rclcpp::Publisher<xv_ros2_msgs::msg::ButtonMsg>::SharedPtr> m_buttonPublisher2;
    std::map<std::string, rclcpp::Publisher<xv_ros2_msgs::msg::ButtonMsg>::SharedPtr> m_buttonPublisher3;
    std::map<std::string, rclcpp::Publisher<xv_ros2_msgs::msg::ButtonMsg>::SharedPtr> m_buttonPublisher4;

    std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> m_rgbPointCloudPublisher;
    std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> m_rgbPointCloud_cameraInfo_pub;

    std::map<std::string, rclcpp::Publisher<rosClamp>::SharedPtr> m_clampPublisher;
    std::map<std::string, rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr> m_service_clamp_start;
    std::map<std::string, rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr> m_service_clamp_stop;

    // rectification flag-service
    std::map<std::string, rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr> m_service_flag_rectification_image;

    // imu 1000
    std::map<std::string, rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr> m_service_flag_imu_1000;

    size_t count_;
    /** 设备发现监控线程。 */
    std::thread m_watchDevThread;
    /** 请求设备发现监控线程停止。 */
    std::atomic_bool m_stopWatchDevices;
    /** 保护设备监控线程可中断等待的互斥量。 */
    std::mutex m_watchDevicesMutex;
    /** 唤醒设备监控线程等待的条件变量。 */
    std::condition_variable m_watchDevicesCondition;
    /** 保护设备封装、publisher 和 service 映射的读写锁。 */
    std::shared_mutex m_deviceStateMutex;

    std::string m_devSerialNumber;
    std::string m_controllerSerialNumber = "Controller";
    std::vector<std::string> m_serialNumbers, m_controllerSerials;
    //
    std::map<std::string, std::shared_ptr<xv_dev_wrapper>> m_deviceMap, m_controllerMap;

    std::map<std::string, std::string> m_frameID;
    std::map<std::string, bool> m_deviceConfig;
    std::shared_ptr<xv::Device> controllerDevice;
    std::string controllerSN;
};
#endif
