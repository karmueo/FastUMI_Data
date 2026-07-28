/**
 * @file xv_ros2_node.cpp
 * @brief 实现 XV SDK ROS 2 节点的设备发现、topic/service 初始化和消息发布。
 */

#include "xv_ros2_node.h"
xvision_ros2_node::xvision_ros2_node(void)
: Node(
    "xvisio_SDK",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(
      true)),
  count_(0), m_stopWatchDevices(false) {}

xvision_ros2_node::~xvision_ros2_node()
{
  m_stopWatchDevices.store(true);
  m_watchDevicesCondition.notify_all();
  if (m_watchDevThread.joinable()) {
    m_watchDevThread.join();
  }
}

void xvision_ros2_node::init(void)
{
  // initFrameIDParameter();
  get_frameId_parameters();
  get_device_config_parameters();
  // initTopicAndServer();

  m_watchDevThread = std::thread(&xvision_ros2_node::watchDevices, this);
}

// module_name-size > 0
std::string get_module_name(std::string sn, std::string module_name)
{
  bool is_slash = module_name[0] == '/';

  return "xv_sdk/SN" + sn + (is_slash ? "" : "/") + module_name;
}

void xvision_ros2_node::initTopicAndServer(std::string sn)
{
  /** 防止设备热插拔初始化与已有设备的数据发布并发访问映射容器。 */
  std::unique_lock<std::shared_mutex> deviceStateLock(m_deviceStateMutex);
  //
  std::string imuPubName = get_module_name(sn, "imu");
  m_imuPublisher.emplace(sn, rclcpp::Publisher<rosImu>::SharedPtr());
  m_imuPublisher[sn] = this->create_publisher<rosImu>(imuPubName.c_str(), 10);

  //
  auto imu_set_flag =
    [this, sn](const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
      /** 当前服务对应的设备封装对象。 */
      auto device = findMappedValue(m_deviceMap, sn);
      response->success = device != nullptr;
      if (!device) {
        response->message = "Device is unavailable.";
        return;
      }
      device->set_imu_flag(request->data);
      if (request->data) {
        response->message = "IMU frame rate increased.";
      } else {
        response->message = "IMU frame rate decreased.";
      }
    };

  std::string service_imu_flag_name = get_module_name(sn, "imu/flag");
  m_service_flag_imu_1000.emplace(
      sn, rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr());
  m_service_flag_imu_1000[sn] = this->create_service<std_srvs::srv::SetBool>(
      service_imu_flag_name.c_str(), imu_set_flag);

  //
  // std::string oriPubName = get_module_name(sn, "orientation");
  // m_orientationPublisher.emplace(sn,
  // rclcpp::Publisher<rosOrientationStamped>::SharedPtr());
  // m_orientationPublisher[sn] =
  // this->create_publisher<rosOrientationStamped>(oriPubName.c_str(), 1);

  //
  auto imuSensor_startOri_callBack =
    [this,
      sn](const std::shared_ptr<std_srvs::srv::Trigger::Request>/*req*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res) -> bool {
      /** 当前服务对应的设备封装对象。 */
      auto device = findMappedValue(m_deviceMap, sn);
      /** 启动姿态流的执行结果。 */
      bool ret = device ? device->startImuOri() : false;
      res->success = ret;
      res->message = ret ? "successed" : "failed";
      return ret;
    };
  std::string oriStart = get_module_name(sn, "start_orientation");
  m_service_imuSensor_startOri.emplace(
      sn, rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr());
  m_service_imuSensor_startOri[sn] =
    this->create_service<std_srvs::srv::Trigger>(oriStart.c_str(),
                                                   imuSensor_startOri_callBack);

  //
  auto imuSensor_stopOri_callBack =
    [this,
      sn](const std::shared_ptr<std_srvs::srv::Trigger::Request>/*req*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res) -> bool {
      /** 当前服务对应的设备封装对象。 */
      auto device = findMappedValue(m_deviceMap, sn);
      /** 停止姿态流的执行结果。 */
      bool ret = device ? device->stopImuOri() : false;
      res->success = ret;
      res->message = ret ? "successed" : "failed";
      return ret;
    };
  std::string oriStop = get_module_name(sn, "stop_orientation");
  m_service_imuSensor_stopOri.emplace(
      sn, rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr());
  m_service_imuSensor_stopOri[sn] =
    this->create_service<std_srvs::srv::Trigger>(oriStop.c_str(),
                                                   imuSensor_stopOri_callBack);

  //
  auto imuSensor_getOri_callback =
    [this,
      sn](const std::shared_ptr<xv_ros2_msgs::srv::GetOrientation_Request> req,
    std::shared_ptr<xv_ros2_msgs::srv::GetOrientation_Response> res)
    -> bool {
      /** 当前服务对应的设备封装对象。 */
      auto device = findMappedValue(m_deviceMap, sn);
      return device ?
             device->getImuOri(res->orientation, req->prediction) : false;
    };
  std::string getOri = get_module_name(sn, "get_orientation");
  m_service_imuSensor_getOri.emplace(
      sn, rclcpp::Service<xv_ros2_msgs::srv::GetOrientation>::SharedPtr());
  m_service_imuSensor_getOri[sn] =
    this->create_service<xv_ros2_msgs::srv::GetOrientation>(
          getOri.c_str(), imuSensor_getOri_callback);

  //
  auto imuSensor_getOriAt_callback =
    [this,
      sn](const std::shared_ptr<xv_ros2_msgs::srv::GetOrientationAt_Request>
    req,
    std::shared_ptr<xv_ros2_msgs::srv::GetOrientationAt_Response> res)
    -> bool {
      /** 当前服务对应的设备封装对象。 */
      auto device = findMappedValue(m_deviceMap, sn);
      return device ?
             device->getImuOriAt(res->orientation, req->timestamp) : false;
    };
  std::string getOriAt = get_module_name(sn, "get_orientation_at");
  m_service_imuSensor_getOriAt.emplace(
      sn, rclcpp::Service<xv_ros2_msgs::srv::GetOrientationAt>::SharedPtr());
  m_service_imuSensor_getOriAt[sn] =
    this->create_service<xv_ros2_msgs::srv::GetOrientationAt>(
          getOriAt.c_str(), imuSensor_getOriAt_callback);


  // auto tof_start_service_callback = [this, sn](const
  // std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
  //                                     std::shared_ptr<std_srvs::srv::Trigger::Response>
  //                                     res) -> bool
  // {
  //     bool ret = m_deviceMap[sn] ? m_deviceMap[sn]->start_tof() : false;
  //     res->success = ret;
  //     res->message = ret ? "successed" : "failed";
  //     return ret;
  // };
  // std::string startTOF = get_module_name(sn,"start_tof";
  // rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr tofStart;
  // m_service_tof_start.emplace(sn, tofStart);
  // m_service_tof_start[sn] =
  // this->create_service<std_srvs::srv::Trigger>(startTOF.c_str(),
  // tof_start_service_callback);

  // auto tof_stop_service_callback = [this, sn](const
  // std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
  //                                     std::shared_ptr<std_srvs::srv::Trigger::Response>
  //                                     res) -> bool
  // {
  //     bool ret = m_deviceMap[sn] ? m_deviceMap[sn]->stop_tof() : false;
  //     res->success = ret;
  //     res->message = ret ? "successed" : "failed";
  //     return ret;
  // };
  // std::string stopTOF = get_module_name(sn,"stop_tof";
  // rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr tofStop;
  // m_service_tof_stop.emplace(sn, tofStop);
  // m_service_tof_stop[sn] =
  // this->create_service<std_srvs::srv::Trigger>(stopTOF.c_str(),
  // tof_stop_service_callback);

  if (m_deviceConfig["tof_enable"]) {
    //
    std::string tofImage = get_module_name(sn, "tof/depth/image_rect_raw");
    m_tof_camera_publisher.emplace(sn, image_transport::Publisher());
    m_tof_camera_publisher[sn] = image_transport::create_publisher(
        this, tofImage.c_str()); // image_transport::create_camera_publisher(this,
                                 // "camera/depth/image_rect_raw");

    //
    std::string tofInfo = get_module_name(sn, "tof/depth/camera_info");
    m_tof_cameraInfo_pub.emplace(sn,
                                 rclcpp::Publisher<rosCamInfo>::SharedPtr());
    m_tof_cameraInfo_pub[sn] =
      this->create_publisher<rosCamInfo>(tofInfo.c_str(), 1);

    /** ToF 原始 IR 强度图 topic。 */
    std::string tofIrImage = get_module_name(sn, "tof/ir/image_raw");
    m_tof_ir_camera_publisher.emplace(sn, image_transport::Publisher());
    m_tof_ir_camera_publisher[sn] = image_transport::create_publisher(
        this, tofIrImage.c_str());

    /** ToF 原始 IR 强度图内参 topic。 */
    std::string tofIrInfo = get_module_name(sn, "tof/ir/camera_info");
    m_tof_ir_cameraInfo_pub.emplace(
      sn, rclcpp::Publisher<rosCamInfo>::SharedPtr());
    m_tof_ir_cameraInfo_pub[sn] =
      this->create_publisher<rosCamInfo>(tofIrInfo.c_str(), 1);
  }

  // auto rgb_start_service_callback = [this, sn](const
  // std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
  //                                     std::shared_ptr<std_srvs::srv::Trigger::Response>
  //                                     res) -> bool
  // {
  //     bool ret = m_deviceMap[sn] ? m_deviceMap[sn]->start_rgb() : false;
  //     res->success = ret;
  //     res->message = ret ? "successed" : "failed";
  //     return ret;
  // };
  // std::string startRGB = get_module_name(sn,"start_rgb)";
  // rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr rgbStartSrv;
  // m_service_rgb_start.emplace(sn, rgbStartSrv);
  // m_service_rgb_start[sn] =
  // this->create_service<std_srvs::srv::Trigger>(startRGB.c_str(),
  // rgb_start_service_callback);

  // auto rgb_stop_service_callback = [this, sn](const
  // std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
  //                                     std::shared_ptr<std_srvs::srv::Trigger::Response>
  //                                     res) -> bool
  // {
  //     bool ret = m_deviceMap[sn] ? m_deviceMap[sn]->stop_rgb() : false;
  //     res->success = ret;
  //     res->message = ret ? "successed" : "failed";
  //     return ret;
  // };
  // std::string stopRGB = get_module_name(sn,"stop_rgb)";
  // rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr rgbStopSrv;
  // m_service_rgb_stop.emplace(sn, rgbStopSrv);
  // m_service_rgb_stop[sn] =
  // this->create_service<std_srvs::srv::Trigger>(stopRGB.c_str(),
  // rgb_stop_service_callback);

  if (m_deviceConfig["rgb_enable"]) {
    // rgb
    //
    std::string rgbImage = get_module_name(sn, "rgb/image");
    m_rgb_camera_publisher.emplace(sn, image_transport::Publisher());
    m_rgb_camera_publisher[sn] =
      image_transport::create_publisher(this, rgbImage.c_str());

    //
    std::string rgbInfo = get_module_name(sn, "rgb/camera_info");
    m_rgb_cameraInfo_pub.emplace(sn,
                                 rclcpp::Publisher<rosCamInfo>::SharedPtr());
    m_rgb_cameraInfo_pub[sn] =
      this->create_publisher<rosCamInfo>(rgbInfo.c_str(), 1);
  }


  if (m_deviceConfig["rgb_fisheye_undistort_enable"]) {
    /** Kalibr RGB 鱼眼校正图像 topic 名称。 */
    std::string rgb_fisheye_image =
      get_module_name(sn, "rgb_fisheye_undistorted/image");
    m_rgb_fisheye_undistorted_publisher.emplace(sn,
                                                image_transport::Publisher());
    m_rgb_fisheye_undistorted_publisher[sn] =
      image_transport::create_publisher(this, rgb_fisheye_image.c_str());

    /** Kalibr RGB 鱼眼校正 CameraInfo topic 名称。 */
    std::string rgb_fisheye_info =
      get_module_name(sn, "rgb_fisheye_undistorted/camera_info");
    m_rgb_fisheye_undistorted_cameraInfo_pub.emplace(
        sn, rclcpp::Publisher<rosCamInfo>::SharedPtr());
    m_rgb_fisheye_undistorted_cameraInfo_pub[sn] =
      this->create_publisher<rosCamInfo>(rgb_fisheye_info.c_str(), 1);
  }


}

void xvision_ros2_node::initFrameIDParameter(void)
{
  this->declare_parameter<std::string>("imu_optical_frame",
                                       "imu_optical_frame");
  this->declare_parameter<std::string>("rgb_optical_frame",
                                       "rgb_optical_frame");
  this->declare_parameter<std::string>("tof_optical_frame",
                                       "tof_optical_frame");
}

void xvision_ros2_node::get_frameId_parameters(void)
{
  rclcpp::Parameter frmaId_param;
  std::vector<std::string> paramNames{
    "imu_optical_frame",
    "rgb_optical_frame",
    "tof_optical_frame"};
  for (auto & param_name : paramNames) {
    bool ret = this->get_parameter_or(
        param_name, frmaId_param, rclcpp::Parameter(param_name, param_name));
    if (!ret) {
      printInfoMsg("get parameter is fail" + param_name);
      continue;
    }

    printInfoMsg(frmaId_param.value_to_string());
    m_frameID[param_name] = frmaId_param.value_to_string();
  }
}

void xvision_ros2_node::get_device_config_parameters(void)
{
  m_deviceConfig["rgb_enable"] = true;
  m_deviceConfig["tof_enable"] = false;
  m_deviceConfig["rgb_fisheye_undistort_enable"] = false;
  bool configState = false;
  for (auto & config_name : m_deviceConfig) {
    bool ret = this->get_parameter_or(config_name.first, configState,
                                      config_name.second);
    if (!ret) {
      printErrorMsg("Launch file has no " + config_name.first + " parameter");
      continue;
    }
    config_name.second = configState;
  }
  for (auto & config : m_deviceConfig) {
    std::cout << config.first << ":" << config.second << std::endl;
  }
}

void xvision_ros2_node::printInfoMsg(const std::string msgString) const
{
  RCLCPP_INFO(this->get_logger(), "xv_sdk: '%s'", msgString.c_str());
}

void xvision_ros2_node::printErrorMsg(const std::string msgString) const
{
  RCLCPP_ERROR(this->get_logger(), "xv_sdk: '%s'", msgString.c_str());
}

void xvision_ros2_node::publishImu(std::string sn, const rosImu & imuMsg)
{
  /** 当前设备的 IMU publisher。 */
  auto publisher = findMappedValue(m_imuPublisher, sn);
  if (publisher) {
    publisher->publish(imuMsg);
  }
}

void xvision_ros2_node::publisheOrientation(
  std::string sn, const rosOrientationStamped & orientationMsg)
{
  /** 当前设备的姿态 publisher。 */
  auto publisher = findMappedValue(m_orientationPublisher, sn);
  if (publisher) {
    publisher->publish(orientationMsg);
  }
}

void xvision_ros2_node::publishFEImage(
  std::string sn, const rosImage & image, const rosCamInfo & cameraInfo,
  xv_dev_wrapper::FE_IMAGE_TYPE imageType)
{
  /** 左右鱼眼图像和内参 publisher 的一致性快照。 */
  std::shared_lock<std::shared_mutex> deviceStateLock(m_deviceStateMutex);
  if (imageType == xv_dev_wrapper::FE_IMAGE_TYPE::LEFT_IMAGE) {
    if (m_fisheyeImageLeftPublisher[sn] &&
      m_fisheyeImageLeftInfoPublisher[sn])
    {
      m_fisheyeImageLeftInfoPublisher[sn]->publish(cameraInfo);
      m_fisheyeImageLeftPublisher[sn].publish(image);
    }
  } else if (imageType == xv_dev_wrapper::FE_IMAGE_TYPE::RIGHT_IMAGE) {
    if (m_fisheyeImageRightPublisher[sn] &&
      m_fisheyeImageRightInfoPublisher[sn])
    {
      m_fisheyeImageRightInfoPublisher[sn]->publish(cameraInfo);
      m_fisheyeImageRightPublisher[sn].publish(image);
    }
  } else if (imageType == xv_dev_wrapper::FE_IMAGE_TYPE::LEFT2_IMAGE) {
    if (m_fisheyeImageLeft2Publisher[sn] &&
      m_fisheyeImageLeft2InfoPublisher[sn])
    {
      m_fisheyeImageLeft2InfoPublisher[sn]->publish(cameraInfo);
      m_fisheyeImageLeft2Publisher[sn].publish(image);
    }
  } else if (imageType == xv_dev_wrapper::FE_IMAGE_TYPE::RIGHT2_IMAGE) {
    if (m_fisheyeImageRight2Publisher[sn] &&
      m_fisheyeImageRight2InfoPublisher[sn])
    {
      m_fisheyeImageRight2InfoPublisher[sn]->publish(cameraInfo);
      m_fisheyeImageRight2Publisher[sn].publish(image);
    }
  }
}

void xvision_ros2_node::publishFEAntiDistortionImage(
  std::string sn, const rosImage & image, const rosCamInfo & cameraInfo,
  xv_dev_wrapper::FE_IMAGE_TYPE imageType)
{
  /** 左右鱼眼校正图像和内参 publisher 的一致性快照。 */
  std::shared_lock<std::shared_mutex> deviceStateLock(m_deviceStateMutex);
  if (imageType == xv_dev_wrapper::FE_IMAGE_TYPE::LEFT_IMAGE) {
    if (m_fisheyeAntiDistortionLeftImagePublisher[sn] &&
      m_fisheyeAntiDistortionCamLeftInfoPublisher[sn])
    {
      m_fisheyeAntiDistortionCamLeftInfoPublisher[sn]->publish(cameraInfo);
      m_fisheyeAntiDistortionLeftImagePublisher[sn].publish(image);
    }
  } else if (imageType == xv_dev_wrapper::FE_IMAGE_TYPE::RIGHT_IMAGE) {
    if (m_fisheyeAntiDistortionRightImagePublisher[sn] &&
      m_fisheyeAntiDistortionCamRightInfoPublisher[sn])
    {
      m_fisheyeAntiDistortionCamRightInfoPublisher[sn]->publish(cameraInfo);
      m_fisheyeAntiDistortionRightImagePublisher[sn].publish(image);
    }
  }
}

void xvision_ros2_node::publishSlamPose(
  std::string sn, const geometry_msgs::msg::PoseStamped & pose)
{
  /** 当前设备的 SLAM 位姿 publisher。 */
  auto publisher = findMappedValue(m_slam_pose_publisher, sn);
  if (publisher) {
    publisher->publish(pose);
  }
}

void xvision_ros2_node::publishSlamTrajectory(
  std::string sn,
  const nav_msgs::msg::Path & path)
{
  /** 当前设备的 SLAM 轨迹 publisher。 */
  auto publisher = findMappedValue(m_slam_path_publisher, sn);
  if (publisher) {
    publisher->publish(path);
  }
}

/**
 * @brief 查询指定序列号对应的图像 publisher。
 * @param publishers 图像 publisher 映射表。
 * @param sn 设备序列号。
 * @return publisher 存在且有效时返回指针；否则返回 nullptr。
 */
image_transport::Publisher xvision_ros2_node::findImagePublisher(
  const std::map<std::string, image_transport::Publisher> & publishers,
  const std::string & sn)
{
  return findMappedValue(publishers, sn);
}

/**
 * @brief 查询指定序列号对应的 camera_info publisher。
 * @param publishers camera_info publisher 映射表。
 * @param sn 设备序列号。
 * @return publisher 存在且有效时返回共享指针；否则返回空指针。
 */
rclcpp::Publisher<rosCamInfo>::SharedPtr
xvision_ros2_node::findCameraInfoPublisher(
  const std::map<std::string, rclcpp::Publisher<rosCamInfo>::SharedPtr> & publishers,
  const std::string & sn)
{
  return findMappedValue(publishers, sn);
}

void xvision_ros2_node::publishTofCameraImage(
  std::string sn,
  const rosImage & image,
  const rosCamInfo & cameraInfo)
{
  /** ToF 图像 publisher。 */
  auto imagePublisher = findImagePublisher(m_tof_camera_publisher, sn);
  /** ToF camera_info publisher。 */
  auto cameraInfoPublisher = findCameraInfoPublisher(m_tof_cameraInfo_pub, sn);
  if (!imagePublisher || !cameraInfoPublisher) {
    return;
  }

  imagePublisher.publish(image);
  cameraInfoPublisher->publish(cameraInfo);
}

/**
 * @brief 发布 ToF 原始 IR 强度图和 CameraInfo。
 * @param sn 设备序列号。
 * @param image mono16 编码的原始 IR 强度图。
 * @param cameraInfo ToF 相机内参。
 */
void xvision_ros2_node::publishTofIrCameraImage(
  std::string sn,
  const rosImage & image,
  const rosCamInfo & cameraInfo)
{
  /** ToF IR 图像 publisher。 */
  auto imagePublisher = findImagePublisher(m_tof_ir_camera_publisher, sn);
  /** ToF IR camera_info publisher。 */
  auto cameraInfoPublisher =
    findCameraInfoPublisher(m_tof_ir_cameraInfo_pub, sn);
  if (!imagePublisher || !cameraInfoPublisher) {
    return;
  }

  imagePublisher.publish(image);
  cameraInfoPublisher->publish(cameraInfo);
}

void xvision_ros2_node::publishRGBCameraImage(
  std::string sn,
  const rosImage & image,
  const rosCamInfo & cameraInfo)
{
  /** RGB 图像 publisher。 */
  auto imagePublisher = findImagePublisher(m_rgb_camera_publisher, sn);
  /** RGB camera_info publisher。 */
  auto cameraInfoPublisher = findCameraInfoPublisher(m_rgb_cameraInfo_pub, sn);
  if (!imagePublisher || !cameraInfoPublisher) {
    return;
  }

  imagePublisher.publish(image);
  cameraInfoPublisher->publish(cameraInfo);
}

void xvision_ros2_node::publishRGBRectCameraImage(
  std::string sn, const rosImage & image, const rosCamInfo & cameraInfo)
{
  /** RGB 去畸变图像 publisher。 */
  auto imagePublisher = findImagePublisher(m_rgb_rect_camera_publisher, sn);
  /** RGB 去畸变 camera_info publisher。 */
  auto cameraInfoPublisher =
    findCameraInfoPublisher(m_rgb_rect_cameraInfo_pub, sn);
  if (!imagePublisher || !cameraInfoPublisher) {
    return;
  }

  imagePublisher.publish(image);
  cameraInfoPublisher->publish(cameraInfo);
}

/**
 * @brief 发布 Kalibr RGB 鱼眼校正后的图像和 CameraInfo。
 * @param sn 设备序列号。
 * @param image RGB8 校正图像。
 * @param cameraInfo 无畸变针孔模型 CameraInfo。
 */
void xvision_ros2_node::publishRGBFisheyeUndistortedCameraImage(
  std::string sn, const rosImage & image, const rosCamInfo & cameraInfo)
{
  /** Kalibr RGB 鱼眼校正图像 publisher。 */
  auto imagePublisher =
    findImagePublisher(m_rgb_fisheye_undistorted_publisher, sn);
  /** Kalibr RGB 鱼眼校正 CameraInfo publisher。 */
  auto cameraInfoPublisher =
    findCameraInfoPublisher(m_rgb_fisheye_undistorted_cameraInfo_pub, sn);
  if (!imagePublisher || !cameraInfoPublisher) {
    return;
  }

  imagePublisher.publish(image);
  cameraInfoPublisher->publish(cameraInfo);
}

void xvision_ros2_node::publishSGBMImage(
  std::string sn, const rosImage & image,
  const rosCamInfo & cameraInfo)
{
  /** SGBM 图像 publisher。 */
  auto imagePublisher = findImagePublisher(m_sgbm_camera_publisher, sn);
  /** SGBM camera_info publisher。 */
  auto cameraInfoPublisher = findCameraInfoPublisher(m_sgbm_cameraInfo_pub, sn);
  if (!imagePublisher || !cameraInfoPublisher) {
    return;
  }

  imagePublisher.publish(image);
  cameraInfoPublisher->publish(cameraInfo);
}

void xvision_ros2_node::publishSGBMRawImage(
  std::string sn,
  const rosImage & image,
  const rosCamInfo & cameraInfo)
{
  /** SGBM raw 图像 publisher。 */
  auto imagePublisher = findImagePublisher(m_sgbm_raw_camera_publisher, sn);
  /** SGBM raw camera_info publisher。 */
  auto cameraInfoPublisher =
    findCameraInfoPublisher(m_sgbm_raw_cameraInfo_pub, sn);
  if (!imagePublisher || !cameraInfoPublisher) {
    return;
  }

  imagePublisher.publish(image);
  cameraInfoPublisher->publish(cameraInfo);
}

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
void xvision_ros2_node::publishRGBDCameraImage(
  std::string sn, const rosImage & rgbImage, const rosImage & depthImage,
  const rosImage & irImage, const rosCamInfo & rgbCameraInfo,
  const rosCamInfo & depthCameraInfo, const rosCamInfo & irCameraInfo)
{
  /** SDK RGBD RGB 图像 publisher。 */
  auto rgbImagePublisher = findImagePublisher(m_rgbd_rgb_camera_publisher, sn);
  if (rgbImagePublisher) {
    rgbImagePublisher.publish(rgbImage);
  }

  /** SDK RGBD depth 图像 publisher。 */
  auto depthImagePublisher =
    findImagePublisher(m_rgbd_depth_camera_publisher, sn);
  if (depthImagePublisher) {
    depthImagePublisher.publish(depthImage);
  }

  /** RGBD IR 图像 publisher。 */
  auto irImagePublisher = findImagePublisher(m_rgbd_ir_camera_publisher, sn);
  if (irImagePublisher) {
    irImagePublisher.publish(irImage);
  }

  /** SDK RGBD RGB camera_info publisher。 */
  auto rgbCameraInfoPublisher =
    findCameraInfoPublisher(m_rgbd_rgb_cameraInfo_pub, sn);
  if (rgbCameraInfoPublisher) {
    rgbCameraInfoPublisher->publish(rgbCameraInfo);
  }

  /** SDK RGBD depth camera_info publisher。 */
  auto depthCameraInfoPublisher =
    findCameraInfoPublisher(m_rgbd_depth_cameraInfo_pub, sn);
  if (depthCameraInfoPublisher) {
    depthCameraInfoPublisher->publish(depthCameraInfo);
  }

  /** RGBD IR camera_info publisher。 */
  auto irCameraInfoPublisher =
    findCameraInfoPublisher(m_rgbd_ir_cameraInfo_pub, sn);
  if (irCameraInfoPublisher) {
    irCameraInfoPublisher->publish(irCameraInfo);
  }
}

void xvision_ros2_node::publishRGBDRawCameraImage(
  std::string sn, const rosImage & image, const rosCamInfo & cameraInfo)
{
  /** RGBD raw 图像 publisher。 */
  auto imagePublisher = findImagePublisher(m_rgbd_raw_camera_publisher, sn);
  /** RGBD raw camera_info publisher。 */
  auto cameraInfoPublisher =
    findCameraInfoPublisher(m_rgbd_raw_cameraInfo_pub, sn);
  if (!imagePublisher || !cameraInfoPublisher) {
    return;
  }

  imagePublisher.publish(image);
  cameraInfoPublisher->publish(cameraInfo);
}

void xvision_ros2_node::publishRGBDRawCameraData(
  std::string sn, const rosColorDepth & colorDephData)
{
  /** 当前设备的 RGBD 原始数据 publisher。 */
  auto publisher = findMappedValue(m_rgbd_raw_data_publisher, sn);
  if (publisher) {
    publisher->publish(colorDephData);
  }
}

void xvision_ros2_node::publishRGBRegisteredCameraImage(
  std::string sn, const rosImage & rgbImage, const rosImage & depthImage,
  const rosCamInfo & cameraInfo)
{
  /** RGB registered 彩色图 publisher。 */
  auto rgbImagePublisher =
    findImagePublisher(m_rgb_registered_image_publisher, sn);
  if (rgbImagePublisher) {
    rgbImagePublisher.publish(rgbImage);
  }

  /** RGB registered 深度图 publisher。 */
  auto depthImagePublisher =
    findImagePublisher(m_rgb_registered_depth_publisher, sn);
  if (depthImagePublisher) {
    depthImagePublisher.publish(depthImage);
  }

  /** RGB registered camera_info publisher。 */
  auto cameraInfoPublisher =
    findCameraInfoPublisher(m_rgb_registered_cameraInfo_pub, sn);
  if (cameraInfoPublisher) {
    cameraInfoPublisher->publish(cameraInfo);
  }
}

void xvision_ros2_node::publishFactoryRGBDCameraImage(
  std::string sn, const rosImage & rgbImage, const rosImage & depthImage)
{
  /** factory RGB-D RGB 图像 publisher。 */
  auto rgbImagePublisher =
    findImagePublisher(m_factory_rgbd_rgb_publisher, sn);
  if (rgbImagePublisher) {
    rgbImagePublisher.publish(rgbImage);
  }

  /** factory RGB-D 深度图 publisher。 */
  auto depthImagePublisher =
    findImagePublisher(m_factory_rgbd_depth_publisher, sn);
  if (depthImagePublisher) {
    depthImagePublisher.publish(depthImage);
  }
}

void xvision_ros2_node::publisheEvent(
  std::string sn,
  const rosEventData & eventMsg)
{
  /** 当前设备的事件 publisher。 */
  auto publisher = findMappedValue(m_eventPublisher, sn);
  if (publisher) {
    publisher->publish(eventMsg);
  }
}

std::string xvision_ros2_node::getFrameID(const std::string & defaultId)
{
  std::string framId;
  if (!this->get_parameter(defaultId, framId) || framId.empty()) {
    framId = defaultId;
  }
  return framId;
}

bool xvision_ros2_node::getConfig(std::string configName)
{
  /** 匹配已公开设备配置的迭代器。 */
  const auto config = m_deviceConfig.find(configName);
  return config != m_deviceConfig.end() && config->second;
}

void xvision_ros2_node::broadcasterTfTransform(
  const geometry_msgs::msg::TransformStamped & transform)
{
  if (m_static_tf_broadcaster) {
    m_static_tf_broadcaster->sendTransform(transform);
  }
}

/**
 * @brief 等待下一次设备扫描，停止请求可立即中断等待。
 * @param timeout 最长等待时间。
 * @return 收到停止请求时返回 true；等待超时时返回 false。
 */
bool xvision_ros2_node::waitForDeviceWatchStop(
  std::chrono::milliseconds timeout)
{
  /** 条件变量等待期间持有的互斥锁。 */
  std::unique_lock<std::mutex> lock(m_watchDevicesMutex);
  return m_watchDevicesCondition.wait_for(
    lock, timeout, [this]() {return m_stopWatchDevices.load();});
}

/**
 * @brief 监控设备连接状态，发现新设备后初始化其 ROS 接口。
 */
void xvision_ros2_node::watchDevices(void)
{
  while (!m_stopWatchDevices.load()) {
    xv::setLogLevel(xv::LogLevel::info);
    std::map<std::string, std::shared_ptr<Device>> device_map;
    std::string json = "";
    std::string jsonPath = "/etc/xvisio/config.json";
    std::ifstream ifs(jsonPath);
    if (ifs.is_open()) {
      std::stringstream fbuf;
      fbuf << ifs.rdbuf();
      json = fbuf.str();
      ifs.close();
      if (waitForDeviceWatchStop(std::chrono::milliseconds(5000))) {
        return;
      }
      device_map = getDevices(.0, json);
    } else {
      if (waitForDeviceWatchStop(std::chrono::milliseconds(5000))) {
        return;
      }
      device_map = getDevices(.0);
    }

    if (m_stopWatchDevices.load()) {
      return;
    }

    for (auto & pair : device_map) {
      m_devSerialNumber = pair.first;
      if (m_serialNumbers.empty()) {
        std::cout << "find new device " << m_devSerialNumber << std::endl;
        m_serialNumbers.push_back(m_devSerialNumber);
        /** 新设备对应的 SDK 封装对象。 */
        auto deviceWrapper = std::make_shared<xv_dev_wrapper>(
            this, device_map[m_devSerialNumber], m_devSerialNumber, 1);
        {
          /** 保护设备封装映射写入的独占锁。 */
          std::unique_lock<std::shared_mutex> deviceStateLock(
            m_deviceStateMutex);
          m_deviceMap.emplace(m_devSerialNumber, deviceWrapper);
        }
        std::cout << "init device topic" << std::endl;
        initTopicAndServer(m_devSerialNumber);
        deviceWrapper->startDeviceStreams();
      } else {
        auto it = find(m_serialNumbers.begin(), m_serialNumbers.end(),
                       m_devSerialNumber);
        if (it == m_serialNumbers.end()) {
          std::cout << "find new device " << m_devSerialNumber << std::endl;
          m_serialNumbers.push_back(m_devSerialNumber);
          /** 新设备对应的 SDK 封装对象。 */
          auto deviceWrapper = std::make_shared<xv_dev_wrapper>(
              this, device_map[m_devSerialNumber], m_devSerialNumber, 1);
          {
            /** 保护设备封装映射写入的独占锁。 */
            std::unique_lock<std::shared_mutex> deviceStateLock(
              m_deviceStateMutex);
            m_deviceMap.emplace(m_devSerialNumber, deviceWrapper);
          }
          std::cout << "init device topic" << std::endl;
          initTopicAndServer(m_devSerialNumber);
          deviceWrapper->startDeviceStreams();
        }
      }
    }
    if (waitForDeviceWatchStop(std::chrono::milliseconds(1000))) {
      return;
    }
  }
}

void xvision_ros2_node::publishLeftControllerPose(
  std::string sn, const geometry_msgs::msg::PoseStamped & pose)
{
  /** 当前控制器的左手位姿 publisher。 */
  auto publisher = findMappedValue(m_controller_left_pose_publisher, sn);
  if (publisher) {
    publisher->publish(pose);
  }
}

void xvision_ros2_node::publishRightControllerPose(
  std::string sn, const geometry_msgs::msg::PoseStamped & pose)
{
  /** 当前控制器的右手位姿 publisher。 */
  auto publisher = findMappedValue(m_controller_right_pose_publisher, sn);
  if (publisher) {
    publisher->publish(pose);
  }
}

void xvision_ros2_node::publishLeftControllerData(
  std::string sn, const rosController & controllerMsg)
{
  /** 当前控制器的左手数据 publisher。 */
  auto publisher = findMappedValue(m_controller_left_data_publisher, sn);
  if (publisher) {
    publisher->publish(controllerMsg);
  }
}

void xvision_ros2_node::publishRightControllerData(
  std::string sn, const rosController & controllerMsg)
{
  /** 当前控制器的右手数据 publisher。 */
  auto publisher = findMappedValue(m_controller_right_data_publisher, sn);
  if (publisher) {
    publisher->publish(controllerMsg);
  }
}

void xvision_ros2_node::publisheButton(
  std::string sn, int buttonType,
  const rosButtonMsg & buttonMsg)
{
  /** 当前按钮类型对应的 publisher。 */
  rclcpp::Publisher<rosButtonMsg>::SharedPtr publisher;
  switch (buttonType) {
    case 1:
      publisher = findMappedValue(m_buttonPublisher1, sn);
      break;
    case 2:
      publisher = findMappedValue(m_buttonPublisher2, sn);
      break;
    case 3:
      publisher = findMappedValue(m_buttonPublisher3, sn);
      break;
    case 4:
      publisher = findMappedValue(m_buttonPublisher4, sn);
      break;
  }
  if (publisher) {
    publisher->publish(buttonMsg);
  }
}

void xvision_ros2_node::publishRGBPointCloud(
  std::string sn,
  rosPointCloud2 pointclouds,
  const rosCamInfo & cameraInfo)
{
  /** 当前设备的 RGB 点云 publisher。 */
  auto pointCloudPublisher = findMappedValue(m_rgbPointCloudPublisher, sn);
  if (pointCloudPublisher) {
    pointCloudPublisher->publish(pointclouds);
  }

  /** 当前设备的 RGB 点云内参 publisher。 */
  auto cameraInfoPublisher =
    findMappedValue(m_rgbPointCloud_cameraInfo_pub, sn);
  if (cameraInfoPublisher) {
    cameraInfoPublisher->publish(cameraInfo);
  }
}

void xvision_ros2_node::publishClamp(std::string sn, rosClamp clamp)
{
  /** 当前设备的夹具状态 publisher。 */
  auto publisher = findMappedValue(m_clampPublisher, sn);
  if (publisher) {
    publisher->publish(clamp);
  }
}
