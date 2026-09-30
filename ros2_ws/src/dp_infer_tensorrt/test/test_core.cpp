// 确定性回归：几何期望值独立于实现，RM75 FK 数值取自现有 Python 实现。
#include "dp_infer_tensorrt/core.hpp"
#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace dp_infer_tensorrt;
namespace {
RawAction identity_action() {
    RawAction result{};
    for (int i = 0; i < HORIZON; ++i) { result[i*10+3] = 1; result[i*10+7] = 1; result[i*10+9] = .4f; }
    return result;
}
void frame(ObservationBuffer &buffer,int64_t stamp,double gripper = .5) {
    buffer.add_image(stamp,std::vector<float>(3*224*224));
    buffer.add_pose(stamp,Eigen::Matrix4d::Identity());
    buffer.add_gripper(stamp,gripper);
}
}
TEST(Image, PaddingRgbAndCompressedPng) {
    std::vector<uint8_t> pixels{1,2,3,4,5,6,99,99,7,8,9,10,11,12,99,99};
    auto rgb = image_to_rgb(pixels,2,2,8,"bgr8");
    EXPECT_EQ(rgb.at<cv::Vec3b>(0,0),cv::Vec3b(3,2,1));
    EXPECT_EQ(rgb.at<cv::Vec3b>(1,1),cv::Vec3b(12,11,10));
    EXPECT_THROW(image_to_rgb(pixels,2,2,5,"rgb8"),std::invalid_argument);
    EXPECT_THROW(image_to_rgb(pixels,2,2,8,"mono8"),std::invalid_argument);
    std::vector<uint8_t> encoded;
    cv::imencode(".png",rgb,encoded);
    EXPECT_EQ(compressed_image_to_rgb(encoded).at<cv::Vec3b>(0,0),cv::Vec3b(1,2,3));
    EXPECT_THROW(compressed_image_to_rgb({}),std::invalid_argument);
}
TEST(Image, LetterboxBlackBorderAndChwChannels) {
    auto image = letterbox_rgb(cv::Mat(100,200,CV_8UC3,cv::Scalar(255,128,0)));
    EXPECT_EQ(image.size(),3*224*224u);
    EXPECT_EQ(image[55*224],0); EXPECT_EQ(image[56*224],1);
    EXPECT_FLOAT_EQ(image[224*224+56*224],128/255.0f);
    EXPECT_EQ(image[2*224*224+56*224],0); EXPECT_EQ(image[168*224],0);
}
TEST(Geometry, RowsOfRotationAndFrozenReference) {
    RawAction action = identity_action();
    for (int i = 0; i < 16; ++i) {
        auto *p = action.data()+i*10;
        p[0] = .1f; p[3] = 0; p[4] = -1; p[6] = 1; p[7] = 0;
    }
    InferenceContext context;
    context.reference_pose.block<3,3>(0,0) = Eigen::AngleAxisd(std::acos(-1.0)/2,Eigen::Vector3d::UnitZ()).toRotationMatrix();
    context.reference_pose(0,3) = 1;
    auto sequence = decode_actions(action,context);
    EXPECT_NEAR(sequence.positions[0].x(),1,1e-8);
    EXPECT_NEAR(sequence.positions[0].y(),.1,1e-8);
    auto rotation = sequence.quaternions[0].toRotationMatrix();
    EXPECT_NEAR(rotation(0,0),-1,1e-12); EXPECT_NEAR(rotation(1,1),-1,1e-12);
    EXPECT_NEAR(sequence.time_from_start[15],.5,1e-12);
    action[3] = action[4] = action[5] = 0;
    EXPECT_THROW(decode_actions(action,context),std::invalid_argument);
}
TEST(Geometry, RelativeHistoryAndEpisodeStartRotation) {
    auto image = std::make_shared<const std::vector<float>>(3*224*224,0);
    Eigen::Matrix4d a = Eigen::Matrix4d::Identity(),b = a;
    b(0,3) = 1;
    b.block<3,3>(0,0) = Eigen::AngleAxisd(std::acos(-1.0)/2,Eigen::Vector3d::UnitZ()).toRotationMatrix();
    auto observations = build_observations({Observation{1,image,a,.25},Observation{2,image,b,.75}},a);
    EXPECT_NEAR(observations["robot0_eef_pos"][0],0,1e-7);
    EXPECT_NEAR(observations["robot0_eef_pos"][1],1,1e-7);
    EXPECT_NEAR(observations["robot0_eef_rot_axis_angle"][1],1,1e-7);
    EXPECT_NEAR(observations["robot0_eef_rot_axis_angle_wrt_start"][7],-1,1e-7);
    EXPECT_FLOAT_EQ(observations["robot0_gripper_width"][1],.75f);
}
TEST(Geometry, SequenceValidationAndQuaternionContinuity) {
    auto sequence = decode_actions(identity_action(),InferenceContext{});
    sequence.quaternions[1].coeffs() *= -2;
    sequence.gripper_openness[0] = -1; sequence.gripper_openness[1] = 2;
    validate_sequence(sequence);
    EXPECT_GT(sequence.quaternions[0].dot(sequence.quaternions[1]),.99);
    EXPECT_DOUBLE_EQ(sequence.gripper_openness[0],0); EXPECT_DOUBLE_EQ(sequence.gripper_openness[1],1);
    sequence.time_from_start[1] = .5;
    EXPECT_THROW(validate_sequence(sequence),std::invalid_argument);
}
TEST(Kinematics, PythonRm75ReferenceAndJointOrder) {
    Kinematics fk(TEST_URDF_PATH);
    auto zero = fk.forward({0,0,0,0,0,0,0});
    EXPECT_NEAR(zero(2,3),.8505,1e-12);
    auto pose = fk.forward({.1,-.2,.3,-.4,.5,-.6,.7});
    Eigen::Matrix4d expected;
    expected << -.378464091268579,-.5939001703087495,-.7099630408179609,-.264847669618701,
                .8125219652268644,.1542344454362061,-.562156376700344,-.12155456365588466,
                .44336552374324467,-.7896165674225831,.4241847342935081,.7265259508318718,
                0,0,0,1;
    EXPECT_TRUE(pose.isApprox(expected,1e-12));
    std::vector<std::string> names{"joint7","joint6","joint5","joint4","joint3","joint2","joint1"};
    auto ordered = ordered_joints(names,{7,6,5,4,3,2,1});
    EXPECT_EQ(ordered[0],1); EXPECT_EQ(ordered[6],7);
    names[0] = names[1]; EXPECT_THROW(ordered_joints(names,{7,6,5,4,3,2,1}),std::invalid_argument);
}
TEST(Buffer, MissingStaleOutOfOrderDuplicateAndReset) {
    ObservationBuffer buffer;
    int64_t now = 1000000000;
    frame(buffer,now); frame(buffer,now-OBS_PERIOD_NS);
    auto history = buffer.poll(now);
    ASSERT_TRUE(history); EXPECT_EQ((*history)[0].stamp_ns,now-OBS_PERIOD_NS);
    EXPECT_EQ((*history)[1].stamp_ns,now); EXPECT_FALSE(buffer.poll(now));
    frame(buffer,now+OBS_PERIOD_NS); EXPECT_TRUE(buffer.poll(now+OBS_PERIOD_NS));
    EXPECT_FALSE(buffer.poll(now+500000000));
    buffer.reset(); EXPECT_EQ(buffer.episode_id,1u); EXPECT_FALSE(buffer.start_pose);
    buffer.add_image(now,std::vector<float>(3*224*224)); EXPECT_FALSE(buffer.poll(now));
    EXPECT_THROW(buffer.add_gripper(now,1.1),std::invalid_argument);
    EXPECT_THROW(ObservationBuffer(.05,.2,.034),std::invalid_argument);
}
TEST(Buffer, FutureAndHistorySpacing) {
    ObservationBuffer buffer;
    frame(buffer,1000000000); frame(buffer,1100000000);
    EXPECT_FALSE(buffer.poll(1000000000)); EXPECT_FALSE(buffer.poll(1100000000));
    frame(buffer,1100000000+OBS_PERIOD_NS);
    EXPECT_TRUE(buffer.poll(1100000000+OBS_PERIOD_NS));
}
TEST(VisualGuard, MarkedBallAndDarkBackground) {
    cv::Mat image(720,1280,CV_8UC3,cv::Scalar(0,0,0));
    cv::circle(image,cv::Point(640,482),55,cv::Scalar(220,220,220),-1);
    cv::circle(image,cv::Point(625,467),12,cv::Scalar(220,70,20),-1);
    cv::circle(image,cv::Point(655,490),12,cv::Scalar(20,110,240),-1);
    auto center = ball_center_bgr(image); ASSERT_TRUE(center);
    EXPECT_NEAR(center->x,640,8); EXPECT_NEAR(center->y,482,8);
    EXPECT_TRUE(ball_near_jaws(center,1280,720));
    EXPECT_FALSE(ball_near_jaws(cv::Point2d(640,200),1280,720));
    EXPECT_FALSE(ball_center_bgr(cv::Mat(720,1280,CV_8UC3,cv::Scalar(0,0,0))));
}
TEST(Ddim, LeadingStepsAndUnsupportedContract) {
    Json::Value config;
    config["num_train_timesteps"] = 50; config["beta_schedule"] = "squaredcos_cap_v2";
    config["prediction_type"] = "epsilon"; config["timestep_spacing"] = "leading";
    config["steps_offset"] = 0; config["clip_sample"] = true; config["clip_sample_range"] = 1.0;
    config["set_alpha_to_one"] = true; config["thresholding"] = false; config["rescale_betas_zero_snr"] = false;
    DdimScheduler scheduler(config);
    EXPECT_EQ(scheduler.steps(8).front().timestep,42); EXPECT_EQ(scheduler.steps(16).front().timestep,45);
    EXPECT_EQ(scheduler.steps(50).front().timestep,49); EXPECT_EQ(scheduler.steps(1).front().timestep,0);
    EXPECT_THROW(scheduler.steps(0),std::invalid_argument);
    EXPECT_THROW(scheduler.steps(51),std::invalid_argument);
    config["thresholding"] = true; EXPECT_THROW(DdimScheduler{config},std::invalid_argument);
}
