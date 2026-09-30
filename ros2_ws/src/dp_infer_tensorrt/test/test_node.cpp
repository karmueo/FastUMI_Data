// 在隔离 ROS domain 中使用确定性引擎检验真实订阅、服务和异步生命周期。
#include "dp_infer_tensorrt/node.hpp"
#include <atomic>
#include <mutex>
#include <thread>

#include <fastumi_interfaces/msg/policy_action_sequence.hpp>
#include <fastumi_interfaces/srv/reset_policy_controller.hpp>
#include <fastumi_interfaces/srv/set_num_inference_steps.hpp>
#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <pluginlib/class_loader.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_srvs/srv/trigger.hpp>

using namespace dp_infer_tensorrt;
using Trigger = std_srvs::srv::Trigger;
using Reset = fastumi_interfaces::srv::ResetPolicyController;
using Steps = fastumi_interfaces::srv::SetNumInferenceSteps;
using SequenceMsg = fastumi_interfaces::msg::PolicyActionSequence;
namespace {
class FakeBackend : public InferenceBackend {
public:
    std::atomic<int> delay_ms{0};
    std::atomic<bool> blocked{false};
    std::mutex mutex;
    std::vector<int> calls;
    RawAction infer(const Observations &,int steps,const RawAction *) override {
        { std::lock_guard<std::mutex> lock(mutex); calls.push_back(steps); }
        while (blocked) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms.load()));
        RawAction result{};
        for (int i = 0; i < HORIZON; ++i) { result[i*10+3] = 1; result[i*10+7] = 1; result[i*10+9] = .4f; }
        return result;
    }
    void warmup(int) override {}
    std::string urdf_sha256() const override { return sha256_file(TEST_URDF_PATH); }
    std::vector<int> captured() { std::lock_guard<std::mutex> lock(mutex); return calls; }
};

class NodeTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { if (!rclcpp::ok()) rclcpp::init(0,nullptr); }
    static void TearDownTestSuite() { rclcpp::shutdown(); }
    std::shared_ptr<FakeBackend> engine;
    std::shared_ptr<DpInferenceNode> inference;
    std::shared_ptr<rclcpp::Node> io;
    rclcpp::executors::SingleThreadedExecutor executor;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr images;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joints;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gripper;
    rclcpp::Subscription<SequenceMsg>::SharedPtr subscription;
    std::vector<rclcpp::ClientBase::SharedPtr> clients;
    std::vector<SequenceMsg> outputs;
    std::vector<uint8_t> pixels;

    void SetUp() override {
        if (!rclcpp::ok()) rclcpp::init(0,nullptr);
        engine = std::make_shared<FakeBackend>();
        io = std::make_shared<rclcpp::Node>("trt_test_io");
        executor.add_node(io);
        images = io->create_publisher<sensor_msgs::msg::CompressedImage>("/wrist_camera/image_raw/compressed",10);
        joints = io->create_publisher<sensor_msgs::msg::JointState>("/joint_states",10);
        gripper = io->create_publisher<std_msgs::msg::Float32>("/motion_control/gripper_state",10);
        subscription = io->create_subscription<SequenceMsg>("/fastumi/policy/action_sequence",10,
            [this](SequenceMsg::ConstSharedPtr message) { outputs.push_back(*message); });
        cv::imencode(".png",cv::Mat(32,64,CV_8UC3,cv::Scalar(20,40,60)),pixels);
    }
    void TearDown() override {
        engine->blocked = false;
        if (inference) { executor.remove_node(inference); inference.reset(); }
        executor.remove_node(io); clients.clear(); subscription.reset(); images.reset(); joints.reset(); gripper.reset(); io.reset();
    }
    void build(bool controller = false,const std::string &plugins = "",bool managed = true,bool sim_time = false) {
        rclcpp::NodeOptions options;
        options.parameter_overrides({rclcpp::Parameter("urdf_path",TEST_URDF_PATH),
            rclcpp::Parameter("require_controller_reset",controller),rclcpp::Parameter("task_control_enabled",managed),
            rclcpp::Parameter("postprocessors",plugins),rclcpp::Parameter("result_timeout_s",.25),
            rclcpp::Parameter("controller_reset_timeout_s",.05),rclcpp::Parameter("use_sim_time",sim_time)});
        inference = std::make_shared<DpInferenceNode>(options,engine); executor.add_node(inference);
        pump(250,true);
    }
    void feed() {
        auto stamp = inference ? inference->now() : io->now();
        sensor_msgs::msg::JointState joint; joint.header.stamp = stamp;
        for (int i = 1; i <= 7; ++i) { joint.name.push_back("joint"+std::to_string(i)); joint.position.push_back(0); }
        joints->publish(joint);
        std_msgs::msg::Float32 width; width.data = .5f; gripper->publish(width);
        sensor_msgs::msg::CompressedImage image; image.header.stamp = stamp; image.format = "png"; image.data = pixels;
        images->publish(image);
    }
    void pump(int milliseconds,bool sensors = false) {
        auto deadline = std::chrono::steady_clock::now()+std::chrono::milliseconds(milliseconds);
        auto next = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() < deadline) {
            if (sensors && std::chrono::steady_clock::now() >= next) { feed(); next += std::chrono::milliseconds(33); }
            executor.spin_some(); std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    rclcpp::Client<Trigger>::SharedFuture request(const std::string &suffix) {
        auto client = io->create_client<Trigger>("/fastumi/policy/"+suffix);
        clients.push_back(client);
        for (int i = 0; i < 100 && !client->service_is_ready(); ++i) pump(10);
        return client->async_send_request(std::make_shared<Trigger::Request>()).share();
    }
    bool call(const std::string &suffix,bool sensors = true) {
        auto future = request(suffix);
        for (int i = 0; i < 400 && future.wait_for(std::chrono::seconds(0)) != std::future_status::ready; ++i) pump(10,sensors);
        if (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        return future.get()->success;
    }
};
}

TEST_F(NodeTest, StartsStopsResetsAndPreservesMessageContract) {
    build(); EXPECT_TRUE(outputs.empty());
    ASSERT_TRUE(call("start_task")); pump(400,true); ASSERT_FALSE(outputs.empty());
    auto output = outputs.back(); EXPECT_EQ(output.header.frame_id,"base_link"); EXPECT_EQ(output.end_frame,"Link7");
    EXPECT_EQ(output.poses.size(),16u); EXPECT_EQ(output.time_from_start.back().nanosec,500000000u);
    EXPECT_NEAR(output.poses.front().position.z,.8505,1e-12);
    auto stamp = rclcpp::Time(output.header.stamp).nanoseconds();
    EXPECT_LE(stamp,io->now().nanoseconds()); EXPECT_LT(io->now().nanoseconds()-stamp,250000000);
    ASSERT_TRUE(call("stop_task")); size_t stopped = outputs.size(); pump(300,true); EXPECT_EQ(outputs.size(),stopped);
    ASSERT_TRUE(call("start_task")); pump(250,true); ASSERT_GT(outputs.size(),stopped);
    EXPECT_GT(outputs.back().episode_id,output.episode_id); EXPECT_GT(outputs.back().sequence_id,output.sequence_id);
    ASSERT_TRUE(call("reset_episode")); EXPECT_FALSE(call("return_to_start"));
}
TEST_F(NodeTest, ChangingStepsFreezesInflightAndRejectsInvalidValues) {
    build(); engine->blocked = true;
    ASSERT_TRUE(call("start_task")); pump(100,true);
    ASSERT_FALSE(engine->captured().empty()); EXPECT_EQ(engine->captured()[0],8);
    auto client = io->create_client<Steps>("/fastumi/policy/set_inference_steps");
    auto req = std::make_shared<Steps::Request>(); req->num_inference_steps = 16;
    auto future = client->async_send_request(req); pump(100,true);
    ASSERT_EQ(future.wait_for(std::chrono::seconds(0)),std::future_status::ready); EXPECT_TRUE(future.get()->success);
    EXPECT_EQ(inference->get_parameter("num_inference_steps").as_int(),16);
    auto invalid = inference->set_parameters_atomically({rclcpp::Parameter("num_inference_steps",51)});
    EXPECT_FALSE(invalid.successful);
    auto type = inference->set_parameters_atomically({rclcpp::Parameter("num_inference_steps",true)});
    EXPECT_FALSE(type.successful); EXPECT_EQ(inference->get_parameter("num_inference_steps").as_int(),16);
    engine->blocked = false; pump(300,true);
    auto calls = engine->captured(); ASSERT_GE(calls.size(),2u); EXPECT_EQ(calls[1],16);
}
TEST_F(NodeTest, StopInvalidatesOldWorkerResult) {
    build(); engine->blocked = true;
    ASSERT_TRUE(call("start_task")); pump(120,true); ASSERT_FALSE(engine->captured().empty());
    ASSERT_TRUE(call("stop_task")); engine->blocked = false; pump(200,true); EXPECT_TRUE(outputs.empty());
    ASSERT_TRUE(call("start_task")); pump(300,true); ASSERT_FALSE(outputs.empty());
    EXPECT_GE(outputs.front().episode_id,3u);
}
TEST_F(NodeTest, ExpiredResultsAndMissingInputsAreNotPublished) {
    build(); engine->delay_ms = 300;
    ASSERT_TRUE(call("start_task")); pump(900,true); EXPECT_TRUE(outputs.empty());
    ASSERT_TRUE(call("stop_task")); pump(300); EXPECT_FALSE(call("start_task",false));
}
TEST_F(NodeTest, ControllerResetWaitsForAckTimesOutAndBlocksInference) {
    build(true,"",false);
    std::shared_ptr<rmw_request_id_t> header;
    auto controller = io->create_service<Reset>("/fastumi/rm75/placo/reset_episode",
        [&header](std::shared_ptr<rmw_request_id_t> h,std::shared_ptr<Reset::Request>) { header = h; });
    pump(100,true);
    auto response = request("reset_episode"); pump(150,true);
    ASSERT_EQ(response.wait_for(std::chrono::seconds(0)),std::future_status::ready); EXPECT_FALSE(response.get()->success);
    ASSERT_TRUE(header); outputs.clear(); pump(200,true); EXPECT_TRUE(outputs.empty());
    // 迟到的旧确认不能解除新 episode 的发布门控。
    Reset::Response late; late.success = true; controller->send_response(*header,late); pump(100,true); EXPECT_TRUE(outputs.empty());
}
TEST_F(NodeTest, ControllerStartAndReturnAreInterruptible) {
    build(true);
    std::shared_ptr<rmw_request_id_t> start_header,home_header;
    auto start_server = io->create_service<Reset>("/fastumi/rm75/placo/start_task",
        [&start_header](std::shared_ptr<rmw_request_id_t> h,std::shared_ptr<Reset::Request>) { start_header = h; });
    auto reset_server = io->create_service<Reset>("/fastumi/rm75/placo/reset_episode",
        [](std::shared_ptr<Reset::Request>,std::shared_ptr<Reset::Response> r) { r->success = true; });
    auto home_server = io->create_service<Reset>("/fastumi/rm75/placo/return_to_start",
        [&home_header](std::shared_ptr<rmw_request_id_t> h,std::shared_ptr<Reset::Request>) { home_header = h; });
    pump(150,true);
    auto start = request("start_task"); pump(100,true); ASSERT_TRUE(start_header);
    ASSERT_TRUE(call("stop_task")); pump(100,true);
    ASSERT_EQ(start.wait_for(std::chrono::seconds(0)),std::future_status::ready); EXPECT_FALSE(start.get()->success);
    Reset::Response late; late.success = true; start_server->send_response(*start_header,late); pump(100,true);
    EXPECT_TRUE(outputs.empty());
    auto home = request("return_to_start"); pump(150,true); ASSERT_TRUE(home_header);
    ASSERT_TRUE(call("stop_task")); pump(100,true);
    ASSERT_EQ(home.wait_for(std::chrono::seconds(0)),std::future_status::ready); EXPECT_FALSE(home.get()->success);
    home_server->send_response(*home_header,late); pump(100,true); EXPECT_TRUE(outputs.empty());
}
TEST_F(NodeTest, PluginOutputValidatedAfterEveryStage) {
    build(false,"dp_infer_tensorrt/TestShift,dp_infer_tensorrt/TestShift");
    ASSERT_TRUE(call("start_task")); pump(250,true); ASSERT_FALSE(outputs.empty());
    EXPECT_NEAR(outputs.back().poses[0].position.x,.02,1e-12);
    ASSERT_TRUE(call("stop_task")); executor.remove_node(inference); inference.reset(); outputs.clear();
    build(false,"dp_infer_tensorrt/TestInvalid"); ASSERT_TRUE(call("start_task")); pump(250,true); EXPECT_TRUE(outputs.empty());
}
TEST_F(NodeTest, PluginExceptionDropsPrediction) {
    build(false,"dp_infer_tensorrt/TestThrow"); ASSERT_TRUE(call("start_task")); pump(250,true); EXPECT_TRUE(outputs.empty());
}
TEST_F(NodeTest, ClockRollbackInvalidatesInflightEpisodeAndPausesTask) {
    build(false,"",true,true);
    auto clock = io->create_publisher<rosgraph_msgs::msg::Clock>("/clock",10);
    auto publish_clock = [&clock](int sec,uint32_t nanosec) {
        rosgraph_msgs::msg::Clock message; message.clock.sec = sec; message.clock.nanosec = nanosec;
        clock->publish(message);
    };
    publish_clock(10,0); pump(50,true);
    engine->blocked = true;
    ASSERT_TRUE(call("start_task"));
    publish_clock(10,33333333); pump(40,true);
    publish_clock(10,66666666); pump(40,true);
    ASSERT_FALSE(engine->captured().empty());
    publish_clock(9,0); pump(50,true); engine->blocked = false;
    publish_clock(9,33333333); pump(50,true);
    EXPECT_TRUE(outputs.empty()); EXPECT_FALSE(call("start_task"));
    ASSERT_TRUE(call("stop_task"));
}
TEST_F(NodeTest, TimedOutStartCannotBeRevivedByLateControllerReply) {
    build(true);
    std::shared_ptr<rmw_request_id_t> header;
    auto start_server = io->create_service<Reset>("/fastumi/rm75/placo/start_task",
        [&header](std::shared_ptr<rmw_request_id_t> h,std::shared_ptr<Reset::Request>) { header = h; });
    auto reset_server = io->create_service<Reset>("/fastumi/rm75/placo/reset_episode",
        [](std::shared_ptr<Reset::Request>,std::shared_ptr<Reset::Response> r) { r->success = true; });
    pump(150,true);
    auto start = request("start_task"); pump(3200,true);
    ASSERT_TRUE(header); ASSERT_EQ(start.wait_for(std::chrono::seconds(0)),std::future_status::ready);
    EXPECT_FALSE(start.get()->success); ASSERT_TRUE(call("stop_task"));
    Reset::Response late; late.success = true; start_server->send_response(*header,late);
    pump(150,true); EXPECT_TRUE(outputs.empty());
}
