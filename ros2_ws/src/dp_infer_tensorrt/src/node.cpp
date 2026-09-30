// ROS 2 任务状态、异步控制器确认、观测与工作线程结果的生命周期管理。
#include "dp_infer_tensorrt/node.hpp"
#include "dp_infer_tensorrt/runtime.hpp"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <mutex>
#include <sstream>
#include <thread>

#include <fastumi_interfaces/msg/policy_action_sequence.hpp>
#include <fastumi_interfaces/srv/reset_policy_controller.hpp>
#include <fastumi_interfaces/srv/set_num_inference_steps.hpp>
#include <opencv2/imgproc.hpp>
#include <pluginlib/class_loader.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace dp_infer_tensorrt {
namespace {
using Clock = std::chrono::steady_clock;
using Trigger = std_srvs::srv::Trigger;
using Reset = fastumi_interfaces::srv::ResetPolicyController;
using Steps = fastumi_interfaces::srv::SetNumInferenceSteps;
using SequenceMsg = fastumi_interfaces::msg::PolicyActionSequence;
int64_t stamp_ns(const builtin_interfaces::msg::Time &stamp) {
    return static_cast<int64_t>(stamp.sec)*1000000000 + stamp.nanosec;
}
SequenceMsg sequence_message(const ActionSequence &sequence,const InferenceContext &context) {
    SequenceMsg output;
    output.header.stamp = rclcpp::Time(context.stamp_ns);
    output.header.frame_id = "base_link"; output.end_frame = "Link7";
    output.episode_id = context.episode_id; output.sequence_id = context.sequence_id;
    for (int i = 0; i < HORIZON; ++i) {
        geometry_msgs::msg::Pose pose;
        const auto &p = sequence.positions[i]; const auto &q = sequence.quaternions[i];
        pose.position.x = p.x(); pose.position.y = p.y(); pose.position.z = p.z();
        pose.orientation.x = q.x(); pose.orientation.y = q.y(); pose.orientation.z = q.z(); pose.orientation.w = q.w();
        output.poses.push_back(pose);
        int64_t nanos = std::llround(sequence.time_from_start[i]*1e9);
        builtin_interfaces::msg::Duration duration;
        duration.sec = nanos/1000000000; duration.nanosec = nanos%1000000000;
        output.time_from_start.push_back(duration);
        output.gripper_openness.push_back(sequence.gripper_openness[i]);
    }
    return output;
}
void positive(double value,const std::string &name) {
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument(name + " must be positive and finite");
}
}  // namespace

struct DpInferenceNode::Impl {
    DpInferenceNode &node;
    std::shared_ptr<InferenceBackend> backend;
    std::unique_ptr<ObservationBuffer> buffer;
    std::unique_ptr<Kinematics> fk;
    std::unique_ptr<pluginlib::ClassLoader<Postprocessor>> plugin_loader;
    std::vector<std::shared_ptr<Postprocessor>> processors;
    rclcpp::Publisher<SequenceMsg>::SharedPtr publisher;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr guard;
    std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions;
    std::vector<rclcpp::Service<Trigger>::SharedPtr> services;
    rclcpp::Service<Steps>::SharedPtr steps_service;
    rclcpp::Client<Reset>::SharedPtr reset_client,start_client,return_client;
    rclcpp::TimerBase::SharedPtr timer;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback;
    std::string state;
    bool managed,require_controller,acknowledged = true;
    int requested_steps = 8;
    double period_s,result_timeout_s,reset_timeout_s;
    int64_t last_clock_ns,last_image_ns = -1;
    uint64_t sequence_id = 0,operation_id = 0;
    Clock::time_point last_started = Clock::time_point::min();

    struct Job { Observations observations; InferenceContext context; int steps; };
    struct Completed {
        std::optional<ActionSequence> sequence;
        InferenceContext context;
        int steps;
        double inference_ms;
        std::string error;
    };
    std::optional<Job> pending;
    bool busy = false;
    std::mutex mutex;
    std::condition_variable condition;
    bool exiting = false;
    std::optional<Job> worker_job;
    std::optional<Completed> completed;
    std::thread worker;

    struct Reply {
        rclcpp::Service<Trigger>::SharedPtr service;
        std::shared_ptr<rmw_request_id_t> header;
    };
    enum class Kind { Start,Stop,Reset,ReturnTaskStop,ReturnStop,ReturnHome,ClockReset };
    struct Operation {
        Kind kind;
        Reply reply;
        uint64_t id;
        rclcpp::Client<Reset>::SharedPtr client;
        rclcpp::Client<Reset>::SharedFuture future;
        int64_t request_id;
        Clock::time_point deadline;
    };
    std::optional<Operation> operation;

    template<class T> T declare(const std::string &name,const T &value,bool read_only = true) {
        rcl_interfaces::msg::ParameterDescriptor descriptor;
        descriptor.read_only = read_only;
        return node.declare_parameter<T>(name,value,descriptor);
    }
    void warn(const std::string &message) {
        RCLCPP_WARN_THROTTLE(node.get_logger(),*node.get_clock(),2000,"%s",message.c_str());
    }
    void close_guard(bool allowed) { std_msgs::msg::Bool message; message.data = allowed; guard->publish(message); }
    void respond(const Reply &reply,bool success,const std::string &message) {
        if (!reply.service) return;
        Trigger::Response response; response.success = success; response.message = message;
        try { reply.service->send_response(*reply.header,response); }
        catch (const std::exception &error) { warn(std::string("Trigger response failed: ")+error.what()); }
    }

    Impl(DpInferenceNode &n,std::shared_ptr<InferenceBackend> engine) : node(n),backend(std::move(engine)) {
        auto engine_dir = declare<std::string>("engine_dir","");
        auto model_manifest = declare<std::string>("model_manifest","");
        auto precision = declare<std::string>("precision","fp16");
        auto device = declare<std::string>("device","cuda:0");
        auto urdf = declare<std::string>("urdf_path","");
        requested_steps = declare<int>("num_inference_steps",8,false);
        validate_steps(requested_steps);
        auto slop = declare<double>("sync_slop_s",0.05);
        auto timeout = declare<double>("input_timeout_s",0.2);
        auto tolerance = declare<double>("history_tolerance_s",0.015);
        auto rate = declare<double>("max_inference_hz",10.0);
        result_timeout_s = declare<double>("result_timeout_s",0.5);
        reset_timeout_s = declare<double>("controller_reset_timeout_s",1.0);
        positive(rate,"max_inference_hz"); positive(result_timeout_s,"result_timeout_s");
        positive(reset_timeout_s,"controller_reset_timeout_s"); period_s = 1/rate;
        buffer = std::make_unique<ObservationBuffer>(slop,timeout,tolerance);
        managed = declare<bool>("task_control_enabled",false);
        require_controller = declare<bool>("require_controller_reset",true);
        state = managed ? "idle" : "running";
        auto plugin_spec = declare<std::string>("postprocessors","");
        if (!backend) {
            if (device.rfind("cuda:",0) != 0 || device.size() <= 5 ||
                device.substr(5).find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("device must be cuda:<nonnegative GPU index>");
            backend = std::make_shared<TensorRtBackend>(RuntimeOptions{engine_dir,model_manifest,precision,std::stoi(device.substr(5))});
        }
        if (urdf.empty() || sha256_file(urdf) != backend->urdf_sha256())
            throw std::invalid_argument("urdf_path must match the model's training URDF SHA-256");
        fk = std::make_unique<Kinematics>(urdf);
        if (!plugin_spec.empty()) {
            plugin_loader = std::make_unique<pluginlib::ClassLoader<Postprocessor>>("dp_infer_tensorrt","dp_infer_tensorrt::Postprocessor");
            std::istringstream stream(plugin_spec); std::string item;
            while (std::getline(stream,item,',')) {
                auto begin = item.find_first_not_of(" \t\n"),end = item.find_last_not_of(" \t\n");
                if (begin != std::string::npos) processors.push_back(plugin_loader->createSharedInstance(item.substr(begin,end-begin+1)));
            }
        }
        backend->warmup(requested_steps);
        publisher = node.create_publisher<SequenceMsg>(declare<std::string>("output_topic","/fastumi/policy/action_sequence"),10);
        guard = node.create_publisher<std_msgs::msg::Bool>(declare<std::string>("close_guard_topic","/fastumi/policy/gripper_close_allowed"),10);
        auto image_topic = declare<std::string>("image_topic","/wrist_camera/image_raw/compressed");
        auto image_type = declare<std::string>("image_type","compressed");
        if (image_type == "raw") {
            subscriptions.push_back(node.create_subscription<sensor_msgs::msg::Image>(image_topic,rclcpp::SensorDataQoS(),
                [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
                    try { on_image(image_to_rgb(message->data,message->height,message->width,message->step,message->encoding),
                                   stamp_ns(message->header.stamp)); }
                    catch (const std::exception &e) { close_guard(false); warn(std::string("Dropped image: ")+e.what()); }
                }));
        } else if (image_type == "compressed") {
            subscriptions.push_back(node.create_subscription<sensor_msgs::msg::CompressedImage>(image_topic,rclcpp::SensorDataQoS(),
                [this](sensor_msgs::msg::CompressedImage::ConstSharedPtr message) {
                    try { on_image(compressed_image_to_rgb(message->data),stamp_ns(message->header.stamp)); }
                    catch (const std::exception &e) { close_guard(false); warn(std::string("Dropped compressed image: ")+e.what()); }
                }));
        } else throw std::invalid_argument("image_type must be raw or compressed");
        subscriptions.push_back(node.create_subscription<sensor_msgs::msg::JointState>(
            declare<std::string>("joint_topic","/joint_states"),rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {
                try { buffer->add_pose(stamp_ns(message->header.stamp),fk->forward(ordered_joints(message->name,message->position))); }
                catch (const std::exception &e) { warn(std::string("Dropped joints: ")+e.what()); }
            }));
        subscriptions.push_back(node.create_subscription<std_msgs::msg::Float32>(
            declare<std::string>("gripper_topic","/motion_control/gripper_state"),rclcpp::SensorDataQoS(),
            [this](std_msgs::msg::Float32::ConstSharedPtr message) {
                try { buffer->add_gripper(node.now().nanoseconds(),message->data); }
                catch (const std::exception &e) { warn(std::string("Dropped gripper: ")+e.what()); }
            }));
        reset_client = node.create_client<Reset>(declare<std::string>("controller_reset_service","/fastumi/rm75/placo/reset_episode"));
        start_client = node.create_client<Reset>(declare<std::string>("controller_start_service","/fastumi/rm75/placo/start_task"));
        return_client = node.create_client<Reset>(declare<std::string>("controller_return_service","/fastumi/rm75/placo/return_to_start"));
        add_service(declare<std::string>("reset_service","/fastumi/policy/reset_episode"),"reset");
        add_service(declare<std::string>("start_service","/fastumi/policy/start_task"),"start");
        add_service(declare<std::string>("stop_service","/fastumi/policy/stop_task"),"stop");
        add_service(declare<std::string>("return_service","/fastumi/policy/return_to_start"),"return");
        parameter_callback = node.add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter> &parameters) {
            rcl_interfaces::msg::SetParametersResult result; result.successful = true;
            for (const auto &p : parameters) {
                if (p.get_name() != "num_inference_steps") continue;
                if (p.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER || p.as_int() < 1 || p.as_int() > 50) {
                    result.successful = false; result.reason = "num_inference_steps must be an integer in [1, 50]"; return result;
                }
            }
            for (const auto &p : parameters) if (p.get_name() == "num_inference_steps") requested_steps = p.as_int();
            return result;
        });
        steps_service = node.create_service<Steps>(declare<std::string>("set_inference_steps_service","/fastumi/policy/set_inference_steps"),
            [this](std::shared_ptr<Steps::Request> request,std::shared_ptr<Steps::Response> response) {
                auto result = node.set_parameters_atomically({rclcpp::Parameter("num_inference_steps",request->num_inference_steps)});
                response->success = result.successful;
                response->message = result.successful ? "Configured denoising steps for the next inference" : result.reason;
            });
        last_clock_ns = node.now().nanoseconds();
        timer = node.create_wall_timer(std::chrono::milliseconds(10),[this] { tick(); });
        RCLCPP_INFO(node.get_logger(),"TensorRT DP ready: base_link -> Link7, 16 actions at 30 Hz, denoise_steps=%d",requested_steps);
        worker = std::thread([this] { work(); });
    }

    ~Impl() {
        if (timer) timer->cancel();
        if (operation && operation->client) operation->client->remove_pending_request(operation->request_id);
        { std::lock_guard<std::mutex> lock(mutex); exiting = true; worker_job.reset(); }
        condition.notify_one();
        if (worker.joinable()) worker.join();
        if (guard && rclcpp::ok(node.get_node_base_interface()->get_context())) close_guard(false);
    }

    void on_image(const cv::Mat &rgb,int64_t stamp) {
        int64_t age = node.now().nanoseconds()-stamp;
        bool allowed = false;
        if (state == "running" && acknowledged && age >= 0 && age <= buffer->timeout_ns) {
            cv::Mat bgr; cv::cvtColor(rgb,bgr,cv::COLOR_RGB2BGR);
            allowed = ball_near_jaws(ball_center_bgr(bgr),bgr.cols,bgr.rows);
        }
        close_guard(allowed); last_image_ns = stamp;
        buffer->add_image(stamp,letterbox_rgb(rgb));
    }

    void add_service(const std::string &name,const std::string &action) {
        // Humble deferred-response 回调：不阻塞 executor，停机服务可抢占正在等候的操作。
        auto callback = [this,action](rclcpp::Service<Trigger>::SharedPtr service,
                                     std::shared_ptr<rmw_request_id_t> header,std::shared_ptr<Trigger::Request>) {
            handle(action,Reply{service,header});
        };
        services.push_back(node.create_service<Trigger>(name,callback));
    }

    void invalidate(const std::string &reason = "Operation was interrupted") {
        ++operation_id; buffer->reset(); pending.reset(); close_guard(false); last_image_ns = -1;
        if (operation) {
            auto old = std::move(*operation); operation.reset();
            if (old.client) old.client->remove_pending_request(old.request_id);
            respond(old.reply,false,reason);
        }
    }

    void begin(Kind kind,const Reply &reply,const rclcpp::Client<Reset>::SharedPtr &client,double timeout) {
        if (!require_controller) { finish(kind,reply,true,"Inference-only operation completed"); return; }
        if (!client->service_is_ready()) { finish(kind,reply,false,"Controller service is unavailable"); return; }
        auto request = std::make_shared<Reset::Request>(); request->episode_id = buffer->episode_id;
        try {
            auto future = client->async_send_request(request);
            auto id = future.request_id;
            operation = Operation{kind,reply,operation_id,client,future.share(),id,
                Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout))};
        } catch (const std::exception &e) { finish(kind,reply,false,e.what()); }
    }

    void finish(Kind kind,const Reply &reply,bool success,const std::string &message) {
        switch (kind) {
        case Kind::Start: state = success ? "running" : "idle"; acknowledged = success; break;
        case Kind::Stop: state = success ? "idle" : "fault"; acknowledged = success; break;
        case Kind::Reset: case Kind::ClockReset: acknowledged = success; break;
        case Kind::ReturnTaskStop:
            if (success) { state = "homing"; invalidate(); begin(Kind::ReturnStop,reply,reset_client,3.0); return; }
            state = "fault"; break;
        case Kind::ReturnStop:
            if (success) { state = "homing"; invalidate(); begin(Kind::ReturnHome,reply,return_client,155.0); return; }
            state = "fault"; break;
        case Kind::ReturnHome: state = success ? "idle" : "fault"; break;
        }
        respond(reply,success,message);
    }

    void handle(const std::string &action,const Reply &reply) {
        if (action == "reset" && !managed) {
            invalidate(); acknowledged = !require_controller;
            begin(Kind::Reset,reply,reset_client,reset_timeout_s); return;
        }
        if (action == "start") {
            if (state == "running") { respond(reply,true,"Task is already running"); return; }
            if (state != "idle") { respond(reply,false,"Cannot start while " + state); return; }
            if (!buffer->fresh(node.now().nanoseconds())) { respond(reply,false,"Image, joints or gripper feedback is stale"); return; }
            state = "starting"; invalidate(); acknowledged = !require_controller;
            begin(Kind::Start,reply,start_client,3.0); return;
        }
        if (action == "stop" || action == "reset") {
            if (state == "idle" && !require_controller) { respond(reply,true,"Task is already stopped"); return; }
            if (state == "stopping") { respond(reply,false,"Task stop is in progress"); return; }
            state = "stopping"; invalidate(); acknowledged = !require_controller;
            begin(Kind::Stop,reply,reset_client,3.0); return;
        }
        if (state == "homing" || state == "stopping" || state == "starting") {
            respond(reply,false,"Cannot return while " + state); return;
        }
        if (!require_controller) {
            state = "idle"; invalidate(); respond(reply,false,"Controller is unavailable"); return;
        }
        bool was_running = state != "idle";
        state = was_running ? "stopping" : "homing";
        invalidate(); acknowledged = false;
        begin(was_running ? Kind::ReturnTaskStop : Kind::ReturnStop,reply,reset_client,3.0);
    }

    void poll_operation() {
        if (!operation) return;
        bool ready = operation->future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        if (!ready && Clock::now() < operation->deadline) return;
        auto old = std::move(*operation); operation.reset();
        if (old.id != operation_id) { old.client->remove_pending_request(old.request_id); respond(old.reply,false,"Operation was superseded"); return; }
        if (!ready) { old.client->remove_pending_request(old.request_id); finish(old.kind,old.reply,false,"Controller service timed out"); return; }
        try { auto response = old.future.get(); finish(old.kind,old.reply,response->success,response->message); }
        catch (const std::exception &e) { finish(old.kind,old.reply,false,e.what()); }
    }

    void work() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock,[this] { return exiting || worker_job.has_value(); });
                if (exiting) return;
                job = std::move(*worker_job); worker_job.reset();
            }
            Completed result; result.context = job.context; result.steps = job.steps;
            auto start = Clock::now();
            try {
                result.sequence = decode_actions(backend->infer(job.observations,job.steps),job.context);
                for (auto &processor : processors) { processor->process(*result.sequence,job.context); validate_sequence(*result.sequence); }
            } catch (const std::exception &e) { result.error = e.what(); }
            catch (...) { result.error = "Unknown inference/postprocessor exception"; }
            result.inference_ms = std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            { std::lock_guard<std::mutex> lock(mutex); completed = std::move(result); }
        }
    }

    void tick() {
        int64_t now = node.now().nanoseconds();
        if (now < last_clock_ns) {
            invalidate("ROS clock moved backwards"); acknowledged = !require_controller;
            if (managed) state = "fault";
            begin(Kind::ClockReset,Reply{},reset_client,reset_timeout_s);
            warn("ROS clock moved backwards; episode reset");
        }
        last_clock_ns = now;
        poll_operation();
        if (last_image_ns < 0 || now < last_image_ns || now-last_image_ns > buffer->timeout_ns) close_guard(false);
        std::optional<Completed> result;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (completed) { result = std::move(completed); completed.reset(); busy = false; }
        }
        if (result && (!managed || state == "running") && acknowledged) {
            int64_t age = now-result->context.stamp_ns;
            if (!result->error.empty()) warn("Inference/postprocessing failed: " + result->error);
            else if (result->context.episode_id != buffer->episode_id) warn("Dropped prediction: episode changed");
            else if (age < 0 || age > std::llround(result_timeout_s*1e9)) warn("Dropped prediction: result expired");
            else {
                publisher->publish(sequence_message(*result->sequence,result->context));
                RCLCPP_INFO(node.get_logger(),"Published sequence=%lu episode=%lu actions=16 denoise_steps=%d inference_ms=%.1f age_ms=%.1f",
                    result->context.sequence_id,result->context.episode_id,result->steps,result->inference_ms,age/1e6);
            }
        }
        // stop_task 在独立启动模式中也暂停发布；显式 start_task 可恢复。
        if (state != "running" || !acknowledged) return;
        auto history = buffer->poll(now);
        if (history) {
            InferenceContext context{(*history)[1].pose,(*history)[1].stamp_ns,buffer->episode_id,0};
            pending = Job{build_observations(*history,*buffer->start_pose),context,requested_steps};
        }
        if (busy || !pending) return;
        auto monotonic = Clock::now();
        if (last_started != Clock::time_point::min() && std::chrono::duration<double>(monotonic-last_started).count() < period_s) return;
        int64_t age = now-pending->context.stamp_ns;
        if (age < 0 || age > buffer->timeout_ns) { pending.reset(); return; }
        pending->context.sequence_id = ++sequence_id;
        pending->steps = requested_steps;
        {
            std::lock_guard<std::mutex> lock(mutex);
            worker_job = std::move(pending); pending.reset(); busy = true;
        }
        last_started = monotonic; condition.notify_one();
    }
};

DpInferenceNode::DpInferenceNode(const rclcpp::NodeOptions &options,std::shared_ptr<InferenceBackend> backend)
    : rclcpp::Node("dp_infer_tensorrt",options),impl_(std::make_unique<Impl>(*this,std::move(backend))) {}
DpInferenceNode::~DpInferenceNode() = default;
}  // namespace dp_infer_tensorrt
