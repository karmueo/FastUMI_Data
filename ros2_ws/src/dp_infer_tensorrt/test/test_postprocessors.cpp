// 用真实 pluginlib 加载验证顺序、输出校验和异常处理。
#include "dp_infer_tensorrt/core.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <stdexcept>

namespace dp_infer_tensorrt {
class TestShift : public Postprocessor {
public:
    void process(ActionSequence &sequence,const InferenceContext &) override {
        for (auto &p : sequence.positions) p.x() += .01;
    }
};
class TestInvalid : public Postprocessor {
public:
    void process(ActionSequence &sequence,const InferenceContext &) override { sequence.time_from_start[0] = 1; }
};
class TestThrow : public Postprocessor {
public:
    void process(ActionSequence &,const InferenceContext &) override { throw std::runtime_error("test plugin failure"); }
};
}
PLUGINLIB_EXPORT_CLASS(dp_infer_tensorrt::TestShift,dp_infer_tensorrt::Postprocessor)
PLUGINLIB_EXPORT_CLASS(dp_infer_tensorrt::TestInvalid,dp_infer_tensorrt::Postprocessor)
PLUGINLIB_EXPORT_CLASS(dp_infer_tensorrt::TestThrow,dp_infer_tensorrt::Postprocessor)
