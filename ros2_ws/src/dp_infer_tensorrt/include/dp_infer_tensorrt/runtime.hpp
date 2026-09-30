#pragma once
// 单工作线程专用的 TensorRT 运行时，管理静态 CUDA 缓冲区。
#include "dp_infer_tensorrt/core.hpp"

namespace dp_infer_tensorrt {
struct RuntimeOptions {
    std::filesystem::path engine_dir;
    std::filesystem::path model_manifest;
    std::string precision = "fp16";
    int device = 0;
};

struct InferenceTiming {
    double encoder_gpu_ms = 0;
    double denoiser_gpu_ms = 0;  // 全部去噪步骤的 GPU 执行耗时。
    double end_to_end_ms = 0;  // CPU 输入到 CPU 动作，包含传输与 DDIM。
};

class TensorRtBackend final : public InferenceBackend {
public:
    explicit TensorRtBackend(const RuntimeOptions &options);
    ~TensorRtBackend() override;
    RawAction infer(const Observations &observations, int steps,
                    const RawAction *initial_noise = nullptr) override;
    void warmup(int steps) override;
    std::string urdf_sha256() const override;
    /// 离线工具输出本次 condition 和耗时；与常规节点使用相同推理核心。
    std::vector<float> condition() const;
    InferenceTiming last_timing() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace dp_infer_tensorrt
