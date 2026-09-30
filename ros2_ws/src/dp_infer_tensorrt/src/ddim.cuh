#pragma once
#include <cuda_runtime_api.h>

namespace dp_infer_tensorrt {
void ddim_update(float *sample, const float *noise, float sqrt_alpha, float sqrt_beta,
                 float sqrt_prev_alpha, float sqrt_prev_beta, cudaStream_t stream);
void unnormalize_action(const float *sample, float *output, const float *scale,
                        const float *offset, cudaStream_t stream);
void check_finite(const float *values, int count, int *invalid, cudaStream_t stream);
}
