// 保持 FP32 运算顺序；禁用 FMA，避免与 diffusers 的分步算子产生额外误差。
#include "ddim.cuh"
#include <cuda_runtime.h>

namespace dp_infer_tensorrt {
__global__ void ddim_kernel(float *sample,const float *noise,float alpha,float beta,float prev_alpha,float prev_beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 160) return;
    float original = (sample[i] - beta * noise[i]) / alpha;
    original = fminf(1.0f,fmaxf(-1.0f,original));
    sample[i] = prev_alpha * original + prev_beta * noise[i];
}
__global__ void unnormalize_kernel(const float *sample,float *output,const float *scale,const float *offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 160) output[i] = (sample[i]-offset[i%10])/scale[i%10];
}
__global__ void finite_kernel(const float *values,int count,int *invalid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && !isfinite(values[i])) atomicExch(invalid,1);
}
void check_finite(const float *values,int count,int *invalid,cudaStream_t stream) {
    finite_kernel<<<(count+255)/256,256,0,stream>>>(values,count,invalid);
}
void ddim_update(float *sample,const float *noise,float alpha,float beta,float prev_alpha,float prev_beta,cudaStream_t stream) {
    ddim_kernel<<<1,256,0,stream>>>(sample,noise,alpha,beta,prev_alpha,prev_beta);
}
void unnormalize_action(const float *sample,float *output,const float *scale,const float *offset,cudaStream_t stream) {
    unnormalize_kernel<<<1,256,0,stream>>>(sample,output,scale,offset);
}
}
