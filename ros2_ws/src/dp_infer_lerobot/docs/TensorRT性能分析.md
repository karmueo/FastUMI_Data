# dp_infer_lerobot TensorRT 后端性能分析（Jetson AGX Orin，2026-10-28）

本报告对比 `dp_infer_lerobot` 三种推理后端在 Orin 上的延迟、分阶段耗时、精度与 ROS 实时性：

- `pytorch`：FP32 权重，卷积默认允许 TF32，开启 `cudnn.benchmark`；
- `tensorrt_fp32`：TensorRT FP32 engine，关闭 TF32；
- `tensorrt_fp16`：TensorRT FP16 engine，I/O 仍为 FP32。

模型为 `dp_umi_v4` 的 `012000`：ResNet18 编码器加 U-Net `[512, 1024, 2048]`，16×10 动作序列，DDIM 去噪。
原始数据（JSON/日志）保存在被 Git 忽略的 `dataset/validation/dp_lerobot_v4/`：
`build_trt_orin.log`、`benchmark_orin.json`、`smoke_trt_fp16_ddim16.json`。

## 1. 结论

| 指标 | PyTorch | TRT FP32 | TRT FP16 |
| --- | --- | --- | --- |
| DDIM16 单次推理（中位） | 648.6 ms | 227.6 ms（2.9×） | **79.8 ms（8.1×）** |
| DDIM16 单步 U-Net | 42.3 ms | 13.8 ms | 4.5 ms |
| 相对 PyTorch 最大位置 / 旋转误差（DDIM16） | — | 0.014 mm / 0.002° | 0.21 mm / 0.023° |
| 加载 + 首次推理 | 5.0 s + 1146 ms | 2.3 s + 77 ms | 2.0 s + 26 ms |

- **建议部署 `tensorrt_fp16` + DDIM16。** 实测比 PyTorch 快 8.1 倍，动作误差远低于验收门槛（1 mm / 0.5° / 0.01）。
- 流水线模式下 FP16 DDIM16 的 ROS 冒烟结果：9.4 Hz，结果年龄最大 165 ms，小于 8 个动作覆盖的 233 ms 预算。
  PyTorch 只能用 2～4 步满足这一预算。
- `tensorrt_fp32` 的精度与 PyTorch 几乎一致，但 DDIM16 需要 228 ms，加上观测同步后超出流水线预算，
  只适合串行模式或 ≤8 步。
- 在 FP16 后端下，U-Net 不再是唯一瓶颈。DDIM16 时，预处理加后处理约 10.6 ms，调度器及其余开销约 15 ms，
  合计约占 1/3（见第 4 节）。

## 2. 测试环境与方法

| 项 | 值 |
| --- | --- |
| 硬件 | Jetson AGX Orin 64 GB（GPU `Orin`，计算能力 8.7） |
| 系统 | L4T R36.4.7（JetPack 6.2.1），功耗模式 MAXN，GPU 频率锁定 1300.5 MHz |
| 软件 | TensorRT 10.3.0（系统包）、torch 2.8.0、CUDA 12.6、diffusers 0.35.2 |
| 调度器 | DDIM（`clip_sample=True`，与训练 DDPM 同噪声表） |
| 输入 | 腕部相机视频 `jingbao/.../file-000.mp4` 的相邻帧对（224×224）+ 合成 TCP 状态 |

测量方法：

- 基准由 `tools/benchmark_policy.py` 运行。每个后端都经 `PolicyEngine` 正式加载，TensorRT 包须通过 `validate_export`。
- 每个步数先预热 3 次，再测 30 次端到端 `predict_raw` 延迟（numpy 输入 → CPU 绝对动作）。
- 分阶段耗时单独测一轮（10 次）：每个阶段前后各同步一次 CUDA。同步打断了 CPU 下发与 GPU 执行的重叠，
  所以各阶段之和略大于端到端延迟（PyTorch DDIM16 约多 70 ms）。

## 3. engine 构建（`tools/build_tensorrt_engines.py`）

源包 `lerobot_onnx` 中的 `.engine` 由 RTX 5090 / TRT 10.13 构建。engine 与 GPU 型号及 TensorRT 版本绑定，
Orin 上不能加载：节点会报 `device/version mismatch; rebuild on this device` 并拒绝启动（已实测）。
因此在本机从源包的 ONNX（IR 8，opset 18；`denoiser.onnx` 为 1.0 GB）重建：

| engine | 构建耗时 | 大小 |
| --- | --- | --- |
| encoder FP16 | 22.5 s | 21.7 MiB |
| denoiser FP16 | 71.6 s | 482.0 MiB |
| encoder FP32 | 5.9 s | 71.0 MiB |
| denoiser FP32 | 16.1 s | 962.5 MiB |

- 全部构建约 2 分钟，workspace 上限 8 GiB。FP16 先构建，FP32 复用了同一 timing cache，FP16 需要搜索更多内核。
- 新包（`lerobot_trt_orin`）只包含 assets 和本机 engine，不复制 ONNX。manifest 用 `source_package` 和 `source_onnx` 记录来源摘要。
- `validation_report.json` 由脚本生成，以本包 PyTorch 后端为基准。PyTorch 后端在 x86 上已验证与 LeRobot 原版逐位一致。

## 4. 延迟

### 4.1 端到端（中位 / P95，括号内为相对 PyTorch 的加速比）

| 后端 | DDIM2 | DDIM4 | DDIM8 | DDIM16 |
| --- | --- | --- | --- | --- |
| pytorch | 100.0 / 100.5 ms | 178.2 / 178.7 ms | 337.0 / 337.9 ms | 648.6 / 649.5 ms |
| tensorrt_fp32 | 40.9 / 41.2 ms（2.4×） | 67.8 / 68.4 ms（2.6×） | 121.0 / 122.6 ms（2.8×） | 227.6 / 227.9 ms（2.9×） |
| tensorrt_fp16 | 21.1 / 21.5 ms（4.7×） | 29.5 / 30.0 ms（6.0×） | 46.4 / 46.8 ms（7.3×） | 79.8 / 80.3 ms（8.1×） |

三种后端的抖动都很小：P95 与中位之差 ≤ 1.6 ms，最大值 ≤ 中位 + 2 ms。
步数越少加速比越低，因为与步数无关的固定开销（预处理、后处理、编码器）占比上升。

### 4.2 分阶段（DDIM16，每次推理平均，ms）

| 阶段 | PyTorch | TRT FP32 | TRT FP16 |
| --- | --- | --- | --- |
| 预处理（归一化、相对位姿） | 4.2 | 4.3 | 4.2 |
| 编码器（ResNet18 ×2 帧） | 7.2 | 2.9 | 1.1 |
| U-Net 单步 | 42.3 | 13.8 | 4.5 |
| U-Net ×16 | 676.7 | 220.4 | 71.5 |
| 调度器及其余开销 | 22.0 | 20.1 | 15.4 |
| 后处理（反归一化、位姿还原、拷回 CPU） | 6.3 | 6.4 | 6.4 |

- U-Net 加速：FP32 为 3.1×，FP16 为 9.5×。FP16 能把 U-Net 中的大卷积放到 Tensor Core 上执行。
- 每步约 1 ms 的“调度器及其余开销”包括：diffusers `DDIMScheduler.step` 的十余个小 kernel、`torch.full` 构造时间步，以及 engine 入队与流同步。在 FP16 DDIM16 下它约占 19%。
- 预处理和后处理与后端无关，合计约 10.6 ms。在 FP16 DDIM2 下占一半（10.6 / 21.1 ms）。

### 4.3 加载与内存

| 后端 | 加载 | 首次推理 | 模型显存（Orin 共享内存） |
| --- | --- | --- | --- |
| pytorch | 5.0 s | 1146 ms（cuDNN 算法搜索） | torch 峰值约 1.16 GiB |
| tensorrt_fp32 | 2.3 s | 77 ms | engine 约 1.0 GiB，torch 仅 12.6 MiB（I/O） |
| tensorrt_fp16 | 2.0 s | 26 ms | engine 约 0.5 GiB，torch 仅 12.6 MiB |

- TensorRT 的加载时间包含对约 1 GB engine 计算 SHA-256（模型包校验）。
- `mem_get_info` 在 Orin 统一内存上受页缓存和其他进程影响，不作为精确数值；engine 文件大小可近似代表权重占用。
- 节点启动时会预热，所以首次推理耗时不影响运行中的结果。

## 5. 精度

### 5.1 构建验收（`validation_report.json`）

条件：DDIM16，8 个视频帧样本 × 3 个噪声种子，最终绝对动作（base→TCP），相对本包 PyTorch。

| 后端 | 位置 mean / P95 / max (mm) | 旋转 max (°) | 夹爪 max | 归一化动作最大差 | 结果 |
| --- | --- | --- | --- | --- | --- |
| tensorrt_fp32 | 0.0045 / 0.0087 / 0.0135 | 0.0024 | 7.6e-5 | 3.1e-4 | 通过 |
| tensorrt_fp16 | 0.065 / 0.121 / 0.208 | 0.023 | 7.9e-4 | 3.2e-3 | 通过 |

### 5.2 随步数变化（基准，4 个样本，max）

| DDIM 步数 | FP32 位置 / 旋转 / 夹爪 | FP16 位置 / 旋转 / 夹爪 |
| --- | --- | --- |
| 2 | 0.029 mm / 0.0069° / 4.4e-5 | 0.42 mm / 0.061° / 1.1e-3 |
| 4 | 0.012 mm / 0.0019° / 1.3e-5 | 0.16 mm / 0.023° / 8.7e-4 |
| 8 | 0.0086 mm / 0.0015° / 2.0e-5 | 0.15 mm / 0.015° / 6.5e-4 |
| 16 | 0.014 mm / 0.0016° / 2.4e-5 | 0.14 mm / 0.017° / 6.5e-4 |

步数越少，每一步 U-Net 输出的误差直接进入最终样本，误差随之变大：FP16 在 DDIM2 时约 0.4 mm。
所有组合都在验收门槛以内。

### 5.3 与工作站 5090 导出验证对照

源包报告的条件是 DDPM 100 步、相对 LeRobot 原版 PyTorch，结果原样收录在新包报告的 `source_validation` 字段：

| 后端 | 5090 / TRT 10.13 位置 / 旋转 max | Orin / TRT 10.3 位置 / 旋转 max（DDIM16） |
| --- | --- | --- |
| TRT FP32 | 0.000087 mm / 0.000011° | 0.0135 mm / 0.0024° |
| TRT FP16 | 0.145 mm / 0.022° | 0.208 mm / 0.023° |

- 两台机器上 FP16 的误差处于同一量级，说明 Orin 重建的 engine 没有引入额外的数值问题。
- Orin 上 FP32 的差值比 5090 大约两个数量级，但仍只有 0.01 mm。推测原因是 Orin 的 PyTorch 参考开启了
  `cudnn.benchmark`，卷积选到了 TF32 算法，而 TRT FP32 关闭了 TF32。
- 两边的调度器与输入不同（DDPM100 + 数据集样本，对比 DDIM16 + 视频帧），只能作量级对照。

## 6. ROS 实时性（`test/smoke_dp_infer_lerobot.py`，不接硬件）

| 配置 | 发布频率 | 节点内推理（中位 / 最大） | 结果年龄（中位 / 最大） | 重置后旧消息 |
| --- | --- | --- | --- | --- |
| PyTorch DDIM2 流水线（历史记录） | 9.0 Hz | 108 / 669 ms（含首次 cuDNN 搜索） | 164 / 198 ms | 0 |
| **TRT FP16 DDIM16 流水线** | 9.4 Hz | 90 / 99 ms | 164 / 165 ms | 0 |

- 结果年龄由三部分组成：等待两帧同步、排队（最高 10 Hz 提交）和推理本身。FP16 DDIM16 的年龄与 PyTorch DDIM2 相当，但去噪步数是后者的 8 倍。
- 串行模式下，每段周期为推理、执行 8 个动作（267 ms）与稳定时间（0.3 s）之和。FP16 DDIM16 的周期约 0.65 s，PyTorch 约 1.2 s。

## 7. 部署建议与后续优化

1. 实机使用 `backend:=tensorrt_fp16 export_dir:=<lerobot_trt_orin>`，保持 DDIM16。参数文件默认仍为 `pytorch`，本次不改默认值。
2. 先在 dry-run 中核对结果年龄与输出位姿，再考虑 `sequential_execution:=false`（流水线）。流水线下 FP16 有余量，FP32 只能用 ≤8 步。
3. 更换 checkpoint、升级 JetPack/TensorRT 或更换设备后，必须重新运行 `build_tensorrt_engines.py`。
   launch 会拒绝 GPU、TensorRT 版本或验证报告不匹配的模型包。
4. 可选优化，按收益排序：
   - 用 CUDA Graph 捕获 FP16 去噪循环（engine 加调度器），去掉每步约 1 ms 的下发开销，DDIM16 预计可再省约 15 ms；
   - 把预处理和后处理中的小算子合并，或移到 CPU/NumPy，目前共约 10.6 ms；
   - 用 FP16 编码器搭配 FP32 U-Net 的混合方案意义不大，因为编码器只占 1～3 ms。
