<!-- 本文档用于说明独立 stereo_camera SDK 元数据观察示例的构建、运行和输出语义。 -->

# stereo_camera SDK 示例

此目录包含一个不依赖 ROS、OpenCV 或现有驱动封装的 C++17 示例。程序直接调用上级 `sdk/` 中的公共头文件和动态库，观察 SDK 返回的帧元数据、逐路 `stereo_camera_parse_frame` 回调速率以及 IMU 样本率。程序会按公共头文件定义安全复制并解析 IMU payload。可选显示窗口通过 SDL2/OpenGL 直接上传原始图像 payload，由 GPU shader 解释 YUYV、NV12 和后备的 iTOF 16 位单通道数据；CPU 不生成 RGB/BGR 中间图像，程序也不会保存像素 payload。

## 构建

显示功能需要 SDL2 和 OpenGL 开发包。在 Ubuntu 上安装：

```bash
sudo apt install libsdl2-dev libgl-dev
```

在仓库根目录执行：

```bash
cmake -S ros2_ws/src/tof_stereo_camera/sdk/example \
      -B /tmp/tof_stereo_sdk_example_build
cmake --build /tmp/tof_stereo_sdk_example_build --parallel 2
```

构建产物为 `/tmp/tof_stereo_sdk_example_build/stereo_camera_example`。CMake 会检查相邻的 `../include/stereo_camera/stereo_camera.h` 和 `../lib/libstereo_camera.so`，并写入构建运行所需的 RPATH。

## 运行

```bash
/tmp/tof_stereo_sdk_example_build/stereo_camera_example
```

默认参数是由 SDK 自动选择设备、主码流 RGB `2048x1536`、启用 iTOF、`YUYV`：

```text
--device PATH
--stream-profile NAME
--format FOURCC
--enable-itof BOOL
--display
--verbose
--help
```

码流档位固定为 `main=2048x1536` 和 `sub=1920x1080`。例如指定设备、子码流和格式：

```bash
/tmp/tof_stereo_sdk_example_build/stereo_camera_example \
    --device /dev/video3 --stream-profile sub --format YUYV
```

程序不提供 `--width` 和 `--height` 自定义分辨率参数，避免请求设备不支持的码流组合。启用 iTOF 时，主、子码流分别协商完整复合帧 `2048x2738` 和 `1920x2362`；关闭 iTOF 时，设备会在 RGB 高度中加入 2 行复合帧头部，实际协商尺寸分别为 `2048x1538` 和 `1920x1082`。SDK 解析后的 RGB 尺寸始终为 `2048x1536` 和 `1920x1080`。

`--help` 只打印用法并在打开相机前成功退出，因此无硬件时也可以验证命令行接口。参数错误会打印错误和用法并返回非零状态。

### 启用或关闭 iTOF

默认启用 iTOF Depth 和 Gray，并下发视频流掩码 `0x85`。如需只启用 RGB，设置 `--enable-itof false`：

```bash
/tmp/tof_stereo_sdk_example_build/stereo_camera_example --enable-itof false
```

视频流掩码不控制 IMU，因此两种模式下设备都会推送 IMU 数据。`--enable-itof` 同时决定采集格式、视频流掩码和启用显示时的窗口布局，与是否创建显示窗口无关。

### 原始格式 GPU 显示

使用默认参数显示 RGB、iTOF Depth 和 iTOF Gray：

```bash
/tmp/tof_stereo_sdk_example_build/stereo_camera_example --display
```

使用 NV12 RGB 输入：

```bash
/tmp/tof_stereo_sdk_example_build/stereo_camera_example --display --format NV12
```

`--display` 只控制是否创建显示窗口。默认 `--enable-itof true` 时，窗口上半区显示 RGB，左下显示 iTOF Depth，右下显示 iTOF Gray；设置 `--enable-itof false` 时，RGB 使用全窗口单窗格布局。关闭窗口或按 `Esc` 会停止采集并正常清理 SDK 资源。各路图像依据实际 FOURCC 按 YUYV packed 或 NV12 双平面布局直接上传 GPU；shader 只负责屏幕颜色或灰度解释，不在 CPU 内存中创建 RGB/BGR 中间图像。

## 输出语义

程序按以下 SDK 生命周期运行：

```text
stereo_camera_open
  -> stereo_camera_set_format
  -> stereo_camera_get_format
  -> stereo_camera_start_stream
  -> stereo_camera_parse_frame 循环
  -> stereo_camera_stop_stream
  -> stereo_camera_close
```

启动后先打印实际协商的宽度、高度和 FOURCC。每个 `sourcetype + stream_id` 第一次出现时打印一行原始 `stereo_camera_frame_t` 字段，包括 `sourcetype`、`stream_id`、`frame_seqidx`、`frame_timestamp`（微秒）、`width`、`height`、`FOURCC`、`data_size`（字节）、`match_state` 和 `frame_seq_count`。

新版 `stereo_camera_imu_data_t` 为 72 字节紧凑结构，字段顺序为 `timestamp`、`idx`、`ax/ay/az`、`gx/gy/gz`、`reverve`。示例在编译期检查关键字段偏移，运行时通过 `memcpy` 逐样本解码，并校验 payload 长度、`frame_seq_count`、`frame_seqidx` 和首样本 `idx`。异常批次会输出去重后的 `invalid IMU batch` 警告，结构无效的 payload 不计入样本数。

首次收到非空 IMU 批次时，程序以 `[imu first]` 前缀打印所有样本。每条样本包含 `timestamp`（微秒）、`idx`、`ax/ay/az`（m/s²）和 `gx/gy/gz`（deg/s，度/秒）。

每约一秒打印窗口统计。`PARSE FPS` 表示主机 `std::chrono::steady_clock` 窗口内 `stereo_camera_parse_frame` 成功返回的路由帧数除以实际窗口时长；`RGB FPS`、`ITOF DEPTH FPS` 和 `ITOF GRAY FPS` 分别表示对应路由的成功帧数除以实际窗口时长。`IMU FPS` 使用成功解码的 IMU 样本数除以实际窗口时长。这些结果是主机接收侧测得的窗口平均值，不等同于设备声明帧率。
当 SDK 的 `stereo_camera_parse_frame` 返回空指针且程序仍在运行时，示例会将其计入当前窗口的 `null_returns`。单个偶发空返回立即重试，不等待；连续空返回阶段每次重试前退避 1ms。仅当距最近成功帧（尚无成功帧时从启动时刻计时）至少 1 秒，才按主机 `std::chrono::steady_clock` 最多约每秒输出一次 `no frame/error` 提示；成功帧会重置连续失败计数和最近成功时间。`[stats]` 头行会显示窗口内空返回数，所有 FPS 只统计成功读取的数据。

使用 `--verbose` 时，每次 `stereo_camera_parse_frame` 成功返回都会打印上述原始字段，IMU 批次还会以 `[imu]` 前缀打印全部已解码样本。终端 I/O 会影响极限 FPS 测量，需观察最高速率时应保持默认输出或重定向输出。

收到 `Ctrl-C`、终止信号、Esc 或窗口关闭事件后，主线程会调用 `stereo_camera_cancel_read` 唤醒读取线程中阻塞的 `stereo_camera_parse_frame`，等待读取线程退出，再调用 `stereo_camera_stop_stream` 和 `stereo_camera_close` 完成正常清理。读取线程每次只提交一个结果，并等待主线程处理完成后再继续解析，确保 SDK 返回的 frame 指针在使用期间保持有效。
