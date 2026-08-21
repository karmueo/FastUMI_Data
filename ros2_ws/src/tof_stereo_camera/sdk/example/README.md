<!-- 本文档用于说明独立 stereo_camera SDK 元数据观察示例的构建、运行和输出语义。 -->

# stereo_camera SDK 示例

此目录包含一个不依赖 ROS、OpenCV 或现有驱动封装的 C++17 示例。程序直接调用上级 `sdk/` 中的公共头文件和动态库，观察 SDK 返回的帧元数据、逐路 `stereo_camera_parse_frame` 回调速率以及 IMU 样本率。可选显示窗口通过 SDL2/OpenGL 直接上传原始 payload，由 GPU shader 解释 YUYV、NV12 和后备的 iTOF 16 位单通道数据；CPU 不生成 RGB/BGR 中间图像，程序也不会保存像素 payload。

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

默认参数是由 SDK 自动选择设备、复合帧 `2048x2738`、`YUYV`：

```text
--device PATH
--width N
--height N
--format FOURCC
--display
--rgb-only
--depth-max N
--gray-max N
--verbose
--help
```

例如指定设备和格式：

```bash
/tmp/tof_stereo_sdk_example_build/stereo_camera_example \
    --device /dev/video3 --width 2048 --height 2738 --format YUYV
```

`--help` 只打印用法并在打开相机前成功退出，因此无硬件时也可以验证命令行接口。参数错误会打印错误和用法并返回非零状态。

### 仅启用 RGB 视频流

使用 `--rgb-only` 会在启动采集前下发视频流掩码 `0x1`，只启用 RGB（bit 0），不推送 iTOF Depth 和 iTOF Gray：

```bash
/tmp/tof_stereo_sdk_example_build/stereo_camera_example --rgb-only
```

视频流掩码不控制 IMU，因此该模式下设备仍会推送 IMU 数据。与 `--display` 同时使用时，窗口只显示 RGB，并使用完整窗口区域。

### 原始格式 GPU 显示

使用默认 YUYV 显示 RGB、iTOF Depth 和 iTOF Gray：

```bash
/tmp/tof_stereo_sdk_example_build/stereo_camera_example --display
```

使用 NV12 RGB 输入：

```bash
/tmp/tof_stereo_sdk_example_build/stereo_camera_example --display --format NV12
```

窗口上半区为 RGB，左下为 iTOF Depth，右下为 iTOF Gray。`--display` 会在启动采集前显式下发视频流掩码 `0x85`，启用 RGB（bit 0）、iTOF Depth（bit 2）和 iTOF Gray（bit 7），设备拒绝命令时会打印 SDK 返回值和 ACK 后退出。关闭窗口或按 `Esc` 会停止采集并正常清理 SDK 资源。RGB、Depth 和 Gray 均优先依据实际 FOURCC 按 YUYV packed 或 NV12 双平面布局直接上传 GPU；非 YUV 的 iTOF payload 使用 16 位无符号单通道纹理后备路径。shader 仅为屏幕显示进行颜色解释或灰度映射，不在 CPU 内存中创建 RGB/BGR 图像。

当其他固件返回非 YUV 的 16 位单通道 iTOF 数据时，深度后备路径默认以 `5000` 为显示上限并采用近处更亮的灰度映射，灰度后备路径默认以 `65535` 为显示上限。画面过暗或过亮时可以调整原始值范围，例如：

```bash
/tmp/tof_stereo_sdk_example_build/stereo_camera_example \
    --display --depth-max 8000 --gray-max 4095
```

当前实测设备的 iTOF Depth 和 Gray 均为 YUYV，两个范围参数不会参与其显示。首帧诊断会打印 16 位误读范围，用于识别 YUYV 中固定约为 `0x80` 的 U/V 色度字节；该扫描每路只执行一次。

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

每约一秒打印窗口统计。`PARSE FPS` 表示主机 `std::chrono::steady_clock` 窗口内 `stereo_camera_parse_frame` 成功返回的路由帧数除以实际窗口时长；`RGB FPS`、`ITOF DEPTH FPS` 和 `ITOF GRAY FPS` 分别表示对应路由的成功帧数除以实际窗口时长。`IMU FPS` 表示窗口内 IMU 样本数除以实际窗口时长；样本数优先采用 `frame_seq_count`，该字段为零且 `data_size` 可推导时才按 SDK 结构体大小作后备估计。这些结果是主机接收侧测得的窗口平均值，不等同于设备声明帧率。统计只读取返回结构体元数据，不访问 payload。
当 SDK 的 `stereo_camera_parse_frame` 返回空指针且程序仍在运行时，示例会将其计入当前窗口的 `null_returns`。单个偶发空返回立即重试，不等待；连续空返回阶段每次重试前退避 1ms。仅当距最近成功帧（尚无成功帧时从启动时刻计时）至少 1 秒，才按主机 `std::chrono::steady_clock` 最多约每秒输出一次 `no frame/error` 提示；成功帧会重置连续失败计数和最近成功时间。`[stats]` 头行会显示窗口内空返回数，所有 FPS 只统计成功读取的数据。

使用 `--verbose` 时，每次 `stereo_camera_parse_frame` 成功返回都会打印上述原始字段。终端 I/O 会影响极限 FPS 测量，需观察最高速率时应保持默认输出或重定向输出。

收到 `Ctrl-C`、终止信号、Esc 或窗口关闭事件后，主线程会调用 `stereo_camera_cancel_read` 唤醒读取线程中阻塞的 `stereo_camera_parse_frame`，等待读取线程退出，再调用 `stereo_camera_stop_stream` 和 `stereo_camera_close` 完成正常清理。读取线程每次只提交一个结果，并等待主线程处理完成后再继续解析，确保 SDK 返回的 frame 指针在使用期间保持有效。
