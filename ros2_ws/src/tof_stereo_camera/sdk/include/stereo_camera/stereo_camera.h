// Copyright 2024 UVC Host Reader Project
//
// stereo_camera — 极简 C 风格 UVC 相机库接口
//
// 使用方式:
//   #include "stereo_camera/stereo_camera.h"
//
// 典型用法 (自动检测相机):
//   stereo_camera_t *cam = stereo_camera_open(NULL);
//   stereo_camera_set_format(cam, 1280, 720, "YUYV");
//   stereo_camera_start_stream(cam);
//   ...
//   stereo_camera_frame_t *frame = nullptr;
//   // 逐帧获取各路流数据，同一帧UVC的所有流遍历完后自动读取下一帧
//   while ((frame = stereo_camera_parse_frame(cam)) != nullptr) {
//     /* 处理 frame->data, frame->frame_timestamp, frame->match_state */
//   }
//   ...
//   stereo_camera_stop_stream(cam);
//   stereo_camera_close(cam);
//
// 典型用法 (指定设备路径):
//   stereo_camera_t *cam = stereo_camera_open("/dev/video3");
//
// 典型用法 (枚举设备):
//   stereo_camera_device_info_t devices[32];
//   int n = stereo_camera_enumerate_devices(devices, 32);
//   for (int i = 0; i < n; i++) {
//     printf("[%d] %s  %s\n", i, devices[i].path, devices[i].card);
//   }

#ifndef STEREO_CAMERA_H_
#define STEREO_CAMERA_H_

#include <stddef.h>
#include <stdint.h>

// 符号可见性控制
#if defined(_MSC_VER)
#ifdef STEREO_CAMERA_EXPORTS
#define STEREO_CAMERA_API __declspec(dllexport)
#else
#define STEREO_CAMERA_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define STEREO_CAMERA_API __attribute__((visibility("default")))
#else
#define STEREO_CAMERA_API
#endif

// 结构体紧凑对齐控制（用于 IMU 等需要与设备端二进制一致的结构）
#if defined(_MSC_VER)
#define STEREO_CAMERA_PACKED_BEGIN __pragma(pack(push, 1))
#define STEREO_CAMERA_PACKED_END __pragma(pack(pop))
#else
#define STEREO_CAMERA_PACKED_BEGIN
#define STEREO_CAMERA_PACKED_END __attribute__((packed))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ═══ UVC XU（扩展单元）命令 ═══════════════════════════════
//
// 经 UVC 扩展单元（Extension Unit，实体 ID=6）向设备下发自定义命令，
// 每一条命令均有设备端 ACK 结果确认。
//
// 统一命令通道 selector（内核 f_uvc.c 已声明 selector 1 = bmControls bit0，
// 无需修改内核描述符；uvcvideo 对未声明 selector 返回 ENOENT）：
#define STEREO_CAMERA_XU_SELECTOR_CMD 1
// 通道负载固定长度（设备端 GET_LEN 返回；uvcvideo 强制 SET_CUR/GET_CUR 的
// wLength == GET_LEN 值，故负载恒定 32B，不足补 0）：
#define STEREO_CAMERA_XU_CMD_SIZE 32
// 负载布局（小端 LE）：[0] sub_cmd u8 + [1..27] param + [28..31] result u32(ACK)
#define STEREO_CAMERA_XU_CMD_OFF_RESULT (STEREO_CAMERA_XU_CMD_SIZE - 4)
// 通道内子命令 ID（负载首字节 sub_cmd；单通道可承载 256 个子命令，
// 数量不受内核 selector 位图限制）：
#define STEREO_CAMERA_XU_SUBCMD_IMU_FREQ 1     // 设定 IMU 采样频率
#define STEREO_CAMERA_XU_SUBCMD_STREAM_MASK 2  // 指定推流视频流 ID 掩码

// 设备端 ACK 结果码（由 GET_CUR 回读状态尾部 4 字节提取）：
#define STEREO_CAMERA_XU_ACK_OK 0      // 设定成功
#define STEREO_CAMERA_XU_ACK_EINVAL 1  // 参数非法
#define STEREO_CAMERA_XU_ACK_EIO 2     // 设备执行失败

// ── 不透明句柄 ─────────────────────────────────────────
// 相机实例句柄，所有操作通过此句柄进行
typedef struct stereo_camera_t stereo_camera_t;

// ── 数据结构 ───────────────────────────────────────────

// 帧率描述（实际帧率 = denominator / numerator）
typedef struct {
    uint32_t numerator;    // 帧率分子
    uint32_t denominator;  // 帧率分母
} stereo_camera_frame_rate_t;

// 格式描述（像素格式 + 分辨率 + 该分辨率下的帧率列表）
#define STEREO_CAMERA_MAX_FRAME_RATES 32
typedef struct {
    uint32_t pixel_format;         // V4L2 FOURCC 像素格式
    char     pixel_format_str[5];  // FOURCC 可读字符串，如 "YUYV"
    int      width;                // 图像宽度
    int      height;               // 图像高度
    int      frame_rate_count;     // frame_rates 数组中的有效帧率数量
    stereo_camera_frame_rate_t
        frame_rates[STEREO_CAMERA_MAX_FRAME_RATES];  // 帧率列表
} stereo_camera_format_info_t;

// 设备支持的最大格式数量
#define STEREO_CAMERA_MAX_FORMATS 64

// 设备信息（用于枚举）
typedef struct {
    char                        path[512];                           // 设备路径，如 "/dev/video3"
    char                        driver[128];                         // 驱动名称，如 "uvcvideo"
    char                        card[256];                           // 设备名称，如 "rk3xxx: UVC Camera"
    char                        bus_info[256];                       // 总线信息，如 "usb-0000:00:14.0-1"
    int                         format_count;                        // formats 数组中的有效格式数量
    stereo_camera_format_info_t formats[STEREO_CAMERA_MAX_FORMATS];  // 支持的格式列表
} stereo_camera_device_info_t;

// Depth↔RGB 时间匹配状态（来自 Gadget 端 DescInfo.matchState）
// 仅在 stream_id 为 Depth/ITOF 的帧上有意义
#define STEREO_MATCH_NONE 0    // 未匹配/启动中
#define STEREO_MATCH_EXACT 1   // 精确匹配（偏差 ≤ 8ms）
#define STEREO_MATCH_APPROX 2  // 近似匹配（8ms < 偏差 ≤ Depth半帧间隔）
#define STEREO_MATCH_STALE 3   // 重复旧数据（本次无新Depth帧）
#define STEREO_MATCH_LOST 4    // 失锁（偏差超限，Depth数据无效）

// 传感器类型枚举（与设备端 SensorMode 一致）
#define STEREO_SENSOR_UNKNOWN 0  // 未知
#define STEREO_SENSOR_RGB 1      // RGB
#define STEREO_SENSOR_ITOF 2     // ITOF
#define STEREO_SENSOR_DTOF 3     // DTOF
#define STEREO_SENSOR_IMU 4      // IMU 6 轴传感器

// 解析后的单路流图像数据
typedef struct {
    uint64_t       frame_timestamp;  // 采集时间戳（微秒，CLOCK_MONOTONIC，0=无效）
    uint64_t       frame_seqidx;     // 该流采集帧序号（StreamInfo.frame_seqidx；0=无效）视频流 = 设备端采集帧序号 IMU 流 = 块内首样本 ImuData[0].idx（IMU 采集样本计数）
    int            stream_id;        // 流 ID
    int            width;            // 图像宽度
    int            height;           // 图像高度
    uint32_t       pixel_format;     // V4L2 FOURCC 像素格式
    unsigned char *data;             // 图像数据（指向内部缓冲区，在下次 parse_frame 前有效）
    int            data_size;        // 数据长度
    int            match_state;      // 时间匹配状态（STEREO_MATCH_* 宏值）
    int            sourcetype;       // 传感器类型（STEREO_SENSOR_* 宏值）
    uint32_t       frame_seq_count;  // 本帧携带样本/帧数（视频流=1；IMU=ImuBlockHeader.count）
} stereo_camera_frame_t;

// ═══ IMU 数据解析说明（Device 端 v2 布局） ═══════════════════
//
// 固件在每帧 UVC 数据的头部区块写入 IMU 批量样本。IMU 不再是独立
// 视频流（不占 StreamInfo[] 条目），布局如下：
//
//   [DescInfo 256B][StreamInfo[] N×64B][ImuBlockHeader 24B][ImuData[count]
//   72B×count][视频像素区]
//
// ImuBlockHeader（24B，紧凑对齐）:
//   uint32_t magic;    // 魔数 'IMUB' = 0x494D5542，用于校验区块存在
//   uint32_t version;  // 块版本，当前 = 1
//   uint32_t capacity; // IMU 块最大样本容量（当前 = 5）
//   uint32_t count;    // 本帧实际有效样本数（0 ≤ count ≤ capacity）
//   uint32_t streamId; // IMU 流 ID（无 StreamInfo，由块头携带）
//   uint32_t format;   // IMU 数据格式标识（如 'IXYZ'）
//
// 每个样本为 stereo_camera_imu_data_t（72B）。count 随 UVC 帧率与 IMU
// 采样率的相对关系在 0~capacity 间浮动（实测以 3 为主，间或 4/5；
// 偶发 0 表示该帧无新 IMU 样本，data 可为空）。
//
// ── 如何通过本接口获取 IMU 数据 ──────────────────────────────
// stereo_camera_parse_frame() 会把该头部区块输出为一路独立流，其特征为：
//   - frame->sourcetype == STEREO_SENSOR_IMU (4)
//   - frame->stream_id == 块头 streamId 字段（非视频流，不使用 frameIdx）
//   - frame->pixel_format == 块头 format 字段（如 'IXYZ'）
//   - frame->data 指向连续的 ImuData[count] 数组
//   - frame->data_size == count * sizeof(stereo_camera_imu_data_t)
//     （count == 0 时 data 可为空 / data_size 为 0）
//   - frame->frame_timestamp == 所在 UVC 帧的帧时间戳
//     （DescInfo.timestamp，微秒）
//
// 完整解析示例（遍历本帧全部 IMU 样本）:
//   stereo_camera_frame_t *frame;
//   while ((frame = stereo_camera_parse_frame(cam)) != nullptr) {
//     if (frame->sourcetype != STEREO_SENSOR_IMU)
//       continue; // 视频流直接跳过，不按 IMU 解析
//     int count = frame->data_size / (int)sizeof(stereo_camera_imu_data_t);
//     const stereo_camera_imu_data_t *imu =
//         (const stereo_camera_imu_data_t *)frame->data;
//     for (int i = 0; i < count; i++) {
//       // 加速度: imu[i].ax/ay/az (m/s²)
//       // 陀螺仪: imu[i].gx/gy/gz（度/秒）
//       // 时间戳: imu[i].timestamp (us, CLOCK_MONOTONIC)
//       // 序号:   imu[i].idx
//     }
//   }
//
// 旧固件兼容：若固件把 IMU 作为独立视频流（而非头部区块），同样由
// sourcetype == STEREO_SENSOR_IMU 识别，解析方式与上方一致。
STEREO_CAMERA_PACKED_BEGIN
typedef struct {
    int64_t timestamp;  // 采集时间戳（微秒，CLOCK_MONOTONIC）
    int64_t idx;        // 帧序号（设备端 int64 帧计数）
    float   ax;         // 加速度 X (m/s²)
    float   ay;         // 加速度 Y (m/s²)
    float   az;         // 加速度 Z (m/s²)
    float   gx;         // 陀螺仪 X（度/秒）
    float   gy;         // 陀螺仪 Y（度/秒）
    float   gz;         // 陀螺仪 Z（度/秒）
    int32_t reverve[8];
} STEREO_CAMERA_PACKED_END stereo_camera_imu_data_t;

// ── 设备枚举 ──────────────────────────────────────────

// 枚举系统上所有 V4L2 视频捕获设备
//
// devices:    输出缓冲区数组，可为 NULL（此时仅返回设备数量）
// max_count:  devices 数组最大容量
// 返回值:     系统上发现的设备总数（可能大于 max_count）
STEREO_CAMERA_API int
stereo_camera_enumerate_devices(stereo_camera_device_info_t *devices,
                                int                          max_count);

// ── 相机生命周期 ──────────────────────────────────────

// 打开 UVC 相机
//
// device_path: 设备路径。
//              - 传 NULL 或空字符串时，按优先级匹配已知的本厂设备名称
//                (rk3xxx, rockchip, rk3588, RK3576)，若无匹配则使用第一个
//                可用的视频捕获设备。
//              - 传非空路径时（如 "/dev/video3"），直接打开指定设备。
// 返回句柄，失败返回 NULL
STEREO_CAMERA_API stereo_camera_t *stereo_camera_open(const char *device_path);

// 关闭相机，释放所有资源
// cam: 由 stereo_camera_open 返回的句柄，可为 NULL
STEREO_CAMERA_API void stereo_camera_close(stereo_camera_t *cam);

// ── 格式协商 ──────────────────────────────────────────

// 设置图像格式
// cam: 相机句柄
// width, height: 期望的分辨率
// format: 像素格式字符串，支持 "YUYV", "NV12"
// 返回 0 表示成功，-1 表示失败
STEREO_CAMERA_API int stereo_camera_set_format(stereo_camera_t *cam, int width,
                                               int height, const char *format);

// 获取实际协商后的格式
// cam: 相机句柄
// width, height, format: 输出参数，可为 NULL
// format 缓冲区至少 5 字节
// 返回 0 表示成功，-1 表示失败
STEREO_CAMERA_API int stereo_camera_get_format(stereo_camera_t *cam, int *width,
                                               int *height, char *format_4cc);

// ── 流控制 ────────────────────────────────────────────

// 启动视频流
// 返回 0 表示成功，-1 表示失败
STEREO_CAMERA_API int stereo_camera_start_stream(stereo_camera_t *cam);

// 停止视频流（释放缓冲区）
// 返回 0 表示成功，-1 表示失败
STEREO_CAMERA_API int stereo_camera_stop_stream(stereo_camera_t *cam);

// 取消阻塞的 stereo_camera_parse_frame 调用（线程安全）
// 设置内部的取消标志，使当前阻塞在 ReadFrame(DQBUF) 中的 parse_frame
// 立即返回 NULL。通常由 CameraWorker::requestStop() 配合 running_=0
// 一起调用，使采集循环能立即退出。
// 在下一轮 start/start_stream 时自动重置。
// cam: 相机句柄，可为 NULL
STEREO_CAMERA_API void stereo_camera_cancel_read(stereo_camera_t *cam);

// ── 帧采集与解析 ──────────────────────────────────────

// 读取并解析一路流数据（阻塞）
//
// 每次调用返回当前 UVC 帧中的一路流数据。当同一帧 UVC 的所有流被遍历完后，
// 下一次调用会自动读取下一帧 UVC 数据并返回其第一路流，以此类推 —— 无需
// 用户管理帧计数或流计数。
//
// 返回的帧指针在下次调用 stereo_camera_parse_frame 前有效，无需释放。
// 返回 NULL 表示出错或无数据。
//
// 典型用法（逐帧遍历）：
//   stereo_camera_frame_t *frame;
//   while ((frame = stereo_camera_parse_frame(cam)) != nullptr) {
//     // 处理 frame->stream_id, frame->data, frame->data_size ...
//     // frame->frame_timestamp 为该路流的采集时间戳
//     // frame->match_state 为该帧的时间匹配状态
//   }
STEREO_CAMERA_API stereo_camera_frame_t *
stereo_camera_parse_frame(stereo_camera_t *cam);

// ── UVC XU 命令（统一抽象入口）────────────────────────

// 命令枚举：值即通道内子命令 ID（sub_cmd，负载首字节）。
// 所有命令共用统一命令通道 selector 1；负载 = [sub_cmd] + param + [ACK 4B]。
// 拓展新命令：在末尾追加枚举值（1..255）+ 参数结构体 + 实现内新增分发分支即可，
// 无需修改内核描述符（子命令数量不受内核 selector 位图限制）。
typedef enum {
    STEREO_CAMERA_XU_CMD_IMU_FREQ    = STEREO_CAMERA_XU_SUBCMD_IMU_FREQ,     // sub_cmd=1
    STEREO_CAMERA_XU_CMD_STREAM_MASK = STEREO_CAMERA_XU_SUBCMD_STREAM_MASK,  // sub_cmd=2
} stereo_camera_xu_cmd_t;

// 每条命令各自的参数结构体（结构化绑定：命令 ↔ 参数类型一一对应，
// 调用方须按 cmd 传入匹配的结构体指针）。
// 负载 = 参数结构体内存（小端序），无需另行序列化。
typedef struct {
    // STEREO_CAMERA_XU_CMD_IMU_FREQ：
    uint32_t accel_hz;  // 加速度计采样频率（Hz），允许 12/25/50/100/200/400/800/1600
    uint32_t gyro_hz;   // 陀螺仪采样频率（Hz），允许 25/50/100/200/400/800/1600/3200
} stereo_camera_xu_imu_freq_param_t;

typedef struct {
    // STEREO_CAMERA_XU_CMD_STREAM_MASK：
    uint32_t stream_mask;  // 推流视频流 ID 掩码（bit i=streamId i：1 推流 / 0 不推；
                           // 仅视频流 RGB/ITOF，IMU 恒推不受掩码影响；默认全推）
} stereo_camera_xu_stream_mask_param_t;

// 统一 UVC XU 命令入口（对外导出的 API）：按 cmd 内部分发——
// 以 param（param_size 字节）作为 SET_CUR 负载下发 → GET_CUR 回读 ACK。
// 新增命令无需新增回调函数，追加枚举值/参数结构体/分发分支即可。
//
// cam:        相机句柄
// cmd:        命令枚举（stereo_camera_xu_cmd_t，值即 XU 选择子）
// param:      命令参数结构体指针（须与 cmd 一一对应，见上方定义）
// param_size: param 字节数（即协议负载长度）
// out_ack:    输出设备端 ACK 结果码（0=成功，1=EINVAL，2=EIO），可为 NULL
// 返回值:     0=命令成功执行且 ACK==0；-1=传输失败、参数非法或设备拒绝
//             （此时可通过 *out_ack 进一步区分原因）
STEREO_CAMERA_API int stereo_camera_xu_command(stereo_camera_t       *cam,
                                               stereo_camera_xu_cmd_t cmd,
                                               const void            *param,
                                               size_t                 param_size,
                                               int                   *out_ack);

// ============================================================================
// 模块内部日志控制
// ============================================================================

// 启用/关闭模块内部文件日志，并指定日志保存路径
// （当前为 IMU 头部排查 dump，默认 /tmp/imu_head_dump.log）
//
// enable:   1=启用（真机排障用），0=关闭（默认）
// log_path: 日志文件保存路径；传 NULL 或空字符串表示使用默认路径
//           （/tmp/imu_head_dump.log）。调用方可通过该参数实现统一联动，
//           将本模块日志与其他日志写入同一目录，便于一次排障集中收集。
//
// 模块内部默认不写任何日志文件。需要真机排障时，通过本接口在运行时
// 打开；排查结束后应重新关闭，避免不受控的日志文件写入与磁盘膨胀。
// 该开关通过接口参数传入（非编译期宏），库/调用方均可按需切换。
STEREO_CAMERA_API void stereo_camera_set_log(int enable, const char *log_path);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // STEREO_CAMERA_H_
