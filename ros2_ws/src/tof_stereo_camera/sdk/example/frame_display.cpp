/**
 * @file frame_display.cpp
 * @brief 使用 SDL2/OpenGL shader 直接显示 SDK 原始 YUV 和 iTOF 图像。
 */

#define GL_GLEXT_PROTOTYPES
#include "frame_display.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <string>

namespace {

/// RGB 视频流 ID。
constexpr int kRgbStreamId = 0;
/// iTOF 深度视频流 ID。
constexpr int kItofDepthStreamId = 2;
/// iTOF 灰度视频流 ID。
constexpr int kItofGrayStreamId = 7;
/// 显示窗口初始宽度。
constexpr int kWindowWidth = 1280;
/// 显示窗口初始高度。
constexpr int kWindowHeight = 800;

/** @brief 按 V4L2 规则生成 FOURCC 整数。 */
constexpr std::uint32_t MakeFourcc(char first, char second, char third,
                                   char fourth) {
  return static_cast<std::uint32_t>(static_cast<unsigned char>(first)) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(second))
          << 8U) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(third))
          << 16U) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(fourth))
          << 24U);
}

/// YUYV 的 V4L2 FOURCC。
constexpr std::uint32_t kFourccYuyv = MakeFourcc('Y', 'U', 'Y', 'V');
/// NV12 的 V4L2 FOURCC。
constexpr std::uint32_t kFourccNv12 = MakeFourcc('N', 'V', '1', '2');

/// 全屏三角形顶点 shader，不需要 CPU 顶点缓冲区。
constexpr const char *kVertexShader = R"GLSL(
#version 330 core
out vec2 texture_coordinate;
void main() {
  const vec2 positions[3] = vec2[3](
      vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  const vec2 coordinates[3] = vec2[3](
      vec2(0.0, 0.0), vec2(2.0, 0.0), vec2(0.0, 2.0));
  gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
  texture_coordinate = coordinates[gl_VertexID];
}
)GLSL";

/// 在 GPU 上解释 YUYV、NV12 和 16 位单通道纹理的 fragment shader。
constexpr const char *kFragmentShader = R"GLSL(
#version 330 core
in vec2 texture_coordinate;
out vec4 fragment_color;
uniform sampler2D byte_texture_0;
uniform sampler2D byte_texture_1;
uniform usampler2D word_texture;
uniform int display_mode;
uniform float value_maximum;

vec3 yuv_to_rgb(float y, float u, float v) {
  float scaled_y = 1.164383 * (y - 0.0625);
  float centered_u = u - 0.5;
  float centered_v = v - 0.5;
  return clamp(vec3(
      scaled_y + 1.596027 * centered_v,
      scaled_y - 0.391762 * centered_u - 0.812968 * centered_v,
      scaled_y + 2.017232 * centered_u), 0.0, 1.0);
}

void main() {
  vec2 coordinate = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);
  if (display_mode == 0) {
    ivec2 size = textureSize(byte_texture_0, 0);
    ivec2 pixel = clamp(ivec2(coordinate * vec2(size)), ivec2(0), size - 1);
    int even_x = pixel.x & ~1;
    vec2 even_sample = texelFetch(byte_texture_0, ivec2(even_x, pixel.y), 0).rg;
    vec2 odd_sample = texelFetch(
        byte_texture_0, ivec2(min(even_x + 1, size.x - 1), pixel.y), 0).rg;
    float y = (pixel.x & 1) == 0 ? even_sample.r : odd_sample.r;
    fragment_color = vec4(yuv_to_rgb(y, even_sample.g, odd_sample.g), 1.0);
    return;
  }
  if (display_mode == 1) {
    float y = texture(byte_texture_0, coordinate).r;
    vec2 uv = texture(byte_texture_1, coordinate).rg;
    fragment_color = vec4(yuv_to_rgb(y, uv.r, uv.g), 1.0);
    return;
  }

  ivec2 size = textureSize(word_texture, 0);
  ivec2 pixel = clamp(ivec2(coordinate * vec2(size)), ivec2(0), size - 1);
  uint raw_value = texelFetch(word_texture, pixel, 0).r;
  float intensity = clamp(float(raw_value) / value_maximum, 0.0, 1.0);
  if (display_mode == 2 && raw_value != uint(0)) {
    intensity = 1.0 - intensity;
  }
  fragment_color = vec4(intensity, intensity, intensity, 1.0);
}
)GLSL";

/** @brief 表示 shader 对一幅原始图像采用的解释方式。 */
enum class DisplayMode : int {
  kYuyv = 0,  ///< YUYV packed 图像。
  kNv12 = 1,  ///< NV12 双平面图像。
  kDepth = 2, ///< 16 位 iTOF 深度图。
  kGray = 3,  ///< 16 位 iTOF 灰度图。
};

/** @brief 保存一路图像对应的 OpenGL 纹理与尺寸。 */
struct ImageSlot {
  /// YUYV、NV12 Y 平面或 16 位单通道纹理。
  GLuint texture_0 = 0;
  /// NV12 UV 平面纹理。
  GLuint texture_1 = 0;
  /// 当前图像宽度。
  int width = 0;
  /// 当前图像高度。
  int height = 0;
  /// 当前 payload 的 shader 解释方式。
  DisplayMode mode = DisplayMode::kYuyv;
  /// 是否已成功上传至少一帧。
  bool ready = false;
};

/** @brief 保存一个窗口区域的位置和大小。 */
struct Pane {
  /// 区域左下角横坐标。
  int x = 0;
  /// 区域左下角纵坐标。
  int y = 0;
  /// 区域宽度。
  int width = 0;
  /// 区域高度。
  int height = 0;
};

/** @brief 编译一个 OpenGL shader 并返回对象 ID。 */
GLuint CompileShader(GLenum type, const char *source, std::string *error) {
  const GLuint shader = glCreateShader(type); // 待编译的 shader 对象。
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint compiled = GL_FALSE; // shader 编译状态。
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (compiled == GL_TRUE) {
    return shader;
  }
  GLint log_length = 0; // 编译日志长度。
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
  std::string log(static_cast<std::size_t>(std::max(log_length, 1)), '\0');
  glGetShaderInfoLog(shader, log_length, nullptr, log.data());
  *error = "OpenGL shader compilation failed: " + log;
  glDeleteShader(shader);
  return 0;
}

/** @brief 创建并链接显示程序。 */
GLuint CreateProgram(std::string *error) {
  const GLuint vertex =
      CompileShader(GL_VERTEX_SHADER, kVertexShader, error); // 顶点 shader。
  if (vertex == 0) {
    return 0;
  }
  const GLuint fragment = CompileShader(GL_FRAGMENT_SHADER, kFragmentShader,
                                        error); // 片元 shader。
  if (fragment == 0) {
    glDeleteShader(vertex);
    return 0;
  }
  const GLuint program = glCreateProgram(); // 链接后的显示程序。
  glAttachShader(program, vertex);
  glAttachShader(program, fragment);
  glLinkProgram(program);
  glDeleteShader(vertex);
  glDeleteShader(fragment);
  GLint linked = GL_FALSE; // 程序链接状态。
  glGetProgramiv(program, GL_LINK_STATUS, &linked);
  if (linked == GL_TRUE) {
    return program;
  }
  GLint log_length = 0; // 链接日志长度。
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
  std::string log(static_cast<std::size_t>(std::max(log_length, 1)), '\0');
  glGetProgramInfoLog(program, log_length, nullptr, log.data());
  *error = "OpenGL program link failed: " + log;
  glDeleteProgram(program);
  return 0;
}

/** @brief 配置纹理采用最近邻采样和边缘截断。 */
void ConfigureTexture() {
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

/** @brief 返回并清除最近一次 OpenGL 错误。 */
bool ReadGlError(const char *operation, std::string *error) {
  const GLenum code = glGetError(); // OpenGL 错误码。
  if (code == GL_NO_ERROR) {
    return true;
  }
  std::ostringstream message; // 可读错误信息。
  message << operation << " failed with OpenGL error 0x" << std::hex << code;
  *error = message.str();
  return false;
}

} // namespace

class FrameDisplay::Impl {
public:
  /** @brief 初始化窗口及所有 OpenGL 对象。 */
  bool Initialize(const FrameDisplayOptions &options, std::string *error) {
    options_ = options;
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
      *error = std::string("SDL_Init failed: ") + SDL_GetError();
      return false;
    }
    sdl_initialized_ = true;
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    window_ = SDL_CreateWindow(
        "stereo_camera | RGB (top) | iTOF Depth (bottom-left) | iTOF Gray "
        "(bottom-right)",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, kWindowWidth,
        kWindowHeight, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (window_ == nullptr) {
      *error = std::string("SDL_CreateWindow failed: ") + SDL_GetError();
      return false;
    }
    context_ = SDL_GL_CreateContext(window_);
    if (context_ == nullptr) {
      *error = std::string("SDL_GL_CreateContext failed: ") + SDL_GetError();
      return false;
    }
    SDL_GL_SetSwapInterval(0);
    program_ = CreateProgram(error);
    if (program_ == 0) {
      return false;
    }
    glGenVertexArrays(1, &vertex_array_);
    glBindVertexArray(vertex_array_);
    glUseProgram(program_);
    glUniform1i(glGetUniformLocation(program_, "byte_texture_0"), 0);
    glUniform1i(glGetUniformLocation(program_, "byte_texture_1"), 1);
    glUniform1i(glGetUniformLocation(program_, "word_texture"), 2);
    mode_location_ = glGetUniformLocation(program_, "display_mode");
    maximum_location_ = glGetUniformLocation(program_, "value_maximum");
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    return ReadGlError("display initialization", error);
  }

  /** @brief 释放由实现对象持有的显示资源。 */
  ~Impl() {
    DeleteSlot(&rgb_);
    DeleteSlot(&depth_);
    DeleteSlot(&gray_);
    if (vertex_array_ != 0) {
      glDeleteVertexArrays(1, &vertex_array_);
    }
    if (program_ != 0) {
      glDeleteProgram(program_);
    }
    if (context_ != nullptr) {
      SDL_GL_DeleteContext(context_);
    }
    if (window_ != nullptr) {
      SDL_DestroyWindow(window_);
    }
    if (sdl_initialized_) {
      SDL_Quit();
    }
  }

  /** @brief 识别路由并上传原始 payload。 */
  bool Update(const stereo_camera_frame_t &frame, std::string *error) {
    error->clear();
    if (frame.sourcetype == STEREO_SENSOR_RGB &&
        frame.stream_id == kRgbStreamId) {
      return UploadYuv(frame, &rgb_, error);
    }
    if (frame.sourcetype == STEREO_SENSOR_ITOF &&
        frame.stream_id == kItofDepthStreamId) {
      if (frame.pixel_format == kFourccYuyv ||
          frame.pixel_format == kFourccNv12) {
        return UploadYuv(frame, &depth_, error);
      }
      return UploadMono16(frame, DisplayMode::kDepth, &depth_, error);
    }
    if (frame.sourcetype == STEREO_SENSOR_ITOF &&
        frame.stream_id == kItofGrayStreamId) {
      if (frame.pixel_format == kFourccYuyv ||
          frame.pixel_format == kFourccNv12) {
        return UploadYuv(frame, &gray_, error);
      }
      return UploadMono16(frame, DisplayMode::kGray, &gray_, error);
    }
    return false;
  }

  /** @brief 处理退出相关窗口事件。 */
  bool PollEvents() {
    SDL_Event event; // 当前 SDL 窗口事件。
    while (SDL_PollEvent(&event) != 0) {
      if (event.type == SDL_QUIT) {
        return false;
      }
      if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) {
        return false;
      }
    }
    return true;
  }

  /** @brief 按固定三窗格布局绘制最新纹理。 */
  void Render() {
    int window_width = 0;  // 当前 drawable 宽度。
    int window_height = 0; // 当前 drawable 高度。
    SDL_GL_GetDrawableSize(window_, &window_width, &window_height);
    glViewport(0, 0, window_width, window_height);
    glClearColor(0.04F, 0.04F, 0.04F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    const int bottom_height = window_height / 2; // 底部窗格高度。
    const std::array<Pane, 3> panes{{
        {0, bottom_height, window_width, window_height - bottom_height},
        {0, 0, window_width / 2, bottom_height},
        {window_width / 2, 0, window_width - window_width / 2, bottom_height},
    }}; // RGB、Depth、Gray 的窗口区域。
    DrawSlot(rgb_, panes[0], 1.0F);
    DrawSlot(depth_, panes[1], static_cast<float>(options_.depth_max));
    DrawSlot(gray_, panes[2], static_cast<float>(options_.gray_max));
    SDL_GL_SwapWindow(window_);
  }

private:
  /** @brief 删除一路纹理并复位状态。 */
  static void DeleteSlot(ImageSlot *slot) {
    if (slot->texture_0 != 0) {
      glDeleteTextures(1, &slot->texture_0);
    }
    if (slot->texture_1 != 0) {
      glDeleteTextures(1, &slot->texture_1);
    }
    *slot = ImageSlot{};
  }

  /** @brief 检查公共图像字段和期望 payload 大小。 */
  static bool ValidateFrame(const stereo_camera_frame_t &frame,
                            std::uint64_t expected_size, std::string *error) {
    if (frame.width <= 0 || frame.height <= 0 || frame.data == nullptr ||
        frame.data_size <= 0) {
      *error = "image frame has invalid dimensions or empty payload";
      return false;
    }
    if (expected_size != static_cast<std::uint64_t>(frame.data_size)) {
      std::ostringstream message; // payload 尺寸诊断。
      message << "image payload size mismatch: expected " << expected_size
              << ", got " << frame.data_size;
      *error = message.str();
      return false;
    }
    return true;
  }

  /** @brief 确保一路纹理的格式和尺寸与新帧一致。 */
  static void AllocateSlot(ImageSlot *slot, int width, int height,
                           DisplayMode mode) {
    if (slot->texture_0 != 0 && slot->width == width &&
        slot->height == height && slot->mode == mode) {
      return;
    }
    DeleteSlot(slot);
    slot->width = width;
    slot->height = height;
    slot->mode = mode;
    glGenTextures(1, &slot->texture_0);
    glBindTexture(GL_TEXTURE_2D, slot->texture_0);
    ConfigureTexture();
    if (mode == DisplayMode::kYuyv) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, width, height, 0, GL_RG,
                   GL_UNSIGNED_BYTE, nullptr);
    } else if (mode == DisplayMode::kNv12) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED,
                   GL_UNSIGNED_BYTE, nullptr);
      glGenTextures(1, &slot->texture_1);
      glBindTexture(GL_TEXTURE_2D, slot->texture_1);
      ConfigureTexture();
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, width / 2, height / 2, 0, GL_RG,
                   GL_UNSIGNED_BYTE, nullptr);
    } else {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_R16UI, width, height, 0, GL_RED_INTEGER,
                   GL_UNSIGNED_SHORT, nullptr);
    }
  }

  /** @brief 上传一帧 YUYV 或 NV12 图像，不在 CPU 侧转换颜色。 */
  static bool UploadYuv(const stereo_camera_frame_t &frame, ImageSlot *slot,
                        std::string *error) {
    const std::uint64_t pixels =
        static_cast<std::uint64_t>(frame.width) *
        static_cast<std::uint64_t>(frame.height); // 图像像素数量。
    if (frame.pixel_format == kFourccYuyv) {
      if ((frame.width % 2) != 0 || !ValidateFrame(frame, pixels * 2U, error)) {
        if ((frame.width % 2) != 0) {
          *error = "YUYV width must be even";
        }
        return false;
      }
      AllocateSlot(slot, frame.width, frame.height, DisplayMode::kYuyv);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, slot->texture_0);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.width, frame.height, GL_RG,
                      GL_UNSIGNED_BYTE, frame.data);
    } else if (frame.pixel_format == kFourccNv12) {
      if ((frame.width % 2) != 0 || (frame.height % 2) != 0 ||
          !ValidateFrame(frame, pixels * 3U / 2U, error)) {
        if ((frame.width % 2) != 0 || (frame.height % 2) != 0) {
          *error = "NV12 width and height must be even";
        }
        return false;
      }
      AllocateSlot(slot, frame.width, frame.height, DisplayMode::kNv12);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, slot->texture_0);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.width, frame.height, GL_RED,
                      GL_UNSIGNED_BYTE, frame.data);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, slot->texture_1);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.width / 2, frame.height / 2,
                      GL_RG, GL_UNSIGNED_BYTE, frame.data + pixels);
    } else {
      *error = "YUV display supports only YUYV or NV12 payloads";
      return false;
    }
    slot->ready = ReadGlError("YUV texture upload", error);
    return slot->ready;
  }

  /** @brief 上传一帧 iTOF 16 位单通道图像。 */
  static bool UploadMono16(const stereo_camera_frame_t &frame, DisplayMode mode,
                           ImageSlot *slot, std::string *error) {
    const std::uint64_t pixels =
        static_cast<std::uint64_t>(frame.width) *
        static_cast<std::uint64_t>(frame.height); // 图像像素数量。
    if (!ValidateFrame(frame, pixels * 2U, error)) {
      return false;
    }
    AllocateSlot(slot, frame.width, frame.height, mode);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, slot->texture_0);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.width, frame.height,
                    GL_RED_INTEGER, GL_UNSIGNED_SHORT, frame.data);
    slot->ready = ReadGlError("iTOF texture upload", error);
    return slot->ready;
  }

  /** @brief 在指定窗格内保持宽高比绘制一路图像。 */
  void DrawSlot(const ImageSlot &slot, const Pane &pane,
                float value_maximum) const {
    if (!slot.ready) {
      return;
    }
    const double image_aspect =
        static_cast<double>(slot.width) /
        static_cast<double>(slot.height); // 图像宽高比。
    const double pane_aspect = static_cast<double>(pane.width) /
                               static_cast<double>(pane.height); // 窗格宽高比。
    int draw_width = pane.width;   // 保持比例后的绘制宽度。
    int draw_height = pane.height; // 保持比例后的绘制高度。
    if (image_aspect > pane_aspect) {
      draw_height =
          static_cast<int>(static_cast<double>(draw_width) / image_aspect);
    } else {
      draw_width =
          static_cast<int>(static_cast<double>(draw_height) * image_aspect);
    }
    const int draw_x = pane.x + (pane.width - draw_width) / 2; // 绘制左边界。
    const int draw_y = pane.y + (pane.height - draw_height) / 2; // 绘制下边界。
    glViewport(draw_x, draw_y, draw_width, draw_height);
    glUseProgram(program_);
    glUniform1i(mode_location_, static_cast<int>(slot.mode));
    glUniform1f(maximum_location_, value_maximum);
    if (slot.mode == DisplayMode::kDepth || slot.mode == DisplayMode::kGray) {
      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D, slot.texture_0);
    } else {
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, slot.texture_0);
    }
    if (slot.mode == DisplayMode::kNv12) {
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, slot.texture_1);
    }
    glDrawArrays(GL_TRIANGLES, 0, 3);
  }

  /// 显示参数。
  FrameDisplayOptions options_;
  /// SDL 窗口对象。
  SDL_Window *window_ = nullptr;
  /// SDL 创建的 OpenGL 上下文。
  SDL_GLContext context_ = nullptr;
  /// OpenGL 显示程序。
  GLuint program_ = 0;
  /// 全屏三角形使用的空顶点数组对象。
  GLuint vertex_array_ = 0;
  /// shader 显示模式 uniform 位置。
  GLint mode_location_ = -1;
  /// shader 单通道最大值 uniform 位置。
  GLint maximum_location_ = -1;
  /// RGB 最新纹理。
  ImageSlot rgb_;
  /// iTOF 深度最新纹理。
  ImageSlot depth_;
  /// iTOF 灰度最新纹理。
  ImageSlot gray_;
  /// 是否已成功初始化 SDL。
  bool sdl_initialized_ = false;
};

/** @copydoc FrameDisplay::FrameDisplay */
FrameDisplay::FrameDisplay() : impl_(std::make_unique<Impl>()) {}

/** @copydoc FrameDisplay::~FrameDisplay */
FrameDisplay::~FrameDisplay() = default;

/** @copydoc FrameDisplay::Initialize */
bool FrameDisplay::Initialize(const FrameDisplayOptions &options,
                              std::string *error) {
  return impl_->Initialize(options, error);
}

/** @copydoc FrameDisplay::Update */
bool FrameDisplay::Update(const stereo_camera_frame_t &frame,
                          std::string *error) {
  return impl_->Update(frame, error);
}

/** @copydoc FrameDisplay::PollEvents */
bool FrameDisplay::PollEvents() { return impl_->PollEvents(); }

/** @copydoc FrameDisplay::Render */
void FrameDisplay::Render() { impl_->Render(); }
