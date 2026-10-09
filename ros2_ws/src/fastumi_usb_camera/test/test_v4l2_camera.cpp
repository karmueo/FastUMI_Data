// Copyright 2026 karmueo
// 模拟内核队列以验证实际采集线程的恢复、退出及资源释放。
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <linux/videodev2.h>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "fastumi_usb_camera/v4l2_camera.hpp"

namespace
{
using fastumi_usb_camera::CameraConfiguration;
using fastumi_usb_camera::V4l2Camera;
using fastumi_usb_camera::V4l2Io;
using namespace std::chrono_literals;

struct Event
{
  short revents{POLLIN};
  int dequeue_errno{0};
  uint32_t flags{0};
  int queue_errno{0};
};

/** @brief 在采集线程提供预设事件，计数可由测试线程安全观察。 */
class FakeV4l2Io : public V4l2Io
{
public:
  std::vector<Event> events;
  bool repeat_last{false};
  int fail_open_number{0};
  int fail_stream_on_number{0};
  std::atomic<int> opens{0}, closes{0}, maps{0}, unmaps{0};
  std::atomic<int> stream_ons{0}, stream_offs{0}, frame_queues{0};
  std::array<uint8_t, 16> jpeg{{0xff, 0xd8, 0xff, 0xd9}};
  // 在 stop/join 后检查调用顺序。
  std::vector<std::string> operations;

  int open_device(const std::string &) override
  {
    operations.push_back("open");
    const int number = ++opens;
    if (number == fail_open_number) {errno = EBUSY; return -1;}
    return number;
  }
  int control(int, unsigned long request, void * argument) override
  {
    switch (request) {
      case VIDIOC_QUERYCAP:
        static_cast<v4l2_capability *>(argument)->capabilities =
          V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING;
        break;
      case VIDIOC_S_FMT:
      case VIDIOC_S_PARM:
      case VIDIOC_REQBUFS:
        break;
      case VIDIOC_QUERYBUF:
        static_cast<v4l2_buffer *>(argument)->length = jpeg.size();
        break;
      case VIDIOC_STREAMON:
        operations.push_back("stream_on");
        ++stream_ons;
        if (opens == fail_stream_on_number) {errno = EIO; return -1;}
        break;
      case VIDIOC_STREAMOFF:
        operations.push_back("stream_off");
        ++stream_offs;
        break;
      case VIDIOC_DQBUF: {
        if (current_.dequeue_errno != 0) {errno = current_.dequeue_errno; return -1;}
        auto * buffer = static_cast<v4l2_buffer *>(argument);
        buffer->index = 0;
        buffer->bytesused = 4;
        buffer->flags = current_.flags;
        break;
      }
      case VIDIOC_QBUF:
        if (static_cast<v4l2_buffer *>(argument)->bytesused != 0) {
          ++frame_queues;
          if (current_.queue_errno != 0) {errno = current_.queue_errno; return -1;}
        }
        break;
      default:
        errno = EINVAL;
        return -1;
    }
    return 0;
  }
  int wait(pollfd * descriptor, int) override
  {
    if (next_ < events.size()) {current_ = events[next_++];}
    else if (repeat_last && !events.empty()) {current_ = events.back();}
    else {std::this_thread::sleep_for(1ms); return 0;}
    descriptor->revents = current_.revents;
    return 1;
  }
  void * map_buffer(int, size_t, off_t) override
  {
    operations.push_back("map");
    ++maps;
    return jpeg.data();
  }
  int unmap_buffer(void *, size_t) override
  {
    operations.push_back("unmap");
    ++unmaps;
    return 0;
  }
  int close_device(int) override
  {
    operations.push_back("close");
    ++closes;
    return 0;
  }
private:
  size_t next_{0};
  Event current_;
};

struct Frames
{
  std::atomic<int> count{0};
  std::atomic<bool> valid{true};
  static void receive(const uint8_t * data, size_t size, void * context)
  {
    auto & frames = *static_cast<Frames *>(context);
    if (size != 4 || data[0] != 0xff || data[3] != 0xd9) {frames.valid = false;}
    ++frames.count;
  }
};

CameraConfiguration config(double timeout = 5.0)
{
  CameraConfiguration result;
  // 仅检查路径存在性；FakeV4l2Io 接管所有设备访问。
  result.video_device = "/dev/null";
  result.frame_timeout_seconds = timeout;
  return result;
}

template<typename Predicate>
bool wait_until(Predicate predicate)
{
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (!predicate() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return predicate();
}

TEST(V4l2Capture, RestartsPollErrorAndReleasesOldMappingsBeforeReopen)
{
  auto io = std::make_shared<FakeV4l2Io>();
  io->events = {{POLLERR}, {POLLIN}};
  Frames frames;
  std::vector<std::string> warnings;
  V4l2Camera camera(config(), Frames::receive, &frames,
    [&](const std::string & message) {warnings.push_back(message);}, io);
  ASSERT_TRUE(wait_until([&]() {return frames.count == 1 || camera.failed();}));
  camera.stop();
  EXPECT_FALSE(camera.failed()) << camera.error_message();
  EXPECT_TRUE(frames.valid);
  EXPECT_EQ(frames.count, 1);
  EXPECT_EQ(camera.recovery_count(), 1U);
  EXPECT_EQ(io->opens, 2);
  EXPECT_EQ(io->closes, 2);
  EXPECT_EQ(io->maps, 8);
  EXPECT_EQ(io->unmaps, 8);
  EXPECT_EQ(io->stream_offs, 2);
  const std::vector<std::string> expected{
    "open", "map", "map", "map", "map", "stream_on",
    "stream_off", "unmap", "unmap", "unmap", "unmap", "close",
    "open", "map", "map", "map", "map", "stream_on",
    "stream_off", "unmap", "unmap", "unmap", "unmap", "close"};
  EXPECT_EQ(io->operations, expected);
  ASSERT_EQ(warnings.size(), 2U);
  EXPECT_NE(warnings[0].find("revents=0x8 POLLERR"), std::string::npos);
  EXPECT_NE(warnings[1].find("valid frame received"), std::string::npos);
  camera.stop();
  EXPECT_EQ(io->closes, 2);
}

TEST(V4l2Capture, DequeuesReadableBufferEvenWithPollError)
{
  auto io = std::make_shared<FakeV4l2Io>();
  io->events = {{POLLIN | POLLERR}, {POLLIN, EAGAIN},
    {POLLIN, 0, V4L2_BUF_FLAG_ERROR}, {POLLIN}};
  Frames frames;
  V4l2Camera camera(config(), Frames::receive, &frames, {}, io);
  ASSERT_TRUE(wait_until([&]() {return frames.count == 2 || camera.failed();}));
  camera.stop();
  EXPECT_FALSE(camera.failed()) << camera.error_message();
  EXPECT_EQ(frames.count, 2);
  EXPECT_EQ(camera.invalid_count(), 1U);
  EXPECT_EQ(io->frame_queues, 3);
  EXPECT_EQ(io->opens, 1);
}

TEST(V4l2Capture, RebuildsQueueAfterDequeueEioWithoutRequeuingUnknownBuffer)
{
  auto io = std::make_shared<FakeV4l2Io>();
  io->events = {{POLLIN, EIO}, {POLLIN}};
  Frames frames;
  V4l2Camera camera(config(), Frames::receive, &frames, {}, io);
  ASSERT_TRUE(wait_until([&]() {return frames.count == 1 || camera.failed();}));
  camera.stop();
  EXPECT_FALSE(camera.failed()) << camera.error_message();
  EXPECT_EQ(frames.count, 1);
  EXPECT_EQ(io->frame_queues, 1);
  EXPECT_EQ(camera.recovery_count(), 1U);
}

TEST(V4l2Capture, RestartsAfterQueueEioAndPollErrorWithEmptyQueue)
{
  auto io = std::make_shared<FakeV4l2Io>();
  io->events = {{POLLIN, 0, 0, EIO}, {POLLIN | POLLERR, EAGAIN}, {POLLIN}};
  Frames frames;
  V4l2Camera camera(config(), Frames::receive, &frames, {}, io);
  ASSERT_TRUE(wait_until([&]() {return frames.count == 2 || camera.failed();}));
  camera.stop();
  EXPECT_FALSE(camera.failed()) << camera.error_message();
  EXPECT_EQ(frames.count, 2);
  EXPECT_EQ(camera.recovery_count(), 2U);
}

TEST(V4l2Capture, BoundsConsecutiveRestartsWithoutValidFrame)
{
  auto io = std::make_shared<FakeV4l2Io>();
  io->events = {{POLLERR}};
  io->repeat_last = true;
  Frames frames;
  V4l2Camera camera(config(), Frames::receive, &frames, {}, io);
  ASSERT_TRUE(wait_until([&]() {return camera.failed();}));
  camera.stop();
  EXPECT_EQ(camera.recovery_count(), 3U);
  EXPECT_EQ(io->opens, 4);
  EXPECT_EQ(io->closes, 4);
  EXPECT_EQ(io->maps, io->unmaps);
  EXPECT_NE(camera.error_message().find("restart limit exhausted"), std::string::npos);
}

TEST(V4l2Capture, ValidFramesResetConsecutiveRestartBudget)
{
  auto io = std::make_shared<FakeV4l2Io>();
  io->events = {{POLLERR}, {POLLIN}, {POLLERR}, {POLLIN},
    {POLLERR}, {POLLIN}, {POLLERR}, {POLLIN}};
  Frames frames;
  V4l2Camera camera(config(), Frames::receive, &frames, {}, io);
  ASSERT_TRUE(wait_until([&]() {return frames.count == 4 || camera.failed();}));
  camera.stop();
  EXPECT_FALSE(camera.failed()) << camera.error_message();
  EXPECT_EQ(frames.count, 4);
  EXPECT_EQ(camera.recovery_count(), 4U);
}

TEST(V4l2Capture, HangupAndInvalidDescriptorFailWithoutRestart)
{
  for (const short event : {POLLHUP, POLLNVAL, POLLHUP | POLLIN | POLLERR}) {
    auto io = std::make_shared<FakeV4l2Io>();
    io->events = {{event}};
    Frames frames;
    V4l2Camera camera(config(), Frames::receive, &frames, {}, io);
    ASSERT_TRUE(wait_until([&]() {return camera.failed();}));
    camera.stop();
    EXPECT_EQ(io->opens, 1);
    EXPECT_EQ(frames.count, 0);
    EXPECT_NE(camera.error_message().find("revents=0x"), std::string::npos);
  }
}

TEST(V4l2Capture, DequeueDeviceRemovalFailsWithoutRestart)
{
  auto io = std::make_shared<FakeV4l2Io>();
  io->events = {{POLLIN, ENODEV}};
  Frames frames;
  V4l2Camera camera(config(), Frames::receive, &frames, {}, io);
  ASSERT_TRUE(wait_until([&]() {return camera.failed();}));
  camera.stop();
  EXPECT_EQ(io->opens, 1);
  EXPECT_NE(camera.error_message().find("VIDIOC_DQBUF"), std::string::npos);
  EXPECT_NE(camera.error_message().find("errno=19"), std::string::npos);
}

TEST(V4l2Capture, RestartFailurePreservesOriginalErrorAndReleasesPartialResources)
{
  for (const bool fail_open : {true, false}) {
    auto io = std::make_shared<FakeV4l2Io>();
    io->events = {{POLLERR}};
    if (fail_open) {io->fail_open_number = 2;} else {io->fail_stream_on_number = 2;}
    Frames frames;
    V4l2Camera camera(config(), Frames::receive, &frames, {}, io);
    ASSERT_TRUE(wait_until([&]() {return camera.failed();}));
    camera.stop();
    EXPECT_EQ(io->opens, 2);
    EXPECT_EQ(io->closes, fail_open ? 1 : 2);
    EXPECT_EQ(io->maps, io->unmaps);
    EXPECT_NE(camera.error_message().find("POLLERR"), std::string::npos);
    EXPECT_NE(camera.error_message().find("capture restart failed"), std::string::npos);
  }
}

TEST(V4l2Capture, StopDuringRecoveryDoesNotReopenOrReportFailure)
{
  auto io = std::make_shared<FakeV4l2Io>();
  io->events = {{POLLERR}};
  Frames frames;
  std::atomic<bool> recovering{false};
  V4l2Camera camera(config(), Frames::receive, &frames,
    [&](const std::string &) {recovering = true;}, io);
  ASSERT_TRUE(wait_until([&]() {return recovering.load();}));
  camera.stop();
  EXPECT_FALSE(camera.failed());
  EXPECT_EQ(io->opens, 1);
  EXPECT_EQ(io->closes, 1);
  EXPECT_EQ(io->maps, io->unmaps);
}

TEST(V4l2Capture, RecoveryDoesNotExtendFrameTimeout)
{
  auto io = std::make_shared<FakeV4l2Io>();
  io->events = {{POLLERR}};
  io->repeat_last = true;
  Frames frames;
  V4l2Camera camera(config(0.15), Frames::receive, &frames, {}, io);
  ASSERT_TRUE(wait_until([&]() {return camera.failed();}));
  camera.stop();
  EXPECT_EQ(camera.recovery_count(), 1U);
  EXPECT_NE(camera.error_message().find("no valid V4L2 camera frame"), std::string::npos);
}

TEST(V4l2Capture, IdleCaptureStillTimesOut)
{
  auto io = std::make_shared<FakeV4l2Io>();
  Frames frames;
  V4l2Camera camera(config(0.02), Frames::receive, &frames, {}, io);
  ASSERT_TRUE(wait_until([&]() {return camera.failed();}));
  camera.stop();
  EXPECT_EQ(io->opens, 1);
  EXPECT_NE(camera.error_message().find("no valid V4L2 camera frame"), std::string::npos);
}
}  // namespace
