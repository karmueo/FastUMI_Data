// Copyright 2026 karmueo

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "fastumi_usb_camera/latest_frame_buffer.hpp"

using fastumi_usb_camera::LatestFrameBuffer;

TEST(LatestFrameBuffer, OverwritesPendingFrameAndOwnsItsData)
{
  LatestFrameBuffer<std::unique_ptr<int>> buffer;
  auto first = std::make_unique<int>(1);
  auto second = std::make_unique<int>(2);
  EXPECT_TRUE(buffer.push(std::move(first)));
  EXPECT_TRUE(buffer.push(std::move(second)));
  EXPECT_EQ(buffer.overwritten_count(), 1U);

  auto result = buffer.wait_and_take();
  ASSERT_TRUE(result.has_value());
  ASSERT_NE(result->get(), nullptr);
  EXPECT_EQ(**result, 2);
}

TEST(LatestFrameBuffer, SlowConsumerNeverBuildsAQueue)
{
  LatestFrameBuffer<int> buffer;
  std::atomic<int> consumed{-1};
  std::thread consumer([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    auto frame = buffer.wait_and_take();
    if (frame) {
      consumed = *frame;
    }
  });
  for (int frame = 0; frame < 100; ++frame) {
    ASSERT_TRUE(buffer.push(frame));
  }
  consumer.join();
  EXPECT_EQ(consumed.load(), 99);
  EXPECT_EQ(buffer.overwritten_count(), 99U);
}

TEST(LatestFrameBuffer, CloseDropsPendingFrameAndWakesConsumer)
{
  LatestFrameBuffer<int> buffer;
  ASSERT_TRUE(buffer.push(7));
  buffer.close();
  EXPECT_FALSE(buffer.wait_and_take().has_value());
  EXPECT_FALSE(buffer.push(8));
}
