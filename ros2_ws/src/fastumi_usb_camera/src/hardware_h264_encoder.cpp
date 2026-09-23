/**
 * @file hardware_h264_encoder.cpp
 * @brief NVIDIA GStreamer implementation of the bounded hardware H.264 encoder.
 */

#include "fastumi_usb_camera/hardware_h264_encoder.hpp"

#include <algorithm>
#include <deque>
#include <iterator>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

namespace fastumi_usb_camera
{

namespace
{

constexpr const char * kRequiredFactories[] = {
  "appsrc", "jpegparse", "nvv4l2decoder", "nvvidconv", "capsfilter",
  "nvv4l2h264enc", "h264parse", "appsink",
};

std::runtime_error pipeline_error(const std::string & detail)
{
  return std::runtime_error("NVIDIA H.264 hardware pipeline: " + detail);
}

}  // namespace

class HardwareH264Encoder::Implementation
{
public:
  Implementation(
    const HardwareH264Configuration & configuration,
    PacketCallback callback)
  : configuration_(configuration), callback_(std::move(callback))
  {
    if (configuration_.width <= 0 || configuration_.height <= 0 ||
      configuration_.fps <= 0 || configuration_.bitrate <= 0 ||
      configuration_.gop_size <= 0)
    {
      throw std::invalid_argument("hardware H.264 dimensions, fps, bitrate and GOP must be positive");
    }
    if (!callback_) {
      throw std::invalid_argument("hardware H.264 packet callback is empty");
    }

    GError * initialization_error = nullptr;
    if (!gst_init_check(nullptr, nullptr, &initialization_error)) {
      const std::string detail = initialization_error ? initialization_error->message : "unknown error";
      if (initialization_error) {
        g_error_free(initialization_error);
      }
      throw pipeline_error("GStreamer initialization failed: " + detail);
    }
    std::string missing;
    if (!HardwareH264Encoder::is_available(&missing)) {
      throw pipeline_error("required GStreamer element is unavailable: " + missing);
    }
    create_pipeline();
  }

  ~Implementation()
  {
    shutdown();
  }

  void stop() noexcept
  {
    if (pipeline_) {
      gst_element_set_state(pipeline_, GST_STATE_NULL);
    }
  }

  bool push(
    uint64_t pts, const std::vector<uint8_t> & jpeg,
    SteadyClock::time_point queued_at)
  {
    if (jpeg.empty() || poll_error()) {
      return false;
    }
    GstBuffer * buffer = gst_buffer_new_allocate(nullptr, jpeg.size(), nullptr);
    if (!buffer) {
      set_error("failed to allocate GStreamer input buffer");
      return false;
    }
    if (gst_buffer_fill(buffer, 0, jpeg.data(), jpeg.size()) != jpeg.size()) {
      gst_buffer_unref(buffer);
      set_error("failed to fill GStreamer input buffer");
      return false;
    }
    GST_BUFFER_PTS(buffer) = pts;
    GST_BUFFER_DTS(buffer) = pts;
    GST_BUFFER_DURATION(buffer) = GST_SECOND / configuration_.fps;
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      pending_.emplace_back(pts, queued_at);
      while (pending_.size() > 8U) {
        pending_.pop_front();
      }
    }
    const GstFlowReturn result = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
    if (result != GST_FLOW_OK) {
      remove_pending(pts);
      set_error("appsrc rejected JPEG frame with flow status " + std::to_string(result));
      return false;
    }
    return !poll_error();
  }

  bool poll_error()
  {
    if (!bus_) {
      return has_error();
    }
    while (GstMessage * message = gst_bus_pop_filtered(
        bus_, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS)))
    {
      if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
        GError * error = nullptr;
        gchar * debug = nullptr;
        gst_message_parse_error(message, &error, &debug);
        std::ostringstream detail;
        detail << (error ? error->message : "unknown GStreamer error");
        if (debug && *debug) {
          detail << " (" << debug << ')';
        }
        if (error) {
          g_error_free(error);
        }
        g_free(debug);
        set_error(detail.str());
      } else {
        set_error("pipeline reached unexpected end of stream");
      }
      gst_message_unref(message);
    }
    return has_error();
  }

  std::string error_message() const
  {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return error_message_;
  }

private:
  static GstFlowReturn on_new_sample(GstAppSink * sink, gpointer user_data) noexcept
  {
    return static_cast<Implementation *>(user_data)->handle_sample(sink);
  }

  GstFlowReturn handle_sample(GstAppSink * sink) noexcept
  {
    GstSample * sample = gst_app_sink_pull_sample(sink);
    if (!sample) {
      set_error("appsink returned an empty H.264 sample");
      return GST_FLOW_ERROR;
    }
    GstBuffer * buffer = gst_sample_get_buffer(sample);
    GstMapInfo mapping{};
    if (!buffer || !gst_buffer_map(buffer, &mapping, GST_MAP_READ)) {
      gst_sample_unref(sample);
      set_error("failed to map H.264 output buffer");
      return GST_FLOW_ERROR;
    }
    HardwareH264Packet packet;
    try {
      const uint64_t output_pts =
        GST_BUFFER_PTS_IS_VALID(buffer) ? GST_BUFFER_PTS(buffer) : 0U;
      const auto [capture_pts, latency_nanoseconds] = take_capture_metadata(output_pts);
      packet.pts = capture_pts;
      packet.keyframe = !GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_DELTA_UNIT);
      packet.latency_nanoseconds = latency_nanoseconds;
      packet.data.assign(mapping.data, mapping.data + mapping.size);
    } catch (const std::exception & error) {
      gst_buffer_unmap(buffer, &mapping);
      gst_sample_unref(sample);
      set_error(std::string("failed to prepare H.264 packet: ") + error.what());
      return GST_FLOW_ERROR;
    } catch (...) {
      gst_buffer_unmap(buffer, &mapping);
      gst_sample_unref(sample);
      set_error("failed to prepare H.264 packet with an unknown exception");
      return GST_FLOW_ERROR;
    }
    gst_buffer_unmap(buffer, &mapping);
    gst_sample_unref(sample);
    try {
      callback_(std::move(packet));
      return GST_FLOW_OK;
    } catch (const std::exception & error) {
      set_error(std::string("packet callback failed: ") + error.what());
    } catch (...) {
      set_error("packet callback failed with an unknown exception");
    }
    return GST_FLOW_ERROR;
  }

  void create_pipeline()
  {
    pipeline_ = gst_pipeline_new("fastumi-hardware-h264");
    appsrc_ = gst_element_factory_make("appsrc", "jpeg-source");
    GstElement * jpegparse = gst_element_factory_make("jpegparse", "jpeg-parser");
    GstElement * decoder = gst_element_factory_make("nvv4l2decoder", "jpeg-decoder");
    GstElement * converter = gst_element_factory_make("nvvidconv", "nv12-converter");
    GstElement * raw_filter = gst_element_factory_make("capsfilter", "nv12-filter");
    GstElement * encoder = gst_element_factory_make("nvv4l2h264enc", "h264-encoder");
    GstElement * h264parse = gst_element_factory_make("h264parse", "h264-parser");
    GstElement * encoded_filter = gst_element_factory_make("capsfilter", "h264-filter");
    appsink_ = gst_element_factory_make("appsink", "h264-sink");
    if (!pipeline_ || !appsrc_ || !jpegparse || !decoder || !converter || !raw_filter ||
      !encoder || !h264parse || !encoded_filter || !appsink_)
    {
      shutdown();
      throw pipeline_error("failed to create one or more GStreamer elements");
    }

    GstCaps * input_caps = gst_caps_new_simple(
      "image/jpeg", "width", G_TYPE_INT, configuration_.width,
      "height", G_TYPE_INT, configuration_.height,
      "framerate", GST_TYPE_FRACTION, configuration_.fps, 1, nullptr);
    GstCaps * raw_caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
    GstCaps * encoded_caps = gst_caps_from_string(
      "video/x-h264,stream-format=byte-stream,alignment=au");
    g_object_set(
      appsrc_, "caps", input_caps, "is-live", TRUE, "format", GST_FORMAT_TIME,
      "block", TRUE, "max-buffers", static_cast<guint64>(1), nullptr);
    g_object_set(decoder, "mjpeg", TRUE, nullptr);
    g_object_set(raw_filter, "caps", raw_caps, nullptr);
    g_object_set(
      encoder, "bitrate", static_cast<guint>(configuration_.bitrate),
      "control-rate", 1, "iframeinterval", static_cast<guint>(configuration_.gop_size),
      "idrinterval", static_cast<guint>(configuration_.gop_size),
      "insert-sps-pps", TRUE, "maxperf-enable", TRUE,
      "num-B-Frames", static_cast<guint>(0), "profile", 0,
      "copy-timestamp", TRUE, nullptr);
    g_object_set(h264parse, "config-interval", -1, nullptr);
    g_object_set(encoded_filter, "caps", encoded_caps, nullptr);
    g_object_set(
      appsink_, "emit-signals", TRUE, "sync", FALSE,
      "max-buffers", static_cast<guint>(2), "drop", FALSE, nullptr);
    gst_caps_unref(input_caps);
    gst_caps_unref(raw_caps);
    gst_caps_unref(encoded_caps);

    gst_bin_add_many(
      GST_BIN(pipeline_), appsrc_, jpegparse, decoder, converter, raw_filter,
      encoder, h264parse, encoded_filter, appsink_, nullptr);
    if (!gst_element_link_many(
        appsrc_, jpegparse, decoder, converter, raw_filter, encoder,
        h264parse, encoded_filter, appsink_, nullptr))
    {
      shutdown();
      throw pipeline_error("failed to link GStreamer elements");
    }
    g_signal_connect(appsink_, "new-sample", G_CALLBACK(on_new_sample), this);
    bus_ = gst_element_get_bus(pipeline_);
    const GstStateChangeReturn state = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (state == GST_STATE_CHANGE_FAILURE) {
      poll_error();
      const std::string detail = error_message();
      shutdown();
      throw pipeline_error("failed to enter PLAYING state" +
              (detail.empty() ? std::string() : ": " + detail));
    }
  }

  void shutdown() noexcept
  {
    stop();
    if (bus_) {
      gst_object_unref(bus_);
      bus_ = nullptr;
    }
    if (pipeline_) {
      gst_object_unref(pipeline_);
      pipeline_ = nullptr;
    }
    appsrc_ = nullptr;
    appsink_ = nullptr;
  }

  bool has_error() const
  {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return !error_message_.empty();
  }

  void set_error(const std::string & detail) noexcept
  {
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (error_message_.empty()) {
      error_message_ = detail;
    }
  }

  void remove_pending(uint64_t pts)
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    const auto found = std::find_if(
      pending_.begin(), pending_.end(),
      [pts](const auto & item) {return item.first == pts;});
    if (found != pending_.end()) {
      pending_.erase(found);
    }
  }

  std::pair<uint64_t, int64_t> take_capture_metadata(uint64_t output_pts)
  {
    const auto now = SteadyClock::now();
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto found = std::find_if(
      pending_.begin(), pending_.end(),
      [output_pts](const auto & item) {return item.first == output_pts;});
    if (found == pending_.end()) {
      // NVIDIA elements may regenerate a constant-rate output timeline from the
      // first input PTS. With B-frames disabled, output order still matches
      // capture order, so FIFO preserves the original camera timestamp.
      found = pending_.begin();
    }
    if (found == pending_.end()) {
      return {output_pts, 0};
    }
    const int64_t latency = std::chrono::duration_cast<std::chrono::nanoseconds>(
      now - found->second).count();
    const uint64_t capture_pts = found->first;
    pending_.erase(pending_.begin(), std::next(found));
    return {capture_pts, latency};
  }

  HardwareH264Configuration configuration_;
  PacketCallback callback_;
  GstElement * pipeline_{nullptr};
  GstElement * appsrc_{nullptr};
  GstElement * appsink_{nullptr};
  GstBus * bus_{nullptr};
  mutable std::mutex error_mutex_;
  std::string error_message_;
  std::mutex pending_mutex_;
  std::deque<std::pair<uint64_t, SteadyClock::time_point>> pending_;
};

HardwareH264Encoder::HardwareH264Encoder(
  const HardwareH264Configuration & configuration,
  PacketCallback callback)
: implementation_(std::make_unique<Implementation>(configuration, std::move(callback)))
{
}

HardwareH264Encoder::~HardwareH264Encoder() = default;

bool HardwareH264Encoder::push(
  uint64_t pts, const std::vector<uint8_t> & jpeg,
  SteadyClock::time_point queued_at)
{
  return implementation_->push(pts, jpeg, queued_at);
}

void HardwareH264Encoder::stop() noexcept
{
  implementation_->stop();
}

bool HardwareH264Encoder::poll_error()
{
  return implementation_->poll_error();
}

std::string HardwareH264Encoder::error_message() const
{
  return implementation_->error_message();
}

bool HardwareH264Encoder::is_available(std::string * missing_factory)
{
  gst_init(nullptr, nullptr);
  for (const char * name : kRequiredFactories) {
    GstElementFactory * factory = gst_element_factory_find(name);
    if (!factory) {
      if (missing_factory) {
        *missing_factory = name;
      }
      return false;
    }
    gst_object_unref(factory);
  }
  return true;
}

}  // namespace fastumi_usb_camera
