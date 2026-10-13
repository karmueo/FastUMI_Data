/**
 * @file output_decimator.hpp
 * @brief 按解码后的帧序号选择 ``0, N, 2N, …`` 发布，供 H.264 接收器各后端共用。
 *
 * 所有压缩包都必须送入解码器以维持参考帧；降频只能减少 raw 发布、夹爪估计和预览
 * 开销，不能在输入端每 N 包丢一包。
 */

#ifndef FASTUMI_USB_CAMERA__OUTPUT_DECIMATOR_HPP_
#define FASTUMI_USB_CAMERA__OUTPUT_DECIMATOR_HPP_

#include <cstdint>
#include <stdexcept>

namespace fastumi_usb_camera
{

/** @brief 解码帧输出降频计数器。 */
class OutputDecimator
{
public:
  /** @throws std::invalid_argument ``every_n`` 不为正。 */
  explicit OutputDecimator(int64_t every_n)
  : every_n_(every_n)
  {
    if (every_n_ < 1) {
      throw std::invalid_argument("output_every_n_frames must be a positive integer");
    }
  }

  /** @return 当前解码帧是否发布；每次调用前进一个帧序号。 */
  bool take()
  {
    const bool selected = (index_ % static_cast<uint64_t>(every_n_)) == 0U;
    ++index_;
    if (!selected) {
      ++skipped_;
    }
    return selected;
  }

  int64_t every_n() const {return every_n_;}
  /** @return 主动跳过（非丢帧）的解码帧数。 */
  uint64_t skipped() const {return skipped_;}

private:
  int64_t every_n_;
  uint64_t index_{0};
  uint64_t skipped_{0};
};

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__OUTPUT_DECIMATOR_HPP_
