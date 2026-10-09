"""高效图像消息必须保持原像素、步长、通道和采集 header。"""

from cv_bridge import CvBridge
import numpy as np
import pytest
from std_msgs.msg import Header
from rclpy.serialization import serialize_message

from dexgraspvla_infer.images import image_message, serialized_image


@pytest.mark.parametrize("encoding,shape", [("mono8", (96, 128)), ("bgr8", (96, 128, 3))])
def test_byte_buffer_image_roundtrip(encoding, shape):
    pixels = np.random.default_rng(7).integers(0, 256, shape, np.uint8)
    header = Header(frame_id="wrist_camera_optical_frame")
    header.stamp.sec, header.stamp.nanosec = 3, 19
    message = image_message(pixels, encoding, header)
    decoded = CvBridge().imgmsg_to_cv2(message, encoding)
    np.testing.assert_array_equal(decoded, pixels)
    assert message.header == header
    assert len(message.data) == pixels.size
    received_header, received = serialized_image(serialize_message(message))
    assert received_header == header
    expected = np.repeat(pixels[:, :, None], 3, axis=2) if encoding == "mono8" else pixels
    np.testing.assert_array_equal(received, expected)


def test_serialized_image_rejects_invalid_or_truncated_wire_data():
    for payload in (b"", b"\x00\x07\x00\x00", b"\x00\x01\x00\x00\x01"):
        with pytest.raises(ValueError):
            serialized_image(payload)
