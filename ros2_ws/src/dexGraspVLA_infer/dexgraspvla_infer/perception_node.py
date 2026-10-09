"""独立 ROS 感知节点：发布保留原采集时间的 jingbao mask。"""

from pathlib import Path
from array import array
import sys

import cv2
from cv_bridge import CvBridge
from fastumi_interfaces.msg import TargetMask
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from dexgraspvla_infer.detector import JingbaoDetector
from dexgraspvla_infer.perception import MaskValidator, SamCutieTracker, TargetPerception
from dexgraspvla_infer.images import image_message, serialized_image


class PerceptionNode(Node):
    """SAM 初始化期间未就绪；跟踪失败输出零 mask，再自动检测。"""

    def __init__(self, worker, sink=None, compute=None):
        super().__init__("dexgraspvla_perception")
        defaults = {
            "asset_root": "", "device": "cuda:0", "image_topic": "/wrist_camera/image_decoded",
            "image_type": "raw", "mask_topic": "/fastumi/perception/target_mask",
            "visualization_topic": "/fastumi/perception/visualization/compressed",
            "confidence": 0.25, "min_mask_area": 0.0005, "max_mask_area": 0.10,
            "max_mask_area_ratio": 4.0, "cutie_max_size": 480,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        get = lambda name: self.get_parameter(name).value
        assets = Path(get("asset_root")).expanduser().resolve()
        sys.path.insert(0, str(assets / "third_party/Cutie"))
        self.worker = worker
        self.sink = sink
        self.bridge = CvBridge()
        self.tracking_id = 0
        self.tracking = False
        self.last_stamp = 0
        if compute is None:
            tracker = SamCutieTracker(
                sam_checkpoint=assets / "weights/segmentation/sam_vit_h_4b8939.pth",
                cutie_checkpoint=assets / "weights/tracking/cutie-base-mega.pth",
                device=get("device"), max_internal_size=get("cutie_max_size"),
                torch_hub_dir=assets / "torch_hub",
                validator=MaskValidator(min_area_fraction=get("min_mask_area"),
                                        max_area_fraction=get("max_mask_area"),
                                        max_area_ratio=get("max_mask_area_ratio")))
            detector = JingbaoDetector(assets / "weights/detection/yolo26s_jingbao.pt",
                                      confidence_threshold=get("confidence"), device=get("device"))
            self.perception = TargetPerception(detector, tracker)
        else:
            from dexgraspvla_infer.compute import PerceptionProxy
            self.perception = PerceptionProxy(compute, asset_root=str(assets), device=get("device"),
                max_internal_size=get("cutie_max_size"), confidence=get("confidence"),
                min_area=get("min_mask_area"), max_area=get("max_mask_area"), area_ratio=get("max_mask_area_ratio"))
        reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.publisher = self.create_publisher(TargetMask, get("mask_topic"), reliable)
        self.visualization = self.create_publisher(CompressedImage, get("visualization_topic"), qos_profile_sensor_data)
        self.image_type = get("image_type")
        if self.image_type not in ("raw", "compressed"):
            raise ValueError("image_type must be raw or compressed (JPEG/PNG)")
        self.create_subscription(Image if self.image_type == "raw" else CompressedImage,
                                 get("image_topic"), self.on_image,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
                                 raw=self.image_type == "raw")
        if sink is None:
            self.create_timer(0.01, worker.poll)

    def on_image(self, message):
        if self.image_type == "raw":
            try:
                header, pixels = serialized_image(message)
            except Exception as error:
                if self.sink is not None:
                    self.sink._fail(f"invalid serialized camera frame: {error}")
                self.get_logger().warning(str(error))
                return
        else:
            header, pixels = message.header, None
        stamp = int(header.stamp.sec) * 10**9 + header.stamp.nanosec
        if stamp <= 0 or stamp == self.last_stamp:
            return
        self.last_stamp = stamp
        if self.sink is not None:
            import time
            self.sink.last_image = time.monotonic()

        def process():
            # Tracker state is owned by the single GPU worker, not ROS callbacks.
            image = (pixels if self.image_type == "raw"
                     else self.bridge.compressed_imgmsg_to_cv2(message, "bgr8"))
            image = image.copy()
            try:
                if self.perception.tracker.initialized:
                    result = self.perception.update(image)
                    initialized = False
                else:
                    result, _ = self.perception.initialize(image)
                    initialized = True
                return image, result.mask, initialized, None
            except Exception as error:
                self.perception.reset()
                return image, np.zeros(image.shape[:2], dtype=np.uint8), False, str(error)

        self.worker.submit("perception", process,
                           lambda result, error: self._publish(header, result, error))

    def _publish(self, header, result, error):
        if error is not None:
            image = np.zeros((1, 1, 3), dtype=np.uint8)
            mask, initialized, detail = np.zeros(image.shape[:2], dtype=np.uint8), False, str(error)
        else:
            image, mask, initialized, detail = result
        message = TargetMask()
        message.header = header
        if initialized:
            self.tracking_id += 1
        if detail:
            message.state = TargetMask.LOST if self.tracking else TargetMask.DETECTING
            self.tracking = False
        else:
            self.tracking = True
            message.state = TargetMask.TRACKING
        message.tracking_id = self.tracking_id
        message.mask = image_message(mask, "mono8", header)
        message.diagnostic = detail or "tracking jingbao"
        if self.sink is not None:
            # Same executor, original header, no duplicate multi-megabyte DDS
            # roundtrip inside the shared GPU process. Standalone mode uses ROS.
            self.sink.buffer.image(int(header.stamp.sec) * 10**9 + header.stamp.nanosec,
                                   header.frame_id, image)
            self.sink._on_mask(message)
            self.sink._tick()
        if self.sink is None or self.publisher.get_subscription_count():
            self.publisher.publish(message)
        if not self.visualization.get_subscription_count():
            return
        overlay = image.copy()
        selected = mask > 0
        overlay[selected] = (0.5 * overlay[selected] + [0, 127, 0]).astype(np.uint8)
        cv2.putText(overlay, f"track={self.tracking_id} state={message.state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        ok, encoded = cv2.imencode(".jpg", overlay)
        if not ok:
            self.get_logger().warning("Cannot encode perception preview")
            return
        preview = CompressedImage()
        preview.format = "jpeg"
        preview.data = array("B", encoded.tobytes())
        preview.header = header
        self.visualization.publish(preview)
