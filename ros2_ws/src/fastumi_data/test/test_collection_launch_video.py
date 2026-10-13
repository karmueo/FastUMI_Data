"""验证采集 launch 的 JPEG/PNG 与 H.264 模式：话题、相机、解码节点和 RViz 适配。"""

from pathlib import Path

import pytest
from launch.utilities import perform_substitutions
import yaml

from test_collection_launch import VIDEO_DEVICE, _expand, _include_name

JPEG_TOPIC = "/umi_camera/image_raw/compressed"
H264_TOPIC = "/umi_camera/image_raw/ffmpeg"


def _by_executable(nodes):
    """按可执行文件名索引捕获的节点参数。"""
    return {node["executable"]: node for node in nodes}


def test_default_topic_follows_image_mode_and_reaches_collection_node() -> None:
    """留空 image_topic 时按模式取默认编码话题，并随模式参数传给采集节点。"""
    for mode, topic in (("compressed", JPEG_TOPIC), ("h264", H264_TOPIC)):
        _, nodes, _, _ = _expand(image_mode=mode)
        parameters = _by_executable(nodes)["collection_node"]["parameters"][1]

        assert parameters["image_mode"] == mode
        assert parameters["image_topic"] == topic
        assert parameters["decoded_image_topic"] == "/umi_camera/image_decoded"


def test_explicit_image_topic_and_qos_options_win() -> None:
    """显式话题（例如 Jetson 发布端）、QoS 和等待时间原样传给采集节点。"""
    _, nodes, _, _ = _expand(
        image_mode="h264", image_topic="/jetson/camera/ffmpeg",
        image_reliability="reliable", image_qos_depth="60", h264_start_timeout_s="8.5",
    )
    parameters = _by_executable(nodes)["collection_node"]["parameters"][1]

    assert parameters["image_topic"] == "/jetson/camera/ffmpeg"
    assert parameters["image_reliability"] == "reliable"
    assert parameters["image_qos_depth"] == 60
    assert parameters["h264_start_timeout_s"] == 8.5
    receiver = _by_executable(nodes)["usb_camera_receiver"]["parameters"][0]
    assert receiver["input_reliability"] == "reliable"
    assert receiver["input_queue_depth"] == 60


@pytest.mark.parametrize("name, value", [
    ("image_mode", "mjpeg"), ("image_reliability", "lossy"),
    ("image_qos_depth", "0"), ("decoder_backend", "tpu"),
    ("decoded_image_topic", "relative"),
])
def test_invalid_image_options_are_rejected(name, value) -> None:
    """未知模式、可靠性、非正深度、解码后端和相对解码话题被拒绝。"""
    with pytest.raises(ValueError):
        _expand(**{name: value})


def test_compressed_mode_uses_qos_aware_decoder_for_gripper_and_rviz() -> None:
    """JPEG 模式：启动自带 QoS 的解码节点，夹爪和 RViz 订阅解码话题。"""
    _, nodes, includes, _ = _expand(
        image_mode="compressed", start_gripper="true", image_qos_depth="12"
    )
    by_exec = _by_executable(nodes)
    decoder = by_exec["image_decoder"]["parameters"][0]
    gripper = next(i for i in includes if _include_name(i).endswith("gripper_openness.launch.py"))

    assert decoder["input_topic"] == JPEG_TOPIC
    assert decoder["output_topic"] == "/umi_camera/image_decoded"
    assert decoder["image_qos_depth"] == 12 and decoder["image_reliability"] == "best_effort"
    assert dict(gripper.launch_arguments)["image_topic"] == "/umi_camera/image_decoded"
    assert "usb_camera_receiver" not in by_exec


def test_h264_mode_uses_receiver_in_auto_mode_by_default() -> None:
    """H.264 模式：用 C++ 接收器的 auto 后端，输入为去掉 /ffmpeg 的基础话题。"""
    _, nodes, _, _ = _expand(image_mode="h264")
    receiver = _by_executable(nodes)["usb_camera_receiver"]["parameters"][0]

    assert receiver["input_topic"] == "/umi_camera/image_raw"
    assert receiver["output_topic"] == "/umi_camera/image_decoded"
    assert receiver["decoder_backend"] == "auto"
    assert receiver["umi_camera.image_raw.ffmpeg.decoders.h264"] == "h264"


def test_decoder_backend_can_be_selected() -> None:
    """解码后端由参数选择。"""
    for backend in ("transport", "cuda"):
        _, nodes, _, _ = _expand(image_mode="h264", decoder_backend=backend)

        assert _by_executable(nodes)["usb_camera_receiver"]["parameters"][0][
            "decoder_backend"
        ] == backend


def test_h264_topic_must_end_with_ffmpeg_for_the_receiver() -> None:
    """接收器订阅 <基础话题>/ffmpeg，其他命名无法解码并明确失败。"""
    with pytest.raises(ValueError, match="/ffmpeg"):
        _expand(image_mode="h264", image_topic="/jetson/camera/h264")


def test_decoder_is_skipped_when_not_needed_or_disabled() -> None:
    """raw、关闭解码、或既无夹爪估计也无 RViz2 时不创建解码节点。"""
    for overrides in (
        {"image_mode": "raw", "start_gripper": "true"},
        {"image_mode": "compressed", "start_decoder": "false", "start_gripper": "true"},
        {"image_mode": "compressed", "use_rviz": "false"},
    ):
        _, nodes, _, _ = _expand(**overrides)
        assert {"image_decoder", "usb_camera_receiver"}.isdisjoint(_by_executable(nodes))


def test_rviz_image_display_follows_the_viewed_topic(tmp_path: Path) -> None:
    """编码模式下 RViz2 使用改写后的配置订阅解码话题，其余显示不变。"""
    _, nodes, _, _ = _expand(image_mode="compressed", decoded_image_topic="/view/decoded")
    config = Path(_by_executable(nodes)["rviz2"]["arguments"][1])
    document = yaml.safe_load(config.read_text("utf-8"))
    displays = {d["Class"]: d for d in document["Visualization Manager"]["Displays"]}

    assert displays["rviz_default_plugins/Image"]["Topic"]["Value"] == "/view/decoded"
    assert displays["rviz_default_plugins/Odometry"]["Topic"]["Value"] == "/vive_tracker/odom"
    assert document["Panels"][0]["Class"] == "fastumi_rviz_plugins/CollectionPanel"


def test_raw_mode_keeps_packaged_rviz_config_and_custom_raw_topic_is_followed() -> None:
    """默认 raw 直接用包内配置；自定义原始话题时 Image 显示同步改写。"""
    _, nodes, _, _ = _expand()
    assert _by_executable(nodes)["rviz2"]["arguments"][1].endswith(
        "/share/fastumi_data/config/collection.rviz"
    )
    _, nodes, _, _ = _expand(image_topic="/other/image_raw")
    document = yaml.safe_load(
        Path(_by_executable(nodes)["rviz2"]["arguments"][1]).read_text("utf-8")
    )
    image = next(
        d for d in document["Visualization Manager"]["Displays"]
        if d["Class"] == "rviz_default_plugins/Image"
    )
    assert image["Topic"]["Value"] == "/other/image_raw"


def test_start_camera_maps_each_mode_to_existing_camera_options() -> None:
    """本地相机：raw 默认、JPEG 加 publish_compressed、H.264 启用软件 FFmpeg 编码。"""
    expectations = {
        "raw": {},
        "compressed": {"publish_compressed": "true"},
        "h264": {"enable_ffmpeg": "true"},
    }
    for mode, extra in expectations.items():
        _, _, includes, _ = _expand(
            image_mode=mode, start_camera="true", video_device=VIDEO_DEVICE
        )
        camera = next(i for i in includes if _include_name(i) == "fastumi_usb_camera/usb_camera.launch.py")
        arguments = dict(camera.launch_arguments)

        assert arguments["namespace"] == "umi_camera"
        for key, value in extra.items():
            assert arguments[key] == value
        assert ("publish_compressed" in arguments) == (mode == "compressed")
        assert ("enable_ffmpeg" in arguments) == (mode == "h264")


def test_h264_camera_gets_a_generated_encoder_config_for_the_topic() -> None:
    """H.264 编码配置的话题和插件参数前缀与 image_topic 同步改写，编码参数保留。"""
    _, _, includes, _ = _expand(
        image_mode="h264", image_topic="/front/image_raw/ffmpeg",
        start_camera="true", video_device=VIDEO_DEVICE,
    )
    camera = next(i for i in includes if _include_name(i).endswith("usb_camera.launch.py"))
    config = Path(dict(camera.launch_arguments)["ffmpeg_config"])
    parameters = yaml.safe_load(config.read_text("utf-8"))["usb_camera_ffmpeg"]["ros__parameters"]

    assert parameters["topic"] == "/front/image_raw"
    assert parameters["front.image_raw.ffmpeg.encoder"] == "libx264"
    assert parameters["front.image_raw.ffmpeg.max_b_frames"] == 0
    assert not any(key.startswith("usb_camera.image_raw.ffmpeg.") for key in parameters)
    assert dict(camera.launch_arguments)["namespace"] == "front"


def test_camera_namespace_requires_the_mode_suffix() -> None:
    """启动本地相机时话题必须带有该模式的后缀，才能推导命名空间。"""
    with pytest.raises(ValueError, match="image_raw/compressed"):
        _expand(
            image_mode="compressed", image_topic="/umi_camera/image_raw",
            start_camera="true", video_device=VIDEO_DEVICE,
        )


def test_auto_mode_follows_the_published_message_type() -> None:
    """默认 auto：连接已有设备时按发布端的类型订阅，并启动对应解码节点。"""
    for found, executable in (
        (("compressed", JPEG_TOPIC), "image_decoder"),
        (("h264", H264_TOPIC), "usb_camera_receiver"),
    ):
        _, nodes, _, _ = _expand(probe=lambda topic, timeout, f=found: f)
        by_exec = _by_executable(nodes)
        parameters = by_exec["collection_node"]["parameters"][1]

        assert parameters["image_mode"] == found[0] and parameters["image_topic"] == found[1]
        assert executable in by_exec
        document = yaml.safe_load(Path(by_exec["rviz2"]["arguments"][1]).read_text("utf-8"))
        image = next(
            d for d in document["Visualization Manager"]["Displays"]
            if d["Class"] == "rviz_default_plugins/Image"
        )
        assert image["Topic"]["Value"] == "/umi_camera/image_decoded"


def test_auto_mode_probes_the_explicit_topic_and_falls_back_by_suffix() -> None:
    """显式话题只探测该话题；探测不到时按后缀推断，没有显式话题则退回 raw。"""
    seen = []

    def probe(topic, timeout):
        """记录探测参数并模拟没有发布端。"""
        seen.append(topic)
        return None

    _, nodes, _, _ = _expand(probe=probe, image_topic="/jetson/camera/ffmpeg")
    assert seen == ["/jetson/camera/ffmpeg"]
    assert _by_executable(nodes)["collection_node"]["parameters"][1]["image_mode"] == "h264"
    _, nodes, _, _ = _expand(probe=probe)
    parameters = _by_executable(nodes)["collection_node"]["parameters"][1]
    assert (parameters["image_mode"], parameters["image_topic"]) == ("raw", "/umi_camera/image_raw")


def test_auto_mode_with_local_camera_is_compressed_without_probing() -> None:
    """启动本地相机时 auto 解析为 compressed（相机 JPEG 原样录制），不需要探测图。"""
    def probe(topic, timeout):
        """不应被调用。"""
        raise AssertionError("不应探测")

    _, nodes, _, _ = _expand(probe=probe, start_camera="true")

    parameters = _by_executable(nodes)["collection_node"]["parameters"][1]
    assert parameters["image_mode"] == "compressed"
    assert parameters["image_topic"] == "/umi_camera/image_raw/compressed"
