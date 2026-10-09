"""无需模型或 ROS 图验证 episode 取消、目标切换和键盘服务语义。"""

import asyncio
import json
from pathlib import Path
import time
from types import MethodType, SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import numpy as np
import yaml
from fastumi_interfaces.msg import TargetMask
from fastumi_interfaces.srv import SetNumInferenceSteps
from std_srvs.srv import Trigger

from dexgraspvla_infer.policy_node import PolicyNode


def node():
    value = NS(state="running", reason="", episode=5, sequence=0, identity=7, latest_identity=7,
               inflight=True, warmed=True, last_submitted=1, last_publish=None, prediction_timing={},
               worker=Mock(), publisher=Mock(), get_logger=lambda: Mock(), runtime=NS(max_steps=50))
    value.worker.cancel.return_value = False
    for method in ("_invalidate", "_result", "_start", "_stop", "_home", "_steps", "_on_mask"):
        setattr(value, method, MethodType(getattr(PolicyNode, method), value))
    value._observation = Mock(return_value=NS(tracking_id=7))
    value._fail = Mock()
    return value


def test_stop_invalidates_inflight_prediction_and_its_errors():
    value = node()
    value._call = AsyncMock(return_value=(True, "stopped"))
    response = asyncio.run(value._stop(None, Trigger.Response()))
    assert response.success and value.state == "idle" and value.episode == 6
    value._result(NS(tracking_id=7), 5, False, 16, 0, None, RuntimeError("old error"))
    value.publisher.publish.assert_not_called()
    value._fail.assert_not_called()


def test_stop_during_start_cannot_enable_old_episode():
    value = node()
    value.state = "idle"
    async def interrupted(*args):
        value._invalidate("idle", "stopped")
        return True, "started"
    value._call = interrupted
    result = asyncio.run(value._start(None, Trigger.Response()))
    assert not result.success and value.state == "idle"


def test_mask_target_change_stops_before_mask_decode():
    value = node()
    value.buffer = NS(masks={})
    message = TargetMask()
    message.state, message.tracking_id = TargetMask.TRACKING, 8
    message.mask.encoding = "mono8"
    value._on_mask(message)
    value._fail.assert_called_once()
    assert "changed" in value._fail.call_args.args[0]


def test_keyboard_steps_apply_to_next_inference_and_validate_range():
    value = node()
    parameters = {"num_inference_steps": 16}
    value.set_parameters = lambda values: parameters.update({p.name: p.value for p in values})
    value.get_parameter = lambda name: NS(value=parameters[name])
    request = SetNumInferenceSteps.Request()
    request.num_inference_steps = 4
    assert value._steps(request, SetNumInferenceSteps.Response()).success
    assert parameters["num_inference_steps"] == 4
    request.num_inference_steps = 51
    assert not value._steps(request, SetNumInferenceSteps.Response()).success
    assert parameters["num_inference_steps"] == 4


def test_next_prediction_skips_masks_acquired_before_previous_dispatch():
    value = node()
    value.inflight = False
    value.last_clock = 0
    value.status_received = time.monotonic()
    value._fresh = lambda: True
    value.get_clock = lambda: NS(now=lambda: NS(nanoseconds=100))
    value.get_parameter = lambda name: NS(value=4)
    value.runtime.predict = Mock()
    value._observation.return_value = NS(stamp_ns=80, tracking_id=7)
    PolicyNode._tick(value)
    assert value.last_submitted == 100 and value.inflight
    value.inflight = False
    value._observation.return_value = NS(stamp_ns=90, tracking_id=7)
    PolicyNode._tick(value)
    assert value.worker.submit.call_count == 1


def test_stop_diagnostic_keeps_inflight_timing_and_invalidates_episode():
    value = node()
    logger = Mock()
    value.get_logger = lambda: logger
    value.prediction_timing = {"submitted": {"steps": 4, "dispatch_age_s": 0.18}}
    value.worker.snapshot.return_value = {"active": "policy", "active_age_s": 0.8}
    value.runtime.last_diagnostic = {"steps": 4}
    value.controller_clients = {"stop": Mock()}
    value.controller_clients["stop"].service_is_ready.return_value = False
    PolicyNode._fail(value, "controller stopped: trajectory horizon exhausted")
    assert value.episode == 6 and value.state == "fault"
    timing = json.loads(logger.error.call_args_list[-1].args[0].split(": ", 1)[1])
    assert timing["prediction"]["submitted"]["steps"] == 4
    assert timing["inflight"]
    assert timing["gpu_worker"]["active"] == "policy"


def test_camera_gap_tolerance_and_disconnect_watchdog(monkeypatch):
    """实机 266 ms 断帧可继续等待，超过阈值仍停车。"""
    value = node()
    config = Path(__file__).parents[1] / "config/dexgraspvla.yaml"
    parameters = yaml.safe_load(config.read_text())["dexgraspvla_infer"]["ros__parameters"]
    value.get_parameter = lambda name: NS(value=parameters[name])
    value.get_clock = lambda: NS(now=lambda: NS(nanoseconds=100_000_000_000))
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    value.last_clock = 0
    value.status_received = 100.0
    value.last_image, value.last_joint = 99.734, 99.997
    value.last_gripper, value.last_tracking = 99.986, 99.858
    value._fresh = MethodType(PolicyNode._fresh, value)
    PolicyNode._tick(value)
    value._fail.assert_not_called()

    # 夹爪/关节反馈仍使用原有的 250 ms 阈值。
    value.last_joint = 99.749
    assert not value._fresh()
    value.last_joint = 99.997

    value.last_image = 99.499
    PolicyNode._tick(value)
    value._fail.assert_called_once()
    assert "camera=0.501s" in value._fail.call_args.args[0]


def test_camera_tolerance_does_not_accept_expired_prediction():
    value = node()
    value.get_clock = lambda: NS(now=lambda: NS(nanoseconds=2_010_000_000))
    value.get_parameter = lambda name: NS(value=1.0)
    observation = NS(stamp_ns=1_000_000_000, tracking_id=7)
    value._result(observation, value.episode, False, 4, time.monotonic(), None, None)
    value._fail.assert_called_once_with("policy result expired")
    value.publisher.publish.assert_not_called()
