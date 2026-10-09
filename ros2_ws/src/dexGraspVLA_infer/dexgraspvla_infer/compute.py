"""独占 GPU 计算进程，隔离 ROS 高频回调的 Python GIL 竞争。"""

import multiprocessing as mp
from threading import Lock
from types import SimpleNamespace
import traceback


def serve(connection):
    """仅此进程加载模型和 CUDA；串行执行两个模型栈，不创建 ROS 节点。"""
    model, perception = None, None
    try:
        while True:
            operation, arguments = connection.recv()
            try:
                if operation == "close":
                    return
                if operation == "load_model":
                    from dexgraspvla_infer.model_runtime import ModelRuntime
                    model = ModelRuntime(**arguments)
                    result = model.max_steps
                elif operation == "load_perception":
                    from pathlib import Path
                    import sys
                    from dexgraspvla_infer.detector import JingbaoDetector
                    from dexgraspvla_infer.perception import MaskValidator, SamCutieTracker, TargetPerception
                    assets = Path(arguments.pop("asset_root"))
                    confidence = arguments.pop("confidence")
                    validator = MaskValidator(
                        min_area_fraction=arguments.pop("min_area"),
                        max_area_fraction=arguments.pop("max_area"),
                        max_area_ratio=arguments.pop("area_ratio"))
                    sys.path.insert(0, str(assets / "third_party/Cutie"))
                    tracker = SamCutieTracker(
                        sam_checkpoint=assets / "weights/segmentation/sam_vit_h_4b8939.pth",
                        cutie_checkpoint=assets / "weights/tracking/cutie-base-mega.pth",
                        torch_hub_dir=assets / "torch_hub", validator=validator, **arguments)
                    detector = JingbaoDetector(assets / "weights/detection/yolo26s_jingbao.pt",
                                              confidence_threshold=confidence, device=arguments["device"])
                    perception = TargetPerception(detector, tracker)
                    result = None
                elif operation in ("predict", "warmup"):
                    observation, steps = arguments
                    value = getattr(model, operation)(observation, steps)
                    result = value, model.last_diagnostic
                elif operation == "initialize":
                    result = perception.initialize(arguments)
                elif operation == "update":
                    result = perception.update(arguments)
                elif operation == "reset":
                    result = perception.reset()
                else:
                    raise ValueError(f"Unknown compute request: {operation}")
                connection.send((True, result))
            except Exception:
                connection.send((False, traceback.format_exc()))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class ComputeProcess:
    """最新任务仍由 GpuWorker 调度；阻塞 IPC 只发生在它的工作线程。"""

    def __init__(self):
        context = mp.get_context("spawn")
        self.connection, remote = context.Pipe()
        self.lock = Lock()
        self.process = context.Process(target=serve, args=(remote,), name="dexgraspvla_compute", daemon=True)
        self.process.start()
        remote.close()

    def call(self, operation, arguments=None):
        with self.lock:
            if not self.process.is_alive():
                raise RuntimeError("GPU compute process exited")
            self.connection.send((operation, arguments))
            success, result = self.connection.recv()
        if not success:
            raise RuntimeError(result)
        return result

    def close(self):
        if self.lock.acquire(timeout=2):
            try:
                if self.process.is_alive():
                    self.connection.send(("close", None))
            except (EOFError, BrokenPipeError):
                pass
            finally:
                self.lock.release()
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.connection.close()


class ModelProxy:
    """保持 ModelRuntime 的节点接口，数值转换在计算进程内完成。"""

    def __init__(self, compute, **parameters):
        self.compute = compute
        self.max_steps = compute.call("load_model", parameters)
        self.last_diagnostic = {}

    def predict(self, observation, steps):
        result, self.last_diagnostic = self.compute.call("predict", (observation, steps))
        return result

    def warmup(self, observation, steps):
        _, self.last_diagnostic = self.compute.call("warmup", (observation, steps))


class PerceptionProxy:
    """跟踪状态只在计算进程内更新；这里只镜像是否已初始化。"""

    def __init__(self, compute, **parameters):
        self.compute = compute
        self.tracker = SimpleNamespace(initialized=False)
        compute.call("load_perception", parameters)

    def initialize(self, image):
        result = self.compute.call("initialize", image)
        self.tracker.initialized = True
        return result

    def update(self, image):
        return self.compute.call("update", image)

    def reset(self):
        self.compute.call("reset")
        self.tracker.initialized = False
