"""解析组件配置并回收本会话创建的 Linux 子进程，不依赖 ROS 或 Qt。"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Callable

import yaml


@dataclass
class Health:
    """按单调接收时间判断数据新鲜度，不将静止位姿误判为断流。"""

    # 每个探针的超时秒数。
    deadlines: dict[str, float]
    # 最近收到的探针时间和有效性。
    samples: dict[str, tuple[float, bool]] = field(default_factory=dict)
    # 可替换时钟供确定性测试使用。
    clock: Callable[[], float] = time.monotonic

    def receive(self, key: str, valid: bool = True) -> None:
        """记录探针接收时间及业务有效性。"""
        self.samples[key] = (self.clock(), bool(valid))

    def check(self) -> tuple[bool, str]:
        """返回整体健康和可直接展示的异常原因。"""
        issues = []
        for key, limit in self.deadlines.items():
            sample = self.samples.get(key)
            if sample is None:
                issues.append(f"{key}: 尚无数据")
            elif self.clock() - sample[0] > limit:
                issues.append(f"{key}: 超时 {self.clock() - sample[0]:.2f}s")
            elif not sample[1]:
                issues.append(f"{key}: 数据无效")
        return not issues, "；".join(issues) or "数据正常"

    def ages(self) -> str:
        """格式化各探针接收间隔供诊断显示。"""
        return ", ".join(
            f"{key}={self.clock() - value[0]:.2f}s"
            for key, value in self.samples.items()
        )


def workspace_root() -> Path:
    """从源码或安装位置查找包含共享虚拟环境的工作区。"""
    for root in Path(__file__).resolve().parents:
        if (root / ".venv-numpy2/bin/python").is_file():
            return root
    raise ValueError("无法定位工作区，请在管理配置中设置 workspace_root")


def read_configuration(path: str) -> dict:
    """读取管理配置并拒绝未知管理模式和非法组件标识。"""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("components"), dict):
        raise ValueError("管理配置必须包含 components 映射")
    for field_name, default in (("discovery_wait_s", 2), ("startup_grace_s", 30), ("save_timeout_s", 300)):
        value = float(config.get(field_name, default))
        if not math.isfinite(value) or value < 0 or (field_name != "discovery_wait_s" and value == 0):
            raise ValueError(f"无效时间参数: {field_name}")
    for key, item in config["components"].items():
        if not isinstance(key, str) or not isinstance(item, dict):
            raise ValueError("组件标识必须为字符串，配置必须为映射")
        if key not in {"arm", "tracker", "gripper", "umi_camera", "estimator", "wrist_camera", "teleop", "recorder"}:
            raise ValueError(f"未知组件: {key}")
        if not isinstance(item.get("parameters", {}), dict) or not isinstance(item.get("label", key), str):
            raise ValueError(f"组件标签或 parameters 无效: {key}")
        if not isinstance(item.get("nodes", []), list) or any(
            not isinstance(name, str) or not name.startswith("/") for name in item.get("nodes", [])
        ):
            raise ValueError(f"nodes 必须为节点全名列表: {key}")
        if not key.isidentifier() or item.get("mode", "auto") not in (
            "auto", "observe", "disabled"
        ):
            raise ValueError(f"无效组件配置: {key}")
        timeout = float(item.get("health_timeout_s", 1.0))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"health_timeout_s 必须为正: {key}")
    return config


def process_table() -> dict[int, tuple[int, str, str]]:
    """读取 PID、父 PID、启动标识及状态，避免 PID 重用时误杀。"""
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            result[int(entry.name)] = (int(fields[1]), fields[19], fields[0])
        except (OSError, ValueError, IndexError):
            continue
    return result


class ManagedProcess:
    """保留子孙进程身份并分阶段结束它们，包括调用 setsid 的子进程。"""

    def __init__(self, command: list[str], environment: dict, log_path: Path,
                 cwd: Path, clock=time.monotonic) -> None:
        """以独立会话启动参数数组，将输出写入日志；启动失败向上传播。"""
        self.clock = clock
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as output:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("process_guard.py")), *command],
                env=environment, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            )
        # 跟踪集合只保存由本对象创建的进程及已发现的后代身份。
        self.identities: dict[int, str] = {}
        self.stop_started: float | None = None
        self.escalation = 0
        self.refresh()

    def refresh(self, table=None) -> dict:
        """发现新后代并清理退出身份；返回当前进程快照。"""
        table = process_table() if table is None else table
        root = table.get(self.process.pid)
        if root and self.process.poll() is None:
            self.identities.setdefault(self.process.pid, root[1])
        self.identities = {
            pid: stamp for pid, stamp in self.identities.items()
            if pid in table and table[pid][1] == stamp and table[pid][2] != "Z"
        }
        changed = True
        while changed:
            changed = False
            for pid, (parent, stamp, state) in table.items():
                if parent in self.identities and pid not in self.identities and state != "Z":
                    self.identities[pid] = stamp
                    changed = True
        return table

    def signal_owned(self, signum: int) -> None:
        """仅向身份仍匹配的受管理进程发信号。"""
        table = self.refresh()
        for pid, stamp in list(self.identities.items()):
            if table.get(pid, (None, None))[1] == stamp:
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    pass

    def stop(self) -> None:
        """开始优雅停止；launch 自行转发信号，保留后代用于超时兜底。"""
        if self.stop_started is not None:
            return
        self.refresh()
        self.stop_started = self.clock()
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
        else:
            self.signal_owned(signal.SIGINT)

    def poll(self, grace: float = 5.0, table=None) -> int | None:
        """非阻塞轮询；宽限后升级信号，子孙全部退出才报告完成。"""
        self.refresh(table)
        code = self.process.poll()
        if code is not None and self.identities and self.stop_started is None:
            self.stop()
        if self.stop_started is not None:
            age = self.clock() - self.stop_started
            if age > grace and self.escalation == 0:
                self.signal_owned(signal.SIGTERM)
                self.escalation = 1
            if age > grace + 2.0 and self.escalation == 1:
                self.signal_owned(signal.SIGKILL)
                self.escalation = 2
        return code if code is not None and not self.identities else None
