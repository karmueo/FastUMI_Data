"""作为每个组件的 Linux 子收割器，确保独立会话的后代不会在父进程退出后残留。"""

import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def descendants():
    """查找当前守护进程的全部后代，包括已被收养的双重 fork 后代。"""
    parents = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
                if fields[0] != "Z":
                    parents[int(path.name)] = int(fields[1])
            except (OSError, IndexError, ValueError):
                pass
    owned = {os.getpid()}
    previous = set()
    while previous != owned:
        previous = set(owned)
        owned.update(pid for pid, parent in parents.items() if parent in owned)
    return owned - {os.getpid()}


def main():
    """执行参数数组，转发退出信号并等待所有后代结束，返回原命令退出码。"""
    # PR_SET_CHILD_SUBREAPER 将孤儿后代交给本进程，避免 setsid 绕过进程组回收。
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "无法设置组件子收割器")
    requested = []
    signal.signal(signal.SIGINT, lambda *_: requested.append(signal.SIGINT))
    signal.signal(signal.SIGTERM, lambda *_: requested.append(signal.SIGTERM))
    # 管理器意外消失时也触发收尾，避免硬件进程失去所有者。
    if library.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "无法设置父进程退出信号")
    if os.getppid() == 1:
        requested.append(signal.SIGTERM)
    child = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL)
    stop_started = None
    stop_grace = float(os.environ.get("TRACKER_COMPONENT_STOP_GRACE", "5"))
    cleanup_started = None
    cleanup_stage = 0
    while True:
        if requested:
            signum = requested.pop(0)
            if stop_started is None:
                stop_started = time.monotonic()
            if child.poll() is None:
                child.send_signal(signum)
        code = child.poll()
        if code is None and stop_started is not None and time.monotonic() - stop_started > stop_grace:
            print("组件退出宽限已超时，强制结束；未完成保存可能丢失", file=sys.stderr, flush=True)
            for pid in descendants():
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            stop_started = None
        if code is not None:
            # 子命令已结束，收割被本守护进程收养的孤儿后代。
            try:
                while os.waitpid(-1, os.WNOHANG)[0] > 0:
                    pass
            except ChildProcessError:
                pass
            owned = descendants()
            if not owned:
                return code if code >= 0 else 128 - code
            now = time.monotonic()
            if cleanup_started is None:
                cleanup_started = now
            age = now - cleanup_started
            stage = 3 if age > 7 else 2 if age > 5 else 1
            if stage > cleanup_stage:
                signum = {1: signal.SIGINT, 2: signal.SIGTERM, 3: signal.SIGKILL}[stage]
                for pid in owned:
                    try:
                        os.kill(pid, signum)
                    except ProcessLookupError:
                        pass
                cleanup_stage = stage
        time.sleep(0.05)


if __name__ == "__main__":
    sys.exit(main())
