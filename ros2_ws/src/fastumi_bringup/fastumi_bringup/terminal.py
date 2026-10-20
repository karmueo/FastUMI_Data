"""键盘工具共用的逐键终端读取。"""

import select
import termios
import tty


class TerminalKeyReader:
    """临时启用逐键输入，并在退出时恢复终端配置。"""

    def __init__(self, stream):
        self.stream = stream
        self.original_settings = None

    def __enter__(self):
        if not self.stream.isatty():
            raise RuntimeError("键盘命令需要交互终端；请在第二个终端运行 ros2 run")
        fd = self.stream.fileno()
        self.original_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
        except BaseException:
            termios.tcsetattr(fd, termios.TCSADRAIN, self.original_settings)
            self.original_settings = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_settings is not None:
            termios.tcsetattr(
                self.stream.fileno(), termios.TCSADRAIN, self.original_settings
            )
            self.original_settings = None

    def read_key(self, timeout_seconds):
        """等待单个按键；超时返回 None。"""
        readable, _, _ = select.select([self.stream], [], [], timeout_seconds)
        return self.stream.read(1) if readable else None
