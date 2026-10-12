"""Check keyboard exits and cleanup without connecting to camera or robot."""

import os
import pty
import sys
import termios
from unittest.mock import Mock

import numpy as np
import pytest

import capture_arm_cam as capture


def test_terminal_escape_and_restore(monkeypatch):
    """Terminal ESC works without Enter and restores settings after an error."""
    master_fd, slave_fd = pty.openpty()
    try:
        original = termios.tcgetattr(slave_fd)
        stdin = Mock()
        stdin.isatty.return_value = True
        stdin.fileno.return_value = slave_fd
        monkeypatch.setattr(sys, "stdin", stdin)
        with pytest.raises(RuntimeError, match="preview failed"):
            with capture.terminal_keyboard() as fd:
                assert not termios.tcgetattr(fd)[3] & termios.ICANON
                assert termios.tcgetattr(fd)[3] & termios.ISIG
                assert not capture.terminal_exit_requested(fd)
                os.write(master_fd, b"\x1b[A")
                assert not capture.terminal_exit_requested(fd)
                os.write(master_fd, b"\x1b")
                assert capture.terminal_exit_requested(fd)
                raise RuntimeError("preview failed")
        assert termios.tcgetattr(slave_fd) == original
    finally:
        os.close(master_fd)
        os.close(slave_fd)


def test_redirected_stdin_is_not_read(monkeypatch):
    """Piped input is left untouched instead of requiring terminal settings."""
    stdin = Mock()
    stdin.isatty.return_value = False
    monkeypatch.setattr(sys, "stdin", stdin)
    with capture.terminal_keyboard() as fd:
        assert fd is None
        assert not capture.terminal_exit_requested(fd)
    stdin.fileno.assert_not_called()


@pytest.mark.parametrize("exit_method", [
    "window_escape", "terminal_escape", "window_close", "ctrl_c",
    "read_failure", "camera_failure", "window_failure",
])
def test_preview_exit_cleans_resources(monkeypatch, tmp_path, exit_method):
    """Every exit releases the camera, window, connection and SDK threads."""
    monkeypatch.setattr(sys, "argv", ["capture_arm_cam.py", "-o", str(tmp_path)])
    stdin = Mock()
    stdin.isatty.return_value = False
    monkeypatch.setattr(sys, "stdin", stdin)
    robot = Mock()
    monkeypatch.setattr(capture, "RobotArmController", Mock(return_value=robot))
    camera = Mock()
    camera.isOpened.return_value = exit_method != "camera_failure"
    camera.get.return_value = 30
    camera.read.return_value = (
        exit_method != "read_failure", np.zeros((4, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(capture.cv2, "VideoCapture", Mock(return_value=camera))
    for name in ("namedWindow", "resizeWindow", "putText", "imshow"):
        monkeypatch.setattr(capture.cv2, name, Mock())
    destroy = Mock()
    monkeypatch.setattr(capture.cv2, "destroyAllWindows", destroy)
    wait_key = Mock(return_value=27 if exit_method == "window_escape" else -1)
    monkeypatch.setattr(capture.cv2, "waitKey", wait_key)
    monkeypatch.setattr(capture.cv2, "getWindowProperty", Mock(return_value=0))
    monkeypatch.setattr(
        capture, "terminal_exit_requested",
        Mock(return_value=exit_method == "terminal_escape"),
    )
    if exit_method == "ctrl_c":
        wait_key.side_effect = KeyboardInterrupt
    if exit_method == "window_failure":
        capture.cv2.namedWindow.side_effect = RuntimeError("preview failed")

    if exit_method == "camera_failure":
        with pytest.raises(SystemExit) as error:
            capture.main()
        assert error.value.code == 1
    elif exit_method == "window_failure":
        with pytest.raises(RuntimeError, match="preview failed"):
            capture.main()
    else:
        capture.main()

    camera.release.assert_called_once_with()
    destroy.assert_called_once_with()
    robot.disconnect.assert_called_once_with()
    robot.arm.rm_destroy.assert_called_once_with()
