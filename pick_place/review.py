"""本地 PyQt5 复核窗口：同步视频、夹爪曲线和可保存的阶段边界。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sys

import h5py
import numpy as np

from pick_place.core import (
    STAGES, delete_annotation, load_manifest, midpoint_index, restore_annotation,
    save_json, update_annotation,
)
from pick_place.pipeline import validate_paths, write_report

# OpenCV 可能设置自己附带的 Qt 插件目录；PyQt5 应使用其自身插件。
if "cv2" in os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", ""):
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


STATUS_TEXT = {"auto": "自动通过", "review": "待核对", "accepted": "确认保留", "excluded": "排除"}


def preview_path(record: dict, output: Path, input_root: Path) -> tuple[Path, str, list[int]]:
    """优先使用同步缓存；质量拒绝时只解码视频供查看，不绕过导出检查。"""
    if record.get("hdf5") and (output / record["hdf5"]).is_file():
        return output / record["hdf5"], "observations/images/front", record["image_stamps_ns"]
    from fastumi_data.bag_images import BagImageDecoder
    from fastumi_data.frame_spool import FrameSpool
    from fastumi_data.image_modes import mode_for_message_type
    from fastumi_data.link7_arrays import letterbox_rgb_uint8
    from fastumi_data.link7_record import open_reader

    preview_key = hashlib.sha256((record["id"] + record["fingerprint"]).encode()).hexdigest()
    cache = output / record.get("cache", f".cache/preview-{preview_key}")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "preview.h5"
    if path.exists():
        with h5py.File(path, "r") as handle:
            return path, "frames", handle["stamps_ns"][:].tolist()
    bag = input_root / record["bag_rel"]
    reader = open_reader(bag)
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    topic = record.get("image_topic", "/umi_camera/image_raw")
    if topic not in types:
        raise ValueError("录制没有可预览的图像话题")
    decoder = BagImageDecoder(mode_for_message_type(types[topic]), "rgb")
    temporary = path.with_suffix(".tmp.h5")
    spool = FrameSpool(temporary)
    try:
        reader = open_reader(bag, [topic])
        while reader.has_next():
            _, data, bag_ns = reader.read_next()
            for frame in decoder.feed(data, bag_ns):
                spool.append(frame.stamp_ns, len(spool), letterbox_rgb_uint8(frame.image))
        for frame in decoder.drain():
            spool.append(frame.stamp_ns, len(spool), letterbox_rgb_uint8(frame.image))
        stamps = spool.timestamps_ns
        if not len(stamps):
            raise ValueError("没有可解码的图像")
        spool.close(delete=False)
        with h5py.File(temporary, "a") as handle:
            handle.create_dataset("stamps_ns", data=stamps)
        os.replace(temporary, path)
        return path, "frames", stamps.tolist()
    finally:
        spool.close(delete=True)


class ReviewWindow(QtWidgets.QMainWindow):
    """逐条复核并原子保存清单；未应用修改在切换和关闭前提示。"""

    def __init__(self, output: Path):
        """打开分段清单，不启动 ROS 节点或改写源录制。"""
        super().__init__()
        self.output = Path(output)
        self.manifest = load_manifest(self.output / "segments.json")
        self.record = None
        self.video = None
        self.stamps = []
        self.dirty = False
        self.loading = False
        self.drag_key = None
        self.setWindowTitle("FastUMI 抓取 / 放置分段复核")
        self.resize(1250, 850)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.next_frame)
        container = QtWidgets.QWidget()
        self.setCentralWidget(container)
        layout = QtWidgets.QHBoxLayout(container)
        left = QtWidgets.QVBoxLayout()
        self.show_all = QtWidgets.QCheckBox("显示全部记录（含已删除，可恢复）")
        self.show_all.toggled.connect(self.refresh_records)
        left.addWidget(self.show_all)
        self.records = QtWidgets.QListWidget()
        self.records.setMinimumWidth(300)
        self.records.currentItemChanged.connect(self.select_record)
        left.addWidget(self.records)
        layout.addLayout(left, 1)
        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 3)
        self.message = QtWidgets.QLabel("请选择记录")
        self.message.setWordWrap(True)
        right.addWidget(self.message)
        self.image = QtWidgets.QLabel()
        self.image.setMinimumHeight(250)
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        right.addWidget(self.image, 2)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.display_frame)
        right.addWidget(self.slider)
        controls = QtWidgets.QHBoxLayout()
        for title, callback in (("上一帧", lambda: self.slider.setValue(self.slider.value() - 1)),
                                ("播放 / 暂停", self.toggle_play), ("下一帧", self.next_frame)):
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(callback)
            controls.addWidget(button)
        self.frame_label = QtWidgets.QLabel()
        controls.addWidget(self.frame_label)
        right.addLayout(controls)
        figure = Figure(figsize=(8, 2.2))
        self.canvas = FigureCanvasQTAgg(figure)
        self.axes = figure.add_subplot(111)
        self.canvas.mpl_connect("button_press_event", self.plot_press)
        self.canvas.mpl_connect("motion_notify_event", self.plot_move)
        self.canvas.mpl_connect("button_release_event", self.plot_release)
        right.addWidget(self.canvas, 1)
        form = QtWidgets.QGridLayout()
        right.addLayout(form)
        self.fields = {}
        labels = (("close_ns", "合拢（相对秒）"), ("open_ns", "打开（相对秒）"),
                  ("start", "起始帧"), ("split", "交界帧"), ("end", "结束帧"))
        for row, (key, label) in enumerate(labels):
            field = QtWidgets.QDoubleSpinBox() if key.endswith("ns") else QtWidgets.QSpinBox()
            if key.endswith("ns"):
                field.setDecimals(6)
                field.setSingleStep(1 / 30)
            field.setRange(-1, 1_000_000)
            field.setSpecialValueText("未设置")
            field.valueChanged.connect(self.mark_dirty)
            self.fields[key] = field
            form.addWidget(QtWidgets.QLabel(label), row, 0)
            form.addWidget(field, row, 1)
            button = QtWidgets.QPushButton("设为当前帧")
            button.clicked.connect(lambda checked=False, name=key: self.use_current(name))
            form.addWidget(button, row, 2)
        recalculate = QtWidgets.QPushButton("按合拢 / 打开事件重算时间中点")
        recalculate.clicked.connect(self.recalculate)
        form.addWidget(recalculate, 5, 0, 1, 3)
        self.decisions, self.reasons = {}, {}
        for row, stage in enumerate(STAGES, 6):
            combo = QtWidgets.QComboBox()
            for status, text in STATUS_TEXT.items():
                if status != "auto":
                    combo.addItem(text, status)
            combo.currentIndexChanged.connect(self.decision_dirty)
            reason = QtWidgets.QLineEdit()
            reason.setPlaceholderText("排除原因或复核备注")
            reason.textEdited.connect(self.decision_dirty)
            self.decisions[stage], self.reasons[stage] = combo, reason
            form.addWidget(QtWidgets.QLabel(stage), row, 0)
            form.addWidget(combo, row, 1)
            form.addWidget(reason, row, 2)
        self.save_button = QtWidgets.QPushButton("应用并保存当前记录")
        self.save_button.clicked.connect(self.save_current)
        right.addWidget(self.save_button)
        actions = QtWidgets.QHBoxLayout()
        self.delete_button = QtWidgets.QPushButton("删除当前记录")
        self.delete_button.setToolTip("将 pick/place 两阶段排除；原始文件保留，可在显示全部记录中恢复")
        self.delete_button.clicked.connect(self.delete_current)
        actions.addWidget(self.delete_button)
        self.restore_button = QtWidgets.QPushButton("恢复当前记录")
        self.restore_button.clicked.connect(self.restore_current)
        actions.addWidget(self.restore_button)
        right.addLayout(actions)
        self.update_actions()
        self.refresh_records()

    def record_label(self, record):
        """返回基于已保存状态的列表文字，已删除记录有单独标识。"""
        annotation = record["annotation"]
        state = "已删除" if annotation.get("deleted") else " / ".join(
            STATUS_TEXT[annotation["stages"][stage]["status"]] for stage in STAGES)
        return record["id"] + "\n" + state

    def update_actions(self):
        """未选中或已删除记录时禁用保存、删除，防止意外重新纳入训练。"""
        deleted = bool(self.record and self.record["annotation"].get("deleted"))
        active = self.record is not None and not deleted
        self.save_button.setEnabled(active)
        self.delete_button.setEnabled(active)
        self.restore_button.setEnabled(deleted)
        for widget in (*self.fields.values(), *self.decisions.values(), *self.reasons.values()):
            widget.setEnabled(active)

    def refresh_records(self):
        """刷新筛选列表，保持未应用修改不会静默丢失。"""
        if self.dirty and not self.resolve_dirty():
            return
        self.records.clear()
        for index, record in enumerate(self.manifest["records"]):
            states = record["annotation"]["stages"]
            if self.show_all.isChecked() or any(value["status"] == "review" for value in states.values()):
                item = QtWidgets.QListWidgetItem(self.record_label(record))
                item.setData(QtCore.Qt.UserRole, index)
                self.records.addItem(item)

    def select_record(self, item, previous):
        """载入一条视频缓存和标注；同步失败的录制仍可查看并排除。"""
        if item is None:
            self.timer.stop()
            if self.video is not None:
                self.video.close()
            self.video, self.record = None, None
            self.stamps = []
            self.image.clear()
            self.frame_label.clear()
            self.message.setText("当前列表没有选中的记录")
            self.axes.clear()
            self.canvas.draw_idle()
            self.update_actions()
            return
        if self.dirty and not self.resolve_dirty():
            self.records.blockSignals(True)
            self.records.setCurrentItem(previous)
            self.records.blockSignals(False)
            return
        self.timer.stop()
        self.loading = True
        self.record = self.manifest["records"][item.data(QtCore.Qt.UserRole)]
        if self.video is not None:
            self.video.close()
            self.video = None
        self.stamps = []
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            path, dataset, self.stamps = preview_path(self.record, self.output, Path(self.manifest["input_root"]))
            self.video = h5py.File(path, "r")
            self.frames = self.video[dataset]
            self.origin = self.stamps[0]
            self.slider.setRange(0, len(self.stamps) - 1)
            self.slider.setValue(0)
            self.populate_fields()
            reasons = self.record["quality_errors"] + [reason["message"] for reason in self.record["events"]["reasons"]]
            self.message.setText(self.record["id"] + "\n" + ("；".join(reasons) or "可自动切分；请结合画面确认任务结果。"))
            self.draw_plot()
            self.display_frame(0)
        except Exception as error:
            self.image.clear()
            self.message.setText(f"无法预览：{error}")
            self.origin = 0
            self.populate_fields()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.loading = False
            self.dirty = False
            self.update_actions()

    def populate_fields(self):
        """把已保存的事件纳秒转换为相对秒，帧边界保持整数。"""
        annotation = self.record["annotation"]
        for key, field in self.fields.items():
            value = annotation[key]
            if key.endswith("ns") and value is not None:
                value = (value - self.origin) / 1e9
            field.setValue(-1 if value is None else value)
        for stage in STAGES:
            decision = annotation["stages"][stage]
            status = "accepted" if decision["status"] == "auto" else decision["status"]
            self.decisions[stage].setCurrentIndex(self.decisions[stage].findData(status))
            self.reasons[stage].setText(decision["reason"])

    def draw_plot(self):
        """绘制原始夹爪曲线与事件、阶段和当前帧标线。"""
        self.axes.clear()
        if self.record.get("gripper_cache"):
            with np.load(self.output / self.record["gripper_cache"]) as arrays:
                seconds = (arrays["stamps_ns"] - self.origin) / 1e9
                self.axes.plot(seconds, arrays["raw"], color="gray", label="gripper")
                invalid = ~arrays["valid"]
                self.axes.scatter(seconds[invalid], arrays["raw"][invalid], color="red", s=10)
        self.axes.set_xlabel("seconds")
        self.axes.set_ylabel("openness")
        self.lines = {}
        colors = {"close_ns": "orange", "open_ns": "green", "start": "blue", "split": "red", "end": "purple"}
        for key, color in colors.items():
            value = self.fields[key].value()
            second = value if key.endswith("ns") else (
                (self.stamps[int(value)] - self.origin) / 1e9 if 0 <= value < len(self.stamps) else -1)
            self.lines[key] = self.axes.axvline(second, color=color, label=key.replace("_ns", ""))
        self.cursor = self.axes.axvline(0, color="black", linestyle=":")
        self.axes.legend(loc="upper right", fontsize=7, ncol=3)
        self.canvas.figure.tight_layout()
        self.canvas.draw_idle()

    def display_frame(self, index):
        """按帧索引随机读取 RGB 图像；图像与曲线共用源时间轴。"""
        if self.video is None or not self.stamps:
            return
        pixels = np.ascontiguousarray(self.frames[index])
        image = QtGui.QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QtGui.QImage.Format_RGB888).copy()
        self.image.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
            self.image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        second = (self.stamps[index] - self.origin) / 1e9
        self.frame_label.setText(f"帧 {index}/{len(self.stamps) - 1} · {second:.3f} 秒")
        if hasattr(self, "cursor"):
            self.cursor.set_xdata([second, second])
            self.canvas.draw_idle()

    def next_frame(self):
        """前进一帧，末尾停止播放。"""
        if self.slider.value() >= self.slider.maximum():
            self.timer.stop()
        else:
            self.slider.setValue(self.slider.value() + 1)

    def toggle_play(self):
        """切换约 30 Hz 的离线视频播放。"""
        self.timer.stop() if self.timer.isActive() else self.timer.start()

    def mark_dirty(self, *_):
        """边界变化使已通过阶段回到待核对，防止沿用旧确认。"""
        if self.loading or self.record is None or self.record["annotation"].get("deleted"):
            return
        self.dirty = True
        for combo in self.decisions.values():
            if combo.currentData() != "excluded":
                combo.setCurrentIndex(combo.findData("review"))

    def decision_dirty(self, *_):
        """记录阶段决策或备注尚未保存。"""
        if not self.loading:
            self.dirty = True

    def use_current(self, key):
        """将边界或事件设置到当前可见图像帧。"""
        if self.stamps and self.record and not self.record["annotation"].get("deleted"):
            index = self.slider.value()
            self.fields[key].setValue((self.stamps[index] - self.origin) / 1e9 if key.endswith("ns") else index)
            self.draw_plot()

    def event_values(self):
        """读取事件值；未编辑字段保留原纳秒，避免小数显示造成精度损失。"""
        values = {}
        for key in ("close_ns", "open_ns"):
            second = self.fields[key].value()
            original = self.record["annotation"][key]
            if original is not None and abs(second - (original - self.origin) / 1e9) <= 0.00000051:
                values[key] = original
            else:
                values[key] = None if second < 0 else self.origin + round(second * 1e9)
        return values

    def recalculate(self):
        """按当前事件字段重新定位时间中点。"""
        if self.record is None or self.record["annotation"].get("deleted"):
            return
        try:
            events = self.event_values()
            if any(value is None for value in events.values()):
                raise ValueError("请先设置合拢和打开时间")
            self.fields["split"].setValue(midpoint_index(self.record["image_stamps_ns"], events["close_ns"], events["open_ns"]))
            self.draw_plot()
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "无法重算", str(error))

    def plot_press(self, event):
        """点击曲线定位；靠近边界标线时开始拖动该边界。"""
        if event.inaxes != self.axes or event.xdata is None or not self.stamps:
            return
        threshold = (self.axes.get_xlim()[1] - self.axes.get_xlim()[0]) * 0.015
        self.drag_key = next((key for key, line in self.lines.items()
                              if abs(line.get_xdata()[0] - event.xdata) < threshold), None)
        self.plot_move(event)

    def plot_move(self, event):
        """拖动时按最近视频帧更新字段，等距取较早帧。"""
        if event.inaxes != self.axes or event.xdata is None or not self.stamps or event.button != 1:
            return
        target = self.origin + round(event.xdata * 1e9)
        index = min(range(len(self.stamps)), key=lambda i: abs(self.stamps[i] - target))
        self.slider.setValue(index)
        if self.drag_key:
            self.use_current(self.drag_key)

    def plot_release(self, _):
        """结束标线拖动。"""
        self.drag_key = None

    def save_current(self) -> bool:
        """验证后原子保存标注与报告；失败时保留原清单。"""
        if self.record is None or self.record["annotation"].get("deleted"):
            return True
        try:
            edited = deepcopy(self.record)
            events = {key: value for key, value in self.event_values().items()
                      if value != edited["annotation"][key]}
            bounds = {key: (None if self.fields[key].value() < 0 else int(self.fields[key].value()))
                      for key in ("start", "split", "end")}
            bounds = {key: value for key, value in bounds.items() if value != edited["annotation"][key]}
            decisions = {stage: {"status": self.decisions[stage].currentData(), "reason": self.reasons[stage].text()}
                         for stage in STAGES}
            update_annotation(edited, events=events, bounds=bounds, decisions=decisions)
            self.persist_record(edited)
            statuses = " / ".join(STATUS_TEXT[value["status"]] for value in edited["annotation"]["stages"].values())
            self.statusBar().showMessage(f"已保存：{statuses}；保存不会自动将待核对改为确认保留", 8000)
            return True
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "无法保存", str(error))
            return False

    def persist_record(self, edited):
        """持久化当前记录并同步列表标签，不在选中变化的信号中销毁列表项。"""
        manifest = deepcopy(self.manifest)
        index = self.manifest["records"].index(self.record)
        manifest["records"][index] = edited
        save_json(self.output / "segments.json", manifest)
        self.manifest, self.record = manifest, edited
        write_report(self.output, self.manifest)
        self.loading = True
        self.populate_fields()
        self.loading = False
        self.dirty = False
        for row in range(self.records.count()):
            item = self.records.item(row)
            if item.data(QtCore.Qt.UserRole) == index:
                item.setText(self.record_label(edited))
        self.draw_plot()
        self.update_actions()

    def delete_current(self):
        """立即排除两阶段并保存；隐藏当前列表项并选择下一条，保留恢复快照。"""
        if self.record is None or self.record["annotation"].get("deleted"):
            return
        try:
            edited = deepcopy(self.record)
            delete_annotation(edited)
            self.persist_record(edited)
            row = self.records.currentRow()
            if not self.show_all.isChecked():
                self.records.item(row).setHidden(True)
                candidates = list(range(row + 1, self.records.count())) + list(range(row - 1, -1, -1))
                next_item = next((self.records.item(i) for i in candidates
                                  if not self.records.item(i).isHidden()), None)
                self.records.setCurrentItem(next_item)
            self.statusBar().showMessage("已删除：两个阶段均已排除。可勾选显示全部记录后恢复；已有数据集需重新导出。", 10000)
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "无法删除", str(error))

    def restore_current(self):
        """恢复删除前标注，立即更新列表与持久化清单。"""
        if self.record is None or not self.record["annotation"].get("deleted"):
            return
        try:
            edited = deepcopy(self.record)
            restore_annotation(edited)
            self.persist_record(edited)
            self.statusBar().showMessage("已恢复删除前的标注；已有数据集需重新导出。", 8000)
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "无法恢复", str(error))

    def resolve_dirty(self) -> bool:
        """切换或关闭前询问尚未应用的修改。"""
        choice = QtWidgets.QMessageBox.question(self, "尚未保存", "保存当前修改？",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
        if choice == QtWidgets.QMessageBox.Cancel:
            return False
        if choice == QtWidgets.QMessageBox.Yes:
            return self.save_current()
        self.dirty = False
        return True

    def closeEvent(self, event):
        """关闭时保护未保存标注并释放 HDF5 文件。"""
        if self.dirty and not self.resolve_dirty():
            event.ignore()
            return
        self.timer.stop()
        if self.video is not None:
            self.video.close()
        event.accept()


def run_review(args) -> int:
    """启动本地窗口；无需运行 roscore、采集节点或硬件。"""
    _, output = validate_paths(args.input, args.output)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = ReviewWindow(output)
    window.show()
    return app.exec_()
