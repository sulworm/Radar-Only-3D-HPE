"""Desktop UI for real-time radar point clouds and pose inference."""

import sys
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QBrush, QPainter, QPen, QPolygonF
from OpenGL import GL
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph.opengl as gl

from radar_pipeline import (
    DEFAULT_PITCH_DEG,
    DEFAULT_RADAR_HEIGHT,
    SEQ_LEN,
)
from pose_filter import DEFAULT_MEASUREMENT_NOISE, DEFAULT_PROCESS_NOISE
from realtime_inference import DEFAULT_MODEL_PATH
from realtime_worker import FrameUpdate, RadarWorker, RuntimeConfig
from ui_config import (
    AXIS_LEGEND_HEIGHT,
    AXIS_LEGEND_TEXT_FONT_PT,
    AXIS_LEGEND_TITLE_FONT_PT,
    AXIS_LEGEND_WIDTH,
    DEFAULT_PORT,
    DISPLAY_THEMES,
    SKELETON_BONES,
    STYLE_TEMPLATE,
    UI_BASE_FONT_PX,
    UI_TITLE_FONT_PX,
)


OVERLAY_GL_OPTIONS = {
    GL.GL_DEPTH_TEST: False,
    GL.GL_BLEND: True,
    GL.GL_CULL_FACE: False,
    "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA),
}


class AxisLegendWidget(QWidget):
    def __init__(self, parent=None, radar_height=DEFAULT_RADAR_HEIGHT):
        super().__init__(parent)
        self._theme = DISPLAY_THEMES["dark"]
        self._radar_height = radar_height
        self.setFixedSize(AXIS_LEGEND_WIDTH, AXIS_LEGEND_HEIGHT)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)

    def set_theme(self, theme):
        self._theme = theme
        self.update()

    def set_radar_height(self, radar_height):
        self._radar_height = radar_height
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        panel_bg = QColor(self._theme["panel_bg"])
        panel_bg.setAlpha(222)
        border = QColor(self._theme["border"])
        border.setAlpha(210)
        painter.setPen(QPen(border, 1))
        painter.setBrush(QBrush(panel_bg))
        painter.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 8, 8)

        title_color = QColor(self._theme["title"])
        text_color = QColor(self._theme["text"])
        muted_color = QColor(self._theme["muted"])
        x_color = QColor("#3b82f6")
        y_color = QColor("#f59e0b")
        z_color = QColor("#22c55e")

        font = painter.font()
        font.setPointSize(AXIS_LEGEND_TITLE_FONT_PT)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(title_color)
        painter.drawText(14, 22, "Radar frame")

        origin = QPointF(60, 104)
        z_axis_top = QPointF(60, 40)
        self._draw_axis(painter, origin, QPointF(118, 104), x_color, "X", QPointF(121, 108))
        self._draw_axis(painter, origin, QPointF(36, 78), y_color, "Y", QPointF(25, 76))
        self._draw_axis(painter, origin, z_axis_top, z_color, "Z", QPointF(64, 40))

        painter.setPen(QPen(title_color, 2))
        painter.setBrush(QBrush(title_color))
        painter.drawEllipse(origin, 4.0, 4.0)
        painter.setPen(muted_color)
        painter.drawText(42, 126, "origin")

        font.setPointSize(AXIS_LEGEND_TEXT_FONT_PT)
        font.setBold(False)
        painter.setFont(font)
        self._draw_radar_height_marker(
            painter,
            origin,
            z_axis_top,
            z_color,
            text_color,
        )
        self._draw_legend_line(painter, 48, x_color, "X radar front", text_color)
        self._draw_legend_line(painter, 70, y_color, "Y lateral", text_color)
        self._draw_legend_line(painter, 92, z_color, "Z height", text_color)
        self._draw_legend_dot(
            painter,
            116,
            z_color,
            f"Radar position {self._radar_height:.2f} m",
            text_color,
        )

    def _draw_axis(self, painter, start, end, color, label, label_pos):
        painter.setPen(QPen(color, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(start, end)
        self._draw_arrow_head(painter, start, end, color)
        painter.setPen(color)
        painter.drawText(label_pos, label)

    def _draw_arrow_head(self, painter, start, end, color):
        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        size = 8
        left = QPointF(
            end.x() - math.cos(angle - math.pi / 6) * size,
            end.y() - math.sin(angle - math.pi / 6) * size,
        )
        right = QPointF(
            end.x() - math.cos(angle + math.pi / 6) * size,
            end.y() - math.sin(angle + math.pi / 6) * size,
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPolygon(QPolygonF([end, left, right]))

    def _draw_legend_line(self, painter, y, color, text, text_color):
        painter.setPen(QPen(color, 3, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(134, y), QPointF(160, y))
        painter.setPen(text_color)
        painter.drawText(170, y + 4, text)

    def _draw_legend_dot(self, painter, y, color, text, text_color):
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(147, y), 4.0, 4.0)
        painter.setPen(text_color)
        painter.drawText(170, y + 4, text)

    def _draw_radar_height_marker(self, painter, origin, z_axis_top, color, text_color):
        height = max(float(self._radar_height), 0.0)
        z_scale_top = max(2.0, height * 1.2)
        ratio = min(height / z_scale_top, 1.0)
        marker_y = origin.y() + (z_axis_top.y() - origin.y()) * ratio
        marker = QPointF(origin.x(), marker_y)

        painter.setPen(QPen(color, 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(marker.x() - 8, marker.y()), QPointF(marker.x() + 8, marker.y()))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(marker, 5.0, 5.0)
        painter.setPen(text_color)
        painter.drawText(QPointF(marker.x() + 14, marker.y() + 4), f"{height:.2f} m")


class RadarRealtimeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.latest_frame = None
        self.received_frames = 0
        self.collection_started_at = None
        self.latest_pose = None
        self.theme_name = "dark"
        self._style_signature = None

        self.setWindowTitle("Real-time mmWave Point Cloud")
        self.resize(1280, 800)
        self._build_ui()
        self._apply_style(force=True)
        self.scan_ports()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=8, elevation=35.26, azimuth=135)
        self.gl_view.setBackgroundColor("#111827")
        layout.addWidget(self.gl_view, 1)

        self.axis_legend = AxisLegendWidget(self.gl_view, DEFAULT_RADAR_HEIGHT)
        self.axis_legend.move(14, 14)
        self.axis_legend.raise_()

        self.grid = gl.GLGridItem()
        self.grid.setSize(8, 8)
        self.grid.setSpacing(0.5, 0.5)
        self.gl_view.addItem(self.grid)

        axis = gl.GLAxisItem()
        axis.setSize(2, 2, 2)
        self.gl_view.addItem(axis)

        dark_theme = DISPLAY_THEMES["dark"]
        self.cloud_outline_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
            size=dark_theme["cloud_outline_size"],
            pxMode=True,
            glOptions=OVERLAY_GL_OPTIONS,
        )
        self.cloud_outline_scatter.setDepthValue(10)
        self.gl_view.addItem(self.cloud_outline_scatter)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
            size=dark_theme["cloud_size"],
            pxMode=True,
            glOptions=OVERLAY_GL_OPTIONS,
        )
        self.scatter.setDepthValue(11)
        self.gl_view.addItem(self.scatter)

        self.joint_outline_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=dark_theme["joint_outline"],
            size=dark_theme["joint_outline_size"],
            pxMode=True,
            glOptions=OVERLAY_GL_OPTIONS,
        )
        self.joint_outline_scatter.setDepthValue(30)
        self.gl_view.addItem(self.joint_outline_scatter)

        self.joint_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=dark_theme["joint"],
            size=dark_theme["joint_size"],
            pxMode=True,
            glOptions=OVERLAY_GL_OPTIONS,
        )
        self.joint_scatter.setDepthValue(31)
        self.gl_view.addItem(self.joint_scatter)

        self.skeleton_outline_lines = []
        for _ in SKELETON_BONES:
            line = gl.GLLinePlotItem(
                pos=np.zeros((0, 3), dtype=np.float32),
                color=dark_theme["bone_outline"],
                width=dark_theme["bone_outline_width"],
                antialias=True,
                glOptions=OVERLAY_GL_OPTIONS,
            )
            line.setDepthValue(20)
            self.gl_view.addItem(line)
            self.skeleton_outline_lines.append(line)

        self.skeleton_lines = []
        for _ in SKELETON_BONES:
            line = gl.GLLinePlotItem(
                pos=np.zeros((0, 3), dtype=np.float32),
                color=dark_theme["bone"],
                width=dark_theme["bone_width"],
                antialias=True,
                glOptions=OVERLAY_GL_OPTIONS,
            )
            line.setDepthValue(21)
            self.gl_view.addItem(line)
            self.skeleton_lines.append(line)

        panel = QWidget()
        panel.setObjectName("controlPanel")
        panel.setMinimumWidth(300)
        self.control_panel = panel
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(10)
        layout.addWidget(panel)

        title = QLabel("Real-time radar")
        title.setObjectName("title")
        subtitle = QLabel("Serial input and front-end correction")
        subtitle.setObjectName("subtitle")
        self.theme_button = QPushButton("Light theme")
        self.theme_button.setObjectName("themeButton")
        self.theme_button.clicked.connect(self.toggle_theme)
        title_row = QHBoxLayout()
        title_row.addWidget(title, 1)
        title_row.addWidget(self.theme_button)
        panel_layout.addLayout(title_row)
        panel_layout.addWidget(subtitle)

        connection_group = QGroupBox("Connection")
        connection_form = QFormLayout(connection_group)
        self.port_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.scan_ports)
        port_row = QHBoxLayout()
        port_row.addWidget(self.port_combo, 1)
        port_row.addWidget(self.refresh_button)
        connection_form.addRow("Serial port", port_row)

        self.baudrate_spin = QSpinBox()
        self.baudrate_spin.setRange(1, 10_000_000)
        self.baudrate_spin.setValue(2_000_000)
        self.baudrate_spin.setSingleStep(100_000)
        connection_form.addRow("Baudrate", self.baudrate_spin)
        panel_layout.addWidget(connection_group)

        correction_group = QGroupBox("Front-end correction")
        correction_form = QFormLayout(correction_group)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(-10.0, 10.0)
        self.height_spin.setDecimals(2)
        self.height_spin.setSingleStep(0.05)
        self.height_spin.setSuffix(" m")
        self.height_spin.setValue(DEFAULT_RADAR_HEIGHT)
        self.height_spin.valueChanged.connect(self.axis_legend.set_radar_height)
        self.height_spin.valueChanged.connect(self._push_runtime_config)
        correction_form.addRow("Radar height", self.height_spin)

        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(-90.0, 90.0)
        self.pitch_spin.setDecimals(1)
        self.pitch_spin.setSingleStep(1.0)
        self.pitch_spin.setSuffix(" deg")
        self.pitch_spin.setValue(DEFAULT_PITCH_DEG)
        self.pitch_spin.valueChanged.connect(self._push_runtime_config)
        correction_form.addRow("Pitch angle", self.pitch_spin)

        self.soft_cluster_checkbox = QCheckBox("Sparse soft-cluster front-end")
        self.soft_cluster_checkbox.setChecked(True)
        self.soft_cluster_checkbox.stateChanged.connect(self._push_runtime_config)
        correction_form.addRow(self.soft_cluster_checkbox)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["Corrected world cloud", "Raw sensor cloud", "Local centered cloud"])
        self.view_combo.currentIndexChanged.connect(self._refresh_cloud)
        correction_form.addRow("View", self.view_combo)
        panel_layout.addWidget(correction_group)

        inference_group = QGroupBox("Pose inference")
        inference_layout = QGridLayout(inference_group)
        self.inference_checkbox = QCheckBox("Enable real-time model")
        self.inference_checkbox.setChecked(DEFAULT_MODEL_PATH.exists())
        self.inference_checkbox.stateChanged.connect(self._sync_model_controls)
        self.inference_checkbox.stateChanged.connect(self._push_runtime_config)
        self.model_path_edit = QLineEdit(str(DEFAULT_MODEL_PATH))
        self.model_path_edit.editingFinished.connect(self._push_runtime_config)
        self.model_browse_button = QPushButton("Browse")
        self.model_browse_button.clicked.connect(self.select_model_file)
        self.kalman_checkbox = QCheckBox("Enable Kalman pose smoothing")
        self.kalman_checkbox.setChecked(DEFAULT_MODEL_PATH.exists())
        self.kalman_checkbox.stateChanged.connect(self._sync_model_controls)
        self.kalman_checkbox.stateChanged.connect(self._push_runtime_config)
        self.kalman_process_noise_spin = QDoubleSpinBox()
        self.kalman_process_noise_spin.setRange(0.0001, 1.0)
        self.kalman_process_noise_spin.setDecimals(4)
        self.kalman_process_noise_spin.setSingleStep(0.001)
        self.kalman_process_noise_spin.setValue(DEFAULT_PROCESS_NOISE)
        self.kalman_process_noise_spin.setToolTip("Higher Q follows motion faster but allows more jitter.")
        self.kalman_process_noise_spin.valueChanged.connect(self._push_runtime_config)
        self.kalman_measurement_noise_spin = QDoubleSpinBox()
        self.kalman_measurement_noise_spin.setRange(0.0001, 10.0)
        self.kalman_measurement_noise_spin.setDecimals(4)
        self.kalman_measurement_noise_spin.setSingleStep(0.01)
        self.kalman_measurement_noise_spin.setValue(DEFAULT_MEASUREMENT_NOISE)
        self.kalman_measurement_noise_spin.setToolTip("Higher R smooths more strongly but adds more delay.")
        self.kalman_measurement_noise_spin.valueChanged.connect(self._push_runtime_config)
        inference_layout.addWidget(self.inference_checkbox, 0, 0, 1, 2)
        inference_layout.addWidget(self.model_path_edit, 1, 0)
        inference_layout.addWidget(self.model_browse_button, 1, 1)
        inference_layout.addWidget(self.kalman_checkbox, 2, 0, 1, 2)
        inference_layout.addWidget(QLabel("Process noise Q"), 3, 0)
        inference_layout.addWidget(self.kalman_process_noise_spin, 3, 1)
        inference_layout.addWidget(QLabel("Measurement noise R"), 4, 0)
        inference_layout.addWidget(self.kalman_measurement_noise_spin, 4, 1)
        panel_layout.addWidget(inference_group)

        recording_group = QGroupBox("Recording")
        recording_layout = QGridLayout(recording_group)
        self.save_checkbox = QCheckBox("Save raw frames")
        self.save_checkbox.stateChanged.connect(self._push_runtime_config)
        self.save_dir_edit = QLineEdit("radar_frames")
        self.save_dir_edit.editingFinished.connect(self._push_runtime_config)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.select_save_dir)
        recording_layout.addWidget(self.save_checkbox, 0, 0, 1, 2)
        recording_layout.addWidget(self.save_dir_edit, 1, 0)
        recording_layout.addWidget(self.browse_button, 1, 1)
        panel_layout.addWidget(recording_group)

        status_group = QGroupBox("Pipeline status")
        status_form = QFormLayout(status_group)
        self.state_label = QLabel("Idle")
        self.fps_label = QLabel("0.0")
        self.raw_count_label = QLabel("0")
        self.local_count_label = QLabel("0")
        self.body_count_label = QLabel("0 + 0")
        self.context_count_label = QLabel("0")
        self.tracker_label = QLabel("n/a")
        self.buffer_label = QLabel(f"0 / {SEQ_LEN}")
        self.center_label = QLabel("(0.00, 0.00, 0.00)")
        self.model_state_label = QLabel("Ready to load" if DEFAULT_MODEL_PATH.exists() else "Model missing")
        self.inference_time_label = QLabel("n/a")
        status_form.addRow("State", self.state_label)
        status_form.addRow("FPS", self.fps_label)
        status_form.addRow("Raw points", self.raw_count_label)
        status_form.addRow("Local points", self.local_count_label)
        status_form.addRow("Body core + halo", self.body_count_label)
        status_form.addRow("Context points", self.context_count_label)
        status_form.addRow("Tracker", self.tracker_label)
        status_form.addRow("Frame buffer", self.buffer_label)
        status_form.addRow("XY center", self.center_label)
        status_form.addRow("Model", self.model_state_label)
        status_form.addRow("Inference", self.inference_time_label)
        panel_layout.addWidget(status_group)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start collection")
        self.start_button.setObjectName("startButton")
        self.start_button.setProperty("collecting", False)
        self.start_button.clicked.connect(self.toggle_collection)
        self.clear_button = QPushButton("Clear view")
        self.clear_button.clicked.connect(self.clear_view)
        button_row.addWidget(self.start_button, 1)
        button_row.addWidget(self.clear_button)
        panel_layout.addLayout(button_row)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(130)
        panel_layout.addWidget(self.log_box)
        panel_layout.addStretch(1)

        self.connection_controls = [
            self.port_combo,
            self.refresh_button,
            self.baudrate_spin,
        ]
        self._sync_model_controls()

    def _apply_style(self, force=False):
        scale = min(self.width() / 1280.0, self.height() / 800.0)
        scale = min(max(scale, 0.78), 1.60)
        base_font = min(max(round(UI_BASE_FONT_PX * scale), 10), 20)
        title_font = min(max(round(UI_TITLE_FONT_PX * scale), 16), 30)
        panel_width = min(max(round(self.width() * 0.29), 320), 520)
        padding = min(max(round(4 * scale), 3), 7)
        button_v_padding = min(max(round(6 * scale), 4), 10)
        button_h_padding = min(max(round(10 * scale), 7), 16)
        signature = (
            self.theme_name,
            base_font,
            title_font,
            panel_width,
            padding,
            button_v_padding,
            button_h_padding,
        )
        if not force and signature == self._style_signature:
            return
        self._style_signature = signature

        colors = DISPLAY_THEMES[self.theme_name]
        self.control_panel.setFixedWidth(panel_width)
        self.log_box.setMinimumHeight(min(max(round(130 * scale), 105), 210))
        self.gl_view.setBackgroundColor(colors["viewport"])
        self.grid.setColor(colors["grid"])
        self.axis_legend.set_theme(colors)

        self.setStyleSheet(
            STYLE_TEMPLATE
            % {
                **colors,
                "base_font": base_font,
                "title_font": title_font,
                "padding": padding,
                "button_v_padding": button_v_padding,
                "button_h_padding": button_h_padding,
            }
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "control_panel"):
            self._apply_style()

    def toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.theme_button.setText("Dark theme" if self.theme_name == "light" else "Light theme")
        self._apply_style(force=True)
        self._refresh_cloud()
        self._refresh_pose()
        self.log(f"Switched to {self.theme_name} appearance")

    def _set_collection_button(self, collecting):
        self.start_button.setProperty("collecting", collecting)
        self.start_button.setText("Stop collection" if collecting else "Start collection")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self.start_button.update()

    def scan_ports(self):
        selected_port = self.port_combo.currentData() or DEFAULT_PORT
        self.port_combo.clear()
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}", port.device)
        if self.port_combo.findData(DEFAULT_PORT) < 0:
            self.port_combo.addItem(f"{DEFAULT_PORT} - default", DEFAULT_PORT)

        index = self.port_combo.findData(selected_port)
        if index >= 0:
            self.port_combo.setCurrentIndex(index)

        self.log(f"Found {len(ports)} serial port(s)")

    def select_save_dir(self):
        selected = QFileDialog.getExistingDirectory(self, "Select raw-frame directory", self.save_dir_edit.text())
        if selected:
            self.save_dir_edit.setText(selected)
            self._push_runtime_config()

    def select_model_file(self):
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select pose model",
            self.model_path_edit.text(),
            "PyTorch model (*.pth *.pt);;All files (*)",
        )
        if selected:
            self.model_path_edit.setText(selected)
            self._push_runtime_config()

    def _sync_model_controls(self, *_):
        model_enabled = self.inference_checkbox.isChecked()
        kalman_enabled = model_enabled and self.kalman_checkbox.isChecked()
        self.model_path_edit.setEnabled(model_enabled)
        self.model_browse_button.setEnabled(model_enabled)
        self.kalman_checkbox.setEnabled(model_enabled)
        self.kalman_process_noise_spin.setEnabled(kalman_enabled)
        self.kalman_measurement_noise_spin.setEnabled(kalman_enabled)

    def _runtime_config(self):
        model_path_text = self.model_path_edit.text().strip()
        save_dir_text = self.save_dir_edit.text().strip()
        return RuntimeConfig(
            radar_height=self.height_spin.value(),
            pitch_deg=self.pitch_spin.value(),
            soft_cluster_frontend=self.soft_cluster_checkbox.isChecked(),
            inference_enabled=self.inference_checkbox.isChecked(),
            model_path=Path(model_path_text).expanduser() if model_path_text else None,
            save_enabled=self.save_checkbox.isChecked(),
            save_dir=Path(save_dir_text).expanduser() if save_dir_text else None,
            kalman_enabled=self.kalman_checkbox.isChecked(),
            kalman_process_noise=self.kalman_process_noise_spin.value(),
            kalman_measurement_noise=self.kalman_measurement_noise_spin.value(),
        )

    def _push_runtime_config(self, *_):
        if self.worker is not None and self.worker.isRunning():
            self.worker.update_runtime_config(self._runtime_config())

    def toggle_collection(self):
        if self.worker is not None and self.worker.isRunning():
            self.stop_collection()
        else:
            self.start_collection()

    def start_collection(self):
        port = self.port_combo.currentData()
        if not port:
            QMessageBox.warning(self, "Missing serial port", "Select a serial port before starting collection.")
            return

        config = self._runtime_config()
        if config.save_enabled and config.save_dir is None:
            QMessageBox.warning(self, "Missing save directory", "Choose a directory for raw frames.")
            return

        if config.inference_enabled:
            if config.model_path is None:
                QMessageBox.warning(self, "Missing pose model", "Choose a model file before starting collection.")
                return
            if not config.model_path.exists():
                QMessageBox.warning(self, "Missing pose model", f"Model file does not exist:\n{config.model_path}")
                return

        self.received_frames = 0
        self.collection_started_at = time.perf_counter()
        self.latest_pose = None
        self._refresh_pose()
        self.buffer_label.setText(f"0 / {SEQ_LEN}")
        self.model_state_label.setText("Loading" if config.inference_enabled else "Disabled")
        self.inference_time_label.setText("n/a")
        self.worker = RadarWorker(
            port=port,
            baudrate=self.baudrate_spin.value(),
            config=config,
        )
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.message.connect(self.log)
        self.worker.failure.connect(self._on_failure)
        self.worker.model_status.connect(self._on_model_status)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

        self._set_settings_enabled(False)
        self._set_collection_button(True)
        self.state_label.setText("Starting")
        self.log(
            f"Starting {port}; correction height={self.height_spin.value():.2f} m, "
            f"pitch={self.pitch_spin.value():.1f} deg; "
            f"front-end={'sparse soft-cluster' if config.soft_cluster_frontend else 'legacy'}"
        )

    def stop_collection(self):
        if self.worker is None:
            return
        self.state_label.setText("Stopping")
        self.start_button.setEnabled(False)
        self.worker.stop()

    def _on_frame(self, update: FrameUpdate):
        self.latest_frame = update.frame
        self.received_frames += 1
        elapsed = time.perf_counter() - self.collection_started_at
        fps = self.received_frames / elapsed if elapsed > 0 else 0.0

        self.state_label.setText("Running")
        self.fps_label.setText(f"{fps:.1f}")
        self.raw_count_label.setText(str(len(update.frame.raw_points)))
        self.local_count_label.setText(str(len(update.frame.local_points)))
        self.body_count_label.setText(
            f"{update.frame.core_point_count} + {update.frame.halo_point_count}"
        )
        self.context_count_label.setText(str(update.frame.context_point_count))
        self.tracker_label.setText(update.frame.tracking_status)
        self.buffer_label.setText(f"{update.buffer_size} / {update.seq_len}")
        self.center_label.setText(
            f"({update.frame.center[0]:+.2f}, {update.frame.center[1]:+.2f}, {update.frame.center[2]:+.2f})"
        )
        if update.pose_world is not None:
            self.latest_pose = update.pose_world
            self.inference_time_label.setText(f"{update.inference_ms:.1f} ms")
        elif self.model_state_label.text() == "Active":
            self.latest_pose = None
            self.inference_time_label.setText(f"Buffering {update.buffer_size} / {update.seq_len}")
        self._refresh_cloud()
        self._refresh_pose()

    def _on_model_status(self, status):
        self.model_state_label.setText(status)
        if status != "Active":
            self.latest_pose = None
            self.inference_time_label.setText("n/a")
            self._refresh_pose()

    def _on_failure(self, message):
        self.state_label.setText("Error")
        if "model" in message.lower():
            self.model_state_label.setText("Error")
        self.log(message)

    def _on_worker_finished(self):
        self._set_settings_enabled(True)
        self.start_button.setEnabled(True)
        self._set_collection_button(False)
        if self.state_label.text() != "Error":
            self.state_label.setText("Idle")
        if self.model_state_label.text() != "Error":
            self.model_state_label.setText("Ready to load" if self.inference_checkbox.isChecked() else "Disabled")
        self.worker = None
        self._sync_model_controls()

    def _refresh_cloud(self):
        if self.latest_frame is None:
            return

        mode = self.view_combo.currentIndex()
        if mode == 1:
            points = self.latest_frame.raw_points[:, :3]
        elif mode == 2:
            points = self.latest_frame.local_points[:, :3]
        else:
            points = self.latest_frame.world_points[:, :3]

        if points.shape[0] == 0:
            self.cloud_outline_scatter.setData(
                pos=np.zeros((0, 3), dtype=np.float32),
                color=np.zeros((0, 4), dtype=np.float32),
            )
            self.scatter.setData(
                pos=np.zeros((0, 3), dtype=np.float32),
                color=np.zeros((0, 4), dtype=np.float32),
            )
            return

        theme = DISPLAY_THEMES[self.theme_name]
        outline_colors = np.tile(
            np.asarray(theme["cloud_outline"], dtype=np.float32),
            (points.shape[0], 1),
        )
        self.cloud_outline_scatter.setData(
            pos=points,
            color=outline_colors,
            size=theme["cloud_outline_size"],
        )
        self.scatter.setData(
            pos=points,
            color=self._point_colors(points),
            size=theme["cloud_size"],
        )

    def _point_colors(self, points):
        z = points[:, 2]
        span = max(float(np.max(z) - np.min(z)), 1e-6)
        level = ((z - np.min(z)) / span).reshape(-1, 1)
        theme = DISPLAY_THEMES[self.theme_name]
        low = np.asarray(theme["cloud_low"], dtype=np.float32)
        high = np.asarray(theme["cloud_high"], dtype=np.float32)
        colors = np.ones((points.shape[0], 4), dtype=np.float32)
        colors[:, :3] = low + (high - low) * level
        return colors

    def _refresh_pose(self):
        if self.latest_pose is None:
            self.joint_outline_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self.joint_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
            for line in self.skeleton_outline_lines:
                line.setData(pos=np.zeros((0, 3), dtype=np.float32))
            for line in self.skeleton_lines:
                line.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        theme = DISPLAY_THEMES[self.theme_name]
        self.joint_outline_scatter.setData(
            pos=self.latest_pose,
            color=theme["joint_outline"],
            size=theme["joint_outline_size"],
        )
        self.joint_scatter.setData(
            pos=self.latest_pose,
            color=theme["joint"],
            size=theme["joint_size"],
        )
        for index, (start, end) in enumerate(SKELETON_BONES):
            bone = np.asarray([self.latest_pose[start], self.latest_pose[end]])
            self.skeleton_outline_lines[index].setData(
                pos=bone,
                color=theme["bone_outline"],
                width=theme["bone_outline_width"],
            )
            self.skeleton_lines[index].setData(
                pos=bone,
                color=theme["bone"],
                width=theme["bone_width"],
            )

    def clear_view(self):
        self.latest_frame = None
        self.latest_pose = None
        self.cloud_outline_scatter.setData(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
        )
        self.scatter.setData(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
        )
        self._refresh_pose()

    def _set_settings_enabled(self, enabled):
        for control in self.connection_controls:
            control.setEnabled(enabled)
        if enabled:
            self._sync_model_controls()

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{timestamp}] {message}")

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(3000):
                self.log("Waiting for serial worker to stop before closing")
                event.ignore()
                return
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = RadarRealtimeWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
