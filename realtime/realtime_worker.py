"""Serial worker with frame-boundary runtime configuration updates."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock

import numpy as np
import serial
from PyQt5.QtCore import QThread, pyqtSignal

from pose_filter import (
    DEFAULT_MEASUREMENT_NOISE,
    DEFAULT_PROCESS_NOISE,
    MultiJointKalmanFilter,
)
from radar_pipeline import SEQ_LEN, RadarFrameProcessor, RadarSerialReader
from realtime_inference import RealtimePosePredictor


@dataclass(frozen=True)
class RuntimeConfig:
    radar_height: float
    pitch_deg: float
    soft_cluster_frontend: bool = True
    inference_enabled: bool = False
    model_path: Path | None = None
    save_enabled: bool = False
    save_dir: Path | None = None
    kalman_enabled: bool = False
    kalman_process_noise: float = DEFAULT_PROCESS_NOISE
    kalman_measurement_noise: float = DEFAULT_MEASUREMENT_NOISE


@dataclass
class FrameUpdate:
    frame: object
    buffer_size: int
    seq_len: int
    pose_world: np.ndarray | None
    inference_ms: float | None


class RadarWorker(QThread):
    frame_ready = pyqtSignal(object)
    message = pyqtSignal(str)
    failure = pyqtSignal(str)
    model_status = pyqtSignal(str)

    def __init__(self, port, baudrate, config):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.saved_frames = 0
        self._config = config
        self._config_version = 0
        self._config_lock = Lock()

    def update_runtime_config(self, config):
        """Queue a complete config snapshot for the next frame boundary."""
        with self._config_lock:
            self._config = config
            self._config_version += 1

    def _runtime_config(self):
        with self._config_lock:
            return self._config, self._config_version

    def run(self):
        reader = RadarSerialReader(self.port, self.baudrate)
        config, applied_version = self._runtime_config()
        processor = RadarFrameProcessor(
            config.radar_height,
            config.pitch_deg,
            use_soft_cluster=config.soft_cluster_frontend,
        )
        predictor = None
        kalman_filter = None
        applied_config = None

        try:
            processor, predictor, kalman_filter = self._apply_config(
                config, applied_config, processor, predictor, kalman_filter
            )
            applied_config = config

            reader.open()
            self.message.emit(f"Serial connected: {self.port} @ {self.baudrate}")
            reader.start_radar()
            self.message.emit("Radar start commands sent")

            while not self.isInterruptionRequested():
                config, version = self._runtime_config()
                if version != applied_version:
                    processor, predictor, kalman_filter = self._apply_config(
                        config, applied_config, processor, predictor, kalman_filter
                    )
                    applied_config = config
                    applied_version = version

                raw_points = reader.read_frame(should_stop=self.isInterruptionRequested)
                if raw_points is None:
                    break

                frame = processor.process(raw_points)
                if config.save_enabled and config.save_dir is not None:
                    self._save_raw_frame(config.save_dir, raw_points)

                pose_world = None
                inference_ms = None
                if predictor is not None:
                    model_window = processor.model_window()
                    if model_window is not None:
                        pose_world, inference_ms = predictor.predict(model_window, frame.center)
                        if kalman_filter is not None:
                            pose_world = kalman_filter.smooth(pose_world)

                self.frame_ready.emit(
                    FrameUpdate(
                        frame=frame,
                        buffer_size=len(processor.frame_buffer),
                        seq_len=processor.seq_len,
                        pose_world=pose_world,
                        inference_ms=inference_ms,
                    )
                )
        except serial.SerialException as exc:
            self.failure.emit(f"Serial failure: {exc}")
        except Exception as exc:
            self.failure.emit(f"Pipeline failure: {exc}")
        finally:
            reader.close()
            self.message.emit("Collection stopped")

    def _apply_config(self, config, previous, processor, predictor, kalman_filter):
        model_changed = (
            previous is None
            or config.inference_enabled != previous.inference_enabled
            or config.model_path != previous.model_path
        )
        correction_changed = (
            previous is None
            or config.radar_height != previous.radar_height
            or config.pitch_deg != previous.pitch_deg
            or config.soft_cluster_frontend != previous.soft_cluster_frontend
        )
        kalman_changed = (
            previous is None
            or config.kalman_enabled != previous.kalman_enabled
            or config.kalman_process_noise != previous.kalman_process_noise
            or config.kalman_measurement_noise != previous.kalman_measurement_noise
            or model_changed
        )

        if model_changed:
            predictor = None
            if config.inference_enabled:
                self.model_status.emit("Loading")
                self.message.emit(f"Loading pose model: {config.model_path}")
                try:
                    predictor = RealtimePosePredictor(config.model_path)
                except Exception as exc:
                    self.model_status.emit("Error")
                    self.message.emit(f"Model failure: {exc}")
                else:
                    self.model_status.emit("Active")
                    self.message.emit(f"Model ready: {predictor.description()}")
            else:
                self.model_status.emit("Disabled")
                self.message.emit("Real-time inference disabled")

        if model_changed or correction_changed:
            seq_len = predictor.seq_len if predictor is not None else SEQ_LEN
            input_channels = predictor.input_channels if predictor is not None else 5
            processor = RadarFrameProcessor(
                config.radar_height,
                config.pitch_deg,
                seq_len=seq_len,
                input_channels=input_channels,
                use_soft_cluster=config.soft_cluster_frontend,
            )
            frontend_name = "sparse soft-cluster" if config.soft_cluster_frontend else "legacy"
            self.message.emit(
                f"Applied correction: height={config.radar_height:.2f} m, "
                f"pitch={config.pitch_deg:.1f} deg; front-end={frontend_name}; "
                "frame buffer reset"
            )

        if kalman_changed:
            if config.kalman_enabled and predictor is not None:
                kalman_filter = MultiJointKalmanFilter(
                    predictor.num_joints,
                    process_noise=config.kalman_process_noise,
                    measurement_noise=config.kalman_measurement_noise,
                )
                self.message.emit(
                    "Kalman pose smoothing enabled: "
                    f"Q={config.kalman_process_noise:.4f}, "
                    f"R={config.kalman_measurement_noise:.4f}"
                )
            else:
                kalman_filter = None
                if previous is not None and previous.kalman_enabled:
                    self.message.emit("Kalman pose smoothing disabled")

        if config.save_enabled and config.save_dir is not None:
            config.save_dir.mkdir(parents=True, exist_ok=True)

        return processor, predictor, kalman_filter

    def stop(self):
        self.requestInterruption()

    def _save_raw_frame(self, save_dir, raw_points):
        self.saved_frames += 1
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
        filename = f"{timestamp}_{self.saved_frames:06d}.npy"
        np.save(save_dir / filename, raw_points)
