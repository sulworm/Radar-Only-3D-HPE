"""Real-time pose-model loading and single-window inference."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from model import build_model


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "Models" / "Training_Results_GeoSTAR_yaw30" / "best_model.pth"


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class RealtimePosePredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, device=None):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")

        self.config_path = self.model_path.with_name("best_model_config.json")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config does not exist: {self.config_path}")

        with self.config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)

        self.model_type = str(config["model_type"])
        self.input_channels = int(config["input_channels"])
        self.seq_len = int(config["seq_len"])
        self.num_joints = int(config.get("num_joints", 13))
        self.num_points = int(config.get("num_points", 128))
        self.radar_channels = str(config.get("radar_channels", "xyzvsnr"))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = build_model(
            model_type=self.model_type,
            input_channels=self.input_channels,
            num_joints=self.num_joints,
            seq_len=self.seq_len,
        )
        state_dict = safe_torch_load(self.model_path, map_location=self.device)
        if isinstance(state_dict, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._warm_up()

    def _warm_up(self):
        """Pay one-time framework and CUDA initialization cost before streaming starts."""
        sample = torch.zeros(
            (1, self.seq_len, self.input_channels, self.num_points),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            self.model(sample)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict(self, model_window, center):
        """Predict the latest world-space pose from a local radar window."""
        model_window = np.asarray(model_window, dtype=np.float32)
        if model_window.ndim != 3:
            raise ValueError(f"Expected a 3D radar window, got {model_window.shape}")
        expected_shape = (self.seq_len, model_window.shape[1], self.input_channels)
        if model_window.shape != expected_shape:
            raise ValueError(
                f"Expected radar window (seq_len, points, channels)={expected_shape}, "
                f"got {model_window.shape}"
            )

        tensor = torch.from_numpy(model_window).permute(0, 2, 1).unsqueeze(0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started_at = time.perf_counter()
        with torch.no_grad():
            output = self.model(tensor.to(self.device))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started_at) * 1000.0

        pose_local = output[0, -1].detach().cpu().numpy().astype(np.float32)
        pose_world = pose_local + np.asarray(center, dtype=np.float32).reshape(1, 3)
        return pose_world, inference_ms

    def description(self):
        return (
            f"{self.model_type} | {self.radar_channels} ({self.input_channels} ch) | "
            f"seq={self.seq_len} | joints={self.num_joints} | {self.device}"
        )
