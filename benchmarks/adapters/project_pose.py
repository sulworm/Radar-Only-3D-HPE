"""Adapter for running this project's trained pose models on public data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import (  # noqa: E402
    build_padded_windows,
    load_model_config,
    preprocess_radar_frame,
    radar_channel_count,
    safe_torch_load,
)
from model import build_model  # noqa: E402


DEFAULT_EPOCH40_MODEL = PROJECT_ROOT / "model" / "Training_Results_GeoSTAR_fulldata_40" / "best_model.pth"


@dataclass(frozen=True)
class ProjectModelBundle:
    model: torch.nn.Module
    config: dict[str, object]
    device: torch.device
    model_path: Path
    model_type: str
    radar_channels: str
    input_channels: int
    seq_len: int
    num_joints: int


def load_project_model(
    model_path: str | Path = DEFAULT_EPOCH40_MODEL,
    model_config: str | Path | None = None,
    model_type: str = "auto",
    device: str | torch.device | None = None,
) -> ProjectModelBundle:
    """Load a trained project model and its saved config."""
    model_path = Path(model_path)
    config_path = Path(model_config) if model_config is not None else None
    config = load_model_config(str(model_path), str(config_path) if config_path is not None else None)
    resolved_model_type = str(config.get("model_type", "baseline") if model_type == "auto" else model_type)
    radar_channels = str(config.get("radar_channels", "xyzvsnr"))
    input_channels = int(config.get("input_channels", radar_channel_count(radar_channels)))
    seq_len = int(config.get("seq_len", 10))
    num_joints = int(config.get("num_joints", 13))

    resolved_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_type=resolved_model_type,
        input_channels=input_channels,
        num_joints=num_joints,
        seq_len=seq_len,
    ).to(resolved_device)
    model.load_state_dict(safe_torch_load(str(model_path), map_location=resolved_device))
    model.eval()

    return ProjectModelBundle(
        model=model,
        config=config,
        device=resolved_device,
        model_path=model_path,
        model_type=resolved_model_type,
        radar_channels=radar_channels,
        input_channels=input_channels,
        seq_len=seq_len,
        num_joints=num_joints,
    )


def preprocess_radar_sequence(
    radar_frames,
    radar_channels: str,
    seq_len: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw per-frame radar arrays into model input windows and centers."""
    rng = np.random.default_rng(seed)
    feature_frames = []
    centers = []
    prev_center = None

    for radar_data in radar_frames:
        radar_features, _radar_local_xyz, center = preprocess_radar_frame(
            np.asarray(radar_data, dtype=np.float32),
            prev_center=prev_center,
            rng=rng,
            radar_channels=radar_channels,
        )
        feature_frames.append(radar_features)
        centers.append(center)
        prev_center = center

    feature_frames = np.asarray(feature_frames, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    input_windows = build_padded_windows(feature_frames, seq_len=seq_len)
    return input_windows, centers


def predict_local_sequence(
    bundle: ProjectModelBundle,
    input_windows: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    """Run one sequence and return one local pose prediction per frame."""
    if input_windows.ndim != 4:
        raise ValueError(f"input_windows must have shape (T, seq_len, points, channels), got {input_windows.shape}")
    if input_windows.shape[0] == 0:
        return np.zeros((0, bundle.num_joints, 3), dtype=np.float32)

    input_tensor = torch.from_numpy(input_windows).float().permute(0, 1, 3, 2)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(input_tensor), batch_size):
            batch = input_tensor[start : start + batch_size].to(bundle.device)
            output = bundle.model(batch)
            predictions.append(output[:, -1, :, :].cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)
