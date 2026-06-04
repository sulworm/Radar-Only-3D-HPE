"""Adapter for the MARS public mmWave pose dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


MARS_JOINT_NAMES = [
    "SpineBase",
    "SpineMid",
    "Neck",
    "Head",
    "SpineShoulder",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HipLeft",
    "KneeLeft",
    "AnkleLeft",
    "FootLeft",
    "HipRight",
    "KneeRight",
    "AnkleRight",
    "FootRight",
]

MARS_SPLITS = ("train", "validate", "test")


@dataclass(frozen=True)
class MarsSplit:
    name: str
    featuremap: np.ndarray
    labels: np.ndarray
    pose: np.ndarray


def default_root() -> Path:
    return Path(__file__).resolve().parents[2] / ".benchmark" / "MARS" / "MARS-main"


def mars_label_to_pose(labels) -> np.ndarray:
    """Convert MARS labels from x[19]+y[19]+z[19] to [frames, 19, 3]."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 57:
        raise ValueError(f"MARS labels must have shape [frames, 57], got {labels.shape}")
    return np.stack([labels[:, 0:19], labels[:, 19:38], labels[:, 38:57]], axis=-1).astype(np.float32)


def pose_to_mars_label(pose) -> np.ndarray:
    pose = np.asarray(pose)
    if pose.ndim != 3 or pose.shape[1:] != (19, 3):
        raise ValueError(f"MARS pose must have shape [frames, 19, 3], got {pose.shape}")
    return np.concatenate([pose[:, :, 0], pose[:, :, 1], pose[:, :, 2]], axis=1)


def featuremap_to_points(featuremap) -> np.ndarray:
    """Flatten MARS [frames, 8, 8, 5] feature maps to [frames, 64, 5]."""
    featuremap = np.asarray(featuremap)
    if featuremap.ndim != 4 or featuremap.shape[1:] != (8, 8, 5):
        raise ValueError(f"MARS featuremap must have shape [frames, 8, 8, 5], got {featuremap.shape}")
    return featuremap.reshape(featuremap.shape[0], 64, 5)


def load_split(root=None, split: str = "train", mmap: bool = False) -> MarsSplit:
    if split not in MARS_SPLITS:
        raise ValueError(f"Unsupported MARS split {split!r}; expected one of {MARS_SPLITS}")
    root = Path(root) if root is not None else default_root()
    feature_path = root / "feature" / f"featuremap_{split}.npy"
    label_path = root / "feature" / f"labels_{split}.npy"
    mmap_mode = "r" if mmap else None
    featuremap = np.load(feature_path, mmap_mode=mmap_mode)
    labels = np.load(label_path, mmap_mode=mmap_mode)
    pose = mars_label_to_pose(labels)
    return MarsSplit(name=split, featuremap=featuremap, labels=labels, pose=pose)


def load_splits(root=None, splits=MARS_SPLITS, mmap: bool = False) -> dict[str, MarsSplit]:
    return {split: load_split(root=root, split=split, mmap=mmap) for split in splits}


def describe(root=None) -> list[dict[str, object]]:
    rows = []
    for split_name, split in load_splits(root=root, mmap=True).items():
        rows.append(
            {
                "split": split_name,
                "featuremap_shape": tuple(split.featuremap.shape),
                "labels_shape": tuple(split.labels.shape),
                "pose_shape": tuple(split.pose.shape),
                "pose_min": float(np.min(split.pose)),
                "pose_max": float(np.max(split.pose)),
            }
        )
    return rows

