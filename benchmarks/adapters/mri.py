"""Adapter for the mRI mmWave pose benchmark dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


SUBJECTS = [f"subject{idx}" for idx in range(1, 21)]
PROTOCOL1_ACTIONS = [f"pose_{idx}" for idx in range(1, 11)] + ["free_form", "walk"]
PROTOCOL2_ACTIONS = [f"pose_{idx}" for idx in range(1, 11)]
OFFICIAL_RESULT_FILES = {
    ("random", "p1"): "random_split_protocol1.npy",
    ("random", "p2"): "random_split_protocol2.npy",
    ("subject", "p1"): "subject_split_protocol1.npy",
    ("subject", "p2"): "subject_split_protocol2.npy",
}


@dataclass(frozen=True)
class MRISubject:
    subject: str
    root: Path

    @property
    def index(self) -> int:
        return int(self.subject.replace("subject", ""))

    @property
    def pose_label_path(self) -> Path:
        return self.root / "aligned_data" / "pose_labels" / f"{self.subject}_all_labels.cpl"

    @property
    def radar_singleframe_path(self) -> Path:
        return self.root / "aligned_data" / "radar" / "singleframe" / f"{self.subject}.csv"

    @property
    def radar_featuremap_path(self) -> Path:
        return self.root / "features" / "radar" / f"{self.subject}_featuremap.npy"


def default_root() -> Path:
    return Path(__file__).resolve().parents[2] / ".benchmark" / "mRI" / "dataset_release"


def normalize_protocol(protocol: str) -> str:
    protocol = protocol.lower().replace("_", "").replace("-", "")
    aliases = {
        "1": "p1",
        "p1": "p1",
        "protocol1": "p1",
        "all": "p1",
        "2": "p2",
        "p2": "p2",
        "protocol2": "p2",
        "rehab": "p2",
    }
    if protocol not in aliases:
        raise ValueError(f"Unsupported mRI protocol: {protocol!r}")
    return aliases[protocol]


def normalize_split(split: str) -> str:
    split = split.lower().replace("_", "").replace("-", "")
    aliases = {
        "random": "random",
        "randomsplit": "random",
        "s1": "random",
        "subject": "subject",
        "subjectsplit": "subject",
        "s2": "subject",
    }
    if split not in aliases:
        raise ValueError(f"Unsupported mRI split: {split!r}")
    return aliases[split]


def protocol_actions(protocol: str) -> list[str]:
    protocol = normalize_protocol(protocol)
    if protocol == "p1":
        return list(PROTOCOL1_ACTIONS)
    return list(PROTOCOL2_ACTIONS)


def list_subjects(root=None) -> list[MRISubject]:
    root = Path(root) if root is not None else default_root()
    subjects = []
    for subject in SUBJECTS:
        item = MRISubject(subject=subject, root=root)
        if item.pose_label_path.exists():
            subjects.append(item)
    return subjects


def subject_item(subject: str | int | MRISubject, root=None) -> MRISubject:
    if isinstance(subject, MRISubject):
        return subject
    if isinstance(subject, int):
        subject = f"subject{subject}"
    subject = str(subject)
    if subject.isdigit():
        subject = f"subject{subject}"
    return MRISubject(subject=subject, root=Path(root) if root is not None else default_root())


def read_pose_label_dict(subject: str | int | MRISubject, root=None) -> dict[str, object]:
    item = subject_item(subject, root=root)
    with item.pose_label_path.open("rb") as handle:
        return pickle.load(handle)


def read_pose(
    subject: str | int | MRISubject,
    root=None,
    label_key: str = "refined_gt_kps",
    aligned_to_radar: bool = False,
) -> np.ndarray:
    """Read pose labels as [T, 17, 3] in the dataset coordinate unit."""
    data = read_pose_label_dict(subject, root=root)
    pose = np.asarray(data[label_key], dtype=np.float32)
    if pose.ndim != 3 or pose.shape[1:] != (3, 17):
        raise ValueError(f"Expected {label_key} shape (T, 3, 17), got {pose.shape}")
    pose = pose.transpose(0, 2, 1).astype(np.float32)
    if not aligned_to_radar:
        return pose
    start, end = radar_frame_range(data)
    return pose[start : end + 1]


def read_radar_featuremap(subject: str | int | MRISubject, root=None) -> np.ndarray:
    item = subject_item(subject, root=root)
    return np.load(item.radar_featuremap_path).astype(np.float32)


def radar_frame_range(label_data_or_subject, root=None) -> tuple[int, int]:
    if isinstance(label_data_or_subject, dict):
        data = label_data_or_subject
    else:
        data = read_pose_label_dict(label_data_or_subject, root=root)
    frames = list(data["radar_avail_frames"])
    if len(frames) != 2:
        raise ValueError(f"Expected radar_avail_frames to contain [start, end], got {frames}")
    return int(frames[0]), int(frames[1])


def action_intervals(subject: str | int | MRISubject, root=None) -> dict[str, tuple[int, int]]:
    data = read_pose_label_dict(subject, root=root)
    return {name: (int(bounds[0]), int(bounds[1])) for name, bounds in data["video_label"].items()}


def protocol_frame_indices(subject: str | int | MRISubject, protocol: str = "p1", root=None) -> np.ndarray:
    """Return indices relative to the radar-aligned feature sequence."""
    data = read_pose_label_dict(subject, root=root)
    radar_start, radar_end = radar_frame_range(data)
    aligned_length = radar_end - radar_start + 1
    mask = np.zeros(aligned_length, dtype=bool)
    intervals = {name: (int(bounds[0]), int(bounds[1])) for name, bounds in data["video_label"].items()}
    for action in protocol_actions(protocol):
        if action not in intervals:
            continue
        start, end = intervals[action]
        start = max(start, radar_start)
        end = min(end, radar_end)
        if end >= start:
            mask[start - radar_start : end - radar_start + 1] = True
    return np.flatnonzero(mask)


def aligned_feature_pose(
    subject: str | int | MRISubject,
    root=None,
    protocol: str | None = None,
    label_key: str = "refined_gt_kps",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return radar feature maps, pose labels, and original camera frame numbers."""
    item = subject_item(subject, root=root)
    features = read_radar_featuremap(item)
    pose = read_pose(item, label_key=label_key, aligned_to_radar=True)
    frame_start, frame_end = radar_frame_range(item)
    frames = np.arange(frame_start, frame_end + 1, dtype=np.int32)

    length = min(features.shape[0], pose.shape[0], frames.shape[0])
    features = features[:length]
    pose = pose[:length]
    frames = frames[:length]

    if protocol is not None:
        indices = protocol_frame_indices(item, protocol=protocol)
        indices = indices[indices < length]
        features = features[indices]
        pose = pose[indices]
        frames = frames[indices]

    return features.astype(np.float32), pose.astype(np.float32), frames


def iter_radar_point_frames(subject: str | int | MRISubject, root=None):
    """Yield raw radar point frames as arrays with columns x,y,z,doppler,intensity."""
    item = subject_item(subject, root=root)
    current_frame = None
    points = []
    with item.radar_singleframe_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame = int(row["Frame #"])
            point = [
                float(row["X"]),
                float(row["Y"]),
                float(row["Z"]),
                float(row["Doppler"]),
                float(row["Intensity"]),
            ]
            if current_frame is None:
                current_frame = frame
            if frame != current_frame:
                yield current_frame, np.asarray(points, dtype=np.float32)
                current_frame = frame
                points = []
            points.append(point)
    if current_frame is not None:
        yield current_frame, np.asarray(points, dtype=np.float32)


def describe(root=None) -> dict[str, object]:
    root = Path(root) if root is not None else default_root()
    subjects = list_subjects(root)
    return {
        "root": str(root),
        "subjects": len(subjects),
        "pose_label_files": sum(1 for _ in (root / "aligned_data" / "pose_labels").glob("*_all_labels.cpl")),
        "radar_featuremap_files": sum(1 for _ in (root / "features" / "radar").glob("*_featuremap.npy")),
        "official_result_files": sum(1 for _ in (root / "model" / "mmWave" / "results").glob("*.npy")),
    }


def official_result_path(root=None, split: str = "random", protocol: str = "p1") -> Path:
    root = Path(root) if root is not None else default_root()
    key = (normalize_split(split), normalize_protocol(protocol))
    return root / "model" / "mmWave" / "results" / OFFICIAL_RESULT_FILES[key]


def read_official_results(root=None, split: str = "random", protocol: str = "p1") -> np.ndarray:
    path = official_result_path(root=root, split=split, protocol=protocol)
    result = np.load(path).astype(np.float64)
    if result.shape != (5, 2):
        raise ValueError(f"Expected official result shape (5, 2), got {result.shape}: {path}")
    return result


def summarize_official_results(result: np.ndarray) -> dict[str, object]:
    """Decode mRI official result array: 3 runs, mean row, and std row."""
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (5, 2):
        raise ValueError(f"Expected result shape (5, 2), got {result.shape}")
    runs = result[:3]
    official_mean = result[3]
    official_std = result[4]
    recomputed_mean = runs.mean(axis=0)
    # The released files compute std after appending the mean row to the three runs.
    recomputed_std = np.vstack([runs, recomputed_mean]).std(axis=0)
    return {
        "runs": runs.tolist(),
        "mean": official_mean.tolist(),
        "std": official_std.tolist(),
        "recomputed_mean": recomputed_mean.tolist(),
        "recomputed_std": recomputed_std.tolist(),
        "mean_abs_diff": np.abs(official_mean - recomputed_mean).tolist(),
        "std_abs_diff": np.abs(official_std - recomputed_std).tolist(),
        "columns": ["mpjpe_mm", "pa_mpjpe_mm"],
    }


def all_official_summaries(root=None) -> dict[str, dict[str, object]]:
    summaries = {}
    for split in ("random", "subject"):
        for protocol in ("p1", "p2"):
            result = read_official_results(root=root, split=split, protocol=protocol)
            summaries[f"{split}_{protocol}"] = summarize_official_results(result)
    return summaries
