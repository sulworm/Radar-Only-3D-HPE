"""Adapter for the MM-Fi radar-only benchmark subset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ALL_SUBJECTS = [f"S{idx:02d}" for idx in range(1, 41)]
ALL_ACTIONS = [f"A{idx:02d}" for idx in range(1, 28)]
ALL_SCENES = ["E01", "E02", "E03", "E04"]

DAILY_ACTIONS = [
    "A02",
    "A03",
    "A04",
    "A05",
    "A13",
    "A14",
    "A17",
    "A18",
    "A19",
    "A20",
    "A21",
    "A22",
    "A23",
    "A27",
]

REHAB_ACTIONS = [
    "A01",
    "A06",
    "A07",
    "A08",
    "A09",
    "A10",
    "A11",
    "A12",
    "A15",
    "A16",
    "A24",
    "A25",
    "A26",
]

S2_TEST_SUBJECTS = ["S05", "S10", "S15", "S20", "S25", "S30", "S35", "S40"]


@dataclass(frozen=True)
class MMFiSequence:
    scene: str
    subject: str
    action: str
    root: Path

    @property
    def sequence_dir(self) -> Path:
        return self.root / self.scene / self.subject / self.action

    @property
    def ground_truth_path(self) -> Path:
        return self.sequence_dir / "ground_truth.npy"

    @property
    def mmwave_dir(self) -> Path:
        return self.sequence_dir / "mmwave"

    def frame_path(self, frame_index: int) -> Path:
        return self.mmwave_dir / f"frame{frame_index:03d}.bin"


def default_root() -> Path:
    return Path(__file__).resolve().parents[2] / ".benchmark" / "MM-Fi" / "MMFI_mmwave"


def normalize_protocol(protocol: str) -> str:
    protocol = protocol.lower().replace("_", "").replace("-", "")
    aliases = {
        "1": "p1",
        "p1": "p1",
        "protocol1": "p1",
        "daily": "p1",
        "2": "p2",
        "p2": "p2",
        "protocol2": "p2",
        "rehab": "p2",
        "rehabilitation": "p2",
        "3": "p3",
        "p3": "p3",
        "protocol3": "p3",
        "all": "p3",
    }
    if protocol not in aliases:
        raise ValueError(f"Unsupported MM-Fi protocol: {protocol!r}")
    return aliases[protocol]


def protocol_actions(protocol: str) -> list[str]:
    protocol = normalize_protocol(protocol)
    if protocol == "p1":
        return list(DAILY_ACTIONS)
    if protocol == "p2":
        return list(REHAB_ACTIONS)
    return list(ALL_ACTIONS)


def scene_for_subject(subject: str) -> str:
    subject_index = int(subject[1:])
    if 1 <= subject_index <= 10:
        return "E01"
    if 11 <= subject_index <= 20:
        return "E02"
    if 21 <= subject_index <= 30:
        return "E03"
    if 31 <= subject_index <= 40:
        return "E04"
    raise ValueError(f"Unknown MM-Fi subject: {subject}")


def subjects_for_scene(scene: str) -> list[str]:
    if scene not in ALL_SCENES:
        raise ValueError(f"Unknown MM-Fi scene: {scene}")
    scene_index = int(scene[1:])
    start = (scene_index - 1) * 10 + 1
    return [f"S{idx:02d}" for idx in range(start, start + 10)]


def read_ground_truth(path_or_sequence) -> np.ndarray:
    path = path_or_sequence.ground_truth_path if isinstance(path_or_sequence, MMFiSequence) else Path(path_or_sequence)
    pose = np.load(path)
    if pose.shape != (297, 17, 3):
        raise ValueError(f"MM-Fi ground truth must have shape (297, 17, 3), got {pose.shape}: {path}")
    return pose.astype(np.float32)


def read_mmwave_frame(path_or_sequence, frame_index: int | None = None) -> np.ndarray:
    if isinstance(path_or_sequence, MMFiSequence):
        if frame_index is None:
            raise ValueError("frame_index is required when reading from a sequence")
        path = path_or_sequence.frame_path(frame_index)
    else:
        path = Path(path_or_sequence)
    frame = np.fromfile(path, dtype=np.float64)
    if frame.size % 5 != 0:
        raise ValueError(f"MM-Fi mmWave frame size is not divisible by 5: {path}")
    return frame.reshape(-1, 5).astype(np.float32)


def _s1_subjects_by_action(actions: list[str], split: str, random_seed: int, random_ratio: float) -> dict[str, set[str]]:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if not 0.0 < random_ratio < 1.0:
        raise ValueError(f"random_ratio must be between 0 and 1, got {random_ratio}")

    action_subjects: dict[str, set[str]] = {}
    seed = random_seed
    subjects = np.array(ALL_SUBJECTS)
    for action in actions:
        rng = np.random.RandomState(seed)
        shuffled = subjects[rng.permutation(len(subjects))]
        cutoff = int(np.floor(random_ratio * len(subjects)))
        selected = shuffled[:cutoff] if split == "train" else shuffled[cutoff:]
        action_subjects[action] = set(selected.tolist())
        seed += 1
    return action_subjects


def split_sequences(
    root=None,
    protocol: str = "p3",
    setting: str = "s2",
    split: str = "test",
    random_seed: int = 0,
    random_ratio: float = 0.8,
    test_scene: str = "E04",
) -> list[MMFiSequence]:
    """Return MM-Fi sequences for an official benchmark setting.

    `setting='s1'` follows the official toolbox style random subject split per action.
    The local toolbox config uses `random_ratio=0.8`; the paper text describes a 3:1 split,
    which can be reproduced by passing `random_ratio=0.75`.
    """
    root = Path(root) if root is not None else default_root()
    setting = setting.lower()
    split = split.lower()
    if setting not in {"s1", "s2", "s3"}:
        raise ValueError(f"Unsupported MM-Fi setting: {setting!r}")
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    actions = protocol_actions(protocol)
    sequences: list[MMFiSequence] = []

    if setting == "s1":
        by_action = _s1_subjects_by_action(actions, split, random_seed, random_ratio)
        for action in actions:
            for subject in ALL_SUBJECTS:
                if subject not in by_action[action]:
                    continue
                scene = scene_for_subject(subject)
                sequences.append(MMFiSequence(scene=scene, subject=subject, action=action, root=root))
        return sequences

    if setting == "s2":
        test_subjects = set(S2_TEST_SUBJECTS)
        selected_subjects = test_subjects if split == "test" else set(ALL_SUBJECTS) - test_subjects
        for subject in ALL_SUBJECTS:
            if subject not in selected_subjects:
                continue
            scene = scene_for_subject(subject)
            for action in actions:
                sequences.append(MMFiSequence(scene=scene, subject=subject, action=action, root=root))
        return sequences

    if test_scene not in ALL_SCENES:
        raise ValueError(f"Unknown test_scene: {test_scene}")
    selected_scenes = {test_scene} if split == "test" else set(ALL_SCENES) - {test_scene}
    for scene in ALL_SCENES:
        if scene not in selected_scenes:
            continue
        for subject in subjects_for_scene(scene):
            for action in actions:
                sequences.append(MMFiSequence(scene=scene, subject=subject, action=action, root=root))
    return sequences


def validate_sequences(sequences: list[MMFiSequence], check_frames: bool = False) -> dict[str, int]:
    missing_gt = 0
    missing_mmwave = 0
    missing_frames = 0
    frame_count = 0
    for sequence in sequences:
        if not sequence.ground_truth_path.exists():
            missing_gt += 1
        if not sequence.mmwave_dir.is_dir():
            missing_mmwave += 1
            continue
        if check_frames:
            for frame_index in range(1, 298):
                frame_count += 1
                if not sequence.frame_path(frame_index).exists():
                    missing_frames += 1
    return {
        "sequences": len(sequences),
        "frames": len(sequences) * 297,
        "checked_frames": frame_count,
        "missing_ground_truth": missing_gt,
        "missing_mmwave_dir": missing_mmwave,
        "missing_frames": missing_frames,
    }


def describe(root=None) -> dict[str, object]:
    root = Path(root) if root is not None else default_root()
    scenes = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("E"))
    gt_count = sum(1 for _ in root.rglob("ground_truth.npy"))
    bin_count = sum(1 for _ in root.rglob("*.bin"))
    return {
        "root": str(root),
        "scenes": scenes,
        "ground_truth_files": gt_count,
        "mmwave_bin_files": bin_count,
    }


def streaming_mean_pose(sequences: list[MMFiSequence]) -> np.ndarray:
    total = np.zeros((17, 3), dtype=np.float64)
    count = 0
    for sequence in sequences:
        pose = read_ground_truth(sequence)
        total += pose.sum(axis=0)
        count += pose.shape[0]
    if count == 0:
        raise ValueError("No frames available for mean pose")
    return (total / count).astype(np.float32)
