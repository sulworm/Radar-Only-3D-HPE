"""Evaluate the current project model on MM-Fi radar-only data.

The default mapping is a pragmatic zero-shot bridge from MM-Fi's 17-joint
H36M-style layout to this project's 13-joint output. Treat the resulting
numbers as a current-model diagnostic, not as an official MM-Fi paper result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.adapters.mmfi import (
    read_ground_truth,
    read_mmwave_frame,
    split_sequences,
    validate_sequences,
)
from benchmarks.adapters.project_pose import (
    DEFAULT_EPOCH40_MODEL,
    load_project_model,
    predict_local_sequence,
    preprocess_radar_sequence,
)
from benchmarks.metrics import mpjpe, pa_mpjpe, pck, root_relative, root_relative_mpjpe


PROJECT13_JOINT_NAMES = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "neck",
    "head",
    "right_shoulder",
    "right_wrist",
    "left_shoulder",
    "left_wrist",
]

H36M17_TO_PROJECT13 = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 16, 11, 13]


def mmfi_camera_to_project_frame(pose: np.ndarray) -> np.ndarray:
    """Map MM-Fi camera coordinates to the project's distance/lateral/height axes."""
    pose = np.asarray(pose, dtype=np.float32)
    return np.stack([pose[..., 2], pose[..., 0], -pose[..., 1]], axis=-1).astype(np.float32)


def parse_mapping(name: str, prediction_joints: int, gt_joints: int) -> list[int]:
    if name == "h36m17_project13":
        if prediction_joints != 13 or gt_joints != 17:
            raise ValueError(
                "h36m17_project13 requires model output 13 joints and MM-Fi GT 17 joints, "
                f"got {prediction_joints} and {gt_joints}"
            )
        return list(H36M17_TO_PROJECT13)
    if name == "identity":
        if prediction_joints != gt_joints:
            raise ValueError(f"identity mapping requires equal joint counts, got {prediction_joints} and {gt_joints}")
        return list(range(gt_joints))
    raise ValueError(f"Unsupported joint mapping: {name}")


def summarize(pred_local: np.ndarray, gt_project: np.ndarray) -> dict[str, object]:
    pred_rr = root_relative(pred_local, root_index=0)
    gt_rr = root_relative(gt_project, root_index=0)
    pck_values = pck(pred_rr, gt_rr, thresholds=(0.05, 0.10, 0.15))
    return {
        "frames": int(pred_local.shape[0]),
        "joints": int(pred_local.shape[1]),
        "root_relative_mpjpe_mm": root_relative_mpjpe(pred_local, gt_project, root_index=0) * 1000.0,
        "pa_mpjpe_mm": pa_mpjpe(pred_local, gt_project, scale=True) * 1000.0,
        "root_relative_pck@50mm": pck_values["pck@0.05"],
        "root_relative_pck@100mm": pck_values["pck@0.1"],
        "root_relative_pck@150mm": pck_values["pck@0.15"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Path to MMFI_mmwave. Defaults to .benchmark/MM-Fi/MMFI_mmwave.")
    parser.add_argument("--model_path", default=str(DEFAULT_EPOCH40_MODEL))
    parser.add_argument("--model_config", default=None)
    parser.add_argument("--model_type", default="auto")
    parser.add_argument("--protocol", default="p2", help="p1, p2, or p3.")
    parser.add_argument("--setting", default="s2", choices=("s1", "s2", "s3"))
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--random_ratio", type=float, default=0.8)
    parser.add_argument("--test_scene", default="E04")
    parser.add_argument("--joint_mapping", default="h36m17_project13", choices=("h36m17_project13", "identity"))
    parser.add_argument("--coordinate_mode", default="mmfi_camera_to_project", choices=("mmfi_camera_to_project", "identity"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    bundle = load_project_model(
        model_path=args.model_path,
        model_config=args.model_config,
        model_type=args.model_type,
        device=args.device,
    )
    sequences = split_sequences(
        root=args.root,
        protocol=args.protocol,
        setting=args.setting,
        split=args.split,
        random_seed=args.random_seed,
        random_ratio=args.random_ratio,
        test_scene=args.test_scene,
    )
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]

    validation = validate_sequences(sequences, check_frames=False)
    mapping = parse_mapping(args.joint_mapping, prediction_joints=bundle.num_joints, gt_joints=17)

    pred_blocks = []
    gt_blocks = []
    sequence_rows = []

    print(
        f"Model={bundle.model_type} path={bundle.model_path} device={bundle.device} "
        f"seq_len={bundle.seq_len} channels={bundle.radar_channels} joints={bundle.num_joints}",
        flush=True,
    )
    print(
        f"MM-Fi protocol={args.protocol} setting={args.setting} split={args.split} "
        f"sequences={len(sequences)} mapping={args.joint_mapping}",
        flush=True,
    )

    for sequence_idx, sequence in enumerate(sequences, start=1):
        gt = read_ground_truth(sequence)
        if args.coordinate_mode == "mmfi_camera_to_project":
            gt_project = mmfi_camera_to_project_frame(gt)
        else:
            gt_project = gt.astype(np.float32)
        gt_project = gt_project[:, mapping, :]

        radar_frames = [read_mmwave_frame(sequence, frame_index) for frame_index in range(1, gt.shape[0] + 1)]
        input_windows, centers = preprocess_radar_sequence(
            radar_frames,
            radar_channels=bundle.radar_channels,
            seq_len=bundle.seq_len,
            seed=args.seed + sequence_idx,
        )
        pred_local = predict_local_sequence(bundle, input_windows, batch_size=args.batch_size)
        if pred_local.shape != gt_project.shape:
            raise ValueError(f"Prediction/GT shape mismatch for {sequence}: {pred_local.shape} != {gt_project.shape}")

        pred_blocks.append(pred_local)
        gt_blocks.append(gt_project)
        row = {
            "sequence": f"{sequence.scene}/{sequence.subject}/{sequence.action}",
            "frames": int(gt.shape[0]),
            "mean_points_per_frame": float(np.mean([frame.shape[0] for frame in radar_frames])),
            "center_mean": centers.mean(axis=0).tolist(),
            **summarize(pred_local, gt_project),
        }
        sequence_rows.append(row)
        print(
            f"[{sequence_idx}/{len(sequences)}] {row['sequence']} "
            f"RR-MPJPE={row['root_relative_mpjpe_mm']:.2f}mm "
            f"PA={row['pa_mpjpe_mm']:.2f}mm",
            flush=True,
        )

    if not pred_blocks:
        raise ValueError("No sequences were evaluated")

    pred_all = np.concatenate(pred_blocks, axis=0)
    gt_all = np.concatenate(gt_blocks, axis=0)
    summary = summarize(pred_all, gt_all)
    result = {
        "note": (
            "Diagnostic zero-shot evaluation of the current project model. "
            "The mapping and coordinate transform are explicit; this is not an official MM-Fi score."
        ),
        "model": {
            "path": str(bundle.model_path),
            "model_type": bundle.model_type,
            "radar_channels": bundle.radar_channels,
            "input_channels": bundle.input_channels,
            "seq_len": bundle.seq_len,
            "num_joints": bundle.num_joints,
            "device": str(bundle.device),
        },
        "dataset": {
            "protocol": args.protocol,
            "setting": args.setting,
            "split": args.split,
            "random_seed": args.random_seed,
            "random_ratio": args.random_ratio,
            "test_scene": args.test_scene,
            "validation": validation,
            "evaluated_sequences": len(sequences),
            "evaluated_frames": int(pred_all.shape[0]),
        },
        "adapter": {
            "joint_mapping": args.joint_mapping,
            "gt_indices": mapping,
            "joint_names": PROJECT13_JOINT_NAMES,
            "coordinate_mode": args.coordinate_mode,
            "metric_scope": "13-joint root-relative and PA diagnostic",
        },
        "summary": summary,
        "sequences": sequence_rows,
    }

    print("Summary:", flush=True)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}", flush=True)
        else:
            print(f"  {key}: {value}", flush=True)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
